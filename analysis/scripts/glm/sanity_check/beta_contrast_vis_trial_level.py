#!/usr/bin/env python3
"""
Trial-level contrast analysis for per-run GLM outputs.

This adapts the condition-level contrast workflow to **trial-level** HDF5 files
produced by `trial_level_glm.py` (betas per event). It computes a linear
contrast over selected trial regressors, derives voxel/vertex-wise stats, and
(optionally) aggregates to ROI level using a Glasser atlas.

Selections
----------
Select trials for the **left** (A) and **right** (B) sides of the contrast via:
  • --lhs_regex / --rhs_regex : regex over `task_regressor_names` (e.g. "_encoding$" vs "_delay$")
  • --lhs_types / --rhs_types : event types (suffix in names), e.g. encoding, delay
  • --lhs_trials / --rhs_trials : explicit trialNumbers to include (integers)
You can mix options; the final set is the union of matches per side. If a trial
matches on both sides, it will be ignored (to avoid self-cancel).

Contrast and weights
--------------------
By default we test the **difference of means** between A and B:
  c assigns +1/|A| to A trials and -1/|B| to B trials (zeros elsewhere).
Use `--weights sum` to use +1 and -1 (sum difference) instead.

Outputs (per run)
-----------------
Under --stats_output mirroring the input tree:
  - *_trial_stats_<label>.h5        # voxel/vertex-level stats for the contrast
        datasets: beta_c, t_vals, p_vals, z_vals, sig_mask, fdr_mask, fdr_corrected_p
        attrs: dof, cXtXc, label, tail("two-sided"), n_lhs, n_rhs, task_col_start, task_col_end
               selected_lhs(names), selected_rhs(names)
Optionally, if --atlas and --roi_output are provided:
  - *_roi_stats_<label>_<tail>.h5   # ROI-level stats (Stouffer or IVW) like other scripts

Example
-------
python beta_contrast_vis_trial_level.py \
  --folder /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/sub-01 \
  --stats_output /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/stats/sub-01 \
  --lhs_types encoding --rhs_types delay \
  --p_thresh 0.05 --output_csv /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/stats/sub-01/summary_enc_vs_delay.csv \
  --atlas /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --roi_output /project/def-pbellec/xuan/fmri_dataset_project/data//trial_level_betas/roi_stats/sub-01 --roi_method ivw --tail two-sided
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
import nibabel as nib

# -----------------------
# Helpers
# -----------------------

def _safe_decode(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x


def _rename_stats_path(folder: Path, h5_path: Path, stats_output_dir: Path, label: str) -> Path:
    rel_path = h5_path.relative_to(folder)
    name = rel_path.name
    tag = f"_trial_stats_{label}.h5"
    if name.endswith("_betas.h5"):
        out_name = name[:-10] + tag
    else:
        out_name = rel_path.stem + tag
    return stats_output_dir / rel_path.parent / out_name


def _rename_roi_stats_path(folder: Path, h5_path: Path, roi_output_dir: Path, label: str, tail: str) -> Path:
    rel_path = h5_path.relative_to(folder)
    name = rel_path.name
    tag = f"_roi_stats_{label}_{tail}.h5"
    if name.endswith("_betas.h5"):
        out_name = name[:-10] + tag
    else:
        out_name = rel_path.stem + tag
    return roi_output_dir / rel_path.parent / out_name


def _find_task_indices(names_task: list[str], regex: str|None, types: list[str]|None, trials: list[int]|None) -> set[int]:
    sel = set()
    if regex:
        pat = re.compile(regex)
        for i, n in enumerate(names_task):
            if pat.search(n):
                sel.add(i)
    if types:
        ts = {t.lower() for t in types}
        for i, n in enumerate(names_task):
            # names like "trial012_encoding" -> type is after last underscore
            t = n.split("_")[-1].lower()
            if t in ts:
                sel.add(i)
    if trials:
        trs = {int(t) for t in trials}
        for i, n in enumerate(names_task):
            m = re.search(r"trial(\d+)", n)
            if m and int(m.group(1)) in trs:
                sel.add(i)
    return sel


# -----------------------
# Core compute per run
# -----------------------

def compute_trial_contrast(h5_path: Path, p_thresh=0.05, lhs_regex=None, rhs_regex=None,
                           lhs_types=None, rhs_types=None, lhs_trials=None, rhs_trials=None,
                           weight_mode="mean", tail="two-sided", verbose=False):
    with h5py.File(h5_path, "r") as f:
        betas_task = f["betas"][()]                 # (P x K_task)
        XtX_inv = f["XtX_inv"][()]                   # (K x K)
        sigma2 = f["sigma2"][()]                     # (P,)
        dof = int(f.attrs["dof"])                   # scalar
        t0 = int(f.attrs["task_col_start"])         # inclusive
        t1 = int(f.attrs["task_col_end"])           # exclusive
        # Names
        if "task_regressor_names" in f.attrs:
            tn = f.attrs["task_regressor_names"]
            names_task = [_safe_decode(x) for x in (tn if isinstance(tn, np.ndarray) else [tn])]
        else:
            # fall back to design_col_names slice
            cn = f["design_col_names"][()]
            all_names = [_safe_decode(x) for x in cn]
            names_task = all_names[t0:t1]

    K_total = XtX_inv.shape[0]
    K_task = betas_task.shape[1]

    # Build selections
    lhs = _find_task_indices(names_task, lhs_regex, lhs_types, lhs_trials)
    rhs = _find_task_indices(names_task, rhs_regex, rhs_types, rhs_trials)
    both = lhs & rhs
    if both:
        # remove overlaps
        lhs = lhs - both
        rhs = rhs - both
    n_lhs = len(lhs)
    n_rhs = len(rhs)
    if n_lhs == 0 or n_rhs == 0:
        raise ValueError(f"Need non-empty LHS ({n_lhs}) and RHS ({n_rhs}) trial sets for contrast.")

    # Weighting
    if weight_mode == "mean":
        wA = 1.0 / n_lhs
        wB = 1.0 / n_rhs
    elif weight_mode == "sum":
        wA = 1.0
        wB = 1.0
    else:
        raise ValueError("weight_mode must be 'mean' or 'sum'")

    # Build c_full over ALL columns
    c_full = np.zeros((K_total,), dtype=np.float64)
    for j in lhs:
        c_full[t0 + j] = +wA
    for j in rhs:
        c_full[t0 + j] = -wB

    # Contrast estimate per parcel: use task betas and a task-only weight vector
    w_task = np.zeros((K_task,), dtype=np.float64)
    for j in lhs:
        w_task[j] = +wA
    for j in rhs:
        w_task[j] = -wB

    beta_c = betas_task @ w_task   # (P,)

    # Variance via full XtX_inv
    cXtXc = float(c_full @ (XtX_inv @ c_full))
    if not np.isfinite(cXtXc) or cXtXc <= 0:
        P = betas_task.shape[0]
        nan = np.full((P,), np.nan, dtype=np.float64)
        return {
            "beta_c": nan.astype(np.float32),
            "t_vals": nan.astype(np.float32),
            "p_vals": nan, "z_vals": nan,
            "sig_mask": np.zeros((P,), dtype=bool),
            "fdr_mask": np.zeros((P,), dtype=bool),
            "fdr_corrected_p": nan,
            "diag": {"dof": dof, "cXtXc": cXtXc, "n_lhs": n_lhs, "n_rhs": n_rhs,
                      "task_col_start": t0, "task_col_end": t1,
                      "lhs_names": [names_task[i] for i in sorted(lhs)],
                      "rhs_names": [names_task[i] for i in sorted(rhs)]},
        }

    var_c = sigma2 * cXtXc
    var_c = np.where(var_c > 0, var_c, np.nan)
    se_c = np.sqrt(var_c)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = beta_c / se_c

    # p-values
    if tail == "two-sided":
        p_vals = 2.0 * t_dist.sf(np.abs(t_vals), df=dof)
    elif tail == "greater":
        p_vals = t_dist.sf(t_vals, df=dof)
    elif tail == "less":
        p_vals = t_dist.cdf(t_vals, df=dof)
    else:
        raise ValueError("tail must be one of: two-sided, greater, less")

    # Convert to z (two-sided by convention for z map; keep sign from beta)
    p_two = 2.0 * t_dist.sf(np.abs(t_vals), df=dof)
    p_two = np.clip(p_two, 1e-300, 1.0)
    z_abs = norm.isf(p_two / 2.0)
    signs = np.sign(beta_c)
    z_vals = signs * z_abs

    # Thresholding (on chosen tail)
    sig_mask = np.isfinite(p_vals) & (p_vals < p_thresh)

    finite = np.isfinite(p_vals)
    fdr_mask = np.zeros_like(sig_mask, dtype=bool)
    fdr_corrected_p = np.full_like(p_vals, np.nan, dtype=np.float64)
    if np.any(finite):
        rej, p_fdr = fdrcorrection(p_vals[finite], alpha=p_thresh)
        fdr_mask[finite] = rej
        fdr_corrected_p[finite] = p_fdr

    return {
        "beta_c": beta_c.astype(np.float32),
        "t_vals": t_vals.astype(np.float32),
        "p_vals": p_vals.astype(np.float64),
        "z_vals": z_vals.astype(np.float64),
        "sig_mask": sig_mask,
        "fdr_mask": fdr_mask,
        "fdr_corrected_p": fdr_corrected_p.astype(np.float64),
        "diag": {"dof": dof, "cXtXc": cXtXc, "n_lhs": n_lhs, "n_rhs": n_rhs,
                  "task_col_start": t0, "task_col_end": t1,
                  "lhs_names": [names_task[i] for i in sorted(lhs)],
                  "rhs_names": [names_task[i] for i in sorted(rhs)]},
    }


# -----------------------
# Optional ROI aggregation
# -----------------------

def aggregate_roi(atlas_path: Path, beta_c: np.ndarray, z_signed_vox: np.ndarray,
                  var_c_vox: np.ndarray, method: str = "ivw", tail: str = "two-sided",
                  alpha: float = 0.05, num_regions: int = 360):
    atlas = nib.load(str(atlas_path)).get_fdata()
    if atlas.ndim == 2:
        atlas = atlas[0]
    atlas = atlas.astype(int)
    P = atlas.shape[0]
    if P != beta_c.shape[0]:
        raise RuntimeError(f"Atlas length ({P}) != data length ({beta_c.shape[0]})")

    from scipy.stats import norm as _norm

    roi_beta_ivw = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_se_ivw   = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_z        = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_p_two    = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_p_greater= np.full((num_regions,), np.nan, dtype=np.float64)
    roi_p_less   = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_n_vox    = np.zeros((num_regions,), dtype=int)

    for roi in range(num_regions):
        idx = np.where(atlas == (roi + 1))[0]
        if idx.size == 0:
            continue
        roi_n_vox[roi] = int(idx.size)

        if method == "stouffer":
            zvals = z_signed_vox[idx]
            valid = np.isfinite(zvals)
            if not np.any(valid):
                continue
            zc = float(np.nansum(zvals[valid]) / np.sqrt(valid.sum()))
            p2 = 2.0 * (1.0 - _norm.cdf(abs(zc)))
            pg = 1.0 - _norm.cdf(zc)
            roi_z[roi] = zc
            roi_p_two[roi] = p2
            roi_p_greater[roi] = pg
            roi_p_less[roi] = 1.0 - pg
        elif method == "ivw":
            b = beta_c[idx]
            v = var_c_vox[idx]
            valid = np.isfinite(b) & np.isfinite(v) & (v > 0)
            if not np.any(valid):
                continue
            w = 1.0 / v[valid]
            sw = float(np.sum(w))
            beta_ivw = float(np.sum(w * b[valid]) / sw)
            se_ivw = float(np.sqrt(1.0 / sw))
            zc = beta_ivw / se_ivw if se_ivw > 0 else np.nan
            p2 = 2.0 * (1.0 - _norm.cdf(abs(zc))) if np.isfinite(zc) else np.nan
            pg = 1.0 - _norm.cdf(zc) if np.isfinite(zc) else np.nan
            roi_beta_ivw[roi] = beta_ivw
            roi_se_ivw[roi] = se_ivw
            roi_z[roi] = zc
            roi_p_two[roi] = p2
            roi_p_greater[roi] = pg
            roi_p_less[roi] = 1.0 - pg if np.isfinite(pg) else np.nan
        else:
            raise ValueError("method must be 'stouffer' or 'ivw'")

    # choose tail
    if tail == "two-sided":
        p_use = roi_p_two
    elif tail == "greater":
        p_use = roi_p_greater
    elif tail == "less":
        p_use = roi_p_less
    else:
        raise ValueError("tail must be one of: two-sided, greater, less")

    finite = np.isfinite(p_use)
    sig_mask = np.zeros_like(p_use, dtype=bool)
    sig_mask[finite] = p_use[finite] < alpha

    fdr_mask = np.zeros_like(sig_mask, dtype=bool)
    fdr_corrected_p = np.full_like(p_use, np.nan, dtype=np.float64)
    if np.any(finite):
        rej, p_fdr = fdrcorrection(p_use[finite], alpha=alpha)
        fdr_mask[finite] = rej
        fdr_corrected_p[finite] = p_fdr

    return {
        "roi_beta_ivw": roi_beta_ivw,
        "roi_se_ivw": roi_se_ivw,
        "roi_z": roi_z,
        "roi_p_two": roi_p_two,
        "roi_p_greater": roi_p_greater,
        "roi_p_less": roi_p_less,
        "sig_mask": sig_mask,
        "fdr_mask": fdr_mask,
        "fdr_corrected_p": fdr_corrected_p,
        "roi_n_vox": roi_n_vox,
    }


# -----------------------
# Batch
# -----------------------

def process_folder(folder, stats_output_dir, output_csv=None, p_thresh=0.05,
                   lhs_regex=None, rhs_regex=None,
                   lhs_types=None, rhs_types=None,
                   lhs_trials=None, rhs_trials=None,
                   weight_mode="mean", tail="two-sided",
                   atlas_path=None, roi_output_dir=None, roi_method="ivw",
                   num_regions=360, verbose=False):
    folder = Path(folder)
    stats_output_dir = Path(stats_output_dir)
    stats_output_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = Path(atlas_path) if atlas_path else None
    roi_output_dir = Path(roi_output_dir) if roi_output_dir else None
    if atlas_path and not roi_output_dir:
        roi_output_dir = stats_output_dir.parent / "roi_stats"
        roi_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Build label for filenames
    def set_label():
        L = []
        if lhs_types: L.append("Ltypes-"+"+".join(lhs_types))
        if lhs_regex: L.append("Lre-"+lhs_regex)
        if lhs_trials: L.append("Ltr-"+"+".join(map(str,lhs_trials)))
        if rhs_types: L.append("Rtypes-"+"+".join(rhs_types))
        if rhs_regex: L.append("Rre-"+rhs_regex)
        if rhs_trials: L.append("Rtr-"+"+".join(map(str,rhs_trials)))
        L.append(weight_mode)
        return "_".join(L) if L else "custom"
    label = set_label()

    for h5_path in folder.rglob("*_betas.h5"):
        try:
            res = compute_trial_contrast(
                h5_path, p_thresh=p_thresh,
                lhs_regex=lhs_regex, rhs_regex=rhs_regex,
                lhs_types=lhs_types, rhs_types=rhs_types,
                lhs_trials=lhs_trials, rhs_trials=rhs_trials,
                weight_mode=weight_mode, tail=tail, verbose=verbose,
            )
        except Exception as e:
            print(f"[!] Skip {h5_path}: {e}")
            continue

        stats_file = _rename_stats_path(folder, h5_path, stats_output_dir, label)
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        beta_c = res["beta_c"]
        t_vals = res["t_vals"]
        p_vals = res["p_vals"]
        z_vals = res["z_vals"]
        sig_mask = res["sig_mask"]
        fdr_mask = res["fdr_mask"]
        fdr_corrected_p = res["fdr_corrected_p"]
        diag = res["diag"]

        # summary row
        finite_t = t_vals[np.isfinite(t_vals)]
        finite_p = p_vals[np.isfinite(p_vals)]
        finite_z = z_vals[np.isfinite(z_vals)]
        rows.append({
            "file": str(stats_file),
            "label": label,
            "tail": tail,
            "n_sig_voxels": int(np.sum(sig_mask)),
            "n_fdr_sig_voxels": int(np.sum(fdr_mask)),
            "max_t": float(np.max(finite_t)) if finite_t.size else np.nan,
            "min_p": float(np.min(finite_p)) if finite_p.size else np.nan,
            "max_z": float(np.max(finite_z)) if finite_z.size else np.nan,
            "dof": int(diag["dof"]),
            "cXtXc": float(diag["cXtXc"]),
            "n_lhs": int(diag["n_lhs"]),
            "n_rhs": int(diag["n_rhs"]),
        })

        # write voxel-level stats h5
        with h5py.File(stats_file, "w") as f:
            f.create_dataset("beta_c", data=beta_c.astype(np.float32))
            f.create_dataset("t_vals", data=t_vals.astype(np.float32))
            f.create_dataset("p_vals", data=p_vals.astype(np.float64))
            f.create_dataset("z_vals", data=z_vals.astype(np.float64))
            f.create_dataset("sig_mask", data=sig_mask.astype(np.uint8))
            f.create_dataset("fdr_mask", data=fdr_mask.astype(np.uint8))
            f.create_dataset("fdr_corrected_p", data=fdr_corrected_p.astype(np.float64))
            f.attrs["dof"] = int(diag["dof"])  # pass-through
            f.attrs["cXtXc"] = float(diag["cXtXc"])  # scalar
            f.attrs["n_lhs"] = int(diag["n_lhs"]) ; f.attrs["n_rhs"] = int(diag["n_rhs"]) ;
            f.attrs["task_col_start"] = int(diag["task_col_start"]) ; f.attrs["task_col_end"] = int(diag["task_col_end"]) ;
            # store selections (names) as strings
            try:
                f.create_dataset("selected_lhs", data=np.array(diag["lhs_names"], dtype=object), dtype=h5py.string_dtype("utf-8"))
                f.create_dataset("selected_rhs", data=np.array(diag["rhs_names"], dtype=object), dtype=h5py.string_dtype("utf-8"))
            except Exception:
                pass
            f.attrs["label"] = label
            f.attrs["tail"] = tail
            f.attrs["p_thresh"] = float(p_thresh)

        # optional ROI aggregation
        if atlas_path is not None and roi_output_dir is not None:
            # compute per-voxel variance for IVW
            var_c_vox = (t_vals.astype(np.float64) ** 2)
            # safer: recompute from beta and t
            with np.errstate(divide='ignore', invalid='ignore'):
                var_c_vox = (beta_c.astype(np.float64) ** 2) / (t_vals.astype(np.float64) ** 2)
            # but t can be 0/NaN; fallback using cXtXc * sigma2 not available here; skip
            # We'll approximate ROI with available beta_c and p->z map
            z_signed_vox = z_vals

            roi = aggregate_roi(atlas_path, beta_c, z_signed_vox, var_c_vox, method=roi_method, tail=tail, alpha=p_thresh, num_regions=num_regions)

            roi_file = _rename_roi_stats_path(folder, h5_path, roi_output_dir, label, tail)
            roi_file.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(roi_file, 'w') as f:
                for k in ["roi_beta_ivw","roi_se_ivw","roi_z","roi_p_two","roi_p_greater","roi_p_less","sig_mask","fdr_mask","fdr_corrected_p","roi_n_vox"]:
                    if k in roi and roi[k] is not None:
                        f.create_dataset(k, data=np.asarray(roi[k]))
                f.attrs['label'] = label
                f.attrs['tail'] = tail
                f.attrs['roi_method'] = roi_method
                f.attrs['alpha'] = float(p_thresh)
                f.attrs['num_regions'] = int(num_regions)

            if verbose:
                print(f"[✓ ROI] {roi_file}")

    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved summary CSV: {output_csv}")
    return df


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Trial-level per-run contrast + optional ROI aggregation")
    ap.add_argument("--folder", required=True, help="Path to folder containing *_betas.h5 (trial-level)")
    ap.add_argument("--stats_output", required=True, help="Path to save voxel-level stats HDF5 outputs")
    ap.add_argument("--p_thresh", type=float, default=0.05, help="Significance/FDR alpha")
    ap.add_argument("--output_csv", default=None, help="Path to save summary CSV")

    # selections
    ap.add_argument("--lhs_regex", default=None, help="Regex for LHS trial columns (e.g., '_encoding$')")
    ap.add_argument("--rhs_regex", default=None, help="Regex for RHS trial columns (e.g., '_delay$')")
    ap.add_argument("--lhs_types", nargs="*", default=None, help="Trial types for LHS (e.g., encoding delay)")
    ap.add_argument("--rhs_types", nargs="*", default=None, help="Trial types for RHS")
    ap.add_argument("--lhs_trials", nargs="*", type=int, default=None, help="Explicit trialNumbers for LHS")
    ap.add_argument("--rhs_trials", nargs="*", type=int, default=None, help="Explicit trialNumbers for RHS")

    ap.add_argument("--weight_mode", choices=["mean","sum"], default="mean", help="Weights for A/B in contrast")
    ap.add_argument("--tail", choices=["two-sided","greater","less"], default="two-sided", help="Tail for p-values")

    # ROI
    ap.add_argument("--atlas", default=None, help="Optional Glasser dlabel.nii path for ROI aggregation")
    ap.add_argument("--roi_output", default=None, help="Output dir for ROI HDF5 (defaults to <root>/roi_stats)")
    ap.add_argument("--roi_method", choices=["stouffer","ivw"], default="ivw")
    ap.add_argument("--num_regions", type=int, default=360)

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    process_folder(
        folder=args.folder,
        stats_output_dir=args.stats_output,
        output_csv=args.output_csv,
        p_thresh=args.p_thresh,
        lhs_regex=args.lhs_regex, rhs_regex=args.rhs_regex,
        lhs_types=args.lhs_types, rhs_types=args.rhs_types,
        lhs_trials=args.lhs_trials, rhs_trials=args.rhs_trials,
        weight_mode=args.weight_mode, tail=args.tail,
        atlas_path=args.atlas, roi_output_dir=args.roi_output, roi_method=args.roi_method,
        num_regions=args.num_regions, verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
