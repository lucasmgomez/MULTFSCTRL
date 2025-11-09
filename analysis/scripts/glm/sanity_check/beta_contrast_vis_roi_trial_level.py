#!/usr/bin/env python3
"""
Per-run ROI-level contrast analysis for **trial-level** GLM outputs (collision-safe).

This script reads trial-level per-run HDF5 files (from your trial-level GLM),
selects two sets of trials (LHS vs RHS), builds a linear contrast, computes
per-voxel beta_c and its variance via `c^T (X^T X)^{-1} c * sigma2`, and then
aggregates to **ROI level** using a Glasser atlas (IVW or Stouffer).

Collision-safe outputs:
- If an ROI stats file already exists but was created from a *different* betas file,
  a short hash of the source path is appended to the filename (e.g., `__h9a3b1c2`).
  Disable with `--no-unique_out` if you intentionally want to overwrite.
- Each ROI HDF5 stores `source_betas_file` in attrs for provenance.

Outputs (mirrors input tree under --roi_output):
  <...>/<base>_roi_stats_<label>_<tail>.h5
    datasets (IVW):
      roi_beta_ivw, roi_se_ivw, roi_z, roi_p_two, roi_p_greater, roi_p_less
    datasets (Stouffer):
      roi_z, roi_p_two, roi_p_greater, roi_p_less
    common:
      sig_mask, fdr_mask, fdr_corrected_p, roi_n_vox
    attrs: subj, ses, task, run, label, tail, roi_method, alpha, num_regions,
           n_lhs, n_rhs, task_col_start, task_col_end,
           selected_lhs (names), selected_rhs (names), source_betas_file

Usage examples
--------------
# Encoding vs Delay at trial level
python beta_contrast_vis_roi_trial_level.py \
  --folder      /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/sub-01 \
  --atlas       /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --roi_output  /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/roi_stats \
  --lhs_types encoding --rhs_types delay \
  --tail two-sided --roi_method ivw --p_thresh 0.05 \
  --output_csv  /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/roi_stats/sub-01/summary_enc_vs_delay.csv

# Same but use Stouffer aggregation and one-sided (encoding > delay)
python trial_level_roi_contrast_safe.py \
  --folder /.../trial_level_betas/sub-01 --atlas /.../Glasser_LR_Dense64k.dlabel.nii \
  --roi_output /.../trial_level_betas/roi_stats --lhs_types encoding --rhs_types delay \
  --roi_method stouffer --tail greater
"""
from __future__ import annotations
import argparse
import re
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import t as t_dist
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

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


def _build_label(lhs_regex, rhs_regex, lhs_types, rhs_types, lhs_trials, rhs_trials, weight_mode):
    L = []
    if lhs_types: L.append("Ltypes-"+"+".join(lhs_types))
    if lhs_regex: L.append("Lre-"+lhs_regex)
    if lhs_trials: L.append("Ltr-"+"+".join(map(str,lhs_trials)))
    if rhs_types: L.append("Rtypes-"+"+".join(rhs_types))
    if rhs_regex: L.append("Rre-"+rhs_regex)
    if rhs_trials: L.append("Rtr-"+"+".join(map(str,rhs_trials)))
    L.append(weight_mode)
    return "_".join(L) if L else "custom"


def _rename_roi_stats_path(folder: Path, h5_path: Path, roi_output_dir: Path, label: str, tail: str) -> Path:
    rel_path = h5_path.relative_to(folder)
    name = rel_path.name
    tag = f"_roi_stats_{label}_{tail}.h5"
    if name.endswith("_betas.h5"):
        out_name = name[:-10] + tag
    else:
        out_name = rel_path.stem + tag
    return roi_output_dir / rel_path.parent / out_name


def _unique_out_path(base_out: Path, source_betas: Path, unique_out: bool = True) -> Path:
    """If base_out exists and was produced from a different betas file, append a short hash."""
    if not unique_out:
        return base_out
    if not base_out.exists():
        return base_out
    prev_src = None
    try:
        with h5py.File(base_out, "r") as f:
            prev_src = f.attrs.get("source_betas_file", None)
            if isinstance(prev_src, (bytes, bytearray)):
                prev_src = prev_src.decode("utf-8", errors="ignore")
    except Exception:
        prev_src = None
    if prev_src and Path(prev_src) == source_betas:
        return base_out
    h = hashlib.md5(str(source_betas).encode("utf-8")).hexdigest()[:8]
    return base_out.with_name(base_out.stem + f"__h{h}" + base_out.suffix)


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


def _parse_ids_from_name(path: Path) -> dict:
    """Best-effort parse of subj/ses/task/run from a filename like
    sub-01_ses-001_task-ctxdm_run-01_betas.h5
    """
    info = {"subj": None, "ses": None, "task": None, "run": None}
    m = re.search(r"(sub-[^_]+)_(ses-[^_]+)_task-([^_]+)_([^_]+)", path.name)
    if m:
        info["subj"], info["ses"], info["task"], info["run"] = m.group(1,2,3,4)
    return info

# -----------------------
# Core per-run ROI compute
# -----------------------

def compute_roi_contrast(h5_path: Path, atlas_path: Path, p_thresh: float,
                         lhs_regex=None, rhs_regex=None,
                         lhs_types=None, rhs_types=None,
                         lhs_trials=None, rhs_trials=None,
                         weight_mode="mean", tail="two-sided",
                         roi_method="ivw", num_regions=360, verbose=False):
    # Load GLM pieces
    with h5py.File(h5_path, "r") as f:
        betas_task = f["betas"][()]                 # (P x K_task)
        XtX_inv = f["XtX_inv"][()]                  # (K x K)
        sigma2 = f["sigma2"][()]                    # (P,)
        dof = int(f.attrs["dof"])                   # scalar
        t0 = int(f.attrs["task_col_start"])         # inclusive
        t1 = int(f.attrs["task_col_end"])           # exclusive
        # Names
        if "task_regressor_names" in f.attrs:
            tn = f.attrs["task_regressor_names"]
            names_task = [_safe_decode(x) for x in (tn if isinstance(tn, np.ndarray) else [tn])]
        else:
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
        lhs = lhs - both
        rhs = rhs - both
    n_lhs = len(lhs)
    n_rhs = len(rhs)
    if n_lhs == 0 or n_rhs == 0:
        raise ValueError(f"Need non-empty LHS ({n_lhs}) and RHS ({n_rhs}) trial sets for contrast.")

    # Weights
    if weight_mode == "mean":
        wA = 1.0 / n_lhs
        wB = 1.0 / n_rhs
    elif weight_mode == "sum":
        wA = 1.0
        wB = 1.0
    else:
        raise ValueError("weight_mode must be 'mean' or 'sum'")

    # Build c over ALL columns to get cXtXc
    c_full = np.zeros((K_total,), dtype=np.float64)
    for j in lhs:
        c_full[t0 + j] = +wA
    for j in rhs:
        c_full[t0 + j] = -wB
    cXtXc = float(c_full @ (XtX_inv @ c_full))
    if not np.isfinite(cXtXc) or cXtXc <= 0:
        raise RuntimeError("Contrast not estimable (c^T (X^T X)^{-1} c <= 0)")

    # Beta contrast via task betas
    w_task = np.zeros((K_task,), dtype=np.float64)
    for j in lhs: w_task[j] = +wA
    for j in rhs: w_task[j] = -wB
    beta_c = betas_task @ w_task   # (P,)

    # Variance per voxel
    var_c_vox = sigma2.astype(np.float64) * cXtXc
    se_c_vox = np.sqrt(var_c_vox)

    # Signed z per voxel (two-sided to get |z|, then restore sign)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vals = beta_c / se_c_vox
    p_two = 2.0 * t_dist.sf(np.abs(t_vals), df=dof)
    p_two = np.clip(p_two, 1e-300, 1.0)
    z_abs = norm.isf(p_two / 2.0)
    z_signed_vox = np.sign(beta_c) * z_abs

    # Load atlas labels (Dense64k dlabel.nii)
    atlas = nib.load(str(atlas_path)).get_fdata()
    if atlas.ndim == 2:
        atlas = atlas[0]
    labels = atlas.astype(int)
    P = labels.shape[0]
    if P != beta_c.shape[0]:
        raise RuntimeError(f"Atlas length ({P}) != data length ({beta_c.shape[0]})")

    # Aggregate to ROI
    roi_beta_ivw = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_se_ivw   = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_z        = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_p_two    = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_p_greater= np.full((num_regions,), np.nan, dtype=np.float64)
    roi_p_less   = np.full((num_regions,), np.nan, dtype=np.float64)
    roi_n_vox    = np.zeros((num_regions,), dtype=int)

    for roi in range(num_regions):
        idx = np.where(labels == (roi + 1))[0]
        if idx.size == 0:
            continue
        roi_n_vox[roi] = int(idx.size)

        if roi_method == "stouffer":
            zvals = z_signed_vox[idx]
            valid = np.isfinite(zvals)
            if not np.any(valid):
                continue
            zc = float(np.nansum(zvals[valid]) / np.sqrt(valid.sum()))
            p2 = 2.0 * (1.0 - norm.cdf(abs(zc)))
            pg = 1.0 - norm.cdf(zc)
            roi_z[roi] = zc
            roi_p_two[roi] = p2
            roi_p_greater[roi] = pg
            roi_p_less[roi] = 1.0 - pg
        elif roi_method == "ivw":
            b = beta_c[idx].astype(np.float64)
            v = var_c_vox[idx].astype(np.float64)
            valid = np.isfinite(b) & np.isfinite(v) & (v > 0)
            if not np.any(valid):
                continue
            w = 1.0 / v[valid]
            sw = float(np.sum(w))
            beta_ivw = float(np.sum(w * b[valid]) / sw)
            se_ivw = float(np.sqrt(1.0 / sw))
            zc = beta_ivw / se_ivw if se_ivw > 0 else np.nan
            p2 = 2.0 * (1.0 - norm.cdf(abs(zc))) if np.isfinite(zc) else np.nan
            pg = 1.0 - norm.cdf(zc) if np.isfinite(zc) else np.nan
            roi_beta_ivw[roi] = beta_ivw
            roi_se_ivw[roi] = se_ivw
            roi_z[roi] = zc
            roi_p_two[roi] = p2
            roi_p_greater[roi] = pg
            roi_p_less[roi] = 1.0 - pg if np.isfinite(pg) else np.nan
        else:
            raise ValueError("roi_method must be 'stouffer' or 'ivw'")

    # Choose p for thresholding
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
    sig_mask[finite] = p_use[finite] < p_thresh

    fdr_mask = np.zeros_like(sig_mask, dtype=bool)
    fdr_corrected_p = np.full_like(p_use, np.nan, dtype=np.float64)
    if np.any(finite):
        rej, p_fdr = fdrcorrection(p_use[finite], alpha=p_thresh)
        fdr_mask[finite] = rej
        fdr_corrected_p[finite] = p_fdr

    diag = {
        "dof": dof,
        "cXtXc": float(cXtXc),
        "n_lhs": n_lhs,
        "n_rhs": n_rhs,
        "task_col_start": t0,
        "task_col_end": t1,
        "lhs_names": [names_task[i] for i in sorted(lhs)],
        "rhs_names": [names_task[i] for i in sorted(rhs)],
    }

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
        "diag": diag,
    }


# -----------------------
# Batch
# -----------------------

def process_folder(folder, atlas_path, roi_output_dir, output_csv=None, p_thresh=0.05,
                   lhs_regex=None, rhs_regex=None,
                   lhs_types=None, rhs_types=None,
                   lhs_trials=None, rhs_trials=None,
                   weight_mode="mean", tail="two-sided",
                   roi_method="ivw", num_regions=360, unique_out=True, verbose=False):
    folder = Path(folder)
    atlas_path = Path(atlas_path)
    roi_output_dir = Path(roi_output_dir)
    roi_output_dir.mkdir(parents=True, exist_ok=True)

    label = _build_label(lhs_regex, rhs_regex, lhs_types, rhs_types, lhs_trials, rhs_trials, weight_mode)

    rows = []
    for h5_path in folder.rglob("*_betas.h5"):
        try:
            res = compute_roi_contrast(
                h5_path, atlas_path, p_thresh,
                lhs_regex=lhs_regex, rhs_regex=rhs_regex,
                lhs_types=lhs_types, rhs_types=rhs_types,
                lhs_trials=lhs_trials, rhs_trials=rhs_trials,
                weight_mode=weight_mode, tail=tail,
                roi_method=roi_method, num_regions=num_regions, verbose=verbose,
            )
        except Exception as e:
            print(f"[!] Skip {h5_path}: {e}")
            continue

        base_out = _rename_roi_stats_path(folder, h5_path, roi_output_dir, label, tail)
        out_path = _unique_out_path(base_out, h5_path, unique_out=unique_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        d = res
        ids = _parse_ids_from_name(h5_path)
        with h5py.File(out_path, 'w') as f:
            # datasets (present where filled)
            for k in ["roi_beta_ivw","roi_se_ivw","roi_z","roi_p_two","roi_p_greater","roi_p_less","sig_mask","fdr_mask","fdr_corrected_p","roi_n_vox"]:
                if d.get(k) is not None:
                    f.create_dataset(k, data=np.asarray(d[k]))
            # attrs
            if ids.get("subj"): f.attrs['subj'] = ids['subj']
            if ids.get("ses"):  f.attrs['ses'] = ids['ses']
            if ids.get("task"): f.attrs['task'] = ids['task']
            if ids.get("run"):  f.attrs['run'] = ids['run']
            f.attrs['label'] = label
            f.attrs['tail'] = tail
            f.attrs['roi_method'] = roi_method
            f.attrs['alpha'] = float(p_thresh)
            f.attrs['num_regions'] = int(num_regions)
            f.attrs['dof'] = int(d['diag']['dof'])
            f.attrs['cXtXc'] = float(d['diag']['cXtXc'])
            f.attrs['n_lhs'] = int(d['diag']['n_lhs'])
            f.attrs['n_rhs'] = int(d['diag']['n_rhs'])
            f.attrs['task_col_start'] = int(d['diag']['task_col_start'])
            f.attrs['task_col_end'] = int(d['diag']['task_col_end'])
            f.attrs['source_betas_file'] = str(h5_path)
            try:
                f.create_dataset('selected_lhs', data=np.array(d['diag']['lhs_names'], dtype=object), dtype=h5py.string_dtype('utf-8'))
                f.create_dataset('selected_rhs', data=np.array(d['diag']['rhs_names'], dtype=object), dtype=h5py.string_dtype('utf-8'))
            except Exception:
                pass

        # summary row
        roi_z = d['roi_z']
        p_two = d['roi_p_two']
        sig = d['sig_mask']
        fdr = d['fdr_mask']
        rows.append({
            'file': str(out_path),
            'label': label,
            'tail': tail,
            'roi_method': roi_method,
            'n_sig_rois': int(np.nansum(sig)),
            'n_fdr_sig_rois': int(np.nansum(fdr)),
            'max_|z|': float(np.nanmax(np.abs(roi_z))) if np.any(np.isfinite(roi_z)) else np.nan,
            'min_p_two': float(np.nanmin(p_two)) if np.any(np.isfinite(p_two)) else np.nan,
        })

        if verbose:
            post = " (collision -> hashed)" if out_path != base_out else ""
            print(f"[✓ ROI] {out_path}{post}")

    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved ROI summary CSV: {output_csv}")
    return df


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Trial-level per-run ROI contrast (Glasser)")
    ap.add_argument("--folder", required=True, help="Path to folder containing *_betas.h5 (trial-level)")
    ap.add_argument("--atlas", required=True, help="Glasser dlabel.nii path (Dense64k)")
    ap.add_argument("--roi_output", required=True, help="Output dir for ROI HDF5")
    ap.add_argument("--p_thresh", type=float, default=0.05, help="Alpha for significance/FDR")
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

    ap.add_argument("--roi_method", choices=["stouffer","ivw"], default="ivw")
    ap.add_argument("--num_regions", type=int, default=360)
    ap.add_argument("--no-unique_out", dest="unique_out", action="store_false", help="Disable collision-safe filenames")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    process_folder(
        folder=args.folder,
        atlas_path=args.atlas,
        roi_output_dir=args.roi_output,
        output_csv=args.output_csv,
        p_thresh=args.p_thresh,
        lhs_regex=args.lhs_regex, rhs_regex=args.rhs_regex,
        lhs_types=args.lhs_types, rhs_types=args.rhs_types,
        lhs_trials=args.lhs_trials, rhs_trials=args.rhs_trials,
        weight_mode=args.weight_mode, tail=args.tail,
        roi_method=args.roi_method, num_regions=args.num_regions,
        unique_out=args.unique_out,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
