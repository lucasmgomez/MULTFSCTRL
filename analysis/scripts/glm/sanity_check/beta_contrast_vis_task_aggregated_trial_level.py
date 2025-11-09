#!/usr/bin/env python3
"""
Aggregate per-run **GLM outputs directly** into task-level ROI contrast stats (within-subject),
for **trial-level** GLM analyses — no dependency on per-run ROI stats files.

What this does
--------------
• Scans your GLM result tree (trial-level *_betas.h5 files).
• For each run, reads betas, XtX_inv, sigma2, and task-slice metadata.
• Builds a contrast between two sets of trial regressors (e.g., encoding vs delay).
• Computes per-VOXEL beta_c and its variance via c^T (X^T X)^{-1} c * sigma2.
• Aggregates to **ROI level** (Glasser) per run using either:
    - stouffer: signed Z combine across voxels in the ROI
    - ivw: inverse-variance–weighted beta + SE across voxels in the ROI
• Finally, combines **across runs (within task)** into a single task-level ROI result:
    - stouffer across runs (for stouffer per-run ROI z)
    - IVW across runs (for IVW per-run ROI beta/SE)

Outputs (hierarchical by subject & task)
----------------------------------------
<out_root>/<subj>/task-<task>/<subj>_task-<task>_taskroi_<label>_<tail>_<method>.h5
  datasets:
    For stouffer: roi_z, roi_p_two, roi_p_greater, roi_p_less, n_runs_per_roi
    For ivw:      roi_beta_ivw, roi_se_ivw, roi_z, roi_p_two, roi_p_greater, roi_p_less, n_runs_per_roi
    Common:       sig_mask, fdr_mask, fdr_corrected_p
  attrs: subj, task, label, tail, roi_method, alpha, num_regions, n_runs_total
  attrs: run_files (basenames) and run_relpaths (relative to --glm_root)

Examples
--------
# Aggregate encoding vs delay for ctxdm, IVW, two-sided
python beta_contrast_vis_task_aggregated_trial_level.py \
  --glm_root   /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas \
  --atlas      /project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii \
  --out_root   /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/task_level_roi_stats \
  --lhs_types encoding --rhs_types delay \
  --task 1back --roi_method ivw --tail two-sided \
  --subj sub-01 \
  --output_csv /project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/task_level_roi_stats/summary_enc_vs_delay_1back.csv

# Same but Stouffer and one-sided (encoding > delay)
python beta_contrast_vis_task_aggregated_trial_level.py \
  --glm_root /project/.../trial_level_betas --atlas /project/.../Glasser_LR_Dense64k.dlabel.nii \
  --out_root /project/.../task_level_roi_stats --lhs_types encoding --rhs_types delay \
  --task ctxdm --roi_method stouffer --tail greater
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from scipy.stats import norm, t as t_dist
from statsmodels.stats.multitest import fdrcorrection

# -----------------------
# Helpers
# -----------------------

def _safe_dec(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)
    return x


def parse_subj_task_from_name(path: Path):
    """Extract subj (e.g., 'sub-01') and task (e.g., 'ctxdm') from filename.
    Assumes stem like: sub-01_ses-003_task-ctxdm_run-01_betas
    """
    stem = path.stem
    msub = re.search(r"(sub-[^_]+)", stem)
    subj = msub.group(1) if msub else "unknown"
    mt = re.search(r"task-([^_]+)", stem)
    task = mt.group(1) if mt else "unknown"
    return subj, task


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

# -----------------------
# Per-run ROI from GLM H5
# -----------------------

def compute_roi_from_glm(h5_path: Path, atlas_labels: np.ndarray, num_regions: int,
                         lhs_regex=None, rhs_regex=None,
                         lhs_types=None, rhs_types=None,
                         lhs_trials=None, rhs_trials=None,
                         weight_mode="mean", roi_method="ivw"):
    """Load GLM pieces, build contrast at voxel-level, then aggregate to ROI for this run.
    Returns (method-dependent):
      - if stouffer: roi_z (R,), valid_mask (R,)
      - if ivw:      roi_beta_ivw (R,), roi_se_ivw (R,), valid_mask (R,)
    Also returns subj, task, and a dict diag with fields for provenance.
    """
    with h5py.File(h5_path, "r") as f:
        betas_task = f["betas"][()]                 # (P x K_task)
        XtX_inv = f["XtX_inv"][()]                   # (K x K)
        sigma2 = f["sigma2"][()]                     # (P,)
        dof = int(f.attrs["dof"])                   # scalar
        t0 = int(f.attrs["task_col_start"])         # inclusive
        t1 = int(f.attrs["task_col_end"])           # exclusive
        # Task names
        if "task_regressor_names" in f.attrs:
            tn = f.attrs["task_regressor_names"]
            names_task = [_safe_dec(x) for x in (tn if isinstance(tn, np.ndarray) else [tn])]
        else:
            cn = f["design_col_names"][()]
            all_names = [_safe_dec(x) for x in cn]
            names_task = all_names[t0:t1]

    K_total = XtX_inv.shape[0]
    K_task = betas_task.shape[1]

    # selections
    lhs = _find_task_indices(names_task, lhs_regex, lhs_types, lhs_trials)
    rhs = _find_task_indices(names_task, rhs_regex, rhs_types, rhs_trials)
    both = lhs & rhs
    if both:
        lhs -= both; rhs -= both
    n_lhs, n_rhs = len(lhs), len(rhs)
    if n_lhs == 0 or n_rhs == 0:
        raise ValueError(f"Need non-empty LHS ({n_lhs}) and RHS ({n_rhs}) trial sets for contrast.")

    # weights within each side
    if weight_mode == "mean":
        wA = 1.0 / n_lhs; wB = 1.0 / n_rhs
    elif weight_mode == "sum":
        wA = 1.0; wB = 1.0
    else:
        raise ValueError("weight_mode must be 'mean' or 'sum'")

    # c over ALL columns
    c_full = np.zeros((K_total,), dtype=np.float64)
    for j in lhs: c_full[t0 + j] = +wA
    for j in rhs: c_full[t0 + j] = -wB
    cXtXc = float(c_full @ (XtX_inv @ c_full))
    if not np.isfinite(cXtXc) or cXtXc <= 0:
        raise RuntimeError("Contrast not estimable: c^T (X^T X)^{-1} c <= 0")

    # beta_c per voxel via task betas (no need to build full beta vec)
    w_task = np.zeros((K_task,), dtype=np.float64)
    for j in lhs: w_task[j] = +wA
    for j in rhs: w_task[j] = -wB
    beta_c = betas_task @ w_task   # (P,)

    # variance & se per voxel
    var_c_vox = sigma2.astype(np.float64) * cXtXc
    se_c_vox = np.sqrt(var_c_vox)

    # t and z per voxel
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vals = beta_c / se_c_vox
    p_two = 2.0 * t_dist.sf(np.abs(t_vals), df=dof)
    p_two = np.clip(p_two, 1e-300, 1.0)
    z_abs = norm.isf(p_two / 2.0)
    z_signed_vox = np.sign(beta_c) * z_abs

    # Aggregate to ROI
    labels = atlas_labels
    if labels.ndim == 2:
        labels = labels[0]
    labels = labels.astype(int)
    P = labels.shape[0]
    if P != beta_c.shape[0]:
        raise RuntimeError(f"Atlas length ({P}) != data length ({beta_c.shape[0]})")

    R = num_regions
    roi_n_vox = np.zeros((R,), dtype=int)

    if roi_method == "stouffer":
        roi_z = np.full((R,), np.nan, dtype=np.float64)
        valid = np.zeros((R,), dtype=bool)
        for roi in range(R):
            idx = np.where(labels == (roi + 1))[0]
            if idx.size == 0:
                continue
            roi_n_vox[roi] = int(idx.size)
            zvals = z_signed_vox[idx]
            ok = np.isfinite(zvals)
            if not np.any(ok):
                continue
            roi_z[roi] = float(np.nansum(zvals[ok]) / np.sqrt(ok.sum()))
            valid[roi] = True
        diag = {"dof":dof, "cXtXc":cXtXc, "n_lhs":n_lhs, "n_rhs":n_rhs, "task_col_start":t0, "task_col_end":t1,
                "lhs_names":[names_task[i] for i in sorted(lhs)], "rhs_names":[names_task[i] for i in sorted(rhs)]}
        return (roi_z, valid, roi_n_vox, diag)

    elif roi_method == "ivw":
        roi_beta = np.full((R,), np.nan, dtype=np.float64)
        roi_se   = np.full((R,), np.nan, dtype=np.float64)
        valid = np.zeros((R,), dtype=bool)
        for roi in range(R):
            idx = np.where(labels == (roi + 1))[0]
            if idx.size == 0:
                continue
            roi_n_vox[roi] = int(idx.size)
            b = beta_c[idx].astype(np.float64)
            v = var_c_vox[idx].astype(np.float64)
            ok = np.isfinite(b) & np.isfinite(v) & (v > 0)
            if not np.any(ok):
                continue
            w = 1.0 / v[ok]
            sw = float(np.sum(w))
            roi_beta[roi] = float(np.sum(w * b[ok]) / sw)
            roi_se[roi] = float(np.sqrt(1.0 / sw))
            valid[roi] = True
        diag = {"dof":dof, "cXtXc":cXtXc, "n_lhs":n_lhs, "n_rhs":n_rhs, "task_col_start":t0, "task_col_end":t1,
                "lhs_names":[names_task[i] for i in sorted(lhs)], "rhs_names":[names_task[i] for i in sorted(rhs)]}
        return (roi_beta, roi_se, valid, roi_n_vox, diag)

    else:
        raise ValueError("roi_method must be 'stouffer' or 'ivw'")

# -----------------------
# Across-run combination
# -----------------------

def combine_runs_stouffer(z_runs: np.ndarray, valid_mask: np.ndarray):
    z = np.where(valid_mask, z_runs, 0.0)
    n = valid_mask.sum(axis=0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_task = z.sum(axis=0) / np.sqrt(n)
    p_two = 2.0 * (1.0 - norm.cdf(np.abs(z_task)))
    p_greater = 1.0 - norm.cdf(z_task)
    p_less = norm.cdf(z_task)
    z_task = np.where(n > 0, z_task, np.nan)
    p_two = np.where(n > 0, p_two, np.nan)
    p_greater = np.where(n > 0, p_greater, np.nan)
    p_less = np.where(n > 0, p_less, np.nan)
    return z_task, p_two, p_greater, p_less, n.astype(int)


def combine_runs_ivw(beta_runs: np.ndarray, se_runs: np.ndarray, valid_mask: np.ndarray):
    var = se_runs**2
    w = np.where(valid_mask & (var > 0), 1.0 / var, 0.0)
    sw = w.sum(axis=0)
    beta = np.where(sw > 0, (w * beta_runs).sum(axis=0) / sw, np.nan)
    se = np.where(sw > 0, np.sqrt(1.0 / sw), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = beta / se
    p_two = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    p_greater = 1.0 - norm.cdf(z)
    p_less = norm.cdf(z)
    n = (valid_mask & np.isfinite(beta_runs) & np.isfinite(se_runs) & (se_runs > 0)).sum(axis=0)
    return beta, se, z, p_two, p_greater, p_less, n.astype(int)

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Task-level ROI contrast directly from GLM betas (trial-level)")
    ap.add_argument("--glm_root", required=True, help="Root with trial-level *_betas.h5 (GLM outputs)")
    ap.add_argument("--atlas", required=True, help="Glasser dlabel.nii path (Dense64k)")
    ap.add_argument("--out_root", required=True, help="Output root for task-level ROI HDF5 files")
    # selection of trials for LHS vs RHS
    ap.add_argument("--lhs_regex", default=None, help="Regex for LHS trial columns (e.g., '_encoding$')")
    ap.add_argument("--rhs_regex", default=None, help="Regex for RHS trial columns (e.g., '_delay$')")
    ap.add_argument("--lhs_types", nargs="*", default=None, help="Trial types for LHS (e.g., encoding delay)")
    ap.add_argument("--rhs_types", nargs="*", default=None, help="Trial types for RHS")
    ap.add_argument("--lhs_trials", nargs="*", type=int, default=None, help="Explicit trialNumbers for LHS")
    ap.add_argument("--rhs_trials", nargs="*", type=int, default=None, help="Explicit trialNumbers for RHS")
    ap.add_argument("--weight_mode", choices=["mean","sum"], default="mean", help="Weights within each side of the contrast")
    # method/tail
    ap.add_argument("--roi_method", choices=["stouffer","ivw"], default="ivw", help="ROI combine within a run")
    ap.add_argument("--tail", choices=["two-sided","greater","less"], default="two-sided", help="Tail for p-values at aggregation step")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for FDR thresholding")
    # scoping / filtering
    ap.add_argument("--subj", default=None, help="Optional: restrict to a single subject (e.g., sub-01)")
    ap.add_argument("--task", default=None, help="Optional: restrict to a single task (e.g., ctxdm)")
    ap.add_argument("--min_runs", type=int, default=1, help="Minimum runs required to produce a task-level file")
    # outputs
    ap.add_argument("--output_csv", default=None, help="Where to save a CSV summary (one row per (subj, task))")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    glm_root = Path(args.glm_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load atlas once
    atlas = nib.load(str(Path(args.atlas))).get_fdata()
    if atlas.ndim == 2:
        atlas = atlas[0]
    atlas = atlas.astype(int)
    num_regions = int(np.max(atlas))  # assume 1..R labeled

    # Discover GLM betas files
    if args.task:
        patt = f"**/*task-{args.task}*_betas.h5"
    else:
        patt = f"**/*_betas.h5"
    files = sorted(glm_root.glob(patt))
    if args.verbose:
        print(f"Found {len(files)} GLM runs matching pattern: {patt}")

    # Group per (subj,task)
    groups: dict[tuple[str,str], list[Path]] = {}
    for fpath in files:
        subj, task = parse_subj_task_from_name(fpath)
        if args.subj and subj != args.subj:
            continue
        groups.setdefault((subj, task), []).append(fpath)

    rows = []

    for (subj, task), run_files in sorted(groups.items()):
        # per-run ROI holders
        z_list = []
        beta_list = []
        se_list = []
        valid_list = []
        n_roi = num_regions
        run_basenames = []
        run_relpaths = []

        for fpath in run_files:
            try:
                if args.roi_method == 'stouffer':
                    roi_z, valid, roi_n_vox, diag = compute_roi_from_glm(
                        fpath, atlas_labels=atlas, num_regions=num_regions,
                        lhs_regex=args.lhs_regex, rhs_regex=args.rhs_regex,
                        lhs_types=args.lhs_types, rhs_types=args.rhs_types,
                        lhs_trials=args.lhs_trials, rhs_trials=args.rhs_trials,
                        weight_mode=args.weight_mode, roi_method='stouffer')
                    z_list.append(roi_z)
                    valid_list.append(valid)
                else:
                    roi_beta, roi_se, valid, roi_n_vox, diag = compute_roi_from_glm(
                        fpath, atlas_labels=atlas, num_regions=num_regions,
                        lhs_regex=args.lhs_regex, rhs_regex=args.rhs_regex,
                        lhs_types=args.lhs_types, rhs_types=args.rhs_types,
                        lhs_trials=args.lhs_trials, rhs_trials=args.rhs_trials,
                        weight_mode=args.weight_mode, roi_method='ivw')
                    beta_list.append(roi_beta)
                    se_list.append(roi_se)
                    valid_list.append(valid)
                run_basenames.append(fpath.name)
                try:
                    run_relpaths.append(str(fpath.relative_to(glm_root)))
                except Exception:
                    run_relpaths.append(fpath.name)
            except Exception as e:
                if args.verbose:
                    print(f"[skip run] {fpath}: {e}")
                continue

        if len(run_basenames) < args.min_runs:
            if args.verbose:
                print(f"[skip] Not enough runs ({len(run_basenames)}) for {subj}, {task}")
            continue

        # Combine across runs
        if args.roi_method == 'stouffer':
            z_runs = np.vstack(z_list) if z_list else np.empty((0, n_roi))
            valid = np.vstack(valid_list) if valid_list else np.empty((0, n_roi), dtype=bool)
            z_task, p_two, p_greater, p_less, n_per_roi = combine_runs_stouffer(z_runs, valid)
        else:
            beta_runs = np.vstack(beta_list) if beta_list else np.empty((0, n_roi))
            se_runs   = np.vstack(se_list)   if se_list   else np.empty((0, n_roi))
            valid     = np.vstack(valid_list) if valid_list else np.empty((0, n_roi), dtype=bool)
            beta_task, se_task, z_task, p_two, p_greater, p_less, n_per_roi = combine_runs_ivw(beta_runs, se_runs, valid)

        # Pick p for thresholding
        if args.tail == 'two-sided':
            p_use = p_two
        elif args.tail == 'greater':
            p_use = p_greater
        else:
            p_use = p_less

        finite = np.isfinite(p_use)
        sig_mask = np.zeros_like(p_use, dtype=bool)
        sig_mask[finite] = p_use[finite] < args.alpha

        fdr_mask = np.zeros_like(sig_mask, dtype=bool)
        fdr_corrected_p = np.full_like(p_use, np.nan, dtype=float)
        if np.any(finite):
            rej, p_fdr = fdrcorrection(p_use[finite], alpha=args.alpha)
            fdr_mask[finite] = rej
            fdr_corrected_p[finite] = p_fdr

        # Hierarchical out path
        out_dir = out_root / subj / f"task-{task}"
        out_dir.mkdir(parents=True, exist_ok=True)
        label = _build_label(args.lhs_regex, args.rhs_regex, args.lhs_types, args.rhs_types, args.lhs_trials, args.rhs_trials, args.weight_mode)
        out_name = f"{subj}_task-{task}_taskroi_{label}_{args.tail}_{args.roi_method}.h5"
        out_path = out_dir / out_name
        with h5py.File(out_path, 'w') as f:
            if args.roi_method == 'stouffer':
                f.create_dataset('roi_z', data=z_task.astype(np.float64))
                f.create_dataset('roi_p_two', data=p_two.astype(np.float64))
                f.create_dataset('roi_p_greater', data=p_greater.astype(np.float64))
                f.create_dataset('roi_p_less', data=p_less.astype(np.float64))
            else:
                f.create_dataset('roi_beta_ivw', data=beta_task.astype(np.float64))
                f.create_dataset('roi_se_ivw', data=se_task.astype(np.float64))
                f.create_dataset('roi_z', data=z_task.astype(np.float64))
                f.create_dataset('roi_p_two', data=p_two.astype(np.float64))
                f.create_dataset('roi_p_greater', data=p_greater.astype(np.float64))
                f.create_dataset('roi_p_less', data=p_less.astype(np.float64))
            f.create_dataset('sig_mask', data=sig_mask.astype(np.uint8))
            f.create_dataset('fdr_mask', data=fdr_mask.astype(np.uint8))
            f.create_dataset('fdr_corrected_p', data=fdr_corrected_p.astype(np.float64))
            f.create_dataset('n_runs_per_roi', data=n_per_roi.astype(np.int32))
            # attrs
            f.attrs['subj'] = subj
            f.attrs['task'] = task
            f.attrs['label'] = label
            f.attrs['tail'] = args.tail
            f.attrs['roi_method'] = args.roi_method
            f.attrs['alpha'] = float(args.alpha)
            f.attrs['num_regions'] = int(n_roi)
            f.attrs['n_runs_total'] = int(len(run_basenames))
            try:
                f.create_dataset('run_files', data=np.array(run_basenames, dtype=object), dtype=h5py.string_dtype('utf-8'))
                f.create_dataset('run_relpaths', data=np.array(run_relpaths, dtype=object), dtype=h5py.string_dtype('utf-8'))
            except Exception:
                pass

        # Summary row
        rows.append({
            'subj': subj,
            'task': task,
            'label': label,
            'tail': args.tail,
            'roi_method': args.roi_method,
            'n_runs': len(run_basenames),
            'n_sig_rois': int(sig_mask.sum()),
            'n_fdr_sig_rois': int(fdr_mask.sum()),
            'min_p': float(np.nanmin(p_use)) if np.any(finite) else np.nan,
            'max_|z|': float(np.nanmax(np.abs(z_task))) if np.any(np.isfinite(z_task)) else np.nan,
            'out_file': str(out_path),
        })

        if args.verbose:
            print(f"[✓] {subj} {task}: combined {len(run_basenames)} runs -> {out_path}")

    df = pd.DataFrame(rows)
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved task-level ROI summary CSV: {args.output_csv}")

    print(df)


if __name__ == '__main__':
    main()
