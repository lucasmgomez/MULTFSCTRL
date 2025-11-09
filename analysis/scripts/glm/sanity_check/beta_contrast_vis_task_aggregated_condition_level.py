#!/usr/bin/env python3
"""
Aggregate per-run ROI contrast stats into task-level ROI stats (within-subject),
for GLM outputs with two task regressors ('encoding', 'delay').

Inputs: ROI stats HDF5 files produced by roi_contrast_glasser.py, which contain
per-run ROI arrays (either Stouffer z's or IVW betas+SEs). This script groups all
runs by (subject, task) and combines them into a single task-level ROI result.

Supported aggregation modes (match the per-run --roi_method):
  • stouffer  — combine run-level signed z per ROI via Stouffer's method across runs
                 Z_task = sum(Z_run) / sqrt(n_runs)
  • ivw       — inverse-variance–weighted (fixed-effect) average of run-level ROI betas
                 w_run = 1/SE_run^2;  beta_task = sum(w*beta)/sum(w);  SE_task = sqrt(1/sum(w))
                 Z_task = beta_task / SE_task

Thresholding tail:
  - two-sided (default), greater (>0), less (<0). FDR applied across ROIs per task.

Outputs (mirrors input tree under --out_root, but grouped at task level):
  sub-XX/task-<task>_taskroi_<contrast>_<tail>_<method>.h5
     datasets:
       For stouffer: roi_z, roi_p_two, roi_p_greater, roi_p_less, n_runs_per_roi
       For ivw:      roi_beta_ivw, roi_se_ivw, roi_z, roi_p_two, roi_p_greater, roi_p_less, n_runs_per_roi
       Common:       sig_mask, fdr_mask, fdr_corrected_p
     attrs: subj, task, contrast, tail, roi_method, alpha, num_regions, n_runs_total
     attrs: run_files (list of contributing run file basenames)

Also writes a CSV summary with one row per (subj, task).

Examples
--------
# Aggregate encoding - delay across runs (ivw), two-sided
python beta_contrast_vis_task_aggregated_condition_level.py \
  --roi_stats_root /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/roi_stats \
  --out_root       /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/task_level_roi_stats \
  --contrast enc-minus-delay --tail two-sided --roi_method ivw \
  --output_csv     /project/def-pbellec/xuan/fmri_dataset_project/data/condition_level_betas/task_level_roi_stats/summary_enc_minus_delay.csv

# Aggregate encoding > 0 (IVW), one-sided (greater)
python beta_contrast_vis_task_aggregated_condition_level.py \
  --roi_stats_root /project/.../roi_stats \
  --out_root       /project/.../task_level_roi_stats \
  --contrast enc --tail greater --roi_method ivw
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import h5py
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection


# -----------------------
# Parse helpers
# -----------------------

def parse_subj_task_from_name(path: Path):
    """Extract subj (e.g., 'sub-01') and task (e.g., 'ctxdm') from filename.
    Assumes original stem started with sub-XX_..._task-<task>_...
    """
    stem = path.stem  # e.g., sub-01_ses-003_task-ctxdm_run-01_roi_stats_enc-minus-delay_two-sided
    # Grab only prefix up to _roi_stats
    stem = stem.split("_roi_stats")[0]
    # subj
    msub = re.search(r"(sub-[^_]+)", stem)
    subj = msub.group(1) if msub else "unknown"
    # task
    mt = re.search(r"task-([^_]+)", stem)
    task = mt.group(1) if mt else "unknown"
    return subj, task


# -----------------------
# Combine across runs
# -----------------------

def combine_runs_stouffer(z_runs: np.ndarray, valid_mask: np.ndarray):
    """z_runs shape (n_runs, n_roi). valid_mask same shape (bool). Returns z_task, p_two, p_greater, p_less, n_per_roi."""
    # replace invalid with 0 when summing; count valids
    z = np.where(valid_mask, z_runs, 0.0)
    n = valid_mask.sum(axis=0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_task = z.sum(axis=0) / np.sqrt(n)
    # p-values
    p_two = 2.0 * (1.0 - norm.cdf(np.abs(z_task)))
    p_greater = 1.0 - norm.cdf(z_task)
    p_less = norm.cdf(z_task)
    # where n==0, set to NaN
    z_task = np.where(n > 0, z_task, np.nan)
    p_two = np.where(n > 0, p_two, np.nan)
    p_greater = np.where(n > 0, p_greater, np.nan)
    p_less = np.where(n > 0, p_less, np.nan)
    return z_task, p_two, p_greater, p_less, n.astype(int)


def combine_runs_ivw(beta_runs: np.ndarray, se_runs: np.ndarray, valid_mask: np.ndarray):
    """IVW over runs: weights w=1/se^2. Shapes (n_runs, n_roi). Returns beta, se, z, p_two, p_greater, p_less, n_per_roi."""
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
    ap = argparse.ArgumentParser(description="Aggregate per-run ROI stats into task-level ROI stats")
    ap.add_argument("--roi_stats_root", required=True, help="Root folder containing per-run ROI HDF5 files (_roi_stats_*.h5)")
    ap.add_argument("--out_root", required=True, help="Output root directory for task-level ROI HDF5 files")
    ap.add_argument("--contrast", default="enc-minus-delay", choices=["enc","delay","enc-minus-delay","delay-minus-enc"], help="Contrast to aggregate")
    ap.add_argument("--tail", default="two-sided", choices=["two-sided","greater","less"], help="Tail for significance")
    ap.add_argument("--roi_method", default="stouffer", choices=["stouffer","ivw"], help="Which per-run ROI method to aggregate")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for FDR")
    ap.add_argument("--output_csv", default=None, help="Path for CSV summary (one row per (subj, task))")
    ap.add_argument("--subj", default=None, help="Optional: restrict to a single subject (e.g., sub-01)")
    ap.add_argument("--min_runs", type=int, default=1, help="Minimum runs required to produce a task-level file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    roi_root = Path(args.roi_stats_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover candidate files matching contrast & tail; we'll still verify attrs
    patt = f"**/*_roi_stats_{args.contrast}_{args.tail}.h5"
    files = sorted(roi_root.glob(patt))
    if args.verbose:
        print(f"Found {len(files)} ROI files matching contrast/tail")

    # Group by (subj, task)
    groups = {}
    for fpath in files:
        subj, task = parse_subj_task_from_name(fpath)
        if args.subj and subj != args.subj:
            continue
        groups.setdefault((subj, task), []).append(fpath)

    rows = []

    for (subj, task), run_files in sorted(groups.items()):
        # Load per-run arrays and check roi_method attribute
        z_list = []
        beta_list = []
        se_list = []
        valid_list = []
        roi_method_ok = True
        n_roi = None
        run_basenames = []

        for fpath in run_files:
            with h5py.File(fpath, 'r') as f:
                method = f.attrs.get('roi_method', 'stouffer')
                contrast = f.attrs.get('contrast', '')
                tail = f.attrs.get('tail', '')
                if method != args.roi_method or contrast != args.contrast or tail != args.tail:
                    # skip files with mismatched method/contrast/tail
                    continue
                if n_roi is None:
                    # choose length from available dataset
                    if args.roi_method == 'stouffer':
                        n_roi = int(f['roi_z'].shape[0])
                    else:
                        n_roi = int(f['roi_beta_ivw'].shape[0])
                if args.roi_method == 'stouffer':
                    arr = f['roi_z'][()]
                    valid = np.isfinite(arr)
                    z_list.append(arr)
                    valid_list.append(valid)
                else:  # ivw
                    b = f['roi_beta_ivw'][()]
                    s = f['roi_se_ivw'][()]
                    valid = np.isfinite(b) & np.isfinite(s) & (s > 0)
                    beta_list.append(b)
                    se_list.append(s)
                    valid_list.append(valid)
                run_basenames.append(fpath.name)

        if n_roi is None:
            if args.verbose:
                print(f"[skip] No matching files for {subj}, {task}")
            continue

        if len(run_basenames) < args.min_runs:
            if args.verbose:
                print(f"[skip] Not enough runs ({len(run_basenames)}) for {subj}, {task}")
            continue

        # Stack arrays to (n_runs, n_roi)
        if args.roi_method == 'stouffer':
            z_runs = np.vstack(z_list) if z_list else np.empty((0, n_roi))
            valid = np.vstack(valid_list) if valid_list else np.empty((0, n_roi), dtype=bool)
            z_task, p_two, p_greater, p_less, n_per_roi = combine_runs_stouffer(z_runs, valid)
        else:
            beta_runs = np.vstack(beta_list) if beta_list else np.empty((0, n_roi))
            se_runs   = np.vstack(se_list)   if se_list   else np.empty((0, n_roi))
            valid     = np.vstack(valid_list) if valid_list else np.empty((0, n_roi), dtype=bool)
            beta_task, se_task, z_task, p_two, p_greater, p_less, n_per_roi = combine_runs_ivw(beta_runs, se_runs, valid)

        # Pick p according to tail for thresholding
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

        # Write HDF5
        out_dir = out_root / subj
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{subj}_task-{task}_taskroi_{args.contrast}_{args.tail}_{args.roi_method}.h5"
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
            f.attrs['contrast'] = args.contrast
            f.attrs['tail'] = args.tail
            f.attrs['roi_method'] = args.roi_method
            f.attrs['alpha'] = float(args.alpha)
            f.attrs['num_regions'] = int(n_roi)
            f.attrs['n_runs_total'] = int(len(run_basenames))
            try:
                f.create_dataset('run_files', data=np.array(run_basenames, dtype=object), dtype=h5py.string_dtype('utf-8'))
            except Exception:
                pass

        # Summary row
        rows.append({
            'subj': subj,
            'task': task,
            'contrast': args.contrast,
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
