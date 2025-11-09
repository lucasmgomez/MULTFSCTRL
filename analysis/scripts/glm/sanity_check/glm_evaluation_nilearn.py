#!/usr/bin/env python3
"""
Fast GLM Evaluation (Voxel-level) for a Single Run

- Randomly sample up to --max_voxels non-zero-variance voxels
- Reconstruct full betas via OLS on saved design_matrix (fast, unwhitened)
- Compute predictions, residuals, R²
- Plot 6 interesting voxels (top-3 R² + 3 random)
- Save plots + short markdown report

Usage:
python evaluate_nilearn_glm_voxels.py \
  --subj sub-01 --task ctxdm --ses ses-001 --run run-01 --tmask 1 \
  --glm_root /project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data/trial_level_betas \
  --fmri_root /project/def-pbellec/xuan/cneuromod.multfs.fmriprep \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/glm_fitting_check_results/nilearn_glm \
  --max_voxels 3000 --seed 123
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")


def get_args():
    p = argparse.ArgumentParser(description="Evaluate voxel-level GLM performance on a random voxel subset")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--task", default="ctxdm")
    p.add_argument("--ses", default="ses-001")
    p.add_argument("--run", default="run-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames dropped at run start (must match GLM)")
    p.add_argument("--glm_root", required=True, help="Folder that contains <subj>/<ses>/func/..._nilearn_betas.h5")
    p.add_argument("--fmri_root", required=True, help="Root of fmriprep outputs (NIfTI preproc)")
    p.add_argument("--output_dir", required=True, help="Where to save plots and report")
    p.add_argument("--max_voxels", type=int, default=3000, help="Randomly sample up to this many voxels")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_glm_h5(h5_path: Path):
    d = {}
    with h5py.File(h5_path, "r") as f:
        d["betas_task"] = f["betas"][:]  # (n_voxels x n_trials)
        d["X"] = f["design_matrix"][:]   # (T_kept x K)
        d["design_col_names"] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f["design_col_names"][:]]
        d["task_regressor_names"] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f.attrs["task_regressor_names"]]
        d["task_col_start"] = int(f.attrs["task_col_start"])
        d["task_col_end"] = int(f.attrs["task_col_end"])
        d["n_voxels"] = int(f.attrs["n_voxels"])
        d["tmask_dropped"] = int(f.attrs["tmask_dropped"])
    return d


def load_fmri_timeseries(fmri_file: Path, tmask: int):
    img = nib.load(str(fmri_file))
    Y = img.get_fdata(dtype=np.float32)  # (X,Y,Z,T)
    T = Y.shape[-1]
    if tmask > 0:
        Y = Y[..., tmask:]
    T_kept = Y.shape[-1]
    return Y.reshape(-1, T_kept).T  # (T_kept x n_voxels)


def compute_r2(Y, Yhat):
    V = Y.shape[1]
    r2 = np.empty(V, dtype=np.float32)
    for v in range(V):
        try:
            r2[v] = r2_score(Y[:, v], Yhat[:, v])
        except Exception:
            r2[v] = np.nan
    return r2


def main():
    args = get_args()
    np.random.seed(args.seed)

    glm_file = Path(args.glm_root) / args.subj / args.ses / "func" / f"{args.subj}_{args.ses}_task-{args.task}_{args.run}_nilearn_betas.h5"
    func_dir = Path(args.fmri_root) / args.subj / args.ses / "func"

    # Resolve NIfTI preproc (run-01 vs run-1)
    run_num = args.run.split("-")[1]
    run_num_short = str(int(run_num))
    candidates = [
        func_dir / f"{args.subj}_{args.ses}_task-{args.task}_run-{run_num}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        func_dir / f"{args.subj}_{args.ses}_task-{args.task}_run-{run_num_short}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        func_dir / f"{args.subj}_{args.ses}_task-{args.task}_run-{run_num}_space-T1w_desc-preproc_bold.nii.gz",
        func_dir / f"{args.subj}_{args.ses}_task-{args.task}_run-{run_num_short}_space-T1w_desc-preproc_bold.nii.gz",
    ]
    fmri_file = None
    for c in candidates:
        if c.exists():
            fmri_file = c
            break
    if fmri_file is None:
        # try wildcard
        hits = list(func_dir.glob(f"{args.subj}_{args.ses}_task-{args.task}_run-{run_num}_space-*_desc-preproc_bold.nii.gz"))
        if not hits:
            hits = list(func_dir.glob(f"{args.subj}_{args.ses}_task-{args.task}_run-{run_num_short}_space-*_desc-preproc_bold.nii.gz"))
        if hits:
            fmri_file = hits[0]
    if fmri_file is None:
        raise FileNotFoundError("fMRI preproc file not found")

    if not glm_file.exists():
        raise FileNotFoundError(f"GLM HDF5 not found: {glm_file}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GLM HDF5:", glm_file)
    G = load_glm_h5(glm_file)
    X = G["X"].astype(np.float32)  # (T_kept x K)
    K = X.shape[1]
    T_kept = X.shape[0]

    print("Loading fMRI time series:", fmri_file)
    Y_full = load_fmri_timeseries(fmri_file, args.tmask)  # (T_kept x n_voxels)
    if Y_full.shape[0] != T_kept:
        raise RuntimeError(f"Time mismatch: design T={T_kept} vs data T={Y_full.shape[0]}")

    # Random sample of non-zero-variance voxels
    stds = np.std(Y_full, axis=0)
    nz = np.where(stds > 0)[0]
    if nz.size == 0:
        raise RuntimeError("All voxels have zero variance after tmask")
    n_sel = min(args.max_voxels, nz.size)
    sampled = np.random.choice(nz, size=n_sel, replace=False)
    sampled.sort()
    print(f"Sampled {n_sel} / {nz.size} non-zero-variance voxels")

    Y = Y_full[:, sampled]  # (T_kept x n_sel)

    # Reconstruct full betas via OLS (unwhitened): β = (XᵀX + λI)⁻¹ Xᵀ Y
    lam = 1e-6
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX + lam * np.eye(K, dtype=np.float32))
    betas_full = (XtX_inv @ (X.T @ Y)).T  # (n_sel x K)

    # Predictions & residuals
    Yhat = X @ betas_full.T  # (T_kept x n_sel)
    resid = Y - Yhat
    r2 = compute_r2(Y, Yhat)

    # Choose 6 interesting voxels: top-3 R² + 3 random
    finite_idx = np.where(np.isfinite(r2))[0]
    if finite_idx.size == 0:
        interesting = np.arange(min(6, Y.shape[1]))
    else:
        top_n = min(3, finite_idx.size)
        top_local = finite_idx[np.argsort(r2[finite_idx])[-top_n:]]
        remain = np.setdiff1d(np.arange(Y.shape[1]), top_local, assume_unique=False)
        rand_n = min(3, remain.size)
        rand_local = np.random.choice(remain, size=rand_n, replace=False) if rand_n > 0 else np.array([], dtype=int)
        interesting = np.unique(np.concatenate([top_local, rand_local]))[:6]

    interesting_global = sampled[interesting]

    # === Plots ===
    # 1) Actual vs Predicted (6 voxels)
    fig, axes = plt.subplots(len(interesting), 1, figsize=(12, 3*len(interesting)))
    if len(interesting) == 1:
        axes = [axes]
    t = np.arange(T_kept)
    for i, idx in enumerate(interesting):
        axes[i].plot(t, Y[:, idx], label="Actual", alpha=0.85)
        axes[i].plot(t, Yhat[:, idx], label="Predicted", alpha=0.85)
        axes[i].set_title(f"Voxel {interesting_global[i]}: Actual vs Predicted")
        axes[i].set_xlabel("Time (frames after tmask)")
        axes[i].set_ylabel("Signal")
        axes[i].grid(alpha=0.3)
        axes[i].legend()
    plt.tight_layout()
    pred_png = out_dir / f"{args.subj}_{args.ses}_task-{args.task}_{args.run}_vox_pred_vs_actual.png"
    plt.savefig(pred_png, dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Residuals (up to 3 voxels)
    m = min(3, len(interesting))
    fig, axes = plt.subplots(2, m, figsize=(5*m, 8))
    if m == 1:
        axes = np.array(axes).reshape(2, 1)
    for i in range(m):
        v = interesting[i]
        axes[0, i].plot(t, resid[:, v])
        axes[0, i].set_title(f"Voxel {interesting_global[i]} Residuals")
        axes[0, i].grid(alpha=0.3)
        vals = resid[:, v]
        axes[1, i].hist(vals, bins=30, density=True, alpha=0.8)
        mu, sd = np.mean(vals), np.std(vals)
        xs = np.linspace(vals.min(), vals.max(), 100)
        ys = (1/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mu)/sd)**2) if sd > 0 else np.zeros_like(xs)
        axes[1, i].plot(xs, ys, "r--", label="Normal fit")
        axes[1, i].legend()
        axes[1, i].grid(alpha=0.3)
    plt.tight_layout()
    resid_png = out_dir / f"{args.subj}_{args.ses}_task-{args.task}_{args.run}_vox_residuals.png"
    plt.savefig(resid_png, dpi=300, bbox_inches="tight")
    plt.close()

    # 3) R² distribution on sampled voxels
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(r2[np.isfinite(r2)], bins=50, density=True, alpha=0.85)
    if np.isfinite(r2).any():
        axs[0].axvline(np.nanmean(r2), linestyle="--", label=f"Mean {np.nanmean(r2):.3f}")
        axs[0].axvline(np.nanmedian(r2), linestyle="--", label=f"Median {np.nanmedian(r2):.3f}")
    axs[0].set_title("R² Distribution (Sampled Voxels)")
    axs[0].set_xlabel("R²"); axs[0].set_ylabel("Density"); axs[0].legend(); axs[0].grid(alpha=0.3)

    axs[1].boxplot(r2[np.isfinite(r2)], vert=True)
    axs[1].set_title("R² Box Plot (Sampled Voxels)"); axs[1].set_ylabel("R²"); axs[1].grid(alpha=0.3)
    plt.tight_layout()
    r2_png = out_dir / f"{args.subj}_{args.ses}_task-{args.task}_{args.run}_vox_r2_distribution.png"
    plt.savefig(r2_png, dpi=300, bbox_inches="tight")
    plt.close()

    # === Report ===
    report = []
    report.append(f"# GLM Evaluation Report (Voxel-level)")
    report.append(f"**Subject:** {args.subj}  \n**Task:** {args.task}  \n**Session:** {args.ses}  \n**Run:** {args.run}")
    report.append("")
    report.append("## Data")
    report.append(f"- Time points (after tmask): {T_kept}")
    report.append(f"- Total non-zero-variance voxels: {nz.size}")
    report.append(f"- Sampled voxels: {n_sel}")
    report.append("")
    report.append("## Model Fit (Sampled Voxels)")
    report.append(f"- Mean R²: {np.nanmean(r2):.3f}")
    report.append(f"- Median R²: {np.nanmedian(r2):.3f}")
    report.append(f"- R² std: {np.nanstd(r2):.3f}")
    report.append(f"- R² range: [{np.nanmin(r2):.3f}, {np.nanmax(r2):.3f}]")
    for thr in (0.0, 0.1, 0.2):
        mask = r2 > thr
        report.append(f"- Voxels with R² > {thr:.1f}: {int(np.nansum(mask))}/{r2.size} ({100*np.nanmean(mask):.1f}%)")
    report.append("")
    report.append("## Interesting Voxels (global flat indices)")
    report.append(f"- {interesting_global.tolist()}")
    report_path = out_dir / f"{args.subj}_{args.ses}_task-{args.task}_{args.run}_vox_evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print("\n=== Summary (Sampled Voxels) ===")
    print(f"Sampled {n_sel} voxels; mean R²={np.nanmean(r2):.3f}, median={np.nanmedian(r2):.3f}, "
          f"range=[{np.nanmin(r2):.3f}, {np.nanmax(r2):.3f}]")
    print("Saved:")
    print("  -", pred_png)
    print("  -", resid_png)
    print("  -", r2_png)
    print("  -", report_path)


if __name__ == "__main__":
    main()
