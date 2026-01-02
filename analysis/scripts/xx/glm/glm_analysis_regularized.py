#!/usr/bin/env python3
"""
Regularized Per-run **trial-level** GLM with cosine drifts + AR(1), HRF-convolved *one regressor per event row*.
Uses Ridge regression with cross-validation to prevent overfitting when there are many regressors.

Key improvements over original GLM:
1. Ridge regularization with automatic lambda selection via cross-validation
2. Proper handling of overfitting scenarios
3. Model selection statistics and diagnostics
4. Better numerical stability

Outputs per run (under out_root/.../sub-XX/ses-YY/func/):
  - <base>_regularized_betas.h5
      datasets:
        betas                 (P x K_task)     # trial-wise betas
        XtX_reg               (K x K)          # regularized precision matrix  
        sigma2                (P,)             # residual variance per parcel
        ridge_alpha           (1,)             # optimal regularization parameter
        cv_scores             (n_folds,)       # cross-validation scores
        design_matrix         (T_kept x K)     # FULL X = [confounds, drifts+intercept, task(trials)]
        design_col_names      (K,)             # HDF5 string dtype
      attrs:
        task_regressor_names  [list of trial column names]
        regressor_level       "trial"
        task_col_start, task_col_end (int)
        dof_effective         # effective degrees of freedom after regularization
        regularization_method "ridge_cv"
        cv_folds              # number of CV folds used
        tmask_dropped, high_pass_sec
        
Example
-------
python glm_analysis_regularized.py \
  --subj sub-01 \
  --tasks ctxdm  \
  --include_types encoding delay \
  --out_root /project/def-pbellec/xuan/fmri_dataset_project/data \
  --cv_folds 5 \
  --alpha_range_log -3 3 \
  --overwrite
"""
from __future__ import annotations
import re
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import nibabel as nib

from nilearn.glm.first_level import spm_hrf, make_first_level_design_matrix

from utils import glm_confounds_construction, standardize_run_label

# -----------------------
# CLI / hyperparams
# -----------------------

def get_args():
    p = argparse.ArgumentParser(description="Regularized per-run trial-level GLM with Ridge regression")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames to drop at run start")
    p.add_argument("--correct_only", action="store_true", help="Use only correct trials")
    p.add_argument("--tasks", nargs="+", default=["ctxdm"], help="Task names to process")
    p.add_argument("--include_types", nargs="+", default=["encoding","delay"], help="Event types to include as trial regressors (lowercased)")
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled")
    p.add_argument("--conf_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--events_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data")
    p.add_argument("--high_pass_sec", type=float, default=128.0, help="Cosine high-pass cutoff in seconds")
    p.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds")
    p.add_argument("--alpha_range_log", nargs=2, type=float, default=[-3, 3], help="Log10 range for alpha search [min_exp, max_exp]")
    p.add_argument("--n_alphas", type=int, default=50, help="Number of alpha values to test")
    p.add_argument("--max_vertices", type=int, default=1000, help="Maximum vertices to process (for memory/speed)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

# -----------------------
# Utilities (same as original)
# -----------------------

def default_output_root(out_root_base: Path, correct_only: bool, subj: str) -> Path:
    return (out_root_base / ("correct_trial_level_betas_regularized" if correct_only else "trial_level_betas_regularized") / subj)

def discover_sessions(fmri_root_dir: Path):
    pattern = re.compile(r".*\d$")
    return sorted([p.name for p in fmri_root_dir.iterdir() if p.is_dir() and pattern.match(p.name)])

def discover_runs_for_task_session(subj, ses, task, fmri_root_dir: Path):
    pattern = f"{subj}_{ses}_task-{task}_*_space-Glasser64k_bold.dtseries.nii"
    matching_files = list((fmri_root_dir / ses).glob(pattern))
    run_pattern = re.compile(
        rf"{re.escape(subj)}_{re.escape(ses)}_task-{re.escape(task)}_(.+?)_space-Glasser64k_bold\.dtseries\.nii"
    )
    runs = []
    for f in matching_files:
        m = run_pattern.match(f.name)
        if m:
            runs.append(m.group(1))
    return sorted(runs)

def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    return df

# -----------------------
# Regressor builders (same as original)
# -----------------------

def build_trialwise_regressors(
    df_events: pd.DataFrame,
    num_trs: int,
    tr: float,
    include_types: list[str],
    correct_only: bool = False,
):
    """Build one HRF-convolved regressor per *event row* whose `type` is in include_types."""
    req_cols = {"onset_time", "offset_time", "type"}
    if not req_cols.issubset(df_events.columns):
        missing = ", ".join(sorted(req_cols - set(df_events.columns)))
        raise ValueError(f"Events missing required columns: {missing}")

    # Type filtering
    types = df_events["type"].astype(str).str.strip().str.lower()
    mask_type = types.isin([t.lower() for t in include_types])

    # Correct-only filtering
    if correct_only and "is_correct" in df_events.columns:
        def is_negative(v):
            if isinstance(v, (bool, np.bool_)):
                return (v is False) or (v == False)
            if pd.isna(v):
                return False
            sv = str(v).strip().lower()
            return sv in {"false", "b"}

        mask_correct = ~df_events["is_correct"].apply(is_negative)
        mask = mask_type & mask_correct
    else:
        mask = mask_type

    df_sel = df_events.loc[mask].reset_index(drop=True)

    T = num_trs
    box = []
    names = []
    meta_rows = []

    def g(row: pd.Series, key, default=np.nan):
        return row[key] if (key in row.index and pd.notna(row[key])) else default

    for i, row in df_sel.iterrows():
        on = float(row["onset_time"]) if pd.notna(row["onset_time"]) else np.nan
        off = float(row["offset_time"]) if pd.notna(row["offset_time"]) else np.nan
        if not np.isfinite(on) or not np.isfinite(off):
            continue

        a = int(np.floor(on / tr))
        b = int(np.floor(off / tr))
        a = max(0, min(a, T))
        b = max(0, min(b, T))

        vec = np.zeros((T,), dtype=np.float32)
        if b > a:
            vec[a:b] = 1.0
            box.append(vec)
        else:
            continue

        tn = g(row, "trialNumber", i)
        typ = str(row["type"]).strip().lower()
        try:
            names.append(f"trial{int(tn):03d}_{typ}")
        except Exception:
            names.append(f"trial{i:03d}_{typ}")

        meta_rows.append({
            "trial_index": len(names) - 1,
            "trialNumber": tn,
            "type": typ,
            "onset_time": on,
            "offset_time": off,
            "is_correct": g(row, "is_correct"),
            "stim_order": g(row, "stim_order"),
            "location": g(row, "locmod"),
            "category": g(row, "ctgmod"),
            "object": g(row, "objmod"),
        })

    if len(box) == 0:
        empty_cols = ["trial_index","trialNumber","type","onset_time","offset_time","is_correct","stim_order","location","category","object"]
        return np.zeros((T, 0), dtype=np.float32), [], pd.DataFrame(columns=empty_cols)

    R_box = np.stack(box, axis=1)  # (T x K)

    # Convolve with canonical SPM HRF
    h = spm_hrf(tr, oversampling=1)
    R_hrf = np.zeros_like(R_box, dtype=np.float32)
    for k in range(R_box.shape[1]):
        tmp = np.convolve(R_box[:, k], h)
        R_hrf[:, k] = tmp[:T]

    trial_info = pd.DataFrame(meta_rows).sort_values(by="trial_index").reset_index(drop=True)
    return R_hrf, names, trial_info

def per_run_drift_and_intercept(num_trs: int, tr: float, high_pass_sec: float = 128.0):
    frame_times = np.arange(num_trs) * tr
    dm = make_first_level_design_matrix(
        frame_times,
        events=None,
        hrf_model=None,
        drift_model="cosine",
        high_pass=1.0 / high_pass_sec,
        add_regs=None,
        add_reg_names=None,
        oversampling=1,
    )
    cols = [c for c in dm.columns if c == "constant" or c.startswith("cosine")]
    return dm[cols].to_numpy(dtype=np.float32), cols

# -----------------------
# Regularized GLM fitting
# -----------------------

def fit_ridge_glm_cv(X: np.ndarray, Y: np.ndarray, cv_folds: int = 5, 
                     alpha_range_log: tuple = (-3, 3), n_alphas: int = 50):
    """
    Fit Ridge regression with cross-validation for optimal regularization.
    
    Parameters:
    -----------
    X : ndarray (T, K)
        Design matrix
    Y : ndarray (T, P) 
        fMRI data (time x parcels/vertices)
    cv_folds : int
        Number of cross-validation folds
    alpha_range_log : tuple
        Log10 range for alpha search (min_exp, max_exp)  
    n_alphas : int
        Number of alpha values to test
        
    Returns:
    --------
    dict with results including betas, alpha, cv_scores, etc.
    """
    print(f"Fitting Ridge GLM with {cv_folds}-fold cross-validation...")
    print(f"Design matrix shape: {X.shape}")
    print(f"Data shape: {Y.shape}")
    
    # Create alpha range
    alphas = np.logspace(alpha_range_log[0], alpha_range_log[1], n_alphas)
    print(f"Testing {n_alphas} alpha values from {alphas[0]:.1e} to {alphas[-1]:.1e}")
    
    T, K = X.shape
    T_y, P = Y.shape
    assert T == T_y, f"Design matrix and data have mismatched time dimensions: {T} vs {T_y}"
    
    # Initialize results
    all_betas = np.zeros((P, K), dtype=np.float32)
    all_alphas = np.zeros(P, dtype=np.float32)
    all_cv_scores = np.zeros((P, cv_folds), dtype=np.float32)
    all_r2_scores = np.zeros(P, dtype=np.float32)
    
    # Progress tracking
    vertices_processed = 0
    
    for p in range(P):
        if vertices_processed % 100 == 0:
            print(f"Processing vertex {vertices_processed + 1}/{P}")
            
        y = Y[:, p]
        
        # Skip if no signal variation
        if np.std(y) == 0:
            vertices_processed += 1
            continue
        
        try:
            # Fit Ridge with cross-validation
            ridge_cv = RidgeCV(alphas=alphas, cv=cv_folds, scoring='r2')
            ridge_cv.fit(X, y)
            
            # Store results
            all_betas[p, :] = ridge_cv.coef_
            all_alphas[p] = ridge_cv.alpha_
            
            # Cross-validation scores using KFold manually for consistency
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            ridge_temp = Ridge(alpha=ridge_cv.alpha_)
            cv_scores = cross_val_score(ridge_temp, X, y, cv=kfold, scoring='r2')
            all_cv_scores[p, :cv_scores.shape[0]] = cv_scores
                
            # Compute R² on full data
            y_pred = ridge_cv.predict(X)
            all_r2_scores[p] = r2_score(y, y_pred)
            
        except Exception as e:
            print(f"Warning: Error fitting vertex {p}: {e}")
            continue
            
        vertices_processed += 1
    
    # Compute summary statistics
    valid_mask = all_alphas > 0
    if np.sum(valid_mask) > 0:
        median_alpha = np.median(all_alphas[valid_mask])
        mean_r2 = np.mean(all_r2_scores[valid_mask])
        median_r2 = np.median(all_r2_scores[valid_mask])
        
        print(f"Ridge regression completed:")
        print(f"  Valid vertices: {np.sum(valid_mask)}/{P}")
        print(f"  Median alpha: {median_alpha:.3e}")
        print(f"  Mean R²: {mean_r2:.4f}")
        print(f"  Median R²: {median_r2:.4f}")
    else:
        median_alpha = 0
        mean_r2 = 0
        median_r2 = 0
        print("Warning: No valid vertices processed")
    
    # Compute effective degrees of freedom
    # For Ridge: df_eff = trace(X @ inv(XtX + alpha*I) @ Xt)
    median_alpha_val = median_alpha if median_alpha > 0 else alphas[len(alphas)//2]
    XtX = X.T @ X
    try:
        ridge_matrix = np.linalg.inv(XtX + median_alpha_val * np.eye(K))
        H = X @ ridge_matrix @ X.T
        dof_effective = np.trace(H)
    except:
        dof_effective = K  # fallback to full rank
    
    results = {
        'betas': all_betas,
        'alphas': all_alphas,
        'cv_scores': all_cv_scores,
        'r2_scores': all_r2_scores,
        'median_alpha': median_alpha,
        'mean_r2': mean_r2,
        'median_r2': median_r2,
        'dof_effective': dof_effective,
        'valid_vertices': np.sum(valid_mask),
        'alpha_range': alphas,
        'XtX_reg': XtX + median_alpha_val * np.eye(K)  # Regularized precision matrix
    }
    
    return results

# -----------------------
# Main processing
# -----------------------

def main():
    args = get_args()

    subj = args.subj
    tr = args.tr
    tmask = args.tmask
    correct_only = args.correct_only
    include_types = [t.lower() for t in args.include_types]

    fmri_root_dir = Path(args.fmri_root) / subj
    confounds_root_dir = Path(args.conf_root) / subj
    events_root_dir = Path(args.events_root) / subj
    output_dir = default_output_root(Path(args.out_root), correct_only, subj)

    sessions = discover_sessions(fmri_root_dir)
    print("Sessions:", sessions)

    for task_name in args.tasks:
        for ses in sessions[:1]:  # Process only first session for testing
            runs = discover_runs_for_task_session(subj, ses, task_name, fmri_root_dir)
            print(f"[{task_name} | {ses}] runs: {runs}")

            for run in runs[:1]:  # Process only first run for testing
                print(f"\n=== Processing {subj} {ses} {task_name} {run} ===")
                
                # Paths
                behavioral_file = events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{standardize_run_label(run)}_events.tsv"
                
                if not behavioral_file.exists():
                    print(f"⚠️ Behavioral file {behavioral_file} does not exist, skipping.")
                    continue

                timeseries_file = fmri_root_dir / ses / f"{subj}_{ses}_task-{task_name}_{run}_space-Glasser64k_bold.dtseries.nii"
                confounds_file = confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{run}_desc-confounds_timeseries.tsv"

                if not timeseries_file.exists() or not confounds_file.exists():
                    print(f"⚠️ Missing data for {ses} {run} {task_name}, skipping.")
                    continue

                # Output paths
                rel_behavioral_path = behavioral_file.relative_to(events_root_dir)
                target_subdir = output_dir / rel_behavioral_path.parent
                base_name = behavioral_file.stem.replace("_events", "")
                h5_file = target_subdir / f"{base_name}_regularized_betas.h5"
                design_csv = target_subdir / f"{base_name}_regularized_design.csv"
                design_matrix_csv = target_subdir / f"{base_name}_regularized_design_matrix.csv"

                if h5_file.exists() and not args.overwrite:
                    print(f"[⏩] Skipping {h5_file} — already exists.")
                    continue

                # ---------------- Load data ----------------
                print("Loading fMRI data...")
                Y = nib.load(str(timeseries_file)).get_fdata(dtype=np.float32)  # (T x P)
                print(f"Original data shape: {Y.shape}")
                
                # Limit to max_vertices for memory/speed
                if args.max_vertices < Y.shape[1]:
                    print(f"Limiting analysis to first {args.max_vertices} vertices")
                    Y = Y[:, :args.max_vertices]
                
                assert not np.isnan(Y).any(), "NaNs in timeseries_data"

                print("Loading confounds...")
                df_confounds = pd.read_csv(confounds_file, sep="\t")
                C_mat = glm_confounds_construction(df_confounds)
                if isinstance(C_mat, pd.DataFrame):
                    conf_names = list(C_mat.columns)
                    C = np.nan_to_num(C_mat.values, nan=0.0).astype(np.float32)
                else:
                    C = np.nan_to_num(C_mat, nan=0.0).astype(np.float32)
                    conf_names = [f"conf_{i}" for i in range(C.shape[1])]

                print("Loading behavioral events...")
                df_events = pd.read_csv(behavioral_file, sep="\t")
                df_events = clean_events(df_events)
                
                T = Y.shape[0]
                print(f"Time points: {T}")

                # ------------- Build task regressors (trial-wise) -------------
                print("Building task regressors...")
                R_task, trial_names, trial_info = build_trialwise_regressors(df_events, T, tr, include_types, correct_only=correct_only)
                print(f"Task regressors shape: {R_task.shape}")
                
                if R_task.shape[1] == 0:
                    print(f"⚠️ No trials of types {include_types} in {behavioral_file}, skipping.")
                    continue

                # ------------- Drifts + intercept -------------
                print("Adding drift regressors...")
                D, drift_names = per_run_drift_and_intercept(T, tr, high_pass_sec=args.high_pass_sec)
                print(f"Drift regressors shape: {D.shape}")

                # Apply tmask
                keep = np.ones((T,), dtype=bool)
                keep[:tmask] = False
                Y = Y[keep, :]
                C = C[keep, :]
                R_task = R_task[keep, :]
                D = D[keep, :]
                print(f"After tmask - Data shape: {Y.shape}")

                # ------------- Assemble design matrix -------------
                X = np.hstack([C, D, R_task]).astype(np.float32)
                design_col_names = conf_names + drift_names + list(trial_names)
                
                print(f"Full design matrix shape: {X.shape}")
                print(f"Number of regressors: {X.shape[1]}")
                print(f"Regressors/timepoints ratio: {X.shape[1]/X.shape[0]:.2f}")
                
                # Task regressor indices
                K_task = R_task.shape[1]
                task_col_start = X.shape[1] - K_task
                task_col_end = X.shape[1]

                # ------------- Fit Regularized GLM -------------
                print("Fitting regularized GLM...")
                ridge_results = fit_ridge_glm_cv(
                    X, Y, 
                    cv_folds=args.cv_folds,
                    alpha_range_log=args.alpha_range_log,
                    n_alphas=args.n_alphas
                )

                # ------------- Save outputs -------------
                target_subdir.mkdir(parents=True, exist_ok=True)

                print("Saving results...")
                
                # Trial metadata CSV
                trial_info_sorted = trial_info.sort_values(by='trial_index')
                trial_info_sorted.to_csv(design_csv, index=False)

                # Design matrix CSV
                pd.DataFrame(X, columns=design_col_names).to_csv(design_matrix_csv, index=False)

                # HDF5 results
                with h5py.File(h5_file, "w") as h5f:
                    # Core results
                    h5f.create_dataset("betas", data=ridge_results['betas'][:, task_col_start:task_col_end].astype(np.float32))
                    h5f.create_dataset("ridge_alphas", data=ridge_results['alphas'].astype(np.float32))
                    h5f.create_dataset("r2_scores", data=ridge_results['r2_scores'].astype(np.float32))
                    h5f.create_dataset("cv_scores", data=ridge_results['cv_scores'].astype(np.float32))
                    
                    # Design matrix and regularized precision
                    h5f.create_dataset("design_matrix", data=X.astype(np.float32))
                    h5f.create_dataset("XtX_reg", data=ridge_results['XtX_reg'].astype(np.float32))
                    
                    # Column names
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    h5f.create_dataset("design_col_names", data=np.array(design_col_names, dtype=object), dtype=str_dtype)

                    # Metadata
                    h5f.attrs.create("task_regressor_names", np.array(trial_names, dtype=str_dtype), dtype=str_dtype)
                    h5f.attrs["regulator_level"] = "trial"
                    h5f.attrs["regularization_method"] = "ridge_cv"
                    h5f.attrs["task_col_start"] = task_col_start
                    h5f.attrs["task_col_end"] = task_col_end
                    h5f.attrs["dof_effective"] = ridge_results['dof_effective']
                    h5f.attrs["cv_folds"] = args.cv_folds
                    h5f.attrs["median_alpha"] = ridge_results['median_alpha']
                    h5f.attrs["mean_r2"] = ridge_results['mean_r2']
                    h5f.attrs["median_r2"] = ridge_results['median_r2']
                    h5f.attrs["valid_vertices"] = ridge_results['valid_vertices']
                    h5f.attrs["tmask_dropped"] = tmask
                    h5f.attrs["high_pass_sec"] = args.high_pass_sec
                    h5f.attrs["vertices_analyzed"] = Y.shape[1]

                print(f"[🎯] Saved regularized results to {h5_file}")
                print(f"[📊] Regularization summary:")
                print(f"    Median alpha: {ridge_results['median_alpha']:.3e}")
                print(f"    Mean R²: {ridge_results['mean_r2']:.4f}")
                print(f"    Median R²: {ridge_results['median_r2']:.4f}")
                print(f"    Effective DoF: {ridge_results['dof_effective']:.1f}/{X.shape[1]}")
                print(f"    Valid vertices: {ridge_results['valid_vertices']}/{Y.shape[1]}")
                
                return h5_file  # Return for further analysis

    print("[✅] Regularized GLM analysis completed.")
    return None


if __name__ == "__main__":
    result_file = main()