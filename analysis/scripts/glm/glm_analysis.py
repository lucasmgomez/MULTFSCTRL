#!/usr/bin/env python3
"""
Per-run **trial-level** GLM with cosine drifts, HRF-convolved *one regressor per event row*.

Switch estimator with --estimator {wls_ar1, ols}:
- wls_ar1 (default): GLS via AR(1) whitening; saves estimator="WLS_AR1" and rho_ar1
- ols:        OLS (optionally with tiny ridge via --ridge); saves estimator="OLS" and rho_ar1=0.0

Simple & robust path resolution:
- Discover runs from the **timeseries** filenames and use that run label everywhere.
- For events/confounds, try small aliases (run-1 ↔ run-01) in order.

Outputs (under out_root/.../sub-XX/ses-YY/func/):
  - <base>_betas.h5
      datasets:
        betas_full       (P x K)          # all regressors
        betas            (P x K_task)     # last block (trial regressors)
        design_matrix    (T_kept x K)
        design_col_names (K,)
        yhat             (T_kept x P) [--save_yhat]
        resid            (T_kept x P) [--save_resid]
      attrs:
        task_regressor_names, regressor_level="trial"
        task_col_start, task_col_end (int)
        tmask_dropped, high_pass_sec
        order_in_X="[confounds, drifts+intercept, task(trials)]"
        estimator="WLS_AR1" or "OLS"
        rho_ar1 (AR(1) estimate if GLS; 0.0 if OLS)

Example:
python glm_analysis.py \
  --subj sub-01 --tasks 1back --include_types encoding delay \
  --out_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay \
  --overwrite --save_yhat --save_resid --estimator ols --ridge 1e-6


for iterate over all subjects/sessions/runs:
    for subj in 2 3 5 6; do
        for task in ctxdm 1back interdms; do
            python glm_analysis.py \
            --subj sub-$(printf "%02d" $subj) \
            --tasks $task \
            --include_types encoding delay \
            --out_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay \
            --overwrite --save_yhat --save_resid --estimator ols --ridge 1e-6
        done
    done
"""
from __future__ import annotations

import re
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import nibabel as nib
from nilearn.glm.first_level import spm_hrf, make_first_level_design_matrix

from utils import glm_confounds_construction  # NOTE: no need for standardize_run_label anymore


# -----------------------
# CLI / hyperparams
# -----------------------

def get_args():
    p = argparse.ArgumentParser(description="Per-run trial-level GLM with cosine drifts + HRF-convolution (OLS/GLS).")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames to drop at run start")
    p.add_argument("--correct_only", action="store_true", help="Use only correct trials")
    p.add_argument("--tasks", nargs="+", default=["ctxdm"], help="Task names to process")
    p.add_argument("--include_types", nargs="+", default=["encoding", "delay"],
                   help="Event types to include as trial regressors (lowercased)")
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled")
    p.add_argument("--conf_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--events_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data")
    p.add_argument("--high_pass_sec", type=float, default=128.0, help="Cosine high-pass cutoff in seconds")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_yhat", action="store_true", help="Save predicted time series (original space)")
    p.add_argument("--save_resid", action="store_true", help="Save residuals Yf - yhat (original space)")
    # Estimator toggle
    p.add_argument("--estimator", choices=["wls_ar1", "ols"], default="wls_ar1",
                   help="wls_ar1 (GLS via AR(1) whitening) or ols (ordinary least squares)")
    p.add_argument("--ridge", type=float, default=0.0,
                   help="Optional L2 (Tikhonov) lambda for OLS only; 0.0 = pure OLS.")
    return p.parse_args()


# -----------------------
# Simple path helpers
# -----------------------

def default_output_root(out_root_base: Path, correct_only: bool, subj: str) -> Path:
    return out_root_base / ("correct_trial_level_betas" if correct_only else "trial_level_betas") / subj


def discover_sessions(fmri_root_dir: Path):
    # sessions like ses-001, ses-016, etc.
    pattern = re.compile(r"^ses-\d+$")
    return sorted([p.name for p in fmri_root_dir.iterdir() if p.is_dir() and pattern.match(p.name)])


def discover_runs_from_timeseries(subj: str, ses: str, task: str, fmri_root_dir: Path):
    """Return run labels directly from dtseries filenames (ground truth)."""
    pat = f"{subj}_{ses}_task-{task}_*_space-Glasser64k_bold.dtseries.nii"
    files = sorted((fmri_root_dir / ses).glob(pat))
    run_pattern = re.compile(
        rf"{re.escape(subj)}_{re.escape(ses)}_task-{re.escape(task)}_(.+?)_space-Glasser64k_bold\.dtseries\.nii"
    )
    runs = []
    for f in files:
        m = run_pattern.match(f.name)
        if m:
            runs.append(m.group(1))  # e.g., 'run-1' or 'run-01'
    return sorted(set(runs))


def run_aliases(run_label: str):
    """Return small alias list like ['run-1', 'run-01'] (unique, in a sensible order)."""
    m = re.match(r"^(run-)(\d+)$", run_label)
    if m:
        prefix, num = m.group(1), m.group(2)
        # make a zero-padded 2-digit variant if needed
        pad = f"{int(num):02d}"
        variants = [f"{prefix}{num}"]
        if pad != num:
            variants.append(f"{prefix}{pad}")
        return variants
    return [run_label]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


# -----------------------
# Trial-wise regressors
# -----------------------

def build_trialwise_regressors(
    df_events: pd.DataFrame,
    num_trs: int,
    tr: float,
    include_types: list[str],
    correct_only: bool = False,
):
    req_cols = {"onset_time", "offset_time", "type"}
    if not req_cols.issubset(df_events.columns):
        missing = ", ".join(sorted(req_cols - set(df_events.columns)))
        raise ValueError(f"Events missing required columns: {missing}")

    print(df_events.head(10))
    types = df_events["type"].astype(str).str.strip().str.lower()
    mask_type = types.isin([t.lower() for t in include_types])

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
    box, names, meta_rows = [], [], []

    def g(row: pd.Series, key, default=np.nan):
        return row[key] if (key in row.index and pd.notna(row[key])) else default

    for i, row in df_sel.iterrows():
        try:
            on = float(row["onset_time"]) if pd.notna(row["onset_time"]) else np.nan
            off = float(row["offset_time"]) if pd.notna(row["offset_time"]) else np.nan
        except Exception:
            continue
        if not np.isfinite(on) or not np.isfinite(off):
            continue

        a = int(np.floor(on / tr))
        b = int(np.ceil(off / tr))
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


# -----------------------
# Drifts + intercept
# -----------------------

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
# Estimators
# -----------------------

def estimate_rho_from_resid(resid_run: np.ndarray) -> float:
    e = resid_run - resid_run.mean(axis=0, keepdims=True)
    num = float(np.sum(e[1:] * e[:-1]))
    den = float(np.sum(e[:-1] * e[:-1]) + 1e-12)
    rho = num / den
    return float(np.clip(rho, -0.99, 0.99))


def whiten_ar1(X: np.ndarray, Y: np.ndarray, rho: float):
    Xw = X.astype(np.float64, copy=True)
    Yw = Y.astype(np.float64, copy=True)
    Xw[1:] -= rho * Xw[:-1]
    Yw[1:] -= rho * Yw[:-1]
    return Xw, Yw


def solve_wls_stable(Xw: np.ndarray, Yw: np.ndarray):
    B, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    return B


def fit_glm_with_ar1(X: np.ndarray, Y: np.ndarray):
    """
    GLS via AR(1) whitening:
      1) OLS pass to estimate residuals
      2) Estimate rho from residuals
      3) Whiten X,Y and solve WLS (lstsq)
    Returns betas_full (P x K) and rho.
    """
    reg0 = LinearRegression(fit_intercept=False).fit(X, Y)
    resid0 = Y - reg0.predict(X)
    rho = estimate_rho_from_resid(resid0)
    Xw, Yw = whiten_ar1(X, Y, rho)
    B = solve_wls_stable(Xw, Yw)             # (K x P)
    betas_full = B.T.astype(np.float32)      # (P x K)
    return betas_full, float(rho)


def fit_glm_with_ols(X: np.ndarray, Y: np.ndarray, ridge: float = 0.0):
    """
    OLS (optionally with tiny ridge):
      β = argmin ||Y - Xβ||^2  (+ ridge * ||β||^2)
    Returns betas_full (P x K).
    """
    X64 = X.astype(np.float64, copy=False)
    Y64 = Y.astype(np.float64, copy=False)

    if ridge and ridge > 0.0:
        K = X64.shape[1]
        XtX = X64.T @ X64
        XtY = X64.T @ Y64
        B = np.linalg.solve(XtX + ridge * np.eye(K), XtY)   # (K x P)
    else:
        B, _, _, _ = np.linalg.lstsq(X64, Y64, rcond=None)  # (K x P)

    betas_full = B.T.astype(np.float32)                     # (P x K)
    return betas_full


# -----------------------
# Main
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
        for ses in sessions:
            # Discover runs directly from timeseries files (ground truth)
            runs = discover_runs_from_timeseries(subj, ses, task_name, fmri_root_dir)
            print(f"[{task_name} | {ses}] runs discovered from timeseries: {runs}")
            if not runs:
                continue

            for run_label in runs:
                # Build candidate labels (run-1 ↔ run-01)
                aliases = run_aliases(run_label)

                # Resolve all three paths with the same alias order
                ts_candidates = [
                    fmri_root_dir / ses / f"{subj}_{ses}_task-{task_name}_{rl}_space-Glasser64k_bold.dtseries.nii"
                    for rl in aliases
                ]
                ev_candidates = [
                    events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{rl}_events.tsv"
                    for rl in aliases
                ]
                cf_candidates = [
                    confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{rl}_desc-confounds_timeseries.tsv"
                    for rl in aliases
                ]

                ts_path = first_existing(ts_candidates)
                if ts_path is None:
                    print(f"⚠️ Timeseries missing for {ses} {run_label} {task_name} "
                          f"(tried {[p.name for p in ts_candidates]}), skipping.")
                    continue

                ev_path = first_existing(ev_candidates)
                if ev_path is None:
                    print(f"⚠️ Behavioral file missing for {ses} {run_label} {task_name} "
                          f"(tried {[p.name for p in ev_candidates]}), skipping.")
                    continue

                cf_path = first_existing(cf_candidates)
                if cf_path is None:
                    print(f"⚠️ Confounds missing for {ses} {run_label} {task_name} "
                          f"(tried {[p.name for p in cf_candidates]}), skipping.")
                    continue

                # Use the label that actually matched the timeseries file for naming
                matched_label = None
                for rl, p in zip(aliases, ts_candidates):
                    if p == ts_path:
                        matched_label = rl
                        break
                run_used = matched_label or run_label
                print(f"[✓] Using run label '{run_used}' for I/O (from timeseries).")

                # Output paths
                target_subdir = output_dir / ses / "func"
                base_name = f"{subj}_{ses}_task-{task_name}_{run_used}"
                h5_file = target_subdir / f"{base_name}_betas.h5"
                design_csv = target_subdir / f"{base_name}_design.csv"
                design_matrix_csv = target_subdir / f"{base_name}_design_matrix.csv"

                if h5_file.exists() and not args.overwrite:
                    print(f"[⏩] Skipping {h5_file} — already exists.")
                    continue

                # ---------------- Load data ----------------
                Y = nib.load(str(ts_path)).get_fdata(dtype=np.float32)  # (T x P)
                assert Y.ndim == 2, f"Timeseries must be 2D (T x P), got shape {Y.shape}"
                assert not np.isnan(Y).any(), "NaNs in timeseries_data"
                T, P = Y.shape

                df_confounds = pd.read_csv(cf_path, sep="\t")
                C_mat = glm_confounds_construction(df_confounds)
                if isinstance(C_mat, pd.DataFrame):
                    conf_names = list(C_mat.columns)
                    C = np.nan_to_num(C_mat.values, nan=0.0).astype(np.float32)
                else:
                    C = np.nan_to_num(C_mat, nan=0.0).astype(np.float32)
                    conf_names = [f"conf_{i}" for i in range(C.shape[1])]

                # Drop near-constant confounds
                if C.shape[0] != T:
                    raise AssertionError(f"Confounds rows ({C.shape[0]}) != timeseries T ({T})")
                keep_idx = [j for j in range(C.shape[1]) if np.std(C[:, j]) >= 1e-8]
                if len(keep_idx) != C.shape[1]:
                    C = C[:, keep_idx]
                    conf_names = [conf_names[j] for j in keep_idx]

                df_events = pd.read_csv(ev_path, sep="\t")
                df_events = clean_events(df_events)

                # ------------- Build task regressors (trial-wise) -------------
                R_task, trial_names, trial_info = build_trialwise_regressors(
                    df_events, T, tr, include_types, correct_only=correct_only
                )
                if R_task.shape[1] == 0:
                    print(f"⚠️ No trials of types {include_types} in {ev_path}, skipping.")
                    continue

                # ------------- Drifts + intercept -------------
                D, drift_names = per_run_drift_and_intercept(T, tr, high_pass_sec=args.high_pass_sec)

                # ------------- tmask -------------
                keep = np.ones((T,), dtype=bool)
                keep[:tmask] = False
                Yf = Y[keep, :].astype(np.float32)
                Ck = C[keep, :]
                Rk = R_task[keep, :]
                Dk = D[keep, :]

                # ------------- Assemble X -------------
                X = np.hstack([Ck, Dk, Rk]).astype(np.float32)
                design_col_names = conf_names + drift_names + list(trial_names)

                # ------------- Fit GLM -------------
                if args.estimator == "wls_ar1":
                    betas_full, rho = fit_glm_with_ar1(X, Yf)  # (P x K), rho float
                    estimator_name = "WLS_AR1"
                else:
                    betas_full = fit_glm_with_ols(X, Yf, ridge=args.ridge)  # (P x K)
                    rho = 0.0
                    estimator_name = "OLS"

                # Task slice and convenience view
                K_task = Rk.shape[1]
                task_col_start = X.shape[1] - K_task
                task_col_end = X.shape[1]
                betas_task = betas_full[:, task_col_start:task_col_end]  # (P x K_task)

                # Debug shapes
                print(f"[{estimator_name}] X: {X.shape}, betas_full: {betas_full.shape}, Yf: {Yf.shape}, betas_task: {betas_task.shape}")

                # ------------- Optional predictions/residuals -------------
                yhat = None
                resid = None
                if args.save_yhat or args.save_resid:
                    yhat = (X @ betas_full.T).astype(np.float32)
                if args.save_resid:
                    resid = (Yf - yhat).astype(np.float32)

                # ------------- Save outputs -------------
                target_subdir.mkdir(parents=True, exist_ok=True)
                trial_info.sort_values(by='trial_index').to_csv(design_csv, index=False)
                pd.DataFrame(X, columns=design_col_names).to_csv(design_matrix_csv, index=False)

                with h5py.File(h5_file, "w") as h5f:
                    h5f.create_dataset("betas_full", data=betas_full.astype(np.float32))   # (P x K)
                    h5f.create_dataset("betas", data=betas_task.astype(np.float32))        # (P x K_task)
                    h5f.create_dataset("design_matrix", data=X.astype(np.float32))         # (T_kept x K)
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    h5f.create_dataset("design_col_names",
                                       data=np.array(design_col_names, dtype=object),
                                       dtype=str_dtype)
                    # Metadata
                    h5f.attrs.create("task_regressor_names", np.array(trial_names, dtype=str_dtype), dtype=str_dtype)
                    h5f.attrs["regressor_level"] = "trial"
                    h5f.attrs["task_col_start"] = int(task_col_start)
                    h5f.attrs["task_col_end"] = int(task_col_end)
                    h5f.attrs["tmask_dropped"] = int(tmask)
                    h5f.attrs["high_pass_sec"] = float(args.high_pass_sec)
                    h5f.attrs["order_in_X"] = "[confounds, drifts+intercept, task(trials)]"
                    h5f.attrs["estimator"] = estimator_name
                    h5f.attrs["rho_ar1"] = float(rho)  # 0.0 for OLS
                    if yhat is not None and args.save_yhat:
                        h5f.create_dataset("yhat", data=yhat, compression="gzip")
                    if resid is not None and args.save_resid:
                        h5f.create_dataset("resid", data=resid, compression="gzip")

                print(f"[🎯] Saved betas + full design to {h5_file}")
                print(f"[🧾] Trial design CSV: {design_csv}")
                print(f"[📐] Full design matrix CSV: {design_matrix_csv}")

    print("[✅] Done.")


if __name__ == "__main__":
    main()
