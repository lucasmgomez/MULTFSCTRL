#!/usr/bin/env python3
"""
Per-run GLM (condition-level) with cosine drifts + AR(1), with separate
'encoding' and 'delay' regressors (HRF-convolved). Also saves the FULL design
matrix (both to HDF5 and CSV).

Outputs per run (under out_root/.../sub-XX/ses-YY/func/):
  - <base>_betas.h5
      datasets:
        betas                 (P x 2)          # columns: ["encoding", "delay"]
        XtX_inv               (K x K)
        sigma2                (P,)
        residuals_shape       (2,)              # (T_kept, P)
        all_regressors_shape  (2,)              # (T_kept, K)
        design_matrix         (T_kept x K)      # FULL X
        design_col_names      (K,)  (HDF5 string dtype)
      attrs:
        task_regressor_names ["encoding", "delay"]
        regressor_level      "condition"
        task_col_start, task_col_end (int)  # inclusive-exclusive
        dof, rho_ar1, tmask_dropped, high_pass_sec
        order_in_X           "[confounds, drifts+intercept, task(encoding,delay)]"
  - <base>_design.csv            # minimal condition list: ["encoding","delay"]
  - <base>_design_matrix.csv     # human-friendly CSV of FULL design matrix
"""

from __future__ import annotations
import os
import re
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import nibabel as nib

from nilearn.glm.first_level import spm_hrf, make_first_level_design_matrix

from utils import glm_confounds_construction, standardize_run_label


# -----------------------
# CLI / hyperparams
# -----------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Per-run GLM with cosine drifts + AR(1), separate encoding and delay regressors."
    )
    p.add_argument("--subj", default="sub-03")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames to drop at run start")
    p.add_argument("--correct_only", action="store_true", help="Use only correct trials")
    p.add_argument("--tasks", nargs="+", default=["ctxdm", "interdms", "1back"])
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled")
    p.add_argument("--conf_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--events_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data")
    p.add_argument("--high_pass_sec", type=float, default=128.0, help="Cosine high-pass cutoff in seconds")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# -----------------------
# Utilities
# -----------------------

def default_output_root(out_root_base: Path, correct_only: bool, subj: str) -> Path:
    return (out_root_base / ("correct_condition_level_betas" if correct_only else "condition_level_betas") / subj)


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
# Regressor builders (encoding & delay)
# -----------------------

def _build_binary_regressor(df_events: pd.DataFrame, num_trs: int, tr: float, event_type: str):
    """
    Build a TR-binned 0/1 boxcar regressor for a given event 'type'.

    Rules (per your spec):
      onset_tr  = ceil(onset_time / TR)
      offset_tr = floor(offset_time / TR)
      include interval [onset_tr, offset_tr) only if onset_tr < offset_tr.
    """
    if "type" not in df_events.columns:
        raise ValueError("Events missing required 'type' column.")
    if not {"onset_time", "offset_time"}.issubset(df_events.columns):
        raise ValueError("Events must include 'onset_time' and 'offset_time' columns (in seconds).")

    typ = df_events["type"].astype(str).str.strip().str.lower()
    df_sel = df_events[typ == event_type].copy()

    r = np.zeros((num_trs, 1), dtype=np.float32)
    for _, row in df_sel.iterrows():
        on = float(row["onset_time"])
        off = float(row["offset_time"])
        if not np.isfinite(on) or not np.isfinite(off):
            continue

        onset_tr = int(np.floor(on / tr))
        offset_tr = int(np.floor(off / tr))

        # clip to bounds
        a = max(0, min(onset_tr, num_trs))
        b = max(0, min(offset_tr, num_trs))
        if a < b:
            r[a:b, 0] += 1.0

    count_ones = int(r[:, 0].sum())
    ratio = count_ones / num_trs if num_trs > 0 else 0.0
    print(f"[{event_type.capitalize()} regressor] 1s: {count_ones}/{num_trs}  (ratio={ratio:.4f})")
    return r


def _convolve_hrf(boxcar: np.ndarray, tr: float) -> np.ndarray:
    """Convolve TR-binned boxcar with SPM canonical HRF."""
    T = boxcar.shape[0]
    h = spm_hrf(tr, oversampling=1)
    out = np.zeros_like(boxcar, dtype=np.float32)
    tmp = np.convolve(boxcar[:, 0], h)
    out[:T, 0] = tmp[:T]
    return out


def build_encoding_regressor(df_events: pd.DataFrame, num_trs: int, tr: float) -> tuple[np.ndarray, list[str]]:
    box = _build_binary_regressor(df_events, num_trs, tr, event_type="encoding")
    return _convolve_hrf(box, tr), ["encoding"]


def build_delay_regressor(df_events: pd.DataFrame, num_trs: int, tr: float) -> tuple[np.ndarray, list[str]]:
    box = _build_binary_regressor(df_events, num_trs, tr, event_type="delay")
    return _convolve_hrf(box, tr), ["delay"]


# -----------------------
# Drifts + intercept (nilearn-style cosine)
# -----------------------

def per_run_drift_and_intercept(num_trs: int, tr: float, high_pass_sec: float = 128.0) -> tuple[np.ndarray, list[str]]:
    """Intercept + cosine drifts like nilearn (T x D), and their column names."""
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
# AR(1) estimation + whitening
# -----------------------

def estimate_rho_from_resid(resid_run: np.ndarray) -> float:
    """
    Estimate AR(1) rho from run residuals (T x P) using shared rho.
    ε_t = ρ ε_{t-1} + u_t; clip rho to [-0.99, 0.99].
    """
    e = resid_run - resid_run.mean(axis=0, keepdims=True)
    num = float(np.sum(e[1:] * e[:-1]))
    den = float(np.sum(e[:-1] * e[:-1]) + 1e-12)
    rho = num / den
    return float(np.clip(rho, -0.99, 0.99))


def whiten_ar1(X: np.ndarray, Y: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply AR(1) whitening to design X and data Y (row-wise)."""
    Xw = X.copy()
    Yw = Y.copy()
    Xw[1:] -= rho * Xw[:-1]
    Yw[1:] -= rho * Yw[:-1]
    return Xw, Yw


def fit_glm_with_ar1(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Two-pass fit:
      1) quick OLS → residuals → estimate rho
      2) whiten X,Y with rho → final OLS
    Returns:
      betas_full (P x K), XtX_inv (K x K), sigma2 (P,), dof (int), residuals (T x P)
    """
    # Pass 1: quick OLS
    reg0 = LinearRegression().fit(X, Y)
    resid0 = Y - reg0.predict(X)
    rho = estimate_rho_from_resid(resid0)

    # Pass 2: whiten and refit
    Xw, Yw = whiten_ar1(X, Y, rho)
    XtX = Xw.T @ Xw
    ridge = 1e-6 * np.eye(XtX.shape[0], dtype=np.float32)
    XtX_inv = np.linalg.inv(XtX + ridge)
    B = XtX_inv @ (Xw.T @ Yw)     # (K x P)
    Yhat = Xw @ B                 # (T x P)
    resid = Yw - Yhat             # (T x P)

    dof = Xw.shape[0] - np.linalg.matrix_rank(Xw)
    dof = int(max(dof, 1))
    sigma2 = (resid**2).sum(axis=0) / dof   # per parcel

    betas_full = B.T                        # (P x K)
    return betas_full, XtX_inv, sigma2, dof, resid


# -----------------------
# Main
# -----------------------

def main():
    args = get_args()

    subj = args.subj
    tr_length = args.tr
    tmask = args.tmask
    correct_only = args.correct_only

    fmri_root_dir = Path(args.fmri_root) / subj
    confounds_root_dir = Path(args.conf_root) / subj
    events_root_dir = Path(args.events_root) / subj
    output_dir = default_output_root(Path(args.out_root), correct_only, subj)

    sessions = discover_sessions(fmri_root_dir)
    print("Sessions:", sessions)

    for task_name in args.tasks:
        for ses in sessions:
            # find runs
            runs = discover_runs_for_task_session(subj, ses, task_name, fmri_root_dir)
            print(f"[{task_name} | {ses}] runs: {runs}")

            for run in runs:
                # Paths
                behavioral_file = events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{standardize_run_label(run)}_events.tsv"
                behavioral_path = behavioral_file
                if not behavioral_path.exists():
                    print(f"⚠️ Behavioral file {behavioral_path} does not exist, skipping.")
                    continue

                timeseries_file = fmri_root_dir / ses / f"{subj}_{ses}_task-{task_name}_{run}_space-Glasser64k_bold.dtseries.nii"
                confounds_file = confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{run}_desc-confounds_timeseries.tsv"

                if not timeseries_file.exists() or not confounds_file.exists():
                    print(f"⚠️ Missing data for {ses} {run} {task_name}, skipping.")
                    continue

                # Output
                rel_behavioral_path = behavioral_path.relative_to(events_root_dir)
                target_subdir = output_dir / rel_behavioral_path.parent
                base_name = behavioral_path.stem.replace("_events", "")  # e.g., sub-01_ses-003_task-1back_run-01
                h5_file = target_subdir / f"{base_name}_betas.h5"
                design_csv = target_subdir / f"{base_name}_design.csv"              # minimal list
                design_matrix_csv = target_subdir / f"{base_name}_design_matrix.csv" # FULL X

                if h5_file.exists() and not args.overwrite:
                    print(f"[⏩] Skipping {h5_file} — already exists.")
                    continue

                # ---------------- Load data ----------------
                Y = nib.load(str(timeseries_file)).get_fdata(dtype=np.float32)  # (T x P)
                assert not np.isnan(Y).any(), "NaNs in timeseries_data"

                df_confounds = pd.read_csv(confounds_file, sep="\t")
                C_mat = glm_confounds_construction(df_confounds)
                if isinstance(C_mat, pd.DataFrame):
                    conf_names = list(C_mat.columns)
                    C = np.nan_to_num(C_mat.values, nan=0.0).astype(np.float32)
                else:
                    C = np.nan_to_num(C_mat, nan=0.0).astype(np.float32)
                    conf_names = [f"conf_{i}" for i in range(C.shape[1])]

                df_events = pd.read_csv(behavioral_path, sep="\t")
                df_events = clean_events(df_events)
                if correct_only and "is_correct" in df_events.columns:
                    df_events = df_events[df_events["is_correct"] == True]

                T = Y.shape[0]
                print(f"shape of timeseries data: {Y.shape} (T x P)")

                # ------------- Build task regressors -------------
                R_enc, enc_names = build_encoding_regressor(df_events, T, tr_length)
                R_dly, dly_names = build_delay_regressor(df_events, T, tr_length)
                R_task = np.hstack([R_enc, R_dly])   # (T x 2)
                task_names = enc_names + dly_names

                # ------------- Drifts + intercept -------------
                D, drift_names = per_run_drift_and_intercept(T, tr_length, high_pass_sec=args.high_pass_sec)

                # tmask skip (apply to all time-dependent arrays)
                keep = np.ones((T,), dtype=bool)
                keep[:tmask] = False
                Y = Y[keep, :]
                C = C[keep, :]
                R_task = R_task[keep, :]
                D = D[keep, :]

                # ------------- Assemble X -------------
                # Order: [confounds, drifts+intercept, task(encoding, delay)]
                X = np.hstack([C, D, R_task]).astype(np.float32)
                Yf = Y.astype(np.float32)

                # Names matching the columns of X
                design_col_names = conf_names + drift_names + list(task_names)

                # ------------- Fit GLM with AR(1) -------------
                betas_full, XtX_inv, sigma2, dof, resid = fit_glm_with_ar1(X, Yf)

                # Indices for task columns (last K_task)
                K_task = R_task.shape[1]            # should be 2
                betas_task = betas_full[:, -K_task:]  # (P x 2)
                task_col_start = X.shape[1] - K_task
                task_col_end = X.shape[1]

                # ------------- Save outputs -------------
                target_subdir.mkdir(parents=True, exist_ok=True)

                # Minimal condition design CSV (so contrasts know column order)
                pd.DataFrame({"condition": task_names}).to_csv(design_csv, index=False)  # ["encoding","delay"]

                # Human-friendly full design matrix CSV
                pd.DataFrame(X, columns=design_col_names).to_csv(design_matrix_csv, index=False)

                with h5py.File(h5_file, "w") as h5f:
                    # Core outputs
                    h5f.create_dataset("betas", data=betas_task.astype(np.float32))        # (P x 2): encoding, delay
                    h5f.create_dataset("all_regressors_shape", data=np.array(X.shape, dtype=np.int32))
                    # Stats for t/z
                    h5f.create_dataset("XtX_inv", data=XtX_inv.astype(np.float32))          # (K x K), K = X.shape[1]
                    h5f.create_dataset("sigma2", data=sigma2.astype(np.float32))            # (P,)
                    h5f.create_dataset("residuals_shape", data=np.array(resid.shape, dtype=np.int32))

                    # --- Full design matrix + names ---
                    h5f.create_dataset("design_matrix", data=X.astype(np.float32))          # (T_kept x K)
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    h5f.create_dataset("design_col_names", data=np.array(design_col_names, dtype=object), dtype=str_dtype)

                    # Metadata
                    h5f.attrs.create("task_regressor_names", np.array(task_names, dtype=str_dtype), dtype=str_dtype)  # ["encoding","delay"]
                    h5f.attrs["regressor_level"] = "condition"
                    h5f.attrs["task_col_start"] = task_col_start
                    h5f.attrs["task_col_end"] = task_col_end
                    h5f.attrs["dof"] = dof
                    # rho of whitened resid should be ~0; store estimate from whitened residuals for reference
                    h5f.attrs["rho_ar1"] = float(estimate_rho_from_resid(resid))
                    h5f.attrs["tmask_dropped"] = tmask
                    h5f.attrs["high_pass_sec"] = args.high_pass_sec
                    h5f.attrs["order_in_X"] = "[confounds, drifts+intercept, task(encoding,delay)]"

                print(f"[🎯] Saved betas + full design to {h5_file}")
                print(f"[🧾] Design (conditions): {design_csv}")
                print(f"[📐] Full design matrix CSV: {design_matrix_csv}")

    print("[✅] Done.")


if __name__ == "__main__":
    main()
