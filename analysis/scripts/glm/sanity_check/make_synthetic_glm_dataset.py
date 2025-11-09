#!/usr/bin/env python3
"""
Generate a synthetic per-run dataset compatible with glm_analysis.py.

It creates:
- fmri_root: glasser_resampled/<subj>/<ses>/<subj>_<ses>_task-<task>_<run>_space-Glasser64k_bold.dtseries.nii
- conf_root: confounds/<subj>/<ses>/func/<subj>_<ses>_task-<task>_<run>_desc-confounds_timeseries.tsv
- events_root: events/<subj>/<ses>/func/<subj>_<ses>_task-<task>_<run>_events.tsv
- ground-truth betas: ground-truth betas/<subj>/<ses>/func/<base>_ground_truth_betas.h5 and _full_design.npz

Notes
-----
- The “dtseries.nii” is a simple 2D NIfTI with shape (T, P) to keep things easy; nibabel
  will still load it fine via `nib.load(...).get_fdata()`.
- Task regressors are built from events via the same binning logic your GLM uses
  (onset/offset in seconds -> TR bins, HRF-convolved later by your pipeline).
- We save *ground-truth* task betas so you can compare to the GLM output betas.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import h5py
from nilearn.glm.first_level import spm_hrf

# ---------- CONFIGURABLE ----------
ROOT = Path("/project/def-pbellec/xuan/fmri_dataset_project/synthetic_data")

subj = "sub-synthetic"
sessions = ["ses-001"]
task = "ctxdm"
runs = ["run-01"]

TR = 1.49                  # seconds
T = 400                    # number of timepoints per run (after acquisition, BEFORE tmask)
P = 128                    # “parcels/vertices” dimension
tmask = 1                  # frames to drop at run start by your GLM

# Events / trials
include_types = ["encoding", "delay"]
num_trials = 20            # trial-level regressors per type (so total trial events = num_trials * len(include_types))
event_dur_sec = 1.49 * 3   # 3 TRs duration for each event
iti_sec = 1.49 * 4         # 4 TRs spacing between successive events (onsets are evenly spaced)

# Confounds
num_confounds = 6          # simple numeric columns; your utils will pick them up
confound_scale = 1.0

# Ground-truth beta magnitudes
signal_scale_task = 2.5    # task betas magnitude
signal_scale_drift = 0.2   # small non-zero drift betas (just to be realistic)
signal_scale_conf = 0.0    # confounds betas; keep 0 unless you want to test confound leakage

# Noise model
rho_ar1 = 0.3              # temporal AR(1)
sigma_noise = 1.0          # innovation std
rng_seed = 2025
# ----------------------------------

rng = np.random.default_rng(rng_seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def nifti_2d_save(data_2d: np.ndarray, out_path: Path):
    """
    Save 2D array (T x P) as a NIfTI image (still OK to name .dtseries.nii).
    """
    img = nib.Nifti1Image(data_2d.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(out_path))

def make_confounds(T: int, num_cols: int = 0, scale: float = 1.0) -> pd.DataFrame:
    """
    Generate fMRIPrep-like confound columns so glm_confounds_construction()
    can find what it needs. Extra columns don't hurt.

    Included:
      - csf, white_matter, global_signal (with power2 and derivative1 variants)
      - trans_x/y/z, rot_x/y/z (with power2 and derivative1 variants)
      - framewise_displacement, dvars, std_dvars
    """
    rng = np.random.default_rng(2025)
    t = np.arange(T)

    def smooth_noise(freq=1):
        return (np.sin(2*np.pi*freq*t/max(T,1)) + 0.1*rng.standard_normal(T)) * scale

    # Base signals
    csf = 0.5 * smooth_noise(1) + 0.05 * rng.standard_normal(T)
    white_matter = 0.5 * smooth_noise(2) + 0.05 * rng.standard_normal(T)
    global_signal = 0.5 * smooth_noise(3) + 0.05 * rng.standard_normal(T)
    
    trans_x = smooth_noise(1)
    trans_y = smooth_noise(2) 
    trans_z = smooth_noise(3)
    
    rot_x = 0.01 * smooth_noise(1)
    rot_y = 0.01 * smooth_noise(2)
    rot_z = 0.01 * smooth_noise(3)

    # Create derivatives (backward difference)
    def derivative(signal):
        deriv = np.zeros_like(signal)
        deriv[1:] = np.diff(signal)
        deriv[0] = 0.0  # First timepoint has no derivative, set to 0 instead of NaN
        return deriv

    conf = {
        # CSF signals
        "csf": csf,
        "csf_power2": csf**2,
        "csf_derivative1": derivative(csf),
        "csf_derivative1_power2": derivative(csf)**2,
        
        # White matter signals
        "white_matter": white_matter,
        "white_matter_power2": white_matter**2,
        "white_matter_derivative1": derivative(white_matter),
        "white_matter_derivative1_power2": derivative(white_matter)**2,
        
        # Global signal
        "global_signal": global_signal,
        "global_signal_power2": global_signal**2,
        "global_signal_derivative1": derivative(global_signal),
        "global_signal_derivative1_power2": derivative(global_signal)**2,

        # Translation motion
        "trans_x": trans_x,
        "trans_x_power2": trans_x**2,
        "trans_x_derivative1": derivative(trans_x),
        "trans_x_derivative1_power2": derivative(trans_x)**2,
        
        "trans_y": trans_y,
        "trans_y_power2": trans_y**2,
        "trans_y_derivative1": derivative(trans_y),
        "trans_y_derivative1_power2": derivative(trans_y)**2,
        
        "trans_z": trans_z,
        "trans_z_power2": trans_z**2,
        "trans_z_derivative1": derivative(trans_z),
        "trans_z_derivative1_power2": derivative(trans_z)**2,

        # Rotation motion  
        "rot_x": rot_x,
        "rot_x_power2": rot_x**2,
        "rot_x_derivative1": derivative(rot_x),
        "rot_x_derivative1_power2": derivative(rot_x)**2,
        
        "rot_y": rot_y,
        "rot_y_power2": rot_y**2,
        "rot_y_derivative1": derivative(rot_y),
        "rot_y_derivative1_power2": derivative(rot_y)**2,
        
        "rot_z": rot_z,
        "rot_z_power2": rot_z**2,
        "rot_z_derivative1": derivative(rot_z),
        "rot_z_derivative1_power2": derivative(rot_z)**2,

        # scalar QC-style series
        "framewise_displacement": np.abs(0.05 * rng.standard_normal(T)).astype(float),
        "dvars": np.abs(0.05 * rng.standard_normal(T)).astype(float),
        "std_dvars": np.abs(0.05 * rng.standard_normal(T)).astype(float),
    }

    df = pd.DataFrame(conf)

    # Keep backward-compat with previous version (optionally add extra columns)
    if num_cols and num_cols > 0:
        # add a few extra nuisance cols with benign signals
        for i in range(num_cols):
            df[f"extra_conf_{i:02d}"] = smooth_noise(i + 1)

    return df


def build_events(TR: float, T: int, num_trials: int, include_types: list[str], event_dur_sec: float, iti_sec: float) -> pd.DataFrame:
    """
    Create a simple alternating block of encoding & delay events spaced by ITI.
    We keep all events within the run; extra are clipped off.
    """
    rows = []
    onset = 0.0
    trial_num = 1
    for i in range(num_trials):
        for typ in include_types:
            offset = onset + event_dur_sec
            if offset >= T * TR - 1e-6:
                break
            rows.append({
                "trialNumber": trial_num,
                "type": typ,
                "onset_time": round(onset, 4),
                "offset_time": round(offset, 4),
                "is_correct": True,  # keep it simple; can mix True/False if desired
                # optional metadata columns used in your script:
                "stim_order": trial_num,
                "locmod": rng.integers(0, 4),
                "ctgmod": rng.integers(0, 9),
                "objmod": rng.integers(0, 100),
            })
            onset = offset + iti_sec
        trial_num += 1
        if onset >= T * TR - 1e-6:
            break
    df = pd.DataFrame(rows)
    return df

def bin_trialwise_boxcars(df_events: pd.DataFrame, T: int, TR: float, include_types: list[str]) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Build TR-binned boxcars and convolve with HRF, exactly matching GLM analysis approach.
    Onset bin = ceil(onset/TR), offset bin = floor(offset/TR); then put 1.0 on [a, b)
    Then convolve with canonical SPM HRF to match glm_analysis.py processing.
    """
    types = df_events["type"].astype(str).str.strip().str.lower()
    mask = types.isin([t.lower() for t in include_types])
    df_sel = df_events.loc[mask].reset_index(drop=True)

    names = []
    meta = []
    R_box = []
    for i, row in df_sel.iterrows():
        on = float(row["onset_time"])
        off = float(row["offset_time"])
        a = int(np.ceil(on / TR))
        b = int(np.floor(off / TR))
        a = max(0, min(a, T))
        b = max(0, min(b, T))
        if b <= a:
            continue
        vec = np.zeros(T, dtype=np.float32)
        vec[a:b] = 1.0
        R_box.append(vec)

        tn = int(row["trialNumber"]) if "trialNumber" in row and pd.notna(row["trialNumber"]) else i + 1
        typ = str(row["type"]).strip().lower()
        names.append(f"trial{tn:03d}_{typ}")
        meta.append({
            "trial_index": len(names)-1,
            "trialNumber": tn,
            "type": typ,
            "onset_time": on,
            "offset_time": off,
            "is_correct": row.get("is_correct", True),
            "stim_order": row.get("stim_order", tn),
            "location": row.get("locmod", np.nan),
            "category": row.get("ctgmod", np.nan),
            "object": row.get("objmod", np.nan),
        })

    if len(R_box) == 0:
        return np.zeros((T, 0), dtype=np.float32), [], pd.DataFrame(columns=[
            "trial_index","trialNumber","type","onset_time","offset_time","is_correct",
            "stim_order","location","category","object"
        ])

    R_box = np.stack(R_box, axis=1)  # (T x K_task)
    
    # CRITICAL: Apply HRF convolution exactly like glm_analysis.py
    h = spm_hrf(TR, oversampling=1)
    R_hrf = np.zeros_like(R_box, dtype=np.float32)
    for k in range(R_box.shape[1]):
        tmp = np.convolve(R_box[:, k], h)
        R_hrf[:, k] = tmp[:T]
    
    trial_info = pd.DataFrame(meta).sort_values("trial_index").reset_index(drop=True)
    return R_hrf, names, trial_info  # Return HRF-convolved regressors

def make_drift_and_intercept(T: int, TR: float, high_pass_sec: float = 128.0) -> tuple[np.ndarray, list[str]]:
    """
    cosine drifts + intercept matching your nilearn call (without needing nilearn here).
    We'll synthesize ~floor(T / (high_pass_sec/TR)) cosine terms + a constant.
    """
    # A light-weight approximation so the GLM sees *some* drift columns before HRF regressors.
    frame_times = np.arange(T) * TR
    period = max(int(np.floor(high_pass_sec / TR)), 1)
    num_cos = max(int(np.floor(T / period)), 1)

    cols = []
    X = []
    for k in range(1, num_cos + 1):
        cols.append(f"cosine_{k:02d}")
        X.append(np.cos(2 * np.pi * k * frame_times / (period * TR)))
    cols.append("constant")
    X.append(np.ones_like(frame_times))
    X = np.stack(X, axis=1).astype(np.float32)  # (T x K_drift)
    return X, cols

def simulate_Y(X: np.ndarray, B: np.ndarray, rho: float, sigma: float, rng) -> np.ndarray:
    """
    Y_t = X_t @ B + E_t, with AR(1) innovations on E across time for each parcel independently.
    X: (T x K), B: (K x P) -> mean signal (T x P)
    """
    T, K = X.shape
    P = B.shape[1]
    mean_signal = X @ B  # (T x P)

    # AR(1) noise per parcel
    E = np.zeros((T, P), dtype=np.float32)
    eps = rng.normal(0.0, sigma, size=(T, P)).astype(np.float32)
    for p in range(P):
        for t in range(1, T):
            E[t, p] = rho * E[t-1, p] + eps[t, p]
        E[0, p] = eps[0, p] / max(1e-6, np.sqrt(1 - rho**2))
    return mean_signal + E

def main():
    # Directory layout
    fmri_root = ROOT / "glasser_resampled" / subj
    conf_root = ROOT / "confounds" / subj
    events_root = ROOT / "reformated_behavior" / subj
    gt_root = ROOT / "ground-truth betas" / subj

    for ses in sessions:
        # Ensure folders exist
        ensure_dir(fmri_root / ses)
        ensure_dir(conf_root / ses / "func")
        ensure_dir(events_root / ses / "func")
        ensure_dir(gt_root / ses / "func")

        for run in runs:
            base = f"{subj}_{ses}_task-{task}_{run}"

            # ---------- 1) Events ----------
            df_events = build_events(TR, T, num_trials, include_types, event_dur_sec, iti_sec)
            events_tsv = events_root / ses / "func" / f"{base}_events.tsv"
            df_events.to_csv(events_tsv, sep="\t", index=False)

            # ---------- 2) Confounds ----------
            df_conf = make_confounds(T, num_confounds, confound_scale)
            conf_tsv = conf_root / ses / "func" / f"{base}_desc-confounds_timeseries.tsv"
            df_conf.to_csv(conf_tsv, sep="\t", index=False)

            # ---------- 3) Build design blocks to define truth ----------
            # Task boxcars (no HRF here; your GLM will convolve later)
            R_task, task_names, trial_info = bin_trialwise_boxcars(df_events, T, TR, include_types)

            # Drifts + intercept (rough approximation; your GLM recomputes its own version too)
            D, drift_names = make_drift_and_intercept(T, TR, high_pass_sec=128.0)

            # Confounds (already made): use numeric matrix
            C = df_conf.to_numpy(dtype=np.float32)
            conf_names = list(df_conf.columns)

            # Full X (the same column group order your GLM expects)
            X = np.hstack([C, D, R_task]).astype(np.float32)
            K_conf, K_drift, K_task = C.shape[1], D.shape[1], R_task.shape[1]
            assert X.shape[1] == (K_conf + K_drift + K_task)

            # ---------- 4) Ground-truth betas ----------
            # Confounds betas (P x K_conf) -> default 0
            B_conf = np.zeros((K_conf, P), dtype=np.float32) + signal_scale_conf * rng.normal(size=(K_conf, P)).astype(np.float32)

            # Drift betas (P x K_drift) -> tiny non-zero
            B_drift = (signal_scale_drift * rng.normal(size=(K_drift, P))).astype(np.float32)

            # Task betas (P x K_task) -> non-zero signals
            # Give each parcel a different strength and sign across trials
            B_task = (signal_scale_task * rng.normal(size=(K_task, P))).astype(np.float32)

            # Stack to (K x P)
            B_full = np.vstack([B_conf, B_drift, B_task]).astype(np.float32)  # (K_conf+K_drift+K_task) x P

            # ---------- 5) Simulate Y with AR(1) noise ----------
            Y = simulate_Y(X, B_full, rho_ar1, sigma_noise, rng)  # (T x P)
            
            # ---------- CRITICAL: Apply tmask to match GLM analysis ----------
            # The GLM analysis drops the first `tmask` frames, so we must do the same
            # for ground-truth comparison
            X_masked = X[tmask:, :].copy()  # (T_kept x K)
            Y_masked = Y[tmask:, :].copy()  # (T_kept x P)
            print(f"Applied tmask={tmask}: X {X.shape} -> {X_masked.shape}, Y {Y.shape} -> {Y_masked.shape}")

            # ---------- 6) Save timeseries as 2D NIfTI named .dtseries.nii ----------
            ts_nii = fmri_root / ses / f"{base}_space-Glasser64k_bold.dtseries.nii"
            nifti_2d_save(Y, ts_nii)  # Save full timeseries for GLM to process

            # ---------- 7) Save ground-truth betas ----------
            gt_h5 = gt_root / ses / "func" / f"{base}_ground_truth_betas.h5"
            gt_npz = gt_root / ses / "func" / f"{base}_full_design_and_betas.npz"

            with h5py.File(gt_h5, "w") as h5f:
                # Save only task betas to match your GLM's `betas` dataset (task columns only)
                h5f.create_dataset("betas", data=B_task.T)  # (P x K_task)
                # Names
                str_dtype = h5py.string_dtype(encoding="utf-8")
                h5f.create_dataset("task_regressor_names", data=np.array(task_names, dtype=object), dtype=str_dtype)
                # Column boundaries in the FULL design (handy for debugging)
                h5f.attrs["task_col_start"] = K_conf + K_drift
                h5f.attrs["task_col_end"] = K_conf + K_drift + K_task
                h5f.attrs["order_in_X"] = "[confounds, drifts+intercept, task(trials)]"
                h5f.attrs["TR"] = TR
                h5f.attrs["rho_ar1"] = rho_ar1
                h5f.attrs["sigma_noise"] = sigma_noise

            # Save the full X and full betas too (useful for debugging)
            # IMPORTANT: Save the masked versions for proper comparison with GLM
            np.savez_compressed(
                gt_npz,
                X=X_masked,          # Use masked design matrix (T_kept x K)
                X_full=X,            # Also save full version for reference
                design_col_names=np.array(conf_names + drift_names + task_names, dtype=object),
                B_full=B_full,       # (K x P)
                B_task=B_task,       # (K_task x P)
                conf_col_count=K_conf,
                drift_col_count=K_drift,
                task_col_count=K_task,
                TR=TR,
                rho_ar1=rho_ar1,
                sigma_noise=sigma_noise,
                tmask=tmask,
            )

            print(f"[✓] Wrote synthetic run: {base}")
            print(f"    - events:     {events_tsv}")
            print(f"    - confounds:  {conf_tsv}")
            print(f"    - timeseries: {ts_nii}")
            print(f"    - GT betas:   {gt_h5}")
            print(f"    - GT full:    {gt_npz}")

if __name__ == "__main__":
    main()
