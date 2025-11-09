#!/usr/bin/env python3
"""
Voxel-level First-Level GLM (Nilearn) with Per-Trial Betas and Full Design Export

Saves outputs under:
  <out_root>/<include_key>/trial_level_betas/<subj>/<ses>/func/
    - <base>_nilearn_betas.h5   # (n_voxels x n_trials)
    - <base>_design.csv         # per-trial metadata aligned to betas order

Where:
  - out_root is expected to be: /project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data
  - include_key ∈ {"encoding", "delay", "encoding_delay"} derived from --include_types

Design mirrors your custom-GLM structure:
  e.g., /data/nilearn_data/delay/trial_level_betas/sub-01/ses-008/func/sub-01_ses-008_task-1back_run-01_design.csv

Model:
  - AR(1) noise
  - SPM HRF
  - Cosine drifts with high-pass cutoff (default 128 s)
  - Optional spatial smoothing (FWHM mm)

Usage example:
python glm_analysis_nilearn.py \
  --subj sub-01 \
  --tasks 1back \
  --include_types encoding \
  --fmri_root /project/def-pbellec/xuan/cneuromod.multfs.fmriprep \
  --conf_root /project/def-pbellec/xuan/cneuromod.multfs.fmriprep \
  --events_root /project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior \
  --out_root /project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data \
  --overwrite
"""

from __future__ import annotations
import re
import h5py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import warnings

from nilearn.glm.first_level import FirstLevelModel
from utils import glm_confounds_construction, standardize_run_label

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# CLI
# -----------------------
def get_args():
    p = argparse.ArgumentParser(description="Voxel-level First-Level GLM (Nilearn) with per-trial betas")
    p.add_argument("--subj", default="sub-01")
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--tmask", type=int, default=1, help="Frames to drop at run start")
    p.add_argument("--correct_only", action="store_true", help="Use only correct trials (if column 'is_correct' exists)")
    p.add_argument("--tasks", nargs="+", default=["ctxdm"], help="Task names to process, e.g., 1back ctxdm dms interdms")
    p.add_argument("--include_types", nargs="+", default=["encoding"], help="Event types to include (e.g., encoding delay)")
    p.add_argument("--fmri_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--conf_root", default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep")
    p.add_argument("--events_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior")
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data",
                   help="Base output root; script will create <out_root>/<include_key>/trial_level_betas/...")
    p.add_argument("--high_pass_sec", type=float, default=128.0, help="Cosine high-pass cutoff in seconds")
    p.add_argument("--smoothing_fwhm", type=float, default=6.0, help="Spatial smoothing FWHM in mm")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# -----------------------
# IO helpers
# -----------------------
def include_key_from_types(include_types: list[str]) -> str:
    """
    Map list like ["encoding"] -> "encoding", ["delay"] -> "delay",
    ["encoding","delay"] or ["delay","encoding"] -> "encoding_delay"
    """
    canon = sorted([t.strip().lower() for t in include_types if t.strip()])
    if canon == ["delay"]:
        return "delay"
    if canon == ["encoding"]:
        return "encoding"
    # default combined
    return "encoding_delay"


def default_output_root(out_root_base: Path, include_key: str, correct_only: bool, subj: str) -> Path:
    return (out_root_base
            / include_key
            / ("correct_trial_level_betas" if correct_only else "trial_level_betas")
            / subj)


def discover_sessions(subj_root: Path):
    pattern = re.compile(r"ses-\d+$")
    return sorted([p.name for p in subj_root.iterdir() if p.is_dir() and pattern.match(p.name)])


def discover_runs_for_task_session(subj, ses, task, fmri_root_dir: Path):
    func_dir = fmri_root_dir / ses / "func"
    if not func_dir.exists():
        return []
    pattern = f"{subj}_{ses}_task-{task}_run-*_space-*_desc-preproc_bold.nii.gz"
    matching = list(func_dir.glob(pattern))
    run_re = re.compile(rf"{re.escape(subj)}_{re.escape(ses)}_task-{re.escape(task)}_run-(\d+)_")
    runs = set()
    for f in matching:
        m = run_re.search(f.name)
        if m:
            runs.add(f"run-{int(m.group(1)):02d}")
    return sorted(runs)


def clean_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    return df


def create_nilearn_events(df_events: pd.DataFrame, include_types: list[str], correct_only: bool = False) -> pd.DataFrame:
    """
    Convert behavioral events -> Nilearn format: columns 'onset', 'duration', 'trial_type'
    trial_type := trial{num:03d}_{type}[_stim{order:02d}]
    """
    req_cols = {"onset_time", "offset_time", "type"}
    if not req_cols.issubset(df_events.columns):
        missing = ", ".join(sorted(req_cols - set(df_events.columns)))
        raise ValueError(f"Events missing required columns: {missing}")

    types = df_events["type"].astype(str).str.strip().str.lower()
    mask_type = types.isin([t.lower() for t in include_types])

    if correct_only and "is_correct" in df_events.columns:
        def is_negative(v):
            if isinstance(v, (bool, np.bool_)):
                return (v is False) or (v == False)
            if pd.isna(v):
                return False
            return str(v).strip().lower() in {"false", "b"}
        mask = mask_type & ~df_events["is_correct"].apply(is_negative)
    else:
        mask = mask_type

    df_sel = df_events.loc[mask].reset_index(drop=True)

    rows = []
    for i, row in df_sel.iterrows():
        on = float(row["onset_time"]) if pd.notna(row["onset_time"]) else np.nan
        off = float(row["offset_time"]) if pd.notna(row["offset_time"]) else np.nan
        if not (np.isfinite(on) and np.isfinite(off)):
            continue
        dur = off - on
        if dur <= 0:
            continue

        trial_num = row.get("trialNumber", i)
        trial_type = str(row["type"]).strip().lower()

        stim_suffix = ""
        if "stim_order" in row.index and pd.notna(row["stim_order"]):
            stim_suffix = f"_stim{int(row['stim_order']):02d}"

        try:
            name = f"trial{int(trial_num):03d}_{trial_type}{stim_suffix}"
        except Exception:
            name = f"trial{i:03d}_{trial_type}{stim_suffix}"

        d = {"onset": on, "duration": dur, "trial_type": name,
             "trial_index": i, "trialNumber": trial_num, "type": trial_type}
        for col in ["is_correct", "stim_order", "locmod", "ctgmod", "objmod"]:
            if col in row.index and pd.notna(row[col]):
                d[col] = row[col]
        rows.append(d)

    return pd.DataFrame(rows)


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
    include_key = include_key_from_types(include_types)

    fmri_root_dir = Path(args.fmri_root) / subj
    confounds_root_dir = Path(args.conf_root) / subj
    events_root_dir = Path(args.events_root) / subj
    output_dir = default_output_root(Path(args.out_root), include_key, correct_only, subj)

    sessions = discover_sessions(fmri_root_dir)
    print("Found sessions:", sessions)
    if not sessions:
        print(f"No sessions found in {fmri_root_dir}")
        return

    for task_name in args.tasks:
        for ses in sessions:
            runs = discover_runs_for_task_session(subj, ses, task_name, fmri_root_dir)
            print(f"[{task_name} | {ses}] runs: {runs}")
            for run in runs:
                print(f"\nProcessing {subj} {ses} {task_name} {run}")

                # Paths
                behavioral_file = events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{standardize_run_label(run)}_events.tsv"
                if not behavioral_file.exists():
                    print(f"⚠️ Missing behavioral file: {behavioral_file}")
                    continue

                run_num = run.split("-")[1]              # '01'
                run_num_short = str(int(run_num))        # '1'
                func_dir = fmri_root_dir / ses / "func"

                candidates = [
                    func_dir / f"{subj}_{ses}_task-{task_name}_run-{run_num}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                    func_dir / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                    func_dir / f"{subj}_{ses}_task-{task_name}_run-{run_num}_space-T1w_desc-preproc_bold.nii.gz",
                    func_dir / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-T1w_desc-preproc_bold.nii.gz",
                ]
                fmri_file = None
                for c in candidates:
                    if c.exists():
                        fmri_file = c
                        break
                if fmri_file is None:
                    # try wildcard
                    hits = list(func_dir.glob(f"{subj}_{ses}_task-{task_name}_run-{run_num}_space-*_desc-preproc_bold.nii.gz"))
                    if not hits:
                        hits = list(func_dir.glob(f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-*_desc-preproc_bold.nii.gz"))
                    if hits:
                        fmri_file = hits[0]
                if fmri_file is None:
                    print(f"⚠️ fMRI preproc file not found for {ses} {run} {task_name}")
                    continue

                conf_candidates = [
                    confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num}_desc-confounds_timeseries.tsv",
                    confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_desc-confounds_timeseries.tsv",
                ]
                confounds_file = None
                for c in conf_candidates:
                    if c.exists():
                        confounds_file = c
                        break
                if confounds_file is None:
                    print(f"⚠️ Confounds file not found for {ses} {run} {task_name}")
                    continue

                # Outputs (mirror events relative sub-structure)
                rel_behavior = behavioral_file.relative_to(events_root_dir)
                target_subdir = output_dir / rel_behavior.parent  # <out>/<include_key>/trial_level_betas/subj/ses/func
                base = behavioral_file.stem.replace("_events", "")
                h5_file = target_subdir / f"{base}_nilearn_betas.h5"
                design_csv = target_subdir / f"{base}_design.csv"
                target_subdir.mkdir(parents=True, exist_ok=True)

                if h5_file.exists() and not args.overwrite:
                    print(f"[⏩] Skip existing: {h5_file}")
                    continue

                try:
                    # ---- Load NIfTI & time-mask ----
                    fmri_img = nib.load(str(fmri_file))
                    n_time = fmri_img.shape[-1]
                    print(f"fMRI shape: {fmri_img.shape} (last dim time={n_time})")

                    if tmask > 0:
                        data = fmri_img.get_fdata(dtype=np.float32)
                        data = data[..., tmask:]  # drop first tmask TRs
                        fmri_img = nib.Nifti1Image(data, fmri_img.affine, fmri_img.header)
                        print(f"After tmask, timepoints: {fmri_img.shape[-1]}")

                    # ---- Events ----
                    df_events = pd.read_csv(behavioral_file, sep="\t")
                    df_events = clean_events(df_events)
                    events_df = create_nilearn_events(df_events, include_types, correct_only)
                    if events_df.empty:
                        print(f"⚠️ No events of types {include_types}, skipping.")
                        continue
                    # shift onsets after tmask
                    if tmask > 0:
                        events_df["onset"] = events_df["onset"] - (tmask * tr)
                        events_df = events_df[events_df["onset"] >= 0].reset_index(drop=True)
                    if events_df.empty:
                        print("⚠️ All events fell before time-zero after tmask shift.")
                        continue

                    # ---- Confounds ----
                    df_conf = pd.read_csv(confounds_file, sep="\t")
                    if tmask > 0:
                        df_conf = df_conf.iloc[tmask:].reset_index(drop=True)

                    C = glm_confounds_construction(df_conf)
                    if isinstance(C, pd.DataFrame):
                        confounds_matrix = np.nan_to_num(C.values, nan=0.0).astype(np.float32)
                    else:
                        confounds_matrix = np.nan_to_num(C, nan=0.0).astype(np.float32)

                    if confounds_matrix.shape[0] != fmri_img.shape[-1]:
                        print(f"❌ Confounds rows ({confounds_matrix.shape[0]}) != fMRI timepoints ({fmri_img.shape[-1]})")
                        continue

                    # ---- GLM ----
                    glm = FirstLevelModel(
                        t_r=tr,
                        noise_model="ar1",
                        standardize=False,
                        hrf_model="spm",
                        drift_model="cosine",
                        high_pass=1.0 / args.high_pass_sec,
                        smoothing_fwhm=args.smoothing_fwhm,
                        minimize_memory=False,
                    )
                    glm = glm.fit(fmri_img, events_df, confounds=confounds_matrix)
                    dm = glm.design_matrices_[0]
                    all_cols = list(dm.columns)
                    print(f"Design matrix: T={dm.shape[0]}, K={dm.shape[1]}")

                    # Identify trial columns
                    trial_cols = [c for c in all_cols if c.lower().startswith("trial")]
                    if not trial_cols:
                        print("⚠️ No trial regressors found, skipping.")
                        continue

                    # Extract voxel betas per trial regressor
                    beta_cols = []
                    beta_maps = []
                    for c in trial_cols:
                        try:
                            eff = glm.compute_contrast(c, output_type="effect_size")
                            beta_maps.append(eff.get_fdata(dtype=np.float32).ravel())
                            beta_cols.append(c)
                        except Exception as e:
                            print(f"⚠️ contrast failed for {c}: {e}")

                    if not beta_maps:
                        print("⚠️ No valid beta maps generated.")
                        continue

                    betas = np.column_stack(beta_maps).astype(np.float32)  # (n_voxels x n_trials)
                    print("Betas shape:", betas.shape)

                    # Save per-trial metadata (align to beta_cols order)
                    trial_meta = (
                        events_df
                        .assign(_trial_order=lambda d: d["trial_type"].map({n:i for i,n in enumerate(beta_cols)}))
                        .dropna(subset=["_trial_order"])
                        .sort_values("_trial_order")
                        .drop(columns=["_trial_order"])
                        .reset_index(drop=True)
                    )
                    trial_meta.to_csv(design_csv, index=False)

                    # Drift names (constant + cosines)
                    drift_cols = [c for c in all_cols if c == "constant" or c.startswith("cosine")]

                    # ---- Save HDF5 ----
                    target_subdir.mkdir(parents=True, exist_ok=True)
                    with h5py.File(h5_file, "w") as h5f:
                        # Core
                        h5f.create_dataset("betas", data=betas)  # (n_voxels x n_trials)
                        h5f.create_dataset("design_matrix", data=dm.values.astype(np.float32))
                        str_dt = h5py.string_dtype(encoding="utf-8")
                        h5f.create_dataset("design_col_names", data=np.array(all_cols, dtype=object), dtype=str_dt)

                        # Attrs
                        h5f.attrs.create("task_regressor_names", np.array(beta_cols, dtype=str_dt), dtype=str_dt)
                        h5f.attrs.create("drift_regressor_names", np.array(drift_cols, dtype=str_dt), dtype=str_dt)
                        h5f.attrs["regressor_level"] = "trial"
                        h5f.attrs["n_voxels"] = int(betas.shape[0])
                        h5f.attrs["n_trials"] = int(betas.shape[1])
                        h5f.attrs["tr"] = float(tr)
                        h5f.attrs["tmask_dropped"] = int(tmask)
                        h5f.attrs["high_pass_sec"] = float(args.high_pass_sec)
                        h5f.attrs["noise_model"] = "ar1"
                        h5f.attrs["hrf_model"] = "spm"
                        h5f.attrs["drift_model"] = "cosine"
                        h5f.attrs["smoothing_fwhm"] = float(args.smoothing_fwhm)
                        h5f.attrs["include_key"] = include_key

                    print(f"[🎯] Saved: {h5_file}")
                    print(f"[🧾] Trial CSV: {design_csv}")

                except Exception as e:
                    print(f"❌ Error on {subj} {ses} {task_name} {run}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("[✅] Done (voxel-level Nilearn GLM).")


if __name__ == "__main__":
    main()
