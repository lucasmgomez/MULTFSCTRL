#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

# # Compute contrasts for ALL task_types discovered from the TSV
# python compute_contrast_tasktype_concat.py --subj sub-01

# # Or specify one or more task_types explicitly
# python compute_contrast_tasktype_concat.py --subj sub-01 --task_types interdms_ctg_ABBA nback_obj

# # If your GLM used --correct_only
# python compute_contrast_tasktype_concat.py --subj sub-01 --correct_only


# -----------------------
# CLI
# -----------------------
def get_args():
    p = argparse.ArgumentParser(description="Compute (encoding - delay) contrast for concatenated GLM by task_type.")
    p.add_argument("--subj", required=True, help="e.g., sub-01")

    # If omitted, task_types are auto-discovered from the study_design TSV
    p.add_argument("--task_types", nargs="*", default=None,
                   help="Explicit task_type(s) (e.g., interdms_ctg_ABBA). If omitted, auto-discover from TSV.")

    # Roots (match your project layout)
    p.add_argument("--out_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data",
                   help="Base where betas/<subj> lives")
    p.add_argument("--results_root", default="/project/def-pbellec/xuan/fmri_dataset_project/results",
                   help="Where to save contrast npy files")
    p.add_argument("--designs_root", default="/project/def-pbellec/xuan/fmri_dataset_project/data/study_designs",
                   help="Where <subj>_design_design_with_converted.tsv lives")

    # Atlas
    p.add_argument("--atlas_path", default="/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
    p.add_argument("--n_parcels", type=int, default=360)

    # Toggle correct_betas vs betas if you used --correct_only in GLM
    p.add_argument("--correct_only", action="store_true")
    return p.parse_args()


# -----------------------
# Study design → task_type discovery
# -----------------------
def read_tasktypes_from_tsv(designs_root: Path, subj: str) -> list[str]:
    """
    Parse .../study_designs/<subj>_design_design_with_converted.tsv to discover task_types.
    task_type is the prefix of block_file_name before '_block_*'.
    """
    tsv = designs_root / f"{subj}_design_design_with_converted.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"Study design TSV not found: {tsv}")

    df = pd.read_csv(tsv, sep=r"\s*\t\s*", engine="python")
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()

    if "block_file_name" not in df.columns or "converted_file_name" not in df.columns:
        raise ValueError("TSV must contain 'block_file_name' and 'converted_file_name' columns.")

    task_types = []
    for _, row in df.iterrows():
        block = str(row["block_file_name"]).strip()
        # task_type = substring before '_block_*'
        m = re.split(r"_block_\d+", block)
        tt = m[0] if m and m[0] else block
        task_types.append(tt)

    # unique & stable order
    seen = set()
    uniq = []
    for t in task_types:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# -----------------------
# Paths and saving
# -----------------------
def glm_output_root(out_root: Path, subj: str, correct_only: bool) -> Path:
    return out_root / ("correct_betas" if correct_only else "betas") / subj

def results_mode_root(results_root: Path, subj: str) -> Path:
    # keep a distinct folder for this contrast mode
    return results_root / subj / "contrasts_tasktype_concat"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_contrast_pair(out_dir: Path, base: str, region_vals: np.ndarray, surface_vals: np.ndarray):
    ensure_dir(out_dir)
    rfile = out_dir / f"{base}_contrast_per_region.npy"
    sfile = out_dir / f"{base}_contrast_surface.npy"
    np.save(rfile, region_vals)
    np.save(sfile, surface_vals)
    print(f"[✅] Saved:\n  ├─ {rfile}\n  └─ {sfile}")


# -----------------------
# Core contrast
# -----------------------
def compute_enc_minus_delay(df_design: pd.DataFrame, betas: np.ndarray) -> np.ndarray:
    """
    Trial-level: df_design should have one row per trial; betas has one column per trial.
    We compute mean(encoding) - mean(delay) per region.
    Fallbacks included if columns differ, but this script expects trial-level from your new GLM.
    """
    K = betas.shape[1]
    # Prefer 'regressor_type'; fall back to 'type' if needed
    if "regressor_type" in df_design.columns:
        types = df_design["regressor_type"].astype(str).str.lower().values
    elif "type" in df_design.columns:
        types = df_design["type"].astype(str).str.lower().values
    else:
        raise RuntimeError("Design CSV missing 'regressor_type' (or 'type') column.")

    if len(df_design) != K:
        # If mismatch, try condition-level 2-column case
        if K == 2:
            # Assume columns [encoding, delay] in that order
            return betas[:, 0] - betas[:, 1]
        raise RuntimeError(
            f"Betas columns ({K}) != design rows ({len(df_design)}). "
            "Expected trial-level design for concatenated GLM."
        )

    enc_mask = (types == "encoding")
    del_mask = (types == "delay")

    if enc_mask.sum() == 0 or del_mask.sum() == 0:
        raise RuntimeError("No encoding or delay trials found in design.")

    mean_enc = np.mean(betas[:, enc_mask], axis=1)
    mean_del = np.mean(betas[:, del_mask], axis=1)
    return mean_enc - mean_del


def project_to_surface(region_vals: np.ndarray, atlas_path: Path, n_parcels: int) -> np.ndarray:
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata().squeeze().astype(int)  # (~64984,)
    surf = np.zeros_like(atlas_data, dtype=float)
    for region_idx in range(1, n_parcels + 1):  # Glasser is 1-indexed
        surf[atlas_data == region_idx] = region_vals[region_idx - 1]
    return surf


# -----------------------
# Main
# -----------------------
def main():
    args = get_args()

    glm_root = glm_output_root(Path(args.out_root), args.subj, args.correct_only)
    results_root = results_mode_root(Path(args.results_root), args.subj)

    # Discover task_types if not provided
    if args.task_types:
        task_types = args.task_types
    else:
        task_types = read_tasktypes_from_tsv(Path(args.designs_root), args.subj)

    if not task_types:
        print(f"[⚠️] No task_types found for {args.subj}. Nothing to do.")
        return

    print(f"[ℹ️] {args.subj}: task_types: {task_types}")

    for task_type in task_types:
        # expected concatenated outputs from your new GLM
        design_csv = glm_root / f"task-{task_type}" / "concat" / f"{args.subj}_task-{task_type}_ses-ALL_run-ALL_design.csv"
        h5_file    = glm_root / f"task-{task_type}" / "concat" / f"{args.subj}_task-{task_type}_ses-ALL_run-ALL_betas.h5"

        if not design_csv.exists() or not h5_file.exists():
            print(f"[⚠️] Missing files for {task_type}: "
                  f"{'design missing' if not design_csv.exists() else ''} "
                  f"{'betas missing' if not h5_file.exists() else ''}".strip())
            continue

        df_design = pd.read_csv(design_csv)
        with h5py.File(h5_file, "r") as f:
            betas = f["betas"][:]  # (n_regions, n_trials) in your trial-level concatenated GLM

        print(f"[🔎] {task_type}: betas {betas.shape}, design {df_design.shape}")

        # Compute contrast per region
        contrast = compute_enc_minus_delay(df_design, betas)  # (n_regions,)

        # Project to surface
        surface = project_to_surface(contrast, Path(args.atlas_path), args.n_parcels)

        # Save
        out_dir = results_root / f"task-{task_type}"
        base = f"{args.subj}_task-{task_type}_concat"
        save_contrast_pair(out_dir, base, contrast, surface)

    print("[✅] Done.")


if __name__ == "__main__":
    main()
