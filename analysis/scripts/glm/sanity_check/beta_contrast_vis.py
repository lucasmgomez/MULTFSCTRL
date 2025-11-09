import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import nibabel as nib
import os

# === Subject and task ===
subj = "sub-01"
task = "interdms"

# === Paths ===
data_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/betas")
glasser_atlas_path = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/Glasser_LR_Dense64k.dlabel.nii")
output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/sanity_check")
output_dir.mkdir(exist_ok=True, parents=True)

# === Load Glasser atlas ===
atlas_img = nib.load(str(glasser_atlas_path))
atlas_data = atlas_img.get_fdata().squeeze().astype(int)  # shape: (64984,)
n_parcels = 360

# === Find all matching runs ===
pattern = f"{subj}_*_task-{task}_*_design.csv"
design_files = list(data_root.rglob(pattern))

print(f"[🔍] Found {len(design_files)} design files for subject {subj}, task {task}")

# === Accumulate contrasts ===
all_contrasts = []

for design_file in design_files:
    h5_file = design_file.with_name(design_file.name.replace("_design.csv", "_betas.h5"))

    if not h5_file.exists():
        print(f"[⚠️] Missing betas file for {design_file}, skipping.")
        continue

    # Load design and betas
    df_design = pd.read_csv(design_file)
    with h5py.File(h5_file, 'r') as f:
        betas = f['betas'][:]  # shape: (n_parcels, n_trials)

    # Compute per-run contrast (encoding - delay)
    encoding_mask = df_design["regressor_type"] == "encoding"
    delay_mask = df_design["regressor_type"] == "delay"

    if encoding_mask.sum() == 0 or delay_mask.sum() == 0:
        print(f"[⚠️] No encoding or delay trials in {design_file.name}, skipping.")
        continue

    mean_encoding = np.mean(betas[:, encoding_mask.values], axis=1)
    mean_delay = np.mean(betas[:, delay_mask.values], axis=1)
    contrast = mean_encoding - mean_delay  # shape: (360,)
    all_contrasts.append(contrast)

# === Average across all runs ===
if not all_contrasts:
    raise RuntimeError("No valid contrast data found.")

avg_contrast = np.mean(np.stack(all_contrasts), axis=0)  # shape: (360,)

# === Project to fsLR32k surface using Glasser atlas
surface_data = np.zeros_like(atlas_data, dtype=float)
for region_idx in range(1, n_parcels + 1):  # Glasser is 1-indexed
    surface_data[atlas_data == region_idx] = avg_contrast[region_idx - 1]

# === Save contrast values ===
region_contrast_file = output_dir / f"{subj}_task-{task}_avg_contrast_per_region.npy"
surface_contrast_file = output_dir / f"{subj}_task-{task}_avg_contrast_surface.npy"
np.save(region_contrast_file, avg_contrast)
np.save(surface_contrast_file, surface_data)

print("[✅] Saved average contrast across runs:")
print(f"  ├─ Region-level:  {region_contrast_file}")
print(f"  └─ Surface-level: {surface_contrast_file}")