#!/usr/bin/env python3
"""
Debug Parcel Loading Issues

This script specifically investigates why most parcels show zero signal in the GLM evaluation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path

def debug_parcel_loading():
    # Paths
    fmri_file = "/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled/sub-01/ses-001/sub-01_ses-001_task-ctxdm_run-1_space-Glasser64k_bold.dtseries.nii"
    
    print("=== DEBUG: Parcel Loading Analysis ===")
    
    # Load raw data
    Y = nib.load(str(fmri_file)).get_fdata(dtype=np.float32)
    print(f"Raw data shape: {Y.shape}")
    
    # Apply tmask (same as GLM analysis)
    tmask = 1
    T = Y.shape[0]
    keep = np.ones((T,), dtype=bool)
    keep[:tmask] = False
    Y = Y[keep, :]
    print(f"After tmask shape: {Y.shape}")
    
    # Check first 20 parcels (what we plot in evaluation)
    max_parcels = 500
    parcels_to_check = [7, 34, 57, 58, 379]  # The ones shown in the plot
    
    print(f"\n=== Checking specific parcels ===")
    for parcel_idx in parcels_to_check:
        if parcel_idx < Y.shape[1]:
            ts = Y[:, parcel_idx]
            print(f"Parcel {parcel_idx}:")
            print(f"  Shape: {ts.shape}")
            print(f"  Range: [{np.min(ts):.6f}, {np.max(ts):.6f}]")
            print(f"  Mean: {np.mean(ts):.6f}")
            print(f"  Std: {np.std(ts):.6f}")
            print(f"  Non-zero values: {np.sum(ts != 0)}/{len(ts)}")
            print(f"  First 5 values: {ts[:5]}")
            print("")
    
    # Check subset used in evaluation
    print(f"=== Checking Y_actual_subset ===")
    Y_actual_subset = Y[:, :max_parcels]
    print(f"Y_actual_subset shape: {Y_actual_subset.shape}")
    
    # Statistics for the subset
    for i in range(min(10, max_parcels)):
        ts = Y_actual_subset[:, i]
        print(f"Subset parcel {i}: range=[{np.min(ts):.2f}, {np.max(ts):.2f}], mean={np.mean(ts):.2f}, std={np.std(ts):.2f}")
    
    # Check for zero variance parcels
    parcel_stds = np.std(Y_actual_subset, axis=0)
    zero_var_parcels = np.sum(parcel_stds == 0)
    print(f"\nZero variance parcels in subset: {zero_var_parcels}/{Y_actual_subset.shape[1]}")
    
    # Find non-zero parcels
    nonzero_parcels = np.where(parcel_stds > 0)[0]
    print(f"Non-zero variance parcels (first 10): {nonzero_parcels[:10]}")
    
    if len(nonzero_parcels) > 0:
        print(f"\nExample of non-zero parcel (#{nonzero_parcels[0]}):")
        example_ts = Y_actual_subset[:, nonzero_parcels[0]]
        print(f"  Range: [{np.min(example_ts):.2f}, {np.max(example_ts):.2f}]")
        print(f"  Mean: {np.mean(example_ts):.2f}")
        print(f"  Std: {np.std(example_ts):.2f}")

if __name__ == "__main__":
    debug_parcel_loading()