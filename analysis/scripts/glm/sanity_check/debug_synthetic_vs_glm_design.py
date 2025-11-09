#!/usr/bin/env python3
"""
Debug differences between synthetic dataset generation and GLM analysis.

This script investigates why the ground-truth betas and fitted betas show
a negative correlation, potentially due to differences in HRF convolution
or design matrix construction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from nilearn.glm.first_level import spm_hrf

# Paths
SYNTHETIC_DATA_ROOT = Path("/project/def-pbellec/xuan/fmri_dataset_project/synthetic_data")
SUBJ = "sub-synthetic"  
SES = "ses-001"
TASK = "ctxdm"
RUN = "run-01" 
BASE = f"{SUBJ}_{SES}_task-{TASK}_{RUN}"

def load_synthetic_design_info():
    """Load the full design matrix and parameters from synthetic generation."""
    npz_path = SYNTHETIC_DATA_ROOT / "ground-truth betas" / SUBJ / SES / "func" / f"{BASE}_full_design_and_betas.npz"
    
    print(f"Loading synthetic design from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    print("Available keys:", list(data.keys()))
    
    return data

def load_glm_design_matrix():
    """Load the GLM-fitted design matrix."""  
    h5_path = SYNTHETIC_DATA_ROOT / "trial_level_betas" / SUBJ / SES / "func" / f"{BASE}_betas.h5"
    
    print(f"Loading GLM design from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        design_matrix = f['design_matrix'][:]
        design_col_names = [str(name) for name in f['design_col_names'][:]]
        task_col_start = f.attrs['task_col_start']
        task_col_end = f.attrs['task_col_end']
        
    return design_matrix, design_col_names, task_col_start, task_col_end

def compare_hrf_convolution():
    """Compare HRF convolution approaches."""
    print("\n" + "="*50)
    print("COMPARING HRF CONVOLUTION")
    print("="*50)
    
    # Load events
    events_path = SYNTHETIC_DATA_ROOT / "reformated_behavior" / SUBJ / SES / "func" / f"{BASE}_events.tsv"
    events = pd.read_csv(events_path, sep='\t')
    print(f"Events shape: {events.shape}")
    print("Sample events:", events.head())
    
    # Parameters from synthetic generation
    TR = 1.49
    T = 400  # Original length
    tmask = 1  # Frames dropped
    T_kept = T - tmask
    
    # Create a simple boxcar for the first trial
    first_trial = events.iloc[0]
    onset = first_trial['onset_time'] 
    offset = first_trial['offset_time']
    
    print(f"First trial: {first_trial['type']}, onset={onset}, offset={offset}")
    
    # Create boxcar (synthetic approach)
    frame_times = np.arange(T) * TR
    boxcar = np.zeros(T)
    a = int(np.ceil(onset / TR))
    b = int(np.floor(offset / TR))
    if b > a:
        boxcar[a:b] = 1.0
    
    # Apply tmask (drop first frame)
    boxcar_masked = boxcar[tmask:]
    frame_times_masked = frame_times[tmask:]
    
    print(f"Boxcar shape: {boxcar.shape} -> {boxcar_masked.shape}")
    print(f"Non-zero timepoints: {np.sum(boxcar > 0)} -> {np.sum(boxcar_masked > 0)}")
    
    # HRF convolution (nilearn approach)
    hrf = spm_hrf(TR, oversampling=1)
    print(f"HRF shape: {hrf.shape}")
    
    # Convolve
    convolved = np.convolve(boxcar_masked, hrf, mode='full')[:len(boxcar_masked)]
    
    print(f"Convolved shape: {convolved.shape}")
    print(f"Convolved max: {np.max(convolved):.4f}")
    
    return boxcar_masked, convolved

def main():
    print("="*60)
    print("DEBUGGING SYNTHETIC vs GLM DESIGN DIFFERENCES")  
    print("="*60)
    
    # Load synthetic design info
    print("\n1. Loading synthetic design information...")
    synthetic_data = load_synthetic_design_info()
    
    print("Synthetic design keys:", list(synthetic_data.keys()))
    print("X shape:", synthetic_data['X'].shape)
    print("B_full shape:", synthetic_data['B_full'].shape)
    print("Task columns:", synthetic_data['task_col_count'])
    
    # Load GLM design matrix
    print("\n2. Loading GLM design matrix...")
    glm_design, glm_names, task_start, task_end = load_glm_design_matrix()
    
    print("GLM design shape:", glm_design.shape)
    print("Task columns range:", task_start, "to", task_end)
    print("Task columns count:", task_end - task_start)
    
    # Compare task regressors
    print("\n3. Comparing task regressor construction...")
    synthetic_task = synthetic_data['X'][:, synthetic_data['conf_col_count'] + synthetic_data['drift_col_count']:]
    glm_task = glm_design[:, task_start:task_end]
    
    print("Synthetic task regressors shape:", synthetic_task.shape)
    print("GLM task regressors shape:", glm_task.shape)
    
    if synthetic_task.shape == glm_task.shape:
        # Compare first regressor  
        corr = np.corrcoef(synthetic_task[:, 0], glm_task[:, 0])[0, 1]
        print(f"Correlation of first regressor: {corr:.4f}")
        
        # Check a few more
        for i in range(min(5, synthetic_task.shape[1])):
            corr = np.corrcoef(synthetic_task[:, i], glm_task[:, i])[0, 1]
            print(f"Regressor {i} correlation: {corr:.4f}")
    else:
        print("ERROR: Shape mismatch between synthetic and GLM task regressors!")
    
    # Compare HRF convolution
    print("\n4. Checking HRF convolution...")
    boxcar, convolved = compare_hrf_convolution()
    
    # Visual comparison
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(boxcar[:50])
    plt.title('Boxcar (first 50 TRs)')
    plt.xlabel('TR')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 3, 2)
    plt.plot(convolved[:50])
    plt.title('HRF-convolved (first 50 TRs)')
    plt.xlabel('TR')
    plt.ylabel('Amplitude')
    
    if synthetic_task.shape == glm_task.shape:
        plt.subplot(2, 3, 3)
        plt.plot(synthetic_task[:50, 0], label='Synthetic')
        plt.plot(glm_task[:50, 0], label='GLM')
        plt.title('First Task Regressor Comparison')
        plt.xlabel('TR')
        plt.ylabel('Amplitude')
        plt.legend()
        
        plt.subplot(2, 3, 4)
        plt.scatter(synthetic_task[:, 0], glm_task[:, 0], alpha=0.5)
        plt.plot([synthetic_task[:, 0].min(), synthetic_task[:, 0].max()], 
                 [synthetic_task[:, 0].min(), synthetic_task[:, 0].max()], 'r--')
        plt.xlabel('Synthetic Task Regressor')
        plt.ylabel('GLM Task Regressor')
        plt.title('Regressor Correlation')
        
        plt.subplot(2, 3, 5)
        plt.hist(synthetic_task.flatten(), alpha=0.5, label='Synthetic', bins=50)
        plt.hist(glm_task.flatten(), alpha=0.5, label='GLM', bins=50)
        plt.xlabel('Regressor Values')
        plt.ylabel('Count')
        plt.title('Regressor Value Distributions')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'comparison_results' / 'design_matrix_debug.png', dpi=300, bbox_inches='tight')
    print(f"Saved debug plot to: {Path(__file__).parent / 'comparison_results' / 'design_matrix_debug.png'}")
    
    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)
    print("Check the saved plots and correlation values above.")
    print("If regressor correlations are negative, there may be a sign flip")
    print("in either the HRF convolution or design matrix construction.")
    print("="*60)

if __name__ == "__main__":
    main()