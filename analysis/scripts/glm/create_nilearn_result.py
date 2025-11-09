#!/usr/bin/env python3
"""
Create a proper Nilearn GLM result for evaluation
"""

import sys
sys.path.append('/project/def-pbellec/xuan/fmri_dataset_project/scripts')

import h5py
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from glm_analysis_nilearn import *

print('🔬 Creating Nilearn GLM result for evaluation...')

# Set up the analysis for ses-001 run-01
subj = 'sub-01'
ses = 'ses-001'  
run = 'run-01'
task_name = 'ctxdm'
tr = 1.49
tmask = 1
include_types = ['encoding', 'delay']
correct_only = False

fmri_root_dir = Path('/project/def-pbellec/xuan/cneuromod.multfs.fmriprep') / subj
confounds_root_dir = Path('/project/def-pbellec/xuan/cneuromod.multfs.fmriprep') / subj  
events_root_dir = Path('/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior') / subj
output_dir = Path('/project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data/trial_level_betas') / subj

# Find files
behavioral_file = events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{run}_events.tsv"
run_num_short = "1"  # We know it's run-1

# Find fMRI and confounds files
fmri_file = fmri_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
confounds_file = confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_desc-confounds_timeseries.tsv"

print(f"Using files:")
print(f"  fMRI: {fmri_file.exists()} - {fmri_file}")
print(f"  Confounds: {confounds_file.exists()} - {confounds_file}")
print(f"  Events: {behavioral_file.exists()} - {behavioral_file}")

if not all([fmri_file.exists(), confounds_file.exists(), behavioral_file.exists()]):
    print("❌ Missing required files")
    sys.exit(1)

# Output paths
target_subdir = output_dir / ses / "func"
base_name = f"{subj}_{ses}_task-{task_name}_{run}"
h5_file = target_subdir / f"{base_name}_nilearn_betas.h5"
design_csv = target_subdir / f"{base_name}_design.csv"

target_subdir.mkdir(parents=True, exist_ok=True)

print(f"Output will be saved to: {h5_file}")

try:
    # Load and prepare data (following our working test)
    print("📥 Loading data...")
    
    # Load fMRI
    fmri_img = nib.load(str(fmri_file))
    print(f"Original fMRI shape: {fmri_img.shape}")
    
    # Apply time masking  
    if tmask > 0:
        fmri_data = fmri_img.get_fdata()
        fmri_data_masked = fmri_data[..., tmask:]
        fmri_img = nib.Nifti1Image(fmri_data_masked, fmri_img.affine, fmri_img.header)
        print(f"After time masking: {fmri_img.shape}")
    
    # Load confounds
    df_confounds = pd.read_csv(confounds_file, sep="\t")
    print(f"Original confounds shape: {df_confounds.shape}")
    
    if tmask > 0:
        df_confounds = df_confounds.iloc[tmask:].reset_index(drop=True)
        print(f"After time masking confounds: {df_confounds.shape}")
    
    # Process confounds
    from utils import glm_confounds_construction
    confounds_processed = glm_confounds_construction(df_confounds)
    if isinstance(confounds_processed, pd.DataFrame):
        confounds_matrix = np.nan_to_num(confounds_processed.values, nan=0.0)
    else:
        confounds_matrix = np.nan_to_num(confounds_processed, nan=0.0)
    print(f"Final confounds matrix shape: {confounds_matrix.shape}")
    
    # Load events
    df_events = pd.read_csv(behavioral_file, sep="\t")
    df_events = clean_events(df_events)
    
    # Create Nilearn events
    events_df = create_nilearn_events(df_events, include_types, correct_only)
    print(f"Found {len(events_df)} events")
    
    if len(events_df) == 0:
        print("❌ No events found")
        sys.exit(1)
    
    # Adjust event onsets for time masking
    if tmask > 0:
        events_df['onset'] = events_df['onset'] - (tmask * tr)
        events_df = events_df[events_df['onset'] >= 0]
        print(f"After onset adjustment: {len(events_df)} events")
    
    print("🧠 Fitting GLM...")
    
    # Create FirstLevelModel
    glm = FirstLevelModel(
        t_r=tr,
        noise_model='ar1',
        standardize=False,
        hrf_model='spm',
        drift_model='cosine',
        high_pass=1.0 / 128.0,  # 128 second high-pass
        smoothing_fwhm=None,
        minimize_memory=True
    )
    
    print("Fitting model...")
    glm = glm.fit(fmri_img, events_df, confounds=confounds_matrix)
    
    print("✅ GLM fitting successful!")
    
    # Extract beta coefficients for task regressors
    design_matrix = glm.design_matrices_[0]
    trial_columns = [col for col in design_matrix.columns if 'trial' in col.lower()]
    print(f"Found {len(trial_columns)} trial regressors")
    
    # Get beta maps for trial regressors  
    beta_maps = []
    print("Computing beta maps...")
    for i, trial_col in enumerate(trial_columns):
        if i % 10 == 0:
            print(f"  Processing trial {i+1}/{len(trial_columns)}")
        try:
            beta_map = glm.compute_contrast(trial_col, output_type='effect_size')
            beta_maps.append(beta_map.get_fdata().flatten())
        except Exception as e:
            print(f"⚠️ Could not compute contrast for {trial_col}: {e}")
            continue
    
    if not beta_maps:
        print("❌ No valid beta maps generated")
        sys.exit(1)
    
    # Stack beta coefficients: (n_vertices, n_trials)
    betas_array = np.column_stack(beta_maps).astype(np.float32)
    print(f"Beta coefficients shape: {betas_array.shape}")
    
    # Basic evaluation metrics
    r2_values = []
    residual_values = []
    
    # Sample vertices for evaluation (to speed up)
    n_sample = min(1000, betas_array.shape[0])
    sample_indices = np.random.choice(betas_array.shape[0], n_sample, replace=False)
    
    print("Computing evaluation metrics...")
    for v_idx in sample_indices:
        # Get timeseries for this vertex
        vertex_data = fmri_img.get_fdata().reshape(-1, fmri_img.shape[-1])[v_idx, :]
        
        # Predict from design matrix and betas (simplified)
        # This is a rough approximation for evaluation
        design_task_cols = [i for i, col in enumerate(design_matrix.columns) if 'trial' in col.lower()]
        if len(design_task_cols) > 0:
            X_task = design_matrix.iloc[:, design_task_cols].values
            if X_task.shape[1] == len(beta_maps):  # Ensure dimension match
                y_pred = X_task @ betas_array[v_idx, :len(beta_maps)]
                y_true = vertex_data
                
                # Compute R²
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10))
                r2_values.append(r2)
                residual_values.append(np.mean(np.abs(y_true - y_pred)))
    
    evaluation_metrics = {
        'r2_mean': np.mean(r2_values) if r2_values else 0.0,
        'r2_median': np.median(r2_values) if r2_values else 0.0,
        'r2_std': np.std(r2_values) if r2_values else 0.0,
        'residual_mean': np.mean(residual_values) if residual_values else 0.0,
        'n_vertices_evaluated': len(sample_indices),
        'n_timepoints': fmri_img.shape[-1],
        'n_regressors': design_matrix.shape[1]
    }
    
    # Save trial information
    trial_info = events_df[['trial_type', 'onset', 'duration', 'type', 'trialNumber']].copy()
    trial_info.to_csv(design_csv, index=False)
    
    # Save GLM results
    print("💾 Saving GLM results...")
    with h5py.File(h5_file, 'w') as h5f:
        # Core outputs (compatible with existing format)
        h5f.create_dataset("betas", data=betas_array)
        h5f.create_dataset("design_matrix", data=design_matrix.values.astype(np.float32))
        
        # Design matrix column names
        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5f.create_dataset("design_col_names", 
                         data=np.array(design_matrix.columns, dtype=object), 
                         dtype=str_dtype)
        
        # Trial information
        h5f.attrs.create("task_regressor_names", 
                       np.array(trial_columns, dtype=str_dtype), 
                       dtype=str_dtype)
        h5f.attrs["regressor_level"] = "trial"
        h5f.attrs["n_trials"] = len(trial_columns)
        h5f.attrs["n_vertices"] = betas_array.shape[0]
        
        # Model parameters
        h5f.attrs["tr"] = tr
        h5f.attrs["high_pass_sec"] = 128.0
        h5f.attrs["tmask_dropped"] = tmask
        h5f.attrs["noise_model"] = "ar1"
        h5f.attrs["hrf_model"] = "spm"
        h5f.attrs["drift_model"] = "cosine"
        
        # Evaluation metrics
        for key, value in evaluation_metrics.items():
            h5f.attrs[f"eval_{key}"] = value
    
    print(f"✅ Nilearn GLM results saved to: {h5_file}")
    print(f"   Shape: {betas_array.shape}")
    print(f"   Trials: {len(trial_columns)}")
    print(f"   Mean R²: {evaluation_metrics['r2_mean']:.4f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)