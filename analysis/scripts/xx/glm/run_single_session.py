#!/usr/bin/env python3
"""
Run GLM analysis on single session for testing
"""

import sys
sys.path.append('/project/def-pbellec/xuan/fmri_dataset_project/scripts')

from glm_analysis_nilearn import *
import argparse

print('🚀 Running Nilearn GLM on ses-001 only...')

# Set up args
args = argparse.Namespace(
    subj='sub-01',
    tr=1.49,
    tmask=1,
    correct_only=False,
    tasks=['ctxdm'],
    include_types=['encoding', 'delay'],
    fmri_root='/project/def-pbellec/xuan/cneuromod.multfs.fmriprep',
    conf_root='/project/def-pbellec/xuan/cneuromod.multfs.fmriprep',
    events_root='/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior',
    out_root='/project/def-pbellec/xuan/fmri_dataset_project/data',
    high_pass_sec=128.0,
    smoothing_fwhm=None,
    max_vertices=1000,
    overwrite=True
)

subj = args.subj
tr = args.tr
tmask = args.tmask
correct_only = args.correct_only
include_types = [t.lower() for t in args.include_types]

fmri_root_dir = Path(args.fmri_root) / subj
confounds_root_dir = Path(args.conf_root) / subj
events_root_dir = Path(args.events_root) / subj
output_dir = default_output_root(Path(args.out_root), correct_only, subj)

# Process only ses-001
sessions = ['ses-001']
print(f"Processing sessions: {sessions}")

for task_name in args.tasks:
    for ses in sessions:
        runs = discover_runs_for_task_session(subj, ses, task_name, fmri_root_dir)
        print(f"[{task_name} | {ses}] runs: {runs}")
        
        for run in runs:
            print(f"\nProcessing {subj} {ses} {task_name} {run}")
            
            # Input file paths
            behavioral_file = events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{run}_events.tsv"
            
            if not behavioral_file.exists():
                print(f"⚠️ Behavioral file not found: {behavioral_file}")
                continue
            
            # fMRI data from fmriprep - handle different run formats
            run_num = run.split('-')[1]  
            run_num_short = str(int(run_num))  
            
            # Try different run number formats and spaces
            possible_files = [
                fmri_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                fmri_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                fmri_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num}_space-T1w_desc-preproc_bold.nii.gz",
                fmri_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-T1w_desc-preproc_bold.nii.gz"
            ]
            
            fmri_file = None
            for candidate in possible_files:
                if candidate.exists():
                    fmri_file = candidate
                    break
            
            # Try alternative space if specific files don't exist
            if fmri_file is None:
                alt_patterns = [
                    f"{subj}_{ses}_task-{task_name}_run-{run_num}_space-*_desc-preproc_bold.nii.gz",
                    f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-*_desc-preproc_bold.nii.gz"
                ]
                for pattern in alt_patterns:
                    matches = list((fmri_root_dir / ses / "func").glob(pattern))
                    if matches:
                        fmri_file = matches[0]
                        break
            
            if fmri_file is None:
                print(f"⚠️ fMRI file not found for {ses} {run} {task_name}")
                continue
            
            # Try different confounds file formats
            confounds_candidates = [
                confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num}_desc-confounds_timeseries.tsv",
                confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_desc-confounds_timeseries.tsv"
            ]
            
            confounds_file = None
            for candidate in confounds_candidates:
                if candidate.exists():
                    confounds_file = candidate
                    break
            
            if confounds_file is None:
                print(f"⚠️ Confounds file not found for {ses} {run} {task_name}")
                continue
            
            # Output paths
            rel_behavioral_path = behavioral_file.relative_to(events_root_dir)
            target_subdir = output_dir / rel_behavioral_path.parent
            base_name = behavioral_file.stem.replace("_events", "")
            
            h5_file = target_subdir / f"{base_name}_nilearn_betas.h5"
            design_csv = target_subdir / f"{base_name}_design.csv"
            
            if h5_file.exists() and not args.overwrite:
                print(f"[⏩] Skipping {h5_file} — already exists.")
                continue
            
            # Create output directory
            target_subdir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Load and prepare data
                print("Loading fMRI data...")
                fmri_img = nib.load(str(fmri_file))
                
                print(f"Original fMRI timepoints: {fmri_img.shape[-1]}")
                
                # Apply time masking if needed
                if tmask > 0:
                    fmri_data = fmri_img.get_fdata()
                    fmri_data_masked = fmri_data[..., tmask:]  # Remove first tmask timepoints
                    fmri_img = nib.Nifti1Image(fmri_data_masked, fmri_img.affine, fmri_img.header)
                    print(f"After time masking: {fmri_img.shape[-1]} timepoints")
                
                # Load events
                print("Processing events...")
                df_events = pd.read_csv(behavioral_file, sep="\t")
                df_events = clean_events(df_events)
                
                # Create Nilearn-format events
                events_df = create_nilearn_events(df_events, include_types, correct_only)
                
                if len(events_df) == 0:
                    print(f"⚠️ No events of types {include_types} found")
                    continue
                
                # Adjust event onsets for time masking
                if tmask > 0:
                    events_df['onset'] = events_df['onset'] - (tmask * tr)
                    events_df = events_df[events_df['onset'] >= 0]
                
                print(f"Found {len(events_df)} events")
                
                # Load confounds
                print("Loading confounds...")
                df_confounds = pd.read_csv(confounds_file, sep="\t")
                
                print(f"Original confounds shape: {df_confounds.shape}")
                
                # Apply time masking to confounds to match fMRI
                if tmask > 0:
                    df_confounds = df_confounds.iloc[tmask:].reset_index(drop=True)
                    print(f"After time masking confounds: {df_confounds.shape}")
                
                # Process confounds using existing function
                confounds_processed = glm_confounds_construction(df_confounds)
                if isinstance(confounds_processed, pd.DataFrame):
                    confounds_matrix = np.nan_to_num(confounds_processed.values, nan=0.0)
                    confounds_columns = list(confounds_processed.columns)
                else:
                    confounds_matrix = np.nan_to_num(confounds_processed, nan=0.0)
                    confounds_columns = [f"conf_{i}" for i in range(confounds_matrix.shape[1])]
                
                print(f"Final confounds matrix shape: {confounds_matrix.shape}")
                print(f"Final fMRI timepoints: {fmri_img.shape[-1]}")
                
                # Check dimension match
                if confounds_matrix.shape[0] != fmri_img.shape[-1]:
                    print(f"❌ Dimension mismatch: confounds {confounds_matrix.shape[0]} vs fMRI {fmri_img.shape[-1]}")
                    continue
                
                # Create FirstLevelModel
                print("Fitting GLM with Nilearn FirstLevelModel...")
                glm = FirstLevelModel(
                    t_r=tr,
                    noise_model='ar1',  # Use AR(1) like the original implementation
                    standardize=False,
                    hrf_model='spm',  # Use SPM canonical HRF
                    drift_model='cosine',
                    high_pass=1.0 / args.high_pass_sec,
                    smoothing_fwhm=args.smoothing_fwhm,
                    minimize_memory=True
                )
                
                # Fit the model
                glm = glm.fit(fmri_img, events_df, confounds=confounds_matrix)
                
                print("GLM fitting completed successfully")
                
                # Extract beta coefficients for task regressors
                design_matrix = glm.design_matrices_[0]
                
                # Find task regressor columns
                trial_columns = [col for col in design_matrix.columns if 'trial' in col.lower()]
                if not trial_columns:
                    print("⚠️ No trial regressors found in design matrix")
                    continue
                
                print(f"Found {len(trial_columns)} trial regressors")
                
                # Get beta maps for trial regressors  
                beta_maps = []
                for trial_col in trial_columns:
                    try:
                        beta_map = glm.compute_contrast(trial_col, output_type='effect_size')
                        beta_maps.append(beta_map.get_fdata().flatten())
                    except Exception as e:
                        print(f"⚠️ Could not compute contrast for {trial_col}: {e}")
                        continue
                
                if not beta_maps:
                    print("⚠️ No valid beta maps generated")
                    continue
                
                # Stack beta coefficients: (n_vertices, n_trials)
                betas_array = np.column_stack(beta_maps).astype(np.float32)
                
                print(f"Beta coefficients shape: {betas_array.shape}")
                
                # Save trial information
                trial_info = events_df[['trial_type', 'onset', 'duration', 'type', 'trialNumber']].copy()
                trial_info.to_csv(design_csv, index=False)
                
                # Perform evaluation
                print("Evaluating GLM performance...")
                evaluation_metrics = evaluate_glm_performance(
                    fmri_img, glm, events_df, target_subdir, base_name, args.max_vertices
                )
                
                # Save GLM results
                print("Saving GLM results...")
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
                    h5f.attrs["high_pass_sec"] = args.high_pass_sec
                    h5f.attrs["tmask_dropped"] = tmask
                    h5f.attrs["noise_model"] = "ar1"
                    h5f.attrs["hrf_model"] = "spm"
                    h5f.attrs["drift_model"] = "cosine"
                    
                    # Evaluation metrics
                    for key, value in evaluation_metrics.items():
                        h5f.attrs[f"eval_{key}"] = value
                
                print(f"[🎯] Saved Nilearn GLM results to {h5_file}")
                print(f"[🧾] Trial design CSV: {design_csv}")
                print(f"[📊] Mean R²: {evaluation_metrics['r2_mean']:.4f}")
                
            except Exception as e:
                print(f"❌ Error processing {subj} {ses} {task_name} {run}: {e}")
                import traceback
                traceback.print_exc()
                continue

print("[✅] Single session GLM analysis completed.")