#!/usr/bin/env python3
"""
Quick test of single run to debug GLM analysis
"""

import sys
sys.path.append('/project/def-pbellec/xuan/fmri_dataset_project/scripts')

from glm_analysis_nilearn import *
import argparse

# Test with just one session/run
print('🧪 Testing Nilearn GLM on single run...')

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
ses = 'ses-001'  
run = 'run-01'
task_name = 'ctxdm'
tmask = args.tmask
tr = args.tr
include_types = args.include_types
correct_only = args.correct_only

print(f'📊 Processing: {subj} {ses} {task_name} {run}')

fmri_root_dir = Path(args.fmri_root) / subj
confounds_root_dir = Path(args.conf_root) / subj  
events_root_dir = Path(args.events_root) / subj
output_dir = default_output_root(Path(args.out_root), correct_only, subj)

# Find files
behavioral_file = events_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{run}_events.tsv"
run_num = run.split('-')[1]  
run_num_short = str(int(run_num))

# Find fMRI file
possible_files = [
    fmri_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    fmri_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_space-T1w_desc-preproc_bold.nii.gz"
]

fmri_file = None
for candidate in possible_files:
    if candidate.exists():
        fmri_file = candidate
        break

if fmri_file is None:
    print("❌ fMRI file not found")
    sys.exit(1)

# Find confounds file  
confounds_candidates = [
    confounds_root_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_run-{run_num_short}_desc-confounds_timeseries.tsv"
]

confounds_file = None
for candidate in confounds_candidates:
    if candidate.exists():
        confounds_file = candidate
        break

if confounds_file is None:
    print("❌ Confounds file not found")
    sys.exit(1)

print(f"✅ Found files:")
print(f"  fMRI: {fmri_file}")
print(f"  Confounds: {confounds_file}")
print(f"  Events: {behavioral_file}")

# Test data loading and dimensions
try:
    print("\n📥 Loading data...")
    
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
    
    # Check dimensions match
    if confounds_matrix.shape[0] != fmri_img.shape[-1]:
        print(f"❌ Dimension mismatch: confounds {confounds_matrix.shape[0]} vs fMRI {fmri_img.shape[-1]}")
        sys.exit(1)
    
    print(f"✅ Dimensions match: {confounds_matrix.shape[0]} timepoints")
    
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
    
    print("\n🧠 Attempting GLM fitting...")
    
    # Create FirstLevelModel with simpler settings for testing
    glm = FirstLevelModel(
        t_r=tr,
        noise_model='ar1',
        standardize=False,
        hrf_model='spm',
        drift_model='cosine',
        high_pass=1.0 / args.high_pass_sec,
        smoothing_fwhm=args.smoothing_fwhm,
        minimize_memory=True  # Use memory saving
    )
    
    print("Fitting model...")
    glm = glm.fit(fmri_img, events_df, confounds=confounds_matrix)
    
    print("✅ GLM fitting successful!")
    
    # Test contrast computation
    design_matrix = glm.design_matrices_[0]
    trial_columns = [col for col in design_matrix.columns if 'trial' in col.lower()]
    print(f"Found {len(trial_columns)} trial regressors")
    
    if trial_columns:
        print("Testing contrast computation...")
        beta_map = glm.compute_contrast(trial_columns[0], output_type='effect_size')
        print(f"✅ Beta map computed: {beta_map.shape}")
        
        print(f"🎯 SUCCESS! Nilearn GLM working correctly")
        print(f"   - {len(trial_columns)} trial regressors")  
        print(f"   - Beta map shape: {beta_map.shape}")
        
        # Save a minimal test result
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ses / "func" / f"{subj}_{ses}_task-{task_name}_{run}_test_success.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(test_file, 'w') as f:
            f.write(f"Nilearn GLM Test Successful\n")
            f.write(f"Subject: {subj}\n")
            f.write(f"Session: {ses}\n") 
            f.write(f"Task: {task_name}\n")
            f.write(f"Run: {run}\n")
            f.write(f"Trial regressors: {len(trial_columns)}\n")
            f.write(f"Beta map shape: {beta_map.shape}\n")
        
        print(f"📄 Test result saved to: {test_file}")
    
except Exception as e:
    print(f"❌ Error during GLM processing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)