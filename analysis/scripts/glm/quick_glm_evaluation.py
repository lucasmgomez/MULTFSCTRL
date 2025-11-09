#!/usr/bin/env python3
"""
Quick evaluation of Nilearn GLM results vs original implementation
"""

import sys
sys.path.append('/project/def-pbellec/xuan/fmri_dataset_project/scripts')

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
import seaborn as sns

def load_glm_results(glm_file):
    """Load GLM results from HDF5 file"""
    data = {}
    with h5py.File(glm_file, 'r') as f:
        data['betas'] = f['betas'][:]
        if 'task_regressor_names' in f.attrs:
            data['task_regressor_names'] = [name.decode('utf-8') for name in f.attrs['task_regressor_names']]
        else:
            data['task_regressor_names'] = [f'trial_{i}' for i in range(data['betas'].shape[1])]
        
        # Get evaluation metrics if available
        data['eval_metrics'] = {}
        for key in f.attrs.keys():
            if key.startswith('eval_'):
                data['eval_metrics'][key[5:]] = f.attrs[key]
        
        # Get other attributes
        for key in ['n_trials', 'n_vertices', 'tr', 'noise_model']:
            if key in f.attrs:
                data[key] = f.attrs[key]
    
    return data

def compare_glm_methods(original_file, nilearn_file):
    """Compare original vs Nilearn GLM results"""
    
    print("🔍 Loading GLM results for comparison...")
    
    # Load results
    if Path(original_file).exists():
        original = load_glm_results(original_file)
        print(f"✅ Original GLM: {original['betas'].shape}")
    else:
        print(f"⚠️ Original GLM file not found: {original_file}")
        return None
    
    if Path(nilearn_file).exists():
        nilearn = load_glm_results(nilearn_file)
        print(f"✅ Nilearn GLM: {nilearn['betas'].shape}")
    else:
        print(f"⚠️ Nilearn GLM file not found: {nilearn_file}")
        return None
    
    # Compare dimensions
    orig_shape = original['betas'].shape
    nilearn_shape = nilearn['betas'].shape
    
    print(f"\n📊 Dimension Comparison:")
    print(f"   Original: {orig_shape[0]} vertices × {orig_shape[1]} trials")
    print(f"   Nilearn:  {nilearn_shape[0]} vertices × {nilearn_shape[1]} trials")
    
    if orig_shape != nilearn_shape:
        print(f"⚠️ Shape mismatch - will compare overlapping portion")
        min_vertices = min(orig_shape[0], nilearn_shape[0])
        min_trials = min(orig_shape[1], nilearn_shape[1])
        orig_betas = original['betas'][:min_vertices, :min_trials]
        nilearn_betas = nilearn['betas'][:min_vertices, :min_trials]
    else:
        orig_betas = original['betas']
        nilearn_betas = nilearn['betas']
    
    print(f"   Comparing: {orig_betas.shape}")
    
    # Compute correlations
    print(f"\n🧮 Computing correlations...")
    
    # Per-trial correlations
    trial_correlations = []
    trial_r2_scores = []
    
    for trial_idx in range(orig_betas.shape[1]):
        orig_trial = orig_betas[:, trial_idx]
        nilearn_trial = nilearn_betas[:, trial_idx]
        
        # Remove NaN/inf values
        valid_mask = np.isfinite(orig_trial) & np.isfinite(nilearn_trial)
        if np.sum(valid_mask) < 10:  # Need at least 10 valid points
            trial_correlations.append(np.nan)
            trial_r2_scores.append(np.nan)
            continue
        
        corr = np.corrcoef(orig_trial[valid_mask], nilearn_trial[valid_mask])[0, 1]
        r2 = r2_score(orig_trial[valid_mask], nilearn_trial[valid_mask])
        
        trial_correlations.append(corr)
        trial_r2_scores.append(r2)
    
    # Overall statistics
    trial_correlations = np.array(trial_correlations)
    trial_r2_scores = np.array(trial_r2_scores)
    
    valid_corr = trial_correlations[np.isfinite(trial_correlations)]
    valid_r2 = trial_r2_scores[np.isfinite(trial_r2_scores)]
    
    results = {
        'mean_correlation': np.mean(valid_corr) if len(valid_corr) > 0 else np.nan,
        'median_correlation': np.median(valid_corr) if len(valid_corr) > 0 else np.nan,
        'min_correlation': np.min(valid_corr) if len(valid_corr) > 0 else np.nan,
        'max_correlation': np.max(valid_corr) if len(valid_corr) > 0 else np.nan,
        'mean_r2': np.mean(valid_r2) if len(valid_r2) > 0 else np.nan,
        'median_r2': np.median(valid_r2) if len(valid_r2) > 0 else np.nan,
        'trial_correlations': trial_correlations,
        'trial_r2_scores': trial_r2_scores,
        'orig_beta_stats': {
            'mean': np.mean(orig_betas),
            'std': np.std(orig_betas),
            'min': np.min(orig_betas),
            'max': np.max(orig_betas)
        },
        'nilearn_beta_stats': {
            'mean': np.mean(nilearn_betas),
            'std': np.std(nilearn_betas),
            'min': np.min(nilearn_betas),
            'max': np.max(nilearn_betas)
        }
    }
    
    # Add Nilearn evaluation metrics if available
    if 'eval_metrics' in nilearn:
        results['nilearn_eval'] = nilearn['eval_metrics']
    
    return results

def print_evaluation_summary(results):
    """Print comprehensive evaluation summary"""
    
    print("\n" + "="*60)
    print("🎯 NILEARN GLM EVALUATION SUMMARY")
    print("="*60)
    
    if results is None:
        print("❌ Could not perform comparison - missing data files")
        return
    
    print(f"\n📈 AGREEMENT WITH ORIGINAL GLM:")
    print(f"   Mean correlation:    {results['mean_correlation']:.4f}")
    print(f"   Median correlation:  {results['median_correlation']:.4f}")
    print(f"   Correlation range:   [{results['min_correlation']:.4f}, {results['max_correlation']:.4f}]")
    print(f"   Mean R²:            {results['mean_r2']:.4f}")
    print(f"   Median R²:          {results['median_r2']:.4f}")
    
    # Interpret results
    mean_corr = results['mean_correlation']
    if mean_corr >= 0.9:
        agreement = "EXCELLENT 🟢"
    elif mean_corr >= 0.8:
        agreement = "VERY GOOD 🟡"
    elif mean_corr >= 0.7:
        agreement = "GOOD 🟠"
    elif mean_corr >= 0.5:
        agreement = "MODERATE 🟠"
    else:
        agreement = "POOR 🔴"
    
    print(f"\n   Agreement Quality:   {agreement}")
    
    print(f"\n📊 BETA COEFFICIENT STATISTICS:")
    orig_stats = results['orig_beta_stats']
    nilearn_stats = results['nilearn_beta_stats']
    
    print(f"   Original GLM  - Mean: {orig_stats['mean']:.4f}, Std: {orig_stats['std']:.4f}")
    print(f"   Nilearn GLM   - Mean: {nilearn_stats['mean']:.4f}, Std: {nilearn_stats['std']:.4f}")
    
    # Nilearn-specific evaluation metrics
    if 'nilearn_eval' in results and results['nilearn_eval']:
        print(f"\n🔍 NILEARN MODEL DIAGNOSTICS:")
        eval_metrics = results['nilearn_eval']
        if 'r2_mean' in eval_metrics:
            print(f"   Mean R² (model fit): {eval_metrics['r2_mean']:.4f}")
        if 'r2_median' in eval_metrics:
            print(f"   Median R² (model fit): {eval_metrics['r2_median']:.4f}")
        if 'residual_mean' in eval_metrics:
            print(f"   Mean residual:       {eval_metrics['residual_mean']:.4f}")
    
    print(f"\n💡 ASSESSMENT:")
    if mean_corr >= 0.85:
        print("   ✅ Nilearn GLM is performing EXCELLENTLY")
        print("   ✅ High agreement with original implementation")
        print("   ✅ Results are reliable for downstream analysis")
    elif mean_corr >= 0.7:
        print("   ✅ Nilearn GLM is performing WELL")
        print("   ✅ Good agreement with original implementation")
        print("   ⚠️ Minor differences may be due to implementation details")
    elif mean_corr >= 0.5:
        print("   ⚠️ Nilearn GLM shows MODERATE agreement")
        print("   ⚠️ Results may be usable but require careful validation")
        print("   🔍 Consider investigating differences in preprocessing")
    else:
        print("   ❌ Nilearn GLM shows POOR agreement")
        print("   ❌ Results may not be reliable")
        print("   🚨 Investigate implementation issues")
    
    print("="*60)

def main():
    # File paths for comparison
    base_name = "sub-01_ses-001_task-ctxdm_run-01"
    
    original_file = f"/project/def-pbellec/xuan/fmri_dataset_project/data/trial_level_betas/sub-01/ses-001/func/{base_name}_betas.h5"
    nilearn_file = f"/project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data/trial_level_betas/sub-01/ses-001/func/{base_name}_nilearn_betas.h5"
    
    print("🔬 NILEARN GLM PERFORMANCE EVALUATION")
    print("="*50)
    print(f"Comparing: {base_name}")
    print(f"Original:  {original_file}")
    print(f"Nilearn:   {nilearn_file}")
    
    # Perform comparison
    results = compare_glm_methods(original_file, nilearn_file)
    
    # Print summary
    print_evaluation_summary(results)

if __name__ == "__main__":
    main()