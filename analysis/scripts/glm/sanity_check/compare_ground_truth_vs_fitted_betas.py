#!/usr/bin/env python3
"""
Compare Ground-Truth vs GLM-Fitted Betas from Synthetic Dataset

This script loads the ground-truth betas from the synthetic dataset generator
and the GLM-fitted betas from glm_analysis.py, then computes various quality
metrics and creates visualizations to assess how well the GLM fitting worked.

Metrics computed:
- Pearson correlation between true and fitted betas
- R² (coefficient of determination) 
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Bias (mean difference)

Visualizations:
- Scatter plot of true vs fitted betas
- Correlation heatmap across parcels  
- Residual analysis plots
- Distribution comparison plots
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuration
SYNTHETIC_DATA_ROOT = Path("/project/def-pbellec/xuan/fmri_dataset_project/synthetic_data")
SUBJ = "sub-synthetic"
SES = "ses-001"
TASK = "ctxdm"  
RUN = "run-01"
BASE = f"{SUBJ}_{SES}_task-{TASK}_{RUN}"

# Paths
GT_BETAS_PATH = SYNTHETIC_DATA_ROOT / "ground-truth betas" / SUBJ / SES / "func" / f"{BASE}_ground_truth_betas.h5"
FITTED_BETAS_PATH = SYNTHETIC_DATA_ROOT / "trial_level_betas" / SUBJ / SES / "func" / f"{BASE}_betas.h5"
OUTPUT_DIR = Path(__file__).parent / "comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_ground_truth_betas():
    """Load ground-truth betas from synthetic dataset generator."""
    print(f"Loading ground-truth betas from: {GT_BETAS_PATH}")
    
    with h5py.File(GT_BETAS_PATH, 'r') as f:
        # Ground-truth betas: (P x K_task) 
        gt_betas = f['betas'][:]
        gt_names = [name.decode('utf-8') for name in f['task_regressor_names'][:]]
        
        print(f"Ground-truth betas shape: {gt_betas.shape}")
        print(f"Number of regressors: {len(gt_names)}")
        print(f"Sample regressor names: {gt_names[:5]}")
        
    return gt_betas, gt_names

def load_fitted_betas():
    """Load GLM-fitted betas from analysis results."""
    print(f"Loading fitted betas from: {FITTED_BETAS_PATH}")
    
    with h5py.File(FITTED_BETAS_PATH, 'r') as f:
        # Fitted betas: (P x K_task)
        fitted_betas = f['betas'][:]
        
        # Handle task regressor names (numpy array of strings)
        raw_names = f.attrs['task_regressor_names']
        fitted_names = [str(name) for name in raw_names]
        
        print(f"Fitted betas shape: {fitted_betas.shape}")
        print(f"Number of regressors: {len(fitted_names)}")
        print(f"Sample regressor names: {fitted_names[:5]}")
        
    return fitted_betas, fitted_names

def compute_quality_metrics(true_betas, fitted_betas):
    """Compute various quality metrics comparing true vs fitted betas."""
    
    # Flatten for overall metrics
    true_flat = true_betas.flatten()
    fitted_flat = fitted_betas.flatten()
    
    # Overall metrics
    correlation, p_value = stats.pearsonr(true_flat, fitted_flat)
    r2 = r2_score(true_flat, fitted_flat)
    mae = mean_absolute_error(true_flat, fitted_flat) 
    rmse = np.sqrt(mean_squared_error(true_flat, fitted_flat))
    bias = np.mean(fitted_flat - true_flat)
    
    # Per-parcel correlations
    n_parcels = true_betas.shape[0]
    parcel_correlations = []
    parcel_r2 = []
    
    for p in range(n_parcels):
        if np.std(true_betas[p, :]) > 1e-10 and np.std(fitted_betas[p, :]) > 1e-10:
            corr, _ = stats.pearsonr(true_betas[p, :], fitted_betas[p, :])
            r2_parcel = r2_score(true_betas[p, :], fitted_betas[p, :])
        else:
            corr, r2_parcel = np.nan, np.nan
        parcel_correlations.append(corr)
        parcel_r2.append(r2_parcel)
    
    parcel_correlations = np.array(parcel_correlations)
    parcel_r2 = np.array(parcel_r2)
    
    # Per-regressor correlations  
    n_regressors = true_betas.shape[1]
    regressor_correlations = []
    regressor_r2 = []
    
    for r in range(n_regressors):
        if np.std(true_betas[:, r]) > 1e-10 and np.std(fitted_betas[:, r]) > 1e-10:
            corr, _ = stats.pearsonr(true_betas[:, r], fitted_betas[:, r])
            r2_reg = r2_score(true_betas[:, r], fitted_betas[:, r])
        else:
            corr, r2_reg = np.nan, np.nan
        regressor_correlations.append(corr)
        regressor_r2.append(r2_reg)
    
    regressor_correlations = np.array(regressor_correlations)
    regressor_r2 = np.array(regressor_r2)
    
    metrics = {
        'overall': {
            'correlation': correlation,
            'p_value': p_value, 
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'bias': bias
        },
        'per_parcel': {
            'correlations': parcel_correlations,
            'r2': parcel_r2,
            'mean_correlation': np.nanmean(parcel_correlations),
            'mean_r2': np.nanmean(parcel_r2)
        },
        'per_regressor': {
            'correlations': regressor_correlations, 
            'r2': regressor_r2,
            'mean_correlation': np.nanmean(regressor_correlations),
            'mean_r2': np.nanmean(regressor_r2)
        }
    }
    
    return metrics

def create_visualizations(true_betas, fitted_betas, metrics, regressor_names):
    """Create comprehensive visualizations comparing true vs fitted betas."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall scatter plot
    ax1 = plt.subplot(3, 3, 1)
    true_flat = true_betas.flatten()
    fitted_flat = fitted_betas.flatten()
    
    plt.scatter(true_flat, fitted_flat, alpha=0.5, s=1)
    plt.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--', lw=2)
    plt.xlabel('Ground-Truth Betas')
    plt.ylabel('GLM-Fitted Betas') 
    plt.title(f'Overall Comparison\nr = {metrics["overall"]["correlation"]:.3f}, R² = {metrics["overall"]["r2"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals vs fitted
    ax2 = plt.subplot(3, 3, 2)
    residuals = fitted_flat - true_flat
    plt.scatter(fitted_flat, residuals, alpha=0.5, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('GLM-Fitted Betas')
    plt.ylabel('Residuals (Fitted - True)')
    plt.title('Residuals vs Fitted Values')
    plt.grid(True, alpha=0.3)
    
    # 3. Q-Q plot of residuals
    ax3 = plt.subplot(3, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    ax4 = plt.subplot(3, 3, 4)
    plt.hist(true_flat, bins=50, alpha=0.7, label='Ground-Truth', density=True)
    plt.hist(fitted_flat, bins=50, alpha=0.7, label='GLM-Fitted', density=True)
    plt.xlabel('Beta Values')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Per-parcel correlation distribution
    ax5 = plt.subplot(3, 3, 5)
    valid_corrs = metrics['per_parcel']['correlations'][~np.isnan(metrics['per_parcel']['correlations'])]
    plt.hist(valid_corrs, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(valid_corrs), color='red', linestyle='--', label=f'Mean = {np.mean(valid_corrs):.3f}')
    plt.xlabel('Correlation (per parcel)')
    plt.ylabel('Count')
    plt.title('Per-Parcel Correlation Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Per-regressor correlation
    ax6 = plt.subplot(3, 3, 6)
    valid_reg_corrs = metrics['per_regressor']['correlations'][~np.isnan(metrics['per_regressor']['correlations'])]
    plt.bar(range(len(valid_reg_corrs)), valid_reg_corrs)
    plt.xlabel('Regressor Index')
    plt.ylabel('Correlation')
    plt.title('Per-Regressor Correlations')
    plt.xticks(range(0, len(valid_reg_corrs), 5))
    plt.grid(True, alpha=0.3)
    
    # 7. Correlation heatmap (sample of parcels)
    ax7 = plt.subplot(3, 3, 7)
    sample_parcels = slice(0, min(20, true_betas.shape[0]))  # First 20 parcels
    sample_regressors = slice(0, min(10, true_betas.shape[1]))  # First 10 regressors
    
    corr_matrix = []
    for p in range(sample_parcels.stop):
        for r in range(sample_regressors.stop):
            if p < len(metrics['per_parcel']['correlations']):
                corr_matrix.append(metrics['per_parcel']['correlations'][p])
            else:
                corr_matrix.append(np.nan)
    
    # Simpler heatmap: show correlation between first few parcels and regressors
    subset_true = true_betas[sample_parcels, sample_regressors]
    subset_fitted = fitted_betas[sample_parcels, sample_regressors]
    
    im = plt.imshow(subset_true, aspect='auto', cmap='RdBu_r')
    plt.colorbar(im, shrink=0.8)
    plt.xlabel('Regressors (sample)')
    plt.ylabel('Parcels (sample)')
    plt.title('Ground-Truth Betas\n(Sample Subset)')
    
    # 8. Fitted betas heatmap
    ax8 = plt.subplot(3, 3, 8)
    im = plt.imshow(subset_fitted, aspect='auto', cmap='RdBu_r')
    plt.colorbar(im, shrink=0.8)
    plt.xlabel('Regressors (sample)')
    plt.ylabel('Parcels (sample)')
    plt.title('GLM-Fitted Betas\n(Sample Subset)')
    
    # 9. Error metrics summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    FITTING QUALITY SUMMARY
    
    Overall Metrics:
    • Correlation: {metrics['overall']['correlation']:.4f}
    • R²: {metrics['overall']['r2']:.4f}
    • MAE: {metrics['overall']['mae']:.4f}
    • RMSE: {metrics['overall']['rmse']:.4f}
    • Bias: {metrics['overall']['bias']:.4f}
    
    Per-Parcel Average:
    • Correlation: {metrics['per_parcel']['mean_correlation']:.4f}
    • R²: {metrics['per_parcel']['mean_r2']:.4f}
    
    Per-Regressor Average:
    • Correlation: {metrics['per_regressor']['mean_correlation']:.4f}
    • R²: {metrics['per_regressor']['mean_r2']:.4f}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ground_truth_vs_fitted_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {OUTPUT_DIR / 'ground_truth_vs_fitted_comparison.png'}")
    
    return fig

def generate_quality_report(metrics, regressor_names):
    """Generate a detailed text report explaining the fit quality."""
    
    report = f"""
# GLM Fitting Quality Assessment Report

## Dataset Information
- Subject: {SUBJ}
- Session: {SES}
- Task: {TASK}
- Run: {RUN}
- Number of parcels: {len(metrics['per_parcel']['correlations'])}
- Number of task regressors: {len(regressor_names)}

## Overall Fitting Performance

### Primary Metrics
- **Pearson Correlation**: {metrics['overall']['correlation']:.4f} (p = {metrics['overall']['p_value']:.2e})
- **R² (Coefficient of Determination)**: {metrics['overall']['r2']:.4f}
- **Mean Absolute Error (MAE)**: {metrics['overall']['mae']:.4f}
- **Root Mean Square Error (RMSE)**: {metrics['overall']['rmse']:.4f}
- **Bias (Mean Difference)**: {metrics['overall']['bias']:.4f}

### Interpretation of Overall Results

**Correlation ({metrics['overall']['correlation']:.4f}):**
"""
    
    # Interpret correlation
    corr = metrics['overall']['correlation']
    if corr > 0.9:
        report += "EXCELLENT - Nearly perfect linear relationship between ground-truth and fitted betas."
    elif corr > 0.8:
        report += "VERY GOOD - Strong linear relationship, indicating accurate GLM fitting."
    elif corr > 0.7:
        report += "GOOD - Moderate to strong relationship, fitting is working well overall."
    elif corr > 0.5:
        report += "FAIR - Moderate relationship, some systematic issues may be present."
    else:
        report += "POOR - Weak relationship, significant fitting problems detected."
    
    report += f"""

**R² ({metrics['overall']['r2']:.4f}):**
"""
    
    # Interpret R²
    r2 = metrics['overall']['r2']
    if r2 > 0.9:
        report += f"EXCELLENT - GLM explains {r2*100:.1f}% of variance in ground-truth betas."
    elif r2 > 0.8:
        report += f"VERY GOOD - GLM explains {r2*100:.1f}% of variance, indicating high accuracy."
    elif r2 > 0.6:
        report += f"GOOD - GLM explains {r2*100:.1f}% of variance, reasonably accurate fitting."
    elif r2 > 0.4:
        report += f"FAIR - GLM explains {r2*100:.1f}% of variance, moderate accuracy."
    else:
        report += f"POOR - GLM explains only {r2*100:.1f}% of variance, fitting issues present."
    
    report += f"""

**Bias ({metrics['overall']['bias']:.4f}):**
"""
    
    # Interpret bias
    bias = abs(metrics['overall']['bias'])
    if bias < 0.01:
        report += "EXCELLENT - Minimal systematic bias in the fitted estimates."
    elif bias < 0.05:
        report += "GOOD - Low systematic bias, estimates are well-centered."
    elif bias < 0.1:
        report += "FAIR - Some systematic bias present, but manageable."
    else:
        report += "POOR - Significant systematic bias detected in fitted estimates."
    
    # Per-parcel analysis
    valid_parcel_corrs = metrics['per_parcel']['correlations'][~np.isnan(metrics['per_parcel']['correlations'])]
    report += f"""

## Per-Parcel Analysis

- **Mean correlation across parcels**: {metrics['per_parcel']['mean_correlation']:.4f}
- **Mean R² across parcels**: {metrics['per_parcel']['mean_r2']:.4f}
- **Number of valid parcels**: {len(valid_parcel_corrs)}
- **Parcels with correlation > 0.8**: {np.sum(valid_parcel_corrs > 0.8)} ({np.sum(valid_parcel_corrs > 0.8)/len(valid_parcel_corrs)*100:.1f}%)
- **Parcels with correlation > 0.5**: {np.sum(valid_parcel_corrs > 0.5)} ({np.sum(valid_parcel_corrs > 0.5)/len(valid_parcel_corrs)*100:.1f}%)
"""
    
    # Per-regressor analysis
    valid_reg_corrs = metrics['per_regressor']['correlations'][~np.isnan(metrics['per_regressor']['correlations'])]
    report += f"""

## Per-Regressor Analysis

- **Mean correlation across regressors**: {metrics['per_regressor']['mean_correlation']:.4f}
- **Mean R² across regressors**: {metrics['per_regressor']['mean_r2']:.4f}
- **Number of valid regressors**: {len(valid_reg_corrs)}
- **Regressors with correlation > 0.8**: {np.sum(valid_reg_corrs > 0.8)} ({np.sum(valid_reg_corrs > 0.8)/len(valid_reg_corrs)*100:.1f}%)
- **Regressors with correlation > 0.5**: {np.sum(valid_reg_corrs > 0.5)} ({np.sum(valid_reg_corrs > 0.5)/len(valid_reg_corrs)*100:.1f}%)
"""
    
    # Overall assessment
    report += f"""

## Overall Assessment

Based on the comprehensive analysis above:

"""
    
    if corr > 0.8 and r2 > 0.6 and bias < 0.05:
        report += """
**CONCLUSION: EXCELLENT GLM PERFORMANCE**

The GLM fitting is working very well:
- Strong correlation indicates linear relationship is preserved
- High R² shows most variance is explained 
- Low bias suggests unbiased estimates
- Consistent performance across parcels and regressors

This synthetic dataset validation demonstrates that the GLM analysis pipeline
is correctly implemented and produces accurate results.
"""
    elif corr > 0.6 and r2 > 0.4:
        report += """
**CONCLUSION: GOOD GLM PERFORMANCE**

The GLM fitting is working adequately:
- Moderate to good correlation shows relationship is captured
- Reasonable R² indicates decent variance explanation
- Performance varies across parcels/regressors but overall acceptable

The GLM analysis pipeline appears to be working correctly with some expected
noise and variability typical in real fMRI analysis.
"""
    else:
        report += """
**CONCLUSION: GLM PERFORMANCE ISSUES DETECTED**

The GLM fitting shows concerning results:
- Low correlation suggests poor relationship recovery
- Low R² indicates poor variance explanation
- Large bias or inconsistent performance across parcels

This may indicate issues with:
- GLM implementation bugs
- Inappropriate model specification
- Numerical instability or convergence problems
- Mismatch between synthetic data generation and GLM assumptions

Further investigation recommended.
"""
    
    report += f"""

## Technical Notes

- All metrics computed on flattened beta matrices
- NaN values excluded from per-parcel/regressor statistics
- Synthetic data generated with AR(1) noise (ρ = 0.3, σ = 1.0)
- GLM includes confound regression and high-pass filtering
- Comparison represents validation of entire analysis pipeline

Generated on: {pd.Timestamp.now()}
"""
    
    return report

def main():
    """Main function to run the comparison analysis."""
    print("="*60)
    print("GROUND-TRUTH vs GLM-FITTED BETAS COMPARISON")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    gt_betas, gt_names = load_ground_truth_betas()
    fitted_betas, fitted_names = load_fitted_betas()
    
    # Verify regressor name alignment
    if gt_names != fitted_names:
        print("WARNING: Regressor names don't match exactly!")
        print(f"GT names: {gt_names[:3]}...")
        print(f"Fitted names: {fitted_names[:3]}...")
    
    # Compute metrics
    print("\n2. Computing quality metrics...")
    metrics = compute_quality_metrics(gt_betas, fitted_betas)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    fig = create_visualizations(gt_betas, fitted_betas, metrics, gt_names)
    
    # Generate report
    print("\n4. Generating quality report...")
    report = generate_quality_report(metrics, gt_names)
    
    # Save report
    report_path = OUTPUT_DIR / 'fitting_quality_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved quality report to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    print(f"Overall Correlation: {metrics['overall']['correlation']:.4f}")
    print(f"Overall R²: {metrics['overall']['r2']:.4f}")  
    print(f"Mean Absolute Error: {metrics['overall']['mae']:.4f}")
    print(f"Root Mean Square Error: {metrics['overall']['rmse']:.4f}")
    print(f"Bias: {metrics['overall']['bias']:.4f}")
    print("\nPer-Parcel Averages:")
    print(f"  Correlation: {metrics['per_parcel']['mean_correlation']:.4f}")
    print(f"  R²: {metrics['per_parcel']['mean_r2']:.4f}")
    print("\nPer-Regressor Averages:")
    print(f"  Correlation: {metrics['per_regressor']['mean_correlation']:.4f}")
    print(f"  R²: {metrics['per_regressor']['mean_r2']:.4f}")
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    main()