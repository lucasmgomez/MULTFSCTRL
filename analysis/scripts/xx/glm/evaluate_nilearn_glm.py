#!/usr/bin/env python3
"""
Nilearn GLM Evaluation Following Official Tutorial

This script evaluates GLM results from glm_analysis_nilearn.py following the official
Nilearn tutorial procedures from plot_predictions_residuals.html:
https://nilearn.github.io/stable/auto_examples/04_glm_first_level/plot_predictions_residuals.html

Features:
1. Sample vertices with non-zero variance (avoid silent vertices)
2. Create predictions and residuals plots following Nilearn patterns
3. Comprehensive performance metrics and brain visualizations
4. Silent vertices percentage analysis
5. Residuals distribution analysis

Usage:
python evaluate_nilearn_glm.py \
  --subj sub-01 \
  --task ctxdm \
  --ses ses-001 \
  --run run-02 \
  --glm_root /project/def-pbellec/xuan/fmri_dat/nilearn_data/trial_level_betas \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/results/nilearn_evaluation
"""

from __future__ import annotations
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for better plots
plt.style.use('default')
sns.set_palette('husl')

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Nilearn GLM results comprehensively")
    parser.add_argument("--subj", default="sub-01", help="Subject ID")
    parser.add_argument("--task", default="ctxdm", help="Task name")
    parser.add_argument("--ses", default="ses-001", help="Session")
    parser.add_argument("--run", default="run-02", help="Run")
    parser.add_argument("--glm_root", 
                       default="/project/def-pbellec/xuan/fmri_dat/nilearn_data/trial_level_betas",
                       help="Root directory containing Nilearn GLM results")
    parser.add_argument("--fmri_root",
                       default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep", 
                       help="Root directory containing original fMRI data")
    parser.add_argument("--output_dir",
                       default="/project/def-pbellec/xuan/fmri_dataset_project/results/nilearn_evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--n_vertices_sample", type=int, default=1000,
                       help="Number of vertices to sample for detailed analysis")
    parser.add_argument("--variance_threshold", type=float, default=1e-8,
                       help="Minimum variance threshold for non-silent vertices")
    parser.add_argument("--compare_original", action="store_true",
                       help="Compare with original GLM implementation if available")
    return parser.parse_args()

def load_nilearn_glm_results(glm_file):
    """Load Nilearn GLM results from HDF5 file"""
    print(f"Loading Nilearn GLM results from: {glm_file}")
    
    data = {}
    with h5py.File(glm_file, 'r') as f:
        # Core data
        data['betas'] = f['betas'][:]  # (n_vertices, n_trials)
        data['design_matrix'] = f['design_matrix'][:]  # (n_timepoints, n_regressors)
        data['design_col_names'] = [name.decode('utf-8') if isinstance(name, bytes) else name for name in f['design_col_names'][:]]
        
        # Attributes
        data['task_regressor_names'] = [name.decode('utf-8') if isinstance(name, bytes) else name for name in f.attrs['task_regressor_names']]
        data['n_trials'] = f.attrs['n_trials']
        data['n_vertices'] = f.attrs['n_vertices']
        data['tr'] = f.attrs.get('tr', 1.49)
        data['noise_model'] = f.attrs.get('noise_model', 'unknown').decode('utf-8') if isinstance(f.attrs.get('noise_model', ''), bytes) else f.attrs.get('noise_model', 'unknown')
        
        # Evaluation metrics if available
        data['eval_metrics'] = {}
        for key in f.attrs.keys():
            if key.startswith('eval_'):
                metric_name = key[5:]  # Remove 'eval_' prefix
                data['eval_metrics'][metric_name] = f.attrs[key]
    
    print(f"Loaded betas shape: {data['betas'].shape}")
    print(f"Number of trials: {data['n_trials']}")
    print(f"Number of vertices: {data['n_vertices']}")
    
    return data

def load_original_glm_results(original_glm_file):
    """Load original GLM results for comparison"""
    if not Path(original_glm_file).exists():
        print(f"Original GLM file not found: {original_glm_file}")
        return None
    
    print(f"Loading original GLM results from: {original_glm_file}")
    
    data = {}
    with h5py.File(original_glm_file, 'r') as f:
        data['betas'] = f['betas'][:]  # (n_vertices, n_trials)
        data['sigma2'] = f['sigma2'][:]  # (n_vertices,)
        data['task_regressor_names'] = [name.decode('utf-8') for name in f.attrs['task_regressor_names']]
        data['dof'] = f.attrs['dof']
        data['rho_ar1'] = f.attrs['rho_ar1']
    
    print(f"Original betas shape: {data['betas'].shape}")
    return data

def load_fmri_data_for_evaluation(fmri_file, tmask=1):
    """Load fMRI data for evaluation (matching GLM preprocessing)"""
    print(f"Loading fMRI data from: {fmri_file}")
    
    img = nib.load(str(fmri_file))
    Y = img.get_fdata(dtype=np.float32)
    
    # Reshape to 2D: (n_timepoints, n_vertices/voxels)
    if Y.ndim == 4:  # Volumetric data (x,y,z,t)
        Y = Y.reshape(-1, Y.shape[-1]).T  # (t, voxels)
    elif Y.ndim == 2:  # Already 2D (t, vertices)
        pass
    else:
        raise ValueError(f"Unexpected fMRI data shape: {Y.shape}")
    
    # Apply tmask (same as in GLM analysis)
    if tmask > 0:
        Y = Y[tmask:, :]
        print(f"Applied tmask: {tmask} TRs dropped, remaining shape: {Y.shape}")
    
    return Y

def select_informative_vertices(Y_actual, betas, n_sample=1000, variance_threshold=1e-8):
    """Select vertices with non-zero variance following Nilearn tutorial principles"""
    print("Selecting informative vertices with non-zero variance...")
    
    n_timepoints, n_vertices_total = Y_actual.shape
    n_vertices_glm = betas.shape[0]
    
    # Use the minimum number of vertices available
    n_vertices = min(n_vertices_glm, n_vertices_total)
    
    # Compute variance across time for each vertex
    vertex_variances = np.var(Y_actual[:, :n_vertices], axis=0)
    
    # Identify non-silent vertices (above variance threshold)
    non_silent_mask = vertex_variances > variance_threshold
    non_silent_indices = np.where(non_silent_mask)[0]
    silent_vertices = np.sum(~non_silent_mask)
    silent_percentage = (silent_vertices / n_vertices) * 100
    
    print(f"Total vertices: {n_vertices:,}")
    print(f"Silent vertices (variance ≤ {variance_threshold}): {silent_vertices:,} ({silent_percentage:.2f}%)")
    print(f"Non-silent vertices: {len(non_silent_indices):,} ({100-silent_percentage:.2f}%)")
    
    # Sample from non-silent vertices
    if len(non_silent_indices) == 0:
        raise ValueError("No non-silent vertices found! Check variance threshold.")
    
    if len(non_silent_indices) > n_sample:
        # Sample vertices with higher variance preferentially
        vertex_variances_filtered = vertex_variances[non_silent_indices]
        # Select top variance vertices and random sample from remaining
        top_variance_count = min(n_sample // 2, len(non_silent_indices))
        top_variance_idx = non_silent_indices[np.argsort(vertex_variances_filtered)[-top_variance_count:]]
        
        remaining_indices = non_silent_indices[~np.isin(non_silent_indices, top_variance_idx)]
        if len(remaining_indices) > 0:
            random_count = min(n_sample - len(top_variance_idx), len(remaining_indices))
            random_idx = np.random.choice(remaining_indices, random_count, replace=False)
            selected_indices = np.concatenate([top_variance_idx, random_idx])
        else:
            selected_indices = top_variance_idx
    else:
        selected_indices = non_silent_indices
    
    selected_indices = np.sort(selected_indices)
    print(f"Selected {len(selected_indices)} informative vertices for detailed analysis")
    
    return selected_indices, silent_percentage, vertex_variances

def compute_predictions_and_residuals(design_matrix, betas, Y_actual, selected_indices):
    """Compute GLM predictions and residuals following Nilearn tutorial pattern"""
    print("Computing predictions and residuals for selected vertices...")
    
    n_eval = len(selected_indices)
    print(f"Evaluating {n_eval} selected vertices")
    
    # Get subset of data
    Y_subset = Y_actual[:, selected_indices]
    betas_subset = betas[selected_indices, :]
    
    # Compute predictions using full design matrix
    # Method: Y_predicted = X * beta_full (reconstructed from GLM fit)
    X = design_matrix  # Full design matrix (n_timepoints x n_regressors)
    
    # For each vertex, we need to estimate full model parameters
    Y_pred = np.zeros_like(Y_subset)
    Y_resid = np.zeros_like(Y_subset)
    r2_values = np.zeros(n_eval)
    
    from sklearn.linear_model import LinearRegression
    
    for v in range(n_eval):
        y_true = Y_subset[:, v]
        
        # Refit full GLM to get predictions (including confounds)
        try:
            reg = LinearRegression(fit_intercept=False).fit(X, y_true)
            y_pred = reg.predict(X)
            
            Y_pred[:, v] = y_pred
            Y_resid[:, v] = y_true - y_pred
            
            # Compute R²
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_values[v] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
        except Exception as e:
            print(f"Warning: GLM fit failed for vertex {selected_indices[v]}: {e}")
            Y_pred[:, v] = np.mean(y_true)
            Y_resid[:, v] = y_true - np.mean(y_true)
            r2_values[v] = 0.0
    
    print(f"Mean R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    
    return Y_subset, Y_pred, Y_resid, r2_values, selected_indices

def plot_predictions_vs_actual_glm_style(Y_actual, Y_pred, vertices_to_plot, vertex_indices, output_dir, base_name):
    """Plot predictions vs actual time series for selected vertices (glm_evaluation.py style)"""
    print("Creating predictions vs actual plots (GLM evaluation style)...")
    
    n_vertices = len(vertices_to_plot)
    fig, axes = plt.subplots(n_vertices, 1, figsize=(12, 3*n_vertices))
    if n_vertices == 1:
        axes = [axes]
    
    for i, vertex_plot_idx in enumerate(vertices_to_plot):
        vertex_id = vertex_indices[vertex_plot_idx]
        time_points = np.arange(Y_actual.shape[0])
        
        axes[i].plot(time_points, Y_actual[:, vertex_plot_idx], 'b-', label='Actual', alpha=0.7)
        axes[i].plot(time_points, Y_pred[:, vertex_plot_idx], 'r-', label='Predicted', alpha=0.7)
        axes[i].set_title(f'Vertex {vertex_id}: Actual vs Predicted Time Series')
        axes[i].set_xlabel('Time Points (TRs)')
        axes[i].set_ylabel('Signal')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f"{base_name}_predictions_vs_actual.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions vs actual plot saved: {output_file}")
    return output_file

def plot_residual_analysis_glm_style(Y_resid, vertices_to_plot, vertex_indices, output_dir, base_name):
    """Plot residual analysis (glm_evaluation.py style)"""
    print("Creating residual analysis plots (GLM evaluation style)...")
    
    n_vertices = len(vertices_to_plot)
    fig, axes = plt.subplots(2, n_vertices, figsize=(4*n_vertices, 8))
    if n_vertices == 1:
        axes = axes.reshape(-1, 1)
    
    for i, vertex_plot_idx in enumerate(vertices_to_plot):
        vertex_id = vertex_indices[vertex_plot_idx]
        
        # Residual time series
        time_points = np.arange(Y_resid.shape[0])
        axes[0, i].plot(time_points, Y_resid[:, vertex_plot_idx], 'g-', alpha=0.7)
        axes[0, i].set_title(f'Vertex {vertex_id}: Residuals')
        axes[0, i].set_xlabel('Time Points (TRs)')
        axes[0, i].set_ylabel('Residual')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Residual distribution
        residuals_vertex = Y_resid[:, vertex_plot_idx]
        axes[1, i].hist(residuals_vertex, bins=30, alpha=0.7, color='green', density=True)
        axes[1, i].set_title(f'Vertex {vertex_id}: Residual Distribution')
        axes[1, i].set_xlabel('Residual Value')
        axes[1, i].set_ylabel('Density')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add normal distribution overlay
        mean_res = np.mean(residuals_vertex)
        std_res = np.std(residuals_vertex)
        if std_res > 0:
            x_norm = np.linspace(residuals_vertex.min(), residuals_vertex.max(), 100)
            y_norm = stats.norm.pdf(x_norm, mean_res, std_res)
            axes[1, i].plot(x_norm, y_norm, 'r--', label=f'Normal fit (μ={mean_res:.3f}, σ={std_res:.3f})')
            axes[1, i].legend()
    
    plt.tight_layout()
    output_file = output_dir / f"{base_name}_residual_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residual analysis plot saved: {output_file}")
    return output_file

def plot_r2_distribution_glm_style(r2_values, output_dir, base_name):
    """Plot R-squared distribution across vertices (glm_evaluation.py style)"""
    print("Creating R-squared distribution plots (GLM evaluation style)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(r2_values, bins=50, alpha=0.7, color='blue', density=True)
    axes[0].axvline(np.mean(r2_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(r2_values):.3f}')
    axes[0].axvline(np.median(r2_values), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(r2_values):.3f}')
    axes[0].set_xlabel('R-squared')
    axes[0].set_ylabel('Density')
    axes[0].set_title('R-squared Distribution Across Vertices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    box_plot = axes[1].boxplot(r2_values, vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    axes[1].set_ylabel('R-squared')
    axes[1].set_title('R-squared Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = f"""Statistics:
Mean: {np.mean(r2_values):.4f}
Median: {np.median(r2_values):.4f}
Std: {np.std(r2_values):.4f}
Min: {np.min(r2_values):.4f}
Max: {np.max(r2_values):.4f}
R² > 0.1: {np.mean(r2_values > 0.1)*100:.1f}%
R² > 0.5: {np.mean(r2_values > 0.5)*100:.1f}%"""
    
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                verticalalignment='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / f"{base_name}_r2_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"R-squared distribution plot saved: {output_file}")
    return output_file

def plot_predictions_and_residuals_nilearn_style(Y_actual, Y_pred, Y_resid, r2_values, vertex_indices, 
                                                 base_name, output_dir, silent_percentage, vertex_variances, n_vertices_plot=6):
    """Create predictions and residuals plots following Nilearn tutorial style"""
    print("Creating Nilearn-style predictions and residuals plots...")
    
    # Select vertices to plot (highest R² values)
    top_vertices_idx = np.argsort(r2_values)[-n_vertices_plot:]
    
    # Create figure with subplots similar to Nilearn tutorial
    fig = plt.figure(figsize=(20, 14))
    
    # Main title with silent vertices info
    fig.suptitle(f'GLM Predictions and Residuals Analysis: {base_name}\nSilent Vertices: {silent_percentage:.1f}%', 
                fontsize=16, y=0.96)
    
    # Top section: Time series plots (2 rows x n_vertices_plot cols)
    gs1 = fig.add_gridspec(2, n_vertices_plot, height_ratios=[1, 1], 
                          top=0.85, bottom=0.55, left=0.05, right=0.95, 
                          hspace=0.3, wspace=0.3)
    
    time_points = np.arange(Y_actual.shape[0])
    
    for i, vertex_idx in enumerate(top_vertices_idx):
        # Real vs Predicted time series
        ax1 = fig.add_subplot(gs1[0, i])
        ax1.plot(time_points, Y_actual[:, vertex_idx], 'b-', label='Actual', alpha=0.8, linewidth=1)
        ax1.plot(time_points, Y_pred[:, vertex_idx], 'r-', label='Predicted', alpha=0.8, linewidth=1)
        ax1.set_title(f'Vertex {vertex_indices[vertex_idx]}\nR² = {r2_values[vertex_idx]:.3f}', 
                     fontsize=10)
        ax1.set_ylabel('Signal')
        if i == 0:
            ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Residuals time series
        ax2 = fig.add_subplot(gs1[1, i])
        ax2.plot(time_points, Y_resid[:, vertex_idx], 'g-', alpha=0.8, linewidth=1)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title(f'Residuals', fontsize=10)
        ax2.set_xlabel('Time (TRs)')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
    
    # Bottom section: Summary plots (2 rows x 3 cols)
    gs2 = fig.add_gridspec(2, 3, top=0.48, bottom=0.05, left=0.05, right=0.95, wspace=0.3, hspace=0.4)
    
    # R² distribution
    ax3 = fig.add_subplot(gs2[0, 0])
    ax3.hist(r2_values, bins=30, alpha=0.7, edgecolor='black', color='skyblue', density=True)
    ax3.axvline(np.mean(r2_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(r2_values):.3f}')
    ax3.axvline(np.median(r2_values), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(r2_values):.3f}')
    ax3.set_xlabel('R² Score')
    ax3.set_ylabel('Density')
    ax3.set_title('R² Distribution Across Vertices')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Predicted vs Actual scatter (sample of points)
    ax4 = fig.add_subplot(gs2[0, 1])
    
    # Subsample for visualization
    n_sample_vertices = min(5, len(top_vertices_idx))
    sample_vertices = top_vertices_idx[-n_sample_vertices:]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_vertices)))
    for i, vertex_idx in enumerate(sample_vertices):
        ax4.scatter(Y_actual[:, vertex_idx], Y_pred[:, vertex_idx], 
                   alpha=0.6, s=20, c=[colors[i]], 
                   label=f'V{vertex_indices[vertex_idx]} (R²={r2_values[vertex_idx]:.2f})')
    
    # Perfect prediction line
    all_actual = Y_actual[:, sample_vertices].flatten()
    all_pred = Y_pred[:, sample_vertices].flatten()
    min_val, max_val = min(all_actual.min(), all_pred.min()), max(all_actual.max(), all_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect fit')
    
    ax4.set_xlabel('Actual Signal')
    ax4.set_ylabel('Predicted Signal')
    ax4.set_title('Predicted vs Actual')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Residuals distribution
    ax5 = fig.add_subplot(gs2[0, 2])
    
    # Add variance analysis plots
    # Vertex variance distribution  
    ax6 = fig.add_subplot(gs2[1, 0])
    selected_variances = vertex_variances[vertex_indices]
    ax6.hist(np.log10(selected_variances + 1e-10), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax6.axvline(np.log10(np.mean(selected_variances)), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(selected_variances):.2e}')
    ax6.set_xlabel('log10(Vertex Variance)')
    ax6.set_ylabel('Count')
    ax6.set_title('Variance Distribution (Selected Vertices)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Model performance summary
    ax7 = fig.add_subplot(gs2[1, 1])
    performance_metrics = {
        'Mean R²': np.mean(r2_values),
        'Median R²': np.median(r2_values),
        'R² > 0.1': np.mean(r2_values > 0.1) * 100,
        'R² > 0.5': np.mean(r2_values > 0.5) * 100,
    }
    metrics_names = list(performance_metrics.keys())
    metrics_values = list(performance_metrics.values())
    bars = ax7.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax7.set_ylabel('Value')
    ax7.set_title('Model Performance Summary')
    ax7.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Residuals vs fitted values
    ax8 = fig.add_subplot(gs2[1, 2])
    sample_vertices = top_vertices_idx[-3:]  # Use top 3 vertices
    colors = plt.cm.Set1(np.linspace(0, 1, len(sample_vertices)))
    
    for i, vertex_idx in enumerate(sample_vertices):
        fitted_values = Y_pred[:, vertex_idx]
        residuals = Y_resid[:, vertex_idx]
        ax8.scatter(fitted_values, residuals, alpha=0.6, s=20, c=[colors[i]], 
                   label=f'V{vertex_indices[vertex_idx]} (R²={r2_values[vertex_idx]:.2f})')
    
    ax8.axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=1)
    ax8.set_xlabel('Fitted Values')
    ax8.set_ylabel('Residuals')
    ax8.set_title('Residuals vs Fitted Values')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    residuals_flat = Y_resid.flatten()
    ax5.hist(residuals_flat, bins=50, alpha=0.7, edgecolor='black', color='lightcoral', density=True)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax5.axvline(np.mean(residuals_flat), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(residuals_flat):.4f}')
    ax5.set_xlabel('Residual Value')
    ax5.set_ylabel('Density')
    ax5.set_title('Residuals Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / f"{base_name}_nilearn_style_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Nilearn-style evaluation plot saved: {plot_path}")
    return plot_path

def compare_with_original(nilearn_data, original_data, output_dir, base_name):
    """Compare Nilearn results with original GLM implementation"""
    print("Comparing with original GLM implementation...")
    
    nilearn_betas = nilearn_data['betas']
    original_betas = original_data['betas']
    
    # Ensure same number of trials
    min_trials = min(nilearn_betas.shape[1], original_betas.shape[1])
    nilearn_betas = nilearn_betas[:, :min_trials]
    original_betas = original_betas[:, :min_trials]
    
    # Compute correlations
    correlations = []
    r2_scores = []
    mae_scores = []
    
    for i in range(min_trials):
        corr = np.corrcoef(nilearn_betas[:, i], original_betas[:, i])[0, 1]
        r2 = r2_score(original_betas[:, i], nilearn_betas[:, i])
        mae = mean_absolute_error(original_betas[:, i], nilearn_betas[:, i])
        
        correlations.append(corr)
        r2_scores.append(r2)
        mae_scores.append(mae)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Nilearn vs Original GLM Comparison: {base_name}', fontsize=14)
    
    # Plot 1: Correlations
    axes[0,0].bar(range(len(correlations)), correlations)
    axes[0,0].set_xlabel('Trial Index')
    axes[0,0].set_ylabel('Pearson Correlation')
    axes[0,0].set_title(f'Beta Correlations (Mean: {np.mean(correlations):.3f})')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: R² scores
    axes[0,1].bar(range(len(r2_scores)), r2_scores)
    axes[0,1].set_xlabel('Trial Index')
    axes[0,1].set_ylabel('R²')
    axes[0,1].set_title(f'R² Scores (Mean: {np.mean(r2_scores):.3f})')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot (sample trial)
    sample_trial = 0
    axes[1,0].scatter(original_betas[:, sample_trial], nilearn_betas[:, sample_trial], 
                     alpha=0.5, s=1)
    min_val = min(original_betas[:, sample_trial].min(), nilearn_betas[:, sample_trial].min())
    max_val = max(original_betas[:, sample_trial].max(), nilearn_betas[:, sample_trial].max())
    axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect agreement')
    axes[1,0].set_xlabel('Original GLM Betas')
    axes[1,0].set_ylabel('Nilearn GLM Betas')
    axes[1,0].set_title(f'Beta Agreement (Trial {sample_trial})')
    axes[1,0].legend()
    
    # Plot 4: MAE scores
    axes[1,1].bar(range(len(mae_scores)), mae_scores)
    axes[1,1].set_xlabel('Trial Index')
    axes[1,1].set_ylabel('Mean Absolute Error')
    axes[1,1].set_title(f'MAE Scores (Mean: {np.mean(mae_scores):.3f})')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = output_dir / f"{base_name}_nilearn_vs_original.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save comparison statistics
    comparison_stats = pd.DataFrame({
        'trial_index': range(min_trials),
        'correlation': correlations,
        'r2_score': r2_scores,
        'mae': mae_scores
    })
    
    comparison_stats_path = output_dir / f"{base_name}_comparison_stats.csv"
    comparison_stats.to_csv(comparison_stats_path, index=False)
    
    print(f"Comparison analysis saved to: {comparison_path}")
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Mean R²: {np.mean(r2_scores):.4f}")
    
    return {
        'mean_correlation': np.mean(correlations),
        'mean_r2': np.mean(r2_scores),
        'mean_mae': np.mean(mae_scores),
        'correlations': correlations,
        'r2_scores': r2_scores,
        'mae_scores': mae_scores
    }

def create_comprehensive_report(nilearn_data, beta_stats, comparison_results, silent_percentage, 
                               vertex_stats, output_dir, base_name):
    """Create comprehensive evaluation report"""
    print("Creating comprehensive evaluation report...")
    
    report_path = output_dir / f"{base_name}_comprehensive_evaluation.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Comprehensive Nilearn GLM Evaluation: {base_name}\n\n")
        
        # Model summary
        f.write("## Model Summary\n\n")
        f.write(f"- **Subject**: {base_name.split('_')[0]}\n")
        f.write(f"- **Session**: {base_name.split('_')[1] if len(base_name.split('_')) > 1 else 'N/A'}\n")
        f.write(f"- **Task**: {base_name.split('_')[2].replace('task-', '') if len(base_name.split('_')) > 2 else 'N/A'}\n")
        f.write(f"- **Number of vertices**: {nilearn_data['n_vertices']:,}\n")
        f.write(f"- **Number of trials**: {nilearn_data['n_trials']}\n")
        f.write(f"- **TR**: {nilearn_data['tr']:.2f}s\n")
        f.write(f"- **Noise model**: {nilearn_data['noise_model']}\n\n")
        
        # Vertex analysis
        f.write("## Vertex Analysis\n\n")
        f.write(f"- **Silent vertices percentage**: {silent_percentage:.2f}%\n")
        f.write(f"- **Informative vertices**: {100-silent_percentage:.2f}%\n")
        f.write(f"- **Mean vertex variance**: {vertex_stats['mean_variance']:.2e}\n")
        f.write(f"- **Median vertex variance**: {vertex_stats['median_variance']:.2e}\n")
        f.write(f"- **Vertices analyzed**: {vertex_stats['n_analyzed']:,}\n\n")
        
        # Trial information
        f.write("## Trial Regressors\n\n")
        for i, trial_name in enumerate(nilearn_data['task_regressor_names']):
            f.write(f"{i+1}. {trial_name}\n")
        f.write("\n")
        
        # Beta statistics
        f.write("## Beta Coefficient Statistics\n\n")
        f.write("| Trial | Mean | Std | Median | Min | Max |\n")
        f.write("|-------|------|-----|--------|-----|-----|\n")
        for i, trial_name in enumerate(nilearn_data['task_regressor_names']):
            f.write(f"| {trial_name} | {beta_stats['mean'][i]:.4f} | {beta_stats['std'][i]:.4f} | "
                   f"{beta_stats['median'][i]:.4f} | {beta_stats['min'][i]:.4f} | {beta_stats['max'][i]:.4f} |\n")
        f.write("\n")
        
        # Evaluation metrics from vertex analysis
        f.write("## Performance Metrics (Selected Vertices)\n\n")
        f.write(f"- **Mean R²**: {vertex_stats['mean_r2']:.4f}\n")
        f.write(f"- **Median R²**: {vertex_stats['median_r2']:.4f}\n")
        f.write(f"- **R² Standard Deviation**: {vertex_stats['std_r2']:.4f}\n")
        f.write(f"- **Vertices with R² > 0.1**: {vertex_stats['r2_above_01']:.1f}%\n")
        f.write(f"- **Vertices with R² > 0.5**: {vertex_stats['r2_above_05']:.1f}%\n")
        f.write(f"- **Mean Absolute Residual**: {vertex_stats['mean_abs_residual']:.4f}\n")
        f.write(f"- **Residual Standard Deviation**: {vertex_stats['std_residual']:.4f}\n\n")
        
        # Comparison with original (if available)
        if comparison_results:
            f.write("## Comparison with Original GLM\n\n")
            f.write(f"- **Mean correlation with original**: {comparison_results['mean_correlation']:.4f}\n")
            f.write(f"- **Mean R² vs original**: {comparison_results['mean_r2']:.4f}\n")
            f.write(f"- **Mean absolute error**: {comparison_results['mean_mae']:.4f}\n\n")
            
            f.write("### Per-Trial Comparison\n\n")
            f.write("| Trial | Correlation | R² | MAE |\n")
            f.write("|-------|-------------|-------|-----|\n")
            for i, trial_name in enumerate(nilearn_data['task_regressor_names']):
                if i < len(comparison_results['correlations']):
                    f.write(f"| {trial_name} | {comparison_results['correlations'][i]:.4f} | "
                           f"{comparison_results['r2_scores'][i]:.4f} | {comparison_results['mae_scores'][i]:.4f} |\n")
            f.write("\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("This evaluation follows the official Nilearn tutorial procedures:\n")
        f.write("1. **Vertex Selection**: Only non-silent vertices (variance > threshold) are analyzed\n")
        f.write("2. **Sampling Strategy**: Mix of high-variance and randomly selected vertices\n")
        f.write("3. **Model Evaluation**: GLM refitting for predictions and residuals analysis\n")
        f.write("4. **Visualization**: Time series plots and statistical summaries\n\n")
        
        # Files generated
        f.write("## Generated Files\n\n")
        f.write("### Comprehensive Analysis\n")
        f.write(f"- Nilearn-style evaluation plot: `{base_name}_nilearn_style_evaluation.png`\n")
        f.write(f"- Beta statistics: `{base_name}_beta_statistics.csv`\n")
        f.write(f"- Vertex analysis: `{base_name}_vertex_analysis.csv`\n")
        f.write("\n### GLM Evaluation Style Plots\n")
        f.write(f"- Predictions vs actual: `{base_name}_predictions_vs_actual.png`\n")
        f.write(f"- Residual analysis: `{base_name}_residual_analysis.png`\n")
        f.write(f"- R-squared distribution: `{base_name}_r2_distribution.png`\n")
        if comparison_results:
            f.write("\n### Comparison Analysis\n")
            f.write(f"- Comparison plot: `{base_name}_nilearn_vs_original.png`\n")
            f.write(f"- Comparison statistics: `{base_name}_comparison_stats.csv`\n")
        f.write(f"\n### Report\n")
        f.write(f"- This report: `{base_name}_comprehensive_evaluation.md`\n")
    
    print(f"Comprehensive report saved to: {report_path}")
    return report_path

def main():
    args = get_args()
    
    # Setup paths
    base_name = f"{args.subj}_{args.ses}_task-{args.task}_{args.run}"
    
    # Nilearn GLM results
    nilearn_glm_file = Path(args.glm_root) / args.subj / args.ses / "func" / f"{base_name}_nilearn_betas.h5"
    
    # Original GLM results (for comparison)
    original_glm_root = str(Path(args.glm_root).parent / "trial_level_betas")
    original_glm_file = Path(original_glm_root) / args.subj / args.ses / "func" / f"{base_name}_betas.h5"
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating: {base_name}")
    print(f"Nilearn GLM file: {nilearn_glm_file}")
    print(f"Original GLM file: {original_glm_file}")
    
    # Check if Nilearn results exist
    if not nilearn_glm_file.exists():
        print(f"❌ Nilearn GLM results not found: {nilearn_glm_file}")
        return
    
    # Load Nilearn GLM results
    nilearn_data = load_nilearn_glm_results(nilearn_glm_file)
    
    # Load fMRI data for evaluation
    print("\nLoading fMRI data for predictions and residuals analysis...")
    
    # Handle different run number formats
    run_num = args.run.split('-')[1]  # This gives us '01' from 'run-01'
    run_num_short = str(int(run_num))  # This gives us '1' from '01'
    
    # Try different fMRI file formats
    fmri_candidates = [
        f"{args.fmri_root}/{args.subj}/{args.ses}/func/{args.subj}_{args.ses}_task-{args.task}_run-{run_num}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        f"{args.fmri_root}/{args.subj}/{args.ses}/func/{args.subj}_{args.ses}_task-{args.task}_run-{run_num_short}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        f"{args.fmri_root}/{args.subj}/{args.ses}/func/{args.subj}_{args.ses}_task-{args.task}_run-{run_num}_space-T1w_desc-preproc_bold.nii.gz",
        f"{args.fmri_root}/{args.subj}/{args.ses}/func/{args.subj}_{args.ses}_task-{args.task}_run-{run_num_short}_space-T1w_desc-preproc_bold.nii.gz"
    ]
    
    fmri_file = None
    for candidate in fmri_candidates:
        if Path(candidate).exists():
            fmri_file = candidate
            break
    
    if fmri_file is None:
        print(f"❌ fMRI file not found. Tried:")
        for candidate in fmri_candidates:
            print(f"   {candidate}")
        return
    
    print(f"Found fMRI file: {fmri_file}")
    Y_actual = load_fmri_data_for_evaluation(fmri_file)
    
    # Select informative vertices (non-silent with variance above threshold)
    selected_indices, silent_percentage, vertex_variances = select_informative_vertices(
        Y_actual, nilearn_data['betas'], n_sample=args.n_vertices_sample, 
        variance_threshold=args.variance_threshold
    )
    
    # Compute predictions and residuals following Nilearn tutorial style
    Y_subset, Y_pred, Y_resid, r2_values, vertex_indices = compute_predictions_and_residuals(
        nilearn_data['design_matrix'], nilearn_data['betas'], Y_actual, selected_indices
    )
    
    # Create Nilearn-style plots
    plot_path = plot_predictions_and_residuals_nilearn_style(
        Y_subset, Y_pred, Y_resid, r2_values, vertex_indices, base_name, output_dir,
        silent_percentage, vertex_variances
    )
    
    # Create GLM evaluation style plots (similar to original glm_evaluation.py)
    print("\nCreating GLM evaluation style figures...")
    
    # Select top vertices for detailed plots (highest R² values)
    n_vertices_detailed = min(6, len(vertex_indices))
    top_vertices_idx = np.argsort(r2_values)[-n_vertices_detailed:]
    
    # 1. Predictions vs Actual time series plots
    pred_plot_path = plot_predictions_vs_actual_glm_style(
        Y_subset, Y_pred, top_vertices_idx, vertex_indices, output_dir, base_name
    )
    
    # 2. Residual analysis plots 
    resid_plot_path = plot_residual_analysis_glm_style(
        Y_resid, top_vertices_idx, vertex_indices, output_dir, base_name
    )
    
    # 3. R-squared distribution plots
    r2_plot_path = plot_r2_distribution_glm_style(
        r2_values, output_dir, base_name
    )
    
    # Load original GLM results for comparison
    comparison_results = None
    if args.compare_original:
        original_data = load_original_glm_results(original_glm_file)
        if original_data:
            comparison_results = compare_with_original(nilearn_data, original_data, output_dir, base_name)
    
    # Compute beta statistics for report
    betas = nilearn_data['betas']
    beta_stats = {
        'mean': np.mean(betas, axis=0),
        'std': np.std(betas, axis=0),
        'median': np.median(betas, axis=0),
        'min': np.min(betas, axis=0),
        'max': np.max(betas, axis=0)
    }
    
    # Compute vertex statistics for report
    vertex_stats = {
        'n_analyzed': len(vertex_indices),
        'mean_variance': np.mean(vertex_variances[vertex_indices]),
        'median_variance': np.median(vertex_variances[vertex_indices]),
        'mean_r2': np.mean(r2_values),
        'median_r2': np.median(r2_values),
        'std_r2': np.std(r2_values),
        'r2_above_01': np.mean(r2_values > 0.1) * 100,
        'r2_above_05': np.mean(r2_values > 0.5) * 100,
        'mean_abs_residual': np.mean(np.abs(Y_resid)),
        'std_residual': np.std(Y_resid)
    }
    
    # Save vertex analysis data
    vertex_analysis_df = pd.DataFrame({
        'vertex_index': vertex_indices,
        'variance': vertex_variances[vertex_indices],
        'r2_score': r2_values,
        'mean_residual': np.mean(Y_resid, axis=0),
        'std_residual': np.std(Y_resid, axis=0)
    })
    
    vertex_analysis_path = output_dir / f"{base_name}_vertex_analysis.csv"
    vertex_analysis_df.to_csv(vertex_analysis_path, index=False)
    print(f"Vertex analysis saved to: {vertex_analysis_path}")
    
    # Create comprehensive report
    report_path = create_comprehensive_report(nilearn_data, beta_stats, comparison_results, 
                                            silent_percentage, vertex_stats, output_dir, base_name)
    
    print("\n[✅] Nilearn GLM evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"- Silent vertices: {silent_percentage:.1f}%")
    print(f"- Mean R² (selected vertices): {vertex_stats['mean_r2']:.4f}")
    print(f"- Vertices with good fit (R² > 0.1): {vertex_stats['r2_above_01']:.1f}%")
    if comparison_results:
        print(f"- Correlation with original GLM: {comparison_results['mean_correlation']:.4f}")

if __name__ == "__main__":
    main()