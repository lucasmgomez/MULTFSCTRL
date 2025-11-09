#!/usr/bin/env python3
"""
Vertex-Level Data Analysis

This script analyzes the fMRI data at the vertex level to understand:
1. How many vertices have zero variance vs actual signal
2. Distribution of signal across the cortical surface
3. Whether the data quality is reasonable at vertex resolution

Usage:
python vertex_level_analysis.py \
  --subj sub-01 \
  --task ctxdm \
  --ses ses-001 \
  --run run-1
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Analyze data at vertex level")
    parser.add_argument("--subj", default="sub-01", help="Subject ID")
    parser.add_argument("--task", default="ctxdm", help="Task name")
    parser.add_argument("--ses", default="ses-001", help="Session")
    parser.add_argument("--run", default="run-1", help="Run")
    parser.add_argument("--tr", type=float, default=1.49, help="TR in seconds")
    parser.add_argument("--tmask", type=int, default=1, help="Frames dropped at run start")
    parser.add_argument("--fmri_root",
                       default="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled", 
                       help="Root directory containing fMRI data")
    parser.add_argument("--output_dir",
                       default="/project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check",
                       help="Output directory for plots")
    parser.add_argument("--n_sample_vertices", type=int, default=20, 
                       help="Number of sample vertices to plot")
    return parser.parse_args()

def analyze_vertex_data(data):
    """Comprehensive analysis of vertex-level data"""
    T, V = data.shape
    print(f"=== VERTEX-LEVEL ANALYSIS ===")
    print(f"Time points: {T}")
    print(f"Total vertices: {V:,}")
    
    # Compute vertex statistics
    vertex_means = np.mean(data, axis=0)
    vertex_stds = np.std(data, axis=0)
    
    # Categorize vertices
    zero_var_vertices = np.sum(vertex_stds == 0)
    nonzero_var_vertices = np.sum(vertex_stds > 0)
    zero_mean_vertices = np.sum(vertex_means == 0)
    
    print(f"\n=== VERTEX STATISTICS ===")
    print(f"Zero variance vertices: {zero_var_vertices:,} ({100*zero_var_vertices/V:.1f}%)")
    print(f"Non-zero variance vertices: {nonzero_var_vertices:,} ({100*nonzero_var_vertices/V:.1f}%)")
    print(f"Zero mean vertices: {zero_mean_vertices:,} ({100*zero_mean_vertices/V:.1f}%)")
    
    # Signal statistics for non-zero vertices
    if nonzero_var_vertices > 0:
        nonzero_mask = vertex_stds > 0
        nonzero_means = vertex_means[nonzero_mask]
        nonzero_stds = vertex_stds[nonzero_mask]
        
        print(f"\n=== NON-ZERO VERTEX SIGNAL STATISTICS ===")
        print(f"Mean signal range: [{np.min(nonzero_means):.1f}, {np.max(nonzero_means):.1f}]")
        print(f"Mean signal average: {np.mean(nonzero_means):.1f}")
        print(f"Signal variability range: [{np.min(nonzero_stds):.1f}, {np.max(nonzero_stds):.1f}]")
        print(f"Signal variability average: {np.mean(nonzero_stds):.1f}")
        
        # Check for anatomically reasonable signal values
        reasonable_signal = np.sum((nonzero_means > 100) & (nonzero_means < 50000))
        print(f"Vertices with reasonable BOLD signal (100-50000): {reasonable_signal:,} ({100*reasonable_signal/len(nonzero_means):.1f}% of non-zero)")
        
        return nonzero_mask, nonzero_means, nonzero_stds
    else:
        return None, None, None

def plot_vertex_statistics(vertex_means, vertex_stds, nonzero_mask, output_dir, subj, task, ses, run):
    """Plot vertex-level statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Vertex mean distribution (all vertices)
    axes[0, 0].hist(vertex_means, bins=100, alpha=0.7, color='blue', density=True)
    axes[0, 0].set_xlabel('Vertex Mean Signal')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distribution of Vertex Means (All Vertices)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Vertex std distribution (all vertices)
    axes[0, 1].hist(vertex_stds, bins=100, alpha=0.7, color='red', density=True)
    axes[0, 1].set_xlabel('Vertex Signal Std')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Vertex Std (All Vertices)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Zero vs non-zero vertices pie chart
    zero_vertices = np.sum(~nonzero_mask)
    nonzero_vertices = np.sum(nonzero_mask)
    
    axes[0, 2].pie([zero_vertices, nonzero_vertices], 
                   labels=[f'Zero variance\n{zero_vertices:,}', f'Non-zero variance\n{nonzero_vertices:,}'],
                   colors=['lightcoral', 'lightblue'],
                   autopct='%1.1f%%')
    axes[0, 2].set_title('Vertex Variance Distribution')
    
    # 4. Non-zero vertex means distribution
    if np.sum(nonzero_mask) > 0:
        nonzero_means = vertex_means[nonzero_mask]
        axes[1, 0].hist(nonzero_means, bins=50, alpha=0.7, color='green', density=True)
        axes[1, 0].set_xlabel('Vertex Mean Signal')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Non-Zero Vertex Means')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Non-zero vertex stds distribution
        nonzero_stds = vertex_stds[nonzero_mask]
        axes[1, 1].hist(nonzero_stds, bins=50, alpha=0.7, color='orange', density=True)
        axes[1, 1].set_xlabel('Vertex Signal Std')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Non-Zero Vertex Std')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Scatter plot: mean vs std for non-zero vertices
        # Sample for plotting if too many points
        n_sample = min(10000, len(nonzero_means))
        sample_idx = np.random.choice(len(nonzero_means), n_sample, replace=False)
        
        axes[1, 2].scatter(nonzero_means[sample_idx], nonzero_stds[sample_idx], 
                          alpha=0.5, s=1, color='purple')
        axes[1, 2].set_xlabel('Vertex Mean Signal')
        axes[1, 2].set_ylabel('Vertex Signal Std')
        axes[1, 2].set_title(f'Mean vs Std (Sample of {n_sample:,} vertices)')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No non-zero vertices', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'No non-zero vertices', ha='center', va='center')
        axes[1, 2].text(0.5, 0.5, 'No non-zero vertices', ha='center', va='center')
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_vertex_statistics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_sample_vertex_timeseries(data, nonzero_mask, n_sample, output_dir, subj, task, ses, run, tr):
    """Plot sample time series from vertices"""
    T, V = data.shape
    time_points = np.arange(T) * tr
    
    # Get non-zero vertices
    nonzero_indices = np.where(nonzero_mask)[0]
    
    if len(nonzero_indices) == 0:
        print("No non-zero vertices to plot!")
        return None
    
    # Sample vertices to plot
    n_to_plot = min(n_sample, len(nonzero_indices))
    if len(nonzero_indices) > n_to_plot:
        # Get diverse sample: some high variance, some random
        vertex_stds = np.std(data, axis=0)
        nonzero_stds = vertex_stds[nonzero_indices]
        
        # Top variance vertices
        n_top = n_to_plot // 2
        top_std_idx = np.argsort(nonzero_stds)[-n_top:]
        top_vertices = nonzero_indices[top_std_idx]
        
        # Random vertices from remaining
        n_random = n_to_plot - n_top
        remaining_idx = nonzero_indices[~np.isin(nonzero_indices, top_vertices)]
        if len(remaining_idx) >= n_random:
            random_vertices = np.random.choice(remaining_idx, n_random, replace=False)
        else:
            random_vertices = remaining_idx
        
        vertices_to_plot = np.concatenate([top_vertices, random_vertices])
    else:
        vertices_to_plot = nonzero_indices[:n_to_plot]
    
    # Plot
    n_cols = min(5, len(vertices_to_plot))
    n_rows = (len(vertices_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, vertex_idx in enumerate(vertices_to_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else (axes[col] if n_cols > 1 else axes[i])
        
        ts = data[:, vertex_idx]
        ax.plot(time_points, ts, 'b-', alpha=0.7, linewidth=0.8)
        ax.set_title(f'Vertex {vertex_idx}\n'
                    f'Mean: {np.mean(ts):.0f}, Std: {np.std(ts):.1f}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Signal')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    total_subplots = n_rows * n_cols
    for i in range(len(vertices_to_plot), total_subplots):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else (axes[col] if n_cols > 1 else axes[i])
        ax.remove()
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_sample_vertex_timeseries.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_vertex_analysis_report(data, nonzero_mask, vertex_means, vertex_stds, 
                                output_dir, subj, task, ses, run, fmri_file):
    """Create comprehensive vertex analysis report"""
    T, V = data.shape
    
    report = []
    report.append(f"# Vertex-Level Data Analysis Report")
    report.append(f"## File: {fmri_file}")
    report.append(f"## Subject: {subj}, Task: {task}, Session: {ses}, Run: {run}")
    report.append("")
    
    # Data dimensions
    report.append("## Data Dimensions")
    report.append(f"- Time points: {T}")
    report.append(f"- Total vertices: {V:,}")
    report.append("")
    
    # Vertex classification
    zero_var = np.sum(~nonzero_mask)
    nonzero_var = np.sum(nonzero_mask)
    zero_mean = np.sum(vertex_means == 0)
    
    report.append("## Vertex Classification")
    report.append(f"- Zero variance vertices: {zero_var:,} ({100*zero_var/V:.1f}%)")
    report.append(f"- Non-zero variance vertices: {nonzero_var:,} ({100*nonzero_var/V:.1f}%)")
    report.append(f"- Zero mean vertices: {zero_mean:,} ({100*zero_mean/V:.1f}%)")
    report.append("")
    
    # Signal quality assessment
    report.append("## Signal Quality Assessment")
    if nonzero_var > 0:
        nonzero_means = vertex_means[nonzero_mask]
        nonzero_stds = vertex_stds[nonzero_mask]
        
        # BOLD signal ranges (typical: 100-50000 for preprocessed data)
        reasonable_signal = np.sum((nonzero_means > 100) & (nonzero_means < 50000))
        low_signal = np.sum(nonzero_means <= 100)
        high_signal = np.sum(nonzero_means >= 50000)
        
        report.append(f"### Signal Amplitude")
        report.append(f"- Vertices with reasonable BOLD signal (100-50000): {reasonable_signal:,} ({100*reasonable_signal/nonzero_var:.1f}%)")
        report.append(f"- Vertices with low signal (≤100): {low_signal:,} ({100*low_signal/nonzero_var:.1f}%)")
        report.append(f"- Vertices with high signal (≥50000): {high_signal:,} ({100*high_signal/nonzero_var:.1f}%)")
        report.append("")
        
        report.append(f"### Signal Statistics (Non-Zero Vertices)")
        report.append(f"- Mean signal range: [{np.min(nonzero_means):.1f}, {np.max(nonzero_means):.1f}]")
        report.append(f"- Mean signal average: {np.mean(nonzero_means):.1f}")
        report.append(f"- Signal variability range: [{np.min(nonzero_stds):.1f}, {np.max(nonzero_stds):.1f}]")
        report.append(f"- Signal variability average: {np.mean(nonzero_stds):.1f}")
        report.append("")
        
        # Temporal variability assessment
        good_variability = np.sum((nonzero_stds > 10) & (nonzero_stds < 1000))
        report.append(f"### Temporal Variability")
        report.append(f"- Vertices with good temporal variability (10-1000): {good_variability:,} ({100*good_variability/nonzero_var:.1f}%)")
        report.append("")
    
    # Overall data quality verdict
    report.append("## Data Quality Verdict")
    
    quality_issues = []
    if zero_var/V > 0.9:
        quality_issues.append("- **CRITICAL**: >90% of vertices have zero variance")
    elif zero_var/V > 0.7:
        quality_issues.append("- **WARNING**: >70% of vertices have zero variance")
    
    if nonzero_var > 0:
        nonzero_means = vertex_means[nonzero_mask]
        reasonable_signal = np.sum((nonzero_means > 100) & (nonzero_means < 50000))
        if reasonable_signal/nonzero_var < 0.5:
            quality_issues.append("- **WARNING**: <50% of active vertices have reasonable BOLD signal amplitudes")
        
        good_variability = np.sum((nonzero_stds > 10) & (nonzero_stds < 1000))
        if good_variability/nonzero_var < 0.5:
            quality_issues.append("- **WARNING**: <50% of active vertices have reasonable temporal variability")
    
    if len(quality_issues) == 0:
        if nonzero_var/V > 0.1:  # >10% active vertices
            report.append("- **GOOD**: Data appears to have reasonable quality for fMRI analysis")
        else:
            report.append("- **QUESTIONABLE**: Very few active vertices, but those present seem reasonable")
    else:
        report.extend(quality_issues)
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    if zero_var/V > 0.8:
        report.append("- Consider using only vertices with non-zero variance for analysis")
        report.append("- High proportion of zero vertices is normal for cortical surface data")
    if nonzero_var > 500:
        report.append("- Sufficient vertices with signal for meaningful analysis")
    else:
        report.append("- Limited number of active vertices - check preprocessing and masking")
    
    # Save report
    report_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_vertex_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    return report_file

def main():
    args = get_args()
    
    # Construct paths
    fmri_root = Path(args.fmri_root) / args.subj / args.ses
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # fMRI data file
    fmri_file = fmri_root / f"{args.subj}_{args.ses}_task-{args.task}_{args.run}_space-Glasser64k_bold.dtseries.nii"
    
    if not fmri_file.exists():
        raise FileNotFoundError(f"fMRI file not found: {fmri_file}")
    
    print(f"Loading fMRI data from: {fmri_file}")
    
    # Load and preprocess data
    Y = nib.load(str(fmri_file)).get_fdata(dtype=np.float32)
    print(f"Raw data shape: {Y.shape}")
    
    # Apply tmask
    if args.tmask > 0:
        T = Y.shape[0]
        keep = np.ones((T,), dtype=bool)
        keep[:args.tmask] = False
        Y = Y[keep, :]
        print(f"After tmask shape: {Y.shape}")
    
    # Analyze vertex-level data
    vertex_means = np.mean(Y, axis=0)
    vertex_stds = np.std(Y, axis=0)
    nonzero_mask = vertex_stds > 0
    
    nonzero_mask, nonzero_means, nonzero_stds = analyze_vertex_data(Y)
    
    print(f"\n=== Creating visualizations ===")
    
    # Create plots
    stats_plot = plot_vertex_statistics(vertex_means, vertex_stds, nonzero_mask, 
                                       output_dir, args.subj, args.task, args.ses, args.run)
    print(f"Vertex statistics plot saved: {stats_plot}")
    
    ts_plot = plot_sample_vertex_timeseries(Y, nonzero_mask, args.n_sample_vertices, 
                                          output_dir, args.subj, args.task, args.ses, args.run, args.tr)
    if ts_plot:
        print(f"Sample time series plot saved: {ts_plot}")
    
    # Create comprehensive report
    report_file = create_vertex_analysis_report(Y, nonzero_mask, vertex_means, vertex_stds,
                                              output_dir, args.subj, args.task, args.ses, args.run, fmri_file)
    print(f"Vertex analysis report saved: {report_file}")

if __name__ == "__main__":
    main()