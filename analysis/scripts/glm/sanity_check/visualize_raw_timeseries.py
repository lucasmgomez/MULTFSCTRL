#!/usr/bin/env python3
"""
Visualize Raw fMRI Time Series

This script loads and visualizes raw fMRI time series to diagnose potential data loading issues.
It checks for:
1. Whether the data contains actual signal variation
2. Data ranges and statistics
3. Potential loading or preprocessing errors

Usage:
python visualize_raw_timeseries.py \
  --subj sub-01 \
  --task ctxdm \
  --ses ses-001 \
  --run run-1 \
  --fmri_root /project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled
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
    parser = argparse.ArgumentParser(description="Visualize raw fMRI time series")
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
    parser.add_argument("--n_parcels", type=int, default=20, 
                       help="Number of parcels to visualize")
    return parser.parse_args()

def load_and_inspect_fmri(fmri_file, tmask=0):
    """Load fMRI data and return basic statistics"""
    print(f"Loading fMRI data from: {fmri_file}")
    
    # Load the data
    img = nib.load(str(fmri_file))
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.get_fdata().dtype}")
    print(f"Image header info:")
    try:
        print(f"  - Dimensions: {img.header.get_data_shape()}")
    except:
        print(f"  - Dimensions: {img.shape}")
    
    # Get the data
    data = img.get_fdata(dtype=np.float32)  # (T x P)
    print(f"Raw data shape: {data.shape}")
    print(f"Raw data dtype: {data.dtype}")
    
    # Basic statistics before tmask
    print(f"\n=== BEFORE tmask (raw data) ===")
    print(f"Data range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    print(f"Data mean: {np.mean(data):.6f}")
    print(f"Data std: {np.std(data):.6f}")
    print(f"Number of NaNs: {np.sum(np.isnan(data))}")
    print(f"Number of Infs: {np.sum(np.isinf(data))}")
    print(f"Number of zeros: {np.sum(data == 0)}")
    
    # Check if all values are the same
    unique_values = np.unique(data)
    print(f"Number of unique values: {len(unique_values)}")
    if len(unique_values) < 10:
        print(f"First 10 unique values: {unique_values[:10]}")
    
    # Apply tmask
    if tmask > 0:
        T = data.shape[0]
        keep = np.ones((T,), dtype=bool)
        keep[:tmask] = False
        data_masked = data[keep, :]
        print(f"\n=== AFTER tmask (dropped first {tmask} timepoints) ===")
        print(f"Masked data shape: {data_masked.shape}")
        print(f"Data range: [{np.min(data_masked):.6f}, {np.max(data_masked):.6f}]")
        print(f"Data mean: {np.mean(data_masked):.6f}")
        print(f"Data std: {np.std(data_masked):.6f}")
        return data_masked
    
    return data

def plot_sample_timeseries(data, n_parcels, output_dir, subj, task, ses, run, tr):
    """Plot sample time series from different parcels"""
    T, P = data.shape
    time_points = np.arange(T) * tr
    
    # Select parcels to plot (random + first few + last few)
    parcel_indices = []
    parcel_indices.extend(range(min(5, P)))  # First 5 parcels
    if P > 10:
        parcel_indices.extend(range(P-5, P))  # Last 5 parcels
    # Add some random parcels
    if P > n_parcels:
        random_parcels = np.random.choice(P, min(n_parcels-10, P-10), replace=False)
        parcel_indices.extend(random_parcels)
    
    parcel_indices = sorted(list(set(parcel_indices)))[:n_parcels]
    
    # Create subplots
    n_cols = min(4, len(parcel_indices))
    n_rows = (len(parcel_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if len(parcel_indices) > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, parcel_idx in enumerate(parcel_indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        ts = data[:, parcel_idx]
        ax.plot(time_points, ts, 'b-', alpha=0.7, linewidth=1)
        ax.set_title(f'Parcel {parcel_idx}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Signal')
        ax.grid(True, alpha=0.3)
        
        # Add statistics in title
        ax.set_title(f'Parcel {parcel_idx}\n'
                    f'Range: [{np.min(ts):.2f}, {np.max(ts):.2f}]\n'
                    f'Mean: {np.mean(ts):.2f}, Std: {np.std(ts):.3f}')
    
    # Remove empty subplots
    for i in range(len(parcel_indices), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.remove()
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_raw_timeseries_sample.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_global_signal_stats(data, output_dir, subj, task, ses, run, tr):
    """Plot global signal statistics"""
    T, P = data.shape
    time_points = np.arange(T) * tr
    
    # Compute global statistics
    mean_signal = np.mean(data, axis=1)  # Mean across parcels at each timepoint
    std_signal = np.std(data, axis=1)    # Std across parcels at each timepoint
    
    # Per-parcel statistics over time
    parcel_means = np.mean(data, axis=0)  # Mean over time for each parcel
    parcel_stds = np.std(data, axis=0)    # Std over time for each parcel
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Global mean signal over time
    axes[0, 0].plot(time_points, mean_signal, 'b-', linewidth=1)
    axes[0, 0].set_title('Global Mean Signal Over Time')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Mean Signal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Global std signal over time
    axes[0, 1].plot(time_points, std_signal, 'r-', linewidth=1)
    axes[0, 1].set_title('Global Signal Variability Over Time')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Signal Std')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution of parcel means
    axes[1, 0].hist(parcel_means, bins=50, alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title('Distribution of Parcel Means')
    axes[1, 0].set_xlabel('Mean Signal')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of parcel standard deviations
    axes[1, 1].hist(parcel_stds, bins=50, alpha=0.7, color='red', density=True)
    axes[1, 1].set_title('Distribution of Parcel Standard Deviations')
    axes[1, 1].set_xlabel('Signal Std')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_global_signal_stats.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_diagnostic_report(data, fmri_file, output_dir, subj, task, ses, run):
    """Create diagnostic report"""
    T, P = data.shape
    
    report = []
    report.append(f"# Raw fMRI Time Series Diagnostic Report")
    report.append(f"## File: {fmri_file}")
    report.append(f"## Subject: {subj}, Task: {task}, Session: {ses}, Run: {run}")
    report.append("")
    
    # Data dimensions
    report.append("## Data Dimensions")
    report.append(f"- Time points: {T}")
    report.append(f"- Parcels: {P}")
    report.append(f"- Total values: {T * P:,}")
    report.append("")
    
    # Data statistics
    report.append("## Data Statistics")
    report.append(f"- Range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    report.append(f"- Mean: {np.mean(data):.6f}")
    report.append(f"- Standard deviation: {np.std(data):.6f}")
    report.append(f"- Median: {np.median(data):.6f}")
    report.append("")
    
    # Data quality checks
    report.append("## Data Quality Checks")
    report.append(f"- Number of NaN values: {np.sum(np.isnan(data)):,}")
    report.append(f"- Number of infinite values: {np.sum(np.isinf(data)):,}")
    report.append(f"- Number of zero values: {np.sum(data == 0):,} ({100*np.sum(data == 0)/(T*P):.2f}%)")
    report.append(f"- Number of unique values: {len(np.unique(data)):,}")
    report.append("")
    
    # Per-timepoint statistics
    mean_per_time = np.mean(data, axis=1)
    std_per_time = np.std(data, axis=1)
    report.append("## Temporal Statistics")
    report.append(f"- Mean signal range over time: [{np.min(mean_per_time):.6f}, {np.max(mean_per_time):.6f}]")
    report.append(f"- Signal variability over time: [{np.min(std_per_time):.6f}, {np.max(std_per_time):.6f}]")
    report.append("")
    
    # Per-parcel statistics
    mean_per_parcel = np.mean(data, axis=0)
    std_per_parcel = np.std(data, axis=0)
    report.append("## Parcel Statistics")
    report.append(f"- Parcel mean range: [{np.min(mean_per_parcel):.6f}, {np.max(mean_per_parcel):.6f}]")
    report.append(f"- Parcel std range: [{np.min(std_per_parcel):.6f}, {np.max(std_per_parcel):.6f}]")
    report.append(f"- Parcels with zero variance: {np.sum(std_per_parcel == 0)}")
    report.append("")
    
    # Potential issues
    report.append("## Potential Issues Detected")
    issues = []
    
    if np.std(data) < 1e-10:
        issues.append("- **CRITICAL**: Data has extremely low variance (likely constant)")
    
    if len(np.unique(data)) < 100:
        issues.append(f"- **WARNING**: Data has very few unique values ({len(np.unique(data))})")
    
    if np.sum(np.isnan(data)) > 0:
        issues.append(f"- **WARNING**: Data contains {np.sum(np.isnan(data))} NaN values")
    
    if np.sum(np.isinf(data)) > 0:
        issues.append(f"- **WARNING**: Data contains {np.sum(np.isinf(data))} infinite values")
    
    if np.sum(std_per_parcel == 0) > P * 0.1:
        issues.append(f"- **WARNING**: {np.sum(std_per_parcel == 0)} parcels ({100*np.sum(std_per_parcel == 0)/P:.1f}%) have zero variance")
    
    if len(issues) == 0:
        report.append("- No major issues detected")
    else:
        report.extend(issues)
    
    report.append("")
    
    # Save report
    report_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_raw_timeseries_diagnostic.md"
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
    
    # Load and inspect data
    data = load_and_inspect_fmri(fmri_file, args.tmask)
    
    print(f"\n=== Creating visualizations ===")
    
    # Create plots
    ts_plot = plot_sample_timeseries(data, args.n_parcels, output_dir, 
                                   args.subj, args.task, args.ses, args.run, args.tr)
    print(f"Sample time series plot saved: {ts_plot}")
    
    stats_plot = plot_global_signal_stats(data, output_dir, 
                                        args.subj, args.task, args.ses, args.run, args.tr)
    print(f"Global signal stats plot saved: {stats_plot}")
    
    # Create diagnostic report
    report_file = create_diagnostic_report(data, fmri_file, output_dir, 
                                         args.subj, args.task, args.ses, args.run)
    print(f"Diagnostic report saved: {report_file}")
    
    print(f"\n=== Summary ===")
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    print(f"Data std: {np.std(data):.6f}")
    print(f"Unique values: {len(np.unique(data)):,}")

if __name__ == "__main__":
    main()