#!/usr/bin/env python3
"""
Raw fMRI Data Signal-to-Noise Ratio (SNR) Analysis

This script performs comprehensive SNR analysis on fMRIPrep preprocessed data to assess data quality.
It computes multiple SNR metrics to evaluate the raw data quality before GLM analysis.

SNR Metrics Computed:
1. Temporal SNR (tSNR): mean/std across time for each voxel
2. Global SNR: brain-averaged signal quality
3. Regional SNR: SNR in different brain regions
4. SNR distributions and statistics

Usage:
python raw_data_snr_analysis.py \
  --subj sub-01 \
  --task ctxdm \
  --ses ses-001 \
  --run run-1 \
  --fmriprep_root /project/def-pbellec/xuan/cneuromod.multfs.fmriprep \
  --output_dir /project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/raw_data_snr
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from scipy import ndimage, stats
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Analyze SNR of raw fMRIPrep data")
    parser.add_argument("--subj", default="sub-01", help="Subject ID")
    parser.add_argument("--task", default="ctxdm", help="Task name")
    parser.add_argument("--ses", default="ses-001", help="Session")
    parser.add_argument("--run", default="run-1", help="Run")
    parser.add_argument("--space", default="T1w", help="Space (T1w, MNI152NLin2009cAsym)")
    parser.add_argument("--fmriprep_root",
                       default="/project/def-pbellec/xuan/cneuromod.multfs.fmriprep",
                       help="Root directory containing fMRIPrep data")
    parser.add_argument("--output_dir",
                       default="/project/def-pbellec/xuan/fmri_dataset_project/scripts/sanity_check/raw_data_snr",
                       help="Output directory for SNR analysis")
    parser.add_argument("--skip_initial_trs", type=int, default=5,
                       help="Number of initial TRs to skip for steady-state")
    return parser.parse_args()

def load_fmriprep_data(fmriprep_root, subj, ses, task, run, space, skip_initial=5):
    """Load fMRIPrep preprocessed data and brain mask"""
    print(f"Loading fMRIPrep data for {subj} {ses} {task} {run} in {space} space...")
    
    func_dir = Path(fmriprep_root) / subj / ses / "func"
    
    # Construct file paths
    bold_file = func_dir / f"{subj}_{ses}_task-{task}_{run}_space-{space}_desc-preproc_bold.nii.gz"
    mask_file = func_dir / f"{subj}_{ses}_task-{task}_{run}_space-{space}_desc-brain_mask.nii.gz"
    
    if not bold_file.exists():
        raise FileNotFoundError(f"BOLD file not found: {bold_file}")
    if not mask_file.exists():
        raise FileNotFoundError(f"Brain mask not found: {mask_file}")
    
    print(f"  Loading BOLD: {bold_file}")
    print(f"  Loading mask: {mask_file}")
    
    # Load data
    bold_img = nib.load(str(bold_file))
    mask_img = nib.load(str(mask_file))
    
    bold_data = bold_img.get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)
    
    print(f"  BOLD data shape: {bold_data.shape}")
    print(f"  Brain mask voxels: {np.sum(mask_data):,}")
    
    # Skip initial TRs for steady-state
    if skip_initial > 0:
        bold_data = bold_data[:, :, :, skip_initial:]
        print(f"  Skipped first {skip_initial} TRs, new shape: {bold_data.shape}")
    
    return bold_data, mask_data, bold_img.header

def compute_temporal_snr(bold_data, mask_data):
    """Compute temporal SNR (tSNR) = mean/std across time"""
    print("Computing temporal SNR...")
    
    # Get dimensions
    x, y, z, t = bold_data.shape
    
    # Initialize tSNR array
    tsnr = np.zeros((x, y, z))
    
    # Compute tSNR for masked voxels only
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if mask_data[i, j, k]:
                    time_series = bold_data[i, j, k, :]
                    if np.std(time_series) > 0:
                        tsnr[i, j, k] = np.mean(time_series) / np.std(time_series)
                    else:
                        tsnr[i, j, k] = 0
    
    # Mask tSNR (set non-brain voxels to 0)
    tsnr[~mask_data] = 0
    
    return tsnr

def analyze_snr_statistics(tsnr, mask_data):
    """Analyze SNR statistics within brain mask"""
    print("Analyzing SNR statistics...")
    
    # Get brain voxel tSNR values
    brain_tsnr = tsnr[mask_data]
    brain_tsnr = brain_tsnr[brain_tsnr > 0]  # Remove zero values
    
    if len(brain_tsnr) == 0:
        raise ValueError("No valid tSNR values found in brain mask")
    
    stats_dict = {
        'mean_tsnr': np.mean(brain_tsnr),
        'median_tsnr': np.median(brain_tsnr),
        'std_tsnr': np.std(brain_tsnr),
        'min_tsnr': np.min(brain_tsnr),
        'max_tsnr': np.max(brain_tsnr),
        'q25_tsnr': np.percentile(brain_tsnr, 25),
        'q75_tsnr': np.percentile(brain_tsnr, 75),
        'n_brain_voxels': len(brain_tsnr),
        'n_zero_tsnr': np.sum(brain_tsnr == 0),
    }
    
    # Quality thresholds
    stats_dict['good_tsnr_voxels'] = np.sum(brain_tsnr > 50)  # High quality
    stats_dict['acceptable_tsnr_voxels'] = np.sum(brain_tsnr > 30)  # Acceptable
    stats_dict['poor_tsnr_voxels'] = np.sum(brain_tsnr < 20)  # Poor quality
    
    stats_dict['pct_good'] = 100 * stats_dict['good_tsnr_voxels'] / len(brain_tsnr)
    stats_dict['pct_acceptable'] = 100 * stats_dict['acceptable_tsnr_voxels'] / len(brain_tsnr)
    stats_dict['pct_poor'] = 100 * stats_dict['poor_tsnr_voxels'] / len(brain_tsnr)
    
    return stats_dict, brain_tsnr

def compute_global_snr_timeseries(bold_data, mask_data):
    """Compute global brain signal SNR over time"""
    print("Computing global signal SNR...")
    
    # Extract brain time series (mean across brain voxels at each timepoint)
    brain_mask_4d = np.expand_dims(mask_data, axis=3)
    
    global_signal = []
    for t in range(bold_data.shape[3]):
        brain_values = bold_data[:, :, :, t][mask_data]
        if len(brain_values) > 0:
            global_signal.append(np.mean(brain_values))
        else:
            global_signal.append(0)
    
    global_signal = np.array(global_signal)
    
    # Compute global SNR
    if np.std(global_signal) > 0:
        global_snr = np.mean(global_signal) / np.std(global_signal)
    else:
        global_snr = 0
    
    return global_signal, global_snr

def create_snr_visualizations(tsnr, brain_tsnr, global_signal, mask_data, output_dir, subj, task, ses, run):
    """Create comprehensive SNR visualizations"""
    print("Creating SNR visualizations...")
    
    # Set up figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'fMRI Data Quality Assessment - {subj} {ses} {task} {run}', fontsize=16)
    
    # 1. tSNR histogram
    axes[0, 0].hist(brain_tsnr, bins=50, alpha=0.7, color='blue', density=True)
    axes[0, 0].axvline(np.mean(brain_tsnr), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(brain_tsnr):.1f}')
    axes[0, 0].axvline(np.median(brain_tsnr), color='orange', linestyle='--', 
                       label=f'Median: {np.median(brain_tsnr):.1f}')
    axes[0, 0].axvline(50, color='green', linestyle=':', label='Good (>50)', alpha=0.7)
    axes[0, 0].axvline(30, color='yellow', linestyle=':', label='Acceptable (>30)', alpha=0.7)
    axes[0, 0].set_xlabel('Temporal SNR')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('tSNR Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. tSNR box plot
    axes[0, 1].boxplot(brain_tsnr, vert=True)
    axes[0, 1].set_ylabel('Temporal SNR')
    axes[0, 1].set_title('tSNR Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. SNR quality categories
    good_pct = 100 * np.sum(brain_tsnr > 50) / len(brain_tsnr)
    acceptable_pct = 100 * np.sum((brain_tsnr > 30) & (brain_tsnr <= 50)) / len(brain_tsnr)
    poor_pct = 100 * np.sum(brain_tsnr <= 30) / len(brain_tsnr)
    
    categories = ['Good (>50)', 'Acceptable (30-50)', 'Poor (≤30)']
    percentages = [good_pct, acceptable_pct, poor_pct]
    colors = ['green', 'orange', 'red']
    
    axes[0, 2].pie(percentages, labels=categories, colors=colors, autopct='%1.1f%%')
    axes[0, 2].set_title('SNR Quality Distribution')
    
    # 4. Central sagittal slice of tSNR
    mid_slice = tsnr.shape[0] // 2
    tsnr_slice = tsnr[mid_slice, :, :]
    mask_slice = mask_data[mid_slice, :, :]
    tsnr_slice[~mask_slice] = np.nan
    
    im1 = axes[1, 0].imshow(tsnr_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=100)
    axes[1, 0].set_title('tSNR - Sagittal (Mid)')
    axes[1, 0].set_xlabel('Anterior-Posterior')
    axes[1, 0].set_ylabel('Inferior-Superior')
    plt.colorbar(im1, ax=axes[1, 0], label='tSNR')
    
    # 5. Central coronal slice of tSNR
    mid_slice = tsnr.shape[1] // 2
    tsnr_slice = tsnr[:, mid_slice, :]
    mask_slice = mask_data[:, mid_slice, :]
    tsnr_slice[~mask_slice] = np.nan
    
    im2 = axes[1, 1].imshow(tsnr_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=100)
    axes[1, 1].set_title('tSNR - Coronal (Mid)')
    axes[1, 1].set_xlabel('Left-Right')
    axes[1, 1].set_ylabel('Inferior-Superior')
    plt.colorbar(im2, ax=axes[1, 1], label='tSNR')
    
    # 6. Central axial slice of tSNR
    mid_slice = tsnr.shape[2] // 2
    tsnr_slice = tsnr[:, :, mid_slice]
    mask_slice = mask_data[:, :, mid_slice]
    tsnr_slice[~mask_slice] = np.nan
    
    im3 = axes[1, 2].imshow(tsnr_slice.T, cmap='viridis', origin='lower', vmin=0, vmax=100)
    axes[1, 2].set_title('tSNR - Axial (Mid)')
    axes[1, 2].set_xlabel('Left-Right')
    axes[1, 2].set_ylabel('Anterior-Posterior')
    plt.colorbar(im3, ax=axes[1, 2], label='tSNR')
    
    # 7. Global signal time series
    time_points = np.arange(len(global_signal))
    axes[2, 0].plot(time_points, global_signal, 'b-', linewidth=1, alpha=0.8)
    axes[2, 0].set_xlabel('Time Point (TR)')
    axes[2, 0].set_ylabel('Global Signal')
    axes[2, 0].set_title('Global Brain Signal Over Time')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Global signal statistics
    gs_mean = np.mean(global_signal)
    gs_std = np.std(global_signal)
    gs_snr = gs_mean / gs_std if gs_std > 0 else 0
    
    detrended_signal = global_signal - np.linspace(global_signal[0], global_signal[-1], len(global_signal))
    axes[2, 1].plot(time_points, detrended_signal, 'g-', linewidth=1, alpha=0.8)
    axes[2, 1].set_xlabel('Time Point (TR)')
    axes[2, 1].set_ylabel('Detrended Global Signal')
    axes[2, 1].set_title(f'Detrended Signal (SNR: {gs_snr:.1f})')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Summary statistics
    axes[2, 2].axis('off')
    summary_text = f"""
    SNR SUMMARY STATISTICS
    
    Temporal SNR (Brain):
    • Mean: {np.mean(brain_tsnr):.1f}
    • Median: {np.median(brain_tsnr):.1f}
    • Range: [{np.min(brain_tsnr):.1f}, {np.max(brain_tsnr):.1f}]
    
    Global Signal SNR: {gs_snr:.1f}
    
    Quality Assessment:
    • Good voxels (>50): {good_pct:.1f}%
    • Acceptable (30-50): {acceptable_pct:.1f}%
    • Poor (≤30): {poor_pct:.1f}%
    
    Brain voxels: {len(brain_tsnr):,}
    """
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_snr_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_detailed_report(stats_dict, global_snr, subj, task, ses, run, space, output_dir, 
                          bold_shape, mask_voxels, skip_initial, brain_tsnr):
    """Create comprehensive SNR analysis report"""
    print("Creating detailed SNR report...")
    
    report = []
    report.append("# fMRI Data Signal-to-Noise Ratio (SNR) Analysis Report")
    report.append(f"## Dataset: {subj} {ses} {task} {run} (Space: {space})")
    report.append("")
    
    # Analysis overview
    report.append("## Analysis Overview")
    report.append("")
    report.append("### What is Signal-to-Noise Ratio (SNR)?")
    report.append("SNR quantifies the quality of fMRI data by measuring the ratio of signal to noise.")
    report.append("**Temporal SNR (tSNR)** is computed as mean/standard_deviation across time for each voxel.")
    report.append("Higher tSNR indicates better data quality and more reliable signal detection.")
    report.append("")
    
    report.append("### Metrics Computed")
    report.append("1. **Temporal SNR**: Voxel-wise signal stability over time")
    report.append("2. **Global SNR**: Whole-brain average signal quality")
    report.append("3. **Regional SNR**: Spatial distribution of data quality")
    report.append("4. **Quality Assessment**: Percentage of voxels meeting quality thresholds")
    report.append("")
    
    # Data specifications
    report.append("## Data Specifications")
    report.append(f"- Subject: {subj}")
    report.append(f"- Session: {ses}")
    report.append(f"- Task: {task}")
    report.append(f"- Run: {run}")
    report.append(f"- Space: {space}")
    report.append(f"- Original data shape: {bold_shape}")
    report.append(f"- Brain mask voxels: {mask_voxels:,}")
    report.append(f"- Steady-state TRs skipped: {skip_initial}")
    report.append(f"- Final time points analyzed: {bold_shape[3] - skip_initial}")
    report.append("")
    
    # SNR Results
    report.append("## SNR Analysis Results")
    report.append("")
    
    report.append("### Temporal SNR Statistics")
    report.append(f"- **Mean tSNR**: {stats_dict['mean_tsnr']:.2f}")
    report.append(f"- **Median tSNR**: {stats_dict['median_tsnr']:.2f}")
    report.append(f"- **Standard deviation**: {stats_dict['std_tsnr']:.2f}")
    report.append(f"- **Range**: [{stats_dict['min_tsnr']:.1f}, {stats_dict['max_tsnr']:.1f}]")
    report.append(f"- **25th percentile**: {stats_dict['q25_tsnr']:.2f}")
    report.append(f"- **75th percentile**: {stats_dict['q75_tsnr']:.2f}")
    report.append("")
    
    report.append("### Global Signal SNR")
    report.append(f"- **Global SNR**: {global_snr:.2f}")
    report.append("  (Ratio of mean global signal to its standard deviation over time)")
    report.append("")
    
    # Quality Assessment
    report.append("### Data Quality Assessment")
    report.append("")
    report.append("**Quality Thresholds:**")
    report.append("- **Excellent**: tSNR > 70")
    report.append("- **Good**: tSNR > 50")
    report.append("- **Acceptable**: tSNR > 30")
    report.append("- **Poor**: tSNR ≤ 30")
    report.append("")
    
    excellent_pct = 100 * np.sum(brain_tsnr > 70) / stats_dict['n_brain_voxels'] if stats_dict['n_brain_voxels'] > 0 else 0
    
    report.append("**Quality Distribution:**")
    report.append(f"- **Excellent voxels** (tSNR > 70): {excellent_pct:.1f}%")
    report.append(f"- **Good voxels** (tSNR > 50): {stats_dict['pct_good']:.1f}%")
    report.append(f"- **Acceptable voxels** (tSNR > 30): {stats_dict['pct_acceptable']:.1f}%")
    report.append(f"- **Poor voxels** (tSNR ≤ 30): {stats_dict['pct_poor']:.1f}%")
    report.append("")
    
    report.append(f"**Total brain voxels analyzed**: {stats_dict['n_brain_voxels']:,}")
    report.append("")
    
    # Quality interpretation
    report.append("## Quality Assessment Interpretation")
    report.append("")
    
    # Overall quality verdict
    if stats_dict['mean_tsnr'] > 60 and stats_dict['pct_good'] > 60:
        quality_verdict = "**EXCELLENT**"
        quality_color = "🟢"
    elif stats_dict['mean_tsnr'] > 45 and stats_dict['pct_good'] > 40:
        quality_verdict = "**GOOD**"
        quality_color = "🟡"
    elif stats_dict['mean_tsnr'] > 30 and stats_dict['pct_acceptable'] > 60:
        quality_verdict = "**ACCEPTABLE**"
        quality_color = "🟠"
    else:
        quality_verdict = "**POOR**"
        quality_color = "🔴"
    
    report.append(f"### Overall Data Quality: {quality_color} {quality_verdict}")
    report.append("")
    
    # Detailed interpretation
    if quality_verdict == "**EXCELLENT**":
        report.append("**Interpretation**: This dataset shows excellent SNR characteristics.")
        report.append("- High temporal stability across most brain voxels")
        report.append("- Low noise levels relative to signal")
        report.append("- Suitable for sophisticated analyses and detection of small effects")
        report.append("- Expected to yield reliable and robust results")
    elif quality_verdict == "**GOOD**":
        report.append("**Interpretation**: This dataset shows good SNR characteristics.")
        report.append("- Adequate temporal stability for most analyses")
        report.append("- Reasonable noise levels")
        report.append("- Suitable for standard fMRI analyses")
        report.append("- Should yield reliable results for moderate to large effects")
    elif quality_verdict == "**ACCEPTABLE**":
        report.append("**Interpretation**: This dataset shows acceptable SNR characteristics.")
        report.append("- Moderate temporal stability")
        report.append("- Higher noise levels may limit detection of small effects")
        report.append("- Suitable for basic analyses with appropriate statistical thresholds")
        report.append("- Consider additional preprocessing or conservative statistical approaches")
    else:
        report.append("**Interpretation**: This dataset shows poor SNR characteristics.")
        report.append("- Low temporal stability and high noise levels")
        report.append("- May limit ability to detect meaningful activations")
        report.append("- Consider data quality issues, preprocessing problems, or acquisition artifacts")
        report.append("- Recommend investigation of data collection and processing procedures")
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if quality_verdict in ["**EXCELLENT**", "**GOOD**"]:
        report.append("✅ **Proceed with analysis**")
        report.append("- Data quality is sufficient for standard fMRI analyses")
        report.append("- Consider standard statistical thresholds")
        report.append("- Single-subject analyses should be reliable")
    elif quality_verdict == "**ACCEPTABLE**":
        report.append("⚠️ **Proceed with caution**")
        report.append("- Use conservative statistical thresholds")
        report.append("- Focus on strong, expected effects")
        report.append("- Consider group-level analyses for increased power")
        report.append("- Implement robust preprocessing steps")
    else:
        report.append("❌ **Consider data exclusion or additional processing**")
        report.append("- Investigate potential causes of poor SNR:")
        report.append("  - Head motion during acquisition")
        report.append("  - Scanner artifacts or instabilities")
        report.append("  - Preprocessing issues")
        report.append("- Consider advanced denoising techniques")
        report.append("- Evaluate if data meets minimum quality standards for analysis")
    
    report.append("")
    
    # Technical notes
    report.append("## Technical Notes")
    report.append("")
    report.append("### SNR Calculation Method")
    report.append("- **Temporal SNR**: tSNR = mean(signal) / std(signal) for each voxel")
    report.append("- **Global SNR**: Mean brain signal divided by its temporal standard deviation")
    report.append("- **Steady-state**: First few TRs excluded to avoid T1 saturation effects")
    report.append("")
    
    report.append("### Quality Thresholds")
    report.append("These thresholds are based on typical fMRI data quality standards:")
    report.append("- **tSNR > 50**: Generally considered good quality for task fMRI")
    report.append("- **tSNR > 30**: Minimum acceptable for most analyses")
    report.append("- **Global SNR > 100**: Indicates good overall data stability")
    report.append("")
    
    report.append("### Factors Affecting SNR")
    report.append("- **Scanner field strength**: Higher field = potentially higher SNR")
    report.append("- **Acquisition parameters**: Voxel size, TR, TE, flip angle")
    report.append("- **Preprocessing**: Motion correction, spatial smoothing, denoising")
    report.append("- **Physiological factors**: Cardiac, respiratory, head motion artifacts")
    report.append("- **Task design**: Rest vs. task can affect signal characteristics")
    report.append("")
    
    # Save report
    report_file = Path(output_dir) / f"{subj}_{ses}_task-{task}_{run}_snr_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    return report_file

def main():
    args = get_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== fMRI SNR Analysis ===")
    print(f"Subject: {args.subj}")
    print(f"Task: {args.task}")
    print(f"Session: {args.ses}")
    print(f"Run: {args.run}")
    print(f"Space: {args.space}")
    print(f"Output directory: {output_dir}")
    print("")
    
    try:
        # Load fMRIPrep data
        bold_data, mask_data, header = load_fmriprep_data(
            args.fmriprep_root, args.subj, args.ses, args.task, args.run, args.space, args.skip_initial_trs
        )
        
        original_shape = (bold_data.shape[0], bold_data.shape[1], bold_data.shape[2], bold_data.shape[3] + args.skip_initial_trs)
        mask_voxels = np.sum(mask_data)
        
        print("")
        
        # Compute temporal SNR
        tsnr = compute_temporal_snr(bold_data, mask_data)
        
        # Analyze SNR statistics
        stats_dict, brain_tsnr = analyze_snr_statistics(tsnr, mask_data)
        
        # Compute global SNR
        global_signal, global_snr = compute_global_snr_timeseries(bold_data, mask_data)
        
        # Create visualizations
        plot_file = create_snr_visualizations(
            tsnr, brain_tsnr, global_signal, mask_data, output_dir, 
            args.subj, args.task, args.ses, args.run
        )
        print(f"Saved SNR visualizations: {plot_file}")
        
        # Create detailed report
        report_file = create_detailed_report(
            stats_dict, global_snr, args.subj, args.task, args.ses, args.run, args.space,
            output_dir, original_shape, mask_voxels, args.skip_initial_trs, brain_tsnr
        )
        print(f"Saved detailed report: {report_file}")
        
        # Print summary
        print(f"\n=== SNR ANALYSIS SUMMARY ===")
        print(f"Mean temporal SNR: {stats_dict['mean_tsnr']:.2f}")
        print(f"Median temporal SNR: {stats_dict['median_tsnr']:.2f}")
        print(f"Global signal SNR: {global_snr:.2f}")
        print(f"Good quality voxels (>50): {stats_dict['pct_good']:.1f}%")
        print(f"Acceptable quality voxels (>30): {stats_dict['pct_acceptable']:.1f}%")
        
        # Quality verdict
        if stats_dict['mean_tsnr'] > 60 and stats_dict['pct_good'] > 60:
            print("🟢 OVERALL QUALITY: EXCELLENT")
        elif stats_dict['mean_tsnr'] > 45 and stats_dict['pct_good'] > 40:
            print("🟡 OVERALL QUALITY: GOOD")
        elif stats_dict['mean_tsnr'] > 30 and stats_dict['pct_acceptable'] > 60:
            print("🟠 OVERALL QUALITY: ACCEPTABLE")
        else:
            print("🔴 OVERALL QUALITY: POOR")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()