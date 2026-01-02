#!/usr/bin/env python3
"""
Comprehensive Nilearn GLM Quality Assessment
Following Nilearn tutorial best practices with noise ceiling calculation
"""

import sys
sys.path.append('/project/def-pbellec/xuan/fmri_dataset_project/scripts')

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting, image
from nilearn.masking import apply_mask
from glm_analysis_nilearn import *

class NilearnQualityAssessment:
    """Comprehensive quality assessment following Nilearn tutorial practices"""
    
    def __init__(self, subj='sub-01', ses='ses-001', run='run-01', task='ctxdm'):
        self.subj = subj
        self.ses = ses  
        self.run = run
        self.task = task
        self.results = {}
        
    def load_data_and_fit_glm(self):
        """Load data and fit GLM following our working pipeline"""
        
        print("🔬 Loading data and fitting GLM...")
        
        # File paths
        fmri_root = Path('/project/def-pbellec/xuan/cneuromod.multfs.fmriprep') / self.subj
        events_root = Path('/project/def-pbellec/xuan/fmri_dataset_project/data/reformated_behavior') / self.subj
        
        # Find files
        behavioral_file = events_root / self.ses / "func" / f"{self.subj}_{self.ses}_task-{self.task}_{self.run}_events.tsv"
        run_num_short = "1"  # We know it's run-1
        fmri_file = fmri_root / self.ses / "func" / f"{self.subj}_{self.ses}_task-{self.task}_run-{run_num_short}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        confounds_file = fmri_root / self.ses / "func" / f"{self.subj}_{self.ses}_task-{self.task}_run-{run_num_short}_desc-confounds_timeseries.tsv"
        
        print(f"Files: {fmri_file.exists()}, {confounds_file.exists()}, {behavioral_file.exists()}")
        
        # Load fMRI data
        fmri_img = nib.load(str(fmri_file))
        print(f"Original fMRI shape: {fmri_img.shape}")
        
        # Apply time masking
        tmask = 1
        tr = 1.49
        if tmask > 0:
            fmri_data = fmri_img.get_fdata()
            fmri_data_masked = fmri_data[..., tmask:]
            fmri_img = nib.Nifti1Image(fmri_data_masked, fmri_img.affine, fmri_img.header)
            print(f"After time masking: {fmri_img.shape}")
        
        # Load and process events
        df_events = pd.read_csv(behavioral_file, sep="\t")
        df_events = clean_events(df_events)
        events_df = create_nilearn_events(df_events, ['encoding', 'delay'], correct_only=False)
        
        # Adjust event onsets for time masking
        if tmask > 0:
            events_df['onset'] = events_df['onset'] - (tmask * tr)
            events_df = events_df[events_df['onset'] >= 0]
        
        print(f"Events: {len(events_df)}")
        
        # Load and process confounds
        df_confounds = pd.read_csv(confounds_file, sep="\t")
        if tmask > 0:
            df_confounds = df_confounds.iloc[tmask:].reset_index(drop=True)
        
        from utils import glm_confounds_construction
        confounds_processed = glm_confounds_construction(df_confounds)
        if isinstance(confounds_processed, pd.DataFrame):
            confounds_matrix = np.nan_to_num(confounds_processed.values, nan=0.0)
        else:
            confounds_matrix = np.nan_to_num(confounds_processed, nan=0.0)
        
        print(f"Confounds shape: {confounds_matrix.shape}")
        
        # Fit GLM
        print("Fitting FirstLevelModel...")
        glm = FirstLevelModel(
            t_r=tr,
            noise_model='ar1',
            standardize=False,
            hrf_model='spm',
            drift_model='cosine',
            high_pass=1.0 / 128.0,
            smoothing_fwhm=None,
            minimize_memory=False  # Need full results for quality checks
        )
        
        glm = glm.fit(fmri_img, events_df, confounds=confounds_matrix)
        print("✅ GLM fit complete")
        
        # Store results
        self.fmri_img = fmri_img
        self.events_df = events_df
        self.glm = glm
        self.confounds_matrix = confounds_matrix
        
        return glm, fmri_img, events_df
    
    def compute_r_squared_map(self):
        """Compute R-squared map following Nilearn tutorial"""
        
        print("📊 Computing R-squared map...")
        
        # Get R-squared map from GLM
        r_squared_img = self.glm.r_square
        r_squared_data = r_squared_img.get_fdata()
        
        # Compute statistics
        r_squared_flat = r_squared_data.flatten()
        r_squared_valid = r_squared_flat[np.isfinite(r_squared_flat)]
        
        stats = {
            'mean': np.mean(r_squared_valid),
            'median': np.median(r_squared_valid),
            'std': np.std(r_squared_valid),
            'min': np.min(r_squared_valid),
            'max': np.max(r_squared_valid),
            'q25': np.percentile(r_squared_valid, 25),
            'q75': np.percentile(r_squared_valid, 75),
            'n_voxels': len(r_squared_valid)
        }
        
        print(f"R² statistics: Mean={stats['mean']:.4f}, Range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Store results
        self.results['r_squared'] = {
            'img': r_squared_img,
            'data': r_squared_data,
            'stats': stats
        }
        
        return r_squared_img, stats
    
    def compute_f_test(self):
        """Compute F-test for overall model fit"""
        
        print("🧮 Computing F-test for model fit...")
        
        try:
            # Get design matrix
            design_matrix = self.glm.design_matrices_[0]
            
            # Find task columns (trials)
            task_columns = [col for col in design_matrix.columns if 'trial' in col.lower()]
            
            if len(task_columns) == 0:
                print("⚠️ No task columns found for F-test")
                return None, None
            
            # Create contrast for F-test (all task regressors vs zero)
            contrast = np.zeros(design_matrix.shape[1])
            task_indices = [design_matrix.columns.get_loc(col) for col in task_columns]
            contrast[task_indices] = 1.0 / len(task_indices)  # Average effect
            
            # Compute F-statistic map
            f_map = self.glm.compute_contrast(contrast, output_type='F')
            
            # Convert F to z-scores for visualization (approximate)
            f_data = f_map.get_fdata()
            f_valid = f_data[np.isfinite(f_data) & (f_data > 0)]
            
            # Convert F to z (approximate for large samples)
            z_data = np.sqrt(f_data.copy())
            z_data[~np.isfinite(f_data)] = 0
            z_img = nib.Nifti1Image(z_data, f_map.affine, f_map.header)
            
            stats = {
                'mean_f': np.mean(f_valid),
                'median_f': np.median(f_valid),
                'max_f': np.max(f_valid),
                'n_significant': np.sum(f_valid > 3.1),  # Approximate z > 3.1 threshold
                'fraction_significant': np.sum(f_valid > 3.1) / len(f_valid)
            }
            
            print(f"F-test: Mean F={stats['mean_f']:.2f}, {stats['n_significant']} significant voxels")
            
            self.results['f_test'] = {
                'f_img': f_map,
                'z_img': z_img, 
                'stats': stats
            }
            
            return f_map, z_img, stats
            
        except Exception as e:
            print(f"❌ Error in F-test computation: {e}")
            return None, None, None
    
    def analyze_residuals(self):
        """Analyze residuals following Nilearn tutorial"""
        
        print("📈 Analyzing residuals...")
        
        try:
            # Get residuals from GLM
            residuals_img = self.glm.residuals
            residuals_data = residuals_img.get_fdata()
            
            # Compute residual statistics
            residuals_flat = residuals_data.flatten()
            residuals_valid = residuals_flat[np.isfinite(residuals_flat)]
            
            # Sample for analysis (avoid memory issues)
            n_sample = min(10000, len(residuals_valid))
            sample_indices = np.random.choice(len(residuals_valid), n_sample, replace=False)
            residuals_sample = residuals_valid[sample_indices]
            
            # Compute statistics
            stats = {
                'mean': np.mean(residuals_valid),
                'std': np.std(residuals_valid),
                'skewness': float(stats.skew(residuals_sample)),
                'kurtosis': float(stats.kurtosis(residuals_sample)),
                'shapiro_stat': None,
                'shapiro_p': None
            }
            
            # Normality test (on sample)
            if len(residuals_sample) >= 20:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(residuals_sample[:5000])  # Max 5000 for shapiro
                    stats['shapiro_stat'] = float(shapiro_stat)
                    stats['shapiro_p'] = float(shapiro_p)
                except:
                    pass
            
            print(f"Residuals: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Skew={stats['skewness']:.3f}")
            
            self.results['residuals'] = {
                'img': residuals_img,
                'data': residuals_data,
                'stats': stats,
                'sample': residuals_sample
            }
            
            return residuals_img, stats
            
        except Exception as e:
            print(f"❌ Error in residuals analysis: {e}")
            return None, None
    
    def compute_noise_ceiling(self, n_splits=100):
        """Compute noise ceiling using split-half reliability"""
        
        print("🔊 Computing noise ceiling using split-half reliability...")
        
        try:
            # Get the fMRI data
            fmri_data = self.fmri_img.get_fdata()
            n_timepoints = fmri_data.shape[-1]
            
            # Reshape to 2D (voxels x time)
            original_shape = fmri_data.shape
            fmri_2d = fmri_data.reshape(-1, n_timepoints)
            
            # Remove zero/invalid voxels
            valid_mask = np.std(fmri_2d, axis=1) > 0
            fmri_valid = fmri_2d[valid_mask, :]
            
            print(f"Computing noise ceiling for {np.sum(valid_mask)} valid voxels")
            
            # Sample voxels for computational efficiency
            max_voxels = 5000
            if fmri_valid.shape[0] > max_voxels:
                voxel_indices = np.random.choice(fmri_valid.shape[0], max_voxels, replace=False)
                fmri_sample = fmri_valid[voxel_indices, :]
            else:
                fmri_sample = fmri_valid
                voxel_indices = np.arange(fmri_valid.shape[0])
            
            # Split-half reliability computation
            correlations = []
            
            for split in range(n_splits):
                # Random split of time points
                time_indices = np.arange(n_timepoints)
                np.random.shuffle(time_indices)
                
                split1_idx = time_indices[:n_timepoints//2]
                split2_idx = time_indices[n_timepoints//2:n_timepoints//2*2]  # Ensure equal length
                
                # Compute mean signal for each split
                signal1 = np.mean(fmri_sample[:, split1_idx], axis=1)
                signal2 = np.mean(fmri_sample[:, split2_idx], axis=1)
                
                # Compute correlation
                valid_mask_split = np.isfinite(signal1) & np.isfinite(signal2)
                if np.sum(valid_mask_split) > 10:
                    corr = np.corrcoef(signal1[valid_mask_split], signal2[valid_mask_split])[0, 1]
                    if np.isfinite(corr):
                        correlations.append(corr)
            
            # Spearman-Brown correction for full-length reliability
            correlations = np.array(correlations)
            split_half_reliability = correlations
            full_length_reliability = (2 * split_half_reliability) / (1 + split_half_reliability)
            
            # Noise ceiling is the square root of reliability (upper bound on prediction accuracy)
            noise_ceiling = np.sqrt(np.maximum(full_length_reliability, 0))
            
            noise_ceiling_stats = {
                'mean': np.mean(noise_ceiling),
                'median': np.median(noise_ceiling),
                'std': np.std(noise_ceiling),
                'min': np.min(noise_ceiling),
                'max': np.max(noise_ceiling),
                'q25': np.percentile(noise_ceiling, 25),
                'q75': np.percentile(noise_ceiling, 75),
                'n_splits': n_splits,
                'n_voxels_tested': fmri_sample.shape[0]
            }
            
            print(f"Noise ceiling: {noise_ceiling_stats['mean']:.3f} ± {noise_ceiling_stats['std']:.3f}")
            print(f"Range: [{noise_ceiling_stats['min']:.3f}, {noise_ceiling_stats['max']:.3f}]")
            
            self.results['noise_ceiling'] = {
                'values': noise_ceiling,
                'split_half_reliability': split_half_reliability,
                'full_length_reliability': full_length_reliability,
                'stats': noise_ceiling_stats
            }
            
            return noise_ceiling, noise_ceiling_stats
            
        except Exception as e:
            print(f"❌ Error in noise ceiling computation: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def create_diagnostic_plots(self, output_dir):
        """Create diagnostic plots following Nilearn tutorial style"""
        
        print("📊 Creating diagnostic plots...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # 1. R-squared histogram
        if 'r_squared' in self.results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            r2_data = self.results['r_squared']['data'].flatten()
            r2_valid = r2_data[np.isfinite(r2_data)]
            
            ax.hist(r2_valid, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(self.results['r_squared']['stats']['mean'], color='red', linestyle='--', 
                      label=f"Mean: {self.results['r_squared']['stats']['mean']:.4f}")
            ax.axvline(self.results['r_squared']['stats']['median'], color='orange', linestyle='--',
                      label=f"Median: {self.results['r_squared']['stats']['median']:.4f}")
            
            ax.set_xlabel('R²')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of R² Values Across Voxels')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'r_squared_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Residuals analysis
        if 'residuals' in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            residuals = self.results['residuals']['sample']
            
            # Histogram
            axes[0,0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            axes[0,0].set_xlabel('Residuals')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title('Residuals Distribution')
            axes[0,0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[0,1])
            axes[0,1].set_title('Q-Q Plot (Normal Distribution)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Box plot
            axes[1,0].boxplot(residuals)
            axes[1,0].set_ylabel('Residuals')
            axes[1,0].set_title('Residuals Box Plot')
            axes[1,0].grid(True, alpha=0.3)
            
            # Statistics text
            stats_text = f"""Residuals Statistics:
Mean: {self.results['residuals']['stats']['mean']:.4f}
Std: {self.results['residuals']['stats']['std']:.4f}
Skewness: {self.results['residuals']['stats']['skewness']:.3f}
Kurtosis: {self.results['residuals']['stats']['kurtosis']:.3f}"""
            
            if self.results['residuals']['stats']['shapiro_p'] is not None:
                stats_text += f"\nShapiro-Wilk p: {self.results['residuals']['stats']['shapiro_p']:.4f}"
            
            axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                          fontsize=12, verticalalignment='center')
            axes[1,1].set_xticks([])
            axes[1,1].set_yticks([])
            axes[1,1].set_title('Residuals Statistics')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'residuals_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Noise ceiling plot
        if 'noise_ceiling' in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            nc_values = self.results['noise_ceiling']['values']
            
            # Histogram
            axes[0].hist(nc_values, bins=30, alpha=0.7, edgecolor='black')
            axes[0].axvline(self.results['noise_ceiling']['stats']['mean'], color='red', linestyle='--',
                           label=f"Mean: {self.results['noise_ceiling']['stats']['mean']:.3f}")
            axes[0].set_xlabel('Noise Ceiling')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Noise Ceiling Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Box plot with percentiles
            axes[1].boxplot(nc_values)
            axes[1].set_ylabel('Noise Ceiling')
            axes[1].set_title('Noise Ceiling Box Plot')
            axes[1].grid(True, alpha=0.3)
            
            # Add text with statistics
            nc_stats = self.results['noise_ceiling']['stats']
            stats_text = f"Mean: {nc_stats['mean']:.3f}\nStd: {nc_stats['std']:.3f}\nRange: [{nc_stats['min']:.3f}, {nc_stats['max']:.3f}]"
            axes[1].text(1.1, 0.5, stats_text, transform=axes[1].transAxes, fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'noise_ceiling_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Diagnostic plots saved to: {output_dir}")
    
    def generate_report(self, output_file):
        """Generate comprehensive quality assessment report"""
        
        print("📝 Generating comprehensive quality report...")
        
        report = f"""# Nilearn GLM Quality Assessment Report

**Subject**: {self.subj}  
**Session**: {self.ses}  
**Run**: {self.run}  
**Task**: {self.task}  
**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration

- **Noise Model**: AR(1) autocorrelation
- **HRF Model**: SPM canonical
- **Drift Model**: Cosine high-pass filter (128s)
- **Standardization**: False
- **Spatial Smoothing**: None

## GLM Fit Quality

"""
        
        # R-squared analysis
        if 'r_squared' in self.results:
            r2_stats = self.results['r_squared']['stats']
            report += f"""### R² Analysis (Variance Explained)

- **Mean R²**: {r2_stats['mean']:.4f}
- **Median R²**: {r2_stats['median']:.4f}
- **Standard Deviation**: {r2_stats['std']:.4f}
- **Range**: [{r2_stats['min']:.4f}, {r2_stats['max']:.4f}]
- **25th-75th Percentile**: [{r2_stats['q25']:.4f}, {r2_stats['q75']:.4f}]
- **Valid Voxels**: {r2_stats['n_voxels']:,}

**Interpretation**: """
            
            if r2_stats['mean'] > 0.1:
                report += "Good model fit - substantial variance explained."
            elif r2_stats['mean'] > 0.05:
                report += "Moderate model fit - typical for task fMRI."
            else:
                report += "Low model fit - may indicate preprocessing issues or weak signal."
        
        # F-test analysis  
        if 'f_test' in self.results:
            f_stats = self.results['f_test']['stats']
            report += f"""

### F-Test Analysis (Overall Model Significance)

- **Mean F-statistic**: {f_stats['mean_f']:.2f}
- **Max F-statistic**: {f_stats['max_f']:.2f}
- **Significant voxels**: {f_stats['n_significant']:,}
- **Fraction significant**: {f_stats['fraction_significant']:.1%}

**Interpretation**: """
            
            if f_stats['fraction_significant'] > 0.1:
                report += "Strong task-related activation detected."
            elif f_stats['fraction_significant'] > 0.05:
                report += "Moderate task-related activation detected."
            else:
                report += "Weak task-related activation - check experimental design."
        
        # Residuals analysis
        if 'residuals' in self.results:
            res_stats = self.results['residuals']['stats']
            report += f"""

### Residuals Analysis (Model Assumptions)

- **Mean**: {res_stats['mean']:.4f}
- **Standard Deviation**: {res_stats['std']:.4f}
- **Skewness**: {res_stats['skewness']:.3f}
- **Kurtosis**: {res_stats['kurtosis']:.3f}"""
            
            if res_stats['shapiro_p'] is not None:
                report += f"\n- **Shapiro-Wilk p-value**: {res_stats['shapiro_p']:.4f}"
                
                if res_stats['shapiro_p'] > 0.05:
                    normality = "Residuals appear normally distributed ✅"
                else:
                    normality = "Residuals deviate from normality ⚠️"
                report += f"\n\n**Normality**: {normality}"
        
        # Noise ceiling analysis
        if 'noise_ceiling' in self.results:
            nc_stats = self.results['noise_ceiling']['stats']
            report += f"""

### Noise Ceiling Analysis (Data Quality Upper Bound)

- **Mean Noise Ceiling**: {nc_stats['mean']:.3f}
- **Standard Deviation**: {nc_stats['std']:.3f}
- **Range**: [{nc_stats['min']:.3f}, {nc_stats['max']:.3f}]
- **25th-75th Percentile**: [{nc_stats['q25']:.3f}, {nc_stats['q75']:.3f}]
- **Splits Used**: {nc_stats['n_splits']}
- **Voxels Tested**: {nc_stats['n_voxels_tested']:,}

**Interpretation**: """
            
            if nc_stats['mean'] > 0.7:
                report += "Excellent data quality - high signal-to-noise ratio."
            elif nc_stats['mean'] > 0.5:
                report += "Good data quality - moderate signal-to-noise ratio."  
            elif nc_stats['mean'] > 0.3:
                report += "Fair data quality - consider preprocessing improvements."
            else:
                report += "Poor data quality - significant preprocessing issues likely."
        
        # Overall assessment
        report += f"""

## Overall Assessment

"""
        
        # Determine overall quality
        quality_indicators = []
        
        if 'r_squared' in self.results:
            if self.results['r_squared']['stats']['mean'] > 0.05:
                quality_indicators.append("✅ R² indicates reasonable model fit")
            else:
                quality_indicators.append("❌ Low R² suggests poor model fit")
        
        if 'f_test' in self.results:
            if self.results['f_test']['stats']['fraction_significant'] > 0.05:
                quality_indicators.append("✅ F-test shows significant task activation")
            else:
                quality_indicators.append("❌ F-test shows minimal task activation")
        
        if 'noise_ceiling' in self.results:
            if self.results['noise_ceiling']['stats']['mean'] > 0.3:
                quality_indicators.append("✅ Noise ceiling indicates usable data quality")
            else:
                quality_indicators.append("❌ Low noise ceiling suggests poor data quality")
        
        for indicator in quality_indicators:
            report += f"- {indicator}\n"
        
        # Recommendations
        report += f"""

## Recommendations

"""
        
        if 'r_squared' in self.results and self.results['r_squared']['stats']['mean'] < 0.05:
            report += "- **Low R²**: Consider adding spatial smoothing, checking preprocessing, or validating experimental design\n"
        
        if 'noise_ceiling' in self.results and self.results['noise_ceiling']['stats']['mean'] < 0.4:
            report += "- **Low noise ceiling**: Investigate motion correction, temporal filtering, and outlier detection\n"
        
        if 'residuals' in self.results:
            if abs(self.results['residuals']['stats']['skewness']) > 2:
                report += "- **Skewed residuals**: Check for outliers and temporal artifacts\n"
            if self.results['residuals']['stats']['shapiro_p'] is not None and self.results['residuals']['stats']['shapiro_p'] < 0.001:
                report += "- **Non-normal residuals**: Consider robust GLM or additional preprocessing\n"
        
        report += """
## Files Generated

- `r_squared_distribution.png`: R² distribution across voxels
- `residuals_analysis.png`: Comprehensive residuals diagnostics  
- `noise_ceiling_analysis.png`: Noise ceiling distribution and statistics

---
*Generated by Nilearn GLM Quality Assessment Pipeline*
"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {output_file}")

def main():
    print("🔬 NILEARN GLM QUALITY ASSESSMENT")
    print("Following Nilearn tutorial best practices")
    print("="*60)
    
    # Initialize assessment
    assessment = NilearnQualityAssessment()
    
    # Load data and fit GLM
    try:
        glm, fmri_img, events_df = assessment.load_data_and_fit_glm()
    except Exception as e:
        print(f"❌ Failed to load data and fit GLM: {e}")
        return
    
    # Run quality checks
    print("\n" + "="*40)
    print("RUNNING QUALITY CHECKS")
    print("="*40)
    
    # 1. R-squared analysis
    assessment.compute_r_squared_map()
    
    # 2. F-test for model significance
    assessment.compute_f_test()
    
    # 3. Residuals analysis
    assessment.analyze_residuals()
    
    # 4. Noise ceiling computation
    assessment.compute_noise_ceiling(n_splits=50)  # Reduced for speed
    
    # Generate outputs
    output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/nilearn_quality_assessment")
    
    # Create diagnostic plots
    assessment.create_diagnostic_plots(output_dir)
    
    # Generate comprehensive report
    report_file = output_dir / "nilearn_glm_quality_report.md"
    assessment.generate_report(report_file)
    
    print("\n" + "="*60)
    print("✅ QUALITY ASSESSMENT COMPLETE")
    print("="*60)
    print(f"📊 Results saved to: {output_dir}")
    print(f"📄 Report: {report_file}")
    
    # Print key findings
    if 'r_squared' in assessment.results:
        r2_mean = assessment.results['r_squared']['stats']['mean']
        print(f"📈 Mean R²: {r2_mean:.4f}")
    
    if 'noise_ceiling' in assessment.results:
        nc_mean = assessment.results['noise_ceiling']['stats']['mean']
        print(f"🔊 Noise Ceiling: {nc_mean:.3f}")
    
    if 'f_test' in assessment.results:
        sig_frac = assessment.results['f_test']['stats']['fraction_significant']
        print(f"🎯 Significant Activation: {sig_frac:.1%}")

if __name__ == "__main__":
    main()