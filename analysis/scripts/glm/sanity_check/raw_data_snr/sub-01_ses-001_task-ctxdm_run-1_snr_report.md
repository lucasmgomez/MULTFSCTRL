# fMRI Data Signal-to-Noise Ratio (SNR) Analysis Report
## Dataset: sub-01 ses-001 ctxdm run-1 (Space: T1w)

## Analysis Overview

### What is Signal-to-Noise Ratio (SNR)?
SNR quantifies the quality of fMRI data by measuring the ratio of signal to noise.
**Temporal SNR (tSNR)** is computed as mean/standard_deviation across time for each voxel.
Higher tSNR indicates better data quality and more reliable signal detection.

### Metrics Computed
1. **Temporal SNR**: Voxel-wise signal stability over time
2. **Global SNR**: Whole-brain average signal quality
3. **Regional SNR**: Spatial distribution of data quality
4. **Quality Assessment**: Percentage of voxels meeting quality thresholds

## Data Specifications
- Subject: sub-01
- Session: ses-001
- Task: ctxdm
- Run: run-1
- Space: T1w
- Original data shape: (76, 90, 71, 184)
- Brain mask voxels: 200,841
- Steady-state TRs skipped: 5
- Final time points analyzed: 179

## SNR Analysis Results

### Temporal SNR Statistics
- **Mean tSNR**: 41.28
- **Median tSNR**: 42.57
- **Standard deviation**: 20.95
- **Range**: [0.0, 190.2]
- **25th percentile**: 25.67
- **75th percentile**: 55.70

### Global Signal SNR
- **Global SNR**: 452.48
  (Ratio of mean global signal to its standard deviation over time)

### Data Quality Assessment

**Quality Thresholds:**
- **Excellent**: tSNR > 70
- **Good**: tSNR > 50
- **Acceptable**: tSNR > 30
- **Poor**: tSNR ≤ 30

**Quality Distribution:**
- **Excellent voxels** (tSNR > 70): 7.7%
- **Good voxels** (tSNR > 50): 35.4%
- **Acceptable voxels** (tSNR > 30): 69.9%
- **Poor voxels** (tSNR ≤ 30): 19.3%

**Total brain voxels analyzed**: 200,475

## Quality Assessment Interpretation

### Overall Data Quality: 🟠 **ACCEPTABLE**

**Interpretation**: This dataset shows acceptable SNR characteristics.
- Moderate temporal stability
- Higher noise levels may limit detection of small effects
- Suitable for basic analyses with appropriate statistical thresholds
- Consider additional preprocessing or conservative statistical approaches

## Recommendations

⚠️ **Proceed with caution**
- Use conservative statistical thresholds
- Focus on strong, expected effects
- Consider group-level analyses for increased power
- Implement robust preprocessing steps

## Technical Notes

### SNR Calculation Method
- **Temporal SNR**: tSNR = mean(signal) / std(signal) for each voxel
- **Global SNR**: Mean brain signal divided by its temporal standard deviation
- **Steady-state**: First few TRs excluded to avoid T1 saturation effects

### Quality Thresholds
These thresholds are based on typical fMRI data quality standards:
- **tSNR > 50**: Generally considered good quality for task fMRI
- **tSNR > 30**: Minimum acceptable for most analyses
- **Global SNR > 100**: Indicates good overall data stability

### Factors Affecting SNR
- **Scanner field strength**: Higher field = potentially higher SNR
- **Acquisition parameters**: Voxel size, TR, TE, flip angle
- **Preprocessing**: Motion correction, spatial smoothing, denoising
- **Physiological factors**: Cardiac, respiratory, head motion artifacts
- **Task design**: Rest vs. task can affect signal characteristics
