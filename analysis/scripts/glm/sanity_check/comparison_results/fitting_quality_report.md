
# GLM Fitting Quality Assessment Report

## Dataset Information
- Subject: sub-synthetic
- Session: ses-001
- Task: ctxdm
- Run: run-01
- Number of parcels: 128
- Number of task regressors: 40

## Overall Fitting Performance

### Primary Metrics
- **Pearson Correlation**: 0.8940 (p = 0.00e+00)
- **R² (Coefficient of Determination)**: 0.7633
- **Mean Absolute Error (MAE)**: 0.9610
- **Root Mean Square Error (RMSE)**: 1.2128
- **Bias (Mean Difference)**: 0.0090

### Interpretation of Overall Results

**Correlation (0.8940):**
VERY GOOD - Strong linear relationship, indicating accurate GLM fitting.

**R² (0.7633):**
GOOD - GLM explains 76.3% of variance, reasonably accurate fitting.

**Bias (0.0090):**
EXCELLENT - Minimal systematic bias in the fitted estimates.

## Per-Parcel Analysis

- **Mean correlation across parcels**: 0.8942
- **Mean R² across parcels**: 0.7414
- **Number of valid parcels**: 128
- **Parcels with correlation > 0.8**: 125 (97.7%)
- **Parcels with correlation > 0.5**: 128 (100.0%)


## Per-Regressor Analysis

- **Mean correlation across regressors**: 0.8946
- **Mean R² across regressors**: 0.7588
- **Number of valid regressors**: 40
- **Regressors with correlation > 0.8**: 39 (97.5%)
- **Regressors with correlation > 0.5**: 40 (100.0%)


## Overall Assessment

Based on the comprehensive analysis above:


**CONCLUSION: EXCELLENT GLM PERFORMANCE**

The GLM fitting is working very well:
- Strong correlation indicates linear relationship is preserved
- High R² shows most variance is explained 
- Low bias suggests unbiased estimates
- Consistent performance across parcels and regressors

This synthetic dataset validation demonstrates that the GLM analysis pipeline
is correctly implemented and produces accurate results.


## Technical Notes

- All metrics computed on flattened beta matrices
- NaN values excluded from per-parcel/regressor statistics
- Synthetic data generated with AR(1) noise (ρ = 0.3, σ = 1.0)
- GLM includes confound regression and high-pass filtering
- Comparison represents validation of entire analysis pipeline

Generated on: 2025-08-25 05:27:33.787837
