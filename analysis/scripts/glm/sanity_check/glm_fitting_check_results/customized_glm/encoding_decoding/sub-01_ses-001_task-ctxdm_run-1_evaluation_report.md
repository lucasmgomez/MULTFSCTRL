# GLM Evaluation Report (Vertex-level)
**Subject:** sub-01  
**Task:** ctxdm  
**Session:** ses-001  
**Run:** run-1

## Model Specs
- # regressors (K): 81
- # task regressors: 60
- # time points after tmask (T_kept): 183
- AR(1) rho: 0.000
- Estimator: OLS

## Vertex Sampling
- Total vertices in run: 64984
- Sampled vertices for plots: 1000

## Fit (Sampled Vertices)
- Mean R²: 0.057
- Median R²: 0.601
- R² std: 4.985
- R² range: [-45.164, 0.931]
- Vertices with R² > 0.0: 988/1000 (98.8%)
- Vertices with R² > 0.1: 988/1000 (98.8%)
- Vertices with R² > 0.2: 988/1000 (98.8%)

## Global R² (reduced subset)
- Evaluated vertices: 2000
- Mean R²: 0.258
- Median R²: 0.603
- R² std: 4.021
- R² range: [-45.980, 0.932]

## Task Beta Consistency (full vs saved task slice)
- Mean |Δ| over sampled vertices: 0
- Max  |Δ| over sampled vertices: 0

## Extremely Negative R²
- Threshold: R² < -1.0
- Count: 618 / 59412 (1.04%)