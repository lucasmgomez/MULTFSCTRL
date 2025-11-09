# GLM Evaluation Report (Vertex-level)
**Subject:** sub-02  
**Task:** ctxdm  
**Session:** ses-001  
**Run:** run-1

## Model Specs
- # regressors (K): 93
- # task regressors: 60
- # time points after tmask (T_kept): 183
- AR(1) rho: 0.000
- Estimator: OLS

## Vertex Sampling
- Total vertices in run: 64984
- Sampled vertices for plots: 1000

## Fit (Sampled Vertices)
- Mean R²: 0.653
- Median R²: 0.663
- R² std: 0.150
- R² range: [0.000, 0.951]
- Vertices with R² > 0.0: 983/1000 (98.3%)
- Vertices with R² > 0.1: 983/1000 (98.3%)
- Vertices with R² > 0.2: 983/1000 (98.3%)

## Global R² (reduced subset)
- Evaluated vertices: 2000
- Mean R²: 0.657
- Median R²: 0.662
- R² std: 0.144
- R² range: [0.000, 0.968]

## Task Beta Consistency (full vs saved task slice)
- Mean |Δ| over sampled vertices: 0
- Max  |Δ| over sampled vertices: 0

## Extremely Negative R²
- Threshold: R² < -1.0
- Count: 0 / 59412 (0.00%)