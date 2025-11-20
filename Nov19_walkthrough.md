# Simulation Study Results (CV-Tuned Fixed Ridge)

## Overview

This simulation evaluates variance component estimators in a Linear Mixed Model (LMM).
**Update:** The Fixed Ridge (FR) estimator now uses **5-fold Cross-Validation (CV)** via `glmnet` to select the regularization parameter $\lambda$.

**Parameters:**

- $n = 1000$
- $N_{sim} = 1000$
- $\kappa \in \{1, 10, 100, 1000, 10000\}$
- $K \in \{2, 10, 100\}$

## Results Table (Average MSE)

| Kappa | K | Naive | Unbias | Cutoff | FR_Plug (CV) | FR_Adj (CV) | EM | Oracle |
|-------|---|-------|--------|--------|--------------|-------------|----|--------|
| 1 | 2 | 6.61e-02 | 6.57e-02 | 6.57e-02 | 6.59e-02 | 6.55e-02 | 6.57e-02 | 5.60e-02 |
| 1 | 10 | 5.97e-02 | 5.94e-02 | 5.94e-02 | 5.94e-02 | 5.92e-02 | 5.94e-02 | 5.58e-02 |
| 1 | 100 | 5.73e-02 | 5.72e-02 | 5.72e-02 | 5.70e-02 | 5.69e-02 | 5.71e-02 | 5.45e-02 |
| 10 | 2 | 1.09e+00 | 9.18e-01 | 9.18e-01 | 9.36e-01 | 7.91e-01 | 6.54e-01 | 5.60e-02 |
| 10 | 10 | 2.83e-01 | 1.77e-01 | 1.77e-01 | 2.13e-01 | 1.41e-01 | 1.26e-01 | 5.58e-02 |
| 10 | 100 | 2.16e-01 | 1.35e-01 | 1.35e-01 | 1.58e-01 | 1.10e-01 | 1.04e-01 | 5.45e-02 |
| 100 | 2 | 2.45e+03 | 1.99e+03 | 1.99e+03 | 64.07 | 51.72 | 1.54e+02 | 5.60e-02 |
| 100 | 10 | 4.41e+02 | 1.55e+02 | 1.55e+02 | 0.34 | **22.19** | 1.28 | 5.58e-02 |
| 100 | 100 | 3.14e+02 | 9.09e+01 | 9.09e+01 | 0.34 | **24.71** | 1.76 | 5.45e-02 |
| 1000 | 2 | 1.20e+07 | 9.78e+06 | 9.79e+06 | 0.35 | **35.36** | 6.83e+04 | 5.60e-02 |
| 1000 | 10 | 2.14e+06 | 7.80e+05 | 7.80e+05 | 0.35 | **47.65** | 3.76 | 5.58e-02 |
| 1000 | 100 | 1.54e+06 | 4.77e+05 | 4.77e+05 | 0.35 | **49.15** | 3.36 | 5.45e-02 |
| 10000 | 2 | 7.32e+10 | 6.01e+10 | 6.01e+10 | 0.36 | **32.72** | 4.28e+07 | 5.60e-02 |
| 10000 | 10 | 1.31e+10 | 4.98e+09 | 4.98e+09 | 0.36 | **43.83** | 4.90 | 5.58e-02 |
| 10000 | 100 | 9.49e+09 | 3.12e+09 | 3.12e+09 | 0.36 | **44.95** | 4.58 | 5.45e-02 |

## Key Observations

1. **Fixed Ridge (Plug-in) Robustness**:
    - `FR_Plug` with CV is extremely robust. Even at $\kappa=10000$, MSE remains low ($\approx 0.36$).
    - This confirms that CV successfully selects a regularization parameter that stabilizes the estimator.

2. **Fixed Ridge (Adjusted) Instability**:
    - `FR_Adj` shows **anomalously high MSE** (ranging from 20 to 50) for $\kappa \ge 100$.
    - **Cause**: This is likely due to a scaling mismatch in the bias correction term. The penalty matrix $D$ constructed from `glmnet`'s `lambda.min` was likely under-scaled relative to the actual objective function used by `glmnet` (due to weight normalization), leading to an incorrect subtraction in the bias correction step.

3. **Comparison**:
    - Standard estimators (Naive, Unbias) fail completely at high $\kappa$.
    - EM is better but still degrades.
    - `FR_Plug` is the most stable estimator in this high-ill-conditioning regime.

## P(f,t) Estimation Results (K=2)

Evaluation of methods for estimating the separable variance surface $P(f,t)$ from noisy observations.

| Method | RMSE (Log Scale) | MAPE |
|:---|---:|---:|
| **Parametric GLM** | **0.039** | **0.033** |
| Gamma GAM | 0.077 | 0.062 |
| Log-WLS | 0.099 | 0.080 |
| Alt-MLE | 0.237 | 0.185 |

**Conclusion:**
The **Parametric GLM** provides the best accuracy as it correctly specifies the functional form. Among flexible methods, the **Gamma GAM** is superior to Log-WLS and Alt-MLE.

## Smoothed Weights Simulation Results

Comparison of LMM estimators when using weights estimated via **Parametric GLM** and **Gamma GAM** (instead of raw noisy weights).
*Fixed Ridge uses fixed $\lambda = 10^{-5}$.*

### Results with Parametric GLM Weights

| Kappa | K | Naive | Unbias | Cutoff | FR_Plug | FR_Adj | EM | Oracle |
|-------|---|-------|--------|--------|---------|--------|----|--------|
| 1 | 2 | 0.058 | 0.058 | 0.058 | 0.058 | 0.058 | 0.058 | 0.056 |
| 1 | 10 | 0.057 | 0.057 | 0.057 | 0.057 | 0.057 | 0.057 | 0.054 |
| 1 | 100 | 0.057 | 0.057 | 0.057 | 0.057 | 0.057 | 0.057 | 0.054 |
| 10 | 2 | 0.214 | 0.137 | 0.137 | 0.162 | 0.114 | 0.108 | 0.056 |
| 10 | 10 | 0.212 | 0.136 | 0.136 | 0.161 | 0.114 | 0.103 | 0.054 |
| 10 | 100 | 0.208 | 0.133 | 0.133 | 0.158 | 0.112 | 0.104 | 0.054 |
| 100 | 2 | 284.4 | 76.7 | 76.7 | 1.43 | 0.46 | 1.87 | 0.056 |
| 100 | 10 | 299.4 | 85.7 | 85.7 | 1.45 | 0.46 | 1.81 | 0.054 |
| 100 | 100 | 283.5 | 79.3 | 79.3 | 1.45 | 0.47 | 1.94 | 0.054 |
| 1000 | 2 | 1.38e6 | 3.90e5 | 3.90e5 | 0.63 | 0.35 | 4.91 | 0.056 |
| 1000 | 10 | 1.48e6 | 4.50e5 | 4.50e5 | 0.63 | 0.34 | 3.81 | 0.054 |
| 1000 | 100 | 1.37e6 | 4.06e5 | 4.06e5 | 0.64 | 0.34 | 4.64 | 0.054 |
| 10000 | 2 | 8.49e9 | 2.59e9 | 2.59e9 | 0.27 | 0.31 | 9.63 | 0.056 |
| 10000 | 10 | 9.16e9 | 2.99e9 | 2.99e9 | 0.26 | 0.30 | 6.26 | 0.054 |
| 10000 | 100 | 8.46e9 | 2.66e9 | 2.66e9 | 0.27 | 0.30 | 6.28 | 0.054 |
