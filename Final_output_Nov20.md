# Simulation Study Results (CV-Tuned Fixed Ridge)

## Overview

This simulation evaluates variance component estimators in a Linear Mixed Model (LMM).
**Update:** The Fixed Ridge (FR) estimator now uses **5-fold Cross-Validation (CV)** via `glmnet` to select the regularization parameter $\lambda$.

**Parameters:**

- $n = 1000$
- $N_{sim} = 1000$
- $\kappa \in \{1, 10, 100, 1000, 10000\}$
- $K \in \{2, 10, 100\}$

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

## Combined Simulation Results (Standard vs Smoothed)

This simulation compares two approaches for estimating the variance components in the LMM:

1. **Standard (Noisy Weights)**: Uses weights $W = 1/\hat{R}$ derived directly from the noisy multiplicative surrogates $\hat{P}_k(f,t)$.
2. **Smoothed (Parametric GLM Weights)**: First estimates the smooth surface $P_k(f,t)$ using a **Parametric Gamma GLM** (log link) on the noisy surrogates, then constructs weights $W = 1/(\hat{P}_1 \hat{P}_2)$.

**Why Parametric GLM?**
The raw surrogates $\hat{P}_k$ are very noisy (Gamma distributed). Using them directly leads to highly variable weights, which can destabilize the LMM estimators, especially for the "Unbias" and "Cutoff" methods. Smoothing the surface $P(f,t)$ before weight construction reduces this variance, potentially leading to more stable and accurate variance component estimates.

**Methodology:**

- **Identical Data**: Both methods are applied to the *exact same* data realizations in each replicate.
- **Independent Tuning**: Both methods use **5-fold Cross-Validation** (via `glmnet`) to select their own optimal $\lambda$ for the Fixed Ridge estimator.

### 1. Standard Results (Noisy Weights)

| Kappa | K | Naive | Unbias | Cutoff | FR_Plug | FR_Adj | EM | Oracle |
|-------|---|-------|--------|--------|---------|--------|----|--------|
| 1 | 2 | 0.064 | 0.063 | 0.063 | 0.278 | 0.282 | 0.063 | 0.055 |
| 1 | 10 | 0.056 | 0.056 | 0.056 | 0.233 | 0.241 | 0.056 | 0.053 |
| 1 | 100 | 0.056 | 0.056 | 0.056 | 0.229 | 0.239 | 0.056 | 0.054 |
| 10 | 2 | 1.01 | 0.85 | 0.85 | 0.333 | 0.364 | 0.62 | 0.055 |
| 10 | 10 | 0.26 | 0.16 | 0.16 | 0.323 | 0.425 | 0.12 | 0.053 |
| 10 | 100 | 0.22 | 0.14 | 0.14 | 0.323 | 0.447 | 0.11 | 0.054 |
| 100 | 2 | 2280 | 1852 | 1853 | 0.348 | 0.388 | 181 | 0.055 |
| 100 | 10 | 404 | 140 | 140 | 0.345 | 0.501 | 1.31 | 0.053 |
| 100 | 100 | 310 | 91 | 91 | 0.345 | 0.544 | 1.76 | 0.054 |
| 1000 | 2 | 1.12e7 | 9.15e6 | 9.15e6 | 0.354 | 0.390 | 1.83e5 | 0.055 |
| 1000 | 10 | 1.96e6 | 7.11e5 | 7.12e5 | 0.353 | 0.490 | 3.31 | 0.053 |
| 1000 | 100 | 1.49e6 | 4.51e5 | 4.51e5 | 0.353 | 0.530 | 4.77 | 0.054 |
| 10000 | 2 | 6.88e10 | 5.68e10 | 5.68e10 | 0.358 | 0.391 | 5.30e8 | 0.055 |
| 10000 | 10 | 1.21e10 | 4.61e9 | 4.61e9 | 0.357 | 0.483 | 6.15 | 0.053 |
| 10000 | 100 | 9.02e9 | 2.89e9 | 2.89e9 | 0.358 | 0.521 | 7.19 | 0.054 |

### 2. Smoothed Results (Parametric GLM Weights)

| Kappa | K | Naive | Unbias | Cutoff | FR_Plug | FR_Adj | EM | Oracle |
|-------|---|-------|--------|--------|---------|--------|----|--------|
| 1 | 2 | 0.058 | 0.057 | 0.057 | 0.229 | 0.239 | 0.057 | 0.055 |
| 1 | 10 | 0.055 | 0.055 | 0.055 | 0.229 | 0.239 | 0.055 | 0.053 |
| 1 | 100 | 0.056 | 0.056 | 0.056 | 0.229 | 0.239 | 0.056 | 0.054 |
| 10 | 2 | 0.21 | 0.13 | 0.13 | 0.322 | 0.449 | 0.10 | 0.055 |
| 10 | 10 | 0.21 | 0.13 | 0.13 | 0.323 | 0.449 | 0.10 | 0.053 |
| 10 | 100 | 0.22 | 0.14 | 0.14 | 0.323 | 0.450 | 0.10 | 0.054 |
| 100 | 2 | 290 | 83 | 83 | 0.345 | 0.547 | 1.84 | 0.055 |
| 100 | 10 | 289 | 84 | 84 | 0.345 | 0.544 | 1.84 | 0.053 |
| 100 | 100 | 301 | 88 | 88 | 0.345 | 0.549 | 1.84 | 0.054 |
| 1000 | 2 | 1.41e6 | 4.24e5 | 4.24e5 | 0.353 | 0.532 | 4.30 | 0.055 |
| 1000 | 10 | 1.40e6 | 4.29e5 | 4.29e5 | 0.353 | 0.529 | 4.69 | 0.053 |
| 1000 | 100 | 1.44e6 | 4.37e5 | 4.37e5 | 0.353 | 0.535 | 4.75 | 0.054 |
| 10000 | 2 | 8.65e9 | 2.77e9 | 2.77e9 | 0.358 | 0.522 | 7.38 | 0.055 |
| 10000 | 10 | 8.56e9 | 2.80e9 | 2.80e9 | 0.357 | 0.518 | 9.27 | 0.053 |
| 10000 | 100 | 8.76e9 | 2.81e9 | 2.81e9 | 0.358 | 0.525 | 7.06 | 0.054 |
