# Simulation Study Results

## Overview
This Monte Carlo study compares variance component estimators in the LMM described in gemini.md/openAI.md across 100 replicates per design/weight setting.
We vary the design condition number (kappa) and the residual weight d.o.f. proxy (K) while holding n=1000, q=110, and L=10 fixed.

**Parameters:**
- n = 1000
- q = 110
- L = 10
- N_sim = 100
- kappa in {1, 10, 100, 1000, 10000}
- K in {2, 10, 100}
- Ridge tuning lambda_ell = (0.25 + 0.02*sqrt(kappa) + 1/(K+1)) * C_ell

## Results Table (Average MSE)
| Kappa | K | Naive | Unbias | Cutoff | FR_Plug | FR_Adj | EM | Oracle |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 6.17e-02 | 6.14e-02 | 6.14e-02 | 6.17e-02 | 6.13e-02 | 6.13e-02 | 5.24e-02 |
| 1 | 10 | 5.46e-02 | 5.45e-02 | 5.45e-02 | 5.46e-02 | 5.45e-02 | 5.45e-02 | 5.16e-02 |
| 1 | 100 | 5.70e-02 | 5.68e-02 | 5.68e-02 | 5.69e-02 | 5.67e-02 | 5.66e-02 | 5.34e-02 |
| 10 | 2 | 9.58e-01 | 8.00e-01 | 8.00e-01 | 9.21e-01 | 7.69e-01 | 5.68e-01 | 5.41e-02 |
| 10 | 10 | 2.70e-01 | 1.70e-01 | 1.70e-01 | 2.57e-01 | 1.64e-01 | 1.21e-01 | 6.02e-02 |
| 10 | 100 | 2.09e-01 | 1.36e-01 | 1.36e-01 | 2.00e-01 | 1.32e-01 | 1.02e-01 | 5.45e-02 |
| 100 | 2 | 1.98e+03 | 1.59e+03 | 1.59e+03 | 2.25e+02 | 1.83e+02 | 1.55e+02 | 5.27e-02 |
| 100 | 10 | 3.42e+02 | 1.00e+02 | 1.00e+02 | 1.98e+01 | 5.67e+00 | 1.09e+00 | 5.92e-02 |
| 100 | 100 | 2.48e+02 | 6.36e+01 | 6.36e+01 | 1.59e+01 | 4.45e+00 | 1.47e+00 | 5.76e-02 |
| 1000 | 2 | 1.22e+07 | 1.01e+07 | 1.01e+07 | 1.41e+02 | 1.20e+02 | 2.88e+05 | 4.53e-02 |
| 1000 | 10 | 2.05e+06 | 7.22e+05 | 7.22e+05 | 4.86e+00 | 1.69e+00 | 2.43e+00 | 5.39e-02 |
| 1000 | 100 | 1.28e+06 | 3.12e+05 | 3.12e+05 | 3.26e+00 | 1.25e+00 | 2.79e+00 | 5.18e-02 |
| 10000 | 2 | 6.13e+10 | 5.01e+10 | 5.01e+10 | 2.70e+01 | 2.38e+01 | 1.40e+07 | 5.82e-02 |
| 10000 | 10 | 1.17e+10 | 4.42e+09 | 4.42e+09 | 4.03e-01 | 3.06e-01 | 4.17e+00 | 5.78e-02 |
| 10000 | 100 | 8.32e+09 | 2.48e+09 | 2.48e+09 | 2.21e-01 | 2.97e-01 | 5.09e+00 | 5.47e-02 |

## Key Observations
1. Naive MoM blows up under strong ill-conditioning: at kappa=10000, K=2 its MSE reaches 6.13e+10, whereas FR_Adj stays near 2.38e+01.
2. Increasing K stabilizes the plug-in weights; e.g., for kappa=100 the Naive MSE drops from 1.98e+03 at K=2 to 2.48e+02 at K=100.
3. EM-ML reduces the large-MSE tail of MoM but still lags behind ridge; at kappa=1000, K=10 it delivers 2.43e+00 vs 1.69e+00 (FR_Adj).
4. The Oracle benchmark (group means of true u^2) sits near 5.34e-02 MSE, indicating the Monte Carlo noise floor for these group sizes.
