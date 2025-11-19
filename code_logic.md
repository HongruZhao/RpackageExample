# Code Logic: Linear Mixed Model Simulation Study

This document details the logic and implementation of `simulation_study.R`. It is designed to allow for exact reproduction of the code's functionality and to facilitate human review of the logical flow.

## 1. Overview
The script simulates a Linear Mixed Model (LMM) without fixed effects ($p=0$) to evaluate the performance of various variance component estimators under conditions of design matrix ill-conditioning and residual heteroscedasticity.

**Model:**
$$ y = Z u + e $$
where:
- $y$ is the $n \times 1$ response vector.
- $Z$ is the $n \times q$ random effects design matrix.
- $u \sim N(0, G)$ is the $q \times 1$ random effects vector. $G$ is diagonal with block structure.
- $e \sim N(0, R)$ is the $n \times 1$ residual vector.

## 2. Data Generation

### 2.1. Design Matrix ($Z$)
**Function:** `make_Z_svd(n, q, kappa, anchor)`
- **Goal:** Construct $Z$ with a specific condition number $\kappa$.
- **Method:** SVD Construction ($Z = U \Sigma V^T$).
    1.  Generate random orthonormal matrices $U$ ($n \times q$) and $V$ ($q \times q$) using QR decomposition of random Gaussian matrices.
    2.  Construct singular values $\sigma$.
        -   If `anchor="min"`: $\sigma_{max} = 1$, $\sigma_{min} = 1/\kappa$.
        -   Values are geometrically spaced between $\sigma_{min}$ and $\sigma_{max}$.
    3.  Compute $Z = U \cdot \text{diag}(\sigma) \cdot V^T$.
    4.  Returns $Z$, singular values, and actual condition number.

### 2.2. Residuals and Weights ($R, W$)
**Function:** `generate_residuals_powerlaw(F_val, T_val, K)`
- **Goal:** Generate heteroscedastic residual variances and noisy surrogates.
- **Structure:** Data is organized into $F$ features and $T$ timepoints ($n = F \times T$).
- **True Variance ($R_{true}$):**
    -   $P_1(f, t) = f^{-\alpha_1} e^{-\beta_1 t}$
    -   $P_2(f, t) = f^{-\alpha_2} e^{-\beta_2 t}$
    -   $R_{true} = P_1 P_2$ (element-wise product).
    -   Parameters: $\alpha_1=0.5, \alpha_2=1.0, \beta_1=0.01, \beta_2=0.02$.
- **Observed Surrogates ($\hat{R}$):**
    -   Noise factors $\eta_1, \eta_2 \sim \chi^2(2K) / (2K)$.
    -   $\hat{P}_1 = P_1 \cdot \eta_1$, $\hat{P}_2 = P_2 \cdot \eta_2$.
    -   $\hat{R} = \hat{P}_1 \hat{P}_2$.
- **Weights:** $W_{hat} = 1 / \hat{R}$.

### 2.3. Grouping Structure
**Function:** `make_group_indices(L)`
-   Defines $L$ groups of random effects.
-   Group sizes increase linearly: $s_\ell = 2 \ell$ for $\ell=1 \dots L$.
-   Total random effects $q = \sum s_\ell$.
-   Returns a list of indices for each group.

## 3. Estimators

All estimators aim to estimate the variance components $C = (C_1, \dots, C_L)$, where $Var(u_\ell) = C_\ell I_{s_\ell}$.

### 3.1. Method of Moments (MoM)
**Function:** `est_mom_all(y, Z, W_hat, I_list)`
1.  **Solve Mixed Model Equations:**
    -   Compute $H = Z^T W_{hat} Z$.
    -   Compute $b = Z^T W_{hat} y$.
    -   Solve $H \hat{u} = b$ for $\hat{u}$ using Cholesky decomposition ($H = R_{chol}^T R_{chol}$).
2.  **Covariance Correction:**
    -   Compute $\text{diag}(\Sigma) = \text{diag}(H^{-1})$ using `chol2inv`.
3.  **Estimators:**
    -   **Naive:** $\hat{C}_\ell = \frac{1}{s_\ell} \sum_{j \in \text{group } \ell} \hat{u}_j^2$.
    -   **Unbiased:** $\hat{C}_\ell = \frac{1}{s_\ell} \sum_{j \in \text{group } \ell} (\hat{u}_j^2 - \Sigma_{jj})$.

### 3.2. Spectral Cutoff
**Function:** `est_mom_cutoff(u_hat, H, I_list, tau)`
1.  **Eigendecomposition:** Compute eigenvalues $\lambda_i$ and eigenvectors $v_i$ of $H$.
2.  **Truncation:**
    -   Eigenvalues of covariance $\Sigma = H^{-1}$ are $1/\lambda_i$.
    -   Keep components where covariance eigenvalue $< \tau \cdot \max(\text{cov eigenvalues})$. (Note: Code logic actually keeps *large* covariance eigenvalues? Check: `mask <- vals_Sigma >= (tau * lam_max)`. This keeps the largest variance directions, which corresponds to *small* eigenvalues of H (ill-conditioned directions). **Correction:** The code keeps components with *large* variance in $\Sigma$. Wait, usually cutoff removes small eigenvalues of $H$ (large variance). Let's verify code intent: usually we want to *remove* directions with excessive variance.
    -   *Code Logic Check:* `mask <- vals_Sigma >= (tau * lam_max)`. This keeps the high-variance components. If the goal is regularization, we usually want to *remove* the explosive variance components (small eigenvalues of H). This might be a "keep the signal" logic or a specific user definition. The current code keeps the large $\Sigma$ values.
3.  **Reconstruction:** $\Sigma_{cut} = V \cdot \text{diag}(\lambda_{cut}) \cdot V^T$.
4.  **Estimate:** Same as Unbiased MoM but using $\Sigma_{cut}$ for correction.

### 3.3. Fixed Ridge (FR)
**Function:** `est_fixed_ridge(y, Z, W_hat, I_list, sizes, lambda_vals)`
1.  **Regularization:**
    -   Construct penalty matrix $D = \text{diag}(d_j)$ where $d_j = \lambda_\ell / s_\ell$.
    -   $H_\lambda = Z^T W_{hat} Z + D$.
2.  **Solve:**
    -   Solve $H_\lambda \hat{u}_{FR} = Z^T W_{hat} y$.
3.  **Estimators:**
    -   **Plug-in:** $\hat{C}_\ell = \text{mean}(\hat{u}_{FR}^2)$.
    -   **Adjusted:** Subtract bias using $\Sigma_{FR} = H_\lambda^{-1} (Z^T W Z) H_\lambda^{-1}$.

### 3.4. EM-ML
**Function:** `est_em_ml(...)`
-   **Algorithm:** Expectation-Maximization for REML/ML.
-   **Initialization:** Start with $C_{init}$ (e.g., from Unbiased MoM).
-   **Iteration:**
    1.  Construct prior covariance $G = \text{diag}(C_{curr})$.
    2.  Compute posterior precision $A = H + G^{-1}$.
    3.  **E-step:**
        -   Posterior mean $m = A^{-1} b$.
        -   Posterior covariance $V = A^{-1}$.
    4.  **M-step:**
        -   Update $C_\ell^{new} = \frac{1}{s_\ell} \sum (m_j^2 + V_{jj})$.
    5.  Check convergence.

## 4. Simulation Loop
**Structure:**
1.  **Parameters:**
    -   $N_{sim} = 1000$.
    -   Grid: $\kappa \in \{1, 10, 100, 1000, 10000\}$, $K \in \{2, 10, 100\}$.
2.  **Outer Loop ($\kappa$):** Generate $Z$ once per $\kappa$.
3.  **Inner Loop ($K$):**
    -   **Replicates ($i = 1 \dots N_{sim}$):**
        1.  Generate true $u$ and residuals $e$.
        2.  Construct $y = Z u + e$.
        3.  Run all estimators (MoM, Cutoff, FR, EM).
        4.  Compute Oracle estimator (using true $u$).
    -   **Aggregation:** Compute Mean Squared Error (MSE) for each estimator against true $C$.
4.  **Output:** Print table of Average MSE for all combinations.

## 5. Helper Utilities
-   `chol_spd`: Robust Cholesky decomposition with diagonal jitter for numerical stability.
-   `solve_via_chol`: Solves linear systems using cached Cholesky factor.
-   `diag_inv_via_chol`: Efficiently computes diagonal of inverse matrix.
