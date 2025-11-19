# Linear Mixed Model Simulation (No Fixed Effects, $p=0$)
**Design, Estimators, and Monte Carlo Comparison**

## 1. Model and Notation

Let $n \in \mathbb{N}$ be the sample size and $q \in \mathbb{N}$ be the number of random-effect coordinates.

### Observations and Distributions
We observe $y \in \mathbb{R}^n$:
$$y = Zu + e$$
where $Z \in \mathbb{R}^{n \times q}$ is the design matrix. The vectors $u$ (random effects) and $e$ (error) satisfy:
$$u \sim \mathcal{N}(0, G), \qquad e \sim \mathcal{N}(0, R), \qquad u \perp e$$

The residual covariance is diagonal:
$$R = \text{diag}(R_1, \dots, R_n), \qquad W := R^{-1} = \text{diag}(w_1, \dots, w_n)$$

### Group Structure
The vector $u$ is partitioned into $L$ groups via disjoint index sets $I_1, \dots, I_L$ such that $\bigcup I_\ell = \{1, \dots, q\}$.
* Let $m_\ell = |I_\ell|$.
* Let $C = (C_1, \dots, C_L)^\top$ be the groupwise variance components ($C_\ell > 0$).

The covariance $G$ is block-diagonal:
$$G = \text{diag}(\underbrace{C_1, \dots, C_1}_{m_1}, \dots, \underbrace{C_L, \dots, C_L}_{m_L})$$

### Marginal Law
Marginally, $y \sim \mathcal{N}(0, ZGZ^\top + R)$. The estimand is the vector $C$.

---

## 2. Design and Residual Variance Structures

### 2.1 Design Matrix ($Z$)
$Z$ is constructed via SVD with controlled conditioning:
$$Z = U \, \text{diag}(s_1, \dots, s_q) \, V^\top$$
* $U, V$: Random orthonormal matrices.
* $s_j$: Singular values following a **Geometric** or **Linear** profile.
* **Condition Number:** $\text{cond}(Z) \approx \kappa$.
* **Anchor:**
    * `min`: Fixes $s_{\max}=1$, shrinks $s_{\min} = 1/\kappa$.
    * `max`: Fixes $s_{\min}=1$, inflates $s_{\max} = \kappa$.

### 2.2 Residual Variance Families (Simulation Specifics)
The index $i$ corresponds to a double index $(t, f)$.
* $F = 10$, $T = n/F$.
* $t = 1, \dots, T$ and $f = 1, \dots, F$.

**True Structure:**
$$R_{f,t} = P_1(f,t)P_2(f,t)$$
where $P_k(f,t) = h_k(f)s_k(t)$ for $k=1,2$.
* $h_k(f) = f^{-\alpha_k}$
* $s_k(t) = \exp(-\beta_k t)$
* **Parameters:** $\alpha_1 = 0.5, \alpha_2 = 1, \beta_1 = 0.01, \beta_2 = 0.02$.

**Observed Estimates (Inputs for Estimation):**
We do not observe $R$. We observe independent estimates $\widehat{P}_1$ and $\widehat{P}_2$.
$$\widehat{P}_k(f,t) = P_k(f,t) \eta_{k,f,t}$$
* $\eta_{1,f,t} \sim \chi^2_{2K_1} / (2K_1)$
* $\eta_{2,f,t} \sim \chi^2_{2K_2} / (2K_2)$
* Assume $\eta$ terms are independent.

**Operational Inputs:**
Define inputs for the estimators as:
$$\widehat{R}_{f,t} = \widehat{P}_1(f,t) \widehat{P}_2(f,t)$$
$$\widehat{W} = \widehat{R}^{-1}$$

---

## 3. Estimators of Variance Components ($C$)

### 3.1 Setting 1: WLS for $u$ and Method of Moments (MoM)

**Base Calculations:**
1.  **WLS Estimator:** $\widehat{u} = (Z^\top \widehat{W} Z)^{-1} Z^\top \widehat{W} y$
2.  **Estimated Variance:** $\widehat{\Sigma} = (Z^\top \widehat{W} Z)^{-1}$

#### Estimator A: Naive MoM (`MoM_naive`)
$$\widehat{C}^{\text{naive}}_\ell = \frac{1}{m_\ell} \sum_{i \in I_\ell} \widehat{u}_i^2$$

#### Estimator B: Unbiased MoM (`MoM_unbias`)
Subtracts the noise variance bias:
$$\widetilde{C}_\ell = \frac{1}{m_\ell} \sum_{i \in I_\ell} (\widehat{u}_i^2 - \widehat{\Sigma}_{ii})$$

#### Estimator C: Spectral Cut-off MoM (`MoM_cutoff`)
Addresses instability in $\widehat{\Sigma}$ due to poor condition numbers.
1.  Compute eigendecomposition: $\widehat{\Sigma} = Q \Lambda Q^\top$.
2.  Threshold eigenvalues:
    $$\lambda_j^{\text{cut}} = \begin{cases} \lambda_j & \text{if } \lambda_j \ge \frac{1}{1000}\max_k(\lambda_k) \\ 0 & \text{otherwise} \end{cases}$$
3.  Reconstruct: $\widehat{\Sigma}_{\text{cut}} = Q \Lambda^{\text{cut}} Q^\top$.
4.  Calculate estimator:
    $$\widehat{C}^{\text{cut}}_\ell = \frac{1}{m_\ell} \sum_{i \in I_\ell} (\widehat{u}_i^2 - (\widehat{\Sigma}_{\text{cut}})_{ii})$$

---

### 3.2 Setting 2: Penalized Fixed Effects (Groupwise Ridge)

Treats $u$ as fixed parameters with a groupwise penalty.
* **Penalty Matrix:** $D_\lambda = \text{diag}(d_1, \dots, d_q)$ where $d_i = \frac{\lambda_{\ell(i)}}{m_{\ell(i)}}$.
* **Ridge Solution:** $\widehat{u}^{\text{FR}}(\lambda) = (Z^\top \widehat{W} Z + D_\lambda)^{-1} Z^\top \widehat{W} y$.

#### Estimator D: Fixed Ridge Plug-in (`FixedRidge_plug`)
$$\widehat{C}^{\text{FR}}_\ell(\lambda) = \frac{1}{m_\ell} \sum_{i \in I_\ell} (\widehat{u}^{\text{FR}}_i(\lambda))^2$$

#### Estimator E: Fixed Ridge Adjusted (`FixedRidge_adj`)
Subtracts noise variance (but leaves shrinkage bias).
1.  Let $H_\lambda = Z^\top \widehat{W} Z + D_\lambda$.
2.  Variance map: $\Sigma^{\text{FR}}(\lambda) = H_\lambda^{-1} Z^\top \widehat{W} Z H_\lambda^{-1}$.
3.  Estimator:
    $$\widetilde{C}^{\text{FR}}_\ell(\lambda) = \frac{1}{m_\ell} \sum_{i \in I_\ell} \left( (\widehat{u}^{\text{FR}}_i(\lambda))^2 - [\Sigma^{\text{FR}}(\lambda)]_{ii} \right)$$

---

### 3.3 Setting 3: EM-ML for $C$

Maximizes marginal log-likelihood via Expectation-Maximization.

**Initialization:**
* $C^{(0)}$ derived from Bias-corrected MoM (floored at $10^{-10}$).

**Iteration:**
1.  **Setup:**
    $$A = \text{diag}(C_{\text{rep}}^{-1}) + Z^\top \widehat{W} Z$$
    $$b = Z^\top \widehat{W} y$$
2.  **E-Step (Posterior Moments):**
    $$m = A^{-1} b$$
    $$V = A^{-1}$$
3.  **M-Step (Update $C$):**
    $$C_{\ell}^{(t+1)} = \frac{1}{m_\ell} \sum_{i \in I_\ell} (m_i^2 + V_{ii})$$

**Convergence:**
Stop when $\max_\ell | (C_\ell^{(t+1)} - C_\ell^{(t)}) / C_\ell^{(t)} | < 10^{-8}$ or $t=500$.

---

## 4. Summary of Defaults

| Quantity | Default Value |
| :--- | :--- |
| **$n$** | 1000 |
| **$L$** (Groups) | 10 |
| **$m_\ell$** | $2\ell$ for $\ell=1,\dots,L$ |
| **$C_\ell$** (Truth) | $0.1 \times \ell$ |
| **Residuals** | Power law inputs (see Sec 2.2) |
| **Design SVD** | Geometric profile; anchor = "min" |
| **Target Cond ($\kappa$)** | User-chosen |
| **Monte Carlo ($N$)** | 1000 replicates |
| **Numerical Solves** | Cholesky with small diagonal ridge if needed |
