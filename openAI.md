# Linear Mixed Model Simulation (No Fixed Effects, p=0)
**Purpose.** This document is a coding specification for implementing a simulation and estimation toolkit for a Gaussian linear mixed model with no fixed effects (\(p=0\)). It extracts the essential math, shapes, algorithms, and defaults from the source note while making requirements explicit for implementation.

---

## 1) Model, Shapes, and Notation

### Core model
- Observed data: \(y \in \mathbb{R}^n\)
- Design: \(Z \in \mathbb{R}^{n \times q}\)
- Random effects: \(u \in \mathbb{R}^q\)
- Noise: \(e \in \mathbb{R}^n\)

Model:
\[
y = Z u + e,\qquad
u \sim \mathcal{N}(0, G),\quad e \sim \mathcal{N}(0, R),\quad u \perp e.
\]

Residual covariance and weights:
\[
R = \mathrm{diag}(R_1,\dots,R_n),\qquad W := R^{-1} = \mathrm{diag}(w_1,\dots,w_n).
\]

### Grouped random effects
- Number of groups: \(L\).
- Group index sets: \(I_1,\dots,I_L \subset \{1,\dots,q\}\) are disjoint and cover \(\{1,\dots,q\}\).
- Group sizes: \(m_\ell := |I_\ell|\).
- Variance components (estimand): \(C = (C_1,\dots,C_L)^\top\) with \(C_\ell > 0\).

Random-effects covariance (diagonal within groups):
\[
G = \mathrm{diag}\big(\underbrace{C_1,\dots,C_1}_{m_1},\underbrace{C_2,\dots,C_2}_{m_2},\dots,\underbrace{C_L,\dots,C_L}_{m_L}\big).
\]

### Marginal distribution
\[
y \sim \mathcal{N}\!\big(0,\; Z G Z^\top + R\big).
\]

---

## 2) Design \(Z\) and Residual Variance \(R\)

### 2.1 Design with controlled conditioning
Construct \(Z\) via an SVD profile:
\[
Z = U\,\mathrm{diag}(s_1,\dots,s_q)\,V^\top,
\]
where \(U \in \mathbb{R}^{n\times q}\), \(V \in \mathbb{R}^{q\times q}\) are orthonormal and the singular values \((s_j)\) control \(\mathrm{cond}(Z)\approx\kappa\).
Profiles:
- **Geometric** or **linear** spacing of \((s_j)\).
Anchors:
- **min** anchor: fix \(s_{\max}=1\), shrink \(s_{\min}=1/\kappa\).
- **max** anchor: fix \(s_{\min}=1\), inflate \(s_{\max}=\kappa\).

Weighted conditioning to monitor:
\[
\kappa\!\left(W^{1/2}Z\right) := \frac{\sigma_{\max}(W^{1/2}Z)}{\sigma_{\min}(W^{1/2}Z)}.
\]

### 2.2 Residual variance via separable factors (simulation-friendly)
- Index \(i\) corresponds to pair \((f,t)\) with \(f=1{:}F\), \(t=1{:}T\), \(n=FT\).
- Construct \(R_{f,t} = P_1(f,t)\,P_2(f,t)\), where \(P_k(f,t) = h_k(f)s_k(t)\).
  - Example (defaults): \(h_1(f) = f^{-0.5}\), \(h_2(f)=f^{-1}\); \(s_1(t)=e^{-0.01t}\), \(s_2(t)=e^{-0.02t}\).
- Observed surrogates: \(\widehat{P}_k(f,t) = P_k(f,t)\,\eta_{k,f,t}\) with \(\eta_{k,f,t}\) approximately \(\chi^2_{2K_k}/(2K_k)\) and independent across \(k\) and \((f,t)\).
- Use \(\widehat{R}_{f,t} := \widehat{P}_1(f,t)\widehat{P}_2(f,t)\); set \(\widehat{W} := \widehat{R}^{-1}\).

---

## 3) Estimation Targets and Shapes

- Estimand: \(C=(C_1,\dots,C_L)^\top\).
- Group mapping supplied as either index sets \(I_\ell\) or a length-\(q\) vector `group_id` with entries in \(\{1,\dots,L\}\).

**Key matrices (do not form explicit inverses in code unless unavoidable):**
- \(H := Z^\top \widehat{W} Z \in \mathbb{R}^{q\times q}\) (SPD in typical cases).
- \(\widehat{\Sigma} := H^{-1}\) (avoid explicit inversion; use solves for \(H x\)=rhs).

---

## 4) Estimators (Formulas + What to Implement)

Below, \(\widehat{W}=\widehat{R}^{-1}\) is treated as plug-in fixed.

### S1. WLS estimator for \(u\)
\[
\widehat{u} = (Z^\top \widehat{W} Z)^{-1} Z^\top \widehat{W} y = H^{-1} Z^\top \widehat{W} y.
\]
- Implementation: compute \(b := Z^\top \widehat{W} y\). Solve \(H x = b\) for \(x=\widehat{u}\) via Cholesky (with a tiny ridge if needed).

Approximate covariance of \(\widehat{u}\) (conditional on \(u\) and fixed \(\widehat{W}\)):
\[
\widehat{\Sigma} := (Z^\top \widehat{W} Z)^{-1} = H^{-1}.
\]
- Implementation: **never** invert \(H\) explicitly. To access \(\mathrm{diag}(\widehat{\Sigma})\), prefer:
  - Solve \(H X = I\) for a small set of probe columns if needed,
  - or compute diagonals by Cholesky-based triangular solves,
  - or use Hutchinson/SLQ if \(q\) is large.

### S1.a. Naïve MoM (per group)
\[
\widehat{C}^{\text{naive}}_\ell = \frac{1}{m_\ell}\sum_{i\in I_\ell} \widehat{u}_i^{\,2}.
\]
**Bias:** upward by \(\bar{\Sigma}_\ell = \tfrac{1}{m_\ell}\sum_{i\in I_\ell} \widehat{\Sigma}_{ii}\).

### S1.b. Bias-corrected MoM
\[
\widetilde{C}_\ell = \frac{1}{m_\ell}\sum_{i\in I_\ell}\bigl(\widehat{u}_i^{\,2} - \widehat{\Sigma}_{ii}\bigr).
\]
- If \(\widehat{W}=W\) and GLS assumptions hold, this is unbiased for \(C_\ell\).

### S2. Spectral Cut-off MoM (stabilized)
Ill-conditioning of \(\widehat{\Sigma}=(Z^\top \widehat{W} Z)^{-1}\) can make \(\widehat{\Sigma}_{ii}\) unstable. Stabilize via truncation:
1) Eigendecompose \(\widehat{\Sigma} = Q \,\mathrm{diag}(\lambda_1,\dots,\lambda_q) Q^\top\) with \(\lambda_1\ge \cdots \ge \lambda_q>0\).
2) Threshold with relative cut \(\tau\) (default: \(\tau=10^{-3}\)):
\[
\lambda_j^{\text{cut}} := \lambda_j\,\mathbf{1}\{\lambda_j \ge \tau\,\lambda_1\}.
\]
3) Form \(\widehat{\Sigma}_{\text{cut}} := Q\,\mathrm{diag}(\lambda_1^{\text{cut}},\dots,\lambda_q^{\text{cut}})Q^\top\).

**Cut-off MoM estimator:**
\[
\widehat{C}^{\text{cut}}_\ell = \frac{1}{m_\ell}\sum_{i\in I_\ell}\!\Bigl(\widehat{u}_i^{\,2} - [\widehat{\Sigma}_{\text{cut}}]_{ii}\Bigr).
\]
- Notes: \(\widehat{\Sigma}_{\text{cut}}\preceq \widehat{\Sigma}\), so cut-off can be mildly downward-biased but numerically stable. As \(\tau\downarrow 0\), it reduces to S1.b.

### S3. Penalized fixed-effects (groupwise ridge)
Treat \(u\) as **fixed** parameters and penalize group energies
\[
C_\ell(u) := \frac{1}{m_\ell}\sum_{i\in I_\ell} u_i^2,\quad \ell=1,\dots,L.
\]
Optimization problem:
\[
\widehat{u}^{\,\mathrm{FR}}(\lambda) \in \arg\min_{u\in\mathbb{R}^q}
\Big\{\tfrac{1}{2}\|y-Zu\|_{\widehat{W}}^2 + \tfrac{1}{2}\sum_{\ell=1}^L \lambda_\ell\,C_\ell(u)\Big\},\quad
\|v\|_{\widehat{W}}^2 := v^\top \widehat{W} v.
\]
Define diagonal penalty
\[
D_\lambda := \mathrm{diag}(d_1,\dots,d_q),\quad d_i = \lambda_{\ell(i)}/m_{\ell(i)}.
\]
Closed-form solution (Cholesky solve):
\[
\widehat{u}^{\,\mathrm{FR}}(\lambda) = \big(Z^\top \widehat{W} Z + D_\lambda\big)^{-1} Z^\top \widehat{W} y.
\]

**Plug-in group energy:**
\[
\widehat{C}^{\,\mathrm{FR}}_\ell(\lambda) = \frac{1}{m_\ell}\sum_{i\in I_\ell} \big(\widehat{u}^{\,\mathrm{FR}}_i(\lambda)\big)^2.
\]

**Optional variance subtraction (still biased for shrinkage):**
- With \(H_\lambda := Z^\top \widehat{W} Z + D_\lambda\),
\[
\Sigma^{\mathrm{FR}}(\lambda) := \mathrm{Var}\!\big(\widehat{u}^{\,\mathrm{FR}}(\lambda)\mid u\big)
= H_\lambda^{-1}\, Z^\top \widehat{W}\,\widehat{R}\,\widehat{W}\, Z\, H_\lambda^{-1}.
\]
- If \(\widehat{W}=\widehat{R}^{-1}\) exactly: \(\Sigma^{\mathrm{FR}}(\lambda)=H_\lambda^{-1} (Z^\top \widehat{W}Z) H_\lambda^{-1}\).
- Adjusted version:
\[
\widetilde{C}^{\,\mathrm{FR}}_\ell(\lambda) = \frac{1}{m_\ell}\sum_{i\in I_\ell}\!\Big(\big(\widehat{u}^{\,\mathrm{FR}}_i(\lambda)\big)^2 - [\Sigma^{\mathrm{FR}}(\lambda)]_{ii}\Big).
\]

**Tuning \(\lambda_\ell\):**
- Cross-validation or generalized cross-validation (GCV) on weighted prediction error \(\|y - Z\widehat{u}^{\,\mathrm{FR}}(\lambda)\|_{\widehat{W}}^2\).
- Use groupwise grids; normalize by \(m_\ell\) (as above) to equalize per-coordinate strength.

### S4. EM–ML for \(C\) (integrating out \(u\))
Let \(C_{\text{rep}}\in\mathbb{R}^q\) repeat \(C_\ell\) across \(I_\ell\). Define
\[
A := G^{-1} + Z^\top \widehat{W} Z = \mathrm{diag}(C_{\text{rep}}^{-1}) + H,\quad
b := Z^\top \widehat{W} y.
\]
Given \(C^{(t)}\) (thus \(G^{(t)}\)), compute conditional moments:
\[
m := \mathbb{E}[u\mid y;C^{(t)}] = A^{-1}b,\qquad
V := \mathrm{Var}(u\mid y;C^{(t)}) = A^{-1}.
\]
M-step (groupwise):
\[
C^{(t+1)}_\ell = \frac{1}{m_\ell}\sum_{i\in I_\ell}\big(m_i^2 + V_{ii}\big),\quad \ell=1,\dots,L.
\]
Stop when \(\max_\ell |(C^{(t+1)}_\ell - C^{(t)}_\ell)/C^{(t)}_\ell| < \texttt{tol}\) or \(t=\texttt{maxit}\), with floor \(C^{(t)}_\ell\ge\varepsilon\).

---

## 5) APIs the Agent Should Implement

Below are language-agnostic function signatures; adapt types to your language of choice.

### Data preparation
- `make_weights(P1_hat: array[F,T], P2_hat: array[F,T]) -> W_hat: diag(n)`
  - Returns \( \widehat{W} = \widehat{R}^{-1}\) with \( \widehat{R}_{f,t} = \widehat{P}_1(f,t)\widehat{P}_2(f,t)\).

- `group_map(I_list: list[indices], q: int) -> group_id: int[q], m: int[L]`

### Linear algebra cores
- `solve_H(H: spd[q,q], b: q) -> u_hat: q`  (Cholesky; add small ridge if needed)
- `diag_of_Hinv(H: spd[q,q]) -> diagSigma: q` (no explicit inverse; use triangular solves)
- `eig_truncate_Sigma(Sigma_diag_eig: (Q, lambdas), tau=1e-3) -> Sigma_cut`

### Estimators
- `est_u_hat(Z, W_hat, y) -> u_hat, H`
- `est_mom_naive(u_hat, group_id, L) -> C_hat_naive[L]`
- `est_mom_unbias(u_hat, diagSigma, group_id, L) -> C_tilde[L]`
- `est_mom_cut(u_hat, Sigma_cut, group_id, L, tau=1e-3) -> C_cut[L]`
- `est_fixedridge_u(Z, W_hat, y, group_id, m, lambda_group[L]) -> u_FR, H_lambda`
- `est_fixedridge_C(u_FR, group_id, L) -> C_FR[L]`
- `est_fixedridge_C_adj(u_FR, H_lambda, H, group_id, L, use_exact_W_equals_Rinv: bool) -> C_FR_tilde[L]`
- `est_em_ml(Z, W_hat, y, group_id, m, C_init[L], tol=1e-8, maxit=500, floor=1e-10) -> C_EM[L]`

### Evaluation (optional)
- `mse_by_group(C_est[L], C_true[L]) -> mse[L]`
- `mse_average(mse[L]) -> float`

---

## 6) Pseudocode Sketches

### S1/S1.b WLS and bias-corrected MoM
```
b = Z.T @ W_hat @ y
H = Z.T @ W_hat @ Z
u_hat = chol_solve(H, b)

# diag(Sigma) without explicit inverse
diagSigma = diag_of_Hinv(H)          # via Cholesky solves

# groupwise aggregations
for ell in 1..L:
    idx = I_ell
    C_naive[ell] = mean(u_hat[idx]**2)
    C_unbias[ell] = mean(u_hat[idx]**2 - diagSigma[idx])
```

### S2 Spectral cut-off MoM
```
# If q is moderate: eigendecompose Sigma via Cholesky of H
# Option A: compute Q, lambdas from Sigma; Option B: eig H and map.
Q, lambdas = eigh_of_Sigma(H)        # conceptual; implement via H if preferred
lam_max = max(lambdas)
mask = (lambdas >= tau * lam_max)
Sigma_cut = Q @ diag(lambdas * mask) @ Q.T

for ell in 1..L:
    idx = I_ell
    C_cut[ell] = mean(u_hat[idx]**2 - diag(Sigma_cut)[idx])
```

### S3 Penalized fixed-effects (groupwise ridge)
```
# Build penalty
d = zeros(q)
for i in 1..q:
    ell = group_id[i]
    d[i] = lambda_group[ell] / m[ell]
D_lambda = diag(d)

H_lambda = H + D_lambda
u_FR = chol_solve(H_lambda, b)

for ell in 1..L:
    idx = I_ell
    C_FR[ell] = mean(u_FR[idx]**2)

# Optional adjusted version
if use_exact_W_equals_Rinv:
    Sigma_FR = inv(H_lambda) @ H @ inv(H_lambda)  # implement via two solves
else:
    Lmat = inv(H_lambda) @ (Z.T @ W_hat)         # implement via solves
    Sigma_FR = Lmat @ R_hat @ Lmat.T

for ell in 1..L:
    idx = I_ell
    C_FR_tilde[ell] = mean(u_FR[idx]**2 - diag(Sigma_FR)[idx])
```

### S4 EM–ML (variance components)
```
C = C_init
repeat until convergence or maxit:
    # Form A = diag(C_rep^-1) + H
    C_rep = repeat_by_group(C, m)      # length q
    A = diag(1.0 / C_rep) + H

    # E-step: m = A^{-1} b, V = A^{-1}
    mvec = chol_solve(A, b)
    diagV = diag_of_Hinv(A)

    # M-step: groupwise update
    for ell in 1..L:
        idx = I_ell
        C_new[ell] = mean(mvec[idx]**2 + diagV[idx])
        C_new[ell] = max(C_new[ell], floor)

    # check convergence
    if max_abs((C_new - C) / C) < tol: break
    C = C_new
```

---

## 7) Numerics & Stability Checklist

- Prefer **Cholesky** solves for SPD systems; inject a tiny ridge \( \delta I\) if needed (e.g., \(\delta=10^{-12}\)–\(10^{-8}\)).
- Avoid explicit matrix inverses; compute diagonals of inverses via triangular solves or stochastic trace estimators when \(q\) is large.
- In spectral cut-off, use relative threshold \(\tau=10^{-3}\) by default.
- Keep group-wise penalty normalized by \(1/m_\ell\) so penalty strength is per-coordinate.
- Enforce nonnegativity floors for EM updates: \(C_\ell \ge \varepsilon\).

---

## 8) Defaults

- \(n=1000\); \(L=10\); \(m_\ell=2\ell\) so \(q=\sum_\ell m_\ell\).
- True components (for simulation): \(C_\ell = 0.1\,\ell\).
- Residual: power-law baseline with separable trends as in §2.2; \(F=10\), \(T=n/F\).
- Design singular values: geometric profile; anchor = “min” (controls \(\kappa\)).
- Monte Carlo runs: \(N=1000\); redraw \(u\) each replicate.
- EM: init from bias-corrected MoM with floor \(10^{-10}\); `tol=1e-8`, `maxit=500`.
- Spectral cut-off: \(\tau=10^{-3}\).
- Linear algebra: Cholesky with small ridge if needed.

---

## 9) Inputs/Outputs Summary

**Inputs.**
- Arrays: \(y\) (n,), \(Z\) (n×q), \(\widehat{P}_1\) (F×T), \(\widehat{P}_2\) (F×T), groups \(I_\ell\) or `group_id`.
- Hyperparameters: \(\tau\) (cut-off), \(\lambda_\ell\) (ridge), EM tolerances, SVD/conditioning knobs.

**Outputs.**
- \( \widehat{u}\), \( \widehat{C}^{\text{naive}}\), \( \widetilde{C}\), \( \widehat{C}^{\text{cut}}\),
- \( \widehat{u}^{\,\mathrm{FR}}\), \( \widehat{C}^{\,\mathrm{FR}}\), \( \widetilde{C}^{\,\mathrm{FR}}\),
- \( C^{\mathrm{EM}}\).
- Optional diagnostics: condition numbers, Cholesky info, convergence traces.

---

## 10) Glossary (symbols)
- \(n\): sample size; \(q\): number of random-effect coordinates.
- \(L\): number of groups; \(I_\ell\): indices of group \(\ell\); \(m_\ell\): group size.
- \(C_\ell\): variance component for group \(\ell\); \(G\): diag covariance of \(u\).
- \(R\): residual covariance (diag); \(W=R^{-1}\): weights.
- \(\widehat{R}\), \(\widehat{W}\): plug-in estimates from \(\widehat{P}_1,\widehat{P}_2\).
- \(H = Z^\top \widehat{W} Z\); \(\widehat{\Sigma} = H^{-1}\).
- \(\tau\): spectral truncation threshold.
- \(\lambda_\ell\): groupwise ridge penalty.
- \(A\): EM system matrix \(\mathrm{diag}(C_{\text{rep}}^{-1}) + H\).

---

## 11) References (for human readers)
- Dempster, Laird, Rubin (1977): EM algorithm.
- Laird, Ware (1982): Random-effects for longitudinal data.
- Searle, Casella, McCulloch (1992): Variance components.
- Petersen, Pedersen (2012): Matrix Cookbook.
- Golub, Van Loan (2013): Matrix Computations.

