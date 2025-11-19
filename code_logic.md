# code_logic.md — Logic & Reconstruction Guide for `simulation_study.R`

**Goal.** This document explains the structure and logic of the provided R script so that:
1) a coding agent can **rebuild the code from scratch (even in another language)**, and  
2) a human reviewer can **spot logic issues quickly**.

The script implements a simulation study for a Gaussian **Linear Mixed Model (LMM) without fixed effects** and compares several estimators of groupwise variance components \(C=(C_1,\ldots,C_L)\). Core building blocks are: grouping utilities, numerical solvers, design construction, residual generator, four estimators (MoM, Cut‑off MoM, Fixed‑Ridge, EM‑ML), and a Monte‑Carlo grid driver.

---

## 0) High‑level Model (what the code assumes)

- **Data model:** \(y = Z u + e\), with \(u \sim \mathcal{N}(0,G)\), \(e \sim \mathcal{N}(0,R)\), \(u \perp e\).
- **Weights:** \(W=R^{-1}\) (diagonal). Estimation uses \(\widehat W\) formed from surrogate residual variances.
- **Groups:** \(q\) coordinates of \(u\) are partitioned into \(L\) disjoint groups \(I_\ell\) of sizes \(m_\ell\). The target \(C_\ell\) is the groupwise variance component.

---

## 1) Function Index (what each function does)

1. **`make_group_indices(L=NULL, group_sizes=NULL)`** → list of index vectors `I_list` and attribute `"sizes"`.
2. **`repeat_C(C, group_sizes)`** → repeat \(C_\ell\) according to `group_sizes` to get length‑\(q\) vector \(C_{\text{rep}}\).
3. **`chol_spd(M, tol=1e-12, max_try=6)`** → robust Cholesky of SPD matrix \(M\), adding a tiny **ridge** if needed.
4. **`solve_via_chol(Rchol, b)`** → solve \(A x = b\) with \(Rchol = \text{chol}(A)\).
5. **`diag_inv_via_chol(Rchol)`** → diagonal of \(A^{-1}\) from Cholesky (uses `chol2inv` in the script).
6. **`make_Z_svd(n, q, kappa=100, seed=123, anchor=c("min","max"))`** → construct \(Z\) with controlled singular values.
7. **`generate_residuals_powerlaw(F_val, T_val, K=10, seed=NULL)`** → build true/estimated residual variances and corresponding weights \(W\), \(\widehat W\).
8. **`est_mom_all(y, Z, W_hat, I_list)`** → GLS for \(\widehat u\); compute **Naïve MoM** and **Bias‑corrected MoM** per group.
9. **`est_mom_cutoff(u_hat, H, I_list, tau=1e-3)`** → **Spectral Cut‑off MoM** using truncated spectrum of \(\widehat\Sigma\).
10. **`est_fixed_ridge(y, Z, W_hat, I_list, sizes, lambda_vals)`** → **Fixed‑effects ridge**: plug‑in and adjusted variants.
11. **`est_em_ml(y, Z, W_hat, I_list, sizes, init_C, tol=1e-8, maxit=500, eps=1e-10)`** → **EM–ML** for \(C\).
12. **Main block** (guarded by `if (sys.nframe() == 0)`) → Monte‑Carlo grid driver, MSE computation, results aggregation/print.

---

## 2) Data Shapes and Invariants

- `n` (int): sample size (default 1000).  
- `L` (int): number of groups (default 10).  
- `sizes` (length‑`L`): group sizes \(m_\ell=2\ell\) (default), hence `q <- sum(sizes)` (with defaults, `q = 110`).  
- `I_list`: list where `I_list[[ell]]` are the indices in group \(\ell\); `attr(I_list, "sizes") == sizes`.
- `Z` (n×q), `y` (n×1), `u` (q×1), `e` (n×1).  
- `W_hat` (n×1 vector) is **diagonal weights**; code multiplies rowwise by `sqrt(W_hat)`.

**Invariants**
- `length(unlist(I_list)) == q`.  
- All SPD matrices passed to Cholesky must be symmetric and (numerically) positive definite.

---

## 3) Grouping Utilities

### `make_group_indices(L=NULL, group_sizes=NULL)`
- If `group_sizes` not provided: builds default sizes \(m_\ell = 2\ell\), \(\ell=1..L\).
- Allocates contiguous integer indices 1..q and splits by group label.
- Returns `I_list` with an attribute `"sizes"` so downstream code can recover `sizes`.

### `repeat_C(C, group_sizes)`
- Validates length consistency; returns \(C_{\text{rep}}\) by repeating each \(C_\ell\) `group_sizes[ell]` times.
- Used by EM to build \(\operatorname{diag}(C_{\text{rep}}^{-1})\) and to simulate `u` from \(N(0,G)\).

---

## 4) Numerical Helpers

### `chol_spd(M, tol=1e-12, max_try=6)`
- Attempts `chol(M + tau I)` with an increasing ridge \(\tau := (10^k)\, \texttt{tol}\,\text{mean}(\text{diag}(M))\), \(k=0,\ldots\).
- Returns the upper‑triangular factor with attributes:
  - `"tau"`: ridge actually used,
  - `"tries"`: how many attempts were needed.
- **Usage**: stabilizes Cholesky for ill‑conditioned matrices (e.g., \(H = Z^\top \widehat W Z\) or \(A=H+\operatorname{diag}(1/C_{\text{rep}})\)).

> **Note.** The current implementation *always* tries \(k=0\) first (nonzero ridge). For strict “no ridge unless needed,” try `chol(M)` once before adding ridge.

### `solve_via_chol(Rchol, b)`
- Solves \(R^\top R x = b\) via forward/back substitution.

### `diag_inv_via_chol(Rchol)`
- Returns `diag(chol2inv(Rchol))` (full inverse formed then diagonal extracted).  
- **Alternative (scale‑up)**: compute diagonal via triangular solves without forming the dense inverse:
  - If \(R\) is the Cholesky factor, `diag(A^{-1}) = colSums( backsolve(R, I)^2 )`.

---

## 5) Design Construction

### `make_Z_svd(n, q, kappa=100, seed=123, anchor=c("min","max"))`
- Builds random orthonormal `U` (n×q) and `V` (q×q) via QR of Gaussian matrices.
- **Singular values \(s\)** (geometric):
  - `anchor="min"`: `s_max=1`, `s_min=1/kappa`.
  - `anchor="max"`: `s_min=1`, `s_max=kappa`.
- Returns `Z = U diag(s) V'`, the singular values `sv <- svd(Z)$d`, and `cond <- max(sv)/min(sv)`.

---

## 6) Residual Generator (Power‑law, separable)

### `generate_residuals_powerlaw(F_val, T_val, K=10, seed=NULL)`
- Constructs grids: `f=1..F_val`, `t=1..T_val` and evaluates
  - \(h_1(f)=f^{-0.5},\, h_2(f)=f^{-1}\),
  - \(s_1(t)=e^{-0.01t},\, s_2(t)=e^{-0.02t}\).
- True components: \(P_1=h_1 s_1\), \(P_2=h_2 s_2\), hence \(R_{\text{true}}=P_1 P_2\), \(W_{\text{true}}=1/R_{\text{true}}\).
- **Observed surrogates** (Welch): \(\widehat P_k = P_k \cdot \eta_k\) with \(\eta_k \sim \chi^2_{2K}/(2K)\) (independent).  
  → \(\widehat R = \widehat P_1 \widehat P_2\), \(\widehat W=1/\widehat R\).

> **Note.** The function uses a **single** `K` for both surrogates. If you need \(K_1, K_2\) per spec, extend the signature and draw `eta1`, `eta2` with separate DOFs.

---

## 7) Estimators

### S1 — WLS for \( \widehat u \) and MoM for \(C\): `est_mom_all(y, Z, W_hat, I_list)`
**Goal.** Compute GLS estimate \(\widehat u\) and groupwise MoM variants.

**Steps.**
1. `Zw <- Z * sqrt(W_hat)` (row‑wise scaling by \(\sqrt{\widehat W}\)).
2. `H <- crossprod(Zw, Zw)` equals \(Z^\top \widehat W Z\).
3. `b <- crossprod(Z, W_hat * y)` equals \(Z^\top \widehat W y\).
4. Solve \(H \widehat u = b\) via `chol_spd(H)` and `solve_via_chol`.
5. `diagSigma <- diag_inv_via_chol(Rchol)` as \(\text{diag}(H^{-1})\).
6. For each group \(I_\ell\):
   - `C_naive[ell] <- mean(u_hat[idx]^2)`.
   - `C_unbias[ell] <- mean(u_hat[idx]^2 - diagSigma[idx])`.

**Returns.** `u_hat`, `H`, `Rchol`, `C_naive`, `C_unbias`.

---

### S2 — Spectral Cut‑off MoM: `est_mom_cutoff(u_hat, H, I_list, tau=1e-3)`
**Goal.** Stabilize the bias‑correction term by truncating small eigenvalues of \(\widehat\Sigma = H^{-1}\).

**Steps.**
1. `eigH <- eigen(H, symmetric=TRUE)` → eigenvalues `vals_H` = \(\mu_j\), eigenvectors `vecs`.
2. Convert to covariance eigenvalues: `vals_Sigma <- 1 / vals_H` = \(\lambda_j\).
3. Threshold (relative to largest \(\lambda_{\max}\)):  
   `mask <- vals_Sigma >= (tau * lam_max)`; `vals_cut <- vals_Sigma * mask`.
4. Build truncated covariance: `Sigma_cut <- vecs %*% diag(vals_cut) %*% t(vecs)`.
5. `diagSigma_cut <- diag(Sigma_cut)`.
6. For each group: `C_cut[ell] <- mean(u_hat[idx]^2 - diagSigma_cut[idx])`.

**Important behavior.**
- If **no eigenvalues** fall below the threshold, then `Sigma_cut == Sigma` and `C_cut == C_unbias`.
- The function **does not recompute \(\widehat u\)** under truncation; it only modifies the correction term.

---

### S3 — Fixed‑effects Ridge: `est_fixed_ridge(y, Z, W_hat, I_list, sizes, lambda_vals)`
**Goal.** Treat \(u\) as fixed and estimate with groupwise ridge penalty \(\sum_\ell \lambda_\ell \frac{1}{m_\ell}\sum_{i\in I_\ell}u_i^2\).

**Steps.**
1. Build penalty diagonal `D` where `D[ii] = lambda_vals[group(i)] / m[group(i)]`.
2. `H_base <- Z' W Z` via `Zw <- Z*sqrt(W_hat)` then `crossprod(Zw, Zw)`.
3. `H_lam <- H_base + D`; `b <- Z' W y`.
4. Solve \(H_{\lambda} u_{FR} = b\) via Cholesky.
5. **Plug‑in:** `C_plug[ell] <- mean(u_FR[idx]^2)`.
6. **Adjusted:** compute `Sigma_FR = H_lam_inv %*% H_base %*% H_lam_inv` and subtract diagonals:  
   `C_adj[ell] <- mean(u_FR[idx]^2 - diag(Sigma_FR)[idx])`.

**Notes.**
- `H_lam_inv` is formed via `chol2inv(Rchol)` (dense); acceptable for moderate `q` (≈110). For large `q`, prefer triangular‑solve formulas.

---

### S4 — EM–ML for \(C\): `est_em_ml(y, Z, W_hat, I_list, sizes, init_C, tol, maxit, eps)`
**Goal.** Maximize the marginal likelihood by integrating out \(u\) using EM.

**Setup.**
- Precompute `H <- Z' W Z` and `b <- Z' W y` once.
- Initialize \(C^{(0)} = \max(\text{init\_C}, \varepsilon)\).

**Iteration (for `iter` in 1..`maxit`).**
1. `C_rep <- repeat_C(C_curr, sizes)`.
2. Form `A <- H` then add `diag(A) <- diag(A) + 1 / C_rep` (i.e., \(A = H + \operatorname{diag}(C_{\text{rep}}^{-1})\)).
3. Cholesky `A`; **E‑step**:  
   `m_vec <- solve_via_chol(Rchol, b)` (i.e., \(m = A^{-1}b\));  
   `diagV <- diag(chol2inv(Rchol))` (i.e., \(\text{diag}(A^{-1})\)).
4. **M‑step:** for each group \(I_\ell\),  
   `C_new[ell] <- mean(m_vec[idx]^2 + diagV[idx])`; then `pmax(C_new, eps)`.
5. Convergence check: `rel_diff <- max(abs((C_new - C_curr) / C_curr))`; stop if `< tol`.

**Returns.** Final `C_curr`.

> **Defaults in function** are `tol=1e-8`, `maxit=500`, `eps=1e-10`.  
> **In the main loop** the call uses `tol=1e-5` and `maxit=50` (stricter speed trade‑off). If you want to match spec defaults exactly, call `est_em_ml` without overriding these parameters.

---

## 8) Main Monte‑Carlo Driver (grid + replicates)

**Constants**
- `N_SIM` (replicates per grid point): set to **1000** in this script.
- `n=1000`, `L=10`, `F_val=10`, `T_val=n/F_val` (so `n = F * T`), `q = sum(sizes)`.
- **Grid:** `kappa_vals <- c(1, 10, 100, 1000, 10000)` (design conditioning),
  `K_vals <- c(2, 10, 100)` (residual Welch DOFs).

**Per grid point (`kappa`, `K`):**
1. Build `Z` via `make_Z_svd` with `anchor="min"`.
2. Preallocate result matrices `res_*` each with dimensions `N_SIM × L`.
3. For each replicate `i`:
   - Draw `u ~ N(0, diag(repeat_C(C_true, sizes)))`.
   - Generate residual structure with `generate_residuals_powerlaw(F_val, T_val, K)`.
   - Draw `e ~ N(0, R_true)` elementwise; set `y <- Z %*% u + e`.
   - Set `W_hat <- res_data$W_hat` (plug‑in weights).
   - **Estimators:**
     * `mom <- est_mom_all(...)` → store `C_naive`, `C_unbias`.
     * `res_cut[i, ] <- est_mom_cutoff(mom$u_hat, mom$H, I_list, tau=1e-3)`.
     * `fr <- est_fixed_ridge(...)` → store `C_plug`, `C_adj`.
     * `C_init <- pmax(mom$C_unbias, 1e-5)`;  
       `res_em[i, ] <- est_em_ml(..., tol=1e-5, maxit=50)` (speed‑oriented overrides).
     * `res_oracle[i, ] <- mean(u[idx]^2)` per group (oracle baseline).
4. After `N_SIM` reps, compute **Average MSE** over groups:  
   `mse_calc <- function(est, truth) mean(colMeans( (est - truth)^2 ))`.
5. Append one row to `results_grid` for this (`kappa`, `K`).

**Output**
- Prints a table of averaged MSEs: Naive, Unbias, Cutoff, FR_Plug, FR_Adj, EM, Oracle.

---

## 9) Logic Issues & Diagnostics (what to check / how to fix)

1) **Cut‑off MoM equals Bias‑corrected MoM.**  
   - In `est_mom_cutoff`, if **no eigenvalues** fall below the threshold (`tau=1e-3`), then `Sigma_cut == Sigma` and `C_cut == C_unbias` by construction.  
   - **Diagnostic**: print number of dropped modes and `cond(H)`:
     ```r
     eigH <- eigen(H, symmetric=TRUE)$values
     condH <- max(eigH) / min(eigH)
     vals_Sigma <- 1 / eigH
     lam_max <- max(vals_Sigma)
     dropped <- sum(vals_Sigma < (tau * lam_max))
     cat(sprintf("cond(H)=%.3g, dropped=%d (tau=%.1e)\n", condH, dropped, tau))
     ```
   - **If you want a difference**: increase `tau` (e.g., `1e-2`), or truncate on `H` and **recompute** \(\hat u\) under truncation.

2) **Ridge consistency (minor).**  
   - `est_mom_all` may add a tiny ridge via `chol_spd(H)`, while `est_mom_cutoff` eigendecomposes the **unridged** `H`. This typically changes results only at machine‑precision. For perfect apples‑to‑apples, either:
     - attempt Cholesky **without** ridge first and add ridge only on failure, or
     - if a ridge `tau` was used, add the same ridge when forming `Sigma_cut` (eigendecompose `H + tau I`).

3) **Full inverse formation for diagonals.**  
   - `diag_inv_via_chol` and the EM step use `chol2inv` (dense inverse). For larger `q`, prefer triangular‑solve approaches to get diagonals of \(A^{-1}\) without materializing \(A^{-1}\).

4) **Welch DOFs coupling.**  
   - `generate_residuals_powerlaw` uses a single `K` for both \(\widehat P_1\) and \(\widehat P_2\). If you require separate \((K_1, K_2)\), extend the function signature and draws accordingly.

5) **EM defaults in main loop.**  
   - Function defaults are `tol=1e-8`, `maxit=500`, `eps=1e-10`; main loop overrides to `tol=1e-5`, `maxit=50` (faster, potentially rougher). Align with your desired default profile.

6) **Condition number monitoring (optional).**  
   - To track the *weighted* conditioning emphasized by the theory, log `kappa(W^{1/2}Z)`:
     ```r
     weighted_sv <- svd(sweep(Z, 1, sqrt(W_hat), "*"), nu=0, nv=0)$d
     kappa_w <- max(weighted_sv) / min(weighted_sv)
     ```

---

## 10) Recovery Blueprint (language‑agnostic)

Implement the following modules in order; unit‑test each before the MC driver:

1. **Grouping**: `make_group_indices`, `repeat_C`  
2. **Numerics**: `chol_spd`, `solve_via_chol`, `diag_inv_via_chol` (or triangular‑solve variant)
3. **Design**: `make_Z_svd`
4. **Residuals**: `generate_residuals_powerlaw`
5. **Estimators**: `est_mom_all`, `est_mom_cutoff`, `est_fixed_ridge`, `est_em_ml`
6. **Driver**: Monte‑Carlo grid, replicates, MSE aggregation

**Key equations to reproduce** (for reference):
- \(H = Z^\top \widehat W Z\), \(b = Z^\top \widehat W y\).
- \(\widehat u = H^{-1} b\) (solve via Cholesky).
- \(\widehat\Sigma = H^{-1}\); Spectral cut‑off via truncating small eigenvalues of \(\widehat\Sigma\).
- Fixed‑ridge: \(u_{FR} = (H + D_\lambda)^{-1} b\), \(\Sigma_{FR} = (H + D_\lambda)^{-1} H (H + D_\lambda)^{-1}\).
- EM: \(A = H + \operatorname{diag}(C_{\text{rep}}^{-1})\); update \(C_\ell \leftarrow \frac{1}{m_\ell}\sum_{i\in I_\ell} (m_i^2 + V_{ii})\).

---

## 11) Configuration Knobs (quick reference)

- `kappa` (design conditioning), `anchor` (“min”/“max”).
- `K` (Welch DOFs for residual surrogates).
- `tau` (cut‑off threshold; default `1e-3`).
- `lambda_vals[1..L]` (groupwise ridge strengths).
- EM controls: `tol`, `maxit`, `eps`, initialization `init_C` (often bias‑corrected MoM floored).

---

## 12) Outputs and Expected Shapes

For each grid point (`kappa`, `K`), you get matrices of shape `N_SIM × L` for each estimator.  
The printed `results_grid` aggregates **average MSE across groups** (a single number per estimator per grid point). If you also need **per‑group MSE**, compute `colMeans((est - truth)^2)` before the averaging step.

---

## 13) Minimal Unit Tests (sanity)

- **GLS dimensions**: `length(u_hat) == q`, `all(dim(H) == c(q,q))`.
- **SPD checks**: smallest eigenvalue of `H` and `A` > 0 after ridge.
- **Group coverage**: `sort(unlist(I_list)) == 1:q` and `length(attr(I_list,"sizes")) == L`.
- **EM monotonicity** (optional): track log‑lik surrogate or norm of updates.

---

## 14) Porting Notes

- Replace `crossprod(Zw, Zw)` with `(Zw.T @ Zw)` (NumPy), or BLAS `dsyrk` in lower‑level languages.
- Replace `chol2inv` with two triangular solves if memory is tight.
- Use a deterministic RNG seeding strategy (`set.seed`) to compare implementations across languages.

---

**End of file.**