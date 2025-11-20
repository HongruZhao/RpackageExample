# P\_k(f,t) Separable Model — Estimation & Tuning Guide

**Goal.** Estimate separable surfaces
\[
P_k(f,t)=h_k(f)\,s_k(t),\qquad k\in\{1,2\},\ f=1,\dots,10,\ t=1,\dots,T=n/F
\]
from multiplicative observations
\[
\widehat{P}_k(f,t) = P_k(f,t)\,\eta_{k,f,t},\qquad \eta_{k,f,t}\sim \chi^2_{2K_k}/(2K_k).
\]
We model each \(k\) separately (set \(K=K_k\) for the run). This guide summarizes modeling options, tuning, and an evaluation plan an agent can implement.

> **Noise law.** If \(X\sim \chi^2_{2K}/(2K)\) then \(X\sim\mathrm{Gamma}(\text{shape}=K,\ \text{rate}=K)\) (mean \(1\), var \(1/K\)). Hence
\[
Y_{ft}:=\widehat{P}(f,t)\mid P_{ft}\ \sim\ \mathrm{Gamma}\!\big(\alpha=K,\ \beta=K/P_{ft}\big),
\]
so \(\mathbb{E}[Y|P]=P\), \(\mathrm{Var}(Y|P)=P^2/K\).

**Identifiability.** The factorization \(P_{ft}=h_f s_t\) is scale–nonidentifiable: \((c\,h_f,\, s_t/c)\) yields the same \(P\). Enforce a normalization such as \(\sum_t \log s_t = 0\) or \(\mathrm{mean}(s)=1\).

---

## 1) Models the agent can fit

### 1.1 Parametric separable (if forms are known)
Assume, for example, \(h(f)=c_h f^{-\alpha}\) and \(s(t)=c_s e^{-\beta t}\). Then
\[
\log P(f,t)=\theta_0-\alpha \log f-\beta t,\quad \theta_0=\log c_h+\log c_s.
\]
**Fit options.**
- **Gamma GLM (log link):**
  ```r
  dat <- data.frame(y = as.vector(P_hat), f = as.vector(f_grid), t = as.vector(t_grid))
  fit <- glm(y ~ log(f) + t, family = Gamma(link = "log"), data = dat)
  P_hat_mean <- fitted(fit)
  ```
- **Nonlinear least squares on log-scale** (if you insist on a fixed form).

Pros: efficient when the form is correct. Cons: sensitive to misspecification.

---

### 1.2 Semiparametric log‑additive (robust default)
Because \(P_{ft}=h_f s_t\), the **log** mean is additive:
\[
\log P_{ft} = a_f + b_t,\quad a_f=\log h_f,\ b_t=\log s_t.
\]
Two implementations:

**(A) WLS on \(\log \widehat P\)** using exact moments of \(\log\eta\). For \(\eta\sim\mathrm{Gamma}(K,K)\):
\[
\mathbb{E}[\log\eta]=\psi(K)-\log K,\qquad \mathrm{Var}(\log\eta)=\psi_1(K),
\]
where \(\psi,\psi_1\) are digamma/trigamma. Define \(Y^\*_{ft}=\log \widehat P_{ft} - (\psi(K)-\log K)\). Then
\[
Y^\*_{ft}=a_f+b_t+\varepsilon_{ft},\quad \mathbb{E}[\varepsilon]=0,\ \mathrm{Var}(\varepsilon)=\psi_1(K).
\]
Fit a two‑way additive model with an identifiability constraint:
```r
psi  <- digamma(K); psi1 <- trigamma(K)
Yc   <- log(P_hat) - (psi - log(K))  # center
Ffac <- factor(f_grid)               # F=10 levels
# use a smooth in t (recommended) or a factor if T is small
fit  <- lm(Yc ~ 0 + Ffac + splines::bs(t_grid, df = 10),
           weights = rep(1/psi1, length(Yc)))
a_hat <- coef(fit)[grep("^Ffac", names(coef(fit)))]
b_smooth <- as.numeric(splines::predict(bs(t_grid, df=10), t_grid) %*%
                       coef(fit)[!grepl("^Ffac", names(coef(fit)))])
b_hat <- b_smooth - mean(b_smooth); a_hat <- a_hat + mean(b_smooth)
h_hat <- exp(a_hat); s_hat <- exp(b_hat)
```

**(B) Gamma GAM (log link)** with factor in \(f\) and smooth in \(t\) (targets the original Gamma likelihood):
```r
library(mgcv)
dat <- data.frame(y = as.vector(P_hat),
                  f = factor(as.vector(f_grid)),
                  t = as.vector(t_grid))
fit <- gam(y ~ 0 + f + s(t, bs = "ps", k = 20),
           family = Gamma(link = "log"), data = dat)
alpha_f  <- coef(fit)[grep("^f", names(coef(fit)))]
smooth_t <- predict(fit, newdata = data.frame(f=levels(dat$f)[1], t=1:max(dat$t)),
                    type="terms")[, "s(t)"]
b_hat <- smooth_t - mean(smooth_t); a_hat <- alpha_f + mean(smooth_t)
h_hat <- exp(a_hat); s_hat <- exp(b_hat)
```

If you need to **fix the Gamma shape at \(K\)** (Welch DOF), use **GAMLSS**:
```r
library(gamlss)
fit <- gamlss(y ~ 0 + f + cs(t),
              sigma.fo = ~ 1,
              family = GA(link = "log"),
              data = dat,
              sigma.fix = TRUE, sigma.start = K)   # fixes shape = K
```

---

### 1.3 Rank‑1 MLE with alternating updates (fast, no dependency)
With \(Y_{ft}\sim\mathrm{Gamma}(K, \text{rate}=K/(h_f s_t))\), the block‑wise optimum leads to **closed‑form updates**:
\[
h_f^{\text{new}}=\frac{1}{T}\sum_{t=1}^T \frac{Y_{ft}}{s_t},\qquad
s_t^{\text{new}}=\frac{1}{F}\sum_{f=1}^F \frac{Y_{ft}}{h_f}.
\]
Enforce scale via \(\mathrm{mean}(s)=1\) at each step.
```r
est_separable_gamma <- function(Y, maxit = 200, tol = 1e-8) {
  F <- nrow(Y); T <- ncol(Y)
  s <- rep(1, T); h <- rowMeans(Y)
  for (it in 1:maxit) {
    h_new <- rowMeans(Y / rep(s, each = F))
    s_new <- colMeans(Y / h_new)
    c <- mean(s_new); s_new <- s_new / c; h_new <- h_new * c
    if (max(abs(h_new - h)/pmax(h,1e-12)) < tol &&
        max(abs(s_new - s)/pmax(s,1e-12)) < tol) break
    h <- h_new; s <- s_new
  }
  list(h = h, s = s, P = h %o% s)
}
```
Notes: exact for constant \(K\); if \(K\) varies with \((f,t)\), replace the means by **weighted** means with weights \(K_{ft}\). Add optional spline‑smoothing to \(\log s_t\) post‑hoc if desired.

---

### 1.4 Bayesian hierarchical (positivity + smoothness + UQ)
Model
\[
Y_{ft}\sim\mathrm{Gamma}\!\big(K,\ \text{rate}=K/(h_f s_t)\big),\quad
\log h_f\sim \text{RW1/2 prior},\ \ \log s_t\sim \text{RW1/2 prior}.
\]
- **`brms` (Stan):**
  ```r
  library(brms)
  fit <- brm(bf(y ~ 0 + f + s(t), family = Gamma(link = "log")),
             data = dat, cores = 4)
  # To approximate known K, put a tight prior on shape or move to INLA/Stan custom.
  ```
- **`INLA`/`TMB`/`nimble`/`rstan`** for fixing shape exactly and custom priors.

---

## 2) Tuning and model selection

- **Primary objective for accuracy:** minimize **out‑of‑sample negative log‑likelihood** (Gamma) or, equivalently, **Gamma deviance**.
- **Cross‑validation:** use K‑fold across time (block CV if serial correlation matters). For WLS‑log models, CV on squared error of centered logs \(Y^\*\).
- **Smoothness (GAM):** choose basis size \(k\) and use REML/GCV (`gam(..., method="REML")`), and `select=TRUE` to allow automatic shrinkage of unnecessary wiggles.
- **Alternating MLE:** if you add roughness penalties to \(\log h,\log s\) (e.g., \(\ell_2\) on second differences), tune penalty weights by CV on Gamma NLL.

---

## 3) Evaluation protocol for comparing models

Let \(\widehat{P}(f,t)\) be the predicted mean.
- **Gamma NLL (per held‑out point)**: \(-\log p_\Gamma(y\mid \widehat{P},K)\).
- **RMSE on \(\log P\)** (centered): root mean squared error of \(Y^\*\) residuals.
- **MAPE on original scale** (optional): \(\mathrm{mean}_{ft}\bigl|y_{ft}-\widehat{P}_{ft}\bigr|/\max(y_{ft},\epsilon)\).

**Train/validation splits.** Use time‑blocked K‑fold CV to avoid leakage; ensure each fold spans all \(f\) levels.

**Outputs to save per model:**
- \(\widehat{h}(1{:}F)\), \(\widehat{s}(1{:}T)\), \(\widehat{P}= \widehat{h}\otimes \widehat{s}\).
- CV scores (Gamma NLL / RMSE\_log / MAPE).
- Diagnostics (plots of \(\widehat{s}(t)\), \(\widehat{h}(f)\), residual QQ on log‑scale).

---

## 4) Recommended R packages (by task)

| Task | Package(s) | Notes |
|---|---|---|
| Gamma GLM/GAM (log link) | **mgcv** | `gam(y ~ 0 + f + s(t), family = Gamma(link="log"))` |
| Gamma regression with **fixed** shape \(K\) | **gamlss** | `family = GA(link="log")`, `sigma.fix=TRUE`, `sigma.start=K` |
| Log‑WLS on centered logs | **stats** | `lm()` with digamma/trigamma centering and weights |
| Bayesian Gamma GAM | **brms**, **INLA**, **TMB**, **nimble** | Full UQ; custom priors; ability to fix/target shape |
| Fast rank‑1 MLE | (none required) | Use the alternating updates above; add smoothing post‑hoc if needed |

---

## 5) Minimal agent API (language‑agnostic)

```
estimate_separable(
  P_hat: matrix[F,T],               # observations \hat P
  K: numeric(1),                    # Welch DOF for this k
  method: {"gamma_gam","log_wls","alt_mle","parametric"},
  options: list                     # e.g., df for spline, priors, penalties
) -> list(h = F, s = T, P = F×T, fit = object, metrics = list)
```

- **`gamma_gam`** ⇒ `mgcv::gam` with `y ~ 0 + f + s(t)`, link=log, family=Gamma; returns \(\hat h,\hat s\) from terms.
- **`log_wls`** ⇒ centered log regression with `lm`; returns \(\hat h,\hat s\).
- **`alt_mle`** ⇒ alternating updates; returns \(\hat h,\hat s\), optionally followed by smoothing of \(\log s\).
- **`parametric`** ⇒ GLM with specified parametric form; returns coefficients and implied \(\hat h,\hat s\).

**Helper: digamma/trigamma centering**
```r
center_log_observation <- function(P_hat_vec, K) {
  log(P_hat_vec) - (digamma(K) - log(K))
}
```

---

## 6) Pointers to the project spec

- Observation model \(\widehat{P}_k(f,t)=P_k(f,t)\eta_{k,f,t}\) and separability \(P_k(f,t)=h_k(f)s_k(t)\) are stated in the project documents; use them as ground truth for the agent’s assumptions. (See the simulation/spec PDFs and the openAI.md extraction.)

---

## 7) Checklist (before comparing models)

- [ ] Normalize identifiability (e.g., `mean(s)=1`), and propagate the scale to \(h\).
- [ ] Use time‑blocked CV; report mean and SE across folds for Gamma NLL.
- [ ] Plot \(\hat s(t)\) and \(\hat h(f)\); check monotone trends if expected (e.g., power law in \(f\)).
- [ ] Log residual diagnostics: mean ≈ 0, variance ≈ \(\psi_1(K)\).
- [ ] Save artifacts: \(\hat h,\hat s,\hat P\), CV traces, and seeds for reproducibility.

---

**Notes for human readers.** This markdown condenses the separable‑surface estimation problem and the exact observation model used in the LMM simulation framework (Welch‑type Gamma noise and product structure). For the broader context (design matrices, weight construction, and estimator definitions), see the project note and the openAI.md extraction included with this repo.
