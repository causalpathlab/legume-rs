# What `cocoa diff` estimates, and why it is identified

This note fixes the definitions behind `cocoa diff` (τ, δ, the confounders V)
and the assumptions under which the reported contrast is causal. Everything
is per topic and per gene; the gene index is dropped.

## Two stages

Individual `i` has cells `j`, each in cell state `s_j`. Exposure `X_i` is
assigned to individuals, so everything that follows the exposure is an
individual-level quantity shared by all of that individual's cells.

**Stage 1 (cells to individual).** Pseudobulks pool cells of similar state
across individuals; their rates `μ` give each individual a count and an
expected count at its own cell states:

```text
E[ y_ij | i, s_j ] = n_ij · μ(s_j) · Λ_i
y_i = Σ_j y_ij,   m_i = Σ_p μ_p n_ip,   y_i | Λ_i ~ Poisson(Λ_i m_i)
```

Pseudobulks are built across individuals (the cell projection is centred
per individual within cell state before binning), a poorly mixed pseudobulk
merges with its sibling up the code tree, and one still holding fewer than a
minimum number of individuals (default 3) is dropped. Because every kept pseudobulk
pools several individuals, the individual × pseudobulk table identifies `μ` and the individual multipliers from the
observed counts alone (up to one scale per gene that cancels in every
reported effect); no imputed counterfactual is needed. This requires the
individual × pseudobulk table to be connected: pseudobulks must mix
individuals. Under the multiplicative assumption below, a pseudobulk seen in
only one exposure is still fine, because its individuals' multipliers are
fixed by the pseudobulks they share with others. Stage 1 uses no exposure
labels, so it is computed once and reused by every permutation draw.

`Λ_i` is individual `i`'s rate multiplier at fixed cell state. Assumption: within a
topic, the individual and the exposure act multiplicatively on the state
baseline, with no interaction between individual and state or between
exposure and state.

**Stage 2 (individuals).** Everything below is about `Λ_i`.

## Definitions

- Potential multipliers `Λ_i(x)`, one per exposure level.
- **τ(x) = E[Λ_i(x)]**, the average exposure effect over the population of
  individuals. The reported contrast is `ψ = log τ(1) − log τ(0)`.
- **δ_i(x) = Λ_i(x) / τ(x)**, the individual effect, with `E[δ_i(x)] = 1`.

## Assumptions

1. **Invariance.** `δ_i(1) = δ_i(0) = δ_i`: the individual effect does not
   depend on exposure, so the exposure multiplies every individual by the same
   `τ(1)/τ(0)`. Then `Λ_i = τ(X_i) · δ_i`, and the average of individual log
   ratios equals `ψ`. The two usual estimands coincide, and the ratio of
   means is the efficient way to estimate it from counts.
2. **Conditional ignorability.** `δ_i ⊥ X_i | V_i` for pre-exposure
   individual covariates `V`.
3. **Positivity.** `0 < P(X = x | V) < 1`.

## Why the δ constraint is the identifying assumption

Each individual is seen under one exposure only, so the data give the product
`τ(X_i) δ_i` and never separate the factors. They are separated by a moment
condition on δ, and that condition is the assumption:

| Assumption | Condition on δ |
|---|---|
| randomized, `δ ⊥ X` | `E[δ ∣ X = x] = 1` (mean-1 δ within each arm) |
| `δ ⊥ X ∣ V` | `E[(X − π(V)) (δ − q(V))] = 0` for any `q`, or for any `π` when `q(V) = E[δ ∣ V]` |

Under confounding (V drives both X and expression) the first condition is
false, and τ absorbs each arm's average of V's effect. The second condition
is the statement "δ carries no exposure information given V". It needs
either the propensity `π(V) = P(X = 1 | V)` or the outcome regression
`q(V) = E[δ | V]` to be right, not both. The estimator below fits `q` only
as a function of `e(V)`.

## Estimation

With invariance, the exposure-free rate `H_i(ψ) = Λ_i e^{−ψ(X_i)}` is
proportional to `δ_i`. Following Dukes & Vansteelandt (2018, Am J Epidemiol,
"A note on G-estimation of causal risk ratios"), `ψ` is fit per gene by a
Gamma GLM with a log link over the individuals,

```text
E[Λ | V, X] = exp( b_0 + b_e' e(V) + ψ(X) )
```

with `e(V)` the propensities `P(X = x | V)`, one column per non-reference
level. With the Gamma working variance the score for `ψ` is the g-estimating
equation (Robins, Mark & Newey 1992; Vansteelandt & Joffe 2014)

```text
Σ_i (1{X_i = x} − e_x(V_i)) (H_i(ψ) − q(V_i)) / q(V_i) = 0,   q(V) = exp(b_0 + b_e' e(V))
```

This is Dukes & Vansteelandt's eq. 6, and it is doubly robust: `ψ` is
consistent if `e` is right, whatever the outcome looks like, or if the outcome
model `q(V) = exp(b_0 + b_e' e(V))` is right, even when `e` is misspecified.
The outcome model is deliberately narrow (V enters only through `e`): adding
V itself as outcome covariates (their eq. 7) widens the second condition but
was not calibrated with few individuals, and was dropped.

- `Λ_i = y_i / m_i`, the raw ratio. The posterior mean would shrink toward a
  model and carry that model's error into the effect.
- `e(V)` is a ridge multinomial logistic regression, not clipped:
  individuals without overlap (`e` near 0 or 1) have `1{X = x} − e ≈ 0` and
  drop out of the equation on their own.
- Other working variances break the equivalence: the weight must be a
  function of V only, and only the Gamma variance (proportional to the
  squared mean) makes it so.
- With a log link and Gamma variance the scoring weights are all one, so each
  step is least squares on the working response `η + Λ/μ − 1` with one
  projection shared by every gene; a gene keeps a step only if its
  quasi-likelihood does not fall.
- Without confounders the model holds only the level indicators and `ψ` is
  the log ratio of means.
- `τ(0) = mean_i H_i`, `τ(x) = τ(0) e^{ψ(x)}`, and `δ_i = H_i / τ(0)`, all
  non-negative.

`m_i` comes from the stage-1 baseline, a rank-1 fit of the individual ×
pseudobulk table that uses no exposure labels and is fit once.

Outputs follow the definitions: `*.effect.parquet` holds τ per level and ψ
per non-reference level for each gene and topic, `*.delta.parquet` holds δ
per gene, individual, and topic (mean one over individuals), and
`*.contrast.parquet` holds ψ averaged over topics.

## Inference

The sharp null is `Λ_i(1) = Λ_i(0)` for every individual. Relabelings are drawn
from the fitted propensity (clipped, for the sampler) by a pairwise-swap
sampler over permutations (Berrett et al. 2020, conditional permutation
test). Each draw keeps every individual's V and δ together and changes only
X, and the statistic is recomputed in full. V and the propensity do not
depend on X, so they are fixed across draws. With more than two levels, each
non-reference level gets its own test, and a global test that all levels
equal the reference uses the sum of squared z over levels with a permutation
p-value.

A uniform shuffle is exact only in the randomized row of the table. An
outcome-adjusted statistic with a uniform shuffle is centred but has the
wrong width when V predicts X.

## Where V comes from

V must be pre-exposure. In order of preference:

1. `--covariate-file`: measured covariates. They enter stage 2 only; the
   pseudobulks of stage 1 come from cell state regardless, so μ tracks cell
   state and V is adjusted once. (Building pseudobulks from V made μ absorb
   part of V's effect, which stage 2 then removed a second time.)
2. `--control-genes`: factors of genes that respond to V but not to X (RUV-g).
3. On request: the top principal components of individual expression
   (`--confounder-pcs auto|K`; `auto` uses the eigenvalue-ratio rule),
   assumed to carry no exposure signal. They are computed without reference
   to X. Residualizing them on X would make them orthogonal to X and make
   the propensity flat. If a component is really the exposure's own program,
   the propensity approaches 0 or 1 and the effective sample size reported in
   the log collapses.

Features are per-topic individual log rates from raw counts, computed
without reference to the exposure labels.

## The intrinsic limitation

Each individual is observed under one exposure only. An individual-level
deviation that is correlated with the exposure and not captured by V cannot
be told apart from τ by any estimator; it lands in τ. δ absorbs deviations
unrelated to the exposure, which become between-individual variance rather
than signal, but by construction it cannot absorb a deviation correlated with
X without making τ and δ inseparable. With few individuals the nuisance fits
also remove measured confounding only partly: the conditional permutation
keeps the test calibrated, and the leftover bias stays in the point estimate.
A sensitivity analysis (how strongly an unmeasured confounder would have to
tilt the exposure odds beyond π(V) to overturn a call) is the natural way to
report this limitation.
