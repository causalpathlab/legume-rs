# Methodological changes since CoCoA-diff

Reference: Park & Kellis (2021), "CoCoA-diff: counterfactual inference for
single-cell gene expression analysis", Genome Biology 22:228,
[doi:10.1186/s13059-021-02438-4](https://doi.org/10.1186/s13059-021-02438-4).

## Original

1. Each cell is matched to cells of the opposite condition (KNN on top
   principal components); its counterfactual is imputed by a Poisson
   regression on the matched cells.
2. Per individual, the observed and counterfactual pseudobulks share a
   label-invariant confounding mean `μ_i`, each with its own
   condition-specific sequencing depth.
3. With `μ_i` fixed, each individual's residual (differential) effect is
   estimated; genes are tested across individuals by a Wilcoxon rank-sum test
   on the adjusted profiles, or a Wald statistic for the average effect.

Confounders are adjusted only through matching; there is no model of
individual-level confounders.

## Now

| Aspect | Original | Now |
|---|---|---|
| Structure | match, then decompose per individual | two stages: cells to individuals (stage 1), individuals (stage 2) |
| Cell-state baseline | confounding mean `μ_i` per individual, identified only with the imputed counterfactual | `μ_p` per pseudobulk of similar cell states, shared across individuals, from a label-free rank-1 fit of the observed counts (see below), fit once; offset `m_i = Σ_p μ_p n_ip` |
| Counterfactual `y0` and matching | KNN matching builds the imputed counterfactual | removed: no matching; stage 1 uses no exposure labels and is computed once for all permutation draws |
| Collider bias (X → cell type ← U) | none | soft topic weights residualized on exposure, so conditioning on cell type does not open X → A ← U → Y |
| Individual heterogeneity | per-individual residual effects | `δ_i`, the exposure-free individual effect left after stage 2 (mean 1); its spread enters through the Gamma working variance and the permutation null |
| Individual-level confounders V | not modelled | covariate file, control genes (RUV-g), or top PCs, in stage 2 only; pseudobulks come from cell state regardless |
| Effect estimator | residual effects with `μ_i` fixed | g-estimation (Dukes & Vansteelandt 2018, eq. 6): per-gene Gamma GLM, `E[Λ ∣ V, X] = exp(b + b_e e(V) + ψ(X))`; doubly robust: consistent if the propensity is right, or if expression depends on V log-linearly through `e(V)` |
| Exposure | two groups | K levels, each against a reference |
| Inference | Wilcoxon rank-sum across individuals, or Wald test of the average effect | permutation over individuals; conditional on the propensity when V is given |
| Outputs | confounder-adjusted pseudobulk profiles, per-individual effects | `τ(x)`, `ψ(x)`, `δ_i` (mean 1), propensities |

## Why pseudobulk-level μ removes the need for y0

Per individual, `y1_i ~ Poisson(n_i μ_i τ(X_i))` gives only the product
`μ_i τ(X_i)`: the baseline and the effect cannot be separated from one
individual's own cells, which is why the original needed the imputed
counterfactual. Pooling the baseline at the pseudobulk level,

```text
y1(i, p) ~ Poisson( n_ip · μ_p · Λ_i ),   Λ_i = τ(X_i) · δ_i
```

is a two-way (cell state × individual) design. When pseudobulks mix
individuals, and the individual × pseudobulk table is connected, it
identifies every `μ_p` and `Λ_i` from the observed counts alone, up to one
scale per gene that cancels in `ψ` and `δ`. Other individuals' cells in the
same state supply what the counterfactual used to, and the matched `y0`
(other individuals' observed cells, reweighted) adds no information; fitted
with the other arm's cells it pulled that arm's effect into the baseline and
inflated contrasts when V was not given. This pooling was introduced in the
Rust implementation (June 2025).

The condition is connectivity: pseudobulks must mix individuals, so that
each individual shares pseudobulks with others. Connectivity alone is not
enough in practice. The multilevel refine used earlier built pseudobulks
within each individual, so most held one individual and μ_p absorbed Λ_i,
and with it most of the exposure effect. Pseudobulks are now binned across
individuals on a projection centred per individual within cell state (a
plain per-individual mean also removes composition). A leaf with fewer than
`--min-individuals-per-pb` individuals merges up the code tree, and one
still short is dropped, with a per-topic report in `*.stage1.parquet`. Under no exposure × cell-state
interaction, a pseudobulk seen in one exposure only is still identified
through its individuals. Cross-exposure matching, kept for a while only to
trim cells without a counterpart, had no distance limit, so it removed a cell
only when the other exposure had essentially no cells of that type nearby; it
did not enforce positivity and made every permutation draw redo stage 1. It
was removed.

## Definitions and assumptions

See `identification.md`:
- `τ(x) = E[Λ_i(x)]`, the average exposure effect.
- `δ_i` is the individual effect, invariant to exposure and independent of X
  given V.
- An individual-level confounder that is correlated with X and not in V cannot be
  separated from `τ`. This is an intrinsic limitation.

## Open

- Rank-based permutation p-values, precision-weighted topic averaging, and a
  sensitivity analysis for unmeasured confounding.
