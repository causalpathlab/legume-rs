# `topic` — single-modality factored counts (with optional NB+copula reference mode)

## Synthetic mode (no `--reference`)

Per-gene per-cell counts under a log-normal topic factor model with
explicit variance decomposition on both axes:

```
log β(g, k) = σ_β · [ √π_topic · u(g, k) + √(1 − π_topic) · v(g) ] − σ_β² / 2
log δ(g, b) =         √π_batch · z(g, b) + √(1 − π_batch) · w(g)
θ(k*, j)   = π_topic + (1 − π_topic) / K
θ(k,  j)   =           (1 − π_topic) / K       for k ≠ k*
λ(g, j)    = (depth / G) · δ(g, B(j)) · Σ_k β(g, k) · θ(k, j)
y(g, j)    ~ Poisson( λ(g, j) )
```

with `u, v, z, w ~ N(0, 1)` iid.

Design invariants:
- `Var(log β) = σ_β²` and `Var(log δ) = 1`, independent of the PVEs.
- `E[β] = 1` (centered log-normal), so `depth` is the **expected**
  library size, emergent — no per-cell rescaling.
- `π_topic` and `π_batch` are **independent** variance shares; both can
  hit 1 simultaneously. `π = 0` collapses the corresponding axis to its
  invariant component (`v(g)` or `w(g)`); `π = 1` removes the invariant
  component entirely.

Optional housekeeping injection: the first `n_housekeeping` rows are
overwritten with `LN(log(fold·mean(β)), σ_hk²)` shared across topics
(by-design topic-invariant high-expression genes).

Optional hierarchical β: `--hierarchical-depth N` replaces the flat
log-normal dictionary with a stick-breaking binary tree of depth N
(K = 2^(N−1) leaf topics). In hierarchical mode the tree's
stick-breaking already encodes topic structure, so `π_topic` blends
only θ.

## Reference-conditioned mode (`--reference <h5/zarr>`)

Two-stage GLM with NB+copula PIT sampling (scDesign / scDesign2 /
scDesign3 lineage):

```
stage 1:  log λ⁰(g, j) = log μ̂(g) + √π_topic · t(g, j) + √π_noise · ε(g, j)
stage 2:  log λ (g, j) = log λ⁰(g, j) + √π_batch · z*(g, b) + √(1 − π_batch) · w(g)
sample :  u = Φ(z*),  y = F⁻¹_NB(u; λ, r̂(g))
```

where:
- `t = z-score_g( log(β · θ) )` per cell, with β drawn as in synthetic mode,
- `μ̂` and `r̂` are method-of-moments fits from the reference,
- `z*` is sampled from a gene-gene Gaussian copula
  (low-rank factor of rank `--batch-rank`, choice of program from
  `--batch-program {random, empirical}`),
- `--depth` rescales `μ̂` so simulated library size matches the target.

`--batch-program empirical` reuses the top columns of the reference's
fitted gene-gene copula factor (its leading co-expression PCs). This is
the worst-case for batch-correction methods: batch axes ride the same
geometry as real co-expression.

## Outputs

| file                 | shape / contents                  |
|----------------------|-----------------------------------|
| `{out}.zarr.zip`     | sparse count matrix `[G × N]`     |
| `.dict.parquet`      | true β `[G × K]`                  |
| `.prop.parquet`      | true θ `[N × K]`                  |
| `.ln_batch.parquet`  | log δ `[G × B]`                   |
| `.batch.gz`          | per-cell batch membership         |
| `.hierarchy.parquet` | only with `--hierarchical-depth`  |
| `.r.parquet`         | per-gene NB dispersion (ref mode) |
| `.hvg.gz`            | HVGs used by the copula (ref mode)|

## Code map

- Synthetic dictionary: `core::sample_lognormal_dictionary`.
- θ: `core::sample_theta_kn`.
- Batch: `core::sample_log_batch_effects`.
- Counts: `core::sample_poisson_triplets`.
- Reference fit: `copula::fit_global_copula`.
- Hierarchical β: `core::generate_hierarchical_dictionary`.
