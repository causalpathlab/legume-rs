# data-beans-sim

Synthetic single-cell-like count generators for the data-beans ecosystem.
Outputs land in the same `zarr` / `zarr.zip` / `h5` backends consumed by
`data-beans`, so simulated datasets share the entire downstream toolchain
(`pinto`, `senna`, `cocoa`, `chickpea`, …) with real ones.

Four subcommands, four use cases:

| subcommand   | what it generates                                                         | reference-conditioned? |
| ------------ | ------------------------------------------------------------------------- | ---------------------- |
| `topic`      | factored count matrix (genes × cells) with topic + batch structure        | yes (`--reference`)    |
| `multimodal` | M coupled count matrices sharing latent topics, modality-specific dicts   | no                     |
| `multiome`   | paired ATAC + RNA with peak–gene ground truth and shared topics           | yes (per modality)     |
| `bulk`       | bulk pseudo-samples by Dirichlet-mixing real single-cell data             | n/a (real-data sampler) |

## Generative model — common scaffolding

All parametric subcommands (everything except `bulk`) share the same
log-space variance decomposition for the two structural axes:

```
log β(g, k) = σ_β · [ √π_topic · u(g, k) + √(1 − π_topic) · v(g) ] − σ_β² / 2
log δ(g, b) =        √π_batch · z(g, b) + √(1 − π_batch) · w(g)
```

with `u, v, z, w ~ N(0, 1)` iid. Constructed so:

- `Var(log β) = σ_β²` and `Var(log δ) = 1`, both independent of the PVEs;
- `E[β] = 1` (centered), so `depth` becomes the **expected library size**;
- `π_topic` and `π_batch` are independent variance shares — both can be 1
  simultaneously. `π = 0` collapses the corresponding axis to its
  invariant component (`v(g)` or `w(g)`); `π = 1` removes the invariant
  component entirely.

Topic proportions θ are sampled as one-hot membership softened toward
uniform by `π_topic`:

```
θ(k*, j) = π_topic + (1 − π_topic) / K
θ(k,  j) =           (1 − π_topic) / K   for k ≠ k*
```

## `topic` — single-modality factored counts

```
Y(g, j) ~ Poisson( (depth / G) · δ(g, B(j)) · Σ_k β(g, k) θ(k, j) )
```

Smoke test:

```bash
data-beans-sim topic \
    --rows 2000 --cols 1000 \
    --factors 8 --batches 3 \
    --pve-topic 0.7 --pve-batch 0.3 \
    --beta-scale 1.0 \
    --depth 1000 \
    --output /tmp/sim_topic
```

Outputs (`/tmp/sim_topic.*`):

| file                       | what                                       |
| -------------------------- | ------------------------------------------ |
| `.zarr` (or `.h5`)         | sparse count matrix [G × N]                |
| `.dict.parquet`            | true β [G × K]                             |
| `.prop.parquet`            | true θ [N × K]                             |
| `.ln_batch.parquet`        | log δ [G × B]                              |
| `.batch.gz`                | batch membership per cell                  |
| `.hierarchy.parquet`       | (only if `--hierarchical-depth N`)          |

### Reference-conditioned mode (`topic --reference <h5/zarr>`)

Switches the count step from `Poisson(λ)` to a **negative binomial coupled
across HVGs by a fitted gene-gene copula** — the scDesign / scDesign2 /
scDesign3 lineage. Two-stage:

```
stage 1:  log λ⁰(g, j) = log μ̂(g) + √π_topic · t(g, j) + √π_noise · ε(g, j)
stage 2:  log λ (g, j) = log λ⁰(g, j) + √π_batch · z(g, b) + √(1 − π_batch) · w(g)
sample :  y(g, j) ~ NB(λ, r̂(g))   via   u = Φ(z*),  y = F⁻¹_NB(u; λ, r̂)
```

where `t = z-score_g(log(β·θ))` per cell, `μ̂` and `r̂` are MoM-fitted from
the reference, and `z*` is drawn from the gene-gene Gaussian copula
(low-rank factor + per-row ridge, kept as a **correlation** matrix so
PIT marginals stay unit-variance).

```bash
data-beans-sim topic \
    --reference real_pbmc.zarr.zip \
    --cols 5000 --factors 10 --batches 4 \
    --pve-topic 0.8 --pve-batch 0.5 --pve-noise 0.1 \
    --batch-rank 3 --batch-program empirical \
    --depth 5000 \
    --output /tmp/sim_topic_ref
```

Extra outputs in reference mode: `.r.parquet` (per-gene NB dispersion r̂),
`.hvg.gz` (HVGs used by the copula). `--depth` rescales `μ̂` so simulated
library size matches the supplied target.

`--batch-rank 0` ⇒ Splatter-style iid log-normal batch shifts.
`--batch-program empirical` reuses the top columns of the reference's
fitted gene-gene copula factor — i.e. the leading PCs of the reference's
empirical co-expression. Worst case for batch-correction methods, since
batch programs ride the same axes as real co-expression.

## `multimodal` — M coupled count matrices, shared topics

Modality-specific dictionaries built from a shared base:

```
β_0(:, k) = softmax_g( W_base[k, :]            )    (reference modality)
β_m(:, k) = softmax_g( W_base[k, :] + Δ_m[k, :] )    (m = 1 .. M − 1)
```

`Δ_m` is sparse spike-and-slab: only `--n-delta-features` genes per topic
are perturbed (slab `~ N(0, σ_Δ²)`). Each β column sums to 1 over genes,
so `depth_m` directly sets `E[lib(j) | m]`.

```bash
data-beans-sim multimodal \
    --rows 2000 --cols 1000 \
    --depth 5000,2000 \
    --factors 6 --batches 2 \
    --base-scale 1.0 --delta-scale 1.5 --n-delta-features 50 \
    --output /tmp/sim_mm
```

`--shared-batch-effects` reuses one δ across modalities; otherwise each
gets its own.

## `multiome` — paired ATAC + RNA with peak–gene ground truth

Shared coarse topics θ_coarse drive ATAC counts via β_atac (peak × topic);
RNA dictionary is *derived* from ATAC via a sparse cis-window indicator
M [G × P]:

```
W [G × K_total] = M · β_ext      (peak-derived RNA dictionary)
β_atac [P × K]  = E_subtype β_ext
```

Optional nested topic structure: `--n-sub-topics K_sub > 1` introduces
RNA subtypes within each coarse topic; ATAC sees only the marginalized
coarse layer, RNA sees the full `K × K_sub` resolution.

```bash
data-beans-sim multiome \
    --out /tmp/sim_mo \
    --n-genes 2000 --n-peaks 10000 --n-cells 2000 \
    --n-topics 6 --n-sub-topics 2 \
    --depth-rna 3000 --depth-atac 1500 \
    --batches 3 \
    --pve-topic 0.8 --pve-batch 0.5
```

Reference-conditioned per modality with `--reference-rna <h5>` and/or
`--reference-atac <h5>` (same NB+copula machinery as `topic --reference`,
fitted independently per modality). Cross-modality coupling stays
implicit through the shared θ and the indicator M.

Outputs:

| file                              | what                                 |
| --------------------------------- | ------------------------------------ |
| `.atac.zarr`, `.rna.zarr`         | sparse count matrices                |
| `.dict.parquet`                   | β_atac [P × K]                       |
| `.derived_dict.parquet`           | W [G × K_total]                      |
| `.prop.parquet`                   | θ_coarse [N × K]                     |
| `.theta_full.parquet`             | (only if K_sub > 1)                  |
| `.beta_ext.parquet`               | β_ext [P × K_total] (only if K_sub > 1) |
| `.ground_truth.tsv.gz`            | true peak-gene links                  |
| `.gene_coords.tsv.gz`             | dummy gene coordinates (chr + TSS)    |
| `.{atac,rna}.ln_batch.parquet`    | per-modality log δ (when bb > 1)      |
| `.batch.gz`                       | batch membership (when bb > 1)        |

## `bulk` — Dirichlet pseudo-bulk from real SC

Sums real single-cell counts into bulk pseudo-samples weighted by a
Dirichlet draw over topics. Reads a `(cell × topic)` parquet/tsv and the
matching SC dataset; outputs `(gene × sample)` bulk matrices alongside
the true topic fractions per sample.

```bash
data-beans-sim bulk \
    --sc-data-file pbmc.zarr \
    --topic-file pbmc.topics.parquet \
    --bulk-samples 200 --cells-per-sample 50 \
    --dirichlet-alpha 0.3 \
    --output /tmp/sim_bulk
```

Useful for benchmarking bulk-deconvolution methods against ground-truth
fractions, since the simulated bulks are convex combinations of cells
with known topic memberships.

## Output backends

`--backend zarr` (default), `--backend h5`, or `--backend zarr --zip` for
a `.zarr.zip` single-file archive. Companion `.parquet` / `.tsv.gz`
files are independent of the backend choice.

`--save-mtx` additionally emits a Matrix Market `.mtx.gz` triplet file
plus `.rows.gz` / `.cols.gz` name files. Useful for cross-checking
against tools that don't read zarr/h5.

## Tips

- **Set `--depth` to your real reference's mean library size** in
  reference mode; it's a multiplicative offset, not a target.
- **`pve_topic` and `pve_batch` are independent** — the model does not
  enforce `π_topic + π_batch ≤ 1`. Set both to 1 for the maximum-stress
  configuration (no shared invariant component on either axis).
- **`pve_topic = 0`** removes topic structure entirely (β collapses to a
  per-gene baseline, θ becomes uniform); useful as a null-model baseline.
- **For benchmarking batch-correction methods** in reference mode, use
  `--batch-program empirical --batch-rank 3` — batch shifts ride the
  reference's empirical gene-gene co-expression PCs, the hardest setting.
- **For CNV simulation**, use the `chickpea` simulator instead — CNV was
  removed from `topic` to keep the generative model clean and aligned
  with the NB+copula reference path.
