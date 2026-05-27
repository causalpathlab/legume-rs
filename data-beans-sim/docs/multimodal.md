# `multimodal` — M coupled count matrices sharing latent topics

Generate `M` modality-specific count matrices from one shared latent
topic axis. Dictionaries are perturbations of a common base, so the
modalities share *cell-type structure* but have *modality-specific
gene programs*.

## Generative model

Shared base + sparse spike-and-slab perturbations:

```
W_base ∈ ℝ^{K × G}      ~ N(0, base_scale²) iid          (or stick-breaking-derived
                                                          logits in hierarchical mode)
W_delta_m[k, g]         = ξ_{m, k, g} · N(0, delta_scale²)
                          with ξ_{m, k, g} = 1 for exactly
                          `n_delta_features` genes per (m, k)
                          (zero elsewhere)
β_0(:, k) = softmax_g( W_base[k, :]              )       (reference modality, m=0)
β_m(:, k) = softmax_g( W_base[k, :] + W_delta_m[k, :] )   (m = 1 .. M − 1)
```

Each β_m column sums to 1 over genes. Per-modality counts (batch
effects independent per modality unless `--shared-batch-effects`):

```
log δ_m(g, b)  = √π_batch · z(g, b) + √(1 − π_batch) · w(g)
λ_m(g, j)      = depth_m · δ_m(g, B(j)) · Σ_k β_m(g, k) · θ(k, j)
Y_m(g, j)      ~ Poisson( λ_m(g, j) )
```

Because each β_m(:, k) sums to 1 deterministically, `depth_m` is exactly
the expected per-cell library size for modality m (emergent — no
per-cell rescaling).

Optional `--hierarchical-depth N` swaps the flat random `W_base` for a
stick-breaking binary tree on `K = 2^(N−1)` leaves.

Optional housekeeping injection (first `n_housekeeping` columns of
`W_base` set to `housekeeping_fold · base_scale`) gives a set of
topic-invariant high-expression genes shared across modalities.

## Knobs

- `--n-delta-features` — per-topic per-modality sparsity of the
  perturbation: the spike-and-slab "k-of-G" budget. Small values
  ⇒ near-replica modalities; large values ⇒ near-independent
  per-modality dictionaries.
- `--delta-scale` — slab SD; how strong the per-modality
  programs are when they exist.
- `--shared-batch-effects` — reuse one δ across modalities (collapses
  the per-modality batch sub-spaces).

## Outputs

Per modality `m ∈ 0..M`:

| file                          | what                              |
|-------------------------------|-----------------------------------|
| `{out}.{m}.zarr.zip`          | counts `[G × N]`                  |
| `{out}.{m}.beta.parquet`      | β_m `[G × K]` (post-softmax)      |
| `{out}.{m}.spike_mask.parquet`| ξ_m  `[K × G]` (m ≥ 1)            |
| `{out}.{m}.ln_batch.parquet`  | log δ_m `[G × B]`                 |
| `{out}.theta.parquet`         | shared θ `[N × K]`                |
| `{out}.w_base.parquet`        | base logits `[K × G]`             |
| `{out}.batch.gz`              | per-cell batch membership         |
| `{out}.hierarchy.parquet`     | only with `--hierarchical-depth`  |

## Code map

`multimodal::generate_multimodal_data` + the shared `core` samplers.
