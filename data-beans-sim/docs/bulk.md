# `bulk` — Dirichlet-mixed pseudo-bulk from real single-cell data

Synthesise bulk pseudo-samples by convex-combining real single-cell
counts. Each sample's mixing weights are a Dirichlet draw over a set
of cell-level topic memberships supplied by the user. Useful for
benchmarking bulk-deconvolution methods against ground-truth
fractions, because the simulated bulks are exact mixtures of cells
with known topic memberships.

## Inputs

- `--sc-data-file <h5/zarr>` — real single-cell counts.
- `--topic-file <parquet|tsv>` — `[cell × topic]` topic memberships.

## Generative model

For each bulk sample `b ∈ 1..B`:

```
1. Draw mixing fractions over topics:
     π_b  ~  Dirichlet(α · 1_K)              α = --dirichlet-alpha

2. Pick cells for this sample:
     m_b              = --cells-per-sample
     For each cell slot s ∈ 1..m_b:
       sample a topic k from Categorical(π_b),
       then sample a cell c uniformly from {c : topic_c = k}.

3. Sum the raw single-cell counts of the selected cells:
     y_b(g)  =  Σ_{c ∈ selected_b}  Y_sc(g, c)
```

The simulated bulks are exact, weighted sums of real cells — no extra
noise model on top, since the single-cell counts already carry the
target's empirical distribution.

## Knobs

- `--dirichlet-alpha` — concentration of the per-sample topic mixture.
  Small α ⇒ near-pure samples (one dominant topic); large α ⇒ near-
  uniform mixtures.
- `--cells-per-sample` — how many real cells are pooled per bulk; sets
  the per-sample read depth (linear in the mean SC library size).
- `--bulk-samples` — number of bulk samples to synthesise.

## Outputs

| file                       | what                                      |
|----------------------------|-------------------------------------------|
| `{out}.bulk.parquet`       | bulk counts `[G × B]`                     |
| `{out}.fractions.parquet`  | true topic fractions per sample `[B × K]` |
| `{out}.cell_indices.tsv.gz`| which SC cells contributed to each bulk   |

## Code map

`deconv::generate_convoluted_data`.
