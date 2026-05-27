# `multiome` — paired ATAC + RNA with peak-gene ground truth

Two-step generative model: cis links are **cell-type-INVARIANT** and
cell-type-specific RNA expression arises because *upstream peaks*
switch on/off across cell types. The peak-gene indicator matrix
`M[G × P]` is fixed at sampling time and is the ground truth for
peak-to-gene inference.

## Step 1 — ATAC from topics

For each peak `p` and cell `j`:

```
A_{p, j}  =  base_p
           + σ · (   √π_topic   · T_{p, j}            # cell-type on/off
                   + √π_priv    · P_{p, j}            # peak-PRIVATE fluctuation
                   + √π_noise   · N_{p, j}            # iid noise
                 [ + √π_batch   · B_{p, j} ] )        # optional batch

T_{p, j}  =  z-scoreⱼ( log( β_atac[p, :] · θ_coarse[:, j] ) )
P_{p, j}  ~  N(0, 1)                                  iid per (peak, cell)
```

The peak budget `{topic, private, noise, batch}` is normalised to sum
to 1 (within `σ`). `--pve-topic` and `--pve-private` are the two main
identifiability knobs:
- Only a peak's **private** part reaches its linked gene; co-active
  bystanders share only the topic part.
- `--pve-private = 0` ⇒ peaks collinear within a cell type
  ⇒ cis links unidentifiable.

`--invariant-causal-fraction` makes a fraction of causal peaks
topic-INVARIANT (their topic mass is folded into `P`) — pure-private
accessibility, cleanly identifiable as a positive control alongside the
harder topic-driven links.

Returns `peak_logits[P × N]` AND a separate regulatory signal
`sig[P × N] = √π_topic·T + √π_priv·P` (no noise/batch), which step 2
inherits.

## Step 2 — RNA conditional on enhancers

Each gene `g` inherits its causal peaks' regulatory signal through a
cell-type-INVARIANT cis link `M[g, :]`:

```
C_{g, j}  =  std_j( Σ_{p ∈ M_g} sig_{p, j} )           over the shared cells

E_{g, j}  =  σ · (   √pve_cis     · C_{g, j}           # inherited from enhancers
                   + √(1 − pve_cis) · N_{g, j}         # gene-intrinsic noise
                 [ + √π_batch     · B_{g, j} ] )       # optional batch
```

The gene has **no topic path of its own** — cell-type specificity must
propagate through its peaks. Unlinked genes are noise only.

`--pve-cis = 0` ⇒ genes fully decoupled from their enhancers (null);
`--pve-cis = 1` ⇒ genes fully enhancer-explained.

Final counts (no-reference mode):
```
y_atac ~ Poisson( depth_atac_j · softmax_p(peak_logits[:, j]) )
y_rna  ~ Poisson( depth_rna_j  · softmax_g(gene_logits[:, j]) )
```
with `depth_*_j ~ LogNormal(depth_*, sd_*)` and centered to preserve mean.

## Optional nested topics

`--n-sub-topics K_sub > 1` introduces RNA subtypes within each coarse
topic. ATAC sees only the coarse marginal (K topics); RNA sees the
full `K × K_sub` resolution. `M` still links peaks (coarse-topic-driven)
to genes (subtype-resolved expression).

## Reference-conditioned mode

`--reference-rna <h5>` and/or `--reference-atac <h5>` switch the
per-modality sampler to the two-stage GLM + NB+copula PIT sampler used
by `topic --reference`. The two modalities are fit independently
(no cross-modality copula); coupling stays implicit through the shared
θ and the indicator `M`. Reference row counts override `--n-genes` /
`--n-peaks`.

## Patchy multiome

`--cell-overlap-fraction f` ∈ [0, 1]:
- `1.0` — every cell appears in both files (matched multiome).
- `0.0` — disjoint cells per modality.
- in between — `floor(N · f)` shared `cell_<i>` plus modality-only
  `atac_cell_<i>` / `rna_cell_<i>`.

## Outputs

| file                          | what                                       |
|-------------------------------|--------------------------------------------|
| `{out}.atac.zarr.zip`         | ATAC counts `[P × N_atac]`                 |
| `{out}.rna.zarr.zip`          | RNA  counts `[G × N_rna]`                  |
| `{out}.dict.parquet`          | β_atac `[P × K]` (marginalised)            |
| `{out}.derived_dict.parquet`  | W `[G × K_total]` = M · β_ext              |
| `{out}.prop.parquet`          | θ_coarse `[N_total × K]`                   |
| `{out}.theta_full.parquet`    | only if K_sub > 1                          |
| `{out}.beta_ext.parquet`      | only if K_sub > 1                          |
| `{out}.ground_truth.tsv.gz`   | (gene, peak) ground-truth cis links        |
| `{out}.gene_coords.tsv.gz`    | dummy gene coordinates (chr + TSS)         |
| `{out}.{atac,rna}.ln_batch.parquet` | per-modality log δ (B > 1)            |
| `{out}.batch.gz`              | unified per-cell batch membership          |

## Code map

`multiome::run_multiome` → `multiome::sample::{sample_dictionary,
sample_nested_topic_proportions, sample_indicator_matrix,
build_derived_dictionary, build_peak_logits, build_gene_logits,
sample_poisson_from_logits}` + reference path via
`copula::fit_global_copula`.
