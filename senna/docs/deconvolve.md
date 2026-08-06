# `senna deconvolve` — bulk deconvolution against an empirical reference

Splits bulk RNA-seq into per-cell-type contributions, and produces BayesPrism's
headline deliverable: a per-sample × per-cell-type expression tensor for
within-type differential expression.

```bash
senna deconvolve --from bge.senna.json \
                 --sc-data counts.zarr \
                 --annotation annot.label_stability.parquet \
                 --bulk bulk.parquet \
                 -o out
```

## Inputs

| input | what it supplies |
|---|---|
| `--from` | a `senna bge` manifest: the gene axis, the embedding width, and (by default) the paths below |
| `--sc-data` | the single-cell counts the reference profiles are measured from |
| `--annotation` | cells × cell types, either annotate layout; a hard membership table is read as one-hot |
| `--bulk` | bulk counts, genes × samples, parquet or delimited text |

`--sc-data` and `--annotation` default to whatever the `--from` manifest
records, so both are usually omitted. Gene names are reconciled through the
shared canonicalizer, so `ENSG…_SYM` on one axis and bare symbols on the other
align without pre-editing.

## The model

Bulk sample `s` is a non-negative mixture of `R` empirical component profiles:

```
ε_{s,g} ~ Gamma(r, r)                                  NB overdispersion, mean 1
y_{s,g} ~ Poisson(ε_{s,g} · Σ_m u_{s,m} · x̄_{g,m})
Z_{s,·,g} ~ Multinomial(y_{s,g}, p),  p_m ∝ u_{s,m} x̄_{g,m}
u_{s,m}   ~ Gamma(a0 + τ·Σ_g Z_{s,m,g},  b0 + τ·Σ_g ε_{s,g} x̄_{g,m})
```

and the reported composition maps components onto cell types through the
readout `A`:

```
fraction_{s,c} = Σ_m u_{s,m} A_{m,c} / Σ_m u_{s,m}
```

Because each `x̄_{·,m}` sums to one over genes, the abundance draw has a common
rate and the induced posterior on the fractions is exactly
`Dirichlet(a0 + τ·n_1, …)` — the Dirichlet–multinomial form BayesPrism uses,
obtained here without giving up the closed-form conjugate update.

### Where the reference comes from

Annotated cells are clustered into archetypes by Leiden on the cell embedding,
and each archetype's profile is the empirical mean of its member cells,
shrunk toward the pooled profile by `--archetype-shrink` pseudo-counts so no
gene has rate zero. The readout `A` is the mean annotation posterior over each
archetype's cells, so label uncertainty rides along instead of being argmaxed
away.

Profiles are **measured, not reconstructed**. An earlier version rebuilt them
from the gene embedding as `exp(ρ_g·t_c + a_g)`; that reconstruction reproduced
a true cell-type profile at only r ≈ 0.8 even with the anchor on the true
centroid, which capped how well any composition could fit. On a real pseudobulk
benchmark the empirical reference gives r = 0.91 against 0.72 for the
reconstruction, and that path has been removed.

### Pooling over granularities

`--archetypes` takes several targets and runs one chain each, pooling the draws.
The partition is a nuisance parameter — nothing in the problem picks a
particular clustering — so conditioning on one is overconfident. Pooling also
yields a between-chain R̂, which reports partition disagreement rather than
hiding it.

## Outputs

| file | contents |
|---|---|
| `{out}.fractions.tsv` | posterior-mean composition, samples × cell types |
| `{out}.fractions_ci.tsv` | mean, sd and 95 % interval per (sample, cell type) |
| `{out}.abundance.tsv` | un-normalised per-type abundance |
| `{out}.expression/{celltype}.parquet` | `E[Z_{s,c,g}]`, samples × genes — the DE input |
| `{out}.anchors.parquet` | component coordinates in the embedding |
| `{out}.archetypes.parquet` | per component: readout row and cell count |
| `{out}.membership.tsv.gz` | cell → component, two headerless columns |
| `{out}.abundance_component.parquet` | abundances **before** the readout |
| `{out}.residual.tsv` | posterior-predictive deviance and Pearson per sample |
| `{out}.convergence.tsv` | split or between-chain R̂ and ESS |
| `{out}.trace.tsv.gz` | fraction trace including warmup |
| `{out}.deconvolve.json` | run manifest |

The last four diagnostics exist because the readout is the one part of the model
the data cannot correct: it is estimated upstream and applied as a fixed linear
map. Writing it out, with the membership that produced it and the pre-readout
abundances, is what lets the reported composition be checked against a known one
at a *perfect* abundance vector — separating a wrong reference from a wrong fit.
All three share one component label and join on it.

## Known limits

Read these before trusting a number.

**Fractions are mRNA shares, not cell shares.** Profiles are normalised over
genes, so an abundance is mRNA mass. Cell types differ substantially in mRNA per
cell — on one real reference, DC carried 2.1× the mean and HSPC 0.44× — so the
two quantities are not interchangeable. Converting requires per-type mRNA
content, which is not yet modelled.

**The reported range is compressed.** `A` is row-stochastic, so `f = Aᵀu` is an
averaging operator: it can only contract dispersion, never expand it. Measured
per-type regression slopes of predicted on true run 0.26–0.71, and about 68 % of
total error is this affine mis-scaling. Direction and ranking are reliable;
effect *magnitudes* are understated. If you have samples of known composition, a
per-type slope and intercept correction takes MAE from 0.056 to 0.018.

**Credible intervals are too narrow.** Coverage of the nominal 95 % interval is
around 10 %. The interval is roughly the right *size* for sampling noise —
median |bias|/sd is 1.5 once the affine bias is removed — and wrong in
*location*. This is a bias wearing a variance costume, so `--count-scale` will
not fix it; it only widens an interval that is centred in the wrong place. Do
not attach a p-value to anything derived from these intervals.

**Closely related types may be unresolvable.** Separability is bounded by the
reference, not the sampler. On one real panel CD4_T and CD8_T profiles had
cosine similarity 0.972, the canonical markers carried 0.23 % of T-cell mRNA,
and CD4 itself showed no fold-change between them at the mRNA level. Where two
types are that close, their split is decided by the prior, not the data. Check
`{out}.archetypes.parquet` for diffuse readout rows before trusting a fine
distinction.

**Leakage inflates in-sample benchmarks.** If the bulk is built from cells that
are also in the reference, an empirical reference partly reads back counts it is
meant to explain. Use `--archetype-cells` with the complement of the bulk cells
for a clean estimate.
