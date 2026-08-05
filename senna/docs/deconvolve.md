# `senna deconvolve` — bulk deconvolution on a feature embedding

Estimate, for each **bulk** RNA-seq sample, what fraction of its signal came from
each cell type — and how many counts of each *gene* came from each cell type.
The second output is the useful one: it lets you do differential expression
*within* a cell type across samples, deconfounded from composition shifts.

This is BayesPrism's deliverable, but the reference is built from a **learned
gene embedding** rather than from empirical per-cell-type means.

```
senna bge <sc.zarr> --skip-etm -o ref          # gene embedding (raw Poisson ρ)
senna deconvolve --from ref.senna.json \
                 -m markers.tsv \
                 --bulk bulk.parquet -o out
```

---

## 1. The idea in one picture

Everything lives in one `H`-dimensional latent space learned by `bge`:

```
            embedding space (H dims)
   gene g  ──►  ρ_g        (how gene g responds to a position)
   cell n  ──►  z_n        (where cell n sits)
   type c  ──►  t_c        (ANCHOR: where cell type c sits)
   bulk s  ──►  z_s        (where bulk sample s sits, by projection)

   Poisson rate of gene g at position t :   μ = exp(ρ_g · t + a_g)
```

`bge` trains a Poisson model on the single-cell data where the rate of an edge
`(gene g, cell n)` is

```
λ_{g,n} = exp( ρ_g · z_n + a_g + b_n )
```

with `a_g` a per-gene offset (`feature_bias`) and `b_n` a per-cell depth term.
Deconvolution reuses exactly this rate — that is the whole reason the reference
is coherent with the trained model.

---

## 2. What is an "anchor"?

> An **anchor** `t_c` is a single `H`-dimensional coordinate that represents
> *"where an average cell of type c sits"* in the embedding.

It is the only thing that distinguishes one cell type from another in this
model. The entire reference expression profile of type `c` is *generated* from
its anchor:

```
μ_{g,c} = exp( ρ_g · t_c + a_g )      ← expected expression of gene g in type c
```

So instead of storing a `G × C` reference matrix of per-cell-type gene means,
we store `C` anchors (`C × H` numbers) and *reconstruct* the profile on demand.

**Where the anchor comes from.** From marker genes. For cell type `c` we take
its marker genes, look up each one's coordinate in the **co-embedding**
(`feature_embedding.parquet`, which places every gene at a softmax-weighted
average of the cell embeddings), and take their IDF-weighted mean:

```
t̂_c = Σ_{g ∈ markers(c)} w_g · coembed_g  /  Σ_{g ∈ markers(c)} w_g
```

This works because a gene's co-embedding coordinate is roughly "the centre of
mass of the cells that express it", so averaging a type's markers lands near
that type's cell centroid. This is the same construction
`senna annotate-by-projection` uses to label cells.

**Why the anchor has uncertainty.** Markers are noisy: some are shared between
types, some are weak. So each anchor carries a Gaussian prior

```
t_c ~ N( t̂_c , Σ_c )
```

with `Σ_c` estimated from the scatter of that type's marker coordinates —
isotropic `σ_c²·I` by default, or a shrunk full `H×H` with `--anchor-cov full`.
A cell type whose markers agree gets a tight prior; an ambiguous one gets a
loose prior, and that looseness propagates into wider fraction intervals. This
is the channel by which *annotation uncertainty enters the deconvolution*.

---

## 3. The generative model

For bulk sample `s`, gene `g`, cell type `c`:

```
μ_{g,c} = exp( ρ_g · t_c + a_g )                     reference profile
ε_{s,g} ~ Gamma(r, r)                                per-gene overdispersion (mean 1)
y_{s,g} ~ Poisson( ε_{s,g} · Σ_c w_{s,c} · μ_{g,c} )  observed bulk counts
```

`w_{s,c} ≥ 0` is the abundance of type `c` in sample `s`; the reported fraction
is `π_{s,c} = w_{s,c} / Σ_c w_{s,c}`.

Poisson rates **add**, which is what makes a bulk sample literally the sum of
its cells — this is why the mixture is exact rather than an approximation.

---

## 4. The algorithm

**Step 0 — load the reference.** Read the gated `ρ` (`dictionary.parquet`), the
co-embedding (`feature_embedding.parquet`) and `a_g` (`feature_bias.parquet`).

**Step 1 — build anchors.** Marker TSV → per-type IDF-weighted centroid `t̂_c`
and spread `Σ_c` (§2).

**Step 2 — project the bulk.** Each bulk sample is embedded by the *same*
frozen-`ρ` Poisson solver that placed the reference cells
(`graph_embedding_util::cell_projection::project_cells`, an IRLS/Newton solve
with a ridge prior). This is the cross-platform bridge: bulk and single cells
end up in one geometry. The projected `z_s` also warm-starts the fractions via a
simplex least-squares fit against the anchors.

**Step 3 — Gibbs sampling.** Each sweep, in order:

| update | distribution | why it's cheap |
|---|---|---|
| `ε_{s,g}` | `Gamma(r + τy, r + τλ)` | conjugate |
| `Z_{s,·,g}` | `Multinomial(y_{s,g}, p)`, `p_c ∝ w_{s,c} μ_{g,c}` | the gene-count split; `ε` cancels |
| `w_{s,c}` | `Gamma(a₀ + τΣ_g Z, b₀ + τΣ_g ε μ)` | Gamma–Poisson conjugate |
| `t_c` | elliptical slice sampling under `N(t̂_c, Σ_c)` | non-conjugate, so ESS |

Samples are independent given the anchors, so the per-sample updates run in
parallel (rayon) with one seeded RNG stream each. The anchor update pools
sufficient statistics across all samples, which is what couples them.

`Z` is the BayesPrism gene-split: it says how many of gene `g`'s counts in
sample `s` came from type `c`. Its posterior mean is the per-cell-type
expression tensor.

**Step 4 — summarise.** Fraction means/sds come from running moments; credible
intervals from streaming P² quantiles (`matrix_util::running_quantile`), so no
per-draw arrays are stored.

---

## 5. Knobs

| flag | default | meaning |
|---|---|---|
| `--warmup` / `--draws` / `--thin` | 500 / 500 / 1 | Gibbs schedule |
| `--nb-dispersion` | 10000 (≈Poisson) | NB `r`; smaller = more overdispersion. **Fixed, not sampled** — freely sampling it is non-identifiable against the fractions (`ε` competes with `w` through the per-type exposure) |
| `--count-scale` | 1.0 | likelihood temperature `τ` (power posterior). `τ<1` discounts the count evidence and widens intervals; the calibration knob |
| `--anchor-cov` | isotropic | `isotropic` or `full` (shrunk `H×H`) anchor prior |
| `--anchor-prior-scale` | 1.0 | multiplies `σ_c`; smaller = anchors held closer to their marker-derived position |
| `--frac-prior-shape/rate` | 1.0 / 1.0 | Gamma prior on `w` |
| `--project-ridge` | 1.0 | ridge for the bulk projection |

---

## 6. Outputs

| file | contents |
|---|---|
| `{out}.fractions.tsv` | `S × C` posterior-mean fractions (rows sum to 1) |
| `{out}.fractions_ci.tsv` | long form: `sample, celltype, mean, sd, q2.5, q97.5` |
| `{out}.abundance.tsv` | posterior-mean `w` (unnormalised) |
| `{out}.expression/{celltype}.parquet` | `S × G` posterior-mean `E[Z]` per type — **DE-ready** |
| `{out}.sample_embedding.parquet` | `S × H` projected bulk (plot next to cells/anchors) |
| `{out}.anchors.parquet` | `C × H` posterior anchors |
| `{out}.residual.tsv` | per-sample Poisson deviance + Pearson fit |

---

## 7. Requirements and caveats

**Must be `bge --skip-etm`.** The raw Poisson `ρ` is persisted as
`dictionary.parquet` only in that mode; the default ETM run overwrites the
dictionary with the topic dictionary `β`. Deconvolve detects this from the
dictionary's *content* (a `β` has columns that are gene log-simplexes) rather
than from manifest bookkeeping, because the manifest's `latent` /
`cell_embedding` slots have swapped roles between bge versions.

**The gate is already baked in.** `bge`'s feature gate multiplies every feature
loading, and `materialize_e_feat` writes the *gated* values into `e_feat`. So
`dictionary.parquet` already is the loading that enters the Poisson rate — do
**not** apply a separate gate correction.

**The reported fraction is cell-like, not mRNA-like.** `μ_{·,c}` is the expected
count vector of *one* average cell of type `c` and is **not** normalized, so
`w_{s,c}` is a cell-count-like quantity and `w/Σw` behaves as a cell fraction.
The mRNA fraction would be `w_c M_c / Σ w M`, where `M_c = Σ_g μ_{g,c}` is the
type's total output. This is the opposite convention to BayesPrism (§9), which
normalizes its reference so its `θ` *is* the mRNA fraction. On the benchmark
`M_c` varies by only ±3%, so the two coincide there and the distinction is
untested; in tissue with very different RNA content per type (neurons vs
erythrocytes) it matters, and `M_c` is the conversion factor.

**Counts, not TPM.** Bulk values are rounded to integers for the multinomial
split.

**Topic-family sources (`masked-topic`, `topic`, `itopic`, `masked-vae`) are
disabled.** They were once offered as an "approximation"; benchmarked on
identical data they score **Pearson 0.08** (noise) against `bge --skip-etm`'s
0.99, so they now fail with an explanatory error rather than return
plausible-looking numbers.

The reason is not that a topic run lacks a feature embedding — it writes
`feature_embedding.parquet` (`D × H` ρ) just like `bge`. It is that **ρ's
partner is missing**: under `bge`, cells and genes share one `H`-dim space
(`cell_embedding` is `N × H`), so `ρ_g · z` is meaningful and bulk can be
projected into it. Under a topic model, ρ pairs with the *topic* embeddings α
(`β = log_softmax_d(ρ·αᵀ)`) while cells live on the `K`-simplex — `latent` is
`N × K` log θ and there is no `H`-dim cell representation at all. Projecting
bulk into a space containing no cells is ill-posed, and the softmax head has no
per-gene additive bias for `a_g` to be.

Two coherent reworks, both of which skip the projection entirely:

1. **Read the reference off β.** `dictionary_empirical.parquet` is already a
   full-resolution per-topic gene simplex — precisely BayesPrism's normalized
   `φ` — and `dispersion.parquet` is a per-gene NB dispersion that would replace
   the hand-set scalar `--nb-dispersion`. Markers would then be needed only to
   map topics onto cell types.
2. **Encode the bulk.** A topic run ships its encoder (`safetensors` +
   `model.json` + `feature_mean.parquet`), and `senna predict` already performs
   encoder-only inference, so a bulk sample can be encoded straight to `θ_bulk`.

Route 1 is also the fix for the reference bottleneck in §8, so a topic run may
end up the *better* deconvolution source rather than the weaker one.

---

## 8. Benchmark (and what is still wrong)

Ground truth via `data-beans-sim` (2000 genes × 3000 cells × 5 types; 20 bulk
mixtures with known Dirichlet proportions):

```
data-beans-sim topic -r 2000 -c 3000 -f 5 --pve-topic 1.0 --depth 5000 -o bench_sc
data-beans-sim bulk  -s bench_sc.zarr -t bench_sc.prop.parquet -n 20 -c 200 -o bench_bulk
```

| metric | value |
|---|---|
| Pearson (est vs true fractions) | **0.986** |
| RMSE | 0.050 |
| 95% CI coverage | **26%** |

**Composition recovery is strong. Uncertainty is not calibrated.** Measured
diagnostics: posterior sd is ~8× smaller than the actual error, and the error
has a systematic per-cell-type component (~25% of MSE).

Investigated causes, with what the data said:

| hypothesis | verdict |
|---|---|
| co-embedding shrinks anchors toward the global centroid | **refuted** — anchors sit within 0.04–0.09 of the true cell-type centroids (cos ≥ 0.997), 96% of true mutual separation |
| mRNA- vs cell-fraction units | **refuted here** — per-type output differs by only ±3% in this simulation |
| ESS anchor drift | **implicated** — posterior anchors end up 0.20–0.98 from truth (5/5 move *away*), yet freezing them (`--anchor-prior-scale 0.01`) *lowers* bias (25%→5%) while *raising* RMSE (0.050→0.088) |
| low-rank reference reconstruction | **the bottleneck** — even at the *true* cell-type centroid, `exp(ρ·t + a)` reproduces the true per-type profile at only r ≈ 0.76–0.87 (raw scale) |

The last row explains the third: because the reconstructed profile is an
imperfect representation of the true one, the sampler moves the anchors far off
their correct positions to compensate, trading bias for fit. The anchors are
absorbing reference-model error.

**Implication.** The accuracy ceiling here is the *reference*, not the sampler
and not the anchors. The natural fix is a hybrid: keep the embedding for what
it is demonstrably good at (projecting bulk, placing anchors from markers,
soft cell-type assignment) but take `μ_{g,c}` from **empirical per-cell-type
pseudobulk** instead of reconstructing it from `ρ`. A cheaper intermediate is a
per-gene multiplicative calibration of `μ` against the reference pseudobulk.

Two further notes on the benchmark itself: the "truth" is the *drawn* Dirichlet
weight rather than the *realised* composition of the 200 sampled cells (a ~0.028
sd discrepancy that no posterior could cover), and `bge` is batch-corrected
while the bulk still carries the simulator's per-gene batch factor.

---

## 9. Relationship to BayesPrism

The **core is the same model**. BayesPrism (Chu et al., *Nat Cancer* 2022) also
augments with a latent `Z_{c,g}` — the reads in the bulk attributable to cell
type `c` and gene `g` — splits each gene's counts multinomially in proportion to
`θ_c φ_{c,g}`, and alternates that with a conjugate draw of the cell-type
weights. Our `Z` step is identical, and our deliverables (fractions + a
per-sample per-cell-type expression tensor for within-type DE) are the same.

The differences:

| | BayesPrism | `senna deconvolve` |
|---|---|---|
| reference | **empirical** per-cell-type mean from scRNA-seq | **reconstructed** from the gene embedding, `exp(ρ·t_c + a)` |
| normalization | `Σ_g φ_{c,g} = 1` for every type | unnormalized; `M_c = Σ_g μ_{g,c}` varies |
| what the weight means | `θ_c` = **mRNA** fraction | `w_c` ≈ **cell** count (`M_c` absorbs mRNA output) |
| weight prior | `Dirichlet(α)` → exact Dirichlet–multinomial conjugacy | independent `Gamma(a₀, b₀ + τM_c)` |
| overdispersion | none (Poisson/multinomial) | optional NB via `ε ~ Gamma(r,r)` |
| tempering | none | optional `τ` (`--count-scale`) |
| reference refinement | **updated-reference (ψ) step** | anchors resampled by ESS |
| across samples | each bulk independent | anchors pool across all samples |

**The precise relationship.** Independent `Gamma(a_c, b)` weights with a
*common* rate `b` induce a `Dirichlet(a_1..a_C)` on the normalized simplex. Our
rate is `b₀ + τM_c`, which is type-specific *only because the reference is
unnormalized*. So if we normalized `μ_{·,c}` to sum to 1, this sampler would
reduce **exactly** to BayesPrism's Dirichlet–multinomial Gibbs, with the NB `ε`
and the temper `τ` as strict additions. In that sense the generative model is
consistent; the divergence is in the reference, not the inference.

**The consequential gap** is the reference. BayesPrism's updated-reference step
exists precisely because a scRNA-seq reference does not match the bulk it is
applied to, and it re-estimates the reference to absorb that mismatch. We have
no equivalent: our anchors are resampled instead, which (§8) moves them off the
true cell-type positions to compensate for reconstruction error rather than
correcting the profile itself. Since the reconstruction is the measured
bottleneck (r ≈ 0.76–0.87), adopting an empirical or calibrated reference is
both the fix and the step that brings us closest to BayesPrism.
