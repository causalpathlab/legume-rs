# `peak-to-gene`: math derivation

Embedding-centric, summary-statistics fine-mapping of cis peak→gene links from
paired RNA + ATAC pseudobulk. Both genes and peaks become vectors in one shared
ATAC-derived latent space; the marginal association and the peak–peak LD are
inner products in that space, fed to SuSiE-RSS.

Code: `src/p2g/{embed,finemap,run}.rs`.

---

## 1. Notation

- `S` pseudobulk samples (metacells), index `s`. Built by random projection +
  multi-level collapse of the matched cells (RNA and ATAC share the **same** `S`
  samples because barcodes are matched).
- RNA: gene `g` has a rate vector `x_g ∈ ℝ^S` (Gamma posterior-mean intensity
  per sample). ATAC: peak `p` has `y_p ∈ ℝ^S`. `G` genes, `P` peaks.
- `log1p(v) = ln(1+v)`, applied elementwise. Overdispersed counts/rates are
  variance-stabilized on the log scale, so we work with `log1p` throughout
  (a Poisson GLM would re-impose Var = mean and re-inflate).

Stabilize and center over samples:

```
Ã_{p,·} = log1p(y_p) − mean_s log1p(y_p)      (Ã ∈ ℝ^{P×S}, each peak row centered)
x̃_g    = log1p(x_g) − mean_s log1p(x_g)       (x̃_g ∈ ℝ^S, gene centered)
```

---

## 2. ATAC embedding (global, pb-level)

Rank-`d` randomized SVD of the standardized ATAC pb matrix:

```
Ã = U Σ Vᵀ,   U ∈ ℝ^{P×d} (UᵀU = I),  Σ = diag(σ₁..σ_d),  V ∈ ℝ^{S×d} (VᵀV = I)
```

- **Peak embedding** `W = U Σ ∈ ℝ^{P×d}`; row `W_p` is peak `p`'s vector.
- **Sample factors** `V ∈ ℝ^{S×d}` — orthonormal latent cell-state axes.
- **Denoised peak profile** over samples: `ŷ_p = (U Σ Vᵀ)_{p,·}ᵀ = V W_pᵀ ∈ ℝ^S`.

At pb level `S` is small and rates are dense, so this SVD is cheap; truncating
to `d` denoises the zero-inflated ATAC by borrowing strength across peaks/samples.

## 3. Gene in the same embedding

Project the gene onto the ATAC sample-factor basis:

```
g̃_g = Vᵀ x̃_g ∈ ℝ^d
```

So **both feature types are name-keyed `d`-vectors in one space**: peak `p ↦ W_p`,
gene `g ↦ g̃_g`. Everything below is inner products between them.

---

## 4. Marginal peak→gene z (univariate regression, in embedding space)

Regress the gene's centered log-profile on the denoised peak profile (single
slope, no intercept since both are centered):

```
x̃_g = β · ŷ_p + ε,   ε ~ N(0, σ_ε² I_S)
```

**OLS slope.** Using `ŷ_p = V W_pᵀ` and `VᵀV = I`:

```
⟨x̃_g, ŷ_p⟩ = x̃_gᵀ V W_pᵀ = (Vᵀx̃_g)ᵀ W_pᵀ = g̃_g · W_p
⟨ŷ_p, ŷ_p⟩ = W_p Vᵀ V W_pᵀ = ‖W_p‖²
⇒  β̂ = (g̃_g · W_p) / ‖W_p‖²
```

**Residual / SE.**

```
RSS = ‖x̃_g‖² − β̂² ‖ŷ_p‖² = ‖x̃_g‖² − (g̃_g · W_p)² / ‖W_p‖²
σ̂²  = RSS / (S − 2)
SE(β̂) = σ̂ / ‖ŷ_p‖ = σ̂ / ‖W_p‖
```

`RSS ≥ 0` always: `(g̃·W_p)²/‖W_p‖² ≤ ‖g̃‖² ≤ ‖x̃‖²` (Cauchy–Schwarz, then `V`
is an orthonormal projection so `‖Vᵀx̃‖ ≤ ‖x̃‖`).

**z-score.**

```
z_{g,p} = β̂ / SE(β̂) = (g̃_g · W_p) / (σ̂ · ‖W_p‖)  =  (g̃_g · Ŵ_p)/σ̂ ,   Ŵ_p = W_p/‖W_p‖
```

Under H₀ (gene unrelated to peak `p`'s accessibility module), `z ~ t_{S−2} ≈ N(0,1)`.
Denoising the regressor `ŷ_p` reduces errors-in-variables attenuation (more
power) without breaking the null.

**PVE (winner's-curse) shrinkage**, `n = S`:

```
z ← z · √( (n−1) / (z² + n−2) )
```

---

## 5. Peak–peak LD (in embedding space)

```
R_{pq} = corr(ŷ_p, ŷ_q),   ⟨ŷ_p, ŷ_q⟩ = W_p VᵀV W_qᵀ = W_p · W_q,   ‖ŷ_p‖ = ‖W_p‖
⇒  R_{pq} = (W_p · W_q) / (‖W_p‖ ‖W_q‖) = Ŵ_p · Ŵ_q   (cosine of peak embeddings)
```

`R` has unit diagonal, is PSD, and has rank ≤ d. `z` and `R` are built from the
**same** `{W_p}` (and `g̃`), so they form a self-consistent RSS pair.

---

## 6. SuSiE-RSS fine-mapping

Per gene, restrict to its cis peaks (within `±cis_window` of the TSS; capped to
the `max_cis` nearest). Marginal `z ∈ ℝ^C`, LD `R ∈ ℝ^{C×C}`.

**Model** (Zhu & Stephens 2017; Wang et al. 2020). The RSS likelihood is
`z ~ N(R β, R)` with **residual variance fixed at 1** (z are standardized, so the
noise level is unit). Sum of `L` single effects: `β = Σ_{l=1}^L b_l`, each
`b_l = γ_l β_l` with `γ_l` one-hot over the `C` peaks (one causal peak per
component) and prior `β_l ~ N(0, σ²₀)`.

**IBSS / CAVI updates** (`σ² = 1`), maintaining `b̄ = Σ_l α_l ⊙ μ_l` and `R b̄`:

```
for each component l:
    r_l = z − R (b̄ − α_l⊙μ_l)                 # residualized marginal
    s²_j   = 1 / (R_jj + 1/σ²₀)                # posterior variance (per peak)
    μ_{lj} = s²_j · (r_l)_j                     # posterior mean
    logBF_j = ½ ln(s²_j/σ²₀) + ½ μ_{lj}²/s²_j + ln π_j
    α_l = softmax_j(logBF_j)                    # single-effect selection
iterate to convergence (max |Δα| < 1e-4)
```

`π_j` is the per-peak inclusion prior (uniform `1/C` by default). Outputs:

```
PIP_j      = 1 − Π_l (1 − α_{lj})
effect_j   = Σ_l α_{lj} μ_{lj}
effect_var = Σ_l α_{lj} (s²_j + μ_{lj}²) − effect_j²
```

**Why `σ²` is fixed, not estimated.** `var(z)` (or `var(ỹ)` in an eigenspace
form) includes the signal; estimating residual variance from it lets an inflated
signal be absorbed as "noise", collapsing all PIPs to the `1/C` prior. Fixed
`σ²=1` is the standard SuSiE-RSS choice and is what makes PIPs concentrate.

---

## 7. Why fine-mapping (and not just the marginal)

Shared cell-state (topic) structure makes a gene co-vary with **many** cis peaks
at once — genuine co-variation, not a statistical artifact, so no marginal
statistic removes it. The cis-window restriction limits candidates; the
**LD-aware** SuSiE-RSS step then explains the marginal `z` pattern with as few
peaks as the correlation structure allows, so colinear peaks share inclusion mass
instead of all scoring as independent links. PVE shrinkage tames z magnitudes.

---

## 8. Output

Per tested (gene, cis-peak): a BGZF BED sorted by `(chr,start,end)`:

```
#chr  start  end  peak_id  gene_id  pip  effect_mean  effect_std  z  distance
```

`distance = |peak_midpoint − TSS|`. All tested pairs are written; the PIP
threshold only drives a summary log line. With `--fdr q > 0`, two columns are
appended — `w_stat` (knockoff importance) and `selected` (0/1) — see §10.

---

## 9. Hierarchical pseudobulk refinement (`--num-levels`)

The collapse builds `L` nested resolutions and refines pb assignments using the
level hierarchy (bottom-up coarsening + sibling-constrained, top-down
refinement; data-beans-alg `refine`). `--num-levels L` sets that depth; we then
use the **refined finest level** as the single sample axis (`S` pseudobulks).

We do **not** pool levels as extra columns: nested levels are repartitions of the
same cells, so pooling adds correlated/redundant columns, not independent signal
— empirically it *reduced* peak-gene recovery on simulated data. The value of the
hierarchy is a better-optimized finest partition, not more columns. `L = 1`
reproduces the single-level pipeline.

---

## 10. Knockoff FDR (`--fdr q`)

SuSiE PIPs rank links but don't bound the error rate across the (many) reported
pairs. An optional **GhostKnockoff** filter (He et al. 2022; model-X knockoffs,
Candès et al. 2018) on each gene's `(z, R)` gives genome-wide FDR control.

**Construction.** Under the RSS null `z ~ N(0, R)`. Regularize `R_λ = (1−λ)R + λI`
(the rank-≤d cosine `R` is otherwise singular). Equicorrelated diagonal
`s = min(1, 2λ_min(R_λ))`, `D = sI` (so `2R_λ − D ⪰ 0`). Sample knockoff
z-scores

```
z̃ | z  ~  N( (R_λ − D) R_λ⁻¹ z ,  2D − D R_λ⁻¹ D )
```

which makes `(z, z̃)` swap-exchangeable: `cov(z,z̃) = R_λ − D`, `var(z̃) = R_λ`.
Knockoffs are drawn from the **raw** (pre-PVE) `z` so the null holds.

**Importance (v1 — z-score contrast).**

```
W_j = |z_j| − |z̃_j|
```

Flip-sign holds by construction (swapping `z_j ↔ z̃_j` negates `W_j`). This
marginal statistic only needs `(z, z̃)` exchangeability, so it is robust to the
embedding `R` being a rank-≤d *approximation* of the true z-correlation. A
SuSiE-PIP importance — running the augmented `(z_aug, R_aug)` with
`R_aug = [[R_λ, R_λ−D],[R_λ−D, R_λ]]` through `finemap_gene` and taking
`W_j = PIP_j − PIP_j̃` — is **deferred to v2**: it is more powerful but, because
IBSS leans on the full (misspecified) `R`, it inflated FDR on the null sim below.

**Pooled filter.** Collect all `(peak,gene)` `W` and take the knockoff+ threshold

```
τ_q = min{ t > 0 : (1 + #{W ≤ −t}) / (#{W ≥ t} ∨ 1) ≤ q },   select W_j ≥ τ_q
```

Pooling across genes (each gene's knockoffs are independent; per-gene flip-sign
holds) controls FDR over the genome-wide **link** set — per-gene `C` is too small
to clear the `1/q` detection threshold. `effect_*`/`pip` columns are unchanged;
`selected`/`w_stat` are added.

**Validation.** Two tests in `src/p2g/knockoff.rs`: a clean-RSS calibration test
(`pooled_fdr_is_controlled`) and an *embedding-pipeline* null test
(`embedding_knockoff_controls_fdr_on_nulls`) that drives `build_atac_embedding` +
`cis_link_stats`, with null peaks loading on factors orthogonal to each gene's
program. The z-score contrast holds FDR there (FDP ≈ 0 at q = 0.1); the
SuSiE-PIP contrast did not (FDP ≈ 0.35), which is why it waits for v2.
