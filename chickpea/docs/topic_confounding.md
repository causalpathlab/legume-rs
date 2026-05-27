# How topics confound peak→gene inference

Why the marginal peak–gene association — and therefore the SuSiE-RSS PIPs and
the knockoff FDR built on it — is *structurally* confounded by shared cell-state
topics. This is derived from the generative model that
`data-beans-sim multiome` actually samples, so the conclusions are exact for the
simulation and qualitatively true for real paired data.

Companion to `peak_to_gene_math.md` (the estimator) — this file explains what the
estimator is fighting against.

---

## 1. Generative model

Cells `i` (or pseudobulk samples `s`) carry topic proportions; topics drive both
modalities through a **shared** peak dictionary, and a gene's RNA is built from
*its* linked peaks' contributions, topic by topic.

```
ATAC:  A_{ip} ~ Poisson( ρ_i · Σ_t θ_{it} β_{tp} )
RNA:   X_{ig} ~ Poisson( τ_i · Σ_t θ_{it} Σ_q β_{tq} M_{gq} )
```

- `θ_{it}` topic proportion of cell `i` in topic `t ∈ [T]`.
- `β_{tp}` topic `t`'s loading on peak `p`. Matrix `B ∈ ℝ^{T×P}`, `B_{tp}=β_{tp}`.
- `M_{gq} ∈ {0,1}` cis link indicator: peak `q` regulates gene `g`. `M ∈ {0,1}^{G×P}`.
- `ρ_i, τ_i` per-cell depth (scalars).

The RNA dictionary is **derived**, not free:

```
W_{gt} = Σ_q β_{tq} M_{gq}   ⇔   W = M Bᵀ ∈ ℝ^{G×T}
```

so `X_{ig} ~ Poisson(τ_i Σ_t θ_{it} W_{gt})`. **A gene is a topic mixture of its
own peaks.** This single fact is the root of everything below: the gene shares
the *exact same* topic axes as its peaks, because it is built from them.

---

## 2. Latent rates (the mean structure)

Confounding lives in the conditional mean, not the Poisson noise, so work with
the depth-normalized rates over samples `s ∈ [S]` (write `Θ ∈ ℝ^{T×S}`,
`Θ_{ts}=θ_{ts}`):

```
peak rate   a_p(s) = Σ_t β_{tp} θ_{ts}        ⇒   A = Bᵀ Θ        (P×S)
gene rate   c_g(s) = Σ_t W_{gt} θ_{ts}        ⇒   C = W Θ = M Bᵀ Θ (G×S)
```

Both `A` and `C` are linear images of the **same** `Θ`. log1p + centering (as in
`peak_to_gene_math.md §1`) variance-stabilizes the Poisson part; it does not
touch the fact that `A` and `C` share `Θ`.

---

## 3. The induced peak–gene covariance

Center topics over samples (`Θ̃`), define the **topic covariance**
`Σ_Θ = (1/S) Θ̃ Θ̃ᵀ ∈ ℝ^{T×T}`. The cross-modality covariance the marginal
statistic estimates is:

```
Cov_s(a_p, c_g) = Σ_{t,u} β_{tp} (Σ_Θ)_{tu} W_{gu}
                = [ Bᵀ Σ_Θ Wᵀ ]_{p,g}
```

Substitute `Wᵀ = (M Bᵀ)ᵀ = B Mᵀ`:

```
Cov(A, C) = Bᵀ Σ_Θ B Mᵀ = K Mᵀ ,     K := Bᵀ Σ_Θ B ∈ ℝ^{P×P}
```

Per gene `g`, with `m_g ∈ {0,1}^P` its true link vector (a row of `M`):

```
┌──────────────────────────────────────────────────────────────┐
│  signal_p(g) = Cov_s(a_p, c_g) = (K m_g)_p = Σ_{q: M_{gq}=1} K_{pq}  │
└──────────────────────────────────────────────────────────────┘
```

The marginal evidence for peak `p` is **not** "is `p` a true peak of `g`"; it is
"how topic-similar is `p` to the true peaks of `g`", measured by the kernel `K`.

---

## 4. The topic kernel `K` is the confounder

```
K_{pq} = Σ_{t,u} β_{tp} (Σ_Θ)_{tu} β_{uq} = ⟨β_{·,p}, Σ_Θ β_{·,q}⟩
```

`K` is the peak–peak similarity *induced by topics*: it is large whenever peaks
`p` and `q` load on the same (co-varying) topics. Two regimes:

- **Ideal (`K` diagonal).** If peaks were topic-orthogonal, `K_{pq}=0` for `p≠q`,
  so `signal_p(g) = K_{pp} M_{gp}` — nonzero **only** at true peaks. Marginal
  association alone would identify `M`. Fine-mapping would be trivial.

- **Reality (`K` dense, low rank).** Peaks share topics, so `K` has large
  off-diagonal blocks. A **non-causal** peak `p` inherits
  `signal_p(g) = Σ_{q true} K_{pq} > 0` purely from its topic overlap with the
  gene's true peaks. That is the confounding — and it is *real co-variation in
  the data*, not a finite-sample artifact, so **no marginal statistic can remove
  it.**

`K = Bᵀ Σ_Θ B` has **rank ≤ T** (number of topics). Hold onto this.

---

## 5. The method's `(z, R)` are both functions of `K`

The estimator's own quantities (`peak_to_gene_math.md §4–5`) are, in expectation:

```
peak–peak LD:  Cov(a_p, a_q) = (Bᵀ Σ_Θ B)_{pq} = K_{pq}
               ⇒  R = D^{-1/2} K D^{-1/2}        (D = diag K) — R is the correlation form of K
marginal z:    z_p ∝ signal_p(g) / sd(a_p) ∝ (K m_g)_p / √K_{pp}
```

So the RSS pair fed to SuSiE is, up to diagonal scaling,

```
z ∝ K m_g ,     R ∝ K .
```

This is *consistent* (same `K`), which is why §6 of the math doc calls them a
self-consistent RSS pair — but it also pins the failure mode precisely.

---

## 6. Why fine-mapping cannot fix it: rank deficiency

SuSiE-RSS inverts `z ≈ R b` for a sparse `b`. With `z ∝ K m_g` and `R ∝ K`, the
*true* `b = m_g` is a solution. But `K` has rank ≤ `T`, so within a cis block of
`C > T` peaks the system is **rank-deficient**:

```
K m_g  =  K (m_g + n)   for every  n ∈ null(K).
```

`m_g` and `m_g + n` produce **identical** `z` and live under the **same** `R`.
Any peak that loads on the gene's active topics is an exchangeable substitute for
the true peak — there is no information in `(z, R)` to break the tie. SuSiE adds
a sparsity prior, but among the many *sparse* `b` consistent with `z` (the true
peak vs. each of its topic-twins) it has no signal to pick the right support.

Consequences, all observed in the e2e run:

- **PIP mass spreads across topic-twins**, or lands on the wrong one.
- **`Σ_j PIP_j → L`** per gene (each of the `L` single-effect components confidently
  claims a *different* topic-twin), instead of `≈ |m_g|`. We measured median
  `ΣPIP ≈ 8` for `L=10`, with genes showing 6+ peaks at `PIP=1.0`.
- The knockoff null `z̃ ~ N(0, R)` conditions on `R ∝ K` only, so it does **not**
  render the topic-twins null — they keep large `|z|` — and the FDP blows past
  the target `q` (empirical 0.6–0.9 at `q=0.1`).

In cis this is worst: nearby peaks are co-regulated (share topics), so the local
`K` block is especially low-rank and the true peak is least distinguishable from
its neighbors.

---

## 7. Confounder *and* mediator: the identifiability floor

Note `signal_p(g) = (K m_g)_p` flows **entirely** through `K = Bᵀ Σ_Θ B` — i.e.
through topics. So in this generative model:

- topics are the **confounder** for a non-causal peak (`gene ← θ → peak`), **and**
- topics are the **mediator** for a causal peak (the true peak acts on the gene
  *only* via its topic loadings `β_{·,q}`, by construction `W = M Bᵀ`).

There is no direct peak→gene channel here. Therefore **adjusting topics away
removes the signal along with the confound** — you can only recover whatever
covariation survives the projection orthogonal to `Θ`:

```
identifiable part of peak p for gene g  =  Cov( a_p − P_Θ a_p ,  c_g − P_Θ c_g )
```

with `P_Θ` the projection onto the topic subspace. In a *pure* topic model both
residuals are ~0, so the honest answer is **"not identifiable from pseudobulk"** —
and a correctly deconfounded estimator should return *no* discovery rather than a
confident wrong one. Real data is only identifiable to the extent a peak has a
**topic-orthogonal**, gene-predictive component (a local/direct accessibility
channel beyond the global programs). That residual is the entire ceiling.

This is why §7 of the math doc ("genuine co-variation, no marginal statistic
removes it") is right that the marginal can't fix it — but LD-aware SuSiE can't
either, because the LD it uses *is* the confounder `K`.

---

## 8. What this implies for deconfounding

To make `(z, R)` reflect the topic-orthogonal residual (the IPW/TMLE-in-summary
idea), partial the topic basis `C ≈ Θ̃ᵀ` (or its leading components — note these
are exactly the top sample-factors `V` from the ATAC SVD) out of the moments
before fine-mapping:

```
z̃_p ∝ a_pᵀ M_⊥ c_g ,   R̃_{pq} ∝ a_pᵀ M_⊥ a_q ,   M_⊥ = I − C(CᵀC)⁻¹Cᵀ
```

i.e. drop the top-`m` topic eigen-components of `K`/`R` and the matching
projection of `z`. Then:

- the SuSiE inputs measure residual (topic-orthogonal) association;
- the knockoff null `z̃_ko ~ N(0, R̃)` becomes the *conditional-on-θ* null, so the
  topic-twins are finally null and the FDP can track `q`;
- `m` (how many topic axes to remove) is the bias–variance dial: too small leaves
  residual confounding (FDP stays high), too large removes signal (power → 0).
  The principled setting is the smallest `m` whose knockoff/permutation FDP hits
  `q`. (The current `--ko-ridge` knob is only a crude proxy for this.)
- **df correction:** removing `rank(C)` directions costs degrees of freedom; the
  RSS effective size becomes `S − rank(C)` and must enter the z scaling, or the
  null re-inflates.

The floor from §7 still binds: where a link is purely topic-mediated, `z̃, R̃ → 0`
and SuSiE correctly finds nothing. Deconfounding fixes *calibration*, not the
*information content* — it converts confident false positives into honest nulls.

---

## 9. Empirical fingerprints (e2e on `data-beans-sim multiome`)

The derivation predicts, and the end-to-end run confirms:

| prediction (this doc)                              | observed                          |
|----------------------------------------------------|-----------------------------------|
| marginal inflated by topic overlap (`K m_g`)       | raw PIP precision ~0.08 @ PIP≥0.9 |
| `ΣPIP → L`, multiple `PIP=1` per gene              | median ΣPIP ≈ 8 (L=10); 6+ at 1.0 |
| LD = `K` can't break topic-twins (rank ≤ T)        | top-1 plateaus ~0.7; wrong loci   |
| knockoff null on `R∝K` misses topic-twins          | FDP 0.6–0.9 at q=0.1              |
| `--ko-ridge` is a power knob, not an FDR fix       | ridge↑ ⇒ recall↑, FDP stays high  |

All numbers are single draws from a non-deterministic projection (unseeded
`rnorm`), but the qualitative pattern is robust across runs — it is structural,
predicted here from `K = Bᵀ Σ_Θ B`, not a tuning accident.
