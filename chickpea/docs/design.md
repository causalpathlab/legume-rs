# chickpea peak-to-gene: design

RNA genes and ATAC peaks share one multiome feature axis, fit by the
`graph-embedding-util` two-phase engine (`senna bge --multiome` recipe). In
phase 1 each cis gene's score mixes in the predicted accessibility of its
cis peaks, and an alignment term pulls its own profile toward that
accessibility, so the two modalities share one feature space. The cis gates
are a distance prior; each link is scored by the prior times the data's
evidence.

## Model

```text
λ_u,m = ⟨e_u, μ_m⟩ + b_m + β_u,k(m)                  module score (β: per-unit modality intercept)
ρ_ug  = ⟨e_u, μ_{m(g)} + r_g⟩ + b_g                  the gene's own score
w_gp  = abc_gp · max(0, θ₀ + θ₁ z(log contact_gp))   the gate: a distance prior
w̃_gp  = w_gp / Σ_q w_gq                              the gene's share of each pair
ã_ug  = Σ_p w̃_gp λ_u,m(p)                            ATAC-guided gene activity
η_ug  = (1 − α) ρ_ug + α ã_ug                        the gene's likelihood score (cis genes)
gap   = mean_g var_u(ρ_ug − ã_ug)                    added as align_weight · n_units · gap
```

- **Two couplings, and they need each other.** The mixture (`--mix α`,
  default 0.5) routes part of each cis gene's likelihood through its peaks;
  alone it never pulled the ATAC modules into the gene space. The alignment
  (`--align-weight`, default 0.1) pulls genes and peak modules together;
  alone and strong it squeezed the cell embedding and scattered weakly
  expressed genes. Both on, the ATAC modules sit among the genes of their
  cell types while the gene space and the cell types keep their structure;
  the mixture carries most of the coupling, so the alignment stays light. Genes without cis pairs
  keep `ρ_ug`; phase 2 and the outputs see the mixed rows
  `μ_{m(g)} + (1 − α) r_g + α ν̃_g` (bias alike).
- **The alignment term** is, per cis gene, the variance across the step's
  units of its own score minus its ATAC-guided activity. Centring across
  units drops every bias, since a gene and its peaks' modules sit on
  different baselines.
- **What it moves.** The gene rows, the peak modules' rows and the gates.
  The units' covariance is detached, so flattening the units cannot close the
  gap.
- `θ` is shared by every gene and trains by Adam through the alignment: the
  gate's shape is the distance profile under which a gene's cis peaks track
  the gene. The shares make `ã` a module-scale score and only `θ₁/θ₀`
  matter. A gene whose gates all close pools nothing.
- `--align-weight` sets the term per unit, next to the likelihood;
  `--align-weight 0 --mix 0` is the plain multiome fit.
- The gate reads contact only. A module term in it (`⟨μ_{m(p)}, ρ_g⟩`) would
  repeat what `λ` already carries and reward the peaks whose module looks most
  like the gene, which the gene's own row already explains.
- `β_u,k` is one intercept per unit and non-reference modality, so a unit's
  ATAC:RNA count split is not written into the embedding. `β` alone is not
  identified (a direction shared by one modality's module rows is another
  intercept), so `μ` is held centred within each modality.
- **L2 decay** (`--weight-decay`, default 1e-4): every feature row a phase-1
  step touches — each modality's module and gene rows — shrinks by
  `1 − lr·wd` before its update. Directions the units never vary in carry no
  gradient to hold them, so they decay instead of keeping their random
  start. The pseudobulk and unit rows and the biases never decay. Keep it
  light: at 0.1 (with the pseudobulk rows decaying too) it separated the
  modalities again and lowered the cell embedding's rank.

## How it is computed

Both scores are linear in `e_u`, so the gap is a quadratic form in the
units' covariance `Σ`:

```text
ρ_ug − ã_ug = ⟨e_u, d_g⟩ + const_g      d_g = μ_{m(g)} + r_g − ν̃_g      ν̃_g = Σ_p w̃_gp μ_{m(p)}
var_u(ρ_ug − ã_ug) = d_gᵀ Σ d_g
```

Once per step, the pool: each gene's gate total, then every pair's
`w̃ μ_{m(p)}` and `w̃ b_{m(p)}` scattered onto its gene (`index_add_rows`,
`[pairs × H]` work, no gene × module block), giving `ν̃` and `c̃`. The mixture
and the alignment both read that one pool. The alignment adds `Σ` over the
step's units (`[H × H]`) and `d Σ` (`[cis genes × H]`), once per step, not
per CPU slice. The mixture's `ν̃` and `c̃` are gathered per gene batch and
folded into the gene rows before the gene-level matmul, so no batch tensor
carries a module axis; a step split over CPU threads shares them as detached
leaves, and their gradient goes back through the pool once.

Gene-level module draws go only to modules with a gene level, each weighted by
the unit's count share on them — the same objective, without spending draws on
module-only modules.

## Link score

After training, each pair's evidence is the correlation across pseudobulk
units of the peak module's score and the gene's:

```text
corr_gp  = corr_u(⟨e_u, μ_{m(p)}⟩, ⟨e_u, ρ_g⟩) = μᵀΣρ / √(μᵀΣμ · ρᵀΣρ)     Σ = cov_u(e_u)
score_gp = w̃_gp · corr_gp
```

The model-denoised analogue of a peak–gene expression correlation: the prior
says which peaks are near, the data says which of them move with the gene. A
negative score is a peak that closes where the gene is on. It is a readout
only; nothing trains on it.

## Feature partition

Modality-pure modules, each modality with its own budget: `--feature-modules`
for RNA (default 1024) and `--peak-modules` for ATAC (default 10000). A
peak's row is its module's row, so the ATAC budget sets how finely peaks
resolve; on the order of the genes with their own rows keeps the two sides
comparable.

- **Flat features** (one rate explains a feature's counts over the finest
  pseudobulks, by the coarsener's Poisson homogeneity test) form each
  modality's background module, with no residual. The test runs on the finest
  level only: on pooled coarser levels it ignores overdispersion and calls
  almost every feature non-flat.
- **Scattered genes (RNA).** k-means places a gene whose profile matches no
  other alone. RNA singletons join the background, and the freed module slots
  split the largest groups in two, a split kept only when both halves reach
  the minimum. Re-clustering without them was tried and only promotes the next
  outliers.
- **No size rule on ATAC.** A peak module is never set aside for being small:
  only activity (the flat test above) decides the ATAC background, so a
  single peak with strong, specific activity keeps a module of its own. A
  minimum of ten peaks was tried; at 10k modules it dissolved a third of the
  cis pairs into the background.
- **Genomic blocks (ATAC).** A module never crosses a 10 Mb window of one
  chromosome, so no module sits in every gene's cis window. Each block gets a
  share of the budget proportional to its informative peaks.
- The ATAC background (flat and near-empty peaks) is never a link candidate:
  pairs to it are dropped before the gates.

## Cis candidates

Peaks whose midpoint lies within `--cis-window` of a gene's TSS, at most
`--max-cis` nearest. ABC contact `(d + c)^-γ`, normalised over the kept
candidates. Pairs of module-only genes are dropped (no row of their own to
align).

## Tried and dropped

- Ranking the candidate cap by initial `⟨μ, ρ⟩`: it scores random tables.
- An agreement term `θ₃ ⟨μ_{m(p)}, ρ_g⟩` inside the gate: it repeated the
  module score and fed on itself (gating on agreement pulls `ρ_g` toward the
  gated modules). It took the largest gate weight while the learned ATAC
  weight stayed near zero. Centring it alone concentrated the winning modules.
- A learned mixture weight, `η = γ₂ ρ + γ₁ ã`: `γ₁` settled near 0.03 and the
  gene rows barely took up the ATAC side. The share is a fixed knob instead.
- Either coupling alone. The mixture alone lined up linked genes and modules
  (cosine −0.09 → +0.19 on the top module) but left the ATAC modules apart in
  the feature embedding (8% of their nearest features genes). The alignment
  alone blended them (53%) but lowered the cell embedding's effective rank
  (≈ 5 → 3) and merged neighbouring cell groups. Together: 47% genes, the
  modules at their cell types, rank ≈ 4, the cell types kept apart.
- Re-clustering everything after setting scattered features aside: it tears
  real groups into pieces under the minimum.
- A flat test at every pseudobulk level: the coarse level calls almost
  everything non-flat.
- The earlier softmax link attention, peak fold-in and ATAC-gene track.

## Open

- Most candidate pairs stay open (the ReLU rarely closes): the gate is a
  soft prior, and selection comes from the link score.
- Choosing the alignment weight from the data (a per-gene mismatch variance
  estimated by empirical Bayes) instead of fixing it.
- The feature rows carry components in directions the units never use (the
  units' covariance has an effective rank of a few); nothing constrains
  them, and they dominate a raw-row UMAP.
- Host memory during setup (the collapse's per-level statistics over every
  feature) dominates the peak.
