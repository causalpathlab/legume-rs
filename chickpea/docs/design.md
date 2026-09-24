# chickpea peak-to-gene: design

RNA genes and ATAC peaks share one multiome feature axis, fit by the
`graph-embedding-util` two-phase engine (`senna bge --multiome` recipe). Cis
peak→gene gates sit inside phase 1, on the RNA gene-level likelihood, and their
trained weights are the links.

## Model

```text
λ_u,m = ⟨e_u, μ_m⟩ + b_m + β_u,k(m)                  module score (β: per-unit modality intercept)
w_gp  = abc_gp · max(0, θ₀ + θ₁ z(log contact_gp) + θ₃ ⟨μ_{m(p)}, ρ_g⟩)
a_ug  = Σ_p w_gp (⟨e_u, μ_{m(p)}⟩ + b_{m(p)})         pooled predicted accessibility
η_ug  = γ₂ ρ_ug + γ₁ a_ug                            the RNA gene score
```

- `θ` and `γ` are shared by every gene; `γ = softplus(raw)`, starting at
  `γ₁ = 0.01` (small, still takes a gradient) and `γ₂ = 1`. They train by Adam.
- `w` is unit-free; cell context enters through `e_u`.
- `β_u,k` is one intercept per unit and non-reference modality, so a unit's
  ATAC:RNA count split is not written into the embedding. `β` alone is not
  identified (a direction shared by one modality's module rows is another
  intercept), so `μ` is held centred within each modality.

## How it is computed

`λ` is linear in `e_u`, so the pooled term is a gene row plus a gene bias:

```text
a_ug = ⟨e_u, ν_g⟩ + c_g        ν_g = Σ_p w_gp μ_{m(p)}      c_g = Σ_p w_gp b_{m(p)}
η_ug = ⟨e_u, γ₂ r_g + γ₁ ν_g⟩ + γ₂ b_g + γ₁ c_g
```

Once per step: `⟨ρ, μ⟩` and `W` on `[cis genes × peak modules]` (a parallel
`index_add_rows` scatter), then `ν = W μ`, `c = W b`. Per gene batch, `ν` and
`c` are gathered and folded into the gene rows before the gene-level matmul, so
no batch tensor carries a module axis. A step split over CPU threads builds the
pool once; the slices share it as detached leaves and its gradient goes back
through the pool once.

Gene-level module draws go only to modules with a gene level, each weighted by
the unit's count share on them — the same objective, without spending draws on
module-only modules.

## Feature partition

Modality-pure modules, the same budget per modality.

- **Flat features** (one rate explains a feature's counts over the finest
  pseudobulks, by the coarsener's Poisson homogeneity test) form each
  modality's background module, with no residual. The test runs on the finest
  level only: on pooled coarser levels it ignores overdispersion and calls
  almost every feature non-flat.
- **Scattered features.** k-means places a feature whose profile matches no
  other alone. Groups under a minimum size join the background — singletons on
  RNA, groups under ten peaks on ATAC — and the freed module slots split the
  largest groups in two, a split kept only when both halves reach the minimum.
  Re-clustering without them was tried and only promotes the next outliers.
- **Genomic blocks (ATAC).** A module never crosses a 10 Mb window of one
  chromosome, so no module sits in every gene's cis window. Each block gets a
  share of the budget proportional to its informative peaks.
- The ATAC background (flat and scattered peaks) is never a link candidate:
  pairs to it are dropped before the gates.

## Cis candidates

Peaks whose midpoint lies within `--cis-window` of a gene's TSS, at most
`--max-cis` nearest. ABC contact `(d + c)^-γ`, normalised over the kept
candidates. Pairs of module-only genes are dropped (no gene level to feed).

## Tried and dropped

- Ranking the candidate cap by initial `⟨μ, ρ⟩`: it scores random tables.
- Centring the agreement term alone: it concentrated the winning modules.
- Re-clustering everything after setting scattered features aside: it tears
  real groups into pieces under the minimum.
- A flat test at every pseudobulk level: the coarse level calls almost
  everything non-flat.
- The earlier softmax link attention, peak fold-in and ATAC-gene track.

## Open

- Most candidate pairs stay open (the ReLU rarely closes): the gates rank
  more than they select.
- ATAC module rows and RNA gene rows blend little in the joint embedding.
- Host memory during setup (the collapse's per-level statistics over every
  feature) dominates the peak.
