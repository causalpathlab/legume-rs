# Chickpea

**CH**romatin **I**nteractions **C**aptured by **K**nitting **P**eaks with
**E**xpression **A**nchors.

Peak-to-gene cis-regulatory linkage from paired single-cell RNA + ATAC.

RNA genes and ATAC peaks share one multiome feature axis, embedded like
`senna bge --multiome`: a wide ATAC modality is module-only, so each peak's row
is its module's row. Inside the phase-1 fit, each gene's score mixes in the
predicted accessibility of its cis peaks, pooled through ReLU gates on
distance, and an alignment term pulls the gene's own profile toward it, so
genes and peaks share one feature space.
Each link is scored by its gate times the data's evidence. The design is in
[`docs/design.md`](docs/design.md).

## Model

```text
λ_u,m = ⟨e_u, μ_m⟩ + b_m + β_u,k(m)                       # module score, β per-unit modality intercept
ρ_ug  = ⟨e_u, μ_{m(g)} + r_g⟩ + b_g                       # the gene's own score
w_gp  = abc_gp · max(0, θ₀ + θ₁ z(log contact_gp))        # the gate: a distance prior
w̃_gp  = w_gp / Σ_q w_gq                                   # the gene's share of each pair
ã_ug  = Σ_p w̃_gp λ_u,m(p)                                 # ATAC-guided gene activity
η_ug  = (1 − α) ρ_ug + α ã_ug                             # the gene's likelihood score, α = --mix
gap   = mean_g var_u(ρ_ug − ã_ug)                         # added to phase 1 × --align-weight

corr_gp  = corr_u(⟨e_u, μ_{m(p)}⟩, ⟨e_u, ρ_g⟩)             # after training: the evidence
score_gp = w̃_gp · corr_gp                                 # the link score
```

`θ` is shared by every gene. The mixture routes part of each cis gene's
likelihood through its peaks, and the written gene rows carry it; the
alignment moves gene rows, peak-module rows and the gates, never the units.
The two need each other: the mixture alone leaves the peaks apart from the
genes, the alignment alone squeezes the cell embedding. `w` is unit-free, and cell context comes in through
`e_u`. `β` keeps each unit's ATAC:RNA count split out of the embedding. The
variance and the correlation run across pseudobulk units.

## Pipeline

`chickpea peak-to-gene` (aliases `p2g`, `peak2gene`):

1. **Cis candidates.** Peaks whose midpoint lies within `--cis-window` of a
   gene's TSS, at most `--max-cis` per gene, nearest first. Each pair carries
   the ABC contact `(d + c)^-γ`, normalised over the gene's candidates. A peak
   near several genes is a candidate for each.
2. **Multiome embedding.** RNA genes and ATAC peaks on one axis, trained with
   the shared two-phase engine: an exact two-level softmax over multilevel
   pseudobulks, then one cell encoder over the joint axis. Modules are
   modality-pure, with a budget per modality (`--feature-modules` for RNA,
   `--peak-modules` for ATAC). Flat features (one rate explains their counts)
   form each modality's background and go module-only; on RNA, singleton
   genes join them. ATAC modules are never set aside by size, so a single
   active peak keeps its own module, and they never cross a 10 Mb genomic
   window.
3. **Gates.** After the partition, pairs of module-only genes and pairs to the
   ATAC background are dropped. The gate scalars train inside phase 1,
   through the alignment term. Afterwards each pair is scored: its gate times
   the correlation across pseudobulks of the peak module's score and the
   gene's.
4. **Per-cluster links.** Cells are clustered (Leiden), and each gene's gate
   shares are re-weighted by each cluster's peak accessibility. Fixed ABC shares
   are reported alongside as the baseline.

Cell QC (on by default, driven by the RNA counts) leaves failed cells in the
embedding but out of clustering, accessibility rates and every cell table.

## Usage

```bash
chickpea peak-to-gene \
  --rna sim.rna.zarr \
  --atac sim.atac.zarr \
  --gene-coords sim.gene_coords.tsv.gz \
  -o out
```

Gene TSS positions come from `--gene-coords` (a `gene<TAB>chr<TAB>tss` TSV with
a header) or `--gff-file` (GFF/GTF); one is required.

Key options (see `chickpea peak-to-gene --help` for all):

| Flag | Default | Meaning |
|------|---------|---------|
| `--rna`, `--atac` | — | paired matrices (zarr/h5) for the same barcodes |
| `--batch` | — | batch labels, one per barcode |
| `--cis-window` | 500000 | max peak-midpoint distance (bp) to a TSS |
| `--max-cis` | 200 | cap on candidate peaks per gene (nearest) |
| `--contact-gamma`, `--contact-pseudocount` | 1, 5000 | ABC contact `(d + c)^-γ` |
| `--mix` | 0.5 | share of each gene's score taken from its cis peaks (0 = exact likelihood) |
| `--align-weight` | 0.1 | weight of the gene-to-cis-peak alignment per unit (0 = off) |
| `--weight-decay` | 1e-4 | L2 decay on the gene and peak embedding rows, never the pseudobulks (0 = off) |
| `--embedding-dim` | 128 | embedding dimension |
| `--epochs` | 1000 | embedding epochs |
| `--feature-modules` | 1024 | RNA gene modules |
| `--peak-modules` | 10000 | ATAC peak modules (each peak's row is its module's) |
| `--module-only-min-rows` | 100000 | ATAC goes module-only at this many peaks (0 = off) |
| `--device`, `--device-no` | cpu, 0 | compute device (`cpu`, `cuda`, `metal`) |
| `--n-clusters` | Leiden | target number of cell clusters |
| `-o`, `--out` | — | output prefix |

Outputs, all `{out}.*.parquet`:

| File | Contents |
|------|----------|
| `links` | one row per kept cis pair: `gene`, `peak`, `distance`, `abc`, `gate` (the trained `w`; `0` for a closed pair), `corr` (the evidence), `score` (`gate × corr`; rank links by this) |
| `links_by_cluster` | `gene_idx`, `peak_idx`, `cluster`, `gate`, `abc`: shares within each cell cluster (indices into the RNA / peak axes) |
| `gene_embedding` | RNA gene rows |
| `peak_embedding`, `peaks` | each peak's module row; `peak`, `chromosome`, `start`, `end` |
| `cell_embedding`, `cell_clusters` | per-cell rows; `cell`, `cluster` |

### Simulation

Paired ATAC + RNA with ground-truth peak-gene links lives in
`data-beans-sim multiome`:

```bash
data-beans-sim multiome \
  --out ./results/sim \
  --n-genes 2000 --n-peaks 10000 --n-cells 5000 \
  --n-topics 10 \
  --linked-gene-fraction 0.3 --n-causal-per-gene 3 \
  --depth-rna 5000 --depth-atac 2000 \
  --rseed 42
```

It writes `{out}.rna.zarr`, `{out}.atac.zarr`, and `{out}.gene_coords.tsv.gz`
(the `--gene-coords` input above). Optional `--reference-rna`/`--reference-atac`
switch a modality to NB+copula sampling fitted from a real reference.

## Installation

```bash
cargo build --release -p chickpea
```
