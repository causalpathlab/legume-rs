# Chickpea

**CH**romatin **I**nteraction **C**aptured by **K**nockoff **P**eak-to-**E**xpression
**A**ssociations.

Peak-to-gene cis-regulatory linkage from paired single-cell RNA + ATAC.

Chickpea links ATAC peaks to genes by **summary-statistics fine-mapping**: it
pseudobulks the matched RNA + ATAC cells, embeds peaks and genes in a shared
ATAC-derived latent space, scores each cis peak–gene pair by a regression
z in that space, and fine-maps per gene with **SuSiE-RSS** over the peak–peak
LD. No neural model. See [`docs/peak_to_gene_math.md`](docs/peak_to_gene_math.md)
for the full derivation.

## Pipeline

`chickpea peak-to-gene` (aliases `p2g`, `peak2gene`):

1. Load paired RNA + ATAC (matched barcodes), validate shared cells.
2. Random projection + multi-level pseudobulk collapse (batch-aware).
3. Global ATAC embedding — rSVD of the standardized log1p ATAC pseudobulk.
4. Per gene: read off the marginal peak→gene `z` and the peak–peak LD `R`,
   either as embedding inner products (default) or — with `--tmle` — as
   leave-one-chromosome-out **deconfounded** residual statistics (see below).
5. SuSiE-RSS fine-mapping per gene → posterior inclusion probabilities (PIPs)
   and effect sizes for each cis peak.
6. *(optional, `--fdr q`)* pooled GhostKnockoff filter on the per-gene `(z, R)`
   to select links at a target genome-wide FDR (the "Knockoff" in the name).
7. Write all tested (gene, cis-peak) links to a sorted BGZF BED.

### Deconfounding cis from topic (`--tmle`)

The shared ATAC embedding reads peak→gene off a low-rank *topic* space, so a
cis enhancer and its co-active bystanders — which share the same cell-type
program — look alike: the topic is a confounder the embedding cannot remove.
`--tmle` swaps the embedding association for a partially-linear, doubly-robust
estimator (Robinson/Chernozhukov DML; named **TMLE** after van der Laan &
Rubin's targeted-learning framing): it regresses both the gene and each peak on
a topic confounder and reads the partial association off the residuals
(residual-on-residual). A peak's *private*, topic-orthogonal fluctuation is the
only part that can reach its gene, so confounded bystanders collapse while
identifiable cis links survive.

The confounder is estimated **leave-one-chromosome-out** (LOCO): the topic is
genome-wide but a gene's cis signal is local, so an embedding of peaks *off* the
gene's chromosome (`--tmle-rank m` factors) captures the trans confounder
without absorbing the cis effect under test — the Neyman-orthogonality split
that keeps the residual `z` honest. On simulated multiome the embedding path's
knockoff FDP runs uncontrolled (≈0.4–0.6, topic confounding), while `--tmle`
brings it near the target with higher power.

## Usage

### Inference

```bash
chickpea peak-to-gene \
  --rna-files sim.rna.zarr \
  --atac-files sim.atac.zarr \
  --gene-coords sim.gene_coords.tsv.gz \
  -o out
```

Gene TSS positions come from either `--gene-coords` (a `gene<TAB>chr<TAB>tss`
TSV) or `--gff-file` (a GFF/GTF annotation); one of the two is required when
`--cis-window > 0`.

Key options (see `chickpea peak-to-gene --help` for all):

| Flag | Default | Meaning |
|------|---------|---------|
| `--rna-files`, `--atac-files` | — | paired matrices (zarr/h5), comma-separated |
| `--batch-files` | — | batch labels, one per file (RNA-then-ATAC order) |
| `--cis-window` | 500000 | bp around each TSS to enumerate cis peaks |
| `--max-cis` | 200 | cap on cis-candidate peaks per gene (nearest) |
| `--proj-dim` / `--sort-dim` | 64 / 14 | pseudobulk projection / binary-sort dims |
| `--embedding-dim` | 50 | ATAC embedding rank `d` |
| `--num-components` | 10 | SuSiE single-effect components `L` |
| `--prior-var` | 5.0 | SuSiE prior effect variance (z-score scale) |
| `--no-pve-adjust` | off | disable winner's-curse z shrinkage |
| `--tmle` | off | LOCO deconfounded residual `z`/`R` instead of the embedding association |
| `--tmle-rank` | 20 | off-chromosome topic confounder factors `m` for `--tmle` |
| `--num-levels` | 1 | hierarchical refinement levels; the refined finest level is used |
| `--fdr` | 0.0 | target FDR for knockoff (z-score contrast) selected links (0 = off) |
| `--ko-ridge` | 0.05 | knockoff LD ridge λ in `R_λ = (1-λ)R + λI` |
| `-o`, `--out` | — | output prefix |

Output `{out}.results.bed.gz`, sorted by `(chr, start, end)`:

```
#chr  start  end  peak_id  gene_id  pip  effect_mean  effect_std  z  distance
```

`distance = |peak_midpoint − TSS|`. All tested pairs are written;
`--pip-threshold` only drives a summary log line. With `--fdr q`, two columns
are appended — `w_stat` (knockoff importance) and `selected` (0/1).

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
