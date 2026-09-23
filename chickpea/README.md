# Chickpea

**CH**romatin **I**nteractions **C**aptured by **K**nitting **P**eaks with
**E**xpression **A**nchors.

Peak-to-gene cis-regulatory linkage from paired single-cell RNA + ATAC.

Peaks enter as gene features. Each gene gets an ATAC version, its cis peaks'
counts aggregated through ABC contact weights, and the gene axis carries two
tracks: RNA and peak-aggregated. Both are embedded against the same pseudobulk
embeddings, with the ATAC row tied to the RNA row by a ridge-shrunk low-rank
offset. Each gene then attends over its own cis peaks, and the attention shares
are the links. The design and its checks are in [`docs/plan.md`](docs/plan.md).

## Pipeline

`chickpea peak-to-gene` (aliases `p2g`, `peak2gene`):

1. **Cis candidates.** Peaks whose midpoint lies within `--cis-window` of a
   gene's TSS, at most `--max-cis` per gene, nearest first. Each pair carries
   the ABC contact `(d + c)^-γ`, normalised over the gene's candidates. A peak
   near several genes is a candidate for each.
2. **Peak-aggregated track.** Every cell's peak counts summed onto genes through
   those weights, streamed once over the ATAC counts.
3. **Two-track embedding.** RNA gene rows (the base track) and peak-aggregated
   rows (base plus offset), trained with the shared two-phase engine: an exact
   two-level softmax over multilevel pseudobulks, then per-cell embeddings.
4. **Peak rows.** Every peak folded in against the finest pseudobulk
   embeddings by one Poisson IRLS step, all peaks at once, streamed from cells.
5. **Localized attention.** Each gene scores its cis peaks by a learned distance
   kernel plus a low-rank content term between gene and peak rows. The pooled
   peak rows are trained to agree with the gene's RNA row. Only the kernel and
   the content map train; nothing is genes × peaks.
6. **Per-cluster links.** Cells are clustered (Leiden), and each gene's shares
   are re-weighted by each cluster's peak accessibility. Fixed ABC shares are
   reported alongside as the baseline.

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
| `--contact-gamma`, `--contact-pseudocount` | 1, 5000 | initial ABC contact `(d + c)^-γ` |
| `--embedding-dim` | 128 | embedding dimension |
| `--epochs` | 1000 | embedding epochs |
| `--offset-rank`, `--offset-l2` | 16, 1.0 | rank and ridge tying a gene's ATAC row to its RNA row |
| `--attention-rank`, `--attention-epochs` | 16, 100 | attention content map rank and epochs |
| `--n-clusters` | Leiden | target number of cell clusters |
| `-o`, `--out` | — | output prefix |

Outputs, all `{out}.*.parquet`:

| File | Contents |
|------|----------|
| `links` | one row per cis pair: `gene`, `peak`, `distance`, `abc`, `attention` (shares sum to 1 per gene) |
| `links_by_cluster` | `gene`, `peak`, `cluster`, `attention`, `abc`: shares within each cell cluster |
| `gene_embedding`, `gene_atac_embedding` | RNA and peak-aggregated gene rows |
| `peak_embedding`, `peaks` | folded-in peak rows; `peak`, `chromosome`, `start`, `end`, `bias` |
| `cell_embedding`, `cell_clusters` | per-cell rows; `cell`, `cluster` |

`{out}.atac_gene.zarr` holds the peak-aggregated gene counts.

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
