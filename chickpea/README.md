# Chickpea

- Chromatin Interactions Captured by Knitting Peaks with Expression Anchors

- Peak-to-gene cis-regulatory linkage from paired single-cell RNA + ATAC.

## Pipeline

`chickpea peak-to-gene` (aliases `p2g`, `peak2gene`):

1. **Pseudobulk** the paired cells into `--num-levels` nested levels
   (data-beans multilevel collapse). Profiles are support-masked posterior
   means: an entry a pseudobulk never observed is 0, not the prior's floor.
2. **Link** every gene to the peaks within `--cis-window` of its TSS with
   `--link-score`:
   - `pearson` (default): Pearson correlation of the `log1p` RNA and ATAC
     pseudobulk profiles; positive correlations above `--min-weight` are kept.
   - `abc`: Engreitz ABC, `A_p·C(d) / Σ_window A·C`, with `A_p` the mean
     pseudobulk accessibility and `C(d) = max(d, 5 kb)^(−0.87) + C(1 Mb)` the
     power-law contact with its pseudocount. Reads ATAC only.
   Per gene the edges are ranked and cut to `--top-k-per-gene` (off by default)
   and `--max-cis`.
3. **Embed** peaks, genes and every level's pseudobulks in one
   graph-embedding-util FNE table: the link relation (weight = score), one
   pseudobulk × gene / pseudobulk × peak relation per SIMBA expression level
   (`--context-bins`, library-normalized `log1p` profiles binned by 1-D
   k-means) and the child → parent edges of the levels. Peaks and genes are
   thus placed by both their cis links and the cellular contexts that express
   them. The finest pseudobulk rows are Leiden-clustered.
4. **Recompute the link within each cluster** (same score on the cluster's
   pseudobulk columns): one link table per `cell_type_id`.
5. **Write** E2G-like `{out}/peaks.parquet`, `clusters.parquet`,
   `peak_gene/chr*.parquet`, plus `{out}.{peak,gene,cell}_embedding.parquet`
   (`cell` = finest pseudobulks) and, with more than one level,
   `{out}.pb_tree_embedding.parquet` (rows `L{level}:{i}`).

ATAC-only (omit `--rna-files`): an ArchR-style gene activity stands in for RNA.
Under `pearson` that makes the link a correlation of peaks with a weighted sum
of themselves, so prefer `--link-score abc` there. The surrogate enters no
count relation; the embedding reads ATAC only.

```bash
chickpea peak-to-gene \
  --rna-files sim.rna.zarr.zip \
  --atac-files sim.atac.zarr.zip \
  --gene-coords sim.gene_coords.tsv.gz \
  --link-score pearson --top-k-per-gene 5 \
  --device cuda \
  -o out
```

The FNE is the costly stage (one edge per nonzero pseudobulk entry);
`--device cuda` needs a build with `--features cuda`.

Tests live under `tests/` (`cargo test -p chickpea`).

## Installation

```bash
cargo build --release -p chickpea
```

## Simulation

Paired ATAC + RNA with ground-truth peak–gene links comes from
`data-beans-sim multiome` (`{out}.rna.zarr.zip`, `{out}.atac.zarr.zip`,
`{out}.gene_coords.tsv.gz`, `{out}.ground_truth.tsv.gz`). Its chromosomes are
short, so pass a matching `--cis-window` (tens of kb) when scoring on it.
