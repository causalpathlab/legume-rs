# PINTO

**P**roximity-based **I**nteraction **N**etwork analysis to dissect **T**issue
**O**rganizations.

PINTO segments spatial transcriptomics tissue into coherent regions by
clustering cell-cell edges of a spatial KNN graph. Each edge carries an
expression profile, and a collapsed Gibbs sampler assigns edges to communities;
per-cell soft membership falls out of the edge labels.

## Installation

```sh
cargo build --release -p pinto
```

The binary lands at `target/release/pinto`.

## Subcommands

### `lc` (link-community) — recommended

Link community detection. Assigns each spatial edge to one of K communities
via collapsed Gibbs sampling on compressed all-gene edge profiles, then
derives per-cell soft membership.

```sh
# Typical 10x Visium run:
pinto lc data.h5 -c tissue_positions.csv -o out

# More communities:
pinto lc data.h5 -c coords.csv -o out --n-communities 25

# External gene-pair network (e.g. BioGRID):
pinto lc data.h5 -c coords.csv -o out \
  --gene-network biogrid_pairs.tsv --n-outer-iter 3

# Expression-only (no coordinates):
pinto lc data.h5 -o out
```

**Outputs:** `{out}.propensity.parquet` (cell × K), `{out}.gene_topic.parquet`
(gene × K), `{out}.link_community.parquet` (edge × 3),
`{out}.coord_pairs.parquet`, `{out}.scores.parquet`, and —
in gene-pair mode — `{out}.gene_graph.parquet`.

### `dsvd` (delta-svd) — cell-pair shared/difference SVD

Decomposes cell-pair expression into shared and difference channels, fits
Poisson-Gamma per pseudobulk sample, and runs randomized SVD for latent
interaction topics.

```sh
pinto dsvd data.h5 -c coords.csv -o out
```

**Outputs:** `{out}.coord_pairs.parquet`, `{out}.dictionary.parquet`,
`{out}.latent.parquet`, `{out}.propensity.parquet`,
`{out}.gene_topic.parquet`, `{out}.delta.parquet` (multi-batch only).

### `prop` (propensity) — re-cluster dsvd edges at a different K

Given dsvd's edge latent codes, runs K-means and derives per-cell propensity.

```sh
pinto prop -z out.latent.parquet -e out.coord_pairs.parquet -o prop
```

**Outputs:** `{out}.propensity.parquet`, `{out}.edge_cluster.parquet`,
and — when expression data is provided — `{out}.genes.parquet`.

## Input data format

- **Expression:** `.h5` or `.zarr` sparse matrices via `data-beans`.
  Convert from `.mtx`: `data-beans from-mtx -r features.tsv.gz -c barcodes.tsv.gz matrix.mtx.gz --backend hdf5 -o data.h5`.
  Multiple files comma-separated for multi-sample runs: `s1.h5,s2.h5`.
- **Coordinates:** CSV/TSV/parquet. First column is the cell barcode; remaining
  columns are spatial coordinates. Defaults recognize Visium
  (`pxl_row_in_fullres`, `pxl_col_in_fullres`) and Xenium
  (`cell_centroid_x`, `cell_centroid_y`) layouts.
- **Batch labels (optional):** plain text, one label per cell per line
  (`-b labels.txt`); pass one file per data file.

## Detailed help

Each subcommand has a full `--help` with algorithm stages, flags, and outputs:

```sh
pinto lc --help
pinto dsvd --help
pinto prop --help
```
