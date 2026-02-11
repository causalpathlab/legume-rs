# PINTO

**P**roximity-based **I**nteraction **N**etwork analysis to dissect **T**issue **O**rganizations

PINTO identifies spatial cell-cell interaction patterns from spatially-resolved
transcriptomics (SRT) data. It constructs spatial cell pairs from KNN graphs,
decomposes pair-level expression into shared/difference channels, and learns
latent interaction topics via randomized SVD.

## Installation

```sh
cargo build --release -p pinto
```

The binary will be at `target/release/pinto`.

## Subcommands

### `delta-svd` — Gene-level shared/difference analysis

Decomposes cell-pair expression into shared and difference channels, fits
Poisson-Gamma models on pseudobulk samples, and runs SVD to learn latent
interaction topics.

```sh
pinto delta-svd data.zarr --coord coords.tsv -o output -k 10 -t 10
```

**Outputs:** `{out}.delta.parquet`, `{out}.coord_pairs.parquet`,
`{out}.dictionary.parquet`, `{out}.latent.parquet`

### `gene-pair-delta-svd` — Gene-gene interaction patterns

Discovers gene-gene co-expression patterns within spatial neighbourhoods.
Builds a gene-gene KNN graph, computes positive interaction deltas, and
applies SVD + Nystrom projection.

```sh
pinto gene-pair-delta-svd data.zarr --coord coords.tsv -o output -k 10 -t 10
```

Optionally supply an external gene network (e.g., BioGRID):

```sh
pinto gene-pair-delta-svd data.zarr --coord coords.tsv \
  --gene-network biogrid_edges.tsv -o output
```

**Outputs:** `{out}.coord_pairs.parquet`, `{out}.gene_graph.parquet`,
`{out}.dictionary.parquet`, `{out}.latent.parquet`

### `propensity` — Vertex propensity from edge clusters

Estimates per-cell propensity scores from edge cluster assignments.

```sh
pinto propensity -z output.latent.parquet -e output.coord_pairs.parquet -o prop
```

**Outputs:** `{out}.propensity.parquet`, `{out}.edge_cluster.parquet`

### `visualize` — Gene network layout

Computes a 2D spectral layout of genes and clusters gene-pair edges by their
dictionary vectors.

```sh
pinto visualize -g output.gene_graph.parquet -d output.dictionary.parquet -o viz
```

**Outputs:** `{out}.gene_coords.parquet`, `{out}.gene_pair_clusters.parquet`

## Input data format

- Expression data: `.zarr` or `.h5` sparse matrices (via `data-beans`).
  Convert from `.mtx` using `data-beans from-mtx`.
- Coordinate files: TSV/CSV/parquet with cell barcodes and spatial coordinates.
- Batch files (optional): one label per cell per line, for batch effect correction.

## Detailed help

Each subcommand supports `--help` with full pipeline stage descriptions and
parameter documentation:

```sh
pinto delta-svd --help
pinto gene-pair-delta-svd --help
pinto propensity --help
pinto visualize --help
```
