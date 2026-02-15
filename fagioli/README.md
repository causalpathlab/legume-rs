# Fagioli

Faceted Association for Genetic variant Identification Of Linear models

## Features

- **eQTL Simulation**: Simulate realistic single-cell eQTL data with cell type heterogeneity
  - Gene-by-gene cis-eQTL effects (TSS ± 1Mb windows)
  - Hybrid genetic architecture (shared + independent causal variants across cell types)
  - Factor model for gene-gene correlations (W × Z factorization)
  - Single-cell count generation with Poisson sampling
- Generalized linear models for genetic associations
- Pseudobulk-based black box variational inference model
- Similar to SuSiE (Sum of Single Effects) but with gradient-based optimization

## Installation

```bash
cargo build --release
```

## Usage

### eQTL Simulation (Single-Cell Mode)

Generate single-cell eQTL data with realistic genetic architecture:

```bash
fagioli sim-eqtl \
  --bed-prefix /path/to/genotypes \
  --chromosome 22 \
  --output ./results/sim \
  --mode single-cell \
  --num-genes 500 \
  --num-cell-types 5 \
  --num-factors 10 \
  --eqtl-gene-proportion 0.4 \
  --shared-eqtl-proportion 0.6 \
  --independent-eqtl-proportion 0.4 \
  --mean-cells-per-individual 1000 \
  --depth-per-cell 5000 \
  --seed 42
```

Or use a GFF/GTF file for gene annotations:

```bash
fagioli sim-eqtl \
  --bed-prefix /path/to/genotypes \
  --gff-file /path/to/genes.gtf \
  --chromosome 22 \
  --left-bound 20000000 \
  --right-bound 30000000 \
  --output ./results/sim \
  --mode single-cell
```

**Output files:**
- `sim.counts.zarr/` or `sim.counts.h5` - Sparse count matrix backend (genes × cells)
  - Row names: Gene IDs with symbols (e.g., `ENSG00000000001_GENE1`)
  - Column names: Cell IDs with individual (e.g., `cell_0@HG00096`)
  - Can be directly loaded with data-beans tools
- `sim.cells.tsv.gz` - Cell annotations (cell_id, individual_id, cell_type)
- `sim.cell_to_individual.tsv.gz` - Cell-to-individual mapping (cell_id, individual_id, individual_index)
- `sim.genes.tsv.gz` - Gene annotations (gene_id, chromosome, tss, strand)
- `sim.eqtl_effects.tsv.gz` - True causal eQTL effects per gene
- `sim.gene_loadings.parquet` - Factor model gene loadings (W)
- `sim.factor_celltype.parquet` - Factor-celltype scores (Z)
- `sim.cell_fractions.parquet` - Individual cell type fractions
- `sim.parameters.json` - All simulation parameters

**Backend options:**
- `--backend zarr` (default): Zarr format for cloud-friendly storage
- `--backend hdf5`: HDF5 format for traditional file-based storage

### General Usage

```bash
fagioli --help
```
