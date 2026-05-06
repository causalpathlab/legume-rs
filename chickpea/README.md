# Chickpea

Topic-Model-Based Peak-Gene Linking

## Features

- **Inference** (`fit-topic`): Jointly recover topic proportions and peak-gene
  linking probabilities from paired RNA + ATAC data via topic model with
  SuSiE fine-mapping
- **Simulation**: now lives in `data-beans-sim multiome` (paired ATAC + RNA
  with ground-truth peak-gene links, optional per-modality NB+copula sampling
  conditioned on a real reference)

## Usage

### Simulation

Use `data-beans-sim multiome`:

```bash
data-beans-sim multiome \
  --out ./results/sim \
  --n-genes 2000 \
  --n-peaks 10000 \
  --n-cells 5000 \
  --n-topics 10 \
  --linked-gene-fraction 0.3 \
  --n-causal-per-gene 3 \
  --depth-rna 5000 \
  --depth-atac 2000 \
  --pve-topic 0.8 \
  --gene-topic-sd 0.3 \
  --rseed 42
```

Optional `--reference-rna <h5/zarr>` and/or `--reference-atac <h5/zarr>`
switch the corresponding modality to a two-stage GLM with NB+copula PIT
sampling fitted from the reference.

### Inference

```bash
chickpea fit-topic \
  --rna-files sim.rna.zarr \
  --atac-files sim.atac.zarr \
  -o out
```

## Installation

```bash
cargo build --release -p chickpea
```
