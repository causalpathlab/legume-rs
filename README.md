# `legume-rs`: Library for Expedited Genomic data analysis with Unsupervised Machine learning Estimators

- This is a command line based tool runs on Unix-like environments.

## Installation

0. Install `Rust` environment, e.g., using [`rustup`](https://rustup.rs/)

1. Clone this repo

2. You can install different binary files in your local executable directory (e.g., `~/.cargo/bin/`).

```{sh}
cargo install --path data-beans # utility functions for sparse data management
cargo install --path senna      # single-cell embedding with nearest neighbourhood adjustment
cargo install --path pinto      # proximity-based interaction network analysis
cargo install --path cocoa-diff # counterfactual confounder adjustment
cargo install --path faba       # BAM file parsing to extract a feature matrix
```

