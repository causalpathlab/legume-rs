<!---------------------------------------------------------------------------->
<!-- Rules for vignettes												    -->
<!-- 																	    -->
<!-- 1. use legume CLI tools											    -->
<!-- 																	    -->
<!-- 2. use R, don't use Python											    -->
<!-- 																	    -->
<!-- 3. save intermediate files with "temp" prefix not to be kept in github -->
<!-- 4. keep intermediate results in temp folder to avoid re-running same steps -->
<!---------------------------------------------------------------------------->

# `legume-rs`: Library for Exploring Genomics Using Machine learning Essentials

This is a command line based tool that runs on Unix-like environments.

## Installation

0. Install `Rust` environment, e.g., using [`rustup`](https://rustup.rs/)

1. Clone this repo

2. Install all binaries to your local executable directory (e.g., `~/.cargo/bin/`):

```sh
make install
```

Alternatively, you can install individual binaries:

```sh
cargo install data-beans
cargo install faba
cargo install --path senna
cargo install --path pinto
cargo install --path cocoa
```

## Tools

### data-beans
Basic utility functions for processing sparse matrices from single-cell omics data.

### faba
**FABA**: Feature statistics Accumulator for Base-pair-level Analysis — lives in
[`causalpathlab/faba`](https://github.com/causalpathlab/faba) (`cargo install faba`).

### senna
**SENNA**: Stochastic data Embedding with Nearest Neighbourhood Adjustment - embedding and dimensionality reduction tools.

### pinto
**PINTO**: Proximity-based Interaction Network analysis to dissect Tissue Organizations - spatial analysis and network tools for tissue organization.

### cocoa
**CoCoA**: Counterfactual Confounder Adjustment for Differential Analysis - confounder adjustment methods.

## Documentation

For more details, visit the [GitHub repository](https://github.com/causalpathlab/legume-rs).
