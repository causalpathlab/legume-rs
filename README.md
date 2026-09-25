# `legume-rs`: Library for Exploring Genomics Using Machine learning Essentials

- This is a command line based tool runs on Unix-like environments.
- **fqtl** (functions for QTL analysis), formerly `fagioli`, now lives in
  [`causalpathlab/fqtl-rs`](https://github.com/causalpathlab/fqtl-rs) (`fqtl` binary).
- **faba** (Feature statistics Accumulator for Base-pair-level Analysis) now lives in
  [`causalpathlab/faba`](https://github.com/causalpathlab/faba) (`cargo install faba`).
- **mung** (Malignancy Unmixing on Normalized Genomes with CNV estimation) now lives in
  [`causalpathlab/mung-cnv`](https://github.com/causalpathlab/mung-cnv)
  (`cargo install mung-cnv`).
- **cocoa** (COunterfactual COnfounder Adjustment) now lives in
  [`causalpathlab/cocoa-rs`](https://github.com/causalpathlab/cocoa-rs)
  (`cargo install cocoa-rs`).
- **pinto** (Proximity-based Interaction Network for Tissue Organization) now lives in
  [`causalpathlab/pinto-rs`](https://github.com/causalpathlab/pinto-rs)
  (`cargo install pinto-rs`).
- Shared libraries are published separately:
  [`legume-numeric`](https://crates.io/crates/legume-numeric),
  [`data-beans`](https://crates.io/crates/data-beans),
  [`hsblock-rs`](https://crates.io/crates/hsblock-rs),
  [`legume-genomic-types`](https://crates.io/crates/legume-genomic-types),
  [`mung-cnv`](https://crates.io/crates/mung-cnv) (Rust lib `cnv`),
  [`legume-enrichment`](https://crates.io/crates/legume-enrichment) (Rust lib `enrichment`),
  [`legume-graph-embedding`](https://crates.io/crates/legume-graph-embedding) (Rust lib `graph_embedding_util`).

## Installation

0. Install `Rust` environment, e.g., using [`rustup`](https://rustup.rs/)

1. Clone this repo

2. Install all binaries to your local executable directory (e.g., `~/.cargo/bin/`):

```sh
make install
```

`make install` auto-detects the right GPU backend (CUDA on Linux when
`nvcc` is on PATH, Metal on macOS, otherwise CPU). It also auto-detects
libhdf5 — see [HDF5 support](#hdf5-support) below.

Alternatively, you can install individual binaries:

```sh
cargo install data-beans             # Data Backend for Extraction And Neighbourhood Search
cargo install data-beans --features sim  # data-beans-sim binary
cargo install faba                   # Feature statistics Accumulator for Base-pair-level Analysis
cargo install --path senna      # Stochastic data Embedding with Nearest Neighbourhood Adjustment
cargo install pinto-rs          # pinto — Proximity-based Interaction Network for Tissue Organization
cargo install --path lupin      # Lexicon-Using Projection for Identity Naming (text, annotate, lineage, describe)
cargo install cocoa-rs          # cocoa — COunterfactual COnfounder Adjustment
cargo install mung-cnv          # mung — Malignancy Unmixing on Normalized Genomes with CNV estimation
```

### Backend selection

```sh
make install-cpu        # CPU only (no GPU features)
make install-cuda       # NVIDIA CUDA + cuDNN
make install-metal      # Apple Metal + Accelerate
make install BACKEND={cpu|cuda|metal}
```

If a GPU build fails (e.g. CUDA toolkit present but broken), each
binary falls back to CPU automatically — `make install` reports a
per-binary summary at the end.

### HDF5 support

`.h5` / `.h5ad` inputs (and the HDF5 sparse backend) require linking
against libhdf5. Because libhdf5 isn't shipped on every host — cluster
login nodes routinely lack it — HDF5 is **opt-in**, and `make install`
turns it on automatically when it can find the library:

| Detected via | Used by |
| --- | --- |
| `$CONDA_PREFIX/{include,lib}` | Active conda env with `hdf5` |
| `dirname $(dirname $(which h5cc))` + file check | HPC modules, manual installs |
| `/opt/homebrew`, `/usr/local`, `/usr` | macOS `brew install hdf5`, system installs |
| `pkg-config --exists hdf5 || hdf5-serial` | Debian/Ubuntu `apt install libhdf5-dev` |

`make help` reports the detected state and (if found) the HDF5 prefix
it picked. Force it either way:

```sh
make install HDF5=on              # opt in (uses detected prefix)
make install HDF5=off             # skip libhdf5 even if detected
HDF5_DIR=/path/to/hdf5 make ...   # override the detected prefix
```

When HDF5 is off, the data-beans `from-h5` / `from-h5ad` / `from-10x-molecule`
subcommands and the `--backend hdf5` option compile out; inputs are
limited to `.mtx`, `.zarr`, `.zarr.zip`, and Xenium-style zarr.

### Output format

Sparse matrices produced by data-beans and faba default to
**`.zarr.zip`** — a single archive file, easier to copy/share than a
`.zarr` directory. Pass `--no-zip` to keep an unzipped `.zarr`
directory (useful when piping into other tools that expect a directory
store).
