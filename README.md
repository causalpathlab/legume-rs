# legume-rs

Library for Exploring Genomics Using Machine learning Essentials — CLI tools
for Unix-like environments.

## In this repo

| Binary | Role |
| --- | --- |
| `senna` | Stochastic embedding with nearest-neighbourhood adjustment |
| `lupin` | Text / annotate / lineage / describe |
| `chickpea` | Peak-to-gene and related multiome workflows |
| `gene-text` | Gene-text utilities |

Supporting workspace crates: `annotate`, `lineage`, `plot-utils`.

## Extracted tools

Install from crates.io (or see each repo):

| Binary | Package | Repo |
| --- | --- | --- |
| `fqtl` | `fqtl` | [fqtl-rs](https://github.com/causalpathlab/fqtl-rs) |
| `faba` | `faba` | [faba](https://github.com/causalpathlab/faba) |
| `mung` | `mung-cnv` | [mung-cnv](https://github.com/causalpathlab/mung-cnv) |
| `cocoa` | `cocoa-rs` | [cocoa-rs](https://github.com/causalpathlab/cocoa-rs) |
| `pinto` | `pinto-rs` | [pinto-rs](https://github.com/causalpathlab/pinto-rs) |
| `data-beans` | `data-beans` | [data-beans](https://crates.io/crates/data-beans) |

Shared libraries (also on crates.io):
[`legume-numeric`](https://crates.io/crates/legume-numeric),
[`data-beans`](https://crates.io/crates/data-beans),
[`hsblock-rs`](https://crates.io/crates/hsblock-rs),
[`legume-genomic-types`](https://crates.io/crates/legume-genomic-types),
[`mung-cnv`](https://crates.io/crates/mung-cnv) (lib `cnv`),
[`legume-enrichment`](https://crates.io/crates/legume-enrichment) (lib `enrichment`),
[`legume-graph-embedding`](https://crates.io/crates/legume-graph-embedding) (lib `graph_embedding_util`).

## Installation

1. Install Rust via [`rustup`](https://rustup.rs/).
2. Clone this repo.
3. Install everything `make` knows about (path + crates.io):

```sh
make install
```

`make install` picks a GPU backend (Metal on macOS, CUDA on Linux if `nvcc`
is on `PATH`, else CPU) and turns on HDF5 when libhdf5 is found. See
`make help`, [Backend](#backend-selection), and [HDF5](#hdf5-support).

Individual installs:

```sh
# crates.io
cargo install data-beans
cargo install data-beans --features sim   # also installs data-beans-sim
cargo install faba
cargo install mung-cnv                    # binary: mung
cargo install cocoa-rs                    # binary: cocoa
cargo install pinto-rs                    # binary: pinto

# this workspace
cargo install --path senna
cargo install --path lupin
cargo install --path chickpea
cargo install --path gene-text
```

### Backend selection

```sh
make install-cpu
make install-cuda
make install-metal
make install BACKEND={cpu|cuda|metal}
```

If a GPU build fails, that binary falls back to CPU; `make install` prints a
per-binary summary.

### HDF5 support

`.h5` / `.h5ad` need libhdf5. HDF5 is opt-in; `make install` enables it when
it finds a usable install:

| Probe | Typical source |
| --- | --- |
| `$CONDA_PREFIX` | conda env with `hdf5` |
| `h5cc` → prefix | HPC modules, manual installs |
| `/opt/homebrew`, `/usr/local`, `/usr` | Homebrew / system |
| `pkg-config hdf5` / `hdf5-serial` | Debian/Ubuntu `libhdf5-dev` |

```sh
make install HDF5=on
make install HDF5=off
HDF5_DIR=/path/to/hdf5 make install
```

With HDF5 off, `from-h5` / `from-h5ad` / `from-10x-molecule` and
`--backend hdf5` are unavailable; use `.mtx`, `.zarr`, `.zarr.zip`, or
Xenium-style zarr.

### Output format

Sparse matrices from data-beans and faba default to **`.zarr.zip`**. Pass
`--no-zip` for an unzipped `.zarr` directory.
