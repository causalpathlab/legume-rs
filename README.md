# `legume-rs`: Library for Exploring Genomics Using Machine learning Essentials

- This is a command line based tool runs on Unix-like environments.

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
cargo install --path data-beans # Data Backend for Extraction And Neighbourhood Search
cargo install --path senna      # Stochastic data Embedding with Nearest Neighbourhood Adjustment
cargo install --path pinto      # Proximity-based Interaction Network for Tissue Organization
cargo install --path cocoa      # COunterfactual COnfounder Adjustment
cargo install --path faba       # Feature extraction from Alignment for Base-pair Annotation
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
