# legume-rs

Library for Exploring Genomics Using Machine learning Essentials — umbrella
repo for the legume CLI family (Unix-like environments).

Product crates now live in their own repositories and on crates.io. This repo
keeps shared docs and a convenience `Makefile` that installs them.

## Tools

| Binary | Package | Repo |
| --- | --- | --- |
| `senna` | `senna-rs` | [senna-rs](https://github.com/causalpathlab/senna-rs) |
| `lupin` | `lupin-rs` | [lupin-rs](https://github.com/causalpathlab/lupin-rs) |
| `chickpea` | `chickpea-rs` | [chickpea-rs](https://github.com/causalpathlab/chickpea-rs) |
| `pinto` | `pinto-rs` | [pinto-rs](https://github.com/causalpathlab/pinto-rs) |
| `cocoa` | `cocoa-rs` | [cocoa-rs](https://github.com/causalpathlab/cocoa-rs) |
| `mung` | `mung-cnv` | [mung-cnv](https://github.com/causalpathlab/mung-cnv) |
| `faba` | `faba` | [faba](https://github.com/causalpathlab/faba) |
| `fqtl` | `fqtl` | [fqtl-rs](https://github.com/causalpathlab/fqtl-rs) |
| `data-beans` | `data-beans` | [data-beans](https://crates.io/crates/data-beans) |

Shared libraries: [`legume-numeric`](https://crates.io/crates/legume-numeric),
[`legume-plot`](https://crates.io/crates/legume-plot),
[`legume-genomic-types`](https://crates.io/crates/legume-genomic-types),
[`legume-enrichment`](https://crates.io/crates/legume-enrichment),
[`legume-graph-embedding`](https://crates.io/crates/legume-graph-embedding),
[`legume-annotate`](https://crates.io/crates/legume-annotate),
[`legume-lineage`](https://crates.io/crates/legume-lineage),
[`legume-gene-text`](https://crates.io/crates/legume-gene-text),
[`hsblock-rs`](https://crates.io/crates/hsblock-rs).

**Ownership:** `senna` = train / embed / predict / cluster / layout.
`lupin` = subsequent analysis (annotate, lineage, pseudotime, describe, plots,
`text-qc` / `word-graph`). There is no separate `gene-text` binary.

## Installation

1. Install Rust via [`rustup`](https://rustup.rs/).
2. Install everything `make` knows about:

```sh
make install
```

Or individually:

```sh
cargo install senna-rs
cargo install lupin-rs
cargo install chickpea-rs
cargo install pinto-rs
cargo install cocoa-rs
cargo install mung-cnv
cargo install data-beans
```

`make install` picks a GPU backend (Metal on macOS, CUDA on Linux if `nvcc`
is on `PATH`, else CPU) and turns on HDF5 when libhdf5 is found. See
`make help`.

### Backend selection

```sh
make install-cpu
make install-cuda
make install-metal
make install BACKEND={cpu|cuda|metal}
```

### HDF5 support

```sh
make install HDF5=on
make install HDF5=off
HDF5_DIR=/path/to/hdf5 make install
```
