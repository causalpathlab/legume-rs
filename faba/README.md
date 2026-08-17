# faba

**F**eature statistics **A**ccumulator for **B**ase-pair-level **A**nalysis

`faba` extracts per-cell genomic features directly from alignment (BAM) files:
RNA modifications (DART-seq m6A, A-to-I editing), alternative polyadenylation
(APA), gene counts, read depth, and SNP genotypes.

Its product is the matrices; nothing here fits a model. The embedding,
trajectory and annotation commands that read them live in
[`senna`](../senna/README.md).

`faba` is one crate of the [legume-rs](https://github.com/causalpathlab/legume-rs)
workspace.

## Installation

### Prerequisites

These must be installed **before** running `cargo install` — Cargo only builds
Rust crates and cannot pull in a C toolchain or system libraries on its own. If
they are missing, the build fails midway in the vendored-htslib build script.

- **Rust** (stable, edition 2021+) — install via [rustup](https://rustup.rs).
- A **C/C++ toolchain** and the system libraries that
  [`rust-htslib`](https://github.com/rust-bio/rust-htslib) builds against
  (it vendors and compiles htslib, which needs the compression dev headers,
  and runs `bindgen`, which needs libclang):

  ```sh
  # Debian / Ubuntu
  sudo apt-get install build-essential clang libclang-dev \
      zlib1g-dev libbz2-dev liblzma-dev pkg-config

  # macOS (Homebrew)
  brew install llvm xz bzip2 zlib
  ```

### Install from GitHub

Install the `faba` binary straight from the repository with `cargo install`
(builds in release mode and drops `faba` into `~/.cargo/bin`):

```sh
cargo install --git https://github.com/causalpathlab/legume-rs.git faba
```

Pin to a specific tag or branch if you need a reproducible build:

```sh
cargo install --git https://github.com/causalpathlab/legume-rs.git --tag v0.2.3 faba
cargo install --git https://github.com/causalpathlab/legume-rs.git --branch main faba
```

### Build from a local clone

Clone the workspace and build (or install) just the `faba` crate:

```sh
git clone https://github.com/causalpathlab/legume-rs.git
cd legume-rs

# build the binary at target/release/faba
cargo build --release -p faba

# or install it onto your PATH
cargo install --path faba
```

### Optional features

| Feature | Enables | Extra requirement |
|---------|---------|-------------------|
| `hdf5`  | HDF5 backend + `.h5`/`.h5ad` readers | `libhdf5` on the build host |

There is no `cuda` / `metal` feature: nothing on the BAM-to-matrix path touches a
GPU. `senna` carries those for the code that does.

Add them with `--features`:

```sh
cargo install --git https://github.com/causalpathlab/legume-rs.git faba --features hdf5
```

### Verify

```sh
faba --help
```

### Troubleshooting

Almost all install failures come from the `hts-sys` build step (it compiles a
vendored copy of htslib and generates bindings with `bindgen`). The fixes:

- **`thread 'main' panicked ... Unable to find libclang`** — `bindgen` can't
  locate libclang. Install `libclang-dev` (Debian/Ubuntu) or `brew install
  llvm` (macOS), then point to it:

  ```sh
  # Linux: the unversioned symlink ships with libclang-dev
  export LIBCLANG_PATH=$(llvm-config --libdir 2>/dev/null || echo /usr/lib/llvm-18/lib)

  # macOS (Homebrew llvm is keg-only)
  export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
  ```

- **`fatal error: zlib.h / bzlib.h / lzma.h: No such file or directory`** — the
  compression **dev headers** are missing (the runtime `.so` alone is not
  enough). Install `zlib1g-dev libbz2-dev liblzma-dev` (Debian/Ubuntu) or
  `brew install xz bzip2 zlib` (macOS).

- **`error: linker 'cc' not found`** — no C toolchain. Install
  `build-essential` (Debian/Ubuntu) or Xcode Command Line Tools
  (`xcode-select --install`) on macOS.

- **`failed to run custom build command for 'hdf5-sys'`** — you passed
  `--features hdf5` without `libhdf5` on the host. Install `libhdf5-dev`
  (Debian/Ubuntu) or `brew install hdf5`, or drop the feature.

If a build fails after fixing a prerequisite, re-run with a clean rebuild of the
C bits: `cargo install --git ... faba --force`.

## Usage

```sh
faba <COMMAND> [OPTIONS]
```

| Command | Purpose |
|---------|---------|
| **Feature profiling — BAM → per-cell features** ||
| `dartseq` (`dart`, `m6a`) | Call DART-seq m6A sites by a WT-vs-MUT control contrast on C-to-T conversions |
| `atoi` (`a2i`, `editing`) | Detect and quantify A-to-I RNA editing sites |
| `apa` (`polya`)           | Quantify alternative polyadenylation sites per cell |
| `genes` (`count-genes`)   | Count reads per gene (single-cell or bulk RNA-seq) |
| `depth` (`rd`)            | Compute read depth over genomic intervals |
| `snp` (`genotype`)        | Discover and genotype SNP variants from BAM pileup |
| `all` (`pipeline`)        | Run the full profiling pipeline: SNP → genes → ATOI → APA → m6A |
| **Inspection & reference** ||
| `pwm`                     | Build a position weight matrix around genomic sites |
| `pileup` (`inspect`)      | ASCII pileup, or a faceted Miami plot, for one gene |
| `metagene` (`mg`)         | Metagene histogram of site positions across gene features |
| `docs`                    | Print the method write-ups compiled into this binary |

Run `faba <COMMAND> --help` for the detailed options of each subcommand.

Embedding (`gem`, `gem-encoder`), annotation (`annotate-gem`), trajectory
(`lineage`, `lineage-plot`) and modality dynamics (`dyn-assoc`) are `senna`
subcommands — they read the matrices above by prefix.

## Methods

The method write-ups are **compiled into the binary** — `include_str!`, not files read at
runtime — so they travel with a `cargo install`ed `faba` or one copied to a cluster with no
checkout beside it, and the build fails if one of them goes missing.

```sh
faba docs                # list what there is
faba docs profiling      # BAM -> per-cell features: m6A, A-to-I, APA, counts, SNPs
```

The annotation and lineage write-ups moved with their subcommands: `senna docs`.

The same files live in [`docs/`](docs/), which also carries the design notes for work that is
planned but **not implemented** (kept separate on purpose — reading a plan as though it described
the code is how people end up debugging things that were never built).

### Examples

```sh
# Gene counts from a single-cell BAM
faba genes sample.bam -g genes.gff -o out/

# A-to-I editing sites (the mask is reusable by other subcommands)
faba atoi sample.bam -g genes.gff -f genome.fa -o out/

# DART-seq m6A: signal (WT APOBEC1-YTH) vs catalytically-dead control (YTHmut).
# A control is REQUIRED — m6A can't be told apart from genomic C/T variation
# without it (--mut / --control / --background are accepted aliases).
faba dartseq wt.bam --control-bam ctrl.bam -g genes.gff -f genome.fa -o out/ \
    --atoi-mask out/atoi_sites.parquet

# Everything in one pass (the m6A step runs only when --control-bam is given,
# otherwise it is skipped; the other steps need no control)
faba all sample.bam -g genes.gff -f genome.fa -o out/ --control-bam ctrl.bam
```

## License

MIT — see the [workspace license](https://github.com/causalpathlab/legume-rs).
