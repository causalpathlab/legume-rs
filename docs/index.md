<!---------------------------------------------------------------------------->
<!-- Rules for vignettes                                                    -->
<!--                                                                        -->
<!-- 1. use legume CLI tools                                                -->
<!-- 2. use R, don't use Python                                             -->
<!-- 3. save intermediate files with "temp" prefix (not kept in github)     -->
<!-- 4. keep intermediate results in temp/ to avoid re-running same steps   -->
<!---------------------------------------------------------------------------->

# legume-rs

CLI tools for genomics / single-cell analysis on Unix-like systems. See the
[GitHub README](https://github.com/causalpathlab/legume-rs) for the full
install and backend notes.

## Quick install

```sh
# from a clone of this repo (path + crates.io binaries):
make install

# or pick packages:
cargo install data-beans faba mung-cnv cocoa-rs pinto-rs
cargo install --path senna
cargo install --path lupin
cargo install --path chickpea
cargo install --path gene-text
```

## Tools in this repo

- **senna** — stochastic embedding with nearest-neighbourhood adjustment
- **lupin** — text, annotate, lineage, describe
- **chickpea** — peak-to-gene / multiome workflows
- **gene-text** — gene-text utilities

## Extracted tools

| Binary | Install | Repo |
| --- | --- | --- |
| `data-beans` | `cargo install data-beans` | crates.io |
| `faba` | `cargo install faba` | [faba](https://github.com/causalpathlab/faba) |
| `mung` | `cargo install mung-cnv` | [mung-cnv](https://github.com/causalpathlab/mung-cnv) |
| `cocoa` | `cargo install cocoa-rs` | [cocoa-rs](https://github.com/causalpathlab/cocoa-rs) |
| `pinto` | `cargo install pinto-rs` | [pinto-rs](https://github.com/causalpathlab/pinto-rs) |
| `fqtl` | see repo | [fqtl-rs](https://github.com/causalpathlab/fqtl-rs) |
