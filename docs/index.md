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
# from a clone of this repo (installs everything from crates.io):
make install

# or pick packages:
cargo install senna-rs lupin-rs chickpea-rs pinto-rs cocoa-rs mung-cnv faba fqtl-rs
cargo install data-beans --features sim
```

## Tools

| Binary | Install | Repo |
| --- | --- | --- |
| `senna` | `cargo install senna-rs` | [senna-rs](https://github.com/causalpathlab/senna-rs) |
| `lupin` | `cargo install lupin-rs` | [lupin-rs](https://github.com/causalpathlab/lupin-rs) |
| `chickpea` | `cargo install chickpea-rs` | [chickpea-rs](https://github.com/causalpathlab/chickpea-rs) |
| `pinto` | `cargo install pinto-rs` | [pinto-rs](https://github.com/causalpathlab/pinto-rs) |
| `cocoa` | `cargo install cocoa-rs` | [cocoa-rs](https://github.com/causalpathlab/cocoa-rs) |
| `mung` | `cargo install mung-cnv` | [mung-cnv](https://github.com/causalpathlab/mung-cnv) |
| `faba` | `cargo install faba` | [faba](https://github.com/causalpathlab/faba) |
| `fqtl` | `cargo install fqtl-rs` | [fqtl-rs](https://github.com/causalpathlab/fqtl-rs) |
| `data-beans` | `cargo install data-beans --features sim` | [data-beans](https://github.com/causalpathlab/data-beans) |
