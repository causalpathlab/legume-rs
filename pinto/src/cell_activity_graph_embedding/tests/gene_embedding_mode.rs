//! The rules between `--gene-embedding-mode` and the flags that only one mode
//! reads, checked before any data is opened.

use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    cage: CellActivityGraphEmbeddingArgs,
}

fn parse(argv: &[&str]) -> CellActivityGraphEmbeddingArgs {
    Cli::try_parse_from(
        ["x", "-o", "out", "--embedding-dim", "64", "data.zarr"]
            .into_iter()
            .chain(argv.iter().copied()),
    )
    .unwrap_or_else(|e| panic!("{e}"))
    .cage
}

#[test]
fn the_lora_knobs_need_the_lora_mode() {
    let ok = parse(&[
        "--gene-embedding",
        "d.parquet",
        "--gene-embedding-mode",
        "lora",
        "--lora-rank",
        "4",
    ]);
    assert!(ok.validate_gene_embedding().is_ok());
    let bare = parse(&[
        "--gene-embedding",
        "d.parquet",
        "--gene-embedding-mode",
        "lora",
    ]);
    assert!(bare.validate_gene_embedding().is_ok());
    let wrong = parse(&[
        "--gene-embedding",
        "d.parquet",
        "--gene-embedding-mode",
        "freeze",
        "--lora-rank",
        "4",
    ]);
    let err = wrong.validate_gene_embedding().unwrap_err().to_string();
    assert!(err.contains("--lora-rank"), "{err}");
}

#[test]
fn the_adapter_residual_needs_the_adapt_mode() {
    let ok = parse(&["--gene-embedding", "d.parquet", "--gene-adapter-residual"]);
    assert!(ok.validate_gene_embedding().is_ok());
    let wrong = parse(&[
        "--gene-embedding",
        "d.parquet",
        "--gene-embedding-mode",
        "lora",
        "--gene-adapter-residual",
    ]);
    assert!(wrong.validate_gene_embedding().is_err());
}

#[test]
fn the_lora_spec_is_validated_against_the_width() {
    let a = parse(&[
        "--gene-embedding",
        "d.parquet",
        "--gene-embedding-mode",
        "lora",
        "--lora-rank",
        "0",
    ]);
    assert!(a.validate_gene_embedding().is_err());
}
