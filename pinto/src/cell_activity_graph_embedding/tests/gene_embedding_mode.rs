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
        ["x", "-o", "out", "data.zarr"]
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

mod embedding_dim {
    use super::parse;
    use crate::cell_activity_graph_embedding::args::DEFAULT_EMBEDDING_DIM;

    fn dim(argv: &[&str], width: Option<usize>) -> anyhow::Result<usize> {
        parse(argv).resolve_embedding_dim(width)
    }

    #[test]
    fn without_a_dictionary_the_flag_or_the_default_decides() {
        assert_eq!(dim(&[], None).unwrap(), DEFAULT_EMBEDDING_DIM);
        assert_eq!(dim(&["--embedding-dim", "32"], None).unwrap(), 32);
        assert_eq!(
            dim(&["--embedding-dim", "auto"], None).unwrap(),
            DEFAULT_EMBEDDING_DIM
        );
    }

    #[test]
    fn a_pinned_dictionary_sets_the_width_and_a_conflicting_flag_is_refused() {
        for mode in ["freeze", "free", "lora"] {
            let a = [
                "--gene-embedding",
                "d.parquet",
                "--gene-embedding-mode",
                mode,
            ];
            assert_eq!(dim(&a, Some(128)).unwrap(), 128, "{mode}");
            let b = [&a[..], &["--embedding-dim", "128"]].concat();
            assert_eq!(dim(&b, Some(128)).unwrap(), 128, "{mode}");
            let c = [&a[..], &["--embedding-dim", "16"]].concat();
            let err = dim(&c, Some(128)).unwrap_err().to_string();
            assert!(err.contains("128") && err.contains("16"), "{mode}: {err}");
        }
    }

    #[test]
    fn the_adapter_keeps_its_own_width() {
        let a = ["--gene-embedding", "d.parquet"];
        assert_eq!(dim(&a, Some(128)).unwrap(), DEFAULT_EMBEDDING_DIM);
        assert_eq!(
            dim(&[&a[..], &["--embedding-dim", "32"]].concat(), Some(128)).unwrap(),
            32
        );
    }

    #[test]
    fn the_lora_rank_is_validated_against_the_resolved_width() {
        let a = [
            "--gene-embedding",
            "d.parquet",
            "--gene-embedding-mode",
            "lora",
        ];
        assert!(dim(&a, Some(128)).is_ok());
        // The default rank equals a narrow dictionary's width: refused, not silently init.
        assert!(dim(&a, Some(16)).is_err());
        assert!(dim(&[&a[..], &["--lora-rank", "4"]].concat(), Some(16)).is_ok());
        assert!(dim(&[&a[..], &["--lora-rank", "0"]].concat(), Some(128)).is_err());
    }
}
