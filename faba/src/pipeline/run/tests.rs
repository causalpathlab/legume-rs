use super::*;
use clap::Parser;

/// `PipelineArgs` is a `clap::Args` (it is flattened into the subcommand), so parsing it
/// standalone needs a wrapper `Parser`.
#[derive(Parser)]
struct Wrap {
    #[command(flatten)]
    p: PipelineArgs,
}

fn parse_defaults() -> PipelineArgs {
    Wrap::parse_from(["faba", "a.bam", "-g", "g.gff", "-f", "g.fa", "-o", "out"]).p
}

/// The Leiden mass-enrichment grouping is gone, so its flags must not parse. A
/// removed flag that still parses is worse than one that errors: the caller
/// believes it asked for grouping and gets an ungrouped run with exit code 0.
#[test]
fn the_removed_grouping_flags_do_not_parse() {
    for flag in [
        "--cluster-resolution",
        "--cluster-knn",
        "--cluster-dim",
        "--cluster-block-size",
        "--cluster-min-row-nnz",
        "--cluster-min-col-nnz",
    ] {
        let parsed = Wrap::try_parse_from([
            "faba", "a.bam", "-g", "g.gff", "-f", "g.fa", "-o", "out", flag, "0.5",
        ]);
        assert!(parsed.is_err(), "{flag} still parses");
    }
}

#[test]
fn summary_records_the_effective_options_not_just_the_inputs() {
    // The summary exists to answer "what settings produced this output?" — including the
    // defaults the run never mentioned. faba's defaults have changed between builds, so a
    // record of only the command line (or, as before, of four input paths) cannot answer it.
    let args = parse_defaults();
    let json = serde_json::to_value(&args).expect("PipelineArgs serializes");

    // A default the user never typed is still recorded.
    assert_eq!(json["max_threads"], 16);
    // The foreign enum goes in by its Debug form rather than being dropped.
    assert!(
        json["backend"].is_string(),
        "backend: {:?}",
        json["backend"]
    );
    // Inputs are still there.
    assert_eq!(json["gff_file"], "g.gff");
    assert_eq!(json["output"], "out");
}
