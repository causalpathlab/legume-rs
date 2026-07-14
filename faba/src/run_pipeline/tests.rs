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

#[test]
fn mass_enrichment_is_off_by_default() {
    let args = parse_defaults();
    assert_eq!(args.enrich.cluster_resolution, 0.0);
    assert!(
        !args.enrich.enabled(),
        "grouping must be OFF unless asked for — it adds an embedding, a kNN graph and \
         Leiden, then multiplies discovery by the group count"
    );

    // …and a positive resolution still turns it on.
    let on = Wrap::parse_from([
        "faba",
        "a.bam",
        "-g",
        "g.gff",
        "-f",
        "g.fa",
        "-o",
        "out",
        "--cluster-resolution",
        "0.5",
    ])
    .p;
    assert!(on.enrich.enabled());
}

#[test]
fn summary_records_the_effective_options_not_just_the_inputs() {
    // The summary exists to answer "what settings produced this output?" — including the
    // defaults the run never mentioned. faba's defaults have changed between builds, so a
    // record of only the command line (or, as before, of four input paths) cannot answer it.
    let args = parse_defaults();
    let json = serde_json::to_value(&args).expect("PipelineArgs serializes");

    // A default the user never typed is still recorded.
    assert_eq!(json["enrich"]["cluster_resolution"], 0.0);
    assert_eq!(json["max_threads"], 16);
    // The foreign enum goes in by its Debug form rather than being dropped.
    assert!(json["backend"].is_string(), "backend: {:?}", json["backend"]);
    // Inputs are still there.
    assert_eq!(json["gff_file"], "g.gff");
    assert_eq!(json["output"], "out");
}
