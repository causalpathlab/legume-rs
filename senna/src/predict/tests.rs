//! Ablation is the flag that turns the score from a reconstruction into a
//! prediction, so its gate has to be exact: a feature named for hiding must
//! leave the encoder's view, and nothing else may move.
//!
//! Driven through `build_remap` rather than the hiding helper directly, because
//! the ordering is half the contract — hiding must happen AFTER the coverage
//! gate, or every ablated run is refused for "missing" the genes it withheld on
//! purpose.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

fn opts_hiding(hidden: &[&str], min_overlap: f32) -> QueryNameOpts {
    QueryNameOpts {
        min_overlap,
        hide: Some(std::sync::Arc::new(
            hidden.iter().map(|s| Box::from(*s)).collect(),
        )),
        ..Default::default()
    }
}

#[test]
fn ablation_hides_exactly_the_named_features() {
    let genes = names(&["a", "b", "c", "d"]);
    // Axes match, so the remap would normally be `None`; hiding forces an
    // identity one, and only the named rows differ from it.
    let out = build_remap(&genes, &genes, &opts_hiding(&["b", "d"], 0.0))
        .expect("remap")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train, vec![Some(0), None, Some(2), None]);
    assert_eq!(out.n_mapped, 2);
}

#[test]
fn hiding_survives_a_real_axis_mismatch() {
    // Query carries a gene the model lacks; hiding must not resurrect it or
    // renumber the survivors.
    let training = names(&["a", "b", "c"]);
    let query = names(&["a", "zzz", "b", "c"]);
    let out = build_remap(&training, &query, &opts_hiding(&["c"], 0.0))
        .expect("remap")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train[1], None, "unmatched gene stays unmatched");
    assert_eq!(out.new_to_train[3], None, "named gene is hidden");
    assert_eq!(out.n_mapped, 2);
}

#[test]
fn a_name_that_matches_nothing_is_an_error_not_a_silent_reconstruction() {
    // The failure this guards: a typo'd file leaves every gene visible, the run
    // succeeds, and the reported number is a plain reconstruction wearing the
    // ablation's name.
    let genes = names(&["a", "b"]);
    assert!(build_remap(&genes, &genes, &opts_hiding(&["zzz"], 0.0)).is_err());
}

#[test]
fn hiding_every_feature_is_an_error() {
    let genes = names(&["a", "b"]);
    assert!(build_remap(&genes, &genes, &opts_hiding(&["a", "b"], 0.0)).is_err());
}

#[test]
fn coverage_is_gated_before_hiding_not_after() {
    // Hiding half the axis must not be read as half the axis going missing.
    // Ordering it the other way refuses every ablated run under any real
    // --min-gene-overlap.
    let genes = names(&["a", "b", "c", "d"]);
    let out = build_remap(&genes, &genes, &opts_hiding(&["a", "b"], 0.9))
        .expect("a 90% floor must still pass: nothing is missing, two are withheld")
        .expect("hiding always yields a remap");
    assert_eq!(out.n_mapped, 2);
}

/// The panel file and the data may disagree on case while naming the same
/// genes. The remap matches lowercased, so the model resolves fine — but the
/// hide set used to match exactly, so a lowercase panel against uppercase rows
/// hid nothing and errored with "matched no feature", pointing at the wrong
/// cause entirely.
#[test]
fn hiding_matches_case_insensitively_like_the_remap_does() {
    let genes = names(&["Cd8a", "GZMB", "ms4a1"]);
    let out = build_remap(&genes, &genes, &opts_hiding(&["CD8A", "Ms4a1"], 0.0))
        .expect("case must not defeat the hide")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train, vec![None, Some(1), None]);
}

/// The scoring cap exists for one reason: a dense block's working set must not
/// be multiplied by every thread. It has to bite at whole-transcriptome width
/// and stay out of the way at the coarsened widths the topic paths score on,
/// which have never had a memory problem.
mod block_concurrency {
    use super::super::{
        block_concurrency, dense_bytes, DEFAULT_PREDICT_BUDGET_BYTES, NB_CHAIN_TENSORS,
    };

    // Every claim below is made AT a stated machine size. The cap is
    // budget / bytes clamped to the thread count, so an assertion against the
    // live `rayon::current_num_threads()` is a different claim on every box
    // (it held on a 32-thread machine and failed on a 64-thread one).
    const BUDGET: usize = DEFAULT_PREDICT_BUDGET_BYTES;
    const THREADS: usize = 64;

    fn whole_transcriptome_dense() -> usize {
        // The reported OOM's shape: ~58k genes at the default minibatch.
        dense_bytes(500, 57_843, NB_CHAIN_TENSORS)
    }

    fn whole_transcriptome_chunked() -> usize {
        // The same shape `score_vae_backend` hands the cap: the encoder input
        // plus one gene chunk's likelihood chain.
        dense_bytes(500, 57_843, 1)
            + dense_bytes(
                500,
                crate::topic::predict_common::SCORE_GENE_CHUNK,
                NB_CHAIN_TENSORS,
            )
    }

    #[test]
    fn a_whole_transcriptome_dense_block_is_capped_well_below_the_thread_count() {
        let conc = block_concurrency(whole_transcriptome_dense(), BUDGET, THREADS);
        assert!(
            conc <= 8,
            "58k-gene dense blocks must not run wide open; got {conc}"
        );
        assert!(conc >= 1, "the cap must always admit at least one block");
    }

    #[test]
    fn a_coarsened_block_is_not_throttled() {
        // What a dense topic model actually scores on after coarsening: the
        // cap must return the full thread count, i.e. change nothing.
        assert_eq!(
            block_concurrency(dense_bytes(500, 2_000, NB_CHAIN_TENSORS), BUDGET, THREADS),
            THREADS
        );
    }

    /// The gene-chunked vae scorer holds the encoder input plus one slice, so
    /// the same 58k-gene query that pins the dense path to a handful of blocks
    /// must get an order of magnitude more once the likelihood stops
    /// materialising `[N, D]`. The budget still applies — it is a memory bound,
    /// not a dense-path special case — so "not throttled" is stated at the
    /// thread count where it is true.
    #[test]
    fn the_chunked_vae_path_is_far_less_throttled_at_the_same_width() {
        let chunked = block_concurrency(whole_transcriptome_chunked(), BUDGET, THREADS);
        let dense = block_concurrency(whole_transcriptome_dense(), BUDGET, THREADS);
        assert!(
            chunked >= 4 * dense && chunked >= 32,
            "chunked {chunked} vs dense {dense} blocks at {THREADS} threads"
        );
    }

    /// Tripwire on the per-block cost of chunked scoring: at the default
    /// budget a 32-thread box runs it wide open. Growing `SCORE_GENE_CHUNK` or
    /// the chain's tensor count past that point is a decision, not a drift.
    #[test]
    fn the_chunked_vae_path_runs_wide_open_on_a_32_thread_box() {
        assert_eq!(
            block_concurrency(whole_transcriptome_chunked(), BUDGET, 32),
            32
        );
    }

    #[test]
    fn an_absurd_block_still_admits_one() {
        assert_eq!(block_concurrency(usize::MAX, BUDGET, THREADS), 1);
    }
}

////////////////////////////////////////////////
// `--bulk` is an alternative input, not an add-on //
////////////////////////////////////////////////

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    args: PredictArgs,
}

fn parse(argv: &[&str]) -> Result<PredictArgs, clap::Error> {
    use clap::Parser;
    Cli::try_parse_from(std::iter::once("senna-predict").chain(argv.iter().copied()))
        .map(|c| c.args)
}

#[test]
fn bulk_alone_parses_with_no_data_files() {
    let a = parse(&["--model", "m", "-o", "p", "--bulk", "counts.parquet"]).expect("parses");
    assert!(a.data_files.is_empty());
    assert_eq!(a.bulk, vec![Box::from("counts.parquet")]);
    assert_eq!(
        a.bulk_table.bulk_orientation,
        crate::embed_common::OrientationArg::Auto
    );
}

#[test]
fn a_data_file_alone_still_parses() {
    let a = parse(&["held.zarr", "--model", "m", "-o", "p"]).expect("parses");
    assert_eq!(a.data_files, vec![Box::from("held.zarr")]);
    assert!(a.bulk.is_empty());
}

/// Both at once is a contradiction to refuse at the command line, not a
/// precedence rule to remember.
#[test]
fn bulk_and_a_data_file_together_are_refused() {
    assert!(parse(&[
        "held.zarr",
        "--model",
        "m",
        "-o",
        "p",
        "--bulk",
        "c.parquet"
    ])
    .is_err());
}

#[test]
fn neither_bulk_nor_a_data_file_is_refused() {
    assert!(parse(&["--model", "m", "-o", "p"]).is_err());
}

#[test]
fn bulk_orientation_is_a_value_enum() {
    let a = parse(&[
        "--model",
        "m",
        "-o",
        "p",
        "--bulk",
        "c.tsv",
        "--bulk-orientation",
        "samples-by-genes",
    ])
    .expect("parses");
    assert_eq!(
        a.bulk_table.bulk_orientation.forced(),
        Some(crate::embed_common::Orientation::SamplesByGenes)
    );
}

#[test]
fn bulk_header_defaults_to_auto_and_parses_yes_no() {
    let a = parse(&["--model", "m", "-o", "p", "--bulk", "c.tsv"]).expect("parses");
    assert_eq!(
        a.bulk_table.bulk_header,
        crate::embed_common::HeaderArg::Auto
    );
    let a = parse(&[
        "--model",
        "m",
        "-o",
        "p",
        "--bulk",
        "c.tsv",
        "--bulk-header",
        "yes",
    ])
    .expect("parses");
    assert_eq!(
        a.bulk_table.bulk_header,
        crate::embed_common::HeaderArg::Yes
    );
    let a = parse(&[
        "--model",
        "m",
        "-o",
        "p",
        "--bulk",
        "c.tsv",
        "--bulk-header",
        "no",
    ])
    .expect("parses");
    assert_eq!(a.bulk_table.bulk_header, crate::embed_common::HeaderArg::No);
}
