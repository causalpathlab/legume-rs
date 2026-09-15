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

/// The gene lists are contractually one file shared across arms, and the arms
/// do not all spell their axes the same way: a model trained under the
/// canonical rule has bare symbols where a list keyed on the raw backend rows
/// has `ENSG..._SYMBOL`. The list must hide the same genes either way.
#[test]
fn hiding_bridges_raw_list_names_onto_a_canonical_axis() {
    let genes = names(&["tspan6", "tnmd", "dpm1"]);
    let out = build_remap(
        &genes,
        &genes,
        &opts_hiding(&["ENSG00000000003_TSPAN6", "ENSG00000000419_DPM1"], 0.0),
    )
    .expect("raw names must resolve onto the canonical axis")
    .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train, vec![None, Some(1), None]);
}

#[test]
fn hiding_bridges_symbol_list_names_onto_a_raw_axis() {
    let genes = names(&[
        "ENSG00000000003_TSPAN6",
        "ENSG00000000005_TNMD",
        "ENSG00000000419_DPM1",
    ]);
    let out = build_remap(&genes, &genes, &opts_hiding(&["TSPAN6", "DPM1"], 0.0))
        .expect("symbols must resolve onto the raw axis")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train, vec![None, Some(1), None]);
}

/// When the list spells names exactly as the axis does, the exact matches are
/// the whole answer: the canonical rule must not widen a hit onto a second row
/// that merely shares a suffix.
#[test]
fn an_exact_hit_does_not_widen_to_suffix_sharing_rows() {
    let genes = names(&["gene_0", "other_0", "gene_1"]);
    let out = build_remap(&genes, &genes, &opts_hiding(&["gene_0"], 0.0))
        .expect("remap")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train, vec![None, Some(1), Some(2)]);
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

    /// A CUDA device is one stream and one cuBLAS handle; blocks driven at it
    /// from a thread pool raced (CUBLAS_STATUS_EXECUTION_FAILED, or a hang).
    /// Off the CPU exactly one block is in flight, and the memory budget —
    /// which only ever lowers the ceiling — cannot raise it back.
    #[test]
    fn off_the_cpu_exactly_one_block_is_in_flight() {
        use crate::topic::common::device_concurrency;
        assert_eq!(device_concurrency(false, THREADS), 1);
        assert_eq!(device_concurrency(true, THREADS), THREADS);
        // How `dense_block_concurrency` composes the two ceilings.
        assert_eq!(
            block_concurrency(1, BUDGET, device_concurrency(false, THREADS)),
            1
        );
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

//////////////////////////////////////////////////////////////////
// `--feature-name-kind` reaches the loader, not only the remap //
//////////////////////////////////////////////////////////////////

/// The flag used to be applied to the query→dictionary remap only, while the
/// backend was still loaded under auto-detection, so `exact` on an
/// exact-trained model still got a canonicalized query. Unset keeps the legacy
/// pairing (loader auto, remap exact-then-flexible); a value now drives both.
mod feature_name_kind_reaches_the_loader {
    use super::parse;
    use auxiliary_data::feature_names::FeatureNameKind;

    const BASE: &[&str] = &["q.zarr", "--model", "m", "-o", "out"];

    fn with<'a>(extra: &'a [&'a str]) -> Vec<&'a str> {
        BASE.iter().chain(extra).copied().collect()
    }

    #[test]
    fn unset_keeps_the_legacy_pairing() {
        let opts = parse(&with(&[])).unwrap().query_name_opts().unwrap();
        assert_eq!(opts.loader_kind, None);
        assert_eq!(opts.kind, FeatureNameKind::Exact);
    }

    #[test]
    fn exact_pins_both_the_loader_and_the_remap() {
        let opts = parse(&with(&["--feature-name-kind", "exact"]))
            .unwrap()
            .query_name_opts()
            .unwrap();
        assert_eq!(opts.loader_kind, Some(FeatureNameKind::Exact));
        assert_eq!(opts.kind, FeatureNameKind::Exact);
    }

    #[test]
    fn explicit_auto_means_loader_auto_and_gene_remap() {
        let opts = parse(&with(&["--feature-name-kind", "auto"]))
            .unwrap()
            .query_name_opts()
            .unwrap();
        assert_eq!(opts.loader_kind, None);
        assert_eq!(opts.kind, FeatureNameKind::Gene { delim: '_' });
    }
}

/// End to end: `senna predict` on a `senna gem` run. Reuses the shared gem
/// synthetic fixture (`crate::gem::test_fixtures`) rather than duplicating
/// it — the same axis / cell shapes `gem::run::tests` and
/// `bge::score::tests` already exercise.
mod gem_predict {
    use super::parse;
    use crate::gem::args::GemArgs;
    use crate::gem::run::run_gem_embedding;
    use crate::gem::test_fixtures::{genes_file, m6a_file, synth, CELLS};
    use clap::Parser;
    use matrix_util::traits::IoOps;

    #[derive(Parser)]
    struct GemCli {
        #[command(flatten)]
        args: GemArgs,
    }

    /// A tiny mixed-axis (count + m6a) gem fit; returns `(genes file, -o
    /// prefix)`. GENE1 carries both count channels and both m6a channels;
    /// GENE2 only `count/spliced`.
    fn fit_gem(dir: &std::path::Path) -> (Box<str>, String) {
        let genes = genes_file(dir);
        let m6a = m6a_file(dir);
        let out = dir.join("run").to_string_lossy().into_owned();
        let cli = GemCli::try_parse_from([
            "senna-gem",
            &genes,
            "--modality",
            &m6a,
            "--epochs",
            "2",
            "--skip-etm",
            "--no-emit-pb-reference",
            "--embedding-dim",
            "4",
            "--phase1-cells-per-pb",
            "0",
            "-o",
            &out,
        ])
        .expect("GemArgs parses");
        run_gem_embedding(&cli.args).expect("gem run must succeed");
        (genes, out)
    }

    #[test]
    fn predict_on_a_gem_run_writes_one_finite_row_per_cell() {
        let dir = tempfile::tempdir().unwrap();
        let (genes, out) = fit_gem(dir.path());
        let pout = dir.path().join("pred").to_string_lossy().into_owned();

        let args = parse(&[&genes, "--model", &out, "-o", &pout]).expect("PredictArgs parses");
        super::predict_model(&args).expect("predict on a gem run");

        let pred = super::Mat::from_parquet(&format!("{pout}.predictive.parquet"))
            .expect("read predictive.parquet");
        assert_eq!(pred.mat.nrows(), CELLS.len(), "one row per cell");
        assert!(
            pred.mat.iter().all(|v| v.is_finite()),
            "every predictive value must be finite: {:?}",
            pred.mat
        );
    }

    /// The bare barcode: gem's training-time union tags every barcode
    /// `{barcode}@{sample}` whenever it loads more than one file (see
    /// `gem::load::load_gem_data`), while predict's query loader has no
    /// notion of gem's per-sample convention and never adds the tag. Strip
    /// it, so the alignment is on cell identity, not on which loader wrote
    /// the row name.
    fn bare_barcode(name: &str) -> &str {
        name.split('@').next().unwrap_or(name)
    }

    /// Mean cosine between rows of `a` and `b`, paired by NAME (not
    /// position) — `predict`'s own row order need not match the training
    /// run's `cell_embedding.parquet` order.
    fn mean_paired_cosine_by_name(
        a: &matrix_util::traits::MatWithNames<super::Mat>,
        b: &matrix_util::traits::MatWithNames<super::Mat>,
    ) -> f32 {
        use std::collections::HashMap;
        let idx: HashMap<&str, usize> = a
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (bare_barcode(r), i))
            .collect();
        let mut total = 0f64;
        let mut n = 0usize;
        for (j, name) in b.rows.iter().enumerate() {
            let i = *idx
                .get(bare_barcode(name))
                .unwrap_or_else(|| panic!("row {name} in b is missing from a: {:?}", a.rows));
            let ra = a.mat.row(i);
            let rb = b.mat.row(j);
            let (na, nb) = (ra.norm(), rb.norm());
            let cos = if na > 1e-10 && nb > 1e-10 {
                f64::from(ra.dot(&rb) / (na * nb))
            } else {
                0.0
            };
            total += cos;
            n += 1;
        }
        (total / n as f64) as f32
    }

    /// Track-aware placement: a query on the run's OWN training data must
    /// land close to where phase 2 itself placed those same cells
    /// (`{out}.cell_embedding.parquet`), not merely somewhere finite.
    ///
    /// The query here is the genes file alone (both count tracks,
    /// `count/spliced` + `count/unspliced` — the axis's m6a rows go
    /// unobserved, same as `a_spliced_only_query_also_succeeds`'s query but
    /// wider), which already carries the defect this test is for: TWO count
    /// tracks needing two Poisson partitions and two intercepts, not one.
    /// A single-partition polish mixes them into one partition with one
    /// intercept — not what phase 2's own per-track polish
    /// (`block_sgd::polish_cells`, Task 4a) fit these cells with — so a
    /// query cell reconstructs somewhere else in H-space. bge itself passes
    /// this same check at ~1.000 cosine (a query on a bge run's own
    /// training data is a degenerate, exact instance of "place a cell where
    /// its own model placed it").
    ///
    /// The modality (m6a) file is deliberately NOT added to the query: a
    /// gem axis's three-field row grammar has no multi-file query-loading
    /// path today (`multiome_layout::query_load` refuses to treat it as a
    /// multiome layout by design, so two files fall back to
    /// `ColumnAlignment::Disjoint` and are read as 12 DISJOINT cells, not 6
    /// cells glued across feature blocks — confirmed empirically, not
    /// assumed). Fixing that loader gap is unrelated to the track-aware
    /// polish this test is for, and is out of this fix round's scope.
    #[test]
    fn predict_reproduces_the_runs_own_cell_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let (genes, out) = fit_gem(dir.path());
        let pout = dir.path().join("pred_repro").to_string_lossy().into_owned();

        let args = parse(&[&genes, "--model", &out, "-o", &pout]).expect("PredictArgs parses");
        super::predict_model(&args).expect("predict on the run's own training data");

        let train = super::Mat::from_parquet(&format!("{out}.cell_embedding.parquet"))
            .expect("read the run's own cell_embedding.parquet");
        let query = super::Mat::from_parquet(&format!("{pout}.latent.parquet"))
            .expect("read predict's latent.parquet");

        let cos = mean_paired_cosine_by_name(&train, &query);
        assert!(
            cos > 0.99,
            "predict must reproduce the run's own cell placement track-aware \
             (mean cosine {cos:.4}, want > 0.99)"
        );
    }

    /// A query carrying only the base (`count/spliced`) rows — no
    /// `unspliced`, no `m6a` at all — the shape a real single-modality query
    /// would have. It must still be placeable: the encoder set's combine
    /// rule means over whichever count tracks a cell has counts on, and a
    /// cell with none on `count/unspliced` still has `count/spliced`.
    #[test]
    fn a_spliced_only_query_also_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let (_genes, out) = fit_gem(dir.path());
        let query = synth(
            dir.path(),
            "query",
            &["GENE1/count/spliced", "GENE2/count/spliced"],
            &CELLS,
        );
        let pout = dir.path().join("pred2").to_string_lossy().into_owned();

        let args = parse(&[&query, "--model", &out, "-o", &pout]).expect("PredictArgs parses");
        super::predict_model(&args).expect("predict with a spliced-only query must succeed");

        let pred = super::Mat::from_parquet(&format!("{pout}.predictive.parquet"))
            .expect("read predictive.parquet");
        assert_eq!(pred.mat.nrows(), CELLS.len());
        assert!(pred.mat.iter().all(|v| v.is_finite()));
    }

    /// A well-formed dense bulk table on the run's own gene axis, so
    /// materialization itself succeeds and the refusal reached is the ONE
    /// this task added (a gem axis names track-grammar rows, not plain
    /// genes), not an unrelated orientation failure.
    #[test]
    fn predict_bulk_is_refused_on_a_gem_run() {
        let dir = tempfile::tempdir().unwrap();
        let (_genes, out) = fit_gem(dir.path());

        let bulk_rows: Vec<Box<str>> = vec![
            "GENE1/count/spliced".into(),
            "GENE1/count/unspliced".into(),
            "GENE2/count/spliced".into(),
        ];
        let bulk_mat = super::Mat::from_row_slice(3, 2, &[3.0, 1.0, 2.0, 4.0, 0.0, 5.0]);
        let bulk_path = dir.path().join("bulk.parquet");
        bulk_mat
            .to_parquet_with_names(
                bulk_path.to_str().unwrap(),
                (Some(&bulk_rows), Some("gene")),
                Some(&[Box::from("s0"), Box::from("s1")]),
            )
            .expect("write the bulk table");
        let bulk_path = bulk_path.to_string_lossy().into_owned();

        let pout = dir.path().join("pred3").to_string_lossy().into_owned();
        let args = parse(&["--model", &out, "-o", &pout, "--bulk", &bulk_path])
            .expect("PredictArgs parses");
        let err = match super::predict_model(&args) {
            Ok(()) => panic!("predict --bulk on a gem run must be refused"),
            Err(e) => e,
        };
        assert!(
            err.to_string()
                .contains("predict --bulk is not available on a gem run"),
            "{err}"
        );
    }

    /// A second query file must be refused, not silently scored against
    /// disjoint cells: `crate::multiome_layout::query_load` does not apply
    /// gem's per-file `@sample` tagging, so two files fall back to
    /// `ColumnAlignment::Disjoint` (see
    /// `predict_reproduces_the_runs_own_cell_embedding`'s doc comment for the
    /// confirmed root cause). One file must still succeed.
    #[test]
    fn a_second_query_file_is_refused_on_a_gem_run_but_one_file_still_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let (genes, out) = fit_gem(dir.path());
        let m6a = m6a_file(dir.path());
        let pout = dir.path().join("pred4").to_string_lossy().into_owned();

        let args = parse(&[&genes, &m6a, "--model", &out, "-o", &pout])
            .expect("PredictArgs parses two query files");
        let err = match super::predict_model(&args) {
            Ok(()) => panic!("predict with two query files on a gem run must be refused"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains(
                "the query loader does not yet join a gene file with modality files for a \
                 gem query"
            ),
            "{err}"
        );

        // One file still succeeds.
        let args = parse(&[&genes, "--model", &out, "-o", &pout]).expect("PredictArgs parses");
        super::predict_model(&args).expect("a single query file must still succeed");
    }
}
