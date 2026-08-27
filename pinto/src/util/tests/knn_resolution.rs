//! How `-k` and `--knn-expr` resolve into the two k's the pipeline actually
//! uses, and which combinations are refused.
//!
//! Two flags, two graphs, and the mapping between them depends on whether
//! `--coord` was given. `-k` sizes the BASE graph; `--knn-expr` sizes the
//! expression graph that is unioned into it. Without coordinates there is
//! nothing to union into, so `--knn-expr` sizes the base graph instead and the
//! augmentation is off.
//!
//! The failures these guard are silent ones. A `k` of 0 does not build an empty
//! graph, it floors to 1-NN, which then fragments into hundreds of components
//! and — in expression mode, where component folding is off — hundreds of
//! batches. And a resolution that returned the same number for both roles would
//! union the expression graph with itself.

use crate::util::input::{KnnExprScope, SrtInputArgs};
use crate::util::metadata::GraphParams;
use clap::Parser;

/// Parse an argv the way a subcommand would, minus the subcommand.
fn args(extra: &[&str]) -> SrtInputArgs {
    let mut argv = vec!["pinto", "d.zarr", "-o", "out"];
    argv.extend_from_slice(extra);
    SrtInputArgs::try_parse_from(argv).expect("args should parse")
}

/// Enough cells that no bound fires, so those rows test resolution alone.
const PLENTY: usize = 1000;

/// The resolution table, as a table.
///
/// Rows are `(argv, base, augment)`. Those two columns are the whole contract:
/// which k builds the base graph, and which k (if any) is unioned into it.
#[test]
fn flags_resolve_to_the_two_ks_the_pipeline_uses() {
    let rows: &[(&[&str], usize, usize)] = &[
        // With coordinates: `-k` sizes the spatial graph, `--knn-expr` the
        // union. Nothing set is the new default — spatial pairs only.
        (&["-c", "coords.csv"], 5, 0),
        (&["-c", "coords.csv", "-k", "12"], 12, 0),
        // Asking for expression pairs is what turns the union on; the spatial
        // side keeps its own default.
        (&["-c", "coords.csv", "--knn-expr", "7"], 5, 7),
        (&["-c", "coords.csv", "-k", "12", "--knn-expr", "7"], 12, 7),
        // Without coordinates the base graph IS the expression graph, so the
        // base must never come out as 0 and the augmentation is always 0.
        (&[], 5, 0),
        // An existing `-k` keeps building the graph it always built: k moves
        // the force-directed layout, so re-sizing it would move every plot.
        (&["-k", "20"], 20, 0),
        // `--knn-expr` is the expression-mode knob.
        (&["--knn-expr", "20"], 20, 0),
        // `--knn-expr 0` asks for no expression pairs. There are none to turn
        // off here, so it falls through rather than sizing the graph at 0.
        (&["--knn-expr", "0"], 5, 0),
        (&["--knn-expr", "0", "-k", "12"], 12, 0),
    ];
    for (argv, base, augment) in rows {
        let c = args(argv);
        assert_eq!(c.base_knn(), *base, "base for {argv:?}");
        assert_eq!(c.augment_knn(), *augment, "augment for {argv:?}");
        let knn = c
            .resolve_knn(PLENTY)
            .unwrap_or_else(|e| panic!("{argv:?} should resolve: {e}"));
        assert_eq!(knn.base, *base, "resolved base for {argv:?}");
        assert_eq!(knn.augment, *augment, "resolved augment for {argv:?}");
    }
}

/// Absence must be distinguishable from the default value, or the resolution
/// above cannot tell rows apart. A stray `default_value_t` would make this
/// `Some(5)` and every "was it set?" branch would silently take the wrong arm.
#[test]
fn unset_flags_parse_as_none() {
    let c = args(&["-c", "coords.csv"]);
    assert_eq!(c.knn_spatial, None);
    assert_eq!(c.knn_expr, None);
}

/// `-k` and `--knn-expr` both set without coordinates name the same graph.
/// This has to be refused rather than warned about: the choice changes the
/// connected-component count, which in expression mode becomes the batch
/// partition, which decides whether batch-effect estimation runs at all.
#[test]
fn no_coords_with_both_flags_is_an_error() {
    let err = args(&["-k", "20", "--knn-expr", "7"])
        .resolve_knn(PLENTY)
        .expect_err("must refuse");
    let msg = err.to_string();
    assert!(msg.starts_with("-k and --knn-expr"), "{msg}");
    assert!(msg.contains("Pass only --knn-expr"), "{msg}");
}

/// A base k of 0 must be refused. The KNN builder floors k at 1 rather than
/// rejecting it, so nothing downstream will catch this.
///
/// Asserted with `starts_with`, not `contains`: `"--knn-expr".contains("-k")`
/// is true, so a `contains` check here could not fail.
#[test]
fn a_zero_base_is_refused() {
    for argv in [vec!["-c", "coords.csv", "-k", "0"], vec!["-k", "0"]] {
        let err = args(&argv).resolve_knn(PLENTY).expect_err("must refuse");
        assert!(err.to_string().starts_with("-k "), "for {argv:?}: {err}");
    }
}

/// Errors must name the flag that actually supplied the value, so the remedy
/// is one the user can act on. Asserted through the messages rather than the
/// private resolver, because the messages are the whole reason it exists.
#[test]
fn errors_name_the_flag_that_supplied_the_value() {
    // A cell count small enough that the upper bound fires on the default k,
    // so every row reaches a message that names a flag.
    let cases: &[(&[&str], &str)] = &[
        (&["-c", "coords.csv"], "-k "),
        (&["-c", "coords.csv", "-k", "9"], "-k "),
        (&["-k", "9"], "-k "),
        (&["--knn-expr", "9"], "--knn-expr "),
        // Nothing typed: in expression mode the knob to point at is the one
        // that names this graph, not the one the help calls spatial.
        (&[], "--knn-expr "),
    ];
    for (argv, want) in cases {
        let err = args(argv).resolve_knn(3).expect_err("bound must fire");
        assert!(
            err.to_string().starts_with(want),
            "for {argv:?} wanted {want:?}, got: {err}"
        );
    }
}

/// 0 means "no expression pairs" and stays legal when there is a spatial graph
/// to fall back on. The `augment` assertion is the point: without it a slip
/// that read 0 as "unset" would silently re-enable augmentation at k=5.
#[test]
fn a_zero_augmentation_is_legal_with_coords() {
    let knn = args(&["-c", "coords.csv", "--knn-expr", "0"])
        .resolve_knn(PLENTY)
        .expect("0 turns augmentation off");
    assert_eq!(knn.augment, 0, "and it must actually be off");
    assert_eq!(knn.base, 5, "the spatial side is untouched");
}

/// k at or above the cell count asks for a complete graph, in either role.
#[test]
fn a_k_beyond_the_cell_count_is_refused() {
    for argv in [
        vec!["-c", "coords.csv", "-k", "1000"],
        vec!["-c", "coords.csv", "--knn-expr", "5000"],
        vec!["--knn-expr", "1000"],
    ] {
        assert!(
            args(&argv).resolve_knn(PLENTY).is_err(),
            "{argv:?} must be refused"
        );
    }
}

/// A fatal k outranks a scope complaint: fixing the scope first would only
/// earn the user a second error on the next run.
#[test]
fn a_fatal_k_is_reported_before_a_scope_complaint() {
    let err = args(&["-c", "coords.csv", "-k", "0", "--knn-expr-scope", "within"])
        .resolve_knn(PLENTY)
        .expect_err("must refuse");
    assert!(err.to_string().starts_with("-k "), "{err}");
}

/// `within` scopes the search to components of the SPATIAL graph. Without
/// coordinates there is none, and the flag cannot be given a meaning there:
/// every cell's k-NN already sit inside its own component.
#[test]
fn within_scope_without_coords_is_an_error() {
    let c = args(&["--knn-expr-scope", "within"]);
    assert_eq!(c.knn_expr_scope, KnnExprScope::Within);
    let err = c.resolve_knn(PLENTY).expect_err("must refuse");
    assert!(err.to_string().contains("needs --coord"), "{err}");
}

/// The case the 0 default creates: coordinates present, but no expression
/// pairs to scope. The flag would scope nothing and say nothing, which is the
/// silent-ignore this validation exists to refuse.
#[test]
fn within_scope_without_expression_pairs_is_an_error() {
    let err = args(&["-c", "coords.csv", "--knn-expr-scope", "within"])
        .resolve_knn(PLENTY)
        .expect_err("must refuse");
    // `--knn-expr` is a substring of `--knn-expr-scope`, which this message
    // opens with, so a bare `contains` could not fail. Match the remedy.
    assert!(
        err.to_string().contains("Pass --knn-expr N"),
        "must name the flag to add: {err}"
    );
}

/// The same scope is fine when there is a spatial graph to partition AND
/// expression pairs to scope.
#[test]
fn within_scope_with_coords_and_pairs_is_fine() {
    args(&[
        "-c",
        "coords.csv",
        "--knn-expr",
        "5",
        "--knn-expr-scope",
        "within",
    ])
    .resolve_knn(PLENTY)
    .expect("within is what this flag is for");
}

/// What the manifest records, taken through the real conversion rather than a
/// hand-built struct. Without this the resolution-to-manifest mapping is
/// unpinned: inverting the scope predicate would swap which runs claim a scope
/// and leave every serde round-trip test green.
#[test]
fn graph_params_record_what_the_run_did() {
    let params = |argv: &[&str]| -> GraphParams {
        (&args(argv).resolve_knn(PLENTY).expect("should resolve")).into()
    };

    let augmented = params(&[
        "-c",
        "coords.csv",
        "--knn-expr",
        "7",
        "--knn-expr-scope",
        "within",
    ]);
    assert_eq!(augmented.knn_base, 5);
    assert_eq!(augmented.knn_expr, 7);
    assert!(augmented.knn_expr > 0, "this run unioned expression pairs in");
    assert_eq!(
        augmented.knn_expr_scope.as_deref(),
        Some("within"),
        "the CLI spelling, not the Debug name"
    );

    let spatial_only = params(&["-c", "coords.csv"]);
    assert_eq!(spatial_only.knn_expr, 0);

    assert_eq!(
        spatial_only.knn_expr_scope, None,
        "no search ran, so no scope may be claimed"
    );

    // Expression mode: the k that built the graph is the base, and there is
    // no augmentation to report.
    let expression = params(&["--knn-expr", "20"]);
    assert_eq!(expression.knn_base, 20);
    assert_eq!(expression.knn_expr, 0);


    assert!(params(&["-c", "coords.csv", "--reciprocal"]).reciprocal);
}
