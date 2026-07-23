use super::*;

/// Build a `QcArgs` with clap's declared defaults, without going through a
/// parser — mirrors what every subcommand gets when the user passes no QC flags.
fn default_args() -> QcArgs {
    QcArgs {
        no_qc: false,
        qc_mads: 5.0,
        qc_min_cell_nnz: 2,
        qc_min_counts: 0.0,
        qc_mito_pattern: None,
        qc_mito_max_frac: None,
        qc_ribo_pattern: None,
        qc_ribo_max_frac: None,
        qc_feature_min_cells: 0,
        qc_report: None,
        qc_histogram: false,
        qc_mad_on_genes: true,
        qc_mad_on_counts: true,
        qc_auto_cutoff: false,
    }
}

/// The MAD gates must REACH `QcConfig`.
///
/// They were hardcoded `false` in `to_config` with no CLI path, which also made
/// `--qc-mads` inert: nothing except a MAD gate reads `n_mads`, so the flag did
/// nothing at all unless `--qc-mito-pattern` happened to be set. This test is
/// the regression guard for that — it fails if either gate is pinned again.
#[test]
fn mad_gates_reach_the_config_and_are_on_by_default() {
    let cfg = default_args().to_config().expect("QC is on by default");
    assert!(cfg.mad_on_n_genes, "--qc-mad-on-genes must reach QcConfig");
    assert!(cfg.mad_on_counts, "--qc-mad-on-counts must reach QcConfig");
    assert_eq!(cfg.n_mads, 5.0, "--qc-mads must reach QcConfig");

    let mut off = default_args();
    off.qc_mad_on_genes = false;
    off.qc_mad_on_counts = false;
    let cfg = off.to_config().unwrap();
    assert!(
        !cfg.mad_on_n_genes && !cfg.mad_on_counts,
        "the gates must be settable OFF"
    );
}

/// `--qc-auto-cutoff` must reach `QcConfig` and stay off by default.
///
/// It was documented in chickpea's help before it existed as a flag, and
/// `auto_cell_cutoff` was hardcoded `false`.
#[test]
fn auto_cutoff_is_reachable_but_off_by_default() {
    assert!(!default_args().to_config().unwrap().auto_cell_cutoff);

    let mut on = default_args();
    on.qc_auto_cutoff = true;
    assert!(on.to_config().unwrap().auto_cell_cutoff);
}

/// Mito stays implicit: it can only run when a pattern selects rows to measure.
/// Without one there is nothing to compute a fraction over, so the gate must
/// stay off rather than silently reporting 0.0 for every cell.
#[test]
fn mito_gate_follows_the_pattern_not_a_flag() {
    assert!(!default_args().to_config().unwrap().mad_on_mito);

    let mut with = default_args();
    with.qc_mito_pattern = Some("(?i)^MT-".to_string());
    assert!(with.to_config().unwrap().mad_on_mito);
}

/// `--no-qc` still short-circuits everything.
#[test]
fn no_qc_disables_the_whole_config() {
    let mut off = default_args();
    off.no_qc = true;
    assert!(off.to_config().is_none());
}
