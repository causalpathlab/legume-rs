//! Stage-1 pseudobulks must pool individuals: a pseudobulk that holds too
//! few individuals cannot separate its cell-state rate from their
//! multipliers, and is dropped. The run reports per topic how many
//! pseudobulks and cells were kept, and how many individuals each kept
//! pseudobulk links.

mod common;
use common::*;

#[test]
fn pseudobulks_pool_individuals_and_the_run_reports_it() {
    let fx = build(&Spec {
        n_genes: 30,
        n_indv: 12,
        n_states: 3,
        cells_per_indv: 60,
        exposure_shifts_states: false,
    });
    let out = fx.out("out");
    diff(&fx, &fx.one_topic(), &fx.exposures(), &[], &out);

    let report = format!("{out}.stage1.parquet");
    let median = column(&report, "median_individuals_per_pseudobulk")[0];
    let kept = column(&report, "cells_kept")[0];
    let dropped = column(&report, "cells_dropped")[0];
    assert!(
        median >= 3.0,
        "median individuals per kept pseudobulk = {median}"
    );
    assert!(
        dropped < 0.5 * (kept + dropped),
        "dropped {dropped} of {} cells",
        kept + dropped
    );
}
