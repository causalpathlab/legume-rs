//! The accumulate pass hands the group model raw counts. There is no
//! per-gene row scale on the sufficient statistics: the model carries a
//! per-gene dispersion prior, so a scale would count the same variance twice
//! and turn the Poisson counts its likelihood expects into something else.
//! It takes no exposure labels: stage 1 depends on cell state only.

use super::*;

/// The accumulate input has no gene-weight slot and no exposure labels.
#[test]
fn accumulate_input_takes_raw_counts_and_no_labels() {
    let input = CocoaCollapseIn {
        min_individuals_per_pb: 3,
        n_genes: 1,
        n_topics: 1,
        n_opt_iter: None,
        hyper_param: None,
        cell_topic_nk: Mat::zeros(1, 1),
    };
    assert_eq!(input.cell_topic_nk.ncols(), input.n_topics);
}
