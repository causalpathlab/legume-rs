//! The matched accumulate pass hands the group model raw counts. There is no
//! per-gene row scale on the sufficient statistics: the model carries a
//! per-gene dispersion prior, so a scale would count the same variance twice
//! and turn the Poisson counts its likelihood expects into something else.

use super::*;

/// The accumulate input has no gene-weight slot.
#[test]
fn accumulate_input_takes_raw_counts_only() {
    let exposure = vec![0usize, 1];
    let input = CocoaCollapseIn {
        n_genes: 1,
        n_topics: 1,
        knn: 1,
        n_opt_iter: None,
        hyper_param: None,
        cell_topic_nk: Mat::zeros(1, 1),
        exposure_assignment: &exposure,
    };
    assert_eq!(input.exposure_assignment.len(), 2);
}

/// The permutation replay takes the same inputs as the accumulate pass and
/// nothing that could rescale a gene.
#[test]
fn replay_signature_has_no_gene_weights() {
    type Replay = fn(
        &MatchCache,
        &Mat,
        &[usize],
        usize,
        usize,
        Option<usize>,
        Option<(f32, f32)>,
    ) -> anyhow::Result<CocoaStat>;
    let _replay: Replay = MatchCache::replay_with_exposure;
}
