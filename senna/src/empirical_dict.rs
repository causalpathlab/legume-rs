//! Empirical NB-Fisher-weighted gene × topic dictionary for downstream
//! annotation.
//!
//! The feature-coarsened β from topic training lives at a reduced resolution
//! (~1k meta-features) and is expanded back to full gene resolution via
//! `expand_log_dict_dk`, which spreads each meta-feature's loading across its
//! constituent genes proportional to the training-time sufficient statistics.
//! That expansion is a lossy approximation — rare informative genes (e.g.
//! hemoglobins for erythroid cells) can get merged into generic meta-features
//! and lose lineage-specific contrast at reporting time.
//!
//! This module computes an alternative *empirical* β at full gene resolution:
//!
//!     β_emp[g, k] = NB_fisher_weight[g] · Σ_p pb_gene[g, p] · pb_theta[p, k]
//!
//! then column-normalizes to the probability simplex. The NB Fisher-info
//! weight is computed via the shared
//! `data_beans_alg::gene_weighting::compute_nb_fisher_weights` (same formula
//! used during DC-Poisson refinement, kept consistent across pinto / senna /
//! chickpea reporting). This is the dictionary `senna annotate` consumes.

use crate::embed_common::Mat;
pub use data_beans_alg::gene_weighting::{apply_gene_weights, compute_nb_fisher_weights};
use matrix_util::traits::MatOps;

/// Build the empirical β: scale `pb_gene · pb_theta` rows by per-gene NB
/// Fisher weights, then column-normalize so each topic column is a
/// probability distribution over genes.
pub fn build_empirical_dictionary(
    pb_gene_gp: &Mat,
    pb_theta_pk: &Mat,
    fisher_weights: &[f32],
) -> Mat {
    debug_assert_eq!(pb_gene_gp.ncols(), pb_theta_pk.nrows());
    debug_assert_eq!(fisher_weights.len(), pb_gene_gp.nrows());

    let mut sum_gk = pb_gene_gp * pb_theta_pk;
    apply_gene_weights(&mut sum_gk, fisher_weights);
    sum_gk.sum_to_one_columns_inplace();
    sum_gk
}
