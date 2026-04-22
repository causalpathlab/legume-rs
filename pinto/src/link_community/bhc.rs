//! Bayesian Hierarchical Clustering over K link communities — shim over the
//! shared implementation in `data_beans_alg::bhc`.
//!
//! Post-hoc bottom-up merge of the K fitted link communities driven by a
//! pairwise Dirichlet-Multinomial marginal-likelihood Bayes factor under an
//! **empirical-Bayes asymmetric Dirichlet prior** centered on the pooled
//! empirical marginal `bg[g]`. See `data_beans_alg::bhc` for the derivation,
//! interpretation table, and the rescaling of `(T, S)` by effective sample
//! size (here: `edge_count[k]`).
//!
//! Housekeeping handling: the empirical-Bayes prior derives its baseline `bg`
//! from the fitted `fine_stats` directly.

pub use data_beans_alg::bhc::{bhc_cut, BhcMerge};

use crate::link_community::model::LinkCommunityStats;
use data_beans_alg::bhc::{bhc_merge as bhc_merge_core, BhcInput};

/// Bottom-up agglomerative merge by pairwise log Bayes factor, driven by the
/// link-community sufficient stats. Effective sample size per cluster is
/// `stats.edge_count[k]` — the number of edges assigned to community `k`.
pub fn bhc_merge(stats: &LinkCommunityStats, gamma: f64) -> Vec<BhcMerge> {
    bhc_merge_core(
        BhcInput {
            k: stats.k,
            m: stats.m,
            gene_sum: &stats.gene_sum,
            size_sum: &stats.size_sum,
            effective_size: &stats.edge_count,
        },
        gamma,
    )
}
