#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! Link community model: sufficient statistics and collapsed Poisson-Gamma scoring.
//!
//! Each edge e has a profile vector y_e ∈ R^M (projected gene counts) and a
//! community assignment z_e ∈ {0..K-1}. The model is:
//!
//!   y_e^g | z_e = k ~ Poisson(s_e · μ_{g,k})
//!   μ_{g,k} ~ Gamma(a0, b0)     (collapsed out)
//!
//! where s_e is a size factor for edge e.

use special::Gamma as SpecialGamma;

/// Projected link profiles stored in row-major f32.
pub struct LinkProfileStore {
    /// Row-major [n_edges × m] gene profiles.
    pub profiles: Vec<f32>,
    /// Size factor per edge: Σ_g profile[e,g].
    pub size_factors: Vec<f32>,
    /// Number of edges.
    pub n_edges: usize,
    /// Projection dimension.
    pub m: usize,
}

impl LinkProfileStore {
    /// Create from a flat row-major buffer.
    pub fn new(profiles: Vec<f32>, n_edges: usize, m: usize) -> Self {
        debug_assert_eq!(profiles.len(), n_edges * m);
        let size_factors: Vec<f32> = (0..n_edges)
            .map(|e| {
                let row = &profiles[e * m..(e + 1) * m];
                row.iter().sum()
            })
            .collect();
        LinkProfileStore {
            profiles,
            size_factors,
            n_edges,
            m,
        }
    }

    /// Get the profile slice for edge `e`.
    #[inline]
    pub fn profile(&self, e: usize) -> &[f32] {
        &self.profiles[e * self.m..(e + 1) * self.m]
    }
}

/// Sufficient statistics for the link community model.
///
/// All accumulators are f64 to prevent drift during incremental updates.
pub struct LinkCommunityStats {
    /// Number of communities.
    pub k: usize,
    /// Projection dimension.
    pub m: usize,
    /// Number of edges.
    pub n_edges: usize,
    /// Per-community per-gene sum: gene_sum[k * m + g] = Σ_{e: z_e=k} y_e^g.
    pub gene_sum: Vec<f64>,
    /// Per-community size factor sum: size_sum[k] = Σ_{e: z_e=k} s_e.
    pub size_sum: Vec<f64>,
    /// Per-community edge count.
    pub edge_count: Vec<usize>,
    /// Current community assignment for each edge.
    pub membership: Vec<usize>,
}

impl LinkCommunityStats {
    /// Build sufficient statistics from profiles and initial labels.
    pub fn from_profiles(profiles: &LinkProfileStore, k: usize, labels: &[usize]) -> Self {
        let m = profiles.m;
        let n_edges = profiles.n_edges;
        debug_assert_eq!(labels.len(), n_edges);

        let mut gene_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut edge_count = vec![0usize; k];

        for e in 0..n_edges {
            let c = labels[e];
            debug_assert!(c < k);
            let row = profiles.profile(e);
            let base = c * m;
            for g in 0..m {
                gene_sum[base + g] += row[g] as f64;
            }
            size_sum[c] += profiles.size_factors[e] as f64;
            edge_count[c] += 1;
        }

        LinkCommunityStats {
            k,
            m,
            n_edges,
            gene_sum,
            size_sum,
            edge_count,
            membership: labels.to_vec(),
        }
    }

    /// Move edge `e` from `old_k` to `new_k`, updating stats incrementally. O(M).
    #[inline]
    pub fn delta_move(
        &mut self,
        e: usize,
        old_k: usize,
        new_k: usize,
        profiles: &LinkProfileStore,
    ) {
        debug_assert_eq!(self.membership[e], old_k);
        let m = self.m;
        let row = profiles.profile(e);
        let sf = profiles.size_factors[e] as f64;

        let old_base = old_k * m;
        let new_base = new_k * m;
        for g in 0..m {
            let v = row[g] as f64;
            self.gene_sum[old_base + g] -= v;
            self.gene_sum[new_base + g] += v;
        }

        self.size_sum[old_k] -= sf;
        self.size_sum[new_k] += sf;
        self.edge_count[old_k] -= 1;
        self.edge_count[new_k] += 1;
        self.membership[e] = new_k;
    }

    /// Recompute all statistics from scratch (drift correction).
    pub fn recompute(&mut self, profiles: &LinkProfileStore) {
        let m = self.m;
        let k = self.k;
        self.gene_sum.iter_mut().for_each(|x| *x = 0.0);
        self.size_sum.iter_mut().for_each(|x| *x = 0.0);
        self.edge_count.iter_mut().for_each(|x| *x = 0);

        for e in 0..self.n_edges {
            let c = self.membership[e];
            debug_assert!(c < k);
            let row = profiles.profile(e);
            let base = c * m;
            for g in 0..m {
                self.gene_sum[base + g] += row[g] as f64;
            }
            self.size_sum[c] += profiles.size_factors[e] as f64;
            self.edge_count[c] += 1;
        }
    }

    /// Total collapsed Poisson-Gamma score across all communities and genes.
    pub fn total_score(&self, a0: f64, b0: f64) -> f64 {
        let m = self.m;
        let mut score = 0.0f64;
        for c in 0..self.k {
            let t_k = self.size_sum[c];
            let base = c * m;
            for g in 0..m {
                let e_kg = self.gene_sum[base + g];
                score += poisson_score(a0, b0, e_kg, t_k);
            }
        }
        score
    }
}

/// Poisson-Gamma conjugate score (collapsed marginal log-likelihood for one cell).
///
/// score = a0 * ln(b0) + lgamma(a0 + edge) - lgamma(a0) - (a0 + edge) * ln(b0 + total)
#[inline]
pub fn poisson_score(a0: f64, b0: f64, edge: f64, total: f64) -> f64 {
    a0 * b0.ln() + SpecialGamma::ln_gamma(a0 + edge).0
        - SpecialGamma::ln_gamma(a0).0
        - (a0 + edge) * (b0 + total).ln()
}

/// Compute log-probabilities for assigning edge `e` to each community.
///
/// For each target community t, computes delta score = score_after - score_before
/// if edge e were moved from its current community to t.
/// The current community gets delta = 0.0 (baseline).
///
/// Complexity: O(K × M).
pub fn compute_log_probs_for_edge(
    e: usize,
    stats: &LinkCommunityStats,
    profiles: &LinkProfileStore,
    a0: f64,
    b0: f64,
    log_probs: &mut [f64],
) {
    let k = stats.k;
    let m = stats.m;
    let current_c = stats.membership[e];
    let row = profiles.profile(e);
    let sf = profiles.size_factors[e] as f64;

    // Current community stats (before removal)
    let old_size = stats.size_sum[current_c];
    let new_size_removed = old_size - sf;

    for t in 0..k {
        if t == current_c {
            log_probs[t] = 0.0;
            continue;
        }

        let target_size = stats.size_sum[t];
        let new_target_size = target_size + sf;

        let mut delta = 0.0f64;
        let src_base = current_c * m;
        let tgt_base = t * m;

        for g in 0..m {
            let y = row[g] as f64;

            // Score change for source community (edge removed)
            let old_e_src = stats.gene_sum[src_base + g];
            let new_e_src = old_e_src - y;
            delta += poisson_score(a0, b0, new_e_src, new_size_removed)
                - poisson_score(a0, b0, old_e_src, old_size);

            // Score change for target community (edge added)
            let old_e_tgt = stats.gene_sum[tgt_base + g];
            let new_e_tgt = old_e_tgt + y;
            delta += poisson_score(a0, b0, new_e_tgt, new_target_size)
                - poisson_score(a0, b0, old_e_tgt, target_size);
        }

        log_probs[t] = delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_profiles(
        n_edges: usize,
        m: usize,
        k: usize,
    ) -> (LinkProfileStore, Vec<usize>) {
        let mut profiles = vec![0.0f32; n_edges * m];
        let mut labels = vec![0usize; n_edges];

        // Create planted structure: edges in community c have higher values in gene c
        for e in 0..n_edges {
            let c = e % k;
            labels[e] = c;
            for g in 0..m {
                let base_val = 1.0;
                let signal = if g % k == c { 5.0 } else { 0.0 };
                profiles[e * m + g] = base_val + signal;
            }
        }

        (LinkProfileStore::new(profiles, n_edges, m), labels)
    }

    #[test]
    fn test_from_profiles_basic() {
        let (store, labels) = make_synthetic_profiles(20, 6, 3);
        let stats = LinkCommunityStats::from_profiles(&store, 3, &labels);

        assert_eq!(stats.k, 3);
        assert_eq!(stats.m, 6);
        assert_eq!(stats.n_edges, 20);

        // Each community should have roughly n_edges/k edges
        for c in 0..3 {
            assert!(stats.edge_count[c] > 0);
        }
        let total_count: usize = stats.edge_count.iter().sum();
        assert_eq!(total_count, 20);
    }

    #[test]
    fn test_delta_move_consistency() {
        let (store, labels) = make_synthetic_profiles(30, 4, 3);
        let mut stats = LinkCommunityStats::from_profiles(&store, 3, &labels);

        // Move edge 0 from community 0 to community 1
        let old_c = stats.membership[0];
        let new_c = (old_c + 1) % 3;
        stats.delta_move(0, old_c, new_c, &store);

        // Recompute from scratch and compare
        let stats_recomputed = LinkCommunityStats::from_profiles(&store, 3, &stats.membership);
        let score_delta = stats.total_score(1.0, 1.0);
        let score_recomputed = stats_recomputed.total_score(1.0, 1.0);

        assert!(
            (score_delta - score_recomputed).abs() < 1e-8,
            "Incremental vs recomputed: {} vs {}",
            score_delta,
            score_recomputed
        );
    }

    #[test]
    fn test_recompute_matches() {
        let (store, labels) = make_synthetic_profiles(20, 4, 2);
        let mut stats = LinkCommunityStats::from_profiles(&store, 2, &labels);

        // Do a few moves
        stats.delta_move(0, 0, 1, &store);
        stats.delta_move(3, 1, 0, &store);
        stats.delta_move(5, 1, 0, &store);

        let score_before = stats.total_score(1.0, 1.0);
        stats.recompute(&store);
        let score_after = stats.total_score(1.0, 1.0);

        assert!(
            (score_before - score_after).abs() < 1e-8,
            "Recompute drift: {} vs {}",
            score_before,
            score_after
        );
    }

    #[test]
    fn test_delta_score_matches_brute_force() {
        let (store, labels) = make_synthetic_profiles(15, 4, 3);
        let stats = LinkCommunityStats::from_profiles(&store, 3, &labels);

        let a0 = 1.0;
        let b0 = 1.0;
        let score_before = stats.total_score(a0, b0);

        let mut log_probs = vec![0.0f64; 3];

        for e in 0..stats.n_edges {
            let current_c = stats.membership[e];

            compute_log_probs_for_edge(e, &stats, &store, a0, b0, &mut log_probs);

            // Current community should have delta = 0
            assert!(
                log_probs[current_c].abs() < 1e-10,
                "e={}: log_prob for current cluster should be 0, got {}",
                e,
                log_probs[current_c]
            );

            // Check other communities by brute force
            for t in 0..3 {
                if t == current_c {
                    continue;
                }

                let mut stats_moved =
                    LinkCommunityStats::from_profiles(&store, 3, &stats.membership);
                stats_moved.delta_move(e, current_c, t, &store);
                let score_after = stats_moved.total_score(a0, b0);
                let expected_delta = score_after - score_before;

                assert!(
                    (log_probs[t] - expected_delta).abs() < 1e-8,
                    "e={}, t={}: computed={:.10}, expected={:.10}",
                    e,
                    t,
                    log_probs[t],
                    expected_delta
                );
            }
        }
    }

    #[test]
    fn test_poisson_score_basic() {
        let score = poisson_score(1.0, 1.0, 0.0, 0.0);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_score_with_data() {
        let score = poisson_score(1.0, 1.0, 5.0, 10.0);
        let expected = 0.0 + 120.0_f64.ln() - 0.0 - 6.0 * 11.0_f64.ln();
        assert!(
            (score - expected).abs() < 1e-10,
            "score={}, expected={}",
            score,
            expected
        );
    }

    #[test]
    fn test_size_factors() {
        let profiles = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let store = LinkProfileStore::new(profiles, 2, 3);
        assert_eq!(store.size_factors[0], 6.0);
        assert_eq!(store.size_factors[1], 15.0);
    }
}
