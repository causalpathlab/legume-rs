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

    /// Extract a sub-store for the given edge indices.
    pub fn subset(&self, edge_indices: &[usize]) -> Self {
        let n = edge_indices.len();
        let m = self.m;
        let mut profiles = vec![0.0f32; n * m];
        for (local_e, &global_e) in edge_indices.iter().enumerate() {
            profiles[local_e * m..(local_e + 1) * m].copy_from_slice(self.profile(global_e));
        }
        LinkProfileStore::new(profiles, n, m)
    }

    /// Collapse columns by module assignment.
    ///
    /// Each column `p` is mapped to `assignments[p]`, and values are summed
    /// within each module. Returns a new store with `n_modules` columns.
    pub fn collapse_modules(&self, assignments: &[usize]) -> Self {
        let n_modules = assignments.iter().copied().max().unwrap_or(0) + 1;
        let n = self.n_edges;
        let mut new_profiles = vec![0.0f32; n * n_modules];
        for e in 0..n {
            let row = self.profile(e);
            let base = e * n_modules;
            for (p, &module) in assignments.iter().enumerate() {
                new_profiles[base + module] += row[p];
            }
        }
        LinkProfileStore::new(new_profiles, n, n_modules)
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

        for (e, &c) in labels.iter().enumerate().take(n_edges) {
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

        let old_slice = &mut self.gene_sum[old_k * m..(old_k + 1) * m];
        for (dst, &v) in old_slice.iter_mut().zip(row.iter()) {
            *dst -= v as f64;
        }
        let new_slice = &mut self.gene_sum[new_k * m..(new_k + 1) * m];
        for (dst, &v) in new_slice.iter_mut().zip(row.iter()) {
            *dst += v as f64;
        }

        self.size_sum[old_k] -= sf;
        self.size_sum[new_k] += sf;
        self.edge_count[old_k] -= 1;
        self.edge_count[new_k] += 1;
        self.membership[e] = new_k;
    }

    /// Build sufficient statistics for a subset of edges (no membership vector).
    ///
    /// Returns only the aggregate stats (gene_sum, size_sum, edge_count) for the
    /// Compute aggregate stats for a subset of edges (identified by global indices).
    pub fn component_stats(
        profiles: &LinkProfileStore,
        k: usize,
        edge_indices: &[usize],
        membership: &[usize],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let m = profiles.m;
        let mut gene_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut edge_count = vec![0usize; k];

        for &e in edge_indices {
            let c = membership[e];
            debug_assert!(c < k);
            let row = profiles.profile(e);
            let base = c * m;
            for g in 0..m {
                gene_sum[base + g] += row[g] as f64;
            }
            size_sum[c] += profiles.size_factors[e] as f64;
            edge_count[c] += 1;
        }

        (gene_sum, size_sum, edge_count)
    }

    /// Compute aggregate stats for a contiguous sub-store with local membership.
    pub fn local_stats(
        profiles: &LinkProfileStore,
        k: usize,
        membership: &[usize],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let m = profiles.m;
        let mut gene_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut edge_count = vec![0usize; k];

        for (e, &c) in membership.iter().enumerate() {
            debug_assert!(c < k);
            let row = profiles.profile(e);
            let base = c * m;
            for g in 0..m {
                gene_sum[base + g] += row[g] as f64;
            }
            size_sum[c] += profiles.size_factors[e] as f64;
            edge_count[c] += 1;
        }

        (gene_sum, size_sum, edge_count)
    }

    /// Apply a delta to the sufficient statistics: self += (new - old).
    ///
    /// Used in the memoized EM: after a component updates its local stats,
    /// the global stats are patched with the difference.
    pub fn apply_delta(
        &mut self,
        old: (&[f64], &[f64], &[usize]),
        new: (&[f64], &[f64], &[usize]),
    ) {
        let km = self.k * self.m;
        for i in 0..km {
            self.gene_sum[i] += new.0[i] - old.0[i];
        }
        for c in 0..self.k {
            self.size_sum[c] += new.1[c] - old.1[c];
            // edge_count delta: new - old (both usize, but difference can be negative)
            self.edge_count[c] =
                (self.edge_count[c] as isize + new.2[c] as isize - old.2[c] as isize) as usize;
        }
    }

    /// Recompute all statistics from scratch (drift correction).
    #[allow(dead_code)]
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
            let slice = &mut self.gene_sum[c * m..(c + 1) * m];
            for (dst, &v) in slice.iter_mut().zip(row.iter()) {
                *dst += v as f64;
            }
            self.size_sum[c] += profiles.size_factors[e] as f64;
            self.edge_count[c] += 1;
        }
    }

    /// Compute Dirichlet-smoothed log mixing weights from edge counts.
    ///
    /// `log_weights[k] = digamma(edge_count[k] + alpha/K) - digamma(n_edges + alpha)`
    pub fn compute_log_weights(&self, alpha: f64) -> Vec<f64> {
        let alpha_k = alpha / self.k as f64;
        let total = self.n_edges as f64 + alpha;
        let dg_total = SpecialGamma::digamma(total);
        (0..self.k)
            .map(|c| SpecialGamma::digamma(self.edge_count[c] as f64 + alpha_k) - dg_total)
            .collect()
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

#[cfg(test)]
/// Amortized classifier for link community assignments.
///
/// Extracts posterior log-rates from converged sufficient statistics and
/// predicts community assignments for new edge profiles via matrix multiply.
///
/// The collapsed Poisson-Gamma posterior gives rates:
///   μ_{g,k} = (a0 + gene_sum[k,g]) / (b0 + size_sum[k])
///
/// The predictive log-probability for assigning edge e to community k is:
///   log P(z_e = k | y_e) ≈ Σ_g y_e^g · log(μ_{g,k}) - s_e · Σ_g μ_{g,k} + log(n_k)
///
/// This is a linear classifier in the profile space, trained analytically
/// from the Gibbs posterior — no gradient descent needed.
pub struct LinkCommunityClassifier {
    /// log(μ_{g,k}): [k × m] row-major.
    pub log_rates: Vec<f64>,
    /// Σ_g μ_{g,k} per community (rate totals for size-factor correction).
    pub rate_totals: Vec<f64>,
    /// log(edge_count[k]) per community (prior from empirical frequencies).
    pub log_prior: Vec<f64>,
    pub k: usize,
    pub m: usize,
}

#[cfg(test)]
impl LinkCommunityClassifier {
    /// Extract a classifier from converged sufficient statistics.
    pub fn from_stats(stats: &LinkCommunityStats, a0: f64, b0: f64) -> Self {
        let k = stats.k;
        let m = stats.m;
        let mut log_rates = vec![0.0f64; k * m];
        let mut rate_totals = vec![0.0f64; k];

        for (c, rt) in rate_totals.iter_mut().enumerate().take(k) {
            let denom = b0 + stats.size_sum[c];
            let gene_slice = &stats.gene_sum[c * m..(c + 1) * m];
            let lr_slice = &mut log_rates[c * m..(c + 1) * m];
            let mut rate_sum = 0.0f64;
            for (lr, &gs) in lr_slice.iter_mut().zip(gene_slice.iter()) {
                let mu = (a0 + gs) / denom;
                *lr = mu.ln();
                rate_sum += mu;
            }
            *rt = rate_sum;
        }

        // Empirical prior: log(edge_count + 1) to avoid -inf for empty communities
        let log_prior: Vec<f64> = stats
            .edge_count
            .iter()
            .map(|&n| ((n as f64) + 1.0).ln())
            .collect();

        LinkCommunityClassifier {
            log_rates,
            rate_totals,
            log_prior,
            k,
            m,
        }
    }

    /// Predict community assignment for a single edge profile.
    ///
    /// Returns the argmax community index.
    #[cfg(test)]
    #[inline]
    pub fn predict_one(&self, profile: &[f32], size_factor: f32) -> usize {
        let sf = size_factor as f64;
        let mut best_k = 0;
        let mut best_score = f64::NEG_INFINITY;
        for c in 0..self.k {
            let base = c * self.m;
            let mut score = self.log_prior[c] - sf * self.rate_totals[c];
            for g in 0..self.m {
                score += profile[g] as f64 * self.log_rates[base + g];
            }
            if score > best_score {
                best_score = score;
                best_k = c;
            }
        }
        best_k
    }

    /// Predict community assignments for all edges in a profile store.
    ///
    /// Returns a label vector of length `profiles.n_edges`.
    #[cfg(test)]
    pub fn predict_labels(&self, profiles: &LinkProfileStore) -> Vec<usize> {
        (0..profiles.n_edges)
            .map(|e| self.predict_one(profiles.profile(e), profiles.size_factors[e]))
            .collect()
    }

    /// Predict community assignments in parallel using rayon.
    #[cfg(test)]
    pub fn predict_labels_parallel(&self, profiles: &LinkProfileStore) -> Vec<usize> {
        use rayon::prelude::*;
        let chunk_size = std::cmp::max(256, profiles.n_edges / rayon::current_num_threads().max(1));
        let indices: Vec<usize> = (0..profiles.n_edges).collect();
        indices
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|&e| self.predict_one(profiles.profile(e), profiles.size_factors[e]))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Precomputed Poisson-Gamma hyperparameter constants for hot-loop scoring.
struct PoissonScoreParams {
    a0: f64,
    b0: f64,
    /// a0 * ln(b0) - lgamma(a0), constant across all calls
    base_term: f64,
}

impl PoissonScoreParams {
    fn new(a0: f64, b0: f64) -> Self {
        Self {
            a0,
            b0,
            base_term: a0 * b0.ln() - SpecialGamma::ln_gamma(a0).0,
        }
    }

    #[inline]
    fn score(&self, edge: f64, total: f64) -> f64 {
        self.base_term + SpecialGamma::ln_gamma(self.a0 + edge).0
            - (self.a0 + edge) * (self.b0 + total).ln()
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
/// When `log_weights` is provided, adds `log_weights[t] - log_weights[current_c]`
/// as a Dirichlet prior contribution to each candidate.
///
/// Complexity: O(K × M).
pub fn compute_log_probs_for_edge(
    e: usize,
    stats: &LinkCommunityStats,
    profiles: &LinkProfileStore,
    a0: f64,
    b0: f64,
    log_weights: Option<&[f64]>,
    log_probs: &mut [f64],
) {
    let k = stats.k;
    let m = stats.m;
    let current_c = stats.membership[e];
    let row = profiles.profile(e);
    let sf = profiles.size_factors[e] as f64;

    let old_size = stats.size_sum[current_c];
    let new_size_removed = old_size - sf;

    let src_slice = &stats.gene_sum[current_c * m..(current_c + 1) * m];

    let ps = PoissonScoreParams::new(a0, b0);

    let lw_current = log_weights.map(|w| w[current_c]).unwrap_or(0.0);

    for (t, lp) in log_probs.iter_mut().enumerate().take(k) {
        if t == current_c {
            *lp = 0.0;
            continue;
        }

        let target_size = stats.size_sum[t];
        let new_target_size = target_size + sf;
        let tgt_slice = &stats.gene_sum[t * m..(t + 1) * m];

        let mut delta = 0.0f64;
        for ((&y_f32, &old_e_src), &old_e_tgt) in
            row.iter().zip(src_slice.iter()).zip(tgt_slice.iter())
        {
            let y = y_f32 as f64;
            delta += ps.score(old_e_src - y, new_size_removed) - ps.score(old_e_src, old_size);
            delta += ps.score(old_e_tgt + y, new_target_size) - ps.score(old_e_tgt, target_size);
        }

        // Dirichlet prior: log(π_t / π_current)
        if let Some(w) = log_weights {
            delta += w[t] - lw_current;
        }

        *lp = delta;
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

            compute_log_probs_for_edge(e, &stats, &store, a0, b0, None, &mut log_probs);

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

    #[test]
    fn test_classifier_recovers_planted() {
        // Create planted profiles: K=3 communities with distinct gene signatures
        let k = 3;
        let m = 6;
        let n_edges = 60;

        let (store, labels) = make_synthetic_profiles(n_edges, m, k);
        let stats = LinkCommunityStats::from_profiles(&store, k, &labels);

        let classifier = LinkCommunityClassifier::from_stats(&stats, 1.0, 1.0);
        let predicted = classifier.predict_labels(&store);

        // Classifier should recover the planted labels exactly
        // (since it's trained on the same data)
        assert_eq!(
            predicted, labels,
            "Classifier should recover planted labels exactly"
        );
    }

    #[test]
    fn test_classifier_generalizes() {
        // Train classifier on one set of edges, test on another
        let k = 2;
        let m = 8;
        let n_train = 100;
        let n_test = 50;

        // Training: planted profiles
        let mut train_profiles = vec![0.0f32; n_train * m];
        let mut train_labels = vec![0usize; n_train];
        for e in 0..n_train {
            let c = e % k;
            train_labels[e] = c;
            for g in 0..m {
                let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
                train_profiles[e * m + g] = signal;
            }
        }
        let train_store = LinkProfileStore::new(train_profiles, n_train, m);
        let stats = LinkCommunityStats::from_profiles(&train_store, k, &train_labels);

        let classifier = LinkCommunityClassifier::from_stats(&stats, 1.0, 1.0);

        // Test: similar profiles with noise
        let mut test_profiles = vec![0.0f32; n_test * m];
        let mut test_labels = vec![0usize; n_test];
        for e in 0..n_test {
            let c = e % k;
            test_labels[e] = c;
            for g in 0..m {
                let signal = if (g < m / 2) == (c == 0) { 8.0 } else { 2.0 };
                test_profiles[e * m + g] = signal;
            }
        }
        let test_store = LinkProfileStore::new(test_profiles, n_test, m);

        let predicted = classifier.predict_labels(&test_store);
        let correct: usize = predicted
            .iter()
            .zip(test_labels.iter())
            .filter(|(&p, &t)| p == t)
            .count();

        assert!(
            correct == n_test,
            "Classifier should generalize to noisy test data: {}/{}",
            correct,
            n_test
        );
    }

    #[test]
    fn test_classifier_parallel_matches_sequential() {
        let k = 3;
        let m = 6;
        let n_edges = 120;

        let (store, labels) = make_synthetic_profiles(n_edges, m, k);
        let stats = LinkCommunityStats::from_profiles(&store, k, &labels);
        let classifier = LinkCommunityClassifier::from_stats(&stats, 1.0, 1.0);

        let seq = classifier.predict_labels(&store);
        let par = classifier.predict_labels_parallel(&store);

        assert_eq!(seq, par, "Parallel predictions should match sequential");
    }
}
