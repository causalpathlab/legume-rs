//! Link community model: sufficient statistics and fast Poisson DC-SBM scoring.
//!
//! Each edge e has a profile vector y_e ∈ R^M (projected gene counts) and a
//! community assignment z_e ∈ {0..K-1}. Assignment score under Poisson DC-SBM
//! with MLE plug-in rates μ_{kg} = (T_{kg} + ε) / (S_k + M·ε):
//!
//!   score(e, k) = Σ_g y_{eg} · ln μ_{kg} = Σ_g y_{eg} · log_rate[k, g]
//!
//! where log_rate[k, g] is factored into per-gene and per-community-size parts
//! for cheap incremental updates:
//!
//!   log_rate[k, g] = log_gene[k, g] + log_size_offset[k]
//!
//! with log_gene[k, g] = ln(T_{kg} + ε), log_size_offset[k] = -ln(S_k + M·ε).
//!
//! Move delta (edge e from src to tgt) is the conditional log-likelihood
//! difference:
//!
//!   Δ = Σ_g y_{eg} · (log_rate[tgt, g] - log_rate[src, g])
//!
//! Hot scoring path uses zero `ln` calls (only lookups + multiplies). Each
//! move recomputes O(nnz(y_e)) log_gene entries per side and two
//! log_size_offset entries — much cheaper than a full row rebuild.
//!
//! The reporting-only `total_score()` computes the conditional-entropy objective
//! J(z) = Σ f(T) - Σ f(S) on demand.

use special::Gamma as SpecialGamma;

const LOG_EPS: f64 = 1e-9;

/// f(x) = x · ln(x), with f(0) = 0 by convention.
#[inline]
pub(crate) fn f_entropy(x: f64) -> f64 {
    if x > 0.0 {
        x * x.ln()
    } else {
        0.0
    }
}

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

    /// Empirical column marginal normalised to a probability distribution (sums to 1).
    ///
    /// Uses f64 accumulation for numerical stability.
    pub(crate) fn empirical_marginal(&self) -> Vec<f64> {
        let mut col_sum = vec![0.0f64; self.m];
        for e in 0..self.n_edges {
            let row = self.profile(e);
            for g in 0..self.m {
                col_sum[g] += row[g] as f64;
            }
        }
        let total: f64 = col_sum.iter().sum::<f64>().max(1.0);
        for v in col_sum.iter_mut() {
            *v /= total;
        }
        col_sum
    }

    /// Apply inverse-frequency weighting in place (degree correction ~ housekeeping).
    ///
    /// Multiplies every profile entry by `w_g = -log(bg[g] + ε)`. Housekeeping
    /// genes (high bg) get small weight; specific genes (low bg) get large
    /// weight. Size factors are recomputed. This is the DC-SBM degree
    /// correction with θ_g = bg[g]: after reweighting, the objective measures
    /// community specificity relative to background.
    pub(crate) fn weight_by_idf(&mut self, bg: &[f64]) {
        debug_assert_eq!(bg.len(), self.m);
        let eps = 1e-12f64;
        let w: Vec<f32> = bg.iter().map(|&p| (-((p + eps).ln())) as f32).collect();
        for e in 0..self.n_edges {
            let row = &mut self.profiles[e * self.m..(e + 1) * self.m];
            let mut s = 0.0f32;
            for (y, &wg) in row.iter_mut().zip(w.iter()) {
                *y *= wg;
                s += *y;
            }
            self.size_factors[e] = s;
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
    /// Cached `ln(gene_sum[k*m + g] + ε)`, kept in sync with `gene_sum`.
    pub(crate) log_gene: Vec<f64>,
    /// Cached `-ln(size_sum[k] + M·ε)`, kept in sync with `size_sum`.
    /// Added to `log_gene[k, g]` this gives `ln(rate_kg)`.
    pub(crate) log_size_offset: Vec<f64>,
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

        let log_gene: Vec<f64> = gene_sum.iter().map(|&t| (t + LOG_EPS).ln()).collect();
        let m_eps = (m as f64) * LOG_EPS;
        let log_size_offset: Vec<f64> = size_sum.iter().map(|&s| -((s + m_eps).ln())).collect();

        LinkCommunityStats {
            k,
            m,
            n_edges,
            gene_sum,
            size_sum,
            edge_count,
            membership: labels.to_vec(),
            log_gene,
            log_size_offset,
        }
    }

    /// Move edge `e` from `old_k` to `new_k`, updating stats incrementally.
    ///
    /// Only the log_gene entries for non-zero profile genes are recomputed
    /// (O(nnz(y_e)) per side). log_size_offset is recomputed for both
    /// affected communities (2 `ln` calls total).
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
        let m_eps = (m as f64) * LOG_EPS;

        let old_base = old_k * m;
        for (g, &y) in row.iter().enumerate() {
            let v = y as f64;
            if v == 0.0 {
                continue;
            }
            let idx = old_base + g;
            self.gene_sum[idx] -= v;
            self.log_gene[idx] = (self.gene_sum[idx] + LOG_EPS).ln();
        }
        let new_base = new_k * m;
        for (g, &y) in row.iter().enumerate() {
            let v = y as f64;
            if v == 0.0 {
                continue;
            }
            let idx = new_base + g;
            self.gene_sum[idx] += v;
            self.log_gene[idx] = (self.gene_sum[idx] + LOG_EPS).ln();
        }

        self.size_sum[old_k] -= sf;
        self.log_size_offset[old_k] = -((self.size_sum[old_k] + m_eps).ln());
        self.size_sum[new_k] += sf;
        self.log_size_offset[new_k] = -((self.size_sum[new_k] + m_eps).ln());
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
            let delta = new.0[i] - old.0[i];
            if delta != 0.0 {
                self.gene_sum[i] += delta;
                self.log_gene[i] = (self.gene_sum[i] + LOG_EPS).ln();
            }
        }
        let m_eps = (self.m as f64) * LOG_EPS;
        for c in 0..self.k {
            let delta = new.1[c] - old.1[c];
            if delta != 0.0 {
                self.size_sum[c] += delta;
                self.log_size_offset[c] = -((self.size_sum[c] + m_eps).ln());
            }
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
            self.edge_count[c] += 1;
            let row = profiles.profile(e);
            let slice = &mut self.gene_sum[c * m..(c + 1) * m];
            for (dst, &v) in slice.iter_mut().zip(row.iter()) {
                *dst += v as f64;
            }
            self.size_sum[c] += profiles.size_factors[e] as f64;
        }

        for (lg, &gs) in self.log_gene.iter_mut().zip(self.gene_sum.iter()) {
            *lg = (gs + LOG_EPS).ln();
        }
        let m_eps = (self.m as f64) * LOG_EPS;
        for (lso, &ss) in self.log_size_offset.iter_mut().zip(self.size_sum.iter()) {
            *lso = -((ss + m_eps).ln());
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

    /// Total conditional-entropy / plug-in multinomial score (higher = better).
    ///
    /// J = Σ_{k,g} f(T_kg) − Σ_k f(S_k), f(x) = x · ln(x).
    /// Computed on demand from `gene_sum` / `size_sum` (not a hot-path call).
    pub fn total_score(&self) -> f64 {
        let mut sum_f_t = 0.0f64;
        for &t in &self.gene_sum {
            sum_f_t += f_entropy(t);
        }
        let mut sum_f_s = 0.0f64;
        for &s in &self.size_sum {
            sum_f_s += f_entropy(s);
        }
        sum_f_t - sum_f_s
    }
}

#[cfg(test)]
/// Amortized classifier for link community assignments (multinomial naive Bayes).
///
/// From converged sufficient stats, π_{kg} = T_{kg} / S_k. For a new edge profile y_e:
///   log P(z_e = k | y_e) ∝ log_prior[k] + Σ_g y_{eg} · log π_{kg}
///
/// Additive ε smoothing keeps log_rates finite for empty communities or zero genes.
pub struct LinkCommunityClassifier {
    /// log(π_{g,k}) with ε smoothing: [k × m] row-major.
    pub log_rates: Vec<f64>,
    /// log(edge_count[k] + 1) per community (empirical prior).
    pub log_prior: Vec<f64>,
    pub k: usize,
    pub m: usize,
}

#[cfg(test)]
impl LinkCommunityClassifier {
    /// Extract a classifier from converged sufficient statistics.
    pub fn from_stats(stats: &LinkCommunityStats) -> Self {
        let k = stats.k;
        let m = stats.m;
        // log_rate[k, g] = log_gene[k, g] + log_size_offset[k]
        let mut log_rates = vec![0.0f64; k * m];
        for c in 0..k {
            let off = stats.log_size_offset[c];
            let base = c * m;
            for g in 0..m {
                log_rates[base + g] = stats.log_gene[base + g] + off;
            }
        }

        // Empirical prior: log(edge_count + 1) to avoid -inf for empty communities
        let log_prior: Vec<f64> = stats
            .edge_count
            .iter()
            .map(|&n| ((n as f64) + 1.0).ln())
            .collect();

        LinkCommunityClassifier {
            log_rates,
            log_prior,
            k,
            m,
        }
    }

    /// Predict community assignment for a single edge profile.
    #[cfg(test)]
    #[inline]
    pub fn predict_one(&self, profile: &[f32]) -> usize {
        let mut best_k = 0;
        let mut best_score = f64::NEG_INFINITY;
        for c in 0..self.k {
            let base = c * self.m;
            let mut score = self.log_prior[c];
            #[allow(clippy::needless_range_loop)]
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
    #[cfg(test)]
    pub fn predict_labels(&self, profiles: &LinkProfileStore) -> Vec<usize> {
        (0..profiles.n_edges)
            .map(|e| self.predict_one(profiles.profile(e)))
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
                    .map(|&e| self.predict_one(profiles.profile(e)))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Compute log-probabilities for assigning edge `e` to each community.
///
/// Fast Poisson DC-SBM conditional scoring using cached `log_gene` /
/// `log_size_offset`:
///
///   score(e, k) = Σ_{g: y>0} y_eg · log_gene[k, g]  +  s_e · log_size_offset[k]
///
/// `log_probs[t] = score(e, t) - score(e, current_c)`, with the current-community
/// slot set to 0 so categorical sampling / argmax operate on the deltas.
///
/// Zero `ln` calls in the hot loop — only lookups, multiplies, and adds.
/// Complexity per edge: O(K · nnz(y_e)).
pub(crate) fn compute_log_probs_for_edge(
    e: usize,
    stats: &LinkCommunityStats,
    profiles: &LinkProfileStore,
    log_weights: Option<&[f64]>,
    log_probs: &mut [f64],
) {
    let k = stats.k;
    let m = stats.m;
    let current_c = stats.membership[e];
    let row = profiles.profile(e);
    let sf = profiles.size_factors[e] as f64;
    let lw_current = log_weights.map(|w| w[current_c]).unwrap_or(0.0);

    // Hoisted: score for the current community (computed once).
    let src_slice = &stats.log_gene[current_c * m..(current_c + 1) * m];
    let src_score = edge_score(row, src_slice, sf, stats.log_size_offset[current_c]);

    for (t, lp) in log_probs.iter_mut().enumerate().take(k) {
        if t == current_c {
            *lp = 0.0;
            continue;
        }
        let tgt_slice = &stats.log_gene[t * m..(t + 1) * m];
        let tgt_score = edge_score(row, tgt_slice, sf, stats.log_size_offset[t]);
        let mut delta = tgt_score - src_score;
        if let Some(w) = log_weights {
            delta += w[t] - lw_current;
        }
        *lp = delta;
    }
}

/// Inner dot product `Σ_{g: y>0} y · log_gene_slice[g] + sf · log_size_offset_k`.
#[inline]
fn edge_score(row: &[f32], log_gene_slice: &[f64], sf: f64, log_size_offset_k: f64) -> f64 {
    let mut s = sf * log_size_offset_k;
    for (&y_f32, &lg) in row.iter().zip(log_gene_slice.iter()) {
        let y = y_f32 as f64;
        if y != 0.0 {
            s += y * lg;
        }
    }
    s
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
        let score_delta = stats.total_score();
        let score_recomputed = stats_recomputed.total_score();

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

        let score_before = stats.total_score();
        stats.recompute(&store);
        let score_after = stats.total_score();

        assert!(
            (score_before - score_after).abs() < 1e-8,
            "Recompute drift: {} vs {}",
            score_before,
            score_after
        );
    }

    #[test]
    fn test_delta_score_matches_hand_computed_dot_product() {
        // Fast DC-SBM scoring: log_probs[t] must equal
        //   Σ_g y_eg · (log_rate[t,g] - log_rate[current_c,g])
        // where log_rate[k,g] = log_gene[k,g] + log_size_offset[k].
        let (store, labels) = make_synthetic_profiles(15, 4, 3);
        let stats = LinkCommunityStats::from_profiles(&store, 3, &labels);
        let mut log_probs = vec![0.0f64; 3];

        for e in 0..stats.n_edges {
            let current_c = stats.membership[e];
            let row = store.profile(e);
            let sf = store.size_factors[e] as f64;
            compute_log_probs_for_edge(e, &stats, &store, None, &mut log_probs);
            assert!(
                log_probs[current_c].abs() < 1e-10,
                "current community must score 0"
            );
            for (t, &got) in log_probs.iter().enumerate() {
                if t == current_c {
                    continue;
                }
                let mut expected =
                    sf * (stats.log_size_offset[t] - stats.log_size_offset[current_c]);
                for (g, &y_f32) in row.iter().enumerate() {
                    let y = y_f32 as f64;
                    if y == 0.0 {
                        continue;
                    }
                    expected += y
                        * (stats.log_gene[t * stats.m + g]
                            - stats.log_gene[current_c * stats.m + g]);
                }
                assert!(
                    (got - expected).abs() < 1e-10,
                    "e={e} t={t}: got={got:.10} expected={expected:.10}"
                );
            }
        }
    }

    #[test]
    fn test_delta_move_keeps_caches_in_sync() {
        // After any sequence of delta_moves, log_gene[k,g] must equal
        // ln(gene_sum[k,g] + ε) and log_size_offset[k] must equal
        // -ln(size_sum[k] + M·ε).
        let (store, labels) = make_synthetic_profiles(30, 5, 3);
        let mut stats = LinkCommunityStats::from_profiles(&store, 3, &labels);
        let m_eps = (stats.m as f64) * LOG_EPS;

        // Apply a handful of moves.
        let moves: &[(usize, usize)] = &[(0, 1), (3, 2), (7, 0), (11, 2), (14, 1), (2, 0)];
        for &(e, new_c) in moves {
            let old_c = stats.membership[e];
            if old_c != new_c {
                stats.delta_move(e, old_c, new_c, &store);
            }
        }

        // Verify every cache entry matches what from-scratch would give.
        for k in 0..stats.k {
            let off_expected = -((stats.size_sum[k] + m_eps).ln());
            assert!(
                (stats.log_size_offset[k] - off_expected).abs() < 1e-12,
                "log_size_offset[{k}] drift"
            );
            for g in 0..stats.m {
                let idx = k * stats.m + g;
                let lg_expected = (stats.gene_sum[idx] + LOG_EPS).ln();
                assert!(
                    (stats.log_gene[idx] - lg_expected).abs() < 1e-12,
                    "log_gene[{k},{g}] drift"
                );
            }
        }

        // Incremental caches must also agree with a fresh rebuild.
        let fresh = LinkCommunityStats::from_profiles(&store, 3, &stats.membership);
        for i in 0..stats.log_gene.len() {
            assert!((stats.log_gene[i] - fresh.log_gene[i]).abs() < 1e-12);
        }
        for k in 0..stats.k {
            assert!((stats.log_size_offset[k] - fresh.log_size_offset[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_idf_weighting_downweights_housekeeping() {
        // Two real communities distinguished by genes 0..m/2 vs m/2..m; one
        // housekeeping-ish gene (constant across all edges) should contribute
        // ~0 after IDF weighting.
        let m = 6;
        let k = 2;
        let n = 20;
        let mut profiles = vec![0.0f32; n * m];
        let mut labels = vec![0usize; n];
        for e in 0..n {
            let c = e % k;
            labels[e] = c;
            for g in 0..m {
                profiles[e * m + g] = if g == m - 1 {
                    100.0 // housekeeping: huge in every edge
                } else if (g < m / 2) == (c == 0) {
                    5.0
                } else {
                    0.5
                };
            }
        }
        let mut store = LinkProfileStore::new(profiles, n, m);

        // Before-weighting totals
        let hk_before: f32 = (0..n).map(|e| store.profile(e)[m - 1]).sum();
        let sp_before: f32 = (0..n).map(|e| store.profile(e)[0]).sum();
        let ratio_before = hk_before / sp_before;

        let mut col_sum = vec![0.0f64; m];
        for e in 0..n {
            for (g, &v) in store.profile(e).iter().enumerate() {
                col_sum[g] += v as f64;
            }
        }
        let total: f64 = col_sum.iter().sum::<f64>();
        col_sum.iter_mut().for_each(|v| *v /= total);

        store.weight_by_idf(&col_sum);

        // After IDF: the housekeeping / specific ratio should be compressed.
        let hk_after: f32 = (0..n).map(|e| store.profile(e)[m - 1]).sum();
        let sp_after: f32 = (0..n).map(|e| store.profile(e)[0]).sum();
        let ratio_after = hk_after / sp_after;
        assert!(
            ratio_after < ratio_before * 0.2,
            "IDF should compress hk:specific ratio: before={:.2}, after={:.2}",
            ratio_before,
            ratio_after
        );

        // Greedy should still recover the planted labels using weighted counts.
        let mut stats = LinkCommunityStats::from_profiles(&store, k, &vec![0usize; n]);
        let mut log_probs = vec![0.0f64; k];
        for _ in 0..10 {
            let mut mv = 0;
            for e in 0..n {
                let cur = stats.membership[e];
                compute_log_probs_for_edge(e, &stats, &store, None, &mut log_probs);
                let mut best = cur;
                let mut best_v = 0.0f64;
                for (t, &v) in log_probs.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = t;
                    }
                }
                if best != cur {
                    stats.delta_move(e, cur, best, &store);
                    mv += 1;
                }
            }
            if mv == 0 {
                break;
            }
        }
        let match_direct: usize = (0..n).filter(|&e| stats.membership[e] == labels[e]).count();
        let match_flip: usize = (0..n)
            .filter(|&e| stats.membership[e] == 1 - labels[e])
            .count();
        assert!(
            match_direct.max(match_flip) >= n - 1,
            "IDF-weighted greedy should recover planted partition"
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

        let classifier = LinkCommunityClassifier::from_stats(&stats);
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

        let classifier = LinkCommunityClassifier::from_stats(&stats);

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
        let classifier = LinkCommunityClassifier::from_stats(&stats);

        let seq = classifier.predict_labels(&store);
        let par = classifier.predict_labels_parallel(&store);

        assert_eq!(seq, par, "Parallel predictions should match sequential");
    }
}
