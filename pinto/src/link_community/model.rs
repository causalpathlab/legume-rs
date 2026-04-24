//! Link community model: sufficient statistics and fast Poisson DC-SBM scoring.
//!
//! Each edge e has a profile vector y_e ∈ R^M (sparsely stored as non-zero
//! `(col, val)` pairs) and a community assignment z_e ∈ {0..K-1}. Assignment
//! score under Poisson DC-SBM with MLE plug-in rates
//! μ_{kg} = (T_{kg} + ε) / (S_k + M·ε):
//!
//!   score(e, k) = Σ_{g: y_eg > 0} y_{eg} · ln μ_{kg}
//!
//! where log_rate[k, g] = log_gene[k, g] + log_size_offset[k] is factored
//! into per-gene and per-community-size parts for cheap incremental updates.
//!
//! Profiles are stored in CSR: for each edge, `(indptr[e]..indptr[e+1])`
//! slices `indices` and `values` hold the non-zero entries. The hot Gibbs
//! loop iterates these pairs directly, avoiding zero-skip branches.

use special::Gamma as SpecialGamma;

pub(crate) const LOG_EPS: f64 = 1e-9;

/// f(x) = x · ln(x), with f(0) = 0 by convention.
#[inline]
pub(crate) fn f_entropy(x: f64) -> f64 {
    if x > 0.0 {
        x * x.ln()
    } else {
        0.0
    }
}

/// CSR-backed per-edge profile store.
///
/// Row `e` has non-zero column indices in
/// `indices[indptr[e]..indptr[e+1]]` and matching values in
/// `values[indptr[e]..indptr[e+1]]`. Indices within a row are sorted
/// ascending (deterministic iteration) and unique.
pub struct LinkProfileStore {
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    /// Size factor per edge: Σ of non-zero values in that row.
    pub size_factors: Vec<f32>,
    pub n_edges: usize,
    pub m: usize,
}

impl LinkProfileStore {
    /// Build from per-edge lists of `(column, value)` pairs. Duplicate
    /// columns in a row are summed; zero values are dropped.
    pub fn from_sparse_rows(mut rows: Vec<Vec<(u32, f32)>>, m: usize) -> Self {
        let n_edges = rows.len();
        let mut indptr = Vec::with_capacity(n_edges + 1);
        indptr.push(0usize);

        let mut nnz_est = 0usize;
        for row in &rows {
            nnz_est += row.len();
        }
        let mut indices: Vec<u32> = Vec::with_capacity(nnz_est);
        let mut values: Vec<f32> = Vec::with_capacity(nnz_est);
        let mut size_factors: Vec<f32> = Vec::with_capacity(n_edges);

        for row in rows.iter_mut() {
            row.sort_by_key(|&(c, _)| c);
            // Merge duplicates and drop zeros.
            let mut sf = 0.0f32;
            let mut i = 0usize;
            while i < row.len() {
                let col = row[i].0;
                let mut val = row[i].1;
                let mut j = i + 1;
                while j < row.len() && row[j].0 == col {
                    val += row[j].1;
                    j += 1;
                }
                if val != 0.0 {
                    indices.push(col);
                    values.push(val);
                    sf += val;
                }
                i = j;
            }
            size_factors.push(sf);
            indptr.push(indices.len());
        }

        LinkProfileStore {
            indptr,
            indices,
            values,
            size_factors,
            n_edges,
            m,
        }
    }

    /// Build from a flat row-major `[n_edges × m]` dense buffer. Non-zero
    /// entries are converted to CSR (projection mode + tests).
    pub fn new(dense: Vec<f32>, n_edges: usize, m: usize) -> Self {
        debug_assert_eq!(dense.len(), n_edges * m);
        let mut rows: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_edges);
        for e in 0..n_edges {
            let base = e * m;
            let mut row = Vec::new();
            for g in 0..m {
                let v = dense[base + g];
                if v != 0.0 {
                    row.push((g as u32, v));
                }
            }
            rows.push(row);
        }
        Self::from_sparse_rows(rows, m)
    }

    /// Sparse row slice for edge `e`: `(columns, values)`.
    #[inline]
    pub fn row(&self, e: usize) -> (&[u32], &[f32]) {
        let start = self.indptr[e];
        let end = self.indptr[e + 1];
        (&self.indices[start..end], &self.values[start..end])
    }

    /// Total non-zero entries across all rows.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Extract a sub-store for the given edge indices.
    pub fn subset(&self, edge_indices: &[usize]) -> Self {
        let n = edge_indices.len();
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0usize);

        let mut nnz_est = 0usize;
        for &e in edge_indices {
            nnz_est += self.indptr[e + 1] - self.indptr[e];
        }
        let mut indices = Vec::with_capacity(nnz_est);
        let mut values = Vec::with_capacity(nnz_est);
        let mut size_factors = Vec::with_capacity(n);

        for &e in edge_indices {
            let (cols, vals) = self.row(e);
            indices.extend_from_slice(cols);
            values.extend_from_slice(vals);
            size_factors.push(self.size_factors[e]);
            indptr.push(indices.len());
        }

        LinkProfileStore {
            indptr,
            indices,
            values,
            size_factors,
            n_edges: n,
            m: self.m,
        }
    }
}

/// Sufficient statistics for the link community model.
///
/// All accumulators are f64 to prevent drift during incremental updates.
pub struct LinkCommunityStats {
    pub k: usize,
    pub m: usize,
    pub n_edges: usize,
    /// Per-community per-gene sum: `gene_sum[k*m + g] = Σ_{e: z_e=k} y_e^g`.
    pub gene_sum: Vec<f64>,
    /// Per-community size factor sum: `size_sum[k] = Σ_{e: z_e=k} s_e`.
    pub size_sum: Vec<f64>,
    /// Per-community edge count.
    pub edge_count: Vec<usize>,
    pub membership: Vec<usize>,
    /// Cached `ln(gene_sum[k*m + g] + ε)`, kept in sync with `gene_sum`.
    pub(crate) log_gene: Vec<f64>,
    /// Cached `-ln(size_sum[k] + M·ε)`, kept in sync with `size_sum`.
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
            let (cols, vals) = profiles.row(e);
            let base = c * m;
            for (&col, &v) in cols.iter().zip(vals.iter()) {
                gene_sum[base + col as usize] += v as f64;
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
    /// Only log_gene entries for non-zero profile columns are recomputed
    /// (O(nnz(y_e)) per side).
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
        let (cols, vals) = profiles.row(e);
        let sf = profiles.size_factors[e] as f64;
        let m_eps = (m as f64) * LOG_EPS;

        let old_base = old_k * m;
        for (&col, &y) in cols.iter().zip(vals.iter()) {
            let idx = old_base + col as usize;
            self.gene_sum[idx] -= y as f64;
            self.log_gene[idx] = (self.gene_sum[idx] + LOG_EPS).ln();
        }
        let new_base = new_k * m;
        for (&col, &y) in cols.iter().zip(vals.iter()) {
            let idx = new_base + col as usize;
            self.gene_sum[idx] += y as f64;
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

    /// Aggregate stats for a subset of edges (identified by global indices).
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
            let (cols, vals) = profiles.row(e);
            let base = c * m;
            for (&col, &v) in cols.iter().zip(vals.iter()) {
                gene_sum[base + col as usize] += v as f64;
            }
            size_sum[c] += profiles.size_factors[e] as f64;
            edge_count[c] += 1;
        }

        (gene_sum, size_sum, edge_count)
    }

    /// Aggregate stats for a contiguous sub-store with local membership.
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
            let (cols, vals) = profiles.row(e);
            let base = c * m;
            for (&col, &v) in cols.iter().zip(vals.iter()) {
                gene_sum[base + col as usize] += v as f64;
            }
            size_sum[c] += profiles.size_factors[e] as f64;
            edge_count[c] += 1;
        }

        (gene_sum, size_sum, edge_count)
    }

    /// Apply a delta to the sufficient statistics: self += (new - old).
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
            let (cols, vals) = profiles.row(e);
            let base = c * m;
            for (&col, &v) in cols.iter().zip(vals.iter()) {
                self.gene_sum[base + col as usize] += v as f64;
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
    pub fn compute_log_weights(&self, alpha: f64) -> Vec<f64> {
        let alpha_k = alpha / self.k as f64;
        let total = self.n_edges as f64 + alpha;
        let dg_total = SpecialGamma::digamma(total);
        (0..self.k)
            .map(|c| SpecialGamma::digamma(self.edge_count[c] as f64 + alpha_k) - dg_total)
            .collect()
    }

    /// Total conditional-entropy / plug-in multinomial score (higher = better).
    pub fn total_score(&self) -> f64 {
        let sum_f_t: f64 = self.gene_sum.iter().map(|&t| f_entropy(t)).sum();
        let sum_f_s: f64 = self.size_sum.iter().map(|&s| f_entropy(s)).sum();
        sum_f_t - sum_f_s
    }

    /// Mutual information between community assignment and gene profile, in nats.
    pub fn mutual_information(&self) -> f64 {
        self.score_and_mi().1
    }

    /// Fused `(total_score, mutual_information)` in one pass.
    pub fn score_and_mi(&self) -> (f64, f64) {
        let mut p_global = vec![0.0f64; self.m];
        let mut sum_f_t = 0.0f64;
        for chunk in self.gene_sum.chunks_exact(self.m) {
            for (acc, &v) in p_global.iter_mut().zip(chunk.iter()) {
                *acc += v;
                sum_f_t += f_entropy(v);
            }
        }
        let mut total = 0.0f64;
        let mut sum_f_s = 0.0f64;
        for &s in &self.size_sum {
            total += s;
            sum_f_s += f_entropy(s);
        }
        let score = sum_f_t - sum_f_s;
        if total <= 0.0 {
            return (score, 0.0);
        }
        let inv_total = 1.0 / total;
        let h_global: f64 = p_global
            .iter()
            .map(|&t| {
                let p = t * inv_total;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        (score, h_global + score * inv_total)
    }
}

#[cfg(test)]
/// Amortized classifier for link community assignments (multinomial naive Bayes).
pub struct LinkCommunityClassifier {
    pub log_rates: Vec<f64>,
    pub log_prior: Vec<f64>,
    pub k: usize,
    pub m: usize,
}

#[cfg(test)]
impl LinkCommunityClassifier {
    pub fn from_stats(stats: &LinkCommunityStats) -> Self {
        let k = stats.k;
        let m = stats.m;
        let mut log_rates = vec![0.0f64; k * m];
        for c in 0..k {
            let off = stats.log_size_offset[c];
            let base = c * m;
            for g in 0..m {
                log_rates[base + g] = stats.log_gene[base + g] + off;
            }
        }

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

    /// Predict community assignment for a single sparse row `(cols, vals)`.
    #[inline]
    pub fn predict_one_sparse(&self, cols: &[u32], vals: &[f32]) -> usize {
        let mut best_k = 0;
        let mut best_score = f64::NEG_INFINITY;
        for c in 0..self.k {
            let base = c * self.m;
            let mut score = self.log_prior[c];
            for (&col, &y) in cols.iter().zip(vals.iter()) {
                score += y as f64 * self.log_rates[base + col as usize];
            }
            if score > best_score {
                best_score = score;
                best_k = c;
            }
        }
        best_k
    }

    pub fn predict_labels(&self, profiles: &LinkProfileStore) -> Vec<usize> {
        (0..profiles.n_edges)
            .map(|e| {
                let (cols, vals) = profiles.row(e);
                self.predict_one_sparse(cols, vals)
            })
            .collect()
    }

    pub fn predict_labels_parallel(&self, profiles: &LinkProfileStore) -> Vec<usize> {
        use rayon::prelude::*;
        let chunk_size = std::cmp::max(256, profiles.n_edges / rayon::current_num_threads().max(1));
        let indices: Vec<usize> = (0..profiles.n_edges).collect();
        indices
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|&e| {
                        let (cols, vals) = profiles.row(e);
                        self.predict_one_sparse(cols, vals)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Compute log-probabilities for assigning edge `e` to each community.
///
/// `log_probs[t] = score(e, t) - score(e, current_c)`, with the current-community
/// slot set to 0 so categorical sampling / argmax operate on deltas.
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
    let (cols, vals) = profiles.row(e);
    let sf = profiles.size_factors[e] as f64;
    let lw_current = log_weights.map(|w| w[current_c]).unwrap_or(0.0);

    let src_slice = &stats.log_gene[current_c * m..(current_c + 1) * m];
    let src_score = edge_score(cols, vals, src_slice, sf, stats.log_size_offset[current_c]);

    for (t, lp) in log_probs.iter_mut().enumerate().take(k) {
        if t == current_c {
            *lp = 0.0;
            continue;
        }
        let tgt_slice = &stats.log_gene[t * m..(t + 1) * m];
        let tgt_score = edge_score(cols, vals, tgt_slice, sf, stats.log_size_offset[t]);
        let mut delta = tgt_score - src_score;
        if let Some(w) = log_weights {
            delta += w[t] - lw_current;
        }
        *lp = delta;
    }
}

/// Sparse inner dot product `Σ y · log_gene_slice[col] + sf · log_size_offset_k`.
#[inline]
fn edge_score(
    cols: &[u32],
    vals: &[f32],
    log_gene_slice: &[f64],
    sf: f64,
    log_size_offset_k: f64,
) -> f64 {
    let mut s = sf * log_size_offset_k;
    for (&col, &y) in cols.iter().zip(vals.iter()) {
        s += y as f64 * log_gene_slice[col as usize];
    }
    s
}
