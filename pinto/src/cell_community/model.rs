//! Flat-K cell community sufficient statistics and Poisson DC-SBM scoring.
//!
//! Ported from `link_community::model` with edge→cell substitution.
//!
//! Each cell u has a profile y_u ∈ R^M and an assignment z_u ∈ {0..K-1}.
//! Plug-in Poisson DC-SBM log-rates:
//!
//!   log_rate[k, g] = log_gene[k, g] + log_size_offset[k]
//!   log_gene[k, g]        = ln(T_{kg} + ε)
//!   log_size_offset[k]    = −ln(S_k + M·ε)
//!
//! with T_{kg} = Σ_{u: z_u=k} y_{ug}, S_k = Σ_{u: z_u=k} s_u.
//!
//! Move delta (cell u from src to tgt):
//!   Δ = Σ_g y_{ug} · (log_rate[tgt, g] − log_rate[src, g])

use super::profiles::CellProfileStore;
use special::Gamma as SpecialGamma;

const LOG_EPS: f64 = 1e-9;

#[inline]
pub(crate) fn f_entropy(x: f64) -> f64 {
    if x > 0.0 {
        x * x.ln()
    } else {
        0.0
    }
}

pub struct CellCommunityStats {
    pub k: usize,
    pub m: usize,
    pub n_cells: usize,
    /// gene_sum[k * m + g] = Σ_{u: z_u=k} y_{ug}
    pub gene_sum: Vec<f64>,
    /// size_sum[k] = Σ_{u: z_u=k} s_u
    pub size_sum: Vec<f64>,
    pub cell_count: Vec<usize>,
    pub membership: Vec<usize>,
    pub(crate) log_gene: Vec<f64>,
    pub(crate) log_size_offset: Vec<f64>,
}

impl CellCommunityStats {
    pub fn from_profiles(profiles: &CellProfileStore, k: usize, labels: &[usize]) -> Self {
        let m = profiles.m;
        let n_cells = profiles.n_cells;
        debug_assert_eq!(labels.len(), n_cells);

        let mut gene_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut cell_count = vec![0usize; k];

        for (u, &c) in labels.iter().enumerate() {
            debug_assert!(c < k);
            let row = profiles.profile(u);
            let base = c * m;
            for g in 0..m {
                gene_sum[base + g] += row[g] as f64;
            }
            size_sum[c] += profiles.size_factors[u] as f64;
            cell_count[c] += 1;
        }

        let log_gene: Vec<f64> = gene_sum.iter().map(|&t| (t + LOG_EPS).ln()).collect();
        let m_eps = (m as f64) * LOG_EPS;
        let log_size_offset: Vec<f64> = size_sum.iter().map(|&s| -((s + m_eps).ln())).collect();

        Self {
            k,
            m,
            n_cells,
            gene_sum,
            size_sum,
            cell_count,
            membership: labels.to_vec(),
            log_gene,
            log_size_offset,
        }
    }

    /// Move cell `u` from `old_k` to `new_k`, updating all caches incrementally.
    #[inline]
    pub fn delta_move(
        &mut self,
        u: usize,
        old_k: usize,
        new_k: usize,
        profiles: &CellProfileStore,
    ) {
        debug_assert_eq!(self.membership[u], old_k);
        let m = self.m;
        let row = profiles.profile(u);
        let sf = profiles.size_factors[u] as f64;
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
        self.cell_count[old_k] -= 1;
        self.cell_count[new_k] += 1;
        self.membership[u] = new_k;
    }

    /// Aggregate stats over a subset of cells identified by global indices.
    pub fn component_stats(
        profiles: &CellProfileStore,
        k: usize,
        cell_indices: &[usize],
        membership: &[usize],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let m = profiles.m;
        let mut gene_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut cell_count = vec![0usize; k];
        for &u in cell_indices {
            let c = membership[u];
            debug_assert!(c < k);
            let row = profiles.profile(u);
            let base = c * m;
            for g in 0..m {
                gene_sum[base + g] += row[g] as f64;
            }
            size_sum[c] += profiles.size_factors[u] as f64;
            cell_count[c] += 1;
        }
        (gene_sum, size_sum, cell_count)
    }

    /// Aggregate stats over a contiguous sub-store using local membership.
    pub fn local_stats(
        profiles: &CellProfileStore,
        k: usize,
        membership: &[usize],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let m = profiles.m;
        let mut gene_sum = vec![0.0f64; k * m];
        let mut size_sum = vec![0.0f64; k];
        let mut cell_count = vec![0usize; k];
        for (u, &c) in membership.iter().enumerate() {
            debug_assert!(c < k);
            let row = profiles.profile(u);
            let base = c * m;
            for g in 0..m {
                gene_sum[base + g] += row[g] as f64;
            }
            size_sum[c] += profiles.size_factors[u] as f64;
            cell_count[c] += 1;
        }
        (gene_sum, size_sum, cell_count)
    }

    /// Patch global stats with (new − old) component delta.
    pub fn apply_delta(
        &mut self,
        old: (&[f64], &[f64], &[usize]),
        new: (&[f64], &[f64], &[usize]),
    ) {
        let km = self.k * self.m;
        for i in 0..km {
            let d = new.0[i] - old.0[i];
            if d != 0.0 {
                self.gene_sum[i] += d;
                self.log_gene[i] = (self.gene_sum[i] + LOG_EPS).ln();
            }
        }
        let m_eps = (self.m as f64) * LOG_EPS;
        for c in 0..self.k {
            let d = new.1[c] - old.1[c];
            if d != 0.0 {
                self.size_sum[c] += d;
                self.log_size_offset[c] = -((self.size_sum[c] + m_eps).ln());
            }
            self.cell_count[c] =
                (self.cell_count[c] as isize + new.2[c] as isize - old.2[c] as isize) as usize;
        }
    }

    pub fn compute_log_weights(&self, alpha: f64) -> Vec<f64> {
        let alpha_k = alpha / self.k as f64;
        let total = self.n_cells as f64 + alpha;
        let dg_total = SpecialGamma::digamma(total);
        (0..self.k)
            .map(|c| SpecialGamma::digamma(self.cell_count[c] as f64 + alpha_k) - dg_total)
            .collect()
    }

    /// J(z) = Σ_{k,g} f(T_{kg}) − Σ_k f(S_k), f(x) = x·ln(x).
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

/// log_probs[t] = score(u, t) − score(u, current); current slot set to 0.
pub(crate) fn compute_log_probs_for_cell(
    u: usize,
    stats: &CellCommunityStats,
    profiles: &CellProfileStore,
    log_weights: Option<&[f64]>,
    log_probs: &mut [f64],
) {
    let k = stats.k;
    let m = stats.m;
    let current_c = stats.membership[u];
    let row = profiles.profile(u);
    let sf = profiles.size_factors[u] as f64;
    let lw_current = log_weights.map(|w| w[current_c]).unwrap_or(0.0);

    let src_slice = &stats.log_gene[current_c * m..(current_c + 1) * m];
    let src_score = cell_score(row, src_slice, sf, stats.log_size_offset[current_c]);

    for (t, lp) in log_probs.iter_mut().enumerate().take(k) {
        if t == current_c {
            *lp = 0.0;
            continue;
        }
        let tgt_slice = &stats.log_gene[t * m..(t + 1) * m];
        let tgt_score = cell_score(row, tgt_slice, sf, stats.log_size_offset[t]);
        let mut delta = tgt_score - src_score;
        if let Some(w) = log_weights {
            delta += w[t] - lw_current;
        }
        *lp = delta;
    }
}

#[inline]
fn cell_score(row: &[f32], log_gene_slice: &[f64], sf: f64, log_size_offset_k: f64) -> f64 {
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

    fn make_synthetic(n: usize, m: usize, k: usize) -> (CellProfileStore, Vec<usize>) {
        let mut profiles = vec![0.0f32; n * m];
        let mut labels = vec![0usize; n];
        for u in 0..n {
            let c = u % k;
            labels[u] = c;
            for g in 0..m {
                profiles[u * m + g] = if g % k == c { 5.0 } else { 0.5 };
            }
        }
        (CellProfileStore::new(profiles, n, m), labels)
    }

    #[test]
    fn test_from_profiles_basic() {
        let (store, labels) = make_synthetic(20, 6, 3);
        let stats = CellCommunityStats::from_profiles(&store, 3, &labels);
        assert_eq!(stats.k, 3);
        assert_eq!(stats.m, 6);
        assert_eq!(stats.n_cells, 20);
        assert_eq!(stats.cell_count.iter().sum::<usize>(), 20);
    }

    #[test]
    fn test_delta_move_consistency() {
        let (store, labels) = make_synthetic(30, 4, 3);
        let mut stats = CellCommunityStats::from_profiles(&store, 3, &labels);
        let old = stats.membership[0];
        let new = (old + 1) % 3;
        stats.delta_move(0, old, new, &store);
        let fresh = CellCommunityStats::from_profiles(&store, 3, &stats.membership);
        for i in 0..stats.gene_sum.len() {
            assert!((stats.gene_sum[i] - fresh.gene_sum[i]).abs() < 1e-9);
            assert!((stats.log_gene[i] - fresh.log_gene[i]).abs() < 1e-12);
        }
        for c in 0..stats.k {
            assert!((stats.size_sum[c] - fresh.size_sum[c]).abs() < 1e-9);
            assert!((stats.log_size_offset[c] - fresh.log_size_offset[c]).abs() < 1e-12);
            assert_eq!(stats.cell_count[c], fresh.cell_count[c]);
        }
    }

    #[test]
    fn test_delta_score_matches_hand_dot_product() {
        let (store, labels) = make_synthetic(15, 4, 3);
        let stats = CellCommunityStats::from_profiles(&store, 3, &labels);
        let mut log_probs = vec![0.0f64; 3];
        for u in 0..stats.n_cells {
            let current = stats.membership[u];
            let row = store.profile(u);
            let sf = store.size_factors[u] as f64;
            compute_log_probs_for_cell(u, &stats, &store, None, &mut log_probs);
            assert!(log_probs[current].abs() < 1e-10);
            for (t, &got) in log_probs.iter().enumerate() {
                if t == current {
                    continue;
                }
                let mut expected =
                    sf * (stats.log_size_offset[t] - stats.log_size_offset[current]);
                for (g, &y) in row.iter().enumerate() {
                    let y = y as f64;
                    if y == 0.0 {
                        continue;
                    }
                    expected += y
                        * (stats.log_gene[t * stats.m + g]
                            - stats.log_gene[current * stats.m + g]);
                }
                assert!(
                    (got - expected).abs() < 1e-10,
                    "u={u} t={t}: got={got} expected={expected}"
                );
            }
        }
    }
}
