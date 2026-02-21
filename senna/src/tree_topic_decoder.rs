#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]
//! Tree-structured topic decoder with two-level multinomial decomposition.
//!
//! P(gene g | topic k) = P(module m(g) | topic k) × P(gene g | module m(g))
//!
//! The outer level (module-topic) uses collapsed Poisson-Gamma scoring
//! with gene module assignments updated via Gibbs sampling.
//! The inner level (within-module) spreads probability proportional to
//! marginal expression.
//!
//! This avoids a global G-way softmax — normalization is over M modules (M << G).

use candle_util::candle_core;
use leiden::clustering::SimpleClustering;
use leiden::leiden::Leiden;
use leiden::network::Graph;
use leiden::{Clustering, Network};
use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use special::Gamma as SpecialGamma;

// ─── Gene profile store ─────────────────────────────────────────────────────

/// Gene count profiles across topics.
///
/// profile[g, k] = Σ_n θ_nk · x_ng  (how much gene g appears in topic k)
pub struct GeneProfileStore {
    /// Row-major [G × K]
    profiles: Vec<f32>,
    /// Per-gene total across topics: Σ_k profile[g, k]
    size_factors: Vec<f32>,
    n_genes: usize,
    n_topics: usize,
}

impl GeneProfileStore {
    /// Build gene profiles from count matrix and topic proportions.
    ///
    /// * `x_ng` - Dense count matrix [N × G], row-major
    /// * `theta_nk` - Topic proportions [N × K], row-major
    /// * `n_cells` - Number of cells (N)
    /// * `n_genes` - Number of genes (G)
    /// * `n_topics` - Number of topics (K)
    ///
    /// Computes profile[g, k] = Σ_n θ_nk · x_ng  (i.e., X^T @ Θ)
    pub fn from_dense(
        x_ng: &[f32],
        theta_nk: &[f32],
        n_cells: usize,
        n_genes: usize,
        n_topics: usize,
    ) -> Self {
        debug_assert_eq!(x_ng.len(), n_cells * n_genes);
        debug_assert_eq!(theta_nk.len(), n_cells * n_topics);

        let mut profiles = vec![0.0f32; n_genes * n_topics];

        // profile = X^T @ Θ  →  [G × K]
        for n in 0..n_cells {
            let x_row = &x_ng[n * n_genes..(n + 1) * n_genes];
            let t_row = &theta_nk[n * n_topics..(n + 1) * n_topics];
            for g in 0..n_genes {
                let xval = x_row[g];
                if xval == 0.0 {
                    continue;
                }
                let base = g * n_topics;
                for k in 0..n_topics {
                    profiles[base + k] += xval * t_row[k];
                }
            }
        }

        let size_factors: Vec<f32> = (0..n_genes)
            .map(|g| {
                let row = &profiles[g * n_topics..(g + 1) * n_topics];
                row.iter().sum()
            })
            .collect();

        GeneProfileStore {
            profiles,
            size_factors,
            n_genes,
            n_topics,
        }
    }

    /// Build gene profiles directly from pseudobulk data (no θ weighting).
    ///
    /// Each pseudobulk sample acts as its own "topic". The profile is just
    /// the transposed count matrix: profile[g, s] = x_sg (gene g in sample s).
    ///
    /// * `x_ns` - Dense pseudobulk matrix [N_samples × G], row-major
    /// * `n_samples` - Number of pseudobulk samples (used as profile dimension)
    /// * `n_genes` - Number of genes (G)
    pub fn from_pseudobulk(x_ns: &[f32], n_samples: usize, n_genes: usize) -> Self {
        debug_assert_eq!(x_ns.len(), n_samples * n_genes);

        // profile[g, s] = x[s, g]  (just transpose)
        let mut profiles = vec![0.0f32; n_genes * n_samples];
        for s in 0..n_samples {
            let x_row = &x_ns[s * n_genes..(s + 1) * n_genes];
            for g in 0..n_genes {
                profiles[g * n_samples + s] = x_row[g];
            }
        }

        let size_factors: Vec<f32> = (0..n_genes)
            .map(|g| {
                let row = &profiles[g * n_samples..(g + 1) * n_samples];
                row.iter().sum()
            })
            .collect();

        GeneProfileStore {
            profiles,
            size_factors,
            n_genes,
            n_topics: n_samples,
        }
    }

    /// Get the K-dimensional profile for gene g.
    #[inline]
    pub fn profile(&self, g: usize) -> &[f32] {
        &self.profiles[g * self.n_topics..(g + 1) * self.n_topics]
    }

    pub fn n_genes(&self) -> usize {
        self.n_genes
    }

    pub fn n_topics(&self) -> usize {
        self.n_topics
    }
}

// ─── Gene module sufficient statistics ──────────────────────────────────────

/// Sufficient statistics for gene modules (adapted from pinto's LinkCommunityStats).
///
/// gene_sum[m * K + k] = Σ_{g: membership[g]=m} profile[g, k]
pub struct GeneModuleStats {
    /// Number of modules
    m: usize,
    /// Number of topics
    k: usize,
    /// Number of genes
    n_genes: usize,
    /// Per-module per-topic count sum: [M × K]
    gene_sum: Vec<f64>,
    /// Per-module total count: [M]
    size_sum: Vec<f64>,
    /// Number of genes per module: [M]
    module_count: Vec<usize>,
    /// Gene → module assignment: [G]
    membership: Vec<usize>,
}

impl GeneModuleStats {
    /// Build from gene profiles and initial assignments.
    pub fn from_profiles(profiles: &GeneProfileStore, m: usize, labels: &[usize]) -> Self {
        let k = profiles.n_topics;
        let n_genes = profiles.n_genes;
        debug_assert_eq!(labels.len(), n_genes);

        let mut gene_sum = vec![0.0f64; m * k];
        let mut size_sum = vec![0.0f64; m];
        let mut module_count = vec![0usize; m];

        for g in 0..n_genes {
            let c = labels[g];
            debug_assert!(c < m);
            let row = profiles.profile(g);
            let base = c * k;
            for t in 0..k {
                gene_sum[base + t] += row[t] as f64;
            }
            size_sum[c] += profiles.size_factors[g] as f64;
            module_count[c] += 1;
        }

        GeneModuleStats {
            m,
            k,
            n_genes,
            gene_sum,
            size_sum,
            module_count,
            membership: labels.to_vec(),
        }
    }

    /// Move gene `g` from `old_m` to `new_m`, updating stats incrementally. O(K).
    #[inline]
    pub fn delta_move(
        &mut self,
        g: usize,
        old_m: usize,
        new_m: usize,
        profiles: &GeneProfileStore,
    ) {
        debug_assert_eq!(self.membership[g], old_m);
        let k = self.k;
        let row = profiles.profile(g);
        let sf = profiles.size_factors[g] as f64;

        let old_base = old_m * k;
        let new_base = new_m * k;
        for t in 0..k {
            let v = row[t] as f64;
            self.gene_sum[old_base + t] -= v;
            self.gene_sum[new_base + t] += v;
        }

        self.size_sum[old_m] -= sf;
        self.size_sum[new_m] += sf;
        self.module_count[old_m] -= 1;
        self.module_count[new_m] += 1;
        self.membership[g] = new_m;
    }

    /// Recompute all statistics from scratch (drift correction).
    pub fn recompute(&mut self, profiles: &GeneProfileStore) {
        let k = self.k;
        self.gene_sum.iter_mut().for_each(|x| *x = 0.0);
        self.size_sum.iter_mut().for_each(|x| *x = 0.0);
        self.module_count.iter_mut().for_each(|x| *x = 0);

        for g in 0..self.n_genes {
            let c = self.membership[g];
            let row = profiles.profile(g);
            let base = c * k;
            for t in 0..k {
                self.gene_sum[base + t] += row[t] as f64;
            }
            self.size_sum[c] += profiles.size_factors[g] as f64;
            self.module_count[c] += 1;
        }
    }

    /// Total collapsed Poisson-Gamma score across all modules and topics.
    pub fn total_score(&self, a0: f64, b0: f64) -> f64 {
        let k = self.k;
        let mut score = 0.0f64;
        for c in 0..self.m {
            let t_m = self.size_sum[c];
            let base = c * k;
            for t in 0..k {
                let e_mk = self.gene_sum[base + t];
                score += poisson_score(a0, b0, e_mk, t_m);
            }
        }
        score
    }

    pub fn n_modules(&self) -> usize {
        self.m
    }

    pub fn n_topics(&self) -> usize {
        self.k
    }

    pub fn membership(&self) -> &[usize] {
        &self.membership
    }

    pub fn module_count(&self) -> &[usize] {
        &self.module_count
    }
}

// ─── Poisson-Gamma score ────────────────────────────────────────────────────

/// Poisson-Gamma conjugate score (collapsed marginal log-likelihood for one cell).
#[inline]
fn poisson_score(a0: f64, b0: f64, edge: f64, total: f64) -> f64 {
    a0 * b0.ln() + SpecialGamma::ln_gamma(a0 + edge).0
        - SpecialGamma::ln_gamma(a0).0
        - (a0 + edge) * (b0 + total).ln()
}

// ─── Gene reassignment scoring ──────────────────────────────────────────────

/// Compute log-probabilities for assigning gene `g` to each module.
///
/// For each target module t, computes delta score if gene g were moved
/// from its current module to t. Current module gets delta = 0.0.
///
/// Complexity: O(M × K).
fn compute_log_probs_for_gene(
    g: usize,
    stats: &GeneModuleStats,
    profiles: &GeneProfileStore,
    a0: f64,
    b0: f64,
    log_probs: &mut [f64],
) {
    let m = stats.m;
    let k = stats.k;
    let current_m = stats.membership[g];
    let row = profiles.profile(g);
    let sf = profiles.size_factors[g] as f64;

    let old_size = stats.size_sum[current_m];
    let new_size_removed = old_size - sf;

    for t in 0..m {
        if t == current_m {
            log_probs[t] = 0.0;
            continue;
        }

        let target_size = stats.size_sum[t];
        let new_target_size = target_size + sf;

        let mut delta = 0.0f64;
        let src_base = current_m * k;
        let tgt_base = t * k;

        for topic in 0..k {
            let y = row[topic] as f64;

            // Source module (gene removed)
            let old_e_src = stats.gene_sum[src_base + topic];
            let new_e_src = old_e_src - y;
            delta += poisson_score(a0, b0, new_e_src, new_size_removed)
                - poisson_score(a0, b0, old_e_src, old_size);

            // Target module (gene added)
            let old_e_tgt = stats.gene_sum[tgt_base + topic];
            let new_e_tgt = old_e_tgt + y;
            delta += poisson_score(a0, b0, new_e_tgt, new_target_size)
                - poisson_score(a0, b0, old_e_tgt, target_size);
        }

        log_probs[t] = delta;
    }
}

// ─── Within-module weights ──────────────────────────────────────────────────

/// Within-module gene distribution: log P(gene g | module m(g)).
pub struct WithinModuleWeights {
    /// [G] log π_{g|m(g)}
    log_weights: Vec<f32>,
}

impl WithinModuleWeights {
    /// Set π_{g|m} proportional to gene g's total expression, normalized within module.
    pub fn from_marginal_expression(
        size_factors: &[f32],
        membership: &[usize],
        n_modules: usize,
    ) -> Self {
        let n_genes = membership.len();
        let eps = 1e-30f64;

        // Compute per-module total
        let mut module_total = vec![0.0f64; n_modules];
        for g in 0..n_genes {
            module_total[membership[g]] += size_factors[g] as f64 + eps;
        }

        let log_weights: Vec<f32> = (0..n_genes)
            .map(|g| {
                let m = membership[g];
                let w = (size_factors[g] as f64 + eps) / module_total[m];
                w.ln() as f32
            })
            .collect();

        WithinModuleWeights { log_weights }
    }

    /// Uniform weights within each module.
    pub fn uniform(membership: &[usize], module_count: &[usize]) -> Self {
        let n_genes = membership.len();
        let log_weights: Vec<f32> = (0..n_genes)
            .map(|g| {
                let m = membership[g];
                let count = module_count[m].max(1) as f64;
                (-(count.ln())) as f32
            })
            .collect();
        WithinModuleWeights { log_weights }
    }
}

// ─── Sampling utilities ─────────────────────────────────────────────────────

/// Sample from a categorical distribution given log-probabilities.
fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
    let max = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_probs.iter().map(|lp| (lp - max).exp()).collect();
    let total: f64 = weights.iter().sum();

    if total <= 0.0 || !total.is_finite() {
        return rng.random_range(0..log_probs.len());
    }

    let u: f64 = rng.random::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= u {
            return i;
        }
    }
    weights.len() - 1
}

/// Pick the index with the highest value.
fn argmax_log(log_probs: &[f64]) -> usize {
    let mut best = 0;
    let mut best_val = log_probs[0];
    for (i, &v) in log_probs.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

// ─── Tree topic decoder ─────────────────────────────────────────────────────

/// Tree-structured topic decoder using two-level multinomial decomposition.
pub struct TreeTopicDecoder {
    stats: GeneModuleStats,
    profiles: GeneProfileStore,
    within_weights: WithinModuleWeights,
    a0: f64,
    b0: f64,
    rng: SmallRng,
    parallel_seed: u64,
}

impl TreeTopicDecoder {
    /// Create a new tree topic decoder with random initial module assignments.
    ///
    /// * `profiles` - Gene count profiles [G × K]
    /// * `n_modules` - Number of gene modules (M)
    /// * `a0` - Gamma shape prior
    /// * `b0` - Gamma rate prior
    /// * `seed` - Random seed
    pub fn new(profiles: GeneProfileStore, n_modules: usize, a0: f64, b0: f64, seed: u64) -> Self {
        let n_genes = profiles.n_genes();
        let mut rng = SmallRng::seed_from_u64(seed);

        // Random initial assignment
        let labels: Vec<usize> = (0..n_genes)
            .map(|_| rng.random_range(0..n_modules))
            .collect();

        let stats = GeneModuleStats::from_profiles(&profiles, n_modules, &labels);
        let within_weights = WithinModuleWeights::from_marginal_expression(
            &profiles.size_factors,
            &labels,
            n_modules,
        );

        TreeTopicDecoder {
            stats,
            profiles,
            within_weights,
            a0,
            b0,
            rng,
            parallel_seed: 0,
        }
    }

    /// Create a tree topic decoder from pre-computed module assignments.
    ///
    /// No Gibbs/greedy sweeps — just builds stats from profiles + given membership.
    pub fn from_modules(
        profiles: GeneProfileStore,
        membership: Vec<usize>,
        n_modules: usize,
        a0: f64,
        b0: f64,
        seed: u64,
    ) -> Self {
        let stats = GeneModuleStats::from_profiles(&profiles, n_modules, &membership);
        let within_weights = WithinModuleWeights::from_marginal_expression(
            &profiles.size_factors,
            &membership,
            n_modules,
        );

        TreeTopicDecoder {
            stats,
            profiles,
            within_weights,
            a0,
            b0,
            rng: SmallRng::seed_from_u64(seed),
            parallel_seed: 0,
        }
    }

    /// Initialize from pseudobulk data: run Gibbs on pseudobulk profiles to find
    /// gene modules, then switch to topic-space profiles for training.
    ///
    /// * `pb_ns` - Pseudobulk matrix [N_samples × G], row-major
    /// * `n_samples` - Number of pseudobulk samples
    /// * `n_genes` - Number of genes
    /// * `n_modules` - Number of gene modules (M)
    /// * `a0`, `b0` - Gamma priors
    /// * `gibbs_sweeps`, `greedy_sweeps` - Sweeps for pseudobulk initialization
    /// * `seed` - Random seed
    ///
    /// After initialization, call `update_profiles()` to switch to topic-space
    /// profiles before the first E-step.
    pub fn from_pseudobulk(
        pb_ns: &[f32],
        n_samples: usize,
        n_genes: usize,
        n_modules: usize,
        a0: f64,
        b0: f64,
        gibbs_sweeps: usize,
        greedy_sweeps: usize,
        seed: u64,
    ) -> Self {
        let pb_profiles = GeneProfileStore::from_pseudobulk(pb_ns, n_samples, n_genes);
        let mut td = Self::new(pb_profiles, n_modules, a0, b0, seed);
        td.gibbs_sweep_parallel(gibbs_sweeps);
        td.greedy_sweep(greedy_sweeps);
        td
    }

    /// Update gene profiles from new data and topic proportions.
    ///
    /// * `x_ng` - Dense count matrix [N × G], row-major
    /// * `theta_nk` - Topic proportions [N × K], row-major
    /// * `n_cells` - Number of cells (N)
    /// * `n_topics` - Number of topics (K)
    ///
    /// The topic dimension may differ from the current profile dimension
    /// (e.g., after pseudobulk initialization where n_topics = n_samples).
    /// Stats are rebuilt with the correct k when the dimension changes.
    pub fn update_profiles(
        &mut self,
        x_ng: &[f32],
        theta_nk: &[f32],
        n_cells: usize,
        n_topics: usize,
    ) {
        let n_genes = self.profiles.n_genes;
        self.profiles = GeneProfileStore::from_dense(x_ng, theta_nk, n_cells, n_genes, n_topics);
        if self.stats.k != n_topics {
            // Dimension changed (e.g., pseudobulk → topic space): rebuild stats
            self.stats = GeneModuleStats::from_profiles(
                &self.profiles,
                self.stats.m,
                &self.stats.membership,
            );
        } else {
            self.stats.recompute(&self.profiles);
        }
    }

    /// Update within-module weights based on current membership and marginal expression.
    pub fn update_within_weights(&mut self) {
        self.within_weights = WithinModuleWeights::from_marginal_expression(
            &self.profiles.size_factors,
            &self.stats.membership,
            self.stats.m,
        );
    }

    /// Run sequential Gibbs sweeps over all genes.
    ///
    /// Returns the total number of gene moves.
    pub fn gibbs_sweep(&mut self, num_sweeps: usize) -> usize {
        let m = self.stats.m;
        let n_genes = self.stats.n_genes;
        let mut log_probs = vec![0.0f64; m];
        let mut total_moves = 0;

        for _sweep in 0..num_sweeps {
            for g in 0..n_genes {
                let old_m = self.stats.membership[g];

                compute_log_probs_for_gene(
                    g,
                    &self.stats,
                    &self.profiles,
                    self.a0,
                    self.b0,
                    &mut log_probs,
                );

                let new_m = sample_categorical_log(&log_probs, &mut self.rng);

                if new_m != old_m {
                    self.stats.delta_move(g, old_m, new_m, &self.profiles);
                    total_moves += 1;
                }
            }
        }

        total_moves
    }

    /// Run parallel (Jacobi-style) Gibbs sweeps.
    ///
    /// All genes compute proposals against a frozen snapshot, then moves
    /// are applied sequentially.
    pub fn gibbs_sweep_parallel(&mut self, num_sweeps: usize) -> usize {
        let m = self.stats.m;
        let n_genes = self.stats.n_genes;

        if self.parallel_seed == 0 {
            self.parallel_seed = self.rng.random::<u64>() | 1;
        }
        let base_seed = self.parallel_seed;

        let mut total_moves = 0;
        let gene_order: Vec<usize> = (0..n_genes).collect();
        let chunk_size = std::cmp::max(256, n_genes / rayon::current_num_threads().max(1));

        for sweep in 0..num_sweeps {
            let sweep_seed = base_seed.wrapping_mul(sweep as u64 + 1);

            // Phase 1: Parallel proposals (read-only on stats)
            let proposals: Vec<usize> = gene_order
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    let mut log_probs = vec![0.0f64; m];
                    chunk
                        .iter()
                        .map(|&g| {
                            compute_log_probs_for_gene(
                                g,
                                &self.stats,
                                &self.profiles,
                                self.a0,
                                self.b0,
                                &mut log_probs,
                            );
                            let vertex_seed = sweep_seed ^ (g as u64).wrapping_mul(2654435761);
                            let mut rng = SmallRng::seed_from_u64(vertex_seed);
                            sample_categorical_log(&log_probs, &mut rng)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            // Phase 2: Sequential apply
            let mut sweep_moves = 0;
            for g in 0..n_genes {
                let old_m = self.stats.membership[g];
                let new_m = proposals[g];
                if new_m != old_m {
                    self.stats.delta_move(g, old_m, new_m, &self.profiles);
                    sweep_moves += 1;
                }
            }
            total_moves += sweep_moves;

            log::info!(
                "  [gibbs {}/{}] moves={}, score={:.2}",
                sweep + 1,
                num_sweeps,
                sweep_moves,
                self.stats.total_score(self.a0, self.b0)
            );
        }

        total_moves
    }

    /// Run greedy (argmax) sweeps. Returns total moves. Stops early if converged.
    pub fn greedy_sweep(&mut self, num_sweeps: usize) -> usize {
        let m = self.stats.m;
        let n_genes = self.stats.n_genes;
        let mut log_probs = vec![0.0f64; m];
        let mut total_moves = 0;

        for sweep in 0..num_sweeps {
            let mut sweep_moves = 0;
            for g in 0..n_genes {
                let old_m = self.stats.membership[g];

                compute_log_probs_for_gene(
                    g,
                    &self.stats,
                    &self.profiles,
                    self.a0,
                    self.b0,
                    &mut log_probs,
                );

                let new_m = argmax_log(&log_probs);

                if new_m != old_m {
                    self.stats.delta_move(g, old_m, new_m, &self.profiles);
                    sweep_moves += 1;
                }
            }
            total_moves += sweep_moves;

            log::info!(
                "  [greedy {}/{}] moves={}, score={:.2}",
                sweep + 1,
                num_sweeps,
                sweep_moves,
                self.stats.total_score(self.a0, self.b0)
            );

            if sweep_moves == 0 {
                break;
            }
        }

        total_moves
    }

    /// Compute log ψ_mk (posterior mean rate) for each module-topic pair.
    ///
    /// ψ_mk = (a0 + count_mk) / (b0 + total_m)
    /// Returns [M × K] in row-major order.
    pub fn module_log_rates(&self) -> Vec<f64> {
        let m = self.stats.m;
        let k = self.stats.k;
        let mut log_rates = vec![0.0f64; m * k];

        for c in 0..m {
            let t_m = self.stats.size_sum[c];
            let base = c * k;
            for t in 0..k {
                let e_mk = self.stats.gene_sum[base + t];
                // Posterior mean of Gamma(a0 + e, b0 + total): (a0 + e) / (b0 + total)
                log_rates[base + t] = (self.a0 + e_mk).ln() - (self.b0 + t_m).ln();
            }
        }

        log_rates
    }

    /// Compute full log P(gene g | topic k) matrix.
    ///
    /// log β_gk = log ψ_{m(g),k} + log π_{g|m(g)} - logsumexp_m(log ψ_{m,k})
    ///
    /// Returns [G × K] in row-major f32.
    pub fn log_beta_gk(&self) -> Vec<f32> {
        let n_genes = self.profiles.n_genes;
        let k = self.stats.k;
        let m = self.stats.m;

        let log_rates = self.module_log_rates(); // [M × K]

        // Compute logsumexp over modules for each topic
        let mut log_norm = vec![0.0f64; k]; // [K]
        for t in 0..k {
            let mut max_val = f64::NEG_INFINITY;
            for c in 0..m {
                let v = log_rates[c * k + t];
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum_exp = 0.0f64;
            for c in 0..m {
                sum_exp += (log_rates[c * k + t] - max_val).exp();
            }
            log_norm[t] = max_val + sum_exp.ln();
        }

        // Build log β
        let mut log_beta = vec![0.0f32; n_genes * k];
        for g in 0..n_genes {
            let c = self.stats.membership[g];
            let log_pi = self.within_weights.log_weights[g];
            let mod_base = c * k;
            let gene_base = g * k;
            for t in 0..k {
                log_beta[gene_base + t] = (log_rates[mod_base + t] - log_norm[t]) as f32 + log_pi;
            }
        }

        log_beta
    }

    /// Materialize log_β as a candle Tensor [K × G] (transposed for matmul).
    ///
    /// The tensor is detached (no gradient). Reconstruction:
    /// log_recon_ng = θ_nk @ log_β_kg
    pub fn log_beta_tensor(
        &self,
        device: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let n_genes = self.profiles.n_genes;
        let k = self.stats.k;
        let log_beta = self.log_beta_gk(); // [G × K] row-major

        // Create [G × K] tensor then transpose to [K × G]
        let tensor =
            candle_core::Tensor::from_vec(log_beta, (n_genes, k), &candle_core::Device::Cpu)?;
        let tensor = tensor.t()?.to_device(device)?;
        Ok(tensor.detach())
    }

    /// Get current gene-to-module assignments.
    pub fn membership(&self) -> &[usize] {
        self.stats.membership()
    }

    /// Get number of modules.
    pub fn n_modules(&self) -> usize {
        self.stats.n_modules()
    }

    /// Get number of topics.
    pub fn n_topics(&self) -> usize {
        self.stats.n_topics()
    }

    /// Get number of genes.
    pub fn n_genes(&self) -> usize {
        self.profiles.n_genes()
    }

    /// Get total collapsed score.
    pub fn total_score(&self) -> f64 {
        self.stats.total_score(self.a0, self.b0)
    }

    /// Get module-topic gene_sum stats (for output).
    pub fn module_gene_sum(&self) -> &[f64] {
        &self.stats.gene_sum
    }
}

// ─── Gene module discovery via co-expression graph + Leiden ─────────────────

/// Discover gene modules from a gene expression matrix using KNN graph + Leiden.
///
/// Builds a gene-gene co-expression graph (genes as rows), partitions it with
/// Leiden community detection (modularity objective), and returns module labels.
///
/// * `pb_mat` - Gene profiles [G × S], row-major (genes as rows, samples as columns)
/// * `n_modules` - Target number of modules (used for resolution tuning)
/// * `knn` - Number of nearest neighbours for gene graph
/// * `resolution` - Leiden resolution (modularity scale, e.g., 1.0)
/// * `seed` - Optional random seed
pub fn discover_gene_modules(
    pb_mat: &DMatrix<f32>,
    n_modules: usize,
    knn: usize,
    resolution: f64,
    seed: Option<u64>,
) -> anyhow::Result<Vec<usize>> {
    let n_genes = pb_mat.nrows();
    let n_samples = pb_mat.ncols();

    if n_genes < 2 {
        anyhow::bail!("Need at least 2 genes for module discovery");
    }

    log::info!(
        "Discovering gene modules: {} genes x {} samples, knn={}, target_modules={}",
        n_genes,
        n_samples,
        knn,
        n_modules,
    );

    // Step 1: Z-score standardize rows (genes) — each gene has mean=0, std=1 across samples.
    // KnnGraph::from_rows uses Euclidean distance, and with z-scored rows,
    // Euclidean distance ∝ Pearson correlation distance.
    let mut gene_mat = pb_mat.clone();
    gene_mat.scale_columns_inplace(); // z-score columns (samples), but we want rows...

    // Actually we need to z-score rows (genes). scale_columns_inplace z-scores columns.
    // Transpose, z-score columns (= original rows), then use from_rows on the transposed matrix.
    let mut gene_mat_t = pb_mat.transpose();
    gene_mat_t.scale_columns_inplace(); // z-score columns of [S × G] = z-score genes
    let gene_mat = gene_mat_t.transpose(); // back to [G × S], now z-scored per gene

    let effective_knn = knn.min(n_genes - 1);

    log::info!(
        "Building gene KNN graph (k={}) for {} genes ...",
        effective_knn,
        n_genes
    );
    let graph = KnnGraph::from_rows(
        &gene_mat,
        KnnGraphArgs {
            knn: effective_knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;

    let mean_degree = if graph.num_nodes() > 0 {
        2.0 * graph.num_edges() as f64 / graph.num_nodes() as f64
    } else {
        0.0
    };
    log::info!(
        "Gene KNN graph: {} nodes, {} edges (mean degree {:.1})",
        graph.num_nodes(),
        graph.num_edges(),
        mean_degree
    );

    // Step 2: Convert to Leiden Network with modularity objective
    let weights = graph.fuzzy_kernel_weights();

    let mut node_degree = vec![0.0f32; n_genes];
    let mut total_edge_weight = 0.0f64;
    for (&(i, j), &w) in graph.edges.iter().zip(weights.iter()) {
        node_degree[i] += w;
        node_degree[j] += w;
        total_edge_weight += w as f64;
    }

    let mut leiden_graph = Graph::with_capacity(n_genes, graph.num_edges());
    for nd in node_degree.iter() {
        leiden_graph.add_node(*nd);
    }
    for (&(i, j), &w) in graph.edges.iter().zip(weights.iter()) {
        leiden_graph.add_edge((i as u32).into(), (j as u32).into(), w);
    }
    let network = Network::new_from_graph(leiden_graph);

    // Scale resolution: modularity γ → CPM resolution = γ / (2m)
    let resolution_scaled = resolution / (2.0 * total_edge_weight);
    log::info!(
        "Modularity resolution={:.4} → scaled={:.6e}",
        resolution,
        resolution_scaled
    );

    // Step 3: Run Leiden with resolution tuning toward target module count
    let seed_val = seed.map(|s| s as usize);

    log::info!(
        "Auto-tuning Leiden resolution to target ~{} modules ...",
        n_modules
    );
    let result = tune_leiden_for_modules(&network, n_genes, n_modules, resolution_scaled, seed_val);

    log::info!(
        "Gene module discovery done: {} modules, sizes: {:?}",
        result.n_clusters,
        {
            let mut sizes = result.cluster_sizes();
            sizes.sort_unstable_by(|a, b| b.cmp(a));
            sizes
        }
    );

    Ok(result.labels)
}

/// Simple cluster result for module discovery.
struct ModuleClusterResult {
    labels: Vec<usize>,
    n_clusters: usize,
}

impl ModuleClusterResult {
    fn cluster_sizes(&self) -> Vec<usize> {
        let mut counts = vec![0; self.n_clusters];
        for &label in &self.labels {
            if label < self.n_clusters {
                counts[label] += 1;
            }
        }
        counts
    }
}

fn run_leiden_modules(
    network: &Network,
    n: usize,
    resolution: f64,
    seed: Option<usize>,
) -> ModuleClusterResult {
    let mut leiden = Leiden::new(resolution, 0.01, seed);
    let mut clustering = SimpleClustering::init_different_clusters(n);

    let max_outer = 10;
    for _iter in 0..max_outer {
        let updated = leiden.iterate(network, &mut clustering);
        if !updated {
            break;
        }
    }

    let labels: Vec<usize> = (0..n).map(|i| clustering.get(i)).collect();
    ModuleClusterResult {
        labels,
        n_clusters: clustering.num_clusters(),
    }
}

fn tune_leiden_for_modules(
    network: &Network,
    n: usize,
    target_k: usize,
    initial_resolution: f64,
    seed: Option<usize>,
) -> ModuleClusterResult {
    let mut lo = 1e-6_f64;
    let mut hi = 10.0_f64;
    let mut best = run_leiden_modules(network, n, initial_resolution, seed);
    let mut best_res = initial_resolution;

    log::info!(
        "  resolution={:.6} → {} modules (target {})",
        initial_resolution,
        best.n_clusters,
        target_k
    );

    if best.n_clusters == target_k {
        return best;
    }

    if best.n_clusters > target_k {
        hi = initial_resolution;
    } else {
        lo = initial_resolution;
    }

    const MAX_SEARCH: usize = 20;

    for step in 0..MAX_SEARCH {
        let mid = (lo + hi) / 2.0;
        let result = run_leiden_modules(network, n, mid, seed);

        log::info!(
            "  step {}: resolution={:.6} → {} modules",
            step + 1,
            mid,
            result.n_clusters
        );

        if result.n_clusters > target_k {
            hi = mid;
        } else {
            lo = mid;
        }

        let cur_diff = (result.n_clusters as isize - target_k as isize).unsigned_abs();
        let best_diff = (best.n_clusters as isize - target_k as isize).unsigned_abs();
        if cur_diff < best_diff {
            best = result;
            best_res = mid;
        }

        if best.n_clusters == target_k || (hi - lo) / hi.max(1e-10) < 1e-4 {
            break;
        }
    }

    log::info!(
        "  best resolution={:.6} → {} modules (target {})",
        best_res,
        best.n_clusters,
        target_k
    );

    best
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create synthetic gene profiles with planted module structure.
    fn make_synthetic_profiles(
        n_genes: usize,
        n_topics: usize,
        n_modules: usize,
    ) -> (GeneProfileStore, Vec<usize>) {
        let mut profiles = vec![0.0f32; n_genes * n_topics];
        let mut labels = vec![0usize; n_genes];

        for g in 0..n_genes {
            let c = g % n_modules;
            labels[g] = c;
            for k in 0..n_topics {
                let base_val = 1.0;
                let signal = if k % n_modules == c { 5.0 } else { 0.0 };
                profiles[g * n_topics + k] = base_val + signal;
            }
        }

        let size_factors: Vec<f32> = (0..n_genes)
            .map(|g| {
                let row = &profiles[g * n_topics..(g + 1) * n_topics];
                row.iter().sum()
            })
            .collect();

        (
            GeneProfileStore {
                profiles,
                size_factors,
                n_genes,
                n_topics,
            },
            labels,
        )
    }

    #[test]
    fn test_from_profiles_basic() {
        let (store, labels) = make_synthetic_profiles(20, 6, 3);
        let stats = GeneModuleStats::from_profiles(&store, 3, &labels);

        assert_eq!(stats.m, 3);
        assert_eq!(stats.k, 6);
        assert_eq!(stats.n_genes, 20);

        for c in 0..3 {
            assert!(stats.module_count[c] > 0);
        }
        let total_count: usize = stats.module_count.iter().sum();
        assert_eq!(total_count, 20);
    }

    #[test]
    fn test_delta_move_consistency() {
        let (store, labels) = make_synthetic_profiles(30, 4, 3);
        let mut stats = GeneModuleStats::from_profiles(&store, 3, &labels);

        let old_c = stats.membership[0];
        let new_c = (old_c + 1) % 3;
        stats.delta_move(0, old_c, new_c, &store);

        let stats_recomputed = GeneModuleStats::from_profiles(&store, 3, &stats.membership);
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
        let mut stats = GeneModuleStats::from_profiles(&store, 2, &labels);

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
        let stats = GeneModuleStats::from_profiles(&store, 3, &labels);

        let a0 = 1.0;
        let b0 = 1.0;
        let score_before = stats.total_score(a0, b0);

        let mut log_probs = vec![0.0f64; 3];

        for g in 0..stats.n_genes {
            let current_c = stats.membership[g];

            compute_log_probs_for_gene(g, &stats, &store, a0, b0, &mut log_probs);

            assert!(
                log_probs[current_c].abs() < 1e-10,
                "g={}: log_prob for current module should be 0, got {}",
                g,
                log_probs[current_c]
            );

            for t in 0..3 {
                if t == current_c {
                    continue;
                }

                let mut stats_moved = GeneModuleStats::from_profiles(&store, 3, &stats.membership);
                stats_moved.delta_move(g, current_c, t, &store);
                let score_after = stats_moved.total_score(a0, b0);
                let expected_delta = score_after - score_before;

                assert!(
                    (log_probs[t] - expected_delta).abs() < 1e-8,
                    "g={}, t={}: computed={:.10}, expected={:.10}",
                    g,
                    t,
                    log_probs[t],
                    expected_delta
                );
            }
        }
    }

    #[test]
    fn test_log_beta_columns_sum_to_one() {
        let (store, labels) = make_synthetic_profiles(20, 4, 3);
        let stats = GeneModuleStats::from_profiles(&store, 3, &labels);
        let within = WithinModuleWeights::from_marginal_expression(&store.size_factors, &labels, 3);

        let decoder = TreeTopicDecoder {
            stats,
            profiles: store,
            within_weights: within,
            a0: 1.0,
            b0: 1.0,
            rng: SmallRng::seed_from_u64(42),
            parallel_seed: 0,
        };

        let log_beta = decoder.log_beta_gk();
        let n_genes = decoder.n_genes();
        let k = decoder.n_topics();

        // For each topic k, sum exp(log_beta[g,k]) over all genes should ≈ 1
        for t in 0..k {
            let mut sum = 0.0f64;
            for g in 0..n_genes {
                sum += (log_beta[g * k + t] as f64).exp();
            }
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "Topic {}: sum of probabilities = {} (expected ~1.0)",
                t,
                sum
            );
        }
    }

    #[test]
    fn test_gibbs_convergence_on_planted_data() {
        // Create data with clear module structure
        let n_genes = 60;
        let n_topics = 6;
        let n_modules = 3;

        let (store, true_labels) = make_synthetic_profiles(n_genes, n_topics, n_modules);

        // Start with random labels
        let mut decoder = TreeTopicDecoder::new(store, n_modules, 1.0, 1.0, 123);

        let score_before = decoder.total_score();

        // Run Gibbs
        decoder.gibbs_sweep(10);
        decoder.greedy_sweep(10);

        let score_after = decoder.total_score();

        // Score should improve
        assert!(
            score_after >= score_before - 1e-6,
            "Score should not decrease: before={}, after={}",
            score_before,
            score_after
        );

        // Check that genes in same true module tend to be in same inferred module
        // (modular recovery — at least same-module genes should cluster together)
        let membership = decoder.membership();
        let mut same_module_agreement = 0;
        let mut same_module_total = 0;
        for g1 in 0..n_genes {
            for g2 in (g1 + 1)..n_genes {
                if true_labels[g1] == true_labels[g2] {
                    same_module_total += 1;
                    if membership[g1] == membership[g2] {
                        same_module_agreement += 1;
                    }
                }
            }
        }

        let agreement_rate = same_module_agreement as f64 / same_module_total as f64;
        assert!(
            agreement_rate > 0.8,
            "Same-module agreement rate too low: {:.3} ({}/{})",
            agreement_rate,
            same_module_agreement,
            same_module_total
        );
    }

    #[test]
    fn test_from_dense_profiles() {
        // 3 cells, 4 genes, 2 topics
        let x_ng = vec![
            1.0, 0.0, 2.0, 0.0, // cell 0
            0.0, 3.0, 0.0, 1.0, // cell 1
            2.0, 1.0, 0.0, 0.0, // cell 2
        ];
        let theta_nk = vec![
            0.7, 0.3, // cell 0
            0.2, 0.8, // cell 1
            0.5, 0.5, // cell 2
        ];

        let store = GeneProfileStore::from_dense(&x_ng, &theta_nk, 3, 4, 2);
        assert_eq!(store.n_genes, 4);
        assert_eq!(store.n_topics, 2);

        // profile[0, :] = 1.0*[0.7,0.3] + 0.0*[0.2,0.8] + 2.0*[0.5,0.5] = [1.7, 1.3]
        let p0 = store.profile(0);
        assert!((p0[0] - 1.7).abs() < 1e-5);
        assert!((p0[1] - 1.3).abs() < 1e-5);
    }

    #[test]
    fn test_within_module_weights_sum_to_one_per_module() {
        let size_factors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let membership = vec![0, 0, 1, 1, 2, 2];
        let n_modules = 3;

        let w =
            WithinModuleWeights::from_marginal_expression(&size_factors, &membership, n_modules);

        // Within each module, weights should sum to ~1
        for m in 0..n_modules {
            let mut sum = 0.0f64;
            for g in 0..6 {
                if membership[g] == m {
                    sum += (w.log_weights[g] as f64).exp();
                }
            }
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Module {}: within-module weight sum = {}",
                m,
                sum
            );
        }
    }

    #[test]
    fn test_parallel_gibbs_produces_valid_assignments() {
        let n_genes = 60;
        let n_topics = 4;
        let n_modules = 3;
        let (store, _) = make_synthetic_profiles(n_genes, n_topics, n_modules);

        let mut decoder = TreeTopicDecoder::new(store, n_modules, 1.0, 1.0, 42);
        let moves = decoder.gibbs_sweep_parallel(5);

        // Should have some moves
        assert!(moves > 0, "Parallel Gibbs should make some moves");

        // All assignments should be valid
        for &m in decoder.membership() {
            assert!(m < n_modules);
        }
    }

    #[test]
    fn test_discover_gene_modules_planted_structure() {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        // Create gene expression data with 3 planted modules of 30 genes each.
        // Genes in the same module have correlated expression across 20 samples.
        let n_genes = 90;
        let n_samples = 20;
        let n_modules = 3;
        let genes_per_module = n_genes / n_modules;

        let mut rng = SmallRng::seed_from_u64(42);
        let noise = Normal::new(0.0f32, 0.3).unwrap();
        let signal = Normal::new(0.0f32, 2.0).unwrap();

        // gene_mat is [G × S]
        let mut gene_mat = DMatrix::<f32>::zeros(n_genes, n_samples);

        // For each module, generate a shared signal pattern across samples
        let mut module_patterns = Vec::new();
        for _m in 0..n_modules {
            let pattern: Vec<f32> = (0..n_samples).map(|_| signal.sample(&mut rng)).collect();
            module_patterns.push(pattern);
        }

        // Each gene gets its module's pattern + noise
        let mut true_labels = vec![0usize; n_genes];
        for g in 0..n_genes {
            let m = g / genes_per_module;
            true_labels[g] = m;
            for s in 0..n_samples {
                gene_mat[(g, s)] = module_patterns[m][s] + noise.sample(&mut rng);
            }
        }

        // Discover modules
        let labels = discover_gene_modules(&gene_mat, n_modules, 10, 1.0, Some(42)).unwrap();

        assert_eq!(labels.len(), n_genes);

        // Check: genes within the same planted module should mostly share a label
        let mut same_module_agreement = 0;
        let mut same_module_total = 0;
        for g1 in 0..n_genes {
            for g2 in (g1 + 1)..n_genes {
                if true_labels[g1] == true_labels[g2] {
                    same_module_total += 1;
                    if labels[g1] == labels[g2] {
                        same_module_agreement += 1;
                    }
                }
            }
        }

        let agreement_rate = same_module_agreement as f64 / same_module_total as f64;
        assert!(
            agreement_rate > 0.7,
            "Same-module agreement rate too low: {:.3} ({}/{})",
            agreement_rate,
            same_module_agreement,
            same_module_total
        );
    }

    #[test]
    fn test_from_modules_constructor() {
        let (store, labels) = make_synthetic_profiles(30, 4, 3);
        let decoder = TreeTopicDecoder::from_modules(store, labels.clone(), 3, 1.0, 1.0, 42);

        assert_eq!(decoder.n_modules(), 3);
        assert_eq!(decoder.n_topics(), 4);
        assert_eq!(decoder.n_genes(), 30);
        assert_eq!(decoder.membership(), &labels);

        // Score should be computable
        let score = decoder.total_score();
        assert!(score.is_finite(), "Score should be finite, got {}", score);
    }
}
