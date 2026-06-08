//! Per-gene NB-Fisher down-weighting of the count (housekeeping) axis.
//!
//! Mirrors `senna bge / topic`: high-mean / high-dispersion genes
//! (ribosomal, housekeeping, library-size drivers) get a bounded `(0, 1]`
//! Fisher weight
//!
//!     w_g = 1 / (1 + π_g · s̄ · φ(μ_g))
//!
//! from the NB mean-variance trend (`data_beans_alg::nb_dispersion`).
//! `w_g → 1` in the Poisson limit (φ → 0) and shrinks toward 0 for genes
//! whose abundance × overdispersion dominates the library.
//!
//! Why here and not in the loss: faba's sampler draws count-component
//! positives **count-proportionally** (τ=1 by default), so a handful of
//! the highest-count housekeeping genes monopolise the gradient into the
//! shared program loadings `z` — the redundancy-collapse seen at K=8. We
//! fold `w_g^penalty` into the count-based **anchor** sampling pools
//! (agg + count-comp) so those genes are simply drawn less often. The
//! m6A / modifier pools are left untouched — that signal is the point of
//! the model and is already modality-balanced via `--tau-modality`.
//!
//! This module also computes per-gene **ubiquity** (`ubiquity_from_count_pool`):
//! the fraction of cells expressing a gene — a *breadth* signal (distinct
//! from Fisher's *magnitude*). Written to `{out}.ubiquity.parquet` as a
//! diagnostic / inverse-propensity signal; not currently fed back into the
//! model.

use data_beans_alg::nb_dispersion::DispersionTrend;

use super::pseudobulk::StratumPool;

/// Per-gene NB-Fisher weights derived from the **cell-axis** count-comp
/// pool. Returns a length-`n_genes` vector in `(0, 1]`; genes with no
/// count mass get `w_g = 1` (nothing to penalise). `n_cells` is the
/// per-gene mean divisor (zero-count cells enter through it, since the
/// pool only stores nonzero `(gene, cell)` entries).
pub fn fisher_weights_from_count_pool(
    count_comp: &StratumPool,
    n_genes: usize,
    n_cells: usize,
) -> Vec<f32> {
    // Each count-comp entry is one (gene, cell) total (spliced+unspliced
    // already summed in `aggregate_pools`), so per-gene Σc and Σc² over
    // entries give the per-cell mean/variance ingredients directly.
    let mut sum = vec![0.0_f64; n_genes];
    let mut sumsq = vec![0.0_f64; n_genes];
    for i in 0..count_comp.len() {
        let g = count_comp.gene_ids[i] as usize;
        let c = count_comp.counts[i] as f64;
        sum[g] += c;
        sumsq[g] += c * c;
    }

    let n = n_cells.max(1) as f64;
    let means: Vec<f32> = sum.iter().map(|&s| (s / n) as f32).collect();
    let vars: Vec<f32> = (0..n_genes)
        .map(|g| {
            let m = sum[g] / n;
            ((sumsq[g] / n) - m * m).max(0.0) as f32
        })
        .collect();

    let trend = DispersionTrend::fit(&means, &vars);
    let total_mass: f64 = sum.iter().sum();
    let avg_s = (total_mass / n) as f32;
    let inv_total = if total_mass > 0.0 {
        1.0 / total_mass as f32
    } else {
        0.0
    };

    (0..n_genes)
        .map(|g| {
            let pi = sum[g] as f32 * inv_total;
            trend.fisher_weight(pi, avg_s, means[g])
        })
        .collect()
}

/// Per-gene **ubiquity** `u_g ∈ (0, 1]` = fraction of cells expressing
/// gene `g`, derived from the **cell-axis** count-comp pool. Each pool
/// entry is one nonzero `(gene, cell)` total (`aggregate_pools` keys on
/// `(gene_id, axis_id)`), so the number of entries per gene is exactly the
/// count of cells with nonzero expression. Divided by `n_cells`.
///
/// Unlike the NB-Fisher weight (which keys on mean × overdispersion and
/// therefore also penalises lineage-restricted *high-mean* markers such as
/// erythroid Hb), ubiquity separates genes that are high *everywhere*
/// (true housekeeping: MT-/RP-, u ≈ 1) from genes that are high in *one*
/// lineage (Hb: u ≈ 0.1). Written to `{out}.ubiquity.parquet` as a
/// diagnostic / inverse-propensity signal (a *breadth* complement to the
/// NB-Fisher *magnitude* weight); not currently consumed by the model.
pub fn ubiquity_from_count_pool(
    count_comp: &StratumPool,
    n_genes: usize,
    n_cells: usize,
) -> Vec<f32> {
    let mut cells_with = vec![0u32; n_genes];
    for i in 0..count_comp.len() {
        let g = count_comp.gene_ids[i] as usize;
        if g < n_genes {
            cells_with[g] += 1;
        }
    }
    let inv_n = 1.0 / n_cells.max(1) as f32;
    cells_with
        .iter()
        .map(|&c| (c as f32 * inv_n).clamp(0.0, 1.0))
        .collect()
}

/// Effective per-gene weight from a raw NB-Fisher weight `w_g` and the
/// `--housekeeping-penalty` exponent: `w_g^penalty`, clamped to `(0, 1]`.
/// `penalty = 0` ⇒ `1.0` (off), `1` is full senna-style attenuation,
/// `> 1` is more aggressive. Single source of truth for both the
/// sampler-side pool reweight ([`apply_to_pool`]) and the likelihood-side
/// per-positive loss weight (precomputed per gene by the sampler).
pub fn effective_weight(raw_fisher: f32, penalty: f32) -> f32 {
    if penalty <= 0.0 {
        1.0
    } else {
        raw_fisher.clamp(0.0, 1.0).powf(penalty)
    }
}

/// Fold per-gene Fisher weights into a count-based anchor pool's sampling
/// weights, in place: `weight_i ← weight_i · w_{gene_i}^penalty`. Weights
/// are renormalised implicitly by `WeightedIndex`, so only the *relative*
/// attenuation matters.
pub fn apply_to_pool(pool: &mut StratumPool, fisher: &[f32], penalty: f32) {
    for i in 0..pool.weights.len() {
        let g = pool.gene_ids[i] as usize;
        pool.weights[i] *= effective_weight(fisher[g], penalty);
    }
}
