//! Per-modality log-rate assembly and Poisson sampling.
//!
//! Reuses `multiome/sample.rs::sample_poisson_from_logits` which softmaxes
//! per-cell columns and scales by `depth_j` — that's what we want here
//! since the simulator targets per-modality library sizes, with all of
//! `α`, `μ`, `r` contributing relative per-row weights.

use nalgebra::DMatrix;

use super::latents::Latents;
use super::MODALITIES;
use crate::multiome::sample_poisson_from_logits;

const EPS_LOG: f32 = 1e-12;
const NEG_BIG: f32 = -50.0;

/// Sparse count entry: `(row_id, cell_id, count)`.
pub type Triplet = (u64, u64, f32);
/// Maps a modifier-modality `row_id` to `(gene_idx, component_idx)`.
pub type RowKey = (usize, usize);

/// Inputs shared by every per-modality sampler: the latents, the
/// precomputed modality-invariant `log((β_topic · θ))` matrix, and the
/// per-cell batch assignment. Bundling these keeps the per-call signatures
/// small and avoids re-threading the same three references everywhere.
pub struct RateContext<'a> {
    pub lats: &'a Latents,
    pub log_topic: &'a DMatrix<f32>,
    pub batch_membership: &'a [usize],
}

/// Precompute the modality-invariant part of `log μ`:
/// `log_topic[g, j] = log((β_topic · θ)_{g, j})`. This [G × N] matrix
/// is shared across all per-modality sampling calls; the modality- and
/// gene-specific additive terms (β_g and δ_{g, B(j)}) are folded in
/// inside `assemble_log_mu`.
pub fn precompute_log_topic(lats: &Latents) -> DMatrix<f32> {
    let lambda = &lats.beta_topic_gk * &lats.theta_kn;
    lambda.map(|v| v.max(EPS_LOG).ln())
}

/// Sample triplets for the count modality (M=0). Rows:
///   row 0 .. G   : `{gene_i}/count/spliced`
///   row G .. 2G  : `{gene_i}/count/unspliced`
///
/// The log-rate is `log α_{g,c} + log μ_{g,j}` with
/// `log μ_{g,j} = β_g + log_topic[g, j] + δ_count_{g, B(j)}`.
/// `sample_poisson_from_logits` softmaxes this per cell and scales by
/// `depth_count`, so library size targets `depth_count`.
pub fn sample_count_modality(
    ctx: &RateContext,
    ln_delta_count: &DMatrix<f32>,
    depth_count: usize,
    rseed: u64,
) -> Vec<Triplet> {
    let lats = ctx.lats;
    let g = lats.beta_g.len();
    let n = lats.theta_kn.ncols();
    let depths = vec![depth_count as f32; n];

    let log_mu = assemble_log_mu(ctx, ln_delta_count);

    // Stack [G × N] for spliced (c=0) and unspliced (c=1) vertically.
    let mut log_rate = DMatrix::<f32>::zeros(2 * g, n);
    for c in 0..2 {
        for gi in 0..g {
            let row = c * g + gi;
            let log_alpha = (lats.alpha_per_mod[0][(c, gi)] + EPS_LOG).ln();
            for j in 0..n {
                log_rate[(row, j)] = log_alpha + log_mu[(gi, j)];
            }
        }
    }

    sample_poisson_from_logits(&log_rate, &depths, rseed)
}

/// Sample triplets for a modifier modality (M ≥ 1). Returns:
/// - triplets `(row_id, cell_id, count)`
/// - `row_keys` mapping `row_id → (gene_idx, component_idx)` so the
///   output writer can build the correct `gene/modality/component_c`
///   row names.
///
/// Rows are emitted only for genes where `φ_{g, m} = 1` AND NOT
/// `held_out(g, m)`. Within each emitted gene, `C_m` consecutive rows
/// (one per component) are produced.
pub fn sample_modifier_modality(
    ctx: &RateContext,
    m_idx: usize,
    held_out: &[Vec<bool>],
    ln_delta_m: &DMatrix<f32>,
    depth_m: usize,
    rseed: u64,
) -> (Vec<Triplet>, Vec<RowKey>) {
    let lats = ctx.lats;
    let g = lats.beta_g.len();
    let n = lats.theta_kn.ncols();
    let c_m = lats.alpha_per_mod[m_idx].nrows();

    // Pick the substrate-positive, non-held-out genes.
    let emit_genes: Vec<usize> = (0..g)
        .filter(|&gi| lats.phi[m_idx][gi] && !held_out[m_idx][gi])
        .collect();
    let d = emit_genes.len() * c_m;
    let row_keys: Vec<RowKey> = emit_genes
        .iter()
        .flat_map(|&gi| (0..c_m).map(move |c| (gi, c)))
        .collect();

    if d == 0 {
        log::warn!(
            "modality '{}' has 0 emitted rows (all substrate-negative or held-out)",
            MODALITIES[m_idx]
        );
        return (Vec::new(), row_keys);
    }

    let log_mu = assemble_log_mu(ctx, ln_delta_m);
    let log_r = build_log_modifier_rate(ctx, m_idx, ln_delta_m);

    // Assemble log_rate [D × N] = log α + log μ + log r for each (g, c).
    let mut log_rate = DMatrix::<f32>::zeros(d, n);
    let alpha_m = &lats.alpha_per_mod[m_idx];
    for (row_id, &(gi, ci)) in row_keys.iter().enumerate() {
        let log_alpha = (alpha_m[(ci, gi)] + EPS_LOG).ln();
        for j in 0..n {
            // φ = 1 by construction (we only emit substrate-positive rows),
            // so log_r already contains the full program-driven term.
            let v = log_alpha + log_mu[(gi, j)] + log_r[(gi, j)];
            log_rate[(row_id, j)] = if v.is_finite() { v } else { NEG_BIG };
        }
    }

    let depths = vec![depth_m as f32; n];
    let triplets = sample_poisson_from_logits(&log_rate, &depths, rseed);

    (triplets, row_keys)
}

/// `log μ_{g, j} = β_g + log_topic[g, j] + δ_m_{g, B(j)}`.
/// `log_topic` is the modality-invariant `log((β_topic · θ))` matrix
/// from [`precompute_log_topic`]; this routine only folds in the
/// per-gene baseline and per-modality batch effect, so the expensive
/// [G × K] · [K × N] matmul runs once per `run_faba`, not once per
/// modality. Returned shape: `[G × N]`.
fn assemble_log_mu(ctx: &RateContext, ln_delta: &DMatrix<f32>) -> DMatrix<f32> {
    let lats = ctx.lats;
    let g = lats.beta_g.len();
    let n = lats.theta_kn.ncols();
    let bb = ln_delta.ncols();

    let mut log_mu = ctx.log_topic.clone();
    for j in 0..n {
        let b = ctx.batch_membership[j];
        for gi in 0..g {
            let delta = if bb > 1 { ln_delta[(gi, b)] } else { 0.0 };
            log_mu[(gi, j)] += lats.beta_g[gi] + delta;
        }
    }
    log_mu
}

/// `log r_{g, m, j} = base_{g, m} + φ_{g, m} · Σ_k z_{g, k}·A_{m, k}·θ_{k, j}
///                  + δ_m_{g, B(j)}`. Returned shape: `[G × N]`.
///
/// For substrate-negative g the program term is gated to zero by φ;
/// downstream callers only emit rows for substrate-positive g so this
/// gate is mainly a safety net.
fn build_log_modifier_rate(
    ctx: &RateContext,
    m_idx: usize,
    ln_delta: &DMatrix<f32>,
) -> DMatrix<f32> {
    let lats = ctx.lats;
    let g = lats.beta_g.len();
    let n = lats.theta_kn.ncols();
    let k_prog = lats.a_mk.ncols();
    let bb = ln_delta.ncols();

    // z .* A[m, :].broadcast — [G × K_prog].
    let mut z_scaled = lats.z_gk.clone();
    for gi in 0..g {
        for k in 0..k_prog {
            z_scaled[(gi, k)] *= lats.a_mk[(m_idx, k)];
        }
    }
    let prog = z_scaled * &lats.theta_kn; // [G × N]

    let mut log_r = DMatrix::<f32>::zeros(g, n);
    for j in 0..n {
        let b = ctx.batch_membership[j];
        for gi in 0..g {
            let phi = if lats.phi[m_idx][gi] { 1.0 } else { 0.0 };
            let delta = if bb > 1 { ln_delta[(gi, b)] } else { 0.0 };
            log_r[(gi, j)] = lats.base_gm[(gi, m_idx)] + phi * prog[(gi, j)] + delta;
        }
    }
    log_r
}
