//! Recast a finished topic model into a Euclidean metric space.
//!
//! Given **frozen** cell topic proportions θ ∈ ℝ^{N×K}, learn a shared cell +
//! gene embedding against the raw counts:
//!
//! ```text
//! α ∈ ℝ^{K×H}   topic embedding
//! ρ ∈ ℝ^{D×H}   gene embedding
//! b ∈ ℝ^D       gene bias
//! Z = θ·α       cell embedding, derived — never a free parameter
//!
//! score(cell c, gene g) = (θ_c·α)·ρ_g + b_g
//! ℓ = log σ(score_pos) + Σ_neg log σ(−score_neg)
//! ```
//!
//! # Why a topic run needs this at all
//!
//! A topic model's β is **multinomial loadings**, not an embedding: the
//! per-topic gene distributions live on a simplex, and inner products between
//! a cell and a gene in that parameterization do not mean anything metric. So
//! marker→type annotation by projection — which needs cells and genes in ONE
//! inner-product space — cannot consume a raw topic run. Fitting α/ρ/b under
//! this bipartite NCE builds that space around the topic result without
//! disturbing it: θ is a constant here, so the cell-side answer the topic
//! model produced is preserved exactly and only the geometry is learned.
//!
//! `H` defaults to `K` at the call site but is a free knob and may exceed it —
//! that extra room is the count-based payoff over a closed-form SVD of β.
//!
//! # Co-embedding is the caller's job, and it is not optional
//!
//! The learned ρ fans out **off** the K-archetype cell simplex (cells live in
//! the convex hull of α), so a joint UMAP of Z and ρ separates genes from
//! cells into two disjoint clouds and every cell↔gene distance is meaningless.
//! Callers must follow training with the SIMBA `si.tl.embed` transform
//! ([`crate::postprocess::feature_coembedding`]), which re-places each gene at
//! the softmax-over-cells weighted average of the *cell* embeddings, landing it
//! on the cell manifold. Cells are the reference and are unchanged, and
//! training is untouched — it is a post-hoc transform of the finished ρ.
//!
//! # Shape
//!
//! File-free by construction: the caller builds [`RestTrainInputs`] from
//! whatever it already has in memory, matching this module's neighbours in
//! [`crate::fit`]. `senna resolve-embedding-space` reads them off disk;
//! `faba gem-encoder` hands over the θ and counts from the run that just
//! finished, with no round-trip.

use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use log::info;
use matrix_util::traits::ConvertMatOps;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

type Mat = nalgebra::DMatrix<f32>;

/// Pre-built, file-free training inputs.
pub struct RestTrainInputs {
    /// θ aligned to the counts' cells: `[N, K]`, row-stochastic and frozen.
    pub theta_aligned: Mat,
    /// Positive-edge gene ids, `[E]`.
    pub edge_gene: Vec<u32>,
    /// Positive-edge cell ids (row in `theta_aligned`), `[E]`.
    pub edge_cell: Vec<u32>,
    /// Positive-edge weights (counts), `[E]`.
    pub edge_w: Vec<f32>,
    /// Per-gene marginal `Σ count`, `[D]` — drives the `marginal^α` negatives.
    pub gene_marginal: Vec<f64>,
    pub n_genes: usize,
}

pub struct RestConfig<'a> {
    pub embedding_dim: usize,
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub neg_alpha: f32,
    pub seed: u64,
    pub dev: &'a Device,
    /// First Ctrl-C sets this; the loop finalizes outputs from the current
    /// parameter state. Caller installs the SIGINT handler before training.
    pub stop: Arc<AtomicBool>,
}

/// Trained host-side bundle.
pub struct TrainedRest {
    /// `[D, H]` gene embedding — **raw ρ, off the cell manifold.** Run
    /// [`crate::postprocess::feature_coembedding`] before writing it anywhere
    /// a reader will compare genes to cells.
    pub rho: Mat,
    /// `[K, H]` topic embedding.
    pub alpha: Mat,
    /// `[N, H]` cell embedding `Z = θ·α`.
    pub z: Mat,
    /// `[D]` per-gene bias.
    pub b_gene: Vec<f32>,
    pub loss_trace: Vec<f32>,
}

/// Fit α/ρ/b against frozen θ under bipartite cell–gene NCE.
pub fn train_rest(
    inputs: &RestTrainInputs,
    config: &RestConfig<'_>,
) -> anyhow::Result<TrainedRest> {
    let RestTrainInputs {
        theta_aligned,
        edge_gene,
        edge_cell,
        edge_w,
        gene_marginal,
        n_genes,
    } = inputs;
    let d = *n_genes;
    let k = theta_aligned.ncols();
    let n = theta_aligned.nrows();
    let h = config.embedding_dim;
    anyhow::ensure!(!edge_gene.is_empty(), "rest: zero positive edges");
    let n_edges = edge_gene.len();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, config.dev);
    let randn = Init::Randn {
        mean: 0.0,
        stdev: 0.1,
    };
    let alpha = vb.get_with_hints((k, h), "alpha", randn)?;
    let rho = vb.get_with_hints((d, h), "rho", randn)?;
    let b_gene = vb.get_with_hints((d,), "b_gene", Init::Const(0.0))?;

    // Frozen θ as a constant device tensor [N, K]. `.contiguous()` because
    // `to_tensor` of a column-major nalgebra matrix is non-contiguous and
    // `index_select` (the per-batch cell gather) requires contiguous sources.
    let theta_t = theta_aligned.to_tensor(config.dev)?.contiguous()?;

    // f64, NOT f32 — `WeightedIndex` builds a running cumulative sum, and in f32
    // that sum SATURATES at 2^24 ≈ 1.68e7: once the total reaches it, adding a
    // typical count of 1–10 no longer changes the value, every later entry
    // stores the same number, and `partition_point` can never return an index
    // at or past that point. Measured on 40 M unit-weight edges, an f32 sampler
    // reached only 41.9 % of them and an f64 sampler reached 100 %. A single
    // 15 k-cell run is already past the limit on total UMIs, so this is the
    // normal case, not an edge case — and it fails SILENTLY, with a healthy
    // decreasing loss while ρ is fit on a prefix of the cells.
    // Fed as an ITERATOR: `WeightedIndex::new` builds its own cumulative `Vec`,
    // so collecting the widened weights first would hold a second full-length
    // f64 copy of the edge list (≈560 MB at 70 M edges) for nothing.
    let pos_picker = WeightedIndex::new(edge_w.iter().map(|&w| f64::from(w)))
        .map_err(|e| anyhow::anyhow!("positive weights: {e}"))?;
    let neg_weights: Vec<f64> = gene_marginal
        .iter()
        .map(|&m| m.max(1.0).powf(f64::from(config.neg_alpha)))
        .collect();
    let neg_picker =
        WeightedIndex::new(&neg_weights).map_err(|e| anyhow::anyhow!("negative weights: {e}"))?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )?;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut loss_trace = Vec::with_capacity(config.epochs);

    info!(
        "rest train: N={n}, D={d}, K={k}, H={h}, edges={n_edges}, {}×{} per epoch, {} negs, lr={}, α={}",
        config.batches_per_epoch,
        config.batch_size,
        config.num_negatives,
        config.learning_rate,
        config.neg_alpha
    );

    'epochs: for epoch in 0..config.epochs {
        let mut epoch_loss = 0f32;
        let mut n_steps = 0usize;
        for _ in 0..config.batches_per_epoch {
            if config.stop.load(Ordering::Relaxed) {
                break;
            }
            let (ci, gp, gn) = sample_batch(
                edge_cell,
                edge_gene,
                config.batch_size,
                config.num_negatives,
                &pos_picker,
                &neg_picker,
                &mut rng,
            );
            let loss = step(
                &theta_t,
                &alpha,
                &rho,
                &b_gene,
                &ci,
                &gp,
                &gn,
                config.num_negatives,
                config.dev,
            )?;
            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()?;
            n_steps += 1;
        }
        let avg = epoch_loss / n_steps.max(1) as f32;
        loss_trace.push(avg);
        if epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == config.epochs {
            info!("epoch {}/{}: loss={:.4}", epoch + 1, config.epochs, avg);
        }
        if config.stop.load(Ordering::SeqCst) {
            info!(
                "Stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                config.epochs
            );
            break 'epochs;
        }
    }

    let rho_host = Mat::from_tensor(&rho)?;
    let alpha_host = Mat::from_tensor(&alpha)?;
    let z = theta_aligned * &alpha_host;
    let b_host: Vec<f32> = b_gene.to_vec1()?;

    Ok(TrainedRest {
        rho: rho_host,
        alpha: alpha_host,
        z,
        b_gene: b_host,
        loss_trace,
    })
}

/// Draw `B` positives (∝ count) + `B*K` negatives (∝ marginal^α). Returns
/// `(cell_idx, gene_pos, gene_neg_flat)` as `Vec<u32>`. Negative collisions
/// with the positive gene are rare and harmless (ignored, like fne).
fn sample_batch(
    edge_cell: &[u32],
    edge_gene: &[u32],
    batch_size: usize,
    num_negatives: usize,
    pos_picker: &WeightedIndex<f64>,
    neg_picker: &WeightedIndex<f64>,
    rng: &mut StdRng,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let b = batch_size;
    let k = num_negatives;
    let mut ci = Vec::with_capacity(b);
    let mut gp = Vec::with_capacity(b);
    let mut gn = Vec::with_capacity(b * k);
    for _ in 0..b {
        let e = pos_picker.sample(rng);
        ci.push(edge_cell[e]);
        gp.push(edge_gene[e]);
        for _ in 0..k {
            gn.push(neg_picker.sample(rng) as u32);
        }
    }
    (ci, gp, gn)
}

#[allow(clippy::too_many_arguments)]
fn step(
    theta_t: &Tensor, // [N, K]
    alpha: &Tensor,   // [K, H]
    rho: &Tensor,     // [D, H]
    b_gene: &Tensor,  // [D]
    ci: &[u32],       // [B] cell ids
    gp: &[u32],       // [B] positive gene ids
    gn: &[u32],       // [B*K] negative gene ids
    k: usize,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let b = ci.len();
    let bk = gn.len();
    let ci_t = Tensor::from_slice(ci, b, dev)?;
    let gp_t = Tensor::from_slice(gp, b, dev)?;
    let gn_t = Tensor::from_slice(gn, bk, dev)?;

    // Z_b = θ_b · α  [B, H] — computed once and reused for the negatives
    // (which share the same B cells), so the negative side is a broadcast
    // dot, not a second θ-gather + matmul.
    let z_b = theta_t.index_select(&ci_t, 0)?.matmul(alpha)?;
    let h = z_b.dim(1)?;
    let rho_pos = rho.index_select(&gp_t, 0)?; // [B, H]
    let b_pos = b_gene.index_select(&gp_t, 0)?; // [B]
    let pos_score = ((&z_b * &rho_pos)?.sum(1)? + &b_pos)?; // [B]

    // Negatives reuse z_b: score_neg[i,j] = z_b[i] · ρ_{neg[i,j]} + b_{neg[i,j]}.
    let rho_neg = rho.index_select(&gn_t, 0)?.reshape((b, k, h))?; // [B, K, H]
    let b_neg = b_gene.index_select(&gn_t, 0)?.reshape((b, k))?; // [B, K]
    let neg_dot = z_b.unsqueeze(1)?.broadcast_mul(&rho_neg)?.sum(2)?; // [B, K]
    let neg_score = (neg_dot + b_neg)?; // [B, K]

    // The crate's own bipartite NCE loss, one module over — same
    // `−(log σ(pos) + Σ log σ(−neg))`, and going through it keeps this trainer
    // on the objective every other feature-side loss here uses.
    Ok(crate::loss::logistic_nce(&pos_score, std::slice::from_ref(&neg_score))?.mean(0)?)
}

#[cfg(test)]
#[path = "resolve_embedding_tests.rs"]
mod tests;
