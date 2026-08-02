//! cage's batched gene-modulated cell-cell chain NCE.
//!
//! Scores `G` independent [`CellChainBatch`]es — one per gene in the current
//! chunk — in a single forward pass, returning `[G, L]` per-(gene, level)
//! losses. Collapses ~18k tiny device forwards per epoch into ~600 big ones.
//!
//! Per-row score, for gene `g` at chain level `ℓ`:
//!
//! ```text
//! s(u, v) = θ_g · (e_u ⊙ e_v) + b_u + b_v
//! θ_g     = z_g ⊙ e_gene[g]     z ~ Bern(pip), redrawn once per EPOCH
//! ```
//!
//! `e_u ⊙ e_v` is the EDGE embedding: the pair's participation in each latent
//! community, high only when both endpoints load on it. So the score asks
//! whether `u` and `v` agree under a per-gene diagonal metric — an adjacency
//! statistic — where the rank-1 form it replaced scored any two cells lying
//! along `θ_g`, adjacent or not.
//!
//! cage's own loss: geu's chain module scores a rank-1 quadratic form and
//! has no gene-modulated variant left. The per-edge NCE itself is geu's, and
//! selectable — see [`NceObjective`] and `--nce-objective`.

use candle_util::candle_core::{Device, Result as CResult, Tensor};
use graph_embedding_util::loss::{
    gather_feature_rows, logistic_nce, softmax_nce, CellChainBatch, NceObjective,
};
use graph_embedding_util::model::JointEmbedModel;

pub struct CageLossOut {
    /// Per-(gene, level) NCE loss, `[G, L]`.
    pub per_level: Tensor,
    /// Mean `|θ_g · (e_u ⊙ e_v)|`, detached — the collapse detector.
    /// Watch it against the loss: if this decays toward zero while the loss
    /// still falls, the ungated cell biases have taken over the objective and
    /// the gene direction is no longer identified.
    pub mean_abs_pair: Tensor,
}

/// Batched gene-modulated per-level loss. `gene_ids[i]` is the gene for
/// `batches[i]`; all batches must share the same `B`, `L` and `K`.
///
/// Levels differ only in their negative pools — the gene direction is the same
/// at every scale. The only modulation is the epoch's spike draw `z`, applied
/// through geu's `gathered_gate_weights`; there is no per-level gate.
///
/// `objective` picks the per-edge NCE, exactly as geu's feature-side loss does
/// (`graph_embedding_util::loss::feat`). Prefer `Softmax` under a sampled mask:
/// the sampler's profiled-Poisson normalizer is the sampled-softmax estimand,
/// so the gate and the loss then optimize the same thing.
pub fn cage_nce_loss_per_level(
    model: &JointEmbedModel,
    batches: Vec<CellChainBatch>,
    gene_ids: &[u32],
    objective: NceObjective,
    dev: &Device,
) -> CResult<CageLossOut> {
    let g = batches.len();
    assert!(g > 0, "non-empty batches required");
    assert_eq!(
        gene_ids.len(),
        g,
        "gene_ids ({}) and batches ({}) length mismatch",
        gene_ids.len(),
        g
    );
    let b = batches[0].left_cells.len();
    let l = batches[0].per_level_neg.len();
    let k = batches[0].n_negatives;
    for cb in &batches {
        assert_eq!(cb.left_cells.len(), b, "cage loss: B mismatch");
        assert_eq!(cb.per_level_neg.len(), l, "cage loss: L mismatch");
        assert_eq!(cb.n_negatives, k, "cage loss: K mismatch");
    }
    if b == 0 {
        let zero = Tensor::zeros((), candle_util::candle_core::DType::F32, dev)?;
        return Ok(CageLossOut {
            per_level: Tensor::zeros((g, l), candle_util::candle_core::DType::F32, dev)?,
            mean_abs_pair: zero,
        });
    }

    let total_b = g * b;
    let total_neg = total_b * k;

    let mut all_left: Vec<u32> = Vec::with_capacity(total_b);
    let mut all_right: Vec<u32> = Vec::with_capacity(total_b);
    let mut all_neg_per_level: Vec<Vec<u32>> =
        (0..l).map(|_| Vec::with_capacity(total_neg)).collect();
    // Replicate each gene id B times so the gene-side gather lines up
    // row-for-row with the [G*B] cell-side gather.
    let mut gene_repeat: Vec<u32> = Vec::with_capacity(total_b);
    for (cb, &gid) in batches.into_iter().zip(gene_ids.iter()) {
        all_left.extend(cb.left_cells);
        all_right.extend(cb.right_cells);
        for _ in 0..b {
            gene_repeat.push(gid);
        }
        for (lvl_idx, lvl_neg) in cb.per_level_neg.into_iter().enumerate() {
            all_neg_per_level[lvl_idx].extend(lvl_neg);
        }
    }

    let left_idx = Tensor::from_vec(all_left, total_b, dev)?;
    let right_idx = Tensor::from_vec(all_right, total_b, dev)?;
    let gene_idx = Tensor::from_vec(gene_repeat, total_b, dev)?;
    let e_left = model.e_cell.index_select(&left_idx, 0)?;
    let b_left = model.b_cell.index_select(&left_idx, 0)?;
    let e_right = model.e_cell.index_select(&right_idx, 0)?;
    let b_right = model.b_cell.index_select(&right_idx, 0)?;

    // Gene rows with whatever gates them applied — geu's own helper, so cage,
    // `senna bge` and `faba gem` share ONE definition of what multiplies a
    // feature loading. It dispatches on what the model actually carries:
    //
    //   - sampled mask installed: this epoch's `z` rows (or the frozen `pip`
    //     between draws), applied to the raw effect;
    //   - learned gate: `σ(S/τ)` from `s_feat`, plus the reparameterization
    //     draw off `e_feat_logstd`, which is what makes the gate variational;
    //   - neither: a plain gather, ungated.
    //
    // It also reads `e_feat_raw` once a gate is on, so a post-training
    // `materialize_e_feat` cannot gate an already-gated table.
    //
    // Re-deriving this locally is what silently broke the learned arm: passing
    // `logits: None` to `gathered_gate_weights` makes it return `None` unless a
    // pip is installed, so the gate Vars never entered the graph, never got a
    // gradient, and `α` stayed pinned at its init (sd = 0.0000 across 18k
    // genes) while the KL alone nudged the mean.
    let e_gene_l = gather_feature_rows(model, &gene_idx)?; // [G*B, D]

    let mut per_gene_per_level: Vec<Tensor> = Vec::with_capacity(l);
    // Left factor is shared across levels and negatives: the anchor's
    // projection along the gene direction. Only the right factor changes.
    // LINEAR in the gene direction: score the gene against the elementwise
    // product of the two endpoints, `theta_g . (e_a * e_b)`, rather than the
    // product of two separate projections.
    //
    // The rank-1 quadratic form it replaces, `e_a^T (theta theta^T) e_b`, is
    // high for ANY two cells lying along `theta` — including distant ones — so
    // it never tested adjacency at all. `diag(theta)` asks whether a and b
    // AGREE, under a per-gene diagonal metric, which is an adjacency statistic.
    // It also lets a gene say dims should ANTI-align (theta_h < 0), which a
    // rank-1 PSD form cannot express.
    // `θ_g ⊙ e_left` does not change across chain levels or between the
    // positive and its negatives, so it is formed ONCE — a 3x saving on the
    // left factor. It does NOT avoid a `[G*B, K, D]` temporary: the negative
    // branch below still materializes one per level, because `broadcast_as` is
    // a view but the multiply that consumes it is not. A batched
    // `e_neg.matmul(theta_left.unsqueeze(2))` would cut that 64x and hit BLAS;
    // not done yet.
    let theta_left = (&e_gene_l * &e_left)?; // [G*B, D]
    let pair_pos = (&theta_left * &e_right)?.sum(1)?; // [G*B]
                                                      // Collapse detector: the bias-free pair term, so its magnitude is not
                                                      // masked by `b_u + b_v` — which is exactly what would be absorbing the
                                                      // objective if the gate were collapsing.
    let mean_abs_pair = pair_pos.abs()?.mean_all()?.detach();
    let pos_score = ((pair_pos + &b_left)? + &b_right)?;

    for lvl_neg in all_neg_per_level.into_iter() {
        let neg_idx = Tensor::from_vec(lvl_neg, total_neg, dev)?;
        let e_neg_flat = model.e_cell.index_select(&neg_idx, 0)?;
        let b_neg_flat = model.b_cell.index_select(&neg_idx, 0)?;
        let h = e_neg_flat.dim(1)?;
        let e_neg = e_neg_flat.reshape((total_b, k, h))?;
        let b_neg = b_neg_flat.reshape((total_b, k))?;
        // Negatives replace the RIGHT endpoint: the edge embedding becomes
        // `e_u * e_w`, scored against the same gene direction.
        let tl_3d = theta_left.unsqueeze(1)?.broadcast_as((total_b, k, h))?;
        let pair_neg = (&e_neg * &tl_3d)?.sum(2)?; // [G*B, K]
        let b_left_2d = b_left.unsqueeze(1)?.broadcast_as((total_b, k))?;
        let neg_score = ((pair_neg + b_left_2d)? + &b_neg)?;
        let negs = std::slice::from_ref(&neg_score);
        let per_edge = match objective {
            NceObjective::Logistic => logistic_nce(&pos_score, negs)?,
            NceObjective::Softmax => softmax_nce(&pos_score, negs)?,
        };
        let per_edge_gb = per_edge.reshape((g, b))?;
        per_gene_per_level.push(per_edge_gb.mean(1)?); // [G]
    }
    Ok(CageLossOut {
        per_level: Tensor::stack(&per_gene_per_level, 1)?,
        mean_abs_pair,
    })
}
