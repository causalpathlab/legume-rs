//! Per-cell latent inference for `senna cell-embedded-topic`.
//!
//! The cell-embedded encoder pools the genuinely sparse single-cell atoms
//! of a pseudobulk; at inference time every cell becomes its own singleton
//! PB (`cell_to_pb` = identity), so [`pack_eval_minibatch`] packs one
//! ordered minibatch per block and `forward_cells_t` returns the per-cell
//! `log θ`. Block-processed exactly like
//! [`super::eval_indexed::evaluate_latent_by_indexed_encoder`].

use super::common::process_blocks;
use super::eval_indexed::refine_indexed_topic_proportions;
use crate::embed_common::*;

use candle_core::{Device, Tensor};
use candle_util::data::{csc_columns_to_indexed_samples, top_k_indices_weighted, IndexedSample};
use candle_util::data::{pack_eval_minibatch, CellEvalPackArgs};
use candle_util::topic_refinement::TopicRefinementConfig;
use candle_util::traits::{CellEncoderT, IndexedDecoderT};

pub(crate) struct EvaluateCellLatentConfig<'a, Dec> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub fg_context_size: usize,
    pub bg_context_size: usize,
    pub dec_context_size: usize,
    pub decoder: &'a Dec,
    pub refine_config: Option<&'a TopicRefinementConfig>,
    /// Same shortlist weights used during training.
    pub shortlist_weights: &'a [f32],
    /// Per-gene NB-Fisher weights, in training-D order — gathered into the
    /// FG/BG/decoder packs.
    pub feature_fisher_weights: &'a [f32],
}

/// Run the cell-embedded encoder over every cell in `data_vec`, returning
/// the `[N, K]` per-cell `log θ` matrix.
pub(crate) fn evaluate_latent_by_cell_embedded_encoder<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    collapsed: &CollapsedOut,
    config: &EvaluateCellLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: CellEncoderT + Send + Sync,
    Dec: IndexedDecoderT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    // Per-batch background top-K, built once: δ is block-invariant. The
    // batch-null profile (`delta` / `mu_residual` posterior mean,
    // transposed to `[B, D]`) is top-K'd per batch row here; the block fn
    // just fans it out by per-cell membership. Absent (single-batch /
    // --ignore-batch) the block falls back to the observed cell counts.
    let per_batch_bg: Option<Vec<IndexedSample>> = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| -> anyhow::Result<_> {
        let delta_bd = x
            .posterior_mean()
            .to_tensor(config.dev)?
            .transpose(0, 1)?
            .contiguous()?;
        Ok(delta_bd
            .to_vec2::<f32>()?
            .iter()
            .map(|row| {
                let (indices, values) =
                    top_k_indices_weighted(row, config.shortlist_weights, config.bg_context_size);
                IndexedSample { indices, values }
            })
            .collect())
    })
    .transpose()?;

    let block_config = EvaluateCellBlockConfig {
        dev: config.dev,
        adj_method: config.adj_method,
        per_batch_bg: per_batch_bg.as_deref(),
        fg_context_size: config.fg_context_size,
        bg_context_size: config.bg_context_size,
        dec_context_size: config.dec_context_size,
        decoder: config.decoder,
        refine_config: config.refine_config,
        shortlist_weights: config.shortlist_weights,
        feature_fisher_weights: config.feature_fisher_weights,
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_cell_embedded_block(block, data_vec, encoder, &block_config)
    })
}

struct EvaluateCellBlockConfig<'a, Dec> {
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    /// Per-batch δ-null top-K samples, pre-built once (δ is block-invariant).
    /// `None` when no batch correction was fit.
    per_batch_bg: Option<&'a [IndexedSample]>,
    fg_context_size: usize,
    bg_context_size: usize,
    dec_context_size: usize,
    decoder: &'a Dec,
    refine_config: Option<&'a TopicRefinementConfig>,
    shortlist_weights: &'a [f32],
    feature_fisher_weights: &'a [f32],
}

fn evaluate_cell_embedded_block<Enc, Dec>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &Enc,
    config: &EvaluateCellBlockConfig<Dec>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: CellEncoderT,
    Dec: IndexedDecoderT,
{
    let (lb, ub) = block;

    // Sparse `[D, N]` at full D — columns are cells. Every sample set is
    // built straight from the stored nonzeros; no dense `[N, D]` is ever
    // materialized.
    let x_dn = data_vec.read_columns_csc(lb..ub)?;
    let n_features = x_dn.nrows();
    let ncols = x_dn.ncols();

    // FG samples: top-K of the observed counts. (Fit-time eval has no
    // gene remap — the data axis already matches the training axis.)
    let cell_samples = csc_columns_to_indexed_samples(
        &x_dn,
        config.shortlist_weights,
        config.fg_context_size,
        None,
    );
    // Decoder-target is the same observed counts — reuse the FG pack when
    // the contexts match instead of re-scanning the columns.
    let output_samples = if config.dec_context_size == config.fg_context_size {
        cell_samples.clone()
    } else {
        csc_columns_to_indexed_samples(
            &x_dn,
            config.shortlist_weights,
            config.dec_context_size,
            None,
        )
    };
    let cell_size_factor: Vec<f32> = (0..ncols)
        .map(|j| x_dn.col(j).values().iter().sum::<f32>().max(1.0))
        .collect();

    // BG samples: the batch-null profile. With a δ estimate, fan the
    // pre-built per-batch top-K out by per-cell membership; without δ it
    // falls back to the observed counts (reusing the FG pack when the
    // contexts match), mirroring training's `build_cell_grouped_loaders`.
    let bg_samples: Vec<IndexedSample> = match config.per_batch_bg {
        Some(per_batch) => {
            let membership =
                crate::topic::common::block_membership(data_vec, config.adj_method, lb, ub)?;
            membership.iter().map(|&b| per_batch[b].clone()).collect()
        }
        None if config.bg_context_size == config.fg_context_size => cell_samples.clone(),
        None => csc_columns_to_indexed_samples(
            &x_dn,
            config.shortlist_weights,
            config.bg_context_size,
            None,
        ),
    };

    let mb = pack_eval_minibatch(
        CellEvalPackArgs {
            cell_samples: &cell_samples,
            cell_size_factor: &cell_size_factor,
            bg_samples: &bg_samples,
            output_samples: &output_samples,
            n_features,
            fg_context_size: config.fg_context_size,
            bg_context_size: config.bg_context_size,
            dec_context_size: config.dec_context_size,
            feature_fisher_weights: config.feature_fisher_weights,
        },
        config.dev,
    )?;

    let (log_z_nk, _) = encoder.forward_cells_t(&mb, false)?;

    // Optional decoder-side refinement against the frozen ETM decoder.
    // Inference uses a uniform `log_q_s`, matching `eval_indexed`.
    let log_z_nk = if let Some(cfg) = config.refine_config {
        let s = mb.output_union_indices.dim(0)?;
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, config.dev)?;
        refine_indexed_topic_proportions(
            &log_z_nk,
            &mb.output_union_indices,
            &mb.output_scatter_pos,
            &mb.output_values,
            Some(&mb.output_values_weight),
            &log_q_s,
            config.decoder,
            cfg,
        )?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}
