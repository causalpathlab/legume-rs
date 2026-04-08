use super::common::{expand_delta_for_block, process_blocks};
use crate::embed_common::*;

use candle_core::{Device, Tensor};
use candle_util::candle_model_traits::*;
use candle_util::candle_topic_refinement::*;

/// Configuration for latent evaluation by encoder
pub(crate) struct EvaluateLatentConfig<'a, Dec> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    pub feature_coarsening: Option<&'a FeatureCoarsening>,
    pub decoder: Option<&'a Dec>,
    pub refine_config: Option<&'a TopicRefinementConfig>,
}

pub(crate) fn evaluate_latent_by_encoder<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    collapsed: &CollapsedOut,
    config: &EvaluateLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
    Dec: DecoderModuleT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    // Delta coarsened to D_coarse — encoder operates at D_coarse
    let delta = match config.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| x.posterior_mean().clone())
    .map(|mut delta_db| {
        if let Some(fc) = config.feature_coarsening {
            delta_db = fc.aggregate_rows_ds(&delta_db);
        }
        delta_db
            .to_tensor(config.dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
    });

    let block_config = EvaluateBlockConfig {
        dev: config.dev,
        delta: delta.as_ref(),
        feature_coarsening: config.feature_coarsening,
        decoder: config.decoder,
        refine_config: config.refine_config,
        adj_method: config.adj_method.clone(),
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_block(block, data_vec, encoder, &block_config)
    })
}

/// Configuration for block-wise evaluation
struct EvaluateBlockConfig<'a, Dec> {
    dev: &'a Device,
    delta: Option<&'a Tensor>,
    feature_coarsening: Option<&'a FeatureCoarsening>,
    decoder: Option<&'a Dec>,
    refine_config: Option<&'a TopicRefinementConfig>,
    adj_method: AdjMethod,
}

fn evaluate_block<Enc, Dec>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &Enc,
    config: &EvaluateBlockConfig<Dec>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let (lb, ub) = block;
    let x0_nd = config
        .delta
        .map(|delta_bm| {
            expand_delta_for_block(data_vec, delta_bm, &config.adj_method, lb, ub, config.dev)
        })
        .transpose()?;

    let x_dn = data_vec.read_columns_csc(lb..ub)?;

    // Coarsen to D_coarse for encoder (same resolution as decoder)
    let x_enc_nd = if let Some(fc) = config.feature_coarsening {
        fc.aggregate_sparse_csc(&x_dn)
            .to_tensor(config.dev)?
            .transpose(0, 1)?
    } else {
        x_dn.to_tensor(config.dev)?.transpose(0, 1)?
    };

    let (log_z_nk, _) = encoder.forward_t(&x_enc_nd, x0_nd.as_ref(), false)?;

    // Apply per-cell refinement (data already at D_coarse)
    let log_z_nk = if let (Some(dec), Some(cfg)) = (config.decoder, config.refine_config) {
        refine_topic_proportions(&log_z_nk, &x_enc_nd, dec, cfg)?
    } else {
        log_z_nk
    };

    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}
