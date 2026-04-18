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
        gene_remap: None,
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_block(block, data_vec, encoder, &block_config)
    })
}

/// Mapping from new-data row indices to training gene positions.
pub(crate) struct GeneRemap {
    /// For each new-data compact row, the training gene position (or None).
    pub new_to_train: Vec<Option<usize>>,
    /// Number of training genes (`D_train`).
    pub d_train: usize,
    /// Number of new genes that mapped to training genes.
    pub n_mapped: usize,
}

/// Build a gene remap from training gene names and new-data gene names.
pub(crate) fn build_gene_remap(
    training_genes: &[Box<str>],
    new_data_genes: &[Box<str>],
) -> GeneRemap {
    let train_pos: rustc_hash::FxHashMap<&str, usize> = training_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_ref(), i))
        .collect();

    let new_to_train: Vec<Option<usize>> = new_data_genes
        .iter()
        .map(|g| train_pos.get(g.as_ref()).copied())
        .collect();

    let n_mapped = new_to_train.iter().filter(|x| x.is_some()).count();
    log::info!(
        "Gene alignment: {}/{} new genes mapped to {}/{} training genes",
        n_mapped,
        new_data_genes.len(),
        n_mapped,
        training_genes.len()
    );

    GeneRemap {
        new_to_train,
        d_train: training_genes.len(),
        n_mapped,
    }
}

/// Evaluate latent states with optional gene remapping and pre-computed delta.
///
/// When `gene_remap` is `Some`, per-block CSC data is scattered from new-data
/// row order to training gene order. When `None`, data is used as-is.
pub(crate) fn evaluate_latent_with_gene_remap<Enc, Dec>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    delta_db: Option<&Mat>,
    gene_remap: Option<&GeneRemap>,
    config: &EvaluateLatentConfig<Dec>,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
    Dec: DecoderModuleT + Send + Sync,
{
    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    let delta = delta_db.map(|db| {
        let mut db = db.clone();
        if let Some(fc) = config.feature_coarsening {
            db = fc.aggregate_rows_ds(&db);
        }
        db.to_tensor(config.dev)
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
        gene_remap,
    };

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        evaluate_block(block, data_vec, encoder, &block_config)
    })
}

/// Scatter CSC rows from new-data order to training gene order.
fn remap_csc_to_dense(csc: &nalgebra_sparse::CscMatrix<f32>, remap: &GeneRemap) -> Mat {
    let ncols = csc.ncols();
    let mut out = Mat::zeros(remap.d_train, ncols);
    for j in 0..ncols {
        let col = csc.col(j);
        for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
            if let Some(row_train) = remap.new_to_train[row_new] {
                out[(row_train, j)] = val;
            }
        }
    }
    out
}

/// Configuration for block-wise evaluation
struct EvaluateBlockConfig<'a, Dec> {
    dev: &'a Device,
    delta: Option<&'a Tensor>,
    feature_coarsening: Option<&'a FeatureCoarsening>,
    decoder: Option<&'a Dec>,
    refine_config: Option<&'a TopicRefinementConfig>,
    adj_method: AdjMethod,
    gene_remap: Option<&'a GeneRemap>,
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

    let x_dn_csc = data_vec.read_columns_csc(lb..ub)?;

    let x_enc_nd = if let Some(remap) = config.gene_remap {
        let x_dn_train = remap_csc_to_dense(&x_dn_csc, remap);
        if let Some(fc) = config.feature_coarsening {
            fc.aggregate_rows_ds(&x_dn_train)
                .to_tensor(config.dev)?
                .transpose(0, 1)?
        } else {
            x_dn_train.to_tensor(config.dev)?.transpose(0, 1)?
        }
    } else if let Some(fc) = config.feature_coarsening {
        fc.aggregate_sparse_csc(&x_dn_csc)
            .to_tensor(config.dev)?
            .transpose(0, 1)?
    } else {
        x_dn_csc.to_tensor(config.dev)?.transpose(0, 1)?
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
