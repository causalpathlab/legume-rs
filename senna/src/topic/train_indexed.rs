//! Senna-side glue for the indexed-topic trainer.
//!
//! The training hot loop lives in [`candle_util::vae::indexed_topic`].
//! This module owns senna-specific bits the candle-util trainer does not
//! see: per-level data assembly from [`CollapsedOut`], bulk-vs-SC delta
//! estimation, bulk evaluation, and the dictionary / feature-embedding
//! writers. The `pub(crate) use` re-exports keep existing call sites
//! (`fit_indexed_topic`, `train_cell_embedded`) on stable import paths.

use super::common::{apply_column_delta, sample_collapsed_data};
use super::eval_indexed::{dense_to_indexed, refine_indexed_topic_proportions, PerGeneContext};
use crate::embed_common::*;

use candle_core::{Device, Tensor};
use candle_util::decoder::EmbeddedTopicDecoder;
use candle_util::encoder::IndexedEmbeddingEncoder;
use candle_util::topic_refinement::TopicRefinementConfig;
use candle_util::traits::*;
use matrix_param::dmatrix_gamma::GammaMatrix;

// Re-export the generic trainer surface so legacy call sites stay put.
pub(crate) use candle_util::vae::indexed_topic::IndexedTrainConfig;

/// Estimate bulk-vs-SC bias as a `GammaMatrix` [`D_sc`, 1].
pub(crate) fn estimate_bulk_delta(bulk_dm: &Mat, collapsed: &CollapsedOut) -> GammaMatrix {
    let mu_adj = collapsed
        .mu_adjusted
        .as_ref()
        .unwrap_or(&collapsed.mu_observed);
    let mu_adj_mean = mu_adj.posterior_mean();

    let n_pbsamp = mu_adj_mean.ncols() as f32;
    let mu_gene_mean: Mat = Mat::from_fn(mu_adj_mean.nrows(), 1, |i, _| {
        mu_adj_mean.row(i).iter().sum::<f32>() / n_pbsamp
    });

    let m = bulk_dm.ncols() as f32;
    let bulk_sum: Mat = Mat::from_fn(bulk_dm.nrows(), 1, |i, _| {
        bulk_dm.row(i).iter().sum::<f32>()
    });

    let expected: Mat = &mu_gene_mean * m;

    let (a0, b0) = (1.0f32, 1.0f32);
    let mut bulk_delta = GammaMatrix::new((bulk_dm.nrows(), 1), a0, b0);
    bulk_delta.update_stat(&bulk_sum, &expected);
    bulk_delta.calibrate();

    bulk_delta
}

/// Materialize per-level `(mixed, batch, target)` `Mat` triples once
/// per training run.
fn build_level_data(
    collapsed_levels: &[CollapsedOut],
) -> anyhow::Result<Vec<(Mat, Option<Mat>, Mat)>> {
    collapsed_levels.iter().map(sample_collapsed_data).collect()
}

/// Senna wrapper around [`candle_util::vae::indexed_topic::train_mixed`].
///
/// Takes the senna-native `CollapsedOut` slice, assembles per-level data,
/// then delegates to the generic trainer. Converts the candle-util
/// `TrainScores` back to senna's struct so existing call sites (which
/// then write `.to_parquet(...)`) keep working unchanged.
pub(crate) fn train_mixed(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[EmbeddedTopicDecoder],
    config: &IndexedTrainConfig,
    bulk_with_deltas: Option<(&Mat, &[GammaMatrix])>,
) -> anyhow::Result<TrainScores> {
    let level_data = build_level_data(collapsed_levels)?;
    let level_refs: Vec<candle_util::vae::indexed_topic::LevelData> = level_data
        .iter()
        .map(|(a, b, c)| (a, b.as_ref(), c))
        .collect();
    let scores = candle_util::vae::indexed_topic::train_mixed(
        &level_refs,
        encoder,
        decoders,
        config,
        bulk_with_deltas,
    )?;
    Ok(TrainScores {
        llik: scores.llik,
        kl: scores.kl,
    })
}

pub(crate) struct BulkEvalConfig<'a, Dec> {
    pub dev: &'a Device,
    pub enc_context_size: usize,
    pub dec_context_size: usize,
    pub refine_config: Option<&'a TopicRefinementConfig>,
    pub decoder: &'a Dec,
    pub gene_names: &'a [Box<str>],
    pub out_prefix: &'a str,
    pub shortlist_weights: &'a [f32],
    pub feature_mean: &'a [f32],
    pub feature_fisher_weights: &'a [f32],
}

/// Evaluate bulk samples using the given encoder/decoder and write results.
pub(crate) fn evaluate_bulk_samples<Enc, Dec>(
    bulk: &BulkDataOut,
    bulk_deltas: &[GammaMatrix],
    encoder: &Enc,
    config: &BulkEvalConfig<Dec>,
) -> anyhow::Result<()>
where
    Enc: IndexedEncoderT,
    Dec: IndexedDecoderT,
{
    info!("Evaluating bulk samples ...");
    let finest_delta = bulk_deltas.last().unwrap();
    let delta_mean = finest_delta.posterior_mean().clone();

    let bulk_nd = bulk.data.transpose();
    let delta_row = delta_mean.transpose();
    let bulk_corrected = apply_column_delta(&bulk_nd, &delta_row, 1e-8);

    let ctx = PerGeneContext {
        feature_mean: Some(config.feature_mean),
        feature_fisher_weights: Some(config.feature_fisher_weights),
    };
    let bulk_tensor = bulk_nd
        .to_tensor(config.dev)?
        .to_dtype(candle_core::DType::F32)?;
    let enc_pack = dense_to_indexed(
        &bulk_tensor,
        config.enc_context_size,
        config.shortlist_weights,
        ctx,
        config.dev,
    )?;
    let (log_z_nk, _) = encoder.forward_indexed_t(
        &enc_pack.indices,
        &enc_pack.values,
        None,
        enc_pack.values_mean.as_ref(),
        None,
        false,
    )?;

    let log_z_nk = if let Some(cfg) = config.refine_config {
        let corrected_tensor = bulk_corrected
            .to_tensor(config.dev)?
            .to_dtype(candle_core::DType::F32)?;
        let dec_pack = dense_to_indexed(
            &corrected_tensor,
            config.dec_context_size,
            config.shortlist_weights,
            ctx,
            config.dev,
        )?;
        let s = dec_pack.union_indices.dim(0)?;
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, config.dev)?;
        refine_indexed_topic_proportions(
            &log_z_nk,
            &dec_pack.union_indices,
            &dec_pack.scatter_pos,
            &dec_pack.values,
            dec_pack.values_weight.as_ref(),
            &log_q_s,
            config.decoder,
            cfg,
        )?
    } else {
        log_z_nk
    };
    let z_nk_bulk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    let z_nk_bulk = Mat::from_tensor(&z_nk_bulk)?;

    z_nk_bulk.to_parquet_with_names(
        &(config.out_prefix.to_string() + ".deconv.parquet"),
        (Some(&bulk.samples), Some("sample")),
        Some(&axis_id_names("T", z_nk_bulk.ncols())),
    )?;

    delta_mean.to_parquet_with_names(
        &(config.out_prefix.to_string() + ".bulk_delta.parquet"),
        (Some(config.gene_names), Some("gene")),
        Some(&axis_id_names("T", delta_mean.ncols())),
    )?;
    info!("Wrote bulk deconvolution results");
    Ok(())
}

/// Pull `tensor` to host and write it to `{out_prefix}.{suffix}.parquet`,
/// labelling its last-dim columns with `{col_prefix}0..H` and rows with
/// `row_names` under the column name `row_axis`.
fn write_tensor_parquet(
    tensor: &Tensor,
    out_prefix: &str,
    suffix: &str,
    row_names: &[Box<str>],
    row_axis: &str,
    col_prefix: &str,
) -> anyhow::Result<()> {
    let host = tensor.to_device(&candle_core::Device::Cpu)?;
    let n_cols = host.dims().last().copied().unwrap_or(0);
    host.to_parquet_with_names(
        &format!("{out_prefix}.{suffix}"),
        (Some(row_names), Some(row_axis)),
        Some(&axis_id_names(col_prefix, n_cols)),
    )?;
    Ok(())
}

/// Write dictionary at full resolution (no coarsening for indexed model).
pub(crate) fn write_indexed_dictionary<Dec: IndexedDecoderT>(
    decoder: &Dec,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    write_tensor_parquet(
        &decoder.get_dictionary()?,
        out_prefix,
        "dictionary.parquet",
        gene_names,
        "gene",
        "T",
    )
}

/// Write the learned per-gene feature embedding ρ `[D, H]` as a parquet.
/// In the ETM factorization this is shared between encoder and decoder, so
/// it's the model's gene-level representation — directly usable for gene-gene
/// similarity, clustering into programs, or initializing downstream models.
pub(crate) fn write_feature_embedding(
    feature_embeddings: &Tensor,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    write_tensor_parquet(
        feature_embeddings,
        out_prefix,
        "feature_embedding.parquet",
        gene_names,
        "gene",
        "H",
    )
}
