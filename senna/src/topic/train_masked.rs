//! Senna-side glue for the masked-topic trainer.
//!
//! The training hot loop lives in [`candle_util::vae::masked_topic`].
//! This module owns senna-specific bits the candle-util trainer does not
//! see: per-level data assembly from [`CollapsedOut`], bulk-vs-SC delta
//! estimation, bulk evaluation, and the dictionary / feature-embedding
//! writers. The `pub(crate) use` re-exports keep existing call sites
//! (the `masked_topic` command, `train_cell_embedded`) on stable import paths.

use super::common::sample_collapsed_data;
use crate::embed_common::*;

use candle_core::Tensor;
use candle_util::encoder::IndexedEmbeddingEncoder;

// Re-export the generic trainer surface so legacy call sites stay put.
pub(crate) use candle_util::vae::masked_topic::IndexedTrainConfig;

/// Materialize per-level `(mixed, batch, target)` `Mat` triples once
/// per training run.
fn build_level_data(
    collapsed_levels: &[CollapsedOut],
) -> anyhow::Result<Vec<(Mat, Option<Mat>, Mat)>> {
    collapsed_levels.iter().map(sample_collapsed_data).collect()
}

/// Senna wrapper around [`candle_util::vae::masked_topic::train_masked`] —
/// the masked-imputation (no-ELBO) embedded topic model.
pub(crate) fn train_masked(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[candle_util::decoder::EmbeddedNbTopicDecoder],
    config: &IndexedTrainConfig,
    mask_fraction: f64,
    opts: &candle_util::vae::masked_topic::MaskedTrainOpts,
) -> anyhow::Result<TrainScores> {
    let level_data = build_level_data(collapsed_levels)?;
    let level_refs: Vec<candle_util::vae::masked_topic::LevelData> = level_data
        .iter()
        .map(|(a, b, c)| (a, b.as_ref(), c))
        .collect();
    let scores = candle_util::vae::masked_topic::train_masked(
        &level_refs,
        encoder,
        decoders,
        config,
        mask_fraction,
        opts,
    )?;
    Ok(TrainScores {
        llik: scores.llik,
        kl: scores.kl,
    })
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

/// Write the `[D, K]` log-β dictionary + the per-gene dispersion `φ` for the
/// masked-imputation NB embedded topic decoder.
pub(crate) fn write_masked_dictionary(
    decoder: &candle_util::decoder::EmbeddedNbTopicDecoder,
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
    )?;
    // Per-gene NB dispersion φ = exp(log_phi) as [D, 1].
    let phi_1d = decoder.phi()?; // [1, D]
    let phi_d1 = phi_1d.transpose(0, 1)?.contiguous()?; // [D, 1]
    write_tensor_parquet(
        &phi_d1,
        out_prefix,
        "dispersion.parquet",
        gene_names,
        "gene",
        "phi",
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
