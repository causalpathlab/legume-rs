use super::common::{apply_column_delta, sample_collapsed_data};
use super::eval_indexed::{dense_to_indexed, refine_indexed_topic_proportions};
use crate::embed_common::*;
use crate::logging::new_progress_bar;

use candle_core::{Device, Tensor};
use candle_nn::AdamW;
use candle_util::candle_encoder_indexed::IndexedEmbeddingEncoder;
use candle_util::candle_indexed_data_loader::*;
use candle_util::candle_indexed_model_traits::*;
use candle_util::candle_topic_refinement::TopicRefinementConfig;
use matrix_param::dmatrix_gamma::GammaMatrix;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct IndexedTrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub learning_rate: f32,
    pub topic_smoothing: f64,
    pub enc_context_size: usize,
    pub dec_context_size: usize,
    pub stop: &'a AtomicBool,
    /// Per-level `[K, D_l]` anchor β prior tensors (pre-transposed, on device).
    pub anchor_prior_per_level: Option<&'a [Tensor]>,
    /// Cross-entropy penalty strength λ applied per minibatch.
    pub anchor_penalty: f32,
    /// Per-gene weights used only to *score* candidates during top-K
    /// shortlist selection (encoder + decoder). Stored values are the raw
    /// pseudobulk row entries — these weights do not enter the likelihood.
    pub shortlist_weights: &'a [f32],
    /// Global L2 gradient norm clip per minibatch (0 = off).
    pub grad_clip: f32,
}

impl IndexedTrainConfig<'_> {
    #[inline]
    fn add_anchor_penalty(&self, loss: Tensor, level: usize) -> anyhow::Result<Tensor> {
        crate::topic::anchor_prior::anchor_penalty_at_level(
            loss,
            self.parameters,
            self.anchor_prior_per_level,
            self.anchor_penalty,
            level,
        )
    }
}

/// Estimate bulk-vs-SC bias as a `GammaMatrix` [`D_sc`, 1].
pub(crate) fn estimate_bulk_delta(bulk_dm: &Mat, collapsed: &CollapsedOut) -> GammaMatrix {
    let mu_adj = collapsed
        .mu_adjusted
        .as_ref()
        .unwrap_or(&collapsed.mu_observed);
    let mu_adj_mean = mu_adj.posterior_mean();

    let n_sc = mu_adj_mean.ncols() as f32;
    let mu_gene_mean: Mat = Mat::from_fn(mu_adj_mean.nrows(), 1, |i, _| {
        mu_adj_mean.row(i).iter().sum::<f32>() / n_sc
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

fn build_indexed_loaders(
    level_data: &[(Mat, Option<Mat>, Mat)],
    config: &IndexedTrainConfig,
) -> anyhow::Result<Vec<IndexedInMemoryData>> {
    level_data
        .iter()
        .map(|(mixed, batch, target)| {
            IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: mixed,
                input_null: batch.as_ref(),
                output: target,
                input_context_size: config.enc_context_size,
                output_context_size: config.dec_context_size,
                input_shortlist_weights: config.shortlist_weights,
                output_shortlist_weights: config.shortlist_weights,
            })
        })
        .collect()
}

fn shuffle_and_precompute(
    loader: &mut IndexedInMemoryData,
    minibatch_size: usize,
    dev: &Device,
) -> anyhow::Result<()> {
    loader.shuffle_minibatch(minibatch_size);
    loader.precompute_all_minibatches(dev)
}

pub(crate) fn train_mixed<Dec>(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[Dec],
    config: &IndexedTrainConfig,
    bulk_with_deltas: Option<(&Mat, &[GammaMatrix])>,
) -> anyhow::Result<TrainScores>
where
    Dec: IndexedDecoderT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.epochs;

    for (level, (collapsed, decoder)) in collapsed_levels.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {}",
            level + 1,
            num_levels,
            collapsed.mu_observed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!("Mixed multi-level training: {num_levels} levels, {total_epochs} epochs");

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        f64::from(config.learning_rate),
    )?;
    let pb = new_progress_bar(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);

    let level_data = build_level_data(collapsed_levels)?;
    let mut data_loaders = build_indexed_loaders(&level_data, config)?;

    // Bulk loader: corrected matrix uses the posterior mean of the bulk
    // delta. Built once per training run.
    let mut bulk_loader: Option<IndexedInMemoryData> =
        if let Some((bulk_full, bulk_deltas)) = bulk_with_deltas {
            let finest_idx = num_levels - 1;
            let bulk_delta = &bulk_deltas[finest_idx];
            let delta_mean = bulk_delta.posterior_mean().transpose();
            let corrected = apply_column_delta(bulk_full, &delta_mean, 1e-8);
            Some(IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: bulk_full,
                input_null: None,
                output: &corrected,
                input_context_size: config.enc_context_size,
                output_context_size: config.dec_context_size,
                input_shortlist_weights: config.shortlist_weights,
                output_shortlist_weights: config.shortlist_weights,
            })?)
        } else {
            None
        };

    for epoch in 0..total_epochs {
        for loader in data_loaders.iter_mut() {
            shuffle_and_precompute(loader, config.minibatch_size, config.dev)?;
        }
        if let Some(bulk_loader) = bulk_loader.as_mut() {
            shuffle_and_precompute(bulk_loader, config.minibatch_size, config.dev)?;
        }

        let mut llik_tot = 0f32;
        let mut kl_tot = 0f32;
        let mut count_tot = 0f32;
        let mut n_tot = 0usize;

        for (level, loader) in data_loaders.iter().enumerate() {
            let decoder = &decoders[level];
            n_tot += loader.num_data();

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b);

                let (log_z_nk, kl) = encoder.forward_indexed_t(
                    &mb.input_union_indices,
                    &mb.input_indexed_x,
                    mb.input_indexed_x_null.as_ref(),
                    true,
                )?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;

                let (_, llik) = decoder.forward_indexed(
                    &log_z_nk,
                    &mb.output_union_indices,
                    &mb.output_indexed_x,
                    &mb.output_log_q_s,
                )?;
                let loss = (&kl - &llik)?.mean_all()?;
                let loss = config.add_anchor_penalty(loss, level)?;

                clip_grads_and_step(&mut adam, &loss, f64::from(config.grad_clip))?;

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                count_tot += mb.output_indexed_x.sum_all()?.to_scalar::<f32>()?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        // Bulk training step (if present) — use finest decoder.
        if let Some(bulk_loader) = bulk_loader.as_ref() {
            let finest_idx = num_levels - 1;
            let finest_decoder = &decoders[finest_idx];

            for b in 0..bulk_loader.num_minibatch() {
                let mb = bulk_loader.minibatch_cached(b);
                let (log_z_nk, kl) = encoder.forward_indexed_t(
                    &mb.input_union_indices,
                    &mb.input_indexed_x,
                    None,
                    true,
                )?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                let (_, llik) = finest_decoder.forward_indexed(
                    &log_z_nk,
                    &mb.output_union_indices,
                    &mb.output_indexed_x,
                    &mb.output_log_q_s,
                )?;
                let loss = (&kl - &llik)?.mean_all()?;
                let loss = config.add_anchor_penalty(loss, finest_idx)?;
                clip_grads_and_step(&mut adam, &loss, f64::from(config.grad_clip))?;

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                count_tot += mb.output_indexed_x.sum_all()?.to_scalar::<f32>()?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        llik_trace.push(llik_tot / count_tot);
        kl_trace.push(kl_tot / n_tot as f32);

        pb.inc(1);

        info!(
            "[epoch {}] llik={} kl={}",
            epoch,
            llik_trace.last().unwrap(),
            kl_trace.last().unwrap()
        );

        if config.stop.load(Ordering::SeqCst) {
            pb.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(TrainScores {
                llik: llik_trace,
                kl: kl_trace,
            });
        }
    }

    pb.finish_and_clear();
    info!("done mixed multi-level training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
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

    let bulk_tensor = bulk_nd
        .to_tensor(config.dev)?
        .to_dtype(candle_core::DType::F32)?;
    let (enc_union, enc_indexed_x) = dense_to_indexed(
        &bulk_tensor,
        config.enc_context_size,
        config.shortlist_weights,
        config.dev,
    )?;
    let (log_z_nk, _) = encoder.forward_indexed_t(&enc_union, &enc_indexed_x, None, false)?;

    let log_z_nk = if let Some(cfg) = config.refine_config {
        let corrected_tensor = bulk_corrected
            .to_tensor(config.dev)?
            .to_dtype(candle_core::DType::F32)?;
        let (dec_union, dec_indexed_x) = dense_to_indexed(
            &corrected_tensor,
            config.dec_context_size,
            config.shortlist_weights,
            config.dev,
        )?;
        let s = dec_union.dim(0)?;
        let log_q_s = Tensor::zeros((1, s), candle_core::DType::F32, config.dev)?;
        refine_indexed_topic_proportions(
            &log_z_nk,
            &dec_union,
            &dec_indexed_x,
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

/// Write dictionary at full resolution (no coarsening for indexed model).
pub(crate) fn write_indexed_dictionary<Dec: IndexedDecoderT>(
    decoder: &Dec,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let dict_tensor = decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?;

    let k_topics = dict_tensor.dims().last().copied().unwrap_or(0);
    dict_tensor.to_parquet_with_names(
        &(out_prefix.to_string() + ".dictionary.parquet"),
        (Some(gene_names), Some("gene")),
        Some(&axis_id_names("T", k_topics)),
    )?;
    Ok(())
}
