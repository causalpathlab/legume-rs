use super::common::{apply_column_delta, sample_collapsed_data};
use super::eval_indexed::{dense_to_indexed, refine_indexed_topic_proportions, PerGeneContext};
use crate::embed_common::*;
use crate::logging::new_progress_bar;

use super::graph_likelihood::{graph_loss, PoissonGraphConfig};
use candle_core::{Device, Tensor};
use candle_nn::AdamW;
use candle_util::candle_decoder_embedded_topic::EmbeddedTopicDecoder;
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
    /// Per-gene weights used to *score* candidates during top-K shortlist
    /// selection (encoder + decoder). Stored values remain raw counts.
    pub shortlist_weights: &'a [f32],
    /// Per-gene Anscombe baseline (length = D_full). When supplied, the
    /// loader gathers it at each cell's encoder top-K positions; the
    /// encoder subtracts it from Anscombe-stabilized values before pooling
    /// — per-gene equivalent of dense `anscombe_residual`'s per-feature
    /// batch centering.
    pub feature_mean: &'a [f32],
    /// Per-gene NB-Fisher info weight (length = D_full). When supplied,
    /// the loader gathers it at each cell's decoder top-K positions; the
    /// decoder multiplies it into the `(value+1).log()` likelihood term
    /// so housekeeping observations contribute less to β's gradient.
    pub feature_fisher_weights: &'a [f32],
    /// Global L2 gradient norm clip per minibatch (0 = off).
    pub grad_clip: f32,
}

impl IndexedTrainConfig<'_> {
    /// Anchor-prior CE penalty for the ETM-factorized indexed decoder. The
    /// full `[K, D]` logits don't live as a single Var — they're
    /// `α · ρᵀ` — so the caller passes the decoder and we materialize the
    /// logits once per minibatch.
    #[inline]
    fn add_anchor_penalty(
        &self,
        loss: Tensor,
        decoder: &EmbeddedTopicDecoder,
        level: usize,
    ) -> anyhow::Result<Tensor> {
        // Skip `full_logits_kd()` ([K, H]·[H, D] matmul on the AD tape)
        // when there's no penalty to apply. `anchor_penalty_at_level_with_logits`
        // re-checks defensively, but this guard is the load-bearing one.
        if self.anchor_prior_per_level.is_none() || self.anchor_penalty <= 0.0 {
            return Ok(loss);
        }
        let logits_kd = decoder.full_logits_kd()?;
        crate::topic::anchor_prior::anchor_penalty_at_level_with_logits(
            loss,
            &logits_kd,
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
                input_mean: Some(config.feature_mean),
                output_fisher_weights: Some(config.feature_fisher_weights),
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

pub(crate) fn train_mixed(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[EmbeddedTopicDecoder],
    config: &IndexedTrainConfig,
    bulk_with_deltas: Option<(&Mat, &[GammaMatrix])>,
    graph_cfg: Option<&PoissonGraphConfig>,
) -> anyhow::Result<TrainScores> {
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
    let prog_bar = new_progress_bar(total_epochs as u64);

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
                input_mean: Some(config.feature_mean),
                output_fisher_weights: Some(config.feature_fisher_weights),
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
                    &mb.input_indices,
                    &mb.input_values,
                    mb.input_values_null.as_ref(),
                    mb.input_values_mean.as_ref(),
                    true,
                )?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;

                let llik = decoder.forward_indexed(
                    &log_z_nk,
                    &mb.output_union_indices,
                    &mb.output_scatter_pos,
                    &mb.output_values,
                    mb.output_values_weight.as_ref(),
                    &mb.output_log_q_s,
                )?;
                let loss = (&kl - &llik)?.mean_all()?;
                let loss = config.add_anchor_penalty(loss, decoder, level)?;

                clip_grads_and_step(&mut adam, &loss, f64::from(config.grad_clip))?;

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                count_tot += mb.output_values.sum_all()?.to_scalar::<f32>()?;

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
                    &mb.input_indices,
                    &mb.input_values,
                    None,
                    mb.input_values_mean.as_ref(),
                    true,
                )?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                let llik = finest_decoder.forward_indexed(
                    &log_z_nk,
                    &mb.output_union_indices,
                    &mb.output_scatter_pos,
                    &mb.output_values,
                    mb.output_values_weight.as_ref(),
                    &mb.output_log_q_s,
                )?;
                let loss = (&kl - &llik)?.mean_all()?;
                let loss = config.add_anchor_penalty(loss, finest_decoder, finest_idx)?;
                clip_grads_and_step(&mut adam, &loss, f64::from(config.grad_clip))?;

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                count_tot += mb.output_values.sum_all()?.to_scalar::<f32>()?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        // Once-per-epoch BKN graph-likelihood gradient step (Poisson on
        // observed feature-pair edges, closed-form non-edge partition).
        // Cheap: O(|E|·H + D·H). Operates on the *full* ρ [D, H] — the
        // global degree-weighted partition Σ_u d̂_u · ρ̃_{u,h} can't be
        // restricted to the minibatch's top-K slice.
        if let Some(cfg) = graph_cfg {
            let g_loss = graph_loss(encoder.feature_embeddings(), cfg)?;
            clip_grads_and_step(&mut adam, &g_loss, f64::from(config.grad_clip))?;
            let d = encoder.feature_embeddings().dim(0)? as f32;
            info!(
                "[epoch {}] graph_nll/gene={:.4}",
                epoch,
                g_loss.to_scalar::<f32>()? / d.max(1.0)
            );
        }

        llik_trace.push(llik_tot / count_tot);
        kl_trace.push(kl_tot / n_tot as f32);

        prog_bar.inc(1);

        info!(
            "[epoch {}] llik={} kl={}",
            epoch,
            llik_trace.last().unwrap(),
            kl_trace.last().unwrap()
        );

        if config.stop.load(Ordering::SeqCst) {
            prog_bar.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(TrainScores {
                llik: llik_trace,
                kl: kl_trace,
            });
        }
    }

    prog_bar.finish_and_clear();
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
