use super::common::{apply_column_delta, sample_collapsed_data};
use super::eval_indexed::{dense_to_indexed, refine_indexed_topic_proportions, PerGeneContext};
use crate::embed_common::*;
use crate::logging::new_progress_bar;

use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use candle_util::candle_decoder_embedded_topic::EmbeddedTopicDecoder;
use candle_util::candle_encoder_indexed::IndexedEmbeddingEncoder;
use candle_util::candle_indexed_data_loader::*;
use candle_util::candle_indexed_model_traits::*;
use candle_util::candle_topic_refinement::TopicRefinementConfig;
use matrix_param::dmatrix_gamma::GammaMatrix;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Wall-clock breakdown of the indexed-topic training hot loop.
///
/// Candle's CPU backend is eager, so an `Instant` straddling each phase
/// attributes time accurately (no async kernels to sync). On CUDA the
/// kernels are async, so the per-phase split is only indicative.
#[derive(Default)]
pub(crate) struct PhaseTimers {
    pub(crate) precompute: Duration,
    pub(crate) encoder_fwd: Duration,
    pub(crate) decoder_fwd: Duration,
    pub(crate) backward: Duration,
    pub(crate) optimize: Duration,
}

impl PhaseTimers {
    pub(crate) fn log_summary(&self) {
        let total =
            self.precompute + self.encoder_fwd + self.decoder_fwd + self.backward + self.optimize;
        let total_s = total.as_secs_f64().max(1e-9);
        let pct = |d: Duration| 100.0 * d.as_secs_f64() / total_s;
        info!(
            "phase timing — precompute {:.1}s ({:.0}%), encoder_fwd {:.1}s ({:.0}%), \
             decoder_fwd {:.1}s ({:.0}%), backward {:.1}s ({:.0}%), opt_step {:.1}s ({:.0}%)",
            self.precompute.as_secs_f64(),
            pct(self.precompute),
            self.encoder_fwd.as_secs_f64(),
            pct(self.encoder_fwd),
            self.decoder_fwd.as_secs_f64(),
            pct(self.decoder_fwd),
            self.backward.as_secs_f64(),
            pct(self.backward),
            self.optimize.as_secs_f64(),
            pct(self.optimize),
        );
    }
}

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
    /// Optional feature-feature graph attached to every level loader so
    /// that the indexed encoder's GCN block sees per-cell sub-adjacency.
    /// `None` skips the GCN branch and keeps the legacy sum-pool path.
    pub feature_graph: Option<Arc<GraphCsr>>,
    /// Explicit L2 penalty `λ_ρ · ‖ρ‖_F²` on the feature embedding
    /// matrix ρ ∈ ℝ^{D × H}. Added to the per-minibatch loss before
    /// backward, so the gradient on ρ gets an extra `2 · λ_ρ · ρ`
    /// shrinkage term. Equivalent to a zero-mean Gaussian prior on ρ
    /// with precision `2 · λ_ρ`. `0.0` disables.
    pub feature_embedding_l2: f32,
    /// AdamW decoupled weight decay applied to *every* parameter
    /// per-step (not just ρ). Different from `feature_embedding_l2` —
    /// this is a post-step parameter shrinkage that doesn't enter the
    /// loss/backward graph. `0.0` disables.
    pub weight_decay: f32,
    /// When `Some(name)`, exclude the named `Var` from AdamW (used to
    /// freeze ρ when its values came from a prior senna run) and skip
    /// the `rho_l2` term (no point regularizing a non-trainable
    /// parameter). The encoder/decoder still reference ρ through the
    /// same `Var`; freezing just keeps the optimizer's hands off.
    pub frozen_feature_var: Option<&'a str>,
}

/// Global-L2-norm clip + dense `AdamW` step from a precomputed `GradStore`.
/// Mirrors [`crate::embed_common::clip_grads_and_step`] but takes the
/// already-computed grads so the caller can time `backward()` separately.
pub(crate) fn clip_and_step_dense(
    adam: &mut AdamW,
    mut grads: candle_core::backprop::GradStore,
    max_norm: f64,
) -> anyhow::Result<()> {
    if max_norm <= 0.0 {
        adam.step(&grads)?;
        return Ok(());
    }
    let ids: Vec<_> = grads.get_ids().copied().collect();
    let mut sumsq: Option<Tensor> = None;
    for id in &ids {
        if let Some(g) = grads.get_id(*id) {
            let s = g.sqr()?.sum_all()?;
            sumsq = Some(match sumsq {
                None => s,
                Some(prev) => (prev + s)?,
            });
        }
    }
    let Some(sumsq) = sumsq else {
        return Ok(());
    };
    let inv_norm = sumsq.sqrt()?.affine(1.0, 1e-6)?.powf(-1.0)?;
    let scale = inv_norm.affine(max_norm, 0.0)?.clamp(0.0_f64, 1.0_f64)?;
    for id in &ids {
        if let Some(g) = grads.get_id(*id) {
            let scaled = g.broadcast_mul(&scale)?;
            grads.insert_id(*id, scaled);
        }
    }
    adam.step(&grads)?;
    Ok(())
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
            let mut loader = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: mixed,
                input_null: batch.as_ref(),
                output: target,
                input_context_size: config.enc_context_size,
                output_context_size: config.dec_context_size,
                input_shortlist_weights: config.shortlist_weights,
                output_shortlist_weights: config.shortlist_weights,
                input_mean: Some(config.feature_mean),
                output_fisher_weights: Some(config.feature_fisher_weights),
            })?;
            loader.set_graph_csr(config.feature_graph.clone());
            Ok(loader)
        })
        .collect()
}

fn shuffle_and_precompute(
    loader: &mut IndexedInMemoryData,
    minibatch_size: usize,
) -> anyhow::Result<()> {
    loader.shuffle_minibatch(minibatch_size);
    loader.precompute_all_minibatches()
}

pub(crate) fn train_mixed(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[EmbeddedTopicDecoder],
    config: &IndexedTrainConfig,
    bulk_with_deltas: Option<(&Mat, &[GammaMatrix])>,
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

    let adam_vars: Vec<Var> = match config.frozen_feature_var {
        None => config.parameters.all_vars(),
        Some(name) => {
            let trainable =
                candle_util::frozen_features::trainable_vars(config.parameters, &[name]);
            info!(
                "Freeze mode: AdamW over {} trainable vars ({} frozen, key='{}')",
                trainable.len(),
                config.parameters.all_vars().len() - trainable.len(),
                name
            );
            trainable
        }
    };
    let mut adam = AdamW::new(
        adam_vars,
        candle_nn::ParamsAdamW {
            lr: f64::from(config.learning_rate),
            weight_decay: f64::from(config.weight_decay),
            ..Default::default()
        },
    )?;
    let prog_bar = new_progress_bar(total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);
    let mut timers = PhaseTimers::default();

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
            let mut bulk = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: bulk_full,
                input_null: None,
                output: &corrected,
                input_context_size: config.enc_context_size,
                output_context_size: config.dec_context_size,
                input_shortlist_weights: config.shortlist_weights,
                output_shortlist_weights: config.shortlist_weights,
                input_mean: Some(config.feature_mean),
                output_fisher_weights: Some(config.feature_fisher_weights),
            })?;
            bulk.set_graph_csr(config.feature_graph.clone());
            Some(bulk)
        } else {
            None
        };

    for epoch in 0..total_epochs {
        let t_pre = Instant::now();
        for loader in data_loaders.iter_mut() {
            shuffle_and_precompute(loader, config.minibatch_size)?;
        }
        if let Some(bulk_loader) = bulk_loader.as_mut() {
            shuffle_and_precompute(bulk_loader, config.minibatch_size)?;
        }
        timers.precompute += t_pre.elapsed();

        let mut llik_tot = 0f32;
        let mut kl_tot = 0f32;
        let mut count_tot = 0f32;
        let mut n_tot = 0usize;

        for (level, loader) in data_loaders.iter().enumerate() {
            let decoder = &decoders[level];
            n_tot += loader.num_data();
            count_tot += loader.total_output_count();

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b).to_device(config.dev)?;
                let sparse_edges = loader.minibatch_sparse_edges(b, config.dev)?;

                let t_enc = Instant::now();
                let (log_z_nk, kl) = encoder.forward_indexed_t(
                    &mb.input_indices,
                    &mb.input_values,
                    mb.input_values_null.as_ref(),
                    mb.input_values_mean.as_ref(),
                    sparse_edges.as_ref(),
                    true,
                )?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                timers.encoder_fwd += t_enc.elapsed();

                let t_dec = Instant::now();
                let llik = decoder.forward_indexed(
                    &log_z_nk,
                    &mb.output_union_indices,
                    &mb.output_scatter_pos,
                    &mb.output_values,
                    mb.output_values_weight.as_ref(),
                    &mb.output_log_q_s,
                )?;
                let mut loss = (&kl - &llik)?.mean_all()?;
                if config.feature_embedding_l2 > 0.0 && config.frozen_feature_var.is_none() {
                    // `mean_all` (not `sum_all`) so λ stays scale-invariant
                    // across `D · H`: λ=1 means per-element shrinkage of one
                    // loss unit, not D·H · mean(ρ²).
                    let rho_l2 = encoder
                        .feature_embeddings()
                        .sqr()?
                        .mean_all()?
                        .affine(f64::from(config.feature_embedding_l2), 0.0)?;
                    loss = (loss + rho_l2)?;
                }
                timers.decoder_fwd += t_dec.elapsed();

                let t_bwd = Instant::now();
                let grads = loss.backward()?;
                timers.backward += t_bwd.elapsed();

                let t_opt = Instant::now();
                clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))?;
                timers.optimize += t_opt.elapsed();

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        // Bulk training step (if present) — use finest decoder.
        if let Some(bulk_loader) = bulk_loader.as_ref() {
            let finest_idx = num_levels - 1;
            let finest_decoder = &decoders[finest_idx];
            count_tot += bulk_loader.total_output_count();

            for b in 0..bulk_loader.num_minibatch() {
                let mb = bulk_loader.minibatch_cached(b).to_device(config.dev)?;
                let sparse_edges = bulk_loader.minibatch_sparse_edges(b, config.dev)?;
                let t_enc = Instant::now();
                let (log_z_nk, kl) = encoder.forward_indexed_t(
                    &mb.input_indices,
                    &mb.input_values,
                    None,
                    mb.input_values_mean.as_ref(),
                    sparse_edges.as_ref(),
                    true,
                )?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                timers.encoder_fwd += t_enc.elapsed();

                let t_dec = Instant::now();
                let llik = finest_decoder.forward_indexed(
                    &log_z_nk,
                    &mb.output_union_indices,
                    &mb.output_scatter_pos,
                    &mb.output_values,
                    mb.output_values_weight.as_ref(),
                    &mb.output_log_q_s,
                )?;
                let mut loss = (&kl - &llik)?.mean_all()?;
                if config.feature_embedding_l2 > 0.0 && config.frozen_feature_var.is_none() {
                    // `mean_all` (not `sum_all`) so λ stays scale-invariant
                    // across `D · H`: λ=1 means per-element shrinkage of one
                    // loss unit, not D·H · mean(ρ²).
                    let rho_l2 = encoder
                        .feature_embeddings()
                        .sqr()?
                        .mean_all()?
                        .affine(f64::from(config.feature_embedding_l2), 0.0)?;
                    loss = (loss + rho_l2)?;
                }
                timers.decoder_fwd += t_dec.elapsed();

                let t_bwd = Instant::now();
                let grads = loss.backward()?;
                timers.backward += t_bwd.elapsed();

                let t_opt = Instant::now();
                clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))?;
                timers.optimize += t_opt.elapsed();

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        llik_trace.push(llik_tot / count_tot);
        kl_trace.push(kl_tot / n_tot as f32);

        prog_bar.inc(1);

        if log::log_enabled!(log::Level::Info) {
            let gamma_str = encoder.gcn_gamma_vec()?.map_or(String::new(), |g| {
                let l2 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
                let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let mean = g.iter().sum::<f32>() / (g.len().max(1) as f32);
                format!(" ‖γ‖={l2:.3e} max|γ|={max_abs:.3e} mean(γ)={mean:+.3e}")
            });
            info!(
                "[epoch {}] llik={} kl={}{}",
                epoch,
                llik_trace.last().unwrap(),
                kl_trace.last().unwrap(),
                gamma_str,
            );
        }

        if config.stop.load(Ordering::SeqCst) {
            prog_bar.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            timers.log_summary();
            return Ok(TrainScores {
                llik: llik_trace,
                kl: kl_trace,
            });
        }
    }

    prog_bar.finish_and_clear();
    info!("done mixed multi-level training");
    timers.log_summary();
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
