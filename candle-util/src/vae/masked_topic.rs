//! Masked-imputation (and ELBO) topic VAE trainer.
//!
//! Hosts both [`train_masked`] (no-ELBO masked-gene NB imputation; simplex-θ
//! or Gaussian-z latent) and [`train_mixed`] (the ELBO path used by pinto
//! `lc-etm`), sharing the indexed top-K data loader and per-level decoders.
//! Drives the shared [`IndexedEmbeddingEncoder`] + per-level
//! [`EmbeddedTopicDecoder`] stack against [`IndexedInMemoryData`]
//! minibatches. The hot loop never materialises `[N, S]` or `[K, D]`;
//! all gather/scatter happens at the per-batch gene union.

use super::{clip_and_step_dense, smooth_topics, PhaseTimers, TrainScores};
use crate::data::indexed::{labeled_bar, GraphCsr, IndexedInMemoryArgs, IndexedInMemoryData};
use crate::decoder::embedded_topic::EmbeddedTopicDecoder;
use crate::decoder::masked_etm::{EmbeddedNbTopicDecoder, MaskedNbTarget};
use crate::encoder::indexed::IndexedEmbeddingEncoder;
use crate::traits::indexed::*;
use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use log::info;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;
use rand::RngExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

type Mat = DMatrix<f32>;

/// Config bundle passed by reference to [`train_mixed`].
pub struct IndexedTrainConfig<'a> {
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
    /// loader gathers it at each sample's encoder top-K positions; the
    /// encoder subtracts it from Anscombe-stabilized values before pooling.
    pub feature_mean: &'a [f32],
    /// Per-gene NB-Fisher info weight (length = D_full). When supplied,
    /// the loader gathers it at each sample's decoder top-K positions; the
    /// decoder multiplies it into the `(value+1).log()` likelihood term
    /// so housekeeping observations contribute less to β's gradient.
    pub feature_fisher_weights: &'a [f32],
    /// Global L2 gradient norm clip per minibatch (0 = off).
    pub grad_clip: f32,
    /// Optional feature-feature graph attached to every level loader so
    /// that the indexed encoder's GCN block sees per-sample sub-adjacency.
    /// `None` skips the GCN branch and keeps the legacy sum-pool path.
    pub feature_graph: Option<Arc<GraphCsr>>,
    /// Explicit L2 penalty `λ_ρ · ‖ρ‖_F²` on the feature embedding
    /// matrix ρ ∈ ℝ^{D × H}. Added to the per-minibatch loss before
    /// backward. `0.0` disables.
    pub feature_embedding_l2: f32,
    /// AdamW decoupled weight decay applied to every parameter per-step
    /// (not just ρ). Post-step parameter shrinkage that doesn't enter the
    /// loss/backward graph. `0.0` disables.
    pub weight_decay: f32,
    /// When `Some(name)`, exclude the named `Var` from AdamW (used to
    /// freeze ρ when its values came from a prior senna run) and skip
    /// the `rho_l2` term (no point regularizing a non-trainable
    /// parameter). The encoder/decoder still reference ρ through the
    /// same `Var`; freezing just keeps the optimizer's hands off.
    pub frozen_feature_var: Option<&'a str>,
    /// Per-level `[K, D_l]` anchor β prior tensors (pre-transposed, on
    /// device). When set with `anchor_penalty > 0`, the trainer adds
    /// `−λ · mean_K Σ_D prior_kd · log_softmax_D(α · ρᵀ)` to the loss at
    /// each minibatch. Anchors topic indices to anchor gene sets and
    /// breaks the K-way permutation symmetry of the ETM-factorized β.
    /// `None` disables; sized to `level_data.len()` when set.
    pub anchor_prior_per_level: Option<&'a [candle_core::Tensor]>,
    /// Cross-entropy penalty strength λ paired with `anchor_prior_per_level`.
    /// 0.0 disables even when the prior is supplied.
    pub anchor_penalty: f32,
}

/// Optional bulk-deconvolution input: `(bulk_full [G, B], deltas_per_level)`.
/// The trainer uses the finest-level delta to build a corrected bulk loader
/// and runs an extra mini-batch step against the finest decoder per epoch.
pub type BulkWithDeltas<'a> = (&'a Mat, &'a [GammaMatrix]);

/// Per-minibatch mask-rate schedule (any-order / absorbing-diffusion style).
#[derive(Clone, Copy, Debug)]
pub enum MaskSchedule {
    /// Constant mask fraction (the `mask_fraction` arg).
    Fixed,
    /// Sample the mask rate uniformly in `[lo, hi]` each minibatch.
    Uniform { lo: f64, hi: f64 },
}

/// Options specific to [`train_masked`], kept off the shared
/// [`IndexedTrainConfig`] so the `train_mixed`/pinto callers are unaffected.
/// [`Default`] reproduces the legacy NB, fixed-rate behavior.
/// Per-gene likelihood for the masked imputation loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskedLikelihood {
    /// Negative binomial — per-gene overdispersed counts (library-scaled,
    /// learnable dispersion φ). The default; best for raw over-dispersed
    /// count data.
    Nb,
    /// Multinomial / categorical — depth-invariant composition, full-vocab
    /// softmax cross-entropy at masked positions (no φ, no library term).
    /// The likelihood the generative ELBO path (`train_mixed`) also uses, so
    /// an ELBO-vs-masked comparison under this option isolates the objective.
    Multinomial,
}

pub struct MaskedTrainOpts {
    pub mask_schedule: MaskSchedule,
    /// Per-gene likelihood for the masked imputation loss.
    pub likelihood: MaskedLikelihood,
    /// **Masked-VAE mode.** When `true`, the encoder emits a reparameterized
    /// **Gaussian** latent `z` (no log-softmax simplex projection) and the loss
    /// gains a `kl_weight · KL(z)` term. `exp(z)` then plays the role of the
    /// per-topic intensities in the NB head (`μ_g = ℓ·Σ_t exp(z_t)·β_{t,g}`), so
    /// the decoder is reused unchanged. `false` = the legacy simplex-`θ`
    /// masked-topic model (deterministic, no KL). NB head only.
    pub latent_gaussian: bool,
    /// KL weight `β` for the Gaussian latent (ignored unless `latent_gaussian`).
    pub kl_weight: f64,
}

impl Default for MaskedTrainOpts {
    fn default() -> Self {
        Self {
            mask_schedule: MaskSchedule::Fixed,
            likelihood: MaskedLikelihood::Nb,
            latent_gaussian: false,
            kl_weight: 1.0,
        }
    }
}

/// Per-level training triple: `(encoder input, optional batch null, decoder target)`.
///
/// All three are borrowed so callers can reuse the same `Mat` as both input
/// and target without cloning a multi-GB matrix.
pub type LevelData<'a> = (&'a Mat, Option<&'a Mat>, &'a Mat);

/// Build per-level [`IndexedInMemoryData`] loaders from pre-built level data.
pub fn build_indexed_loaders(
    level_data: &[LevelData],
    config: &IndexedTrainConfig,
) -> anyhow::Result<Vec<IndexedInMemoryData>> {
    level_data
        .iter()
        .map(|&(mixed, batch, target)| {
            let mut loader = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: mixed,
                input_null: batch,
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

/// Shuffle the loader's row order then precompute all minibatches.
pub fn shuffle_and_precompute(
    loader: &mut IndexedInMemoryData,
    minibatch_size: usize,
) -> anyhow::Result<()> {
    loader.shuffle_minibatch(minibatch_size);
    loader.precompute_all_minibatches()
}

/// Multinomial ETM decoder exposing its full `α·ρᵀ [K, D]` logits for the
/// anchor-prior cross-entropy, computed lazily inside [`apply_anchor_ce`] only
/// when the penalty is active. (The masked NB trainer already holds these
/// logits for its log-partition, so it calls [`apply_anchor_ce_from_logits`]
/// directly rather than through this trait.)
trait AnchorLogitsKd {
    fn anchor_logits_kd(&self) -> candle_core::Result<Tensor>;
}
impl AnchorLogitsKd for EmbeddedTopicDecoder {
    fn anchor_logits_kd(&self) -> candle_core::Result<Tensor> {
        self.full_logits_kd()
    }
}

/// Cross-entropy penalty `−λ · mean_K Σ_D prior · log_softmax_D(logits)`
/// added to the loss. Anchors topic indices to the supplied per-topic
/// gene-prior distribution; breaks the K-way permutation symmetry of
/// the ETM-factorized β. No-op when `prior` is `None` or `lambda ≤ 0`.
fn apply_anchor_ce<D: AnchorLogitsKd>(
    loss: candle_core::Tensor,
    decoder: &D,
    prior: Option<&candle_core::Tensor>,
    lambda: f32,
) -> candle_core::Result<candle_core::Tensor> {
    let Some(prior) = prior else {
        return Ok(loss);
    };
    if lambda <= 0.0 {
        return Ok(loss);
    }
    let logits_kd = decoder.anchor_logits_kd()?;
    apply_anchor_ce_from_logits(loss, &logits_kd, prior, lambda)
}

/// Anchor-CE core operating on precomputed `[K, D]` logits. Lets a caller that
/// already has `full_logits_kd` (e.g. the masked trainer, which also needs it
/// for the NB log-partition) avoid recomputing the `[K, D]` product.
fn apply_anchor_ce_from_logits(
    loss: candle_core::Tensor,
    logits_kd: &candle_core::Tensor,
    prior: &candle_core::Tensor,
    lambda: f32,
) -> candle_core::Result<candle_core::Tensor> {
    let log_prob = candle_nn::ops::log_softmax(logits_kd, logits_kd.rank() - 1)?;
    let ce = (prior * &log_prob)?.sum(1)?.neg()?;
    let pen = (ce.mean_all()? * f64::from(lambda))?;
    loss + pen
}

/// Apply a per-gene posterior-mean delta correction to `bulk_full`.
///
/// `delta_row` is `[1, G]` (a row of per-gene corrections, broadcast across
/// samples). Output values are clamped at `min_val` to avoid zero rates.
fn apply_column_delta(bulk_full: &Mat, delta_row: &Mat, min_val: f32) -> Mat {
    let mut out = bulk_full.clone();
    let g = bulk_full.nrows();
    let b = bulk_full.ncols();
    for j in 0..b {
        for i in 0..g {
            let v = bulk_full[(i, j)] - delta_row[(0, i)];
            out[(i, j)] = v.max(min_val);
        }
    }
    out
}

/// V-cycle / mini-batch training of the shared indexed encoder + per-level
/// decoders.
///
/// `level_data[i] = (input_i, batch_i, target_i)` — caller's responsibility
/// to assemble. `decoders[i]` is the per-level decoder; `encoder` is shared
/// across levels.
///
/// `bulk_with_deltas` is the senna-specific bulk-deconvolution input;
/// callers (e.g. pinto) that don't have bulk just pass `None`.
pub fn train_mixed(
    level_data: &[LevelData],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[EmbeddedTopicDecoder],
    config: &IndexedTrainConfig,
    bulk_with_deltas: Option<BulkWithDeltas>,
) -> anyhow::Result<TrainScores> {
    let num_levels = level_data.len();
    let total_epochs = config.epochs;

    for (level, (&(mixed, _, _), decoder)) in level_data.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {}",
            level + 1,
            num_levels,
            mixed.ncols(),
            decoder.dim_obs(),
        );
    }

    info!("Mixed multi-level training: {num_levels} levels, {total_epochs} epochs");

    let adam_vars: Vec<Var> = match config.frozen_feature_var {
        None => config.parameters.all_vars(),
        Some(name) => {
            let trainable = crate::frozen_features::trainable_vars(config.parameters, &[name]);
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
    let prog_bar = labeled_bar("Epochs", total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);
    let mut timers = PhaseTimers::default();

    let mut data_loaders = build_indexed_loaders(level_data, config)?;

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
                loss = apply_anchor_ce(
                    loss,
                    decoder,
                    config.anchor_prior_per_level.map(|p| &p[level]),
                    config.anchor_penalty,
                )?;
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
                    let rho_l2 = encoder
                        .feature_embeddings()
                        .sqr()?
                        .mean_all()?
                        .affine(f64::from(config.feature_embedding_l2), 0.0)?;
                    loss = (loss + rho_l2)?;
                }
                loss = apply_anchor_ce(
                    loss,
                    finest_decoder,
                    config.anchor_prior_per_level.map(|p| &p[finest_idx]),
                    config.anchor_penalty,
                )?;
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

/// Masked-imputation training (no ELBO / no KL) for the embedded topic model.
///
/// Per minibatch, the cell's top-K genes are randomly split into **visible**
/// (encoder input) and **masked** (held-out targets). The encoder pools the
/// visible genes into a deterministic `log θ`; the NB embedded-topic decoder
/// imputes the masked genes (`μ = residual·ℓ·θβ`) and the loss is the NB
/// log-likelihood on masked positions only. No posterior, no KL → no
/// posterior collapse. Pseudobulk masking also simulates the PB→single-cell
/// sparsity the amortized encoder must handle at inference.
///
/// Separate from [`train_mixed`] (the ELBO path, kept for `pinto lc-etm`).
pub fn train_masked(
    level_data: &[LevelData],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[EmbeddedNbTopicDecoder],
    config: &IndexedTrainConfig,
    mask_fraction: f64,
    opts: &MaskedTrainOpts,
) -> anyhow::Result<TrainScores> {
    let num_levels = level_data.len();
    let total_epochs = config.epochs;

    for (level, (&(mixed, _, _), decoder)) in level_data.iter().zip(decoders.iter()).enumerate() {
        info!(
            "Level {}/{}: {} samples, decoder dim {} (masked-imputation ETM)",
            level + 1,
            num_levels,
            mixed.ncols(),
            decoder.dim_obs(),
        );
    }
    info!(
        "Masked-imputation training: {num_levels} levels, {total_epochs} epochs, mask={mask_fraction}"
    );

    let adam_vars: Vec<Var> = match config.frozen_feature_var {
        None => config.parameters.all_vars(),
        Some(name) => crate::frozen_features::trainable_vars(config.parameters, &[name]),
    };
    let mut adam = AdamW::new(
        adam_vars,
        candle_nn::ParamsAdamW {
            lr: f64::from(config.learning_rate),
            weight_decay: f64::from(config.weight_decay),
            ..Default::default()
        },
    )?;
    let prog_bar = labeled_bar("Epochs", total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    // No KL in the masked objective; keep a zero column the same length as
    // `llik` so `TrainScores::to_parquet` sees equal-length columns.
    let mut kl_trace = Vec::with_capacity(total_epochs);
    let mut data_loaders = build_indexed_loaders(level_data, config)?;

    for epoch in 0..total_epochs {
        for loader in data_loaders.iter_mut() {
            shuffle_and_precompute(loader, config.minibatch_size)?;
        }

        let mut metric_tot = 0f32;
        let mut metric_cnt = 0f32;
        let mut kl_tot = 0f32;
        let mut kl_cnt = 0f32;

        for (level, loader) in data_loaders.iter().enumerate() {
            let decoder = &decoders[level];

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b).to_device(config.dev)?;

                // Visible/masked split over the cell's real (value>0) top-K.
                // Pads (value==0) are neither visible (no ρ₀ contamination)
                // nor scored.
                let real = mb.input_values.gt(0.0)?.to_dtype(candle_core::DType::F32)?;
                let rate = match opts.mask_schedule {
                    MaskSchedule::Fixed => mask_fraction,
                    MaskSchedule::Uniform { lo, hi } => {
                        lo + rand::rng().random::<f64>() * (hi - lo)
                    }
                };
                let rnd = Tensor::rand(0f32, 1f32, mb.input_values.shape(), config.dev)?;
                let drop = rnd.lt(rate)?.to_dtype(candle_core::DType::F32)?;
                let masked = real.mul(&drop)?;
                let visible = (&real - &masked)?;

                // Masked-VAE: reparameterized Gaussian `z` (no softmax) + KL.
                // Legacy masked-topic: deterministic simplex `log θ`, no KL.
                // In both cases the NB head reads this as its per-topic
                // intensity log — `exp(z)` for the VAE, `θ` for the topic model.
                let (log_z, kl_opt) = if opts.latent_gaussian {
                    let (z, kl) = encoder.forward_indexed_masked_gaussian(
                        &mb.input_indices,
                        &mb.input_values,
                        mb.input_values_null.as_ref(),
                        mb.input_values_mean.as_ref(),
                        &visible,
                        true,
                    )?;
                    (z, Some(kl))
                } else {
                    let log_z = encoder.forward_indexed_masked(
                        &mb.input_indices,
                        &mb.input_values,
                        mb.input_values_null.as_ref(),
                        mb.input_values_mean.as_ref(),
                        &visible,
                        true,
                    )?;
                    (smooth_topics(log_z, config.topic_smoothing)?, None)
                };

                // full_kd (α·ρᵀ [K,D]) — the per-topic log-partition for the NB
                // head, and (when active) the anchor-prior CE.
                let anchor_active =
                    config.anchor_penalty > 0.0 && config.anchor_prior_per_level.is_some();
                let full_kd = decoder.full_logits_kd()?;

                let (mut loss, batch_metric, batch_count) = {
                    let logz_11k = EmbeddedNbTopicDecoder::log_partition_from_logits(&full_kd)?;
                    let lib_n1 = (mb.input_values.sum_keepdim(1)? + 1.0)?;
                    let target = MaskedNbTarget {
                        indices: &mb.input_indices,
                        residual: mb.input_values_null.as_ref(),
                        values: &mb.input_values,
                        lib: &lib_n1,
                        mask: &masked,
                    };
                    let llik = match opts.likelihood {
                        MaskedLikelihood::Nb => {
                            decoder.impute_masked_nb(&log_z, &target, &logz_11k)?
                        }
                        MaskedLikelihood::Multinomial => {
                            decoder.impute_masked_multinomial(&log_z, &target, &logz_11k)?
                        }
                    };
                    let m = llik.sum_all()?.to_scalar::<f32>()?;
                    let c = masked.sum_all()?.to_scalar::<f32>()?;
                    (llik.mean_all()?.neg()?, m, c)
                };
                // Masked-VAE KL bottleneck: β · mean_N KL(z ‖ N(0, I)).
                if let Some(kl) = kl_opt {
                    kl_tot += kl.sum_all()?.to_scalar::<f32>()?;
                    kl_cnt += kl.dim(0)? as f32;
                    loss = (loss + kl.mean_all()?.affine(opts.kl_weight, 0.0)?)?;
                }
                if config.feature_embedding_l2 > 0.0 && config.frozen_feature_var.is_none() {
                    let rho_l2 = encoder
                        .feature_embeddings()
                        .sqr()?
                        .mean_all()?
                        .affine(f64::from(config.feature_embedding_l2), 0.0)?;
                    loss = (loss + rho_l2)?;
                }
                if anchor_active {
                    if let Some(prior) = config.anchor_prior_per_level.map(|p| &p[level]) {
                        loss = apply_anchor_ce_from_logits(
                            loss,
                            &full_kd,
                            prior,
                            config.anchor_penalty,
                        )?;
                    }
                }

                let grads = loss.backward()?;
                clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))?;

                metric_tot += batch_metric;
                metric_cnt += batch_count;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        let per_metric = if metric_cnt > 0.0 {
            metric_tot / metric_cnt
        } else {
            0.0
        };
        let per_kl = if kl_cnt > 0.0 { kl_tot / kl_cnt } else { 0.0 };
        llik_trace.push(per_metric);
        kl_trace.push(per_kl);
        prog_bar.inc(1);
        if log::log_enabled!(log::Level::Info) {
            let kl_msg = if opts.latent_gaussian {
                format!(" kl/cell={per_kl:.4}")
            } else {
                String::new()
            };
            info!("[epoch {epoch}] masked-NB llik/gene={per_metric:.4}{kl_msg}");
        }
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
    info!("done masked-imputation training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}
