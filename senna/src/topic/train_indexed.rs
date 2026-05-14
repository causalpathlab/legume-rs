use super::common::{apply_column_delta, sample_collapsed_data};
use super::eval_indexed::{dense_to_indexed, refine_indexed_topic_proportions, PerGeneContext};
use crate::embed_common::*;
use crate::logging::new_progress_bar;

use super::graph_likelihood::{graph_loss, PoissonGraphConfig};
use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use candle_util::candle_decoder_embedded_topic::EmbeddedTopicDecoder;
use candle_util::candle_encoder_indexed::IndexedEmbeddingEncoder;
use candle_util::candle_indexed_data_loader::*;
use candle_util::candle_indexed_model_traits::*;
use candle_util::candle_topic_refinement::TopicRefinementConfig;
use matrix_param::dmatrix_gamma::GammaMatrix;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Wall-clock breakdown of the indexed-topic training hot loop.
///
/// Candle's CPU backend is eager, so an `Instant` straddling each phase
/// attributes time accurately (no async kernels to sync). Used to decide
/// empirically whether the large-`H` bottleneck is the forward pass
/// (→ pooling restructure) or the optimizer step (→ lazy ρ update).
#[derive(Default)]
struct PhaseTimers {
    precompute: Duration,
    encoder_fwd: Duration,
    decoder_fwd: Duration,
    backward: Duration,
    optimize: Duration,
}

impl PhaseTimers {
    fn log_summary(&self) {
        let total = self.precompute
            + self.encoder_fwd
            + self.decoder_fwd
            + self.backward
            + self.optimize;
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
    /// Run the BKN graph likelihood only for the first N epochs, then drop
    /// it. 0 = never drop (graph term stays active for all epochs).
    pub graph_warmup_epochs: usize,
    /// The shared ρ `[D, H]` feature-embedding `Var`. When `lazy_rho` is
    /// set, ρ is pulled out of the stock `AdamW` and stepped sparsely via
    /// [`LazyRhoAdamW`] over each minibatch's touched rows.
    pub rho_var: &'a Var,
    /// Use the lazy (touched-row-only) optimizer for ρ. Disable to fall
    /// back to dense `AdamW` over all of ρ — for A/B benchmarking.
    pub lazy_rho: bool,
}

/// Lazy (sparse) AdamW for the shared ρ `[D, H]` embedding table.
///
/// Stock `AdamW` rewrites all of ρ — plus two `[D, H]` moment buffers —
/// every minibatch, even though only the `touched_rho_indices` rows carry
/// a nonzero gradient. This applies the identical AdamW update math to
/// just the touched rows via `index_select` → update → `index_add`,
/// turning the per-step cost from O(D·H) to O(T·H), T = touched rows.
///
/// Differences from dense `AdamW` (both standard for sparse-embedding
/// optimizers — cf. PyTorch `SparseAdam` / TF `LazyAdam`):
///  - untouched rows are not weight-decayed this step (embedding tables
///    are conventionally excluded from weight decay anyway);
///  - momentum on untouched rows is not advanced — no `beta^gap`
///    catch-up — so a long-dormant row carries slightly larger momentum
///    when next touched. Negligible for rows touched every few steps.
struct LazyRhoAdamW {
    rho: Var,
    first_moment: Tensor,
    second_moment: Tensor,
    step_t: usize,
    params: ParamsAdamW,
}

impl LazyRhoAdamW {
    fn new(rho: Var, learning_rate: f32) -> anyhow::Result<Self> {
        let first_moment = Tensor::zeros(rho.shape(), rho.dtype(), rho.device())?;
        let second_moment = Tensor::zeros(rho.shape(), rho.dtype(), rho.device())?;
        let params = ParamsAdamW {
            lr: f64::from(learning_rate),
            ..Default::default()
        };
        Ok(Self {
            rho,
            first_moment,
            second_moment,
            step_t: 0,
            params,
        })
    }

    /// Apply one AdamW step to the rows of ρ named by `touched`, using
    /// `grad_rho` (the dense `[D, H]` gradient from `backward()`, mostly
    /// zero) scaled by the scalar `clip_scale` from the global-norm clip.
    fn step_touched(
        &mut self,
        grad_rho: &Tensor,
        touched: &Tensor,
        clip_scale: &Tensor,
    ) -> anyhow::Result<()> {
        self.step_t += 1;
        let p = &self.params;
        let scale_m = 1.0 / (1.0 - p.beta1.powi(self.step_t as i32));
        let scale_v = 1.0 / (1.0 - p.beta2.powi(self.step_t as i32));
        let lr_lambda = p.lr * p.weight_decay;

        let g = grad_rho
            .index_select(touched, 0)?
            .broadcast_mul(clip_scale)?; // [T, H]
        let m_t = self.first_moment.index_select(touched, 0)?;
        let v_t = self.second_moment.index_select(touched, 0)?;
        let theta_t = self.rho.as_tensor().index_select(touched, 0)?;

        let next_m = (m_t.affine(p.beta1, 0.0)? + g.affine(1.0 - p.beta1, 0.0)?)?;
        let next_v = (v_t.affine(p.beta2, 0.0)? + g.sqr()?.affine(1.0 - p.beta2, 0.0)?)?;
        let m_hat = next_m.affine(scale_m, 0.0)?;
        let v_hat = next_v.affine(scale_v, 0.0)?;
        let adjusted = (m_hat / v_hat.sqrt()?.affine(1.0, p.eps)?)?;
        let next_theta =
            (theta_t.affine(1.0 - lr_lambda, 0.0)? - adjusted.affine(p.lr, 0.0)?)?;

        // Scatter per-row deltas back into the full [D, H] tensors.
        // `detach()` is load-bearing: the moments are plain tensors, so
        // without it each `index_add` would chain onto the previous step's
        // graph and the autograd tape would grow unboundedly across the
        // run. ρ is a `Var` — `set` copies in-place, so it stays a leaf.
        self.first_moment = self
            .first_moment
            .index_add(touched, &(next_m - &m_t)?, 0)?
            .detach();
        self.second_moment = self
            .second_moment
            .index_add(touched, &(next_v - &v_t)?, 0)?
            .detach();
        let rho_next = self
            .rho
            .as_tensor()
            .index_add(touched, &(next_theta - &theta_t)?, 0)?;
        self.rho.set(&rho_next)?;
        Ok(())
    }
}

/// Global-L2-norm clip + dense `AdamW` step from a precomputed `GradStore`.
/// Mirrors [`crate::embed_common::clip_grads_and_step`] but takes the
/// already-computed grads so the caller can time `backward()` separately.
fn clip_and_step_dense(
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

/// Global-L2-norm clip + step from a precomputed `GradStore`, with ρ
/// handled by `LazyRhoAdamW`.
///
/// ρ is excluded from `adam`'s var set; its gradient is pulled from the
/// `GradStore` and stepped lazily over `touched`. The global grad norm
/// still includes ρ's contribution, computed from the touched rows only
/// (`O(T·H)`) — the untouched rows of ρ's dense grad are exactly zero, so
/// the sum-of-squares is identical to the dense computation.
fn clip_and_step_lazy_rho(
    adam: &mut AdamW,
    lazy_rho: &mut LazyRhoAdamW,
    mut grads: candle_core::backprop::GradStore,
    max_norm: f64,
    touched: &Tensor,
) -> anyhow::Result<()> {
    let rho_id = lazy_rho.rho.id();
    let rho_grad = grads.get_id(rho_id).cloned();
    let dev = lazy_rho.rho.device().clone();
    let ids: Vec<_> = grads.get_ids().copied().collect();

    // Global sum-of-squares: dense for non-ρ params, touched-only for ρ.
    let mut sumsq: Option<Tensor> = None;
    for id in &ids {
        if *id == rho_id {
            continue;
        }
        if let Some(g) = grads.get_id(*id) {
            let s = g.sqr()?.sum_all()?;
            sumsq = Some(match sumsq {
                None => s,
                Some(prev) => (prev + s)?,
            });
        }
    }
    if let Some(ref rg) = rho_grad {
        let s = rg.index_select(touched, 0)?.sqr()?.sum_all()?;
        sumsq = Some(match sumsq {
            None => s,
            Some(prev) => (prev + s)?,
        });
    }

    // Clip scale = min(1, max_norm / (‖g‖ + eps)); 1 when clipping is off.
    let scale = match (max_norm > 0.0, &sumsq) {
        (true, Some(sumsq)) => {
            let inv = sumsq.sqrt()?.affine(1.0, 1e-6)?.powf(-1.0)?;
            inv.affine(max_norm, 0.0)?.clamp(0.0_f64, 1.0_f64)?
        }
        _ => Tensor::ones((), candle_core::DType::F32, &dev)?,
    };

    // Scale non-ρ grads in place, then step the stock optimizer.
    for id in &ids {
        if *id == rho_id {
            continue;
        }
        if let Some(g) = grads.get_id(*id) {
            let scaled = g.broadcast_mul(&scale)?;
            grads.insert_id(*id, scaled);
        }
    }
    adam.step(&grads)?;

    if let Some(rg) = rho_grad {
        lazy_rho.step_touched(&rg, touched, &scale)?;
    }
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

    // ρ is stepped by `LazyRhoAdamW` when `lazy_rho` is set, so it must be
    // excluded from the stock optimizer's var set to avoid a double update.
    let rho_id = config.rho_var.id();
    let adam_vars: Vec<Var> = if config.lazy_rho {
        config
            .parameters
            .all_vars()
            .into_iter()
            .filter(|v| v.id() != rho_id)
            .collect()
    } else {
        config.parameters.all_vars()
    };
    let mut adam = AdamW::new_lr(adam_vars, f64::from(config.learning_rate))?;
    let mut lazy_rho = if config.lazy_rho {
        Some(LazyRhoAdamW::new(
            config.rho_var.clone(),
            config.learning_rate,
        )?)
    } else {
        None
    };
    info!(
        "ρ optimizer: {}",
        if config.lazy_rho {
            "lazy (touched-row sparse AdamW)"
        } else {
            "dense AdamW"
        }
    );
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

            // One-time diagnostic: how large is the per-minibatch touched
            // ρ-row set relative to D? Determines the lazy-optimizer ceiling.
            if epoch == 0 && loader.num_minibatch() > 0 {
                let t = loader.minibatch_cached(0).touched_rho_indices.dim(0)?;
                let d = config.rho_var.dim(0)?;
                info!(
                    "level {} mb0: touched {} / {} ρ rows ({:.0}%)",
                    level + 1,
                    t,
                    d,
                    100.0 * t as f64 / d as f64
                );
            }

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b).to_device(config.dev)?;

                let t_enc = Instant::now();
                let (log_z_nk, kl) = encoder.forward_indexed_t(
                    &mb.input_indices,
                    &mb.input_values,
                    mb.input_values_null.as_ref(),
                    mb.input_values_mean.as_ref(),
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
                let loss = (&kl - &llik)?.mean_all()?;
                timers.decoder_fwd += t_dec.elapsed();

                let t_bwd = Instant::now();
                let grads = loss.backward()?;
                timers.backward += t_bwd.elapsed();

                let t_opt = Instant::now();
                match lazy_rho.as_mut() {
                    Some(lr) => clip_and_step_lazy_rho(
                        &mut adam,
                        lr,
                        grads,
                        f64::from(config.grad_clip),
                        &mb.touched_rho_indices,
                    )?,
                    None => clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))?,
                }
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
                let t_enc = Instant::now();
                let (log_z_nk, kl) = encoder.forward_indexed_t(
                    &mb.input_indices,
                    &mb.input_values,
                    None,
                    mb.input_values_mean.as_ref(),
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
                let loss = (&kl - &llik)?.mean_all()?;
                timers.decoder_fwd += t_dec.elapsed();

                let t_bwd = Instant::now();
                let grads = loss.backward()?;
                timers.backward += t_bwd.elapsed();

                let t_opt = Instant::now();
                match lazy_rho.as_mut() {
                    Some(lr) => clip_and_step_lazy_rho(
                        &mut adam,
                        lr,
                        grads,
                        f64::from(config.grad_clip),
                        &mb.touched_rho_indices,
                    )?,
                    None => clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))?,
                }
                timers.optimize += t_opt.elapsed();

                llik_tot += llik.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl.sum_all()?.to_scalar::<f32>()?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        // Once-per-epoch BKN graph-likelihood gradient step (Poisson on
        // observed feature-pair edges, closed-form non-edge partition).
        // Cheap: O(|E|·H + D·H). Operates on the decoupled ρ_graph; the
        // encoder's ρ feels the graph only via the per-minibatch tether
        // added inside `add_graph_tether`.
        //
        // `graph_warmup_epochs > 0` gates this step alone — after warmup,
        // ρ_graph is effectively frozen at its best-informed state and the
        // tether keeps streaming structural information to ρ_enc without
        // continuing to pull ρ_graph further into a low-rank attractor.
        let graph_active = match config.graph_warmup_epochs {
            0 => true,
            n => epoch < n,
        };
        if let (true, Some(cfg)) = (graph_active, graph_cfg) {
            let g_loss = graph_loss(cfg)?;
            clip_grads_and_step(&mut adam, &g_loss, f64::from(config.grad_clip))?;
            let d = cfg.rho_graph.dim(0)? as f32;
            info!(
                "[epoch {}] graph_nll/gene={:.4}",
                epoch,
                g_loss.to_scalar::<f32>()? / d.max(1.0)
            );
        } else if epoch == config.graph_warmup_epochs && graph_cfg.is_some() {
            info!(
                "[epoch {}] dropping graph likelihood (warmup={} reached); \
                 ρ_graph frozen, tether continues",
                epoch, config.graph_warmup_epochs,
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
