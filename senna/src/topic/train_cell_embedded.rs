//! Training loop for `senna cell-embedded-topic` — the hierarchical
//! cell→PB pooling topic model.
//!
//! Same model *type* as `indexed-topic` (shared ρ `[D, H]`, ETM-factorized
//! decoder, multi-level PB training, lazy-ρ optimizer). The only change is
//! structural: the encoder pools the genuinely sparse single-cell atoms of
//! each PB instead of consuming a dense PB profile, so the per-minibatch
//! touched ρ-row set shrinks to single-digit % of D and the lazy-ρ
//! optimizer pays off end-to-end.
//!
//! The optimizer/timer helpers (`LazyRhoAdamW`, `clip_and_step_*`,
//! `PhaseTimers`) are reused verbatim from [`super::train_indexed`].
//!
//! The bulk-data path is deferred: bulk samples have no member cells, so
//! they'd each be a singleton group. Revisit when bulk joint deconvolution
//! is needed for this model.

use super::common::sample_collapsed_data;
use super::train_indexed::{clip_and_step_dense, PhaseTimers};
use crate::embed_common::*;
use crate::logging::new_progress_bar;

use candle_core::{Device, Tensor, Var};
use candle_nn::AdamW;
use candle_util::candle_cell_grouped_data_loader::*;
use candle_util::candle_decoder_embedded_topic::EmbeddedTopicDecoder;
use candle_util::candle_encoder_cell_embedded::CellEmbeddedEncoder;
use candle_util::candle_indexed_data_loader::{csc_columns_to_indexed_samples, IndexedSample};
use candle_util::candle_indexed_model_traits::*;
use indicatif::ParallelProgressIterator;
use log::warn;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

////////////////////////////////////////////////////////////////////////
// Config
////////////////////////////////////////////////////////////////////////

pub(crate) struct CellEmbeddedTrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub learning_rate: f32,
    pub topic_smoothing: f64,
    /// Encoder foreground context window (top-K features per **cell**).
    pub fg_context_size: usize,
    /// Encoder background context window (top-K features per **PB**).
    pub bg_context_size: usize,
    /// Decoder context window (top-K features per PB).
    pub dec_context_size: usize,
    /// Cap on member cells sampled per PB per minibatch (0 = use all).
    /// Bounds `M` (and the touched-ρ union) at coarse PB levels; a fresh
    /// subsample is drawn every epoch — that resampling is the within-PB
    /// SGD stochasticity.
    pub fg_cells_per_pb: usize,
    pub stop: &'a AtomicBool,
    /// Per-gene weights used to score top-K candidates (cell + BG + decoder).
    pub shortlist_weights: &'a [f32],
    /// Per-gene NB-Fisher info weight gathered into the FG/BG/decoder packs.
    pub feature_fisher_weights: &'a [f32],
    /// Global L2 gradient norm clip per minibatch (0 = off).
    pub grad_clip: f32,
}

////////////////////////////////////////////////////////////////////////
// Cell extraction (level-independent, shared via Arc)
////////////////////////////////////////////////////////////////////////

/// Extract the per-cell sparse top-K samples + library-size factors once.
///
/// Single cells are the genuinely sparse atoms — read them straight from
/// the sparse backend as CSC blocks, top-K-select per column, and sum each
/// column's raw nonzeros for the DC-Poisson degree-correction factor
/// `s_c = Σ_g y_cg` (floored ≥ 1 so the encoder's per-cell division can't
/// blow up). The result is level-independent and shared across every PB
/// level by `Arc`.
pub(crate) fn extract_cell_samples(
    data_vec: &SparseIoVec,
    shortlist_weights: &[f32],
    fg_context_size: usize,
    block_size: Option<usize>,
) -> anyhow::Result<(Vec<IndexedSample>, Vec<f32>)> {
    let n_cells = data_vec.num_columns();
    let n_features = data_vec.num_rows();
    let jobs = create_jobs(n_cells, n_features, block_size);

    let prog_bar = new_progress_bar(jobs.len() as u64);
    let mut chunks: Vec<(usize, Vec<IndexedSample>, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(prog_bar.clone())
        .map(|&(lb, ub)| -> anyhow::Result<_> {
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let samples =
                csc_columns_to_indexed_samples(&csc, shortlist_weights, fg_context_size, None);
            let sizes: Vec<f32> = (0..csc.ncols())
                .map(|j| csc.col(j).values().iter().sum::<f32>().max(1.0))
                .collect();
            Ok((lb, samples, sizes))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    prog_bar.finish_and_clear();

    chunks.sort_by_key(|c| c.0);
    let mut cell_samples = Vec::with_capacity(n_cells);
    let mut cell_size_factor = Vec::with_capacity(n_cells);
    for (_, samples, sizes) in chunks {
        cell_samples.extend(samples);
        cell_size_factor.extend(sizes);
    }
    anyhow::ensure!(
        cell_samples.len() == n_cells,
        "extracted {} cell samples but data has {} cells",
        cell_samples.len(),
        n_cells,
    );
    Ok((cell_samples, cell_size_factor))
}

////////////////////////////////////////////////////////////////////////
// Per-level loader construction
////////////////////////////////////////////////////////////////////////

fn build_cell_grouped_loaders(
    collapsed_levels: &[CollapsedOut],
    cell_to_pb_per_level: &[Vec<usize>],
    cell_samples: &Arc<Vec<IndexedSample>>,
    cell_size_factor: &Arc<Vec<f32>>,
    n_features: usize,
    config: &CellEmbeddedTrainConfig,
) -> anyhow::Result<Vec<CellGroupedInMemoryData>> {
    let mut warned_no_bg = false;
    collapsed_levels
        .iter()
        .zip(cell_to_pb_per_level.iter())
        .map(|(collapsed, c2p)| {
            // FG = member cells; BG = the PB μ_residual profile; decoder
            // target = the batch-adjusted (or observed) PB profile.
            let (mixed, batch, target) = sample_collapsed_data(collapsed)?;
            let bg_source = match batch {
                Some(b) => b,
                None => {
                    if !warned_no_bg {
                        warn!(
                            "no μ_residual at this level (single-batch / --ignore-batch); \
                             using the observed PB profile as the encoder background"
                        );
                        warned_no_bg = true;
                    }
                    mixed
                }
            };
            CellGroupedInMemoryData::new(CellGroupedArgs {
                cell_samples: cell_samples.clone(),
                cell_size_factor: cell_size_factor.clone(),
                cell_to_pb: c2p,
                bg_source: &bg_source,
                target_source: &target,
                n_features,
                fg_context_size: config.fg_context_size,
                bg_context_size: config.bg_context_size,
                dec_context_size: config.dec_context_size,
                fg_cells_per_pb: config.fg_cells_per_pb,
                shortlist_weights: config.shortlist_weights,
                feature_fisher_weights: config.feature_fisher_weights,
            })
        })
        .collect()
}

////////////////////////////////////////////////////////////////////////
// Training loop
////////////////////////////////////////////////////////////////////////

/// Multi-level cell-embedded topic training. Same epoch/level/minibatch
/// skeleton as [`super::train_indexed::train_mixed`]; the encoder forward
/// pass swaps to `forward_cells_t` over a cell-grouped minibatch, and the
/// bulk path is skipped (deferred).
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_mixed_cell(
    collapsed_levels: &[CollapsedOut],
    cell_to_pb_per_level: &[Vec<usize>],
    cell_samples: Arc<Vec<IndexedSample>>,
    cell_size_factor: Arc<Vec<f32>>,
    encoder: &CellEmbeddedEncoder,
    decoders: &[EmbeddedTopicDecoder],
    config: &CellEmbeddedTrainConfig,
) -> anyhow::Result<TrainScores> {
    let num_levels = collapsed_levels.len();
    anyhow::ensure!(
        cell_to_pb_per_level.len() == num_levels,
        "cell_to_pb_per_level has {} levels, collapsed_levels has {num_levels}",
        cell_to_pb_per_level.len(),
    );
    anyhow::ensure!(
        decoders.len() == num_levels,
        "decoders has {} levels, collapsed_levels has {num_levels}",
        decoders.len(),
    );
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
    info!("Mixed cell-embedded multi-level training: {num_levels} levels, {total_epochs} epochs");

    let adam_vars: Vec<Var> = config.parameters.all_vars();
    let mut adam = AdamW::new_lr(adam_vars, f64::from(config.learning_rate))?;

    let prog_bar = new_progress_bar(total_epochs as u64);
    let mut llik_trace = Vec::with_capacity(total_epochs);
    let mut kl_trace = Vec::with_capacity(total_epochs);
    let mut timers = PhaseTimers::default();

    let mut data_loaders = build_cell_grouped_loaders(
        collapsed_levels,
        cell_to_pb_per_level,
        &cell_samples,
        &cell_size_factor,
        encoder.n_features(),
        config,
    )?;

    for epoch in 0..total_epochs {
        let t_pre = Instant::now();
        for loader in data_loaders.iter_mut() {
            loader.shuffle_minibatch(config.minibatch_size);
            loader.precompute_all_minibatches()?;
        }
        timers.precompute += t_pre.elapsed();

        // Per-epoch loss accumulators kept on-device: pulling a scalar
        // every minibatch forces a GPU→CPU sync per step and stalls the
        // pipeline. Accumulate detached partials, sync once per epoch.
        let mut llik_acc = Tensor::zeros((), candle_core::DType::F32, config.dev)?;
        let mut kl_acc = Tensor::zeros((), candle_core::DType::F32, config.dev)?;
        let mut count_tot = 0f32;
        let mut n_tot = 0usize;

        for (level, loader) in data_loaders.iter().enumerate() {
            let decoder = &decoders[level];
            n_tot += loader.num_data();
            count_tot += loader.total_output_count();

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b).to_device(config.dev)?;

                let t_enc = Instant::now();
                let (log_z_nk, kl) = encoder.forward_cells_t(&mb, true)?;
                let log_z_nk = smooth_topics(log_z_nk, config.topic_smoothing)?;
                timers.encoder_fwd += t_enc.elapsed();

                let t_dec = Instant::now();
                let llik = decoder.forward_indexed(
                    &log_z_nk,
                    &mb.output_union_indices,
                    &mb.output_scatter_pos,
                    &mb.output_values,
                    Some(&mb.output_values_weight),
                    &mb.output_log_q_s,
                )?;
                let loss = (&kl - &llik)?.mean_all()?;
                timers.decoder_fwd += t_dec.elapsed();

                let t_bwd = Instant::now();
                let grads = loss.backward()?;
                timers.backward += t_bwd.elapsed();

                let t_opt = Instant::now();
                clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))?;
                timers.optimize += t_opt.elapsed();

                llik_acc = (llik_acc + llik.sum_all()?.detach())?;
                kl_acc = (kl_acc + kl.sum_all()?.detach())?;

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        // Single GPU→CPU sync per epoch (vs one per minibatch).
        let llik_tot = llik_acc.to_scalar::<f32>()?;
        let kl_tot = kl_acc.to_scalar::<f32>()?;
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
    info!("done mixed cell-embedded multi-level training");
    timers.log_summary();
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}
