//! Masked-imputation trainer for `faba gem-encoder` — the `u + δ → s` objective.
//!
//! Drives [`GemIndexedEncoder`] + per-level [`GemEtmDecoder`] against
//! [`GemIndexedData`] minibatches. There is no ELBO and no KL: the masking IS
//! the regularizer. (A Gaussian/KL head was tried and removed — see
//! [`crate::encoder::gem_encoder::GemIndexedEncoder::forward_latent`].)
//!
//! # The masking schedule — ONE latent
//!
//! Masking is **per gene** at rate `r`, with ONE Bernoulli draw **shared by
//! both tracks**: hiding a gene hides it wholly, and both tracks are scored
//! there. This is the masked-language-modelling convention (BERT, Geneformer,
//! scGPT) and, more importantly here, it is what gives `δ` a monopoly.
//!
//! # Why there is exactly one latent
//!
//! The mature-vs-nascent contrast for cell `c`, gene `g` is
//!
//! ```text
//! log p^s − log p^u  =  ⟨z_c, δ_g⟩          z_c = θ_c·α
//! ```
//!
//! so it can be produced EITHER by `δ` moving (feature side) OR by `z` moving
//! (latent side). Cross-modal masking — hiding one track entirely — hands the
//! encoder two different inputs and therefore a latent delta to use, and
//! nothing in the objective says which channel should carry the signal.
//!
//! Measured, the encoder takes it. Under cross-modal the two-pass
//! `‖Δz‖` was **1.43×** the latent's own spread about its mean: the encoder's
//! response to WHICH TRACK IT SAW exceeded its response to which cell it was
//! looking at. `δ` degenerated with it — the median rank of twelve canonical
//! lineage markers fell from ~200 (of 34 179) under joint masking to ~33 400,
//! i.e. they became the LEAST topic-specific genes in the model. Cross-modal
//! also starves `δ` directly: its `s2u` half scores the nascent track, whose
//! dictionary is `ρ` with no `δ` in it at all.
//!
//! One shared draw removes the channel structurally. Both tracks are predicted
//! from ONE θ, so the only thing that can make `p^s` and `p^u` differ is `δ`.
//!
//! This is the same structure as **DeltaTopic** (Zhang et al., Cell Genomics
//! 2023): a common cellular topic space, with the spliced/unspliced
//! relationship carried on the gene side.
//!
//! The cell-level delta is not lost — it is recovered POST HOC by
//! [`fit_theta_to_track`], which fits θ to each track separately against the
//! frozen dictionaries. Estimating `δ` first and reading the latent delta out of
//! it orders the two rather than letting them compete: a constraint by
//! construction rather than by penalty.
//!
//! # What counts as observed
//!
//! Both tracks use `o = 1[x > 0]`: a zero-valued slot is not scored.
//!
//! This is not obvious for the nascent track, and the first version did the
//! opposite. Top-K is chosen on pooled `s + u`, so a gene can be selected on its
//! mature counts alone with `x^u = 0`, and one can argue that zero is a genuine
//! observation ("no nascent transcription here") rather than a missing entry.
//! That argument is sound in principle but loses in practice: scoring every
//! selected slot means the nascent objective becomes mostly "predict zero",
//! which is easy, and since each track is normalized by its own scored-position
//! count (see Loss below) the easy task collects equal weight. Measured on a
//! six-sample fit, switching nascent to nonzero-only moved the mature likelihood
//! from −4.27 to −2.88 and the splice-ratio check from r = 0.33 to 0.37.
//!
//! Set [`GemTrainOpts::nascent_observed_nonzero_only`] to `false` to score the
//! zeros anyway.
//!
//! # Loss
//!
//! ```text
//! L = − mean_N(LL^τ)      τ = the ONE track this minibatch's mode scores
//!     + λ_δ · mean(δ²)    the splice-ratio ridge
//!     + λ_ρ · mean(ρ²)
//! ```
//!
//! One common per-cell factor, matching senna (`masked_topic`'s
//! `llik.mean_all()`), which is the MLE form. An earlier version divided each
//! track by its own scored-POSITION count; that is not a depth correction — it
//! equalized the two tracks against each other, so the objective converged to a
//! mixture whose weights depended on the realized `n_s/n_u` ratio.

use super::{clip_and_step_dense, smooth_topics};
use crate::data::indexed::{labeled_bar, GemIndexedData, GemMinibatchData};
use crate::decoder::gem_etm::{GemEtmDecoder, GemMaskedTarget, Track};
use crate::encoder::gem_encoder::{GemEncoderInput, GemIndexedEncoder};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use log::{info, warn};
use std::sync::atomic::{AtomicBool, Ordering};

/// Model-type tag persisted in `{out}.gem.json`.
///
/// Kept explicit about the simplex map even though there is now only one, so a
/// reader can still tell a current `{out}.latent.parquet` apart from an older
/// `gem-encoder-sbp` file — those exist on disk and need the OTHER map. Applying
/// the wrong one does not error; it yields a plausible but wrong `θ`.
pub const MODEL_TYPE: &str = "gem-encoder-softmax";

/// Per-gene likelihood for the masked imputation loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemLikelihood {
    /// Negative binomial — over-dispersed counts, library-scaled, learnable φ.
    Nb,
    /// Multinomial — depth-invariant composition. Because the two tracks are
    /// normalized independently, this scores the nascent and mature
    /// *compositions*, the closest thing to modelling the splice ratio directly.
    /// The default: under NB the deepest pseudobulks dominate the objective when
    /// sample depth is uneven, which it usually is.
    Multinomial,
}

pub struct GemTrainOpts {
    pub likelihood: GemLikelihood,
    /// Per-gene hide probability. ONE Bernoulli draw per gene, shared by both
    /// tracks. 0.15 is the masked-language-modelling convention (BERT's rate,
    /// and Geneformer's over transcriptomes).
    pub mask_fraction: f64,
    /// Ridge `λ_δ` on the splice-ratio offset. **Off by default**: `δ_g` is a
    /// per-gene embedding in `R^H` contracted with `α`, so `⟨α_t, δ_g⟩` is
    /// rank-`H` by construction and never a free gene-by-topic matrix. The ridge
    /// is a second constraint on top of that one, and it does not pay for itself
    /// — measured on 3 wt libraries, `λ_δ = 0` recovered canonical markers
    /// BETTER than `λ_δ = 1` (rank 239 vs 276 in `β`, 124 vs 204 in `δ`) for
    /// 0.011 of splice-ratio `r`. See `faba gem-topic --help` for the table.
    pub delta_l2: f32,
    /// Ridge `λ_ρ` on the nascent gene embedding.
    pub feature_embedding_l2: f32,
    /// Uniform smoothing of the simplex before the decoder reads it.
    pub topic_smoothing: f64,
    /// Score the nascent track only where its count is positive. **On** by
    /// default; setting it `false` also scores the zeros, which is defensible in
    /// principle but empirically costs fit — see the module docs.
    pub nascent_observed_nonzero_only: bool,
}

impl Default for GemTrainOpts {
    fn default() -> Self {
        Self {
            likelihood: GemLikelihood::Multinomial,
            mask_fraction: 0.15,
            delta_l2: 0.0,
            feature_embedding_l2: 1.0,
            topic_smoothing: 0.01,
            nascent_observed_nonzero_only: true,
        }
    }
}

pub struct GemTrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip: f32,
    pub stop: &'a AtomicBool,
}

/// Per-epoch training trace. Unlike the single-track `TrainScores`, the two
/// tracks are reported separately — a pooled number hides the failure mode this
/// model actually has (see [`GemScores::mechanism_llik`]).
#[derive(Default)]
pub struct GemScores {
    /// Mature-track log-likelihood per scored position, all modes pooled.
    pub mature_llik: Vec<f32>,
    /// Nascent-track log-likelihood per scored position, all modes pooled.
    pub nascent_llik: Vec<f32>,
    /// Mature-track log-likelihood per scored position **restricted to the
    /// `u→s` mode** — the mechanism's own fit.
    ///
    /// This is the load-bearing diagnostic. If it plateaus while `mature_llik`
    /// keeps improving, the model is imputing mature counts from the mature
    /// context it happens to see in joint mode and has learned nothing about
    /// `u → s`; the velocity would then be noise.
    pub mechanism_llik: Vec<f32>,
    /// `mean(δ²)^½` — collapse toward 0 means the splice-ratio program has been
    /// regularized away and velocity is not trustworthy.
    pub delta_norm: Vec<f32>,
}

impl GemScores {
    /// Last finite entry of a trace.
    ///
    /// A per-mode likelihood is `NaN` for an epoch in which that mode was never
    /// drawn — honest per-epoch, but a caller reading `last()` would then get
    /// `NaN` and silently skip whatever check it was making (every comparison
    /// against `NaN` is false). Summaries and diagnostics go through here.
    #[must_use]
    pub fn last_finite(trace: &[f32]) -> f32 {
        trace
            .iter()
            .rev()
            .copied()
            .find(|x| x.is_finite())
            .unwrap_or(f32::NAN)
    }
}

/// Accumulators for one epoch, kept in one place so the epoch loop stays
/// readable and every reported number has exactly one definition.
#[derive(Default)]
struct EpochAcc {
    mature_ll: f64,
    mature_n: f64,
    nascent_ll: f64,
    nascent_n: f64,
    mech_ll: f64,
    mech_n: f64,
    skipped: usize,
}

impl EpochAcc {
    fn per(ll: f64, n: f64) -> f32 {
        if n > 0.0 {
            (ll / n) as f32
        } else {
            f32::NAN
        }
    }
}

/// Masked visibility for one minibatch: which slots each track shows the
/// encoder, and which it scores.
struct MaskDraw {
    nascent_visible: Tensor,
    mature_visible: Tensor,
    nascent_masked: Tensor,
    mature_masked: Tensor,
}

/// Build the visibility / scoring masks for `mode`.
///
/// `slot_valid` is 1 on real genes and 0 on padding; padding is never visible
/// and never scored, in any mode. Note the asymmetry in what counts as
/// *observed*: mature needs a positive count, nascent does not (see module
/// docs).
fn draw_masks(
    mb: &GemMinibatchData,
    rate: f64,
    nascent_nonzero_only: bool,
    dev: &Device,
) -> candle_core::Result<MaskDraw> {
    let valid = &mb.slot_valid;
    let mature_obs = mb
        .mature_observed
        .gt(0.0)?
        .to_dtype(DType::F32)?
        .mul(valid)?;
    let nascent_obs = if nascent_nonzero_only {
        mb.nascent_observed
            .gt(0.0)?
            .to_dtype(DType::F32)?
            .mul(valid)?
    } else {
        valid.clone()
    };

    // ONE draw, SHARED by both tracks. If the two were masked independently the
    // model could read a gene's own nascent count to predict that same gene's
    // mature count — a same-gene leak `δ` would absorb without learning anything
    // transferable. Shared, a masked gene's `s` and `u` must BOTH come from
    // other genes, and `δ` is the only thing that can make them differ.
    //
    // Rate 0 hides nothing, which is the "everything observed" pattern the
    // inference pass wants — so that path shares this function instead of
    // re-deriving observedness.
    let rnd = Tensor::rand(0f32, 1f32, valid.shape(), dev)?;
    let drop = rnd.lt(rate)?.to_dtype(DType::F32)?;
    let n_masked = nascent_obs.mul(&drop)?;
    let m_masked = mature_obs.mul(&drop)?;
    Ok(MaskDraw {
        nascent_visible: (&nascent_obs - &n_masked)?,
        mature_visible: (&mature_obs - &m_masked)?,
        nascent_masked: n_masked,
        mature_masked: m_masked,
    })
}

/// Map the raw latent to the log-simplex the ETM decoder consumes, then smooth.
///
/// The decoder is a mixture over factors and needs `Σ_t θ_t = 1`; the map is
/// [`theta_log_simplex`], the single owner of that contract.
fn decoder_log_theta(raw_z: &Tensor, topic_smoothing: f64) -> candle_core::Result<Tensor> {
    smooth_topics(theta_log_simplex(raw_z)?, topic_smoothing)
}

/// The ONE map from the raw latent (`{out}.latent.parquet`) to `log θ`.
///
/// Public, and used by every consumer — the trainer, the velocity operator and
/// the cell-embedding writer — because re-deriving it is a silent-failure mode
/// rather than a compile error. That has already happened once: a second head
/// was removed while `faba`'s `operator_velocity` went on applying it, so two
/// per-cell outputs of the same run disagreed about what θ was and the headline
/// `velocity.parquet` used a θ the model never produced.
///
/// A stick-breaking alternative was measured against this one on a six-sample
/// fit, everything else fixed, and lost on every axis: effective rank 1.33 vs
/// **3.14** (K = 20), θ_max 0.88 vs **0.41**, and 61 % vs **0.05 %** of cells
/// with a top factor above 0.9. One run per arm and no seed control, so read it
/// as a large gap rather than a precise one. It is gone; softmax is the map.
pub fn theta_log_simplex(raw_z: &Tensor) -> candle_core::Result<Tensor> {
    candle_nn::ops::log_softmax(raw_z, 1)
}

/// Score one track's masked slots. Returns `(per-cell llik [N], scored count)`.
fn score_track(
    decoder: &GemEtmDecoder,
    log_theta: &Tensor,
    mb: &GemMinibatchData,
    track: Track,
    masked: &Tensor,
    likelihood: GemLikelihood,
) -> candle_core::Result<(Tensor, f32)> {
    // The DECODER TARGET, which is not the encoder's input when batch
    // adjustment is on: the encoder reads the batch-mixed observation with the
    // residual as its batch signal, and the decoder is scored against the
    // batch-free `μ_adjusted`. Falling back to the encoder's own values is the
    // un-adjusted behaviour.
    //
    // THREE mixed/adjusted seams, all deliberate, none of them obvious:
    //
    // 1. WHICH slots are scored comes from the MIXED values — `draw_masks`
    //    derives observedness from `mb.*_values`, not from the target. Since the
    //    collapse defines `support(μ_adjusted) = (observed ∪ imputed) > 0`, the
    //    adjusted support is a SUPERSET, so masking on mixed scores only real
    //    observations and never asks the model to "predict 0" at a position the
    //    adjustment invented. Reverse that and the nascent track would regain
    //    exactly the predict-zero pathology the module header warns about.
    // 2. WHICH genes are in the top-K comes from the mixed values too, since
    //    selection happens in the loader on `input`. The encoder's context
    //    defines the scoring set, which is what keeps the two sides aligned.
    // 3. `lib` below is therefore the ADJUSTED library size, not the observed
    //    one. That is the right scale to compare an adjusted prediction against,
    //    but it only matters to the NB head — the multinomial one ignores `lib`.
    // Every per-track quantity is selected in ONE match, so a track can never be
    // handed another track's tensor. The residual in particular was previously
    // hardcoded to `mature_residual` for both tracks — the exact per-track
    // confusion `GemMinibatchData::nascent_residual` exists to prevent, since a
    // leftover `r^u/r^s` is itself a splice-ratio distortion and lands in `δ`.
    let (values, fallback, residual) = match track {
        Track::Nascent => (
            &mb.nascent_adjusted,
            &mb.nascent_observed,
            &mb.nascent_residual,
        ),
        Track::Mature => (
            &mb.mature_adjusted,
            &mb.mature_observed,
            &mb.mature_residual,
        ),
    };
    let adjusted = values.is_some();
    let values = values.as_ref().unwrap_or(fallback);
    // Each track carries its OWN library size, so the sparse nascent track is
    // not rescaled by the much deeper mature one.
    let lib = (values.sum_keepdim(1)? + 1.0)?;
    let target = GemMaskedTarget {
        indices: &mb.gene_indices,
        // Against an already-adjusted target there is nothing to restore, so
        // the residual must NOT be reapplied — doing so would put the batch
        // effect back into the prediction while the target has none. This is
        // also what makes the multinomial head correct without a special case:
        // it ignores `residual` entirely, which was an asymmetry only while the
        // target was the batch-mixed matrix.
        // Against an already-adjusted target there is nothing to restore.
        residual: if adjusted { None } else { residual.as_ref() },
        values,
        lib: &lib,
        mask: masked,
        values_weight: mb.values_weight.as_ref(),
    };
    let full_kg = decoder.full_logits_kg(track)?;
    let logz = GemEtmDecoder::log_partition_from_logits(&full_kg)?;
    let ll = match likelihood {
        GemLikelihood::Nb => decoder.impute_masked_nb(log_theta, &target, track, &logz)?,
        GemLikelihood::Multinomial => {
            decoder.impute_masked_multinomial(log_theta, &target, track, &logz)?
        }
    };
    let count = masked.sum_all()?.to_scalar::<f32>()?;
    Ok((ll, count))
}

/// Train the splice-aware masked model over per-level pseudobulk data.
///
/// `level_data[i]` is level `i`'s loader; `decoders[i]` its decoder. The encoder
/// is shared across levels.
pub fn train_masked_gem(
    level_data: &mut [GemIndexedData],
    encoder: &GemIndexedEncoder,
    decoders: &[GemEtmDecoder],
    config: &GemTrainConfig,
    opts: &GemTrainOpts,
) -> anyhow::Result<GemScores> {
    anyhow::ensure!(
        level_data.len() == decoders.len(),
        "train_masked_gem: {} data level(s) but {} decoder(s)",
        level_data.len(),
        decoders.len()
    );
    for (level, data) in level_data.iter().enumerate() {
        let (u, s) = data.total_counts();
        info!(
            "Level {}/{}: {} samples, {} genes, top-K {} (nascent {:.3e} / mature {:.3e} counts)",
            level + 1,
            level_data.len(),
            data.num_data(),
            data.n_genes(),
            data.context_size(),
            u,
            s,
        );
    }
    info!(
        "gem-encoder training: {} epochs, per-gene mask rate {} (ONE draw, both tracks)",
        config.epochs, opts.mask_fraction,
    );

    let adam_vars: Vec<Var> = config.parameters.all_vars();
    let mut adam = AdamW::new(
        adam_vars,
        candle_nn::ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )?;

    let prog = labeled_bar("Epochs", config.epochs as u64);
    let mut scores = GemScores::default();

    for epoch in 0..config.epochs {
        for data in level_data.iter_mut() {
            data.shuffle_minibatch(config.minibatch_size);
            data.precompute_all_minibatches()?;
        }

        let mut acc = EpochAcc::default();

        for (level, data) in level_data.iter().enumerate() {
            let decoder = &decoders[level];
            for b in 0..data.num_minibatch() {
                let mb = data.minibatch_cached(b).to_device(config.dev)?;
                let masks = draw_masks(
                    &mb,
                    opts.mask_fraction,
                    opts.nascent_observed_nonzero_only,
                    config.dev,
                )?;

                let input = GemEncoderInput {
                    gene_indices: &mb.gene_indices,
                    nascent_observed: &mb.nascent_observed,
                    mature_observed: &mb.mature_observed,
                    nascent_residual: mb.nascent_residual.as_ref(),
                    mature_residual: mb.mature_residual.as_ref(),
                    nascent_mean: mb.nascent_mean.as_ref(),
                    mature_mean: mb.mature_mean.as_ref(),
                    nascent_visible: &masks.nascent_visible,
                    mature_visible: &masks.mature_visible,
                };
                let raw_z = encoder.forward_latent(&input, true)?;
                let log_theta = decoder_log_theta(&raw_z, opts.topic_smoothing)?;

                // Cross-modal: the HIDDEN track is the scored one, and it is
                // the only one. Computing the other would produce `ll ≡ 0,
                // n = 0` — a full `[K,G]` logits matmul, log-partition and
                // mixture rate for nothing.
                // Both tracks scored at the masked genes: the shared draw hides
                // a gene wholly, so `δ` is the only route between them.
                let (score_mature, score_nascent) = (true, true);
                let mut ll_s = None;
                let mut n_s = 0.0;
                if score_mature {
                    let (ll, n) = score_track(
                        decoder,
                        &log_theta,
                        &mb,
                        Track::Mature,
                        &masks.mature_masked,
                        opts.likelihood,
                    )?;
                    ll_s = Some(ll);
                    n_s = n;
                }
                let mut ll_u = None;
                let mut n_u = 0.0;
                if score_nascent {
                    let (ll, n) = score_track(
                        decoder,
                        &log_theta,
                        &mb,
                        Track::Nascent,
                        &masks.nascent_masked,
                        opts.likelihood,
                    )?;
                    ll_u = Some(ll);
                    n_u = n;
                }

                // MEAN OVER CELLS — a single common factor, matching senna
                // (`masked_topic.rs`: `llik.mean_all()`, with the summed
                // likelihood and scored-position count kept for REPORTING only).
                //
                // This is the MLE form. The earlier per-track `/n_τ` divided
                // each track by its own scored-POSITION count, which is not a
                // library size and not a depth correction — it equalized the two
                // tracks against each other, making the objective converge to a
                // mixture whose weights depend on the realized `n_s/n_u` ratio.
                // `w_u` now carries the track balance explicitly instead.
                //
                // A track with nothing scored in this mode simply contributes no
                // term — accumulating into a detached zero instead would hand
                // `backward()` a constant with no path to any parameter.
                // Only tracks that actually scored contribute a term; a track with
                // nothing scored must not add a detached zero, which would hand
                // `backward()` a constant with no path to any parameter. Nothing
                // scored at all (an all-padding minibatch) means no signal, so
                // skip rather than step on the penalties alone — that would only
                // shrink the embeddings.
                let mut terms: Vec<Tensor> = Vec::with_capacity(2);
                if let (Some(ll), true) = (&ll_s, n_s > 0.0) {
                    terms.push(ll.mean_all()?.neg()?);
                }
                if let (Some(ll), true) = (&ll_u, n_u > 0.0) {
                    terms.push(ll.mean_all()?.neg()?);
                }
                if terms.is_empty() {
                    continue;
                }
                let mut loss = Tensor::stack(&terms, 0)?.sum(0)?;

                if opts.delta_l2 > 0.0 {
                    let pen = encoder
                        .delta_embeddings()
                        .sqr()?
                        .mean_all()?
                        .affine(f64::from(opts.delta_l2), 0.0)?;
                    loss = (loss + pen)?;
                }
                if opts.feature_embedding_l2 > 0.0 {
                    let pen = encoder
                        .feature_embeddings()
                        .sqr()?
                        .mean_all()?
                        .affine(f64::from(opts.feature_embedding_l2), 0.0)?;
                    loss = (loss + pen)?;
                }

                let grads = loss.backward()?;
                if !clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))? {
                    acc.skipped += 1;
                }

                // A track the mode did not score contributes nothing to the
                // trace either — its `n` is 0, so the per-observation mean is
                // unaffected.
                let tot = |ll: &Option<Tensor>| -> candle_core::Result<f64> {
                    Ok(match ll {
                        Some(t) => f64::from(t.sum_all()?.to_scalar::<f32>()?),
                        None => 0.0,
                    })
                };
                let ll_s_tot = tot(&ll_s)?;
                let ll_u_tot = tot(&ll_u)?;
                acc.mature_ll += ll_s_tot;
                acc.mature_n += f64::from(n_s);
                acc.nascent_ll += ll_u_tot;
                acc.nascent_n += f64::from(n_u);
                // "Mechanism" likelihood is now just the mature term: with one
                // shared draw every minibatch predicts mature through `ρ + δ`.
                acc.mech_ll += ll_s_tot;
                acc.mech_n += f64::from(n_s);

                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        let delta_norm = encoder
            .delta_embeddings()
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
            .sqrt();
        scores
            .mature_llik
            .push(EpochAcc::per(acc.mature_ll, acc.mature_n));
        scores
            .nascent_llik
            .push(EpochAcc::per(acc.nascent_ll, acc.nascent_n));
        scores
            .mechanism_llik
            .push(EpochAcc::per(acc.mech_ll, acc.mech_n));
        scores.delta_norm.push(delta_norm);

        prog.set_message(format!(
            "s={:.3} u={:.3}",
            scores.mature_llik[epoch], scores.nascent_llik[epoch]
        ));
        prog.inc(1);

        if acc.skipped > 0 {
            warn!(
                "[epoch {epoch}] skipped {} optimizer step(s): non-finite gradient norm. \
                 Lower --learning-rate or --grad-clip if this persists.",
                acc.skipped
            );
        }
        if log::log_enabled!(log::Level::Info) {
            info!(
                "[epoch {epoch}] llik/s={:.4} llik/u={:.4} llik/mech={:.4} \
                 |delta|={:.4}",
                scores.mature_llik[epoch],
                scores.nascent_llik[epoch],
                scores.mechanism_llik[epoch],
                delta_norm,
            );
        }

        if config.stop.load(Ordering::SeqCst) {
            prog.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(scores);
        }
    }

    prog.finish_and_clear();
    info!("done gem-encoder training");
    Ok(scores)
}

/// Encode one minibatch with every observed slot of both tracks visible.
///
/// Rate 0 hides nothing, so this shares [`draw_masks`] rather than re-deriving
/// observedness. `train = false`: BatchNorm uses running statistics and no
/// latent noise is drawn, so inference is deterministic.
fn infer_latent(
    encoder: &GemIndexedEncoder,
    mb: &GemMinibatchData,
    nascent_nonzero_only: bool,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    let masks = draw_masks(mb, 0.0, nascent_nonzero_only, dev)?;
    let input = GemEncoderInput {
        gene_indices: &mb.gene_indices,
        nascent_observed: &mb.nascent_observed,
        mature_observed: &mb.mature_observed,
        nascent_residual: mb.nascent_residual.as_ref(),
        mature_residual: mb.mature_residual.as_ref(),
        nascent_mean: mb.nascent_mean.as_ref(),
        mature_mean: mb.mature_mean.as_ref(),
        nascent_visible: &masks.nascent_visible,
        mature_visible: &masks.mature_visible,
    };
    encoder.forward_latent(&input, false)
}

/// Encode one minibatch to its latent — ONE pass, everything observed.
///
/// There is no `z_u`/`z_s` pair and no `Δz`. Those came from cross-modal
/// passes, which were the latent delta this model deliberately does not have
/// (see the module header). The cell-level delta is recovered instead by
/// [`fit_theta_to_track`], from the frozen dictionaries.
pub fn infer_minibatch(
    encoder: &GemIndexedEncoder,
    mb: &GemMinibatchData,
    nascent_nonzero_only: bool,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    infer_latent(encoder, mb, nascent_nonzero_only, dev)
}

/// Which center of the retained `θ` draws [`fit_theta_posterior`] reports.
///
/// The two are Bayes estimators under different losses, and the right one is
/// decided by the CONTRAST the caller forms downstream — not by taste.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaMean {
    /// `E[θ]`: average the probabilities. Coherent with a LINEAR contrast
    /// (`θ^u − θ^s`), which is what `faba`'s velocity forms today, and what
    /// makes `θ^u·α − θ^s·α = (θ^u − θ^s)·α` exact — the identity the H-space
    /// velocity field relies on.
    ///
    /// Lands on the simplex for free: a convex combination of simplex points is
    /// a simplex point.
    Arithmetic,
    /// The compositional center: average `log θ`, then close. This is the
    /// closed geometric mean, the center under Aitchison geometry, and it is
    /// the coherent choice for a LOG-RATIO contrast (`log(θ^u/θ^s)`) — which
    /// compositional data analysis argues is the meaningful difference between
    /// two compositions, since a linear difference of compositions is not
    /// itself compositional.
    ///
    /// Not the default, and the reason is downstream: pairing this center with
    /// `faba`'s current linear difference is incoherent in a way that BITES.
    /// The geometric mean sits below the arithmetic one, by a margin that grows
    /// with the spread of the draws — and the nascent track is the sparse, more
    /// diffuse one, so the shrinkage would land unevenly on the two halves of a
    /// quantity computed by differencing them. Switch this on in the same
    /// change that moves the contrast to a log ratio, not before.
    Geometric,
}

/// How hard to work at closing the amortization gap in [`fit_theta_to_track`].
pub struct ThetaFitConfig {
    /// Elliptical-slice transitions per cell.
    pub n_steps: usize,
    /// Safety cap on slice shrinkages within one transition.
    pub max_shrink: usize,
    /// Trailing states averaged into the returned `θ` — the rest are burn-in.
    ///
    /// Costs nothing: the transitions run either way, and this only decides how
    /// many of the resulting states are kept instead of discarded. `1`
    /// reproduces the single-final-draw behaviour this fit used to have.
    pub n_keep: usize,
    /// Which center of the retained draws to report — see [`ThetaMean`].
    pub mean: ThetaMean,
}

impl Default for ThetaFitConfig {
    fn default() -> Self {
        Self {
            n_steps: 16,
            max_shrink: 50,
            n_keep: 8,
            mean: ThetaMean::Arithmetic,
        }
    }
}

/// Fit `log θ` to ONE track's counts against the FROZEN decoder — the post-hoc
/// cell-level delta.
///
/// # What this is for
///
/// The model has one latent, so it cannot express "this cell's nascent state
/// differs from its mature state" during training — by design, since letting it
/// do so lets the encoder absorb the contrast that `δ` exists to carry (module
/// header). The difference is recovered here instead: fit θ separately to each
/// track under that track's dictionary, and take `θ^u − θ^s`.
///
/// Ordering matters and is the whole point. `δ` is estimated first, from the
/// joint-masked likelihood; the latent delta is then READ OUT of the frozen
/// dictionaries. The two are no longer simultaneously free, so the
/// identifiability problem is dissolved by construction rather than penalized.
///
/// # Why ESS, warm-started
///
/// [`crate::mcmc::batched_ess_steps`] transitions AWAY from the initial state
/// toward the posterior, so starting from the encoder's own `z` is what makes
/// this a fix for the **amortization gap** rather than an unrelated
/// optimization. Starting from a uniform θ would discard the encoder entirely
/// and measure something else.
///
/// # Why the posterior MEAN, not the final draw
///
/// ESS returns samples, so keeping only the last one wrote a single draw into
/// the latent as if it were a point estimate — and `Δθ = θ^u − θ^s` differences
/// two independent chains, so it carried two independent sampling errors with
/// no way to tell them from biology. Averaging the trailing `cfg.n_keep` states
/// costs nothing (the transitions run regardless; they were simply discarded)
/// and cuts that noise by the number of INDEPENDENT draws they are worth —
/// fewer than `cfg.n_keep`, since the states are autocorrelated.
///
/// ("ESS" throughout this file is elliptical slice sampling, never effective
/// sample size; the MCMC literature uses the acronym for the latter, so the
/// count of independent draws is spelled out rather than abbreviated here.)
///
/// The average is taken over `θ`, NOT over `z`. `softmax(mean z) ≠ mean
/// softmax(z)`, and `Δθ` is a difference of COMPOSITIONS, so `E[θ]` is the
/// estimand; averaging in `z` would return a sharper-than-posterior point,
/// biased toward the leading topic and worst on the thin nascent track. See
/// [`crate::mcmc::batched_ess_posterior`].
///
/// [`fit_theta_posterior`] returns the per-cell spread alongside the mean —
/// which is what makes "is this cell's `Δθ` real, or sampling noise?"
/// answerable at all.
///
/// # The prior, stated
///
/// Elliptical slice sampling supplies an `N(0, I)` prior on `z` itself. The
/// model TRAINS with no prior — a Gaussian/KL head was removed after collapsing
/// the latent to effective rank 1.03. That is not an argument against this: a
/// prior regularizing a per-cell fit at inference is a different operation from
/// a KL pulling the whole latent during training.
pub fn fit_theta_to_track(
    decoder: &GemEtmDecoder,
    mb: &GemMinibatchData,
    init_raw_z: &Tensor,
    track: Track,
    cfg: &ThetaFitConfig,
) -> candle_core::Result<Tensor> {
    Ok(fit_theta_posterior(decoder, mb, init_raw_z, track, cfg)?.0)
}

/// [`fit_theta_to_track`] with the per-cell spread kept: returns
/// `(log θ posterior mean [N, T], θ posterior SD [N, T])`.
///
/// The SD is on whatever scale the center was taken — `θ` for
/// [`ThetaMean::Arithmetic`], `log θ` for [`ThetaMean::Geometric`] — so that the
/// spread always matches the geometry of the point it describes. Do not compare
/// the two across a change of `cfg.mean`.
///
/// Read it as a LOWER BOUND on the posterior spread: consecutive ESS states are
/// autocorrelated, so `cfg.n_keep` retained states are worth fewer than
/// `cfg.n_keep` independent draws. It ranks cells honestly without being a
/// calibrated standard deviation.
pub fn fit_theta_posterior(
    decoder: &GemEtmDecoder,
    mb: &GemMinibatchData,
    init_raw_z: &Tensor,
    track: Track,
    cfg: &ThetaFitConfig,
) -> candle_core::Result<(Tensor, Tensor)> {
    // Score every OBSERVED position of this track — not a masking draw. The fit
    // asks "what θ best explains the counts this cell actually has", so nothing
    // is held out.
    let valid = &mb.slot_valid;
    let observed = match track {
        Track::Mature => mb
            .mature_observed
            .gt(0.0)?
            .to_dtype(DType::F32)?
            .mul(valid)?,
        Track::Nascent => mb
            .nascent_observed
            .gt(0.0)?
            .to_dtype(DType::F32)?
            .mul(valid)?,
    };

    // Fit against the DECODER TARGET, the same selection `score_track` makes:
    // `μ_adjusted` when batch adjustment produced one, else the encoder's own
    // values.
    let values = match track {
        Track::Nascent => mb.nascent_adjusted.as_ref().unwrap_or(&mb.nascent_observed),
        Track::Mature => mb.mature_adjusted.as_ref().unwrap_or(&mb.mature_observed),
    };
    let counts = values.mul(&observed)?;

    // EVERYTHING that does not depend on `z` is hoisted and DETACHED.
    //
    // This is not only speed. Calling into the decoder inside the closure
    // touches its `Var`s, so each of the `n_steps` transitions built a fresh
    // autograd graph that lived until the fit returned — ×2 fits per block,
    // which exhausted GPU memory on a real run. Elliptical slice sampling is
    // gradient-free and never needs a graph; detaching here is what makes that
    // true in practice rather than only in principle.
    //
    // `get_dictionary` is `log_softmax_g`, so this is the multinomial
    // composition — which is the right target regardless of the training
    // likelihood, because θ IS a composition. It also makes the fit
    // depth-invariant, so the sparser nascent track is not penalized for its
    // depth.
    let log_beta_gt = decoder.get_dictionary(track)?.detach(); // [G, T]
    let (n, kc) = mb.gene_indices.dims2()?;
    let t = log_beta_gt.dim(1)?;
    let lb_nkt = log_beta_gt
        .index_select(&mb.gene_indices.flatten_all()?, 0)?
        .reshape((n, kc, t))?; // [N, K_ctx, T]

    let lnpdf = |z: &Tensor| -> candle_core::Result<Tensor> {
        let log_theta = theta_log_simplex(z)?; // [N, T]
                                               // log p_nk = logsumexp_t ( log θ_nt + log β_nkt ), stabilized.
        let sum_nkt = lb_nkt.broadcast_add(&log_theta.unsqueeze(1)?)?;
        let m = sum_nkt.max_keepdim(2)?;
        let log_p = (m.squeeze(2)? + sum_nkt.broadcast_sub(&m)?.exp()?.sum(2)?.log()?)?;
        counts.mul(&log_p)?.sum(1) // [N]
    };

    // Accumulate a MAP of z, never z itself — see the "posterior MEAN" section
    // above. Which map depends on the center being reported.
    let init = init_raw_z.detach();
    match cfg.mean {
        ThetaMean::Arithmetic => {
            let (theta_mean, theta_sd) = crate::mcmc::batched_ess_posterior(
                &init,
                &lnpdf,
                cfg.n_steps,
                cfg.n_keep,
                cfg.max_shrink,
                |z| theta_log_simplex(z)?.exp(),
            )?;
            // A mean of simplex points is itself on the simplex, so this stays a
            // valid log θ. The clamp is only against an f32 underflow to exactly
            // 0 in a topic no retained state gave mass — `ln(0)` would put a
            // −inf in the latent.
            let log_theta = theta_mean.clamp(1e-20f64, 1.0f64)?.log()?;
            Ok((log_theta, theta_sd))
        }
        ThetaMean::Geometric => {
            let (log_mean, log_sd) = crate::mcmc::batched_ess_posterior(
                &init,
                &lnpdf,
                cfg.n_steps,
                cfg.n_keep,
                cfg.max_shrink,
                theta_log_simplex,
            )?;
            // `log_softmax` IS the closure operator in log space: it renormalizes
            // the averaged log θ back onto the simplex, which the geometric mean
            // does NOT reach on its own (unlike the arithmetic one). No clamp is
            // needed on this branch — `log_softmax` is `z − logsumexp(z)`, finite
            // for any finite input, so nothing can underflow to −inf on the way.
            let log_theta = candle_nn::ops::log_softmax(&log_mean, 1)?;
            Ok((log_theta, log_sd))
        }
    }
}

#[cfg(test)]
#[path = "masked_gem_tests.rs"]
mod tests;
