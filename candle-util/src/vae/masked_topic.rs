//! Masked-imputation topic VAE trainer.
//!
//! Hosts [`train_masked`] (no-ELBO masked-gene NB imputation; simplex-θ or
//! Gaussian-z latent) over the **window-free** dense loader
//! ([`crate::data::masked_dense`]): the encoder reads every gene of a
//! minibatch row, zeros included, and the mask is drawn over the whole gene
//! axis. Drives the shared [`IndexedEmbeddingEncoder`] + per-level
//! [`EmbeddedNbTopicDecoder`] stack.
//!
//! One draw, two consumers: `visible_nd` is what the encoder may look at and
//! `1 − visible_nd` is what the decoder is scored on, so the hidden set has a
//! single definition rather than two that have to be kept in step.

use super::{clip_and_step_dense_all, smooth_topics, TrainScores};
use crate::data::indexed::labeled_bar;
use crate::data::masked_dense::{DenseMaskedLevel, DenseMaskedMinibatch, MaskedDraw};
use crate::decoder::coarsening_map::CoarseningMap;
use crate::decoder::masked_etm::{EmbeddedNbTopicDecoder, MaskedDenseTarget, ModuleTarget};
use crate::encoder::indexed::IndexedEmbeddingEncoder;
pub use crate::lora::LoraPlus;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use log::{info, warn};
use matrix_util::rand_util::mix_seed;
use nalgebra::DMatrix;
use std::sync::atomic::{AtomicBool, Ordering};

pub use crate::data::masked_dense::MaskSchedule;

type Mat = DMatrix<f32>;

/// Config bundle passed by reference to [`train_masked`].
pub struct IndexedTrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a Device,
    pub epochs: usize,
    /// `Some(frac)`: on CUDA, probe one forward per candidate size and
    /// SHRINK `minibatch_size` when free device memory says so (never
    /// grow past it). `None`, CPU, or an unavailable query keep the
    /// configured size exactly.
    pub gpu_mem_fraction: Option<f32>,
    pub minibatch_size: usize,
    pub learning_rate: f32,
    pub topic_smoothing: f64,
    pub stop: &'a AtomicBool,
    /// Per-gene mean rate `μ_d` (length = D). The encoder divides by it — with
    /// the per-row batch null — before the Anscombe residual, so what it reads
    /// is the cell's deviation from the gene's typical rate.
    pub feature_mean: &'a [f32],
    /// Global L2 gradient norm clip per minibatch (0 = off).
    pub grad_clip: f32,
    /// Explicit L2 penalty `λ_ρ · ‖ρ‖_F²` on the feature embedding
    /// matrix ρ ∈ ℝ^{D × H}. Added to the per-minibatch loss before
    /// backward. `0.0` disables.
    pub feature_embedding_l2: f32,
    /// AdamW decoupled weight decay applied to every parameter per-step
    /// (not just ρ). Post-step parameter shrinkage that doesn't enter the
    /// loss/backward graph. `0.0` disables.
    pub weight_decay: f32,
    /// The feature table's anchor, when a prior run supplied it: the named
    /// `Var` (the encoder's `feature.embeddings`) stays out of AdamW and out of
    /// the ridge, and under LoRA its shared factor takes its own group at the
    /// LoRA+ rate. The encoder/decoder still reference the table through the
    /// same `Var`; anchoring keeps the optimizer's hands off.
    pub feature_anchor: Option<FeatureAnchor<'a>>,
}

/// See [`IndexedTrainConfig::feature_anchor`].
#[derive(Clone, Copy, Debug)]
pub struct FeatureAnchor<'a> {
    /// The pinned table's `Var`.
    pub base_var: &'a str,
    /// The LoRA+ split for the residual's shared factor, when there is one.
    pub lora: Option<LoraPlus<'a>>,
}

/// Options specific to [`train_masked`], kept off the shared
/// [`IndexedTrainConfig`] so callers that only need the config are unaffected.
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
    /// The likelihood a generative ELBO path would also use, so a comparison
    /// under this option isolates the objective rather than the likelihood.
    Multinomial,
}

/// Which latent head the masked encoder uses to turn pooled visible genes into
/// the per-topic log-intensity `log θ` the NB/multinomial imputation head reads.
///
/// `Softmax` and `StickBreaking` are both deterministic point estimates with no
/// KL (the masked objective alone prevents collapse); they differ only in how
/// the latent reaches the decoder. `Gaussian` is deterministic too: it used to
/// reparameterize and add a KL toward `N(0, I)`, and at the default weight
/// that pulled `z` to zero and every row's θ to uniform.
///
/// This is a pure identity tag — a `Copy`, round-trippable value used by the
/// train dispatch, the inference dispatch, and model persistence alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentHead {
    /// Deterministic simplex `log_softmax(z)` — exchangeable topics. The legacy
    /// masked-topic default.
    Softmax,
    /// Deterministic **stick-breaking** simplex — ordered, exchangeability-
    /// broken topics with a self-pruning tail. Same no-KL objective as
    /// `Softmax`, only the final simplex map differs.
    StickBreaking,
    /// Unconstrained **Gaussian-style** latent `z` (no simplex projection in
    /// the encoder). The decoder reads it through `log_softmax` (see
    /// [`decoder_log_theta`]); the latent written out is the raw `z`.
    Gaussian,
}

pub struct MaskedTrainOpts {
    pub mask_schedule: MaskSchedule,
    /// Per-gene likelihood for the masked imputation loss.
    pub likelihood: MaskedLikelihood,
    /// Latent head: simplex (softmax / stick-breaking) or unconstrained
    /// Gaussian-style `z`; all deterministic, none with a KL. See [`LatentHead`].
    pub latent: LatentHead,
    /// Train on **Poisson draws** from the pseudobulk rate rows, redrawn every
    /// epoch, instead of the rates themselves.
    ///
    /// The rows this trainer is handed are per-pseudobulk mean rates: dense and
    /// smooth. What the encoder is fed at inference is a single cell's raw
    /// counts: sparse integers. Drawing `x_pg ~ Poisson(μ_pg)` — a synthetic
    /// cell at the pseudobulk's own average depth — makes the training rows
    /// look like the inference rows, at the cost of one top-K repack per epoch.
    ///
    /// In an A/B on a targeted panel the cell-level held-out imputation
    /// likelihood improved, while the cell latent became *sharper*. The encoder
    /// is near one-hot on whatever distribution it trained on and softer off it,
    /// so this is a likelihood lever, not a remedy for a one-hot latent.
    pub poisson_thin: bool,
    /// Seed for the trainer's own stochastic draws: the hidden set and its
    /// rate under [`MaskSchedule::Uniform`] — drawn per epoch and keyed on
    /// `(epoch, level, row)` — and
    /// [`MaskedTrainOpts::poisson_thin`]'s per-epoch draw. Each is keyed on a
    /// disjoint sub-stream of this seed, so all are reproducible independently
    /// of the thread count, the batch size and the shuffle.
    pub seed: u64,
}

impl Default for MaskedTrainOpts {
    fn default() -> Self {
        Self {
            mask_schedule: MaskSchedule::Fixed,
            likelihood: MaskedLikelihood::Nb,
            latent: LatentHead::Softmax,
            poisson_thin: false,
            seed: 42,
        }
    }
}

/// One Poisson draw per entry of a rate matrix. Zero (or non-finite) rates draw 0.
///
/// Column-parallel with a thread-local RNG: this runs once per epoch over the
/// whole `[P × D]` pseudobulk table, so it has to be a few milliseconds, not a
/// second.
fn poisson_draw(rates: &Mat, seed: u64) -> Mat {
    use rand::{rngs::SmallRng, SeedableRng};
    use rand_distr::{Distribution, Poisson};
    use rayon::prelude::*;
    let nrows = rates.nrows();
    let mut out = rates.clone();
    out.as_mut_slice()
        .par_chunks_mut(nrows.max(1))
        .enumerate()
        .for_each(|(col_idx, col): (usize, &mut [f32])| {
            // Seeded PER COLUMN, not per worker: rayon assigns chunks to threads in
            // a scheduling-dependent order, so a thread-local `rand::rng()` would
            // make the draw depend on the thread count and on nothing the user sets.
            // Keying on the column index pins the output to `seed` alone.
            let mut rng =
                SmallRng::seed_from_u64(matrix_util::rand_util::mix_seed(seed, col_idx as u64));
            for v in col.iter_mut() {
                *v = if *v > 0.0 && v.is_finite() {
                    Poisson::new(f64::from(*v)).map_or(0.0, |p| p.sample(&mut rng) as f32)
                } else {
                    0.0
                };
            }
        });
    out
}

////////////////////////////////////////////////////////
// Seeded per-step draws: the mask and its rate       //
/// Seed for one epoch's draws on one level: the hidden set and the per-row
/// mask rate under [`MaskSchedule::Uniform`]. Keyed on
/// `(epoch, level)` so a draw depends only on the run seed and where in the
/// schedule it happens; the row is keyed inside the loader. The `"mask"` name
/// keeps this in a different sub-stream from the Poisson thinning draw.
#[must_use]
pub fn epoch_seed(seed: u64, epoch: usize, level: usize) -> u64 {
    let salt = ((epoch as u64) << 32) | (level as u64);
    mix_seed(matrix_util::rand_util::name_seed(seed, "mask"), salt)
}

/// The decoder's module-level view of ONE dense minibatch.
///
/// Built from the same `visible_nd` the encoder pooled under, which is what
/// makes "the genes the encoder could not see" and "the genes the decoder is
/// scored on" the same set by construction rather than by two call sites
/// agreeing: the decoder's scored indicator is `1 − visible_share`, and under
/// the identity map that is `1 − visible_nd` exactly.
pub struct DenseModuleTargets {
    /// `[N, M]` the whole row's counts, summed into modules.
    pub values_nm: Tensor,
    /// `[N, M]` the counts at the genes the encoder DID see.
    pub visible_counts_nm: Tensor,
    /// `[N, M]` the pinned within-module share the encoder saw.
    pub visible_share_nm: Tensor,
    /// `[N, 1]` the row's library over every gene, `+1`.
    pub lib_n1: Tensor,
}

/// Aggregate one dense minibatch into the decoder's modules.
///
/// Under the identity map every aggregation is a no-op clone, so a full-
/// resolution run pays nothing for the generality.
pub fn dense_module_targets(
    modules: &CoarseningMap,
    target_nd: &Tensor,
    visible_nd: &Tensor,
) -> candle_core::Result<DenseModuleTargets> {
    let values_nm = modules.aggregate_columns(target_nd)?;
    let visible_counts_nm = modules.aggregate_columns(&(target_nd * visible_nd)?)?;
    let share_1d = modules.log_share_1d().exp()?; // 1 under the identity map
    let visible_share_nm = modules.aggregate_columns(&visible_nd.broadcast_mul(&share_1d)?)?;
    // Modules partition the genes, so the module totals sum to the row total.
    let lib_n1 = (values_nm.sum_keepdim(1)? + 1.0)?;
    Ok(DenseModuleTargets {
        values_nm,
        visible_counts_nm,
        visible_share_nm,
        lib_n1,
    })
}

/// Borrowed masked-encoder inputs: the per-cell packed top-K plus the
/// visible-slot mask. Grouping them lets the trainer and encoder-only inference
/// share one dispatch entry point ([`masked_encode`]) instead of spelling the
/// six-argument encoder call out per head at each site.
pub struct MaskedEncoderInput<'a> {
    pub indices: &'a Tensor,
    pub values: &'a Tensor,
    pub values_null: Option<&'a Tensor>,
    pub values_mean: Option<&'a Tensor>,
    pub visible_mask: &'a Tensor,
}

/// Run the masked encoder under `head`, returning the raw per-topic latent
/// `[N, K]` (`log θ` for the simplex heads, `z` for Gaussian).
///
/// Single source of truth for the head → encoder-forward dispatch shared by
/// [`train_masked`] and senna's encoder-only inference. The encoder itself
/// stays head-agnostic (three plain forwards, no `LatentHead` dependency).
/// Every head is deterministic and none carries a KL; the trainer's only
/// post-processing is the decoder coupling in [`decoder_log_theta`].
pub fn masked_encode(
    encoder: &IndexedEmbeddingEncoder,
    head: LatentHead,
    input: &MaskedEncoderInput,
    train: bool,
) -> candle_core::Result<Tensor> {
    match head {
        LatentHead::Gaussian => encoder.forward_indexed_masked_gaussian(
            input.indices,
            input.values,
            input.values_null,
            input.values_mean,
            input.visible_mask,
            train,
        ),
        LatentHead::StickBreaking => encoder.forward_indexed_masked_stick(
            input.indices,
            input.values,
            input.values_null,
            input.values_mean,
            input.visible_mask,
            train,
        ),
        LatentHead::Softmax => encoder.forward_indexed_masked(
            input.indices,
            input.values,
            input.values_null,
            input.values_mean,
            input.visible_mask,
            train,
        ),
    }
}

/// Borrowed **window-free** masked-encoder inputs: the dense row, its per-row
/// batch null, the per-gene mean, and the visible mask over every gene.
pub struct MaskedDenseInput<'a> {
    pub x_nd: &'a Tensor,
    pub x0_nd: Option<&'a Tensor>,
    pub mean_1d: Option<&'a Tensor>,
    pub visible_nd: &'a Tensor,
}

/// Run the window-free masked encoder under `head`, returning the raw per-topic
/// latent `[N, K]` (`log θ` for the simplex heads, `z` for Gaussian).
///
/// The dense sibling of [`masked_encode`]; the single source of truth for the
/// head → encoder-forward dispatch on the dense path, shared by
/// [`train_masked`] and senna's encoder-only inference.
pub fn masked_encode_dense(
    encoder: &IndexedEmbeddingEncoder,
    head: LatentHead,
    input: &MaskedDenseInput,
    train: bool,
) -> candle_core::Result<Tensor> {
    match head {
        LatentHead::Gaussian => encoder.forward_dense_masked_gaussian(
            input.x_nd,
            input.x0_nd,
            input.mean_1d,
            input.visible_nd,
            train,
        ),
        LatentHead::StickBreaking => encoder.forward_dense_masked_stick(
            input.x_nd,
            input.x0_nd,
            input.mean_1d,
            input.visible_nd,
            train,
        ),
        LatentHead::Softmax => encoder.forward_dense_masked(
            input.x_nd,
            input.x0_nd,
            input.mean_1d,
            input.visible_nd,
            train,
        ),
    }
}

/// Project a masked-encoder latent onto the **log-simplex** the NB /
/// multinomial decoder heads consume, then apply `topic_smoothing`.
///
/// The simplex heads already emit `log θ`. The Gaussian head emits a raw
/// unconstrained `z`, which is *not* a log-simplex — and feeding that straight
/// to the decoder left the per-topic intensity `exp(z)` unbounded (`exp(8) ≈
/// 2981` at the encoder clamp, against `≤ 1` for the simplex heads). The head
/// then drove itself into the ±8 clamp, where the gradient is exactly zero, and
/// the encoder stopped learning for the rest of the run while the likelihood
/// trace still looked alive. Projecting with `log_softmax` makes masked-VAE
/// differ from masked-topic in exactly one respect — the latent it writes out
/// is the raw `z`, not `log θ`.
///
/// This is the *decoder coupling only*. The latent written to
/// `{out}.latent.parquet` stays the raw Gaussian `z`; see [`LatentHead`].
pub fn decoder_log_theta(
    raw_z: Tensor,
    head: LatentHead,
    topic_smoothing: f64,
) -> candle_core::Result<Tensor> {
    let log_theta = match head {
        LatentHead::Gaussian => candle_nn::ops::log_softmax(&raw_z, 1)?,
        LatentHead::Softmax | LatentHead::StickBreaking => raw_z,
    };
    smooth_topics(log_theta, topic_smoothing)
}

/// Per-level training triple: `(encoder input, optional batch null, decoder
/// target)`, each **`[D, P]`** — genes down, pseudobulk samples across, the
/// layout the collapsed posterior is sampled in. [`DenseMaskedLevel::from_mats`]
/// makes the `[P, D]` resident rows out of it without transposing anything.
///
/// All three are borrowed so callers can reuse the same `Mat` as both input
/// and target without cloning a multi-GB matrix.
pub type LevelData<'a> = (&'a Mat, Option<&'a Mat>, &'a Mat);

/// Upload every level's dense rows once (see [`DenseMaskedLevel`]).
///
/// A level whose target IS its input — the same `Mat` behind both borrows,
/// which is what "no batch-adjusted target" looks like here — uploads one
/// device buffer, not two; [`DenseMaskedLevel::from_mats`] owns that test.
fn resident_dense_levels(
    level_data: &[LevelData],
    config: &IndexedTrainConfig,
    dev: &Device,
) -> anyhow::Result<Vec<DenseMaskedLevel>> {
    level_data
        .iter()
        .map(|&(mixed, batch, target)| {
            DenseMaskedLevel::from_mats(mixed, batch, target, config.feature_mean, dev)
        })
        .collect()
}

/// What one minibatch's forward reports back to the epoch loop: the loss, and
/// device-side sums the loop accumulates without a host round trip.
struct StepLoss {
    loss: Tensor,
    /// Σ log-likelihood over the scored units (modules, or genes under the
    /// identity map).
    llik_sum: Tensor,
    /// Number of scored units.
    units_sum: Tensor,
}

/// Per-epoch sums, kept on the device and read back once.
pub(crate) struct EpochAccum {
    llik: Tensor,
    scored: Tensor,
}

impl EpochAccum {
    pub(crate) fn new(dev: &Device) -> candle_core::Result<Self> {
        let zero = || Tensor::zeros((), DType::F32, dev);
        Ok(Self {
            llik: zero()?,
            scored: zero()?,
        })
    }

    /// Fold one step in.
    pub(crate) fn add(&mut self, llik_sum: &Tensor, units_sum: &Tensor) -> candle_core::Result<()> {
        self.llik = (&self.llik + llik_sum.detach())?;
        self.scored = (&self.scored + units_sum.detach())?;
        Ok(())
    }

    /// llik per scored unit — the one host read of the epoch.
    pub(crate) fn read(&self) -> candle_core::Result<f32> {
        let scored = self.scored.to_scalar::<f32>()?;
        Ok(if scored > 0.0 {
            self.llik.to_scalar::<f32>()? / scored
        } else {
            0.0
        })
    }
}

/// `(llik [N], scored units [N])` at each row's hidden genes — the
/// full-resolution scorer, where a "module" is a gene and the row's hidden ids
/// are exactly what the decoder answers for.
///
/// The library is the whole row's total (`+1`), as the module form's is: what
/// changed is where the elementwise likelihood is evaluated, not the mean it is
/// evaluated at.
fn score_hidden_genes(
    decoder: &EmbeddedNbTopicDecoder,
    log_z: &Tensor,
    likelihood: MaskedLikelihood,
    mb: &DenseMaskedMinibatch,
    full_kd: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    let lib_n1 = (mb.target_nd.sum_keepdim(1)? + 1.0)?;
    let target = MaskedDenseTarget {
        values: &mb.target_nd,
        residual: None,
        lib: &lib_n1,
        hidden_ids: &mb.hidden_ids,
        hidden_weight: mb.hidden_weight.as_ref(),
    };
    let llik = match likelihood {
        MaskedLikelihood::Nb => decoder.impute_dense_nb(log_z, &target, full_kd)?,
        MaskedLikelihood::Multinomial => {
            decoder.impute_dense_multinomial(log_z, &target, full_kd)?
        }
    };
    let (n, dh) = mb.hidden_ids.dims2()?;
    let units = match mb.hidden_weight.as_ref() {
        Some(w) => w.sum(1)?,
        None => Tensor::full(dh as f32, n, llik.device())?,
    };
    Ok((llik, units))
}

/// One minibatch's full forward loss, shared by the epoch loop and the GPU
/// memory probe so the probe measures exactly the forward a real step retains.
/// Everything the step needs — the dense row, its null, and the mask — arrives
/// in `mb`, drawn by the loader for the epoch.
fn masked_minibatch_loss(
    encoder: &IndexedEmbeddingEncoder,
    decoder: &EmbeddedNbTopicDecoder,
    config: &IndexedTrainConfig,
    opts: &MaskedTrainOpts,
    mb: &DenseMaskedMinibatch,
    mean_1d: &Tensor,
    ridge_step: f64,
) -> anyhow::Result<StepLoss> {
    // Masked-VAE: unconstrained `z` (no softmax in the encoder).
    // Masked-topic: simplex `log θ` (softmax or stick-breaking). All
    // deterministic, none with a KL; `decoder_log_theta` below is the only
    // head-specific step before the NB head.
    let raw_z = masked_encode_dense(
        encoder,
        opts.latent,
        &MaskedDenseInput {
            x_nd: &mb.x_nd,
            x0_nd: mb.x0_nd.as_ref(),
            mean_1d: Some(mean_1d),
            visible_nd: &mb.visible_nd,
        },
        true,
    )?;
    // Every head reaches the decoder as a smoothed log-simplex; the
    // Gaussian `z` is projected with `log_softmax` first. See
    // [`decoder_log_theta`].
    let log_z = decoder_log_theta(raw_z, opts.latent, config.topic_smoothing)?;

    // The module logits `(α − ᾱ)·ρ̄ᵀ + log π` — `[K, M]`, or `[K, D]` under the
    // identity map — and the module-level view of what the encoder saw. The
    // pool and this view read the SAME `visible_nd`, so the scored set is the
    // hidden set, not a second approximation of it.
    let full_kd = decoder.full_logits_kd()?;
    let (llik, units) = if decoder.coarsening().is_identity() {
        // Full gene resolution: the scored set IS the row's hidden ids, so the
        // likelihood is evaluated at `[N, d_h]` and nowhere else. The module
        // form below is the same number term for term (pinned by
        // `module_scorer_matches_the_dense_gene_scorer_under_the_identity_map`)
        // but reaches it by computing every gene and multiplying the visible
        // majority away.
        score_hidden_genes(decoder, &log_z, opts.likelihood, mb, &full_kd)?
    } else {
        let t = dense_module_targets(decoder.coarsening(), &mb.target_nd, &mb.visible_nd)?;
        // These rows are the batch-FREE targets, so β is fit to composition the
        // collapse already corrected; the per-row offset belongs to cell-level
        // scoring, where counts are mixed.
        let module_target = ModuleTarget {
            values: &t.values_nm,
            visible_counts: &t.visible_counts_nm,
            visible_share: &t.visible_share_nm,
            residual: None,
            lib: &t.lib_n1,
        };
        match opts.likelihood {
            MaskedLikelihood::Nb => {
                decoder.score_unseen_modules_nb(&log_z, &module_target, &full_kd)?
            }
            MaskedLikelihood::Multinomial => {
                decoder.score_unseen_modules_multinomial(&log_z, &module_target, &full_kd)?
            }
        }
    };
    let llik_sum = llik.sum_all()?;
    let units_sum = units.sum_all()?;
    // Per scored unit, so every penalty below is on the same scale as the
    // number the epoch log reports.
    let mut loss = llik_sum.neg()?.div(&units_sum.clamp(1.0, f64::INFINITY)?)?;
    // An anchored table's residual is shrunk instead: its ridge at this
    // step's weight (the caller spreads the per-epoch weight over the steps).
    if ridge_step > 0.0 {
        if let Some(r) = encoder.features().lora_ridge()? {
            loss = (loss + r.affine(ridge_step, 0.0)?)?;
        }
    }
    if config.feature_embedding_l2 > 0.0 && config.feature_anchor.is_none() {
        // Shrink what is actually free: the dictionary under modules, the table
        // otherwise. Penalizing composed rows would charge every member of a
        // module for the same shared vector.
        let rho_l2 = encoder
            .features()
            .ridge_table()
            .sqr()?
            .mean_all()?
            .affine(f64::from(config.feature_embedding_l2), 0.0)?;
        loss = (loss + rho_l2)?;
    }
    Ok(StepLoss {
        loss,
        llik_sum,
        units_sum,
    })
}

/// Masked-imputation training (no ELBO / no KL) for the embedded topic model.
///
/// Per epoch and row, the loader splits the gene axis into **visible**
/// (encoder input) and **hidden** at a fixed count per row. The
/// encoder pools the visible genes into a deterministic `log θ`; the
/// embedded-topic decoder imputes every gene the encoder did not see
/// (`μ = ℓ·θβ`) and the loss is the log-likelihood on those positions. No posterior, no KL →
/// no posterior collapse. Pseudobulk masking also simulates the PB→single-cell
/// sparsity the amortized encoder must handle at inference.
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
            "Level {}/{}: {} samples, decoder dim {} over {} genes (masked-imputation ETM)",
            level + 1,
            num_levels,
            mixed.ncols(),
            decoder.dim_obs(),
            decoder.n_features(),
        );
    }
    info!(
        "Masked-imputation training: {num_levels} levels, {total_epochs} epochs, mask={mask_fraction}"
    );

    // Every decoder's pinned background stays out of the optimizer, plus the
    // frozen ρ when a prior run supplied it.
    let pinned: Vec<String> = {
        let suffix = format!(".{}", crate::decoder::masked_etm::BACKGROUND_VAR);
        let tbl = config.parameters.data().lock().unwrap();
        tbl.keys()
            .filter(|name| name.ends_with(&suffix))
            .cloned()
            .collect()
    };
    let frozen: Vec<&str> = pinned
        .iter()
        .map(String::as_str)
        .chain(config.feature_anchor.map(|a| a.base_var))
        .chain(config.feature_anchor.and_then(|a| a.lora).map(|l| l.v_var))
        .collect();
    let adam_vars: Vec<Var> = crate::frozen_features::trainable_vars(config.parameters, &frozen);
    let mut adams = vec![AdamW::new(
        adam_vars,
        candle_nn::ParamsAdamW {
            lr: f64::from(config.learning_rate),
            weight_decay: f64::from(config.weight_decay),
            ..Default::default()
        },
    )?];
    if let Some(l) = config.feature_anchor.and_then(|a| a.lora) {
        adams.push(l.optimizer(config.parameters, config.learning_rate)?);
    }
    let prog_bar = labeled_bar("Epochs", total_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_epochs);
    // No KL in the masked objective; keep a zero column the same length as
    // `llik` so `TrainScores::to_parquet` sees equal-length columns.
    let draw = MaskedDraw {
        schedule: opts.mask_schedule,
        mask_fraction,
    };
    // Each level's dense rows go to the device once; every epoch is one shuffle
    // of the row order plus the epoch's mask draw. The decoder's targets are
    // the batch-free rows, whole; the scored set is what the mask hid.
    let mut levels = resident_dense_levels(level_data, config, config.dev)?;

    // On CUDA, optionally shrink the minibatch size to fit free device
    // memory. The probe runs the exact forward a training step retains
    // (loss held un-backwarded); `auto_chunk_size` reserves half the
    // measured budget for backward's gradient copies.
    let minibatch_size = match (config.gpu_mem_fraction, levels.first()) {
        (Some(frac), Some(level0)) => {
            let cap = config.minibatch_size;
            crate::device::auto_chunk_size(config.dev, cap, 16.min(cap), frac, |n| {
                // Cycled, not truncated: training pads the last batch by
                // resampling, so the probe must measure `n` real rows even
                // when the level holds fewer.
                let mb = level0
                    .probe_minibatch(n, epoch_seed(opts.seed, 0, 0), &draw)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let fwd = masked_minibatch_loss(
                    encoder,
                    &decoders[0],
                    config,
                    opts,
                    &mb,
                    level0.feature_mean_1d(),
                    0.0,
                )
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                Ok(fwd.loss)
            })
            .unwrap_or(cap)
        }
        _ => config.minibatch_size,
    };

    // Under Poisson thinning the rows are redrawn per epoch and the level is
    // rebuilt from the draw.
    for epoch in 0..total_epochs {
        // A fresh synthetic-cell draw per epoch, so no row is ever the same
        // twice — the same role the Gamma jitter plays for the dense trainers,
        // one level down (counts around the rate, not rates around the
        // posterior). The null is the level's own and is not redrawn.
        if opts.poisson_thin {
            // Draw the INPUT and the TARGET separately, each from its own rates.
            // They are not the same matrix: `sample_collapsed_data` sets the target
            // to `mu_adjusted` — the batch-FREE rates — whenever a batch-aware
            // collapse ran, and reusing the input draw for both would train the
            // decoder to reproduce the batch effect the collapse just removed.
            let thinned: Vec<(Mat, Option<&Mat>, Mat)> = level_data
                .iter()
                .enumerate()
                .map(|(level, &(mixed, batch, target))| {
                    let epoch_salt = (epoch as u64) << 32 | level as u64;
                    let x = poisson_draw(mixed, mix_seed(opts.seed, epoch_salt));
                    // A distinct salt, or input and target would be the same draw
                    // wherever `mu_adjusted` is absent and both point at `mixed`.
                    let y = if std::ptr::eq(mixed, target) {
                        x.clone()
                    } else {
                        poisson_draw(target, mix_seed(opts.seed, !epoch_salt))
                    };
                    (x, batch, y)
                })
                .collect();
            let refs: Vec<LevelData> = thinned.iter().map(|(x, b, y)| (x, *b, y)).collect();
            levels = resident_dense_levels(&refs, config, config.dev)?;
        }

        let mut acc = EpochAccum::new(config.dev)?;
        let mut skipped_steps = 0usize;

        for (level, lv) in levels.iter().enumerate() {
            let decoder = &decoders[level];
            let ep = lv.begin_epoch(epoch_seed(opts.seed, epoch, level), &draw, minibatch_size)?;
            // The per-epoch LoRA ridge, spread over every level's batches.
            let ridge_step = config.feature_anchor.and_then(|a| a.lora).map_or(0.0, |l| {
                f64::from(l.ridge) / (levels.len() * ep.n_batches().max(1)) as f64
            });
            for b in 0..ep.n_batches() {
                let mb = ep.batch(b)?;
                let fwd = masked_minibatch_loss(
                    encoder,
                    decoder,
                    config,
                    opts,
                    &mb,
                    lv.feature_mean_1d(),
                    ridge_step,
                )?;
                acc.add(&fwd.llik_sum, &fwd.units_sum)?;
                let grads = fwd.loss.backward()?;
                if !clip_and_step_dense_all(&mut adams, grads, f64::from(config.grad_clip))? {
                    skipped_steps += 1;
                }
                if config.stop.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        let per_metric = acc.read()?;
        llik_trace.push(per_metric);
        prog_bar.set_message(format!("llik={per_metric:.3}"));
        prog_bar.inc(1);
        // A skipped step means the gradient overflowed. Parameters are intact
        // (the step was dropped, not applied), but a run that keeps skipping is
        // diverging and its latent will be junk — say so rather than let the
        // llik trace look healthy while nothing is learning.
        if skipped_steps > 0 {
            warn!(
                "[epoch {epoch}] skipped {skipped_steps} optimizer step(s): \
                 non-finite gradient norm. Lower --learning-rate or --grad-clip \
                 if this persists."
            );
        }
        // Per scored unit: every gene the encoder did not see, or every
        // module it did not see completely — so this is not comparable
        // across module maps, nor to a windowed run, which hid only the
        // masked share of a top-K context.
        info!("[epoch {epoch}] masked llik/unit={per_metric:.4}");
        if config.stop.load(Ordering::SeqCst) {
            prog_bar.finish_and_clear();
            info!("Stopping early at epoch {epoch}");
            return Ok(TrainScores {
                kl: vec![0.0; llik_trace.len()],
                llik: llik_trace,
            });
        }
    }

    prog_bar.finish_and_clear();
    info!("done masked-imputation training");
    Ok(TrainScores {
        kl: vec![0.0; llik_trace.len()],
        llik: llik_trace,
    })
}

#[cfg(test)]
#[path = "masked_topic_tests.rs"]
mod masked_topic_tests;
