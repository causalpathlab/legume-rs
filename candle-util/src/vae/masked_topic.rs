//! Masked-imputation topic VAE trainer.
//!
//! Hosts [`train_masked`] (no-ELBO masked-gene NB imputation; simplex-θ or
//! Gaussian-z latent), sharing the indexed top-K data loader across levels.
//! Drives the shared [`IndexedEmbeddingEncoder`] + per-level
//! [`EmbeddedNbTopicDecoder`] stack against [`IndexedInMemoryData`]
//! minibatches. The hot loop never materialises `[N, S]` or `[K, D]`;
//! all gather/scatter happens at the per-batch gene union.

use super::{clip_and_step_dense, smooth_topics, TrainScores};
use crate::data::indexed::{labeled_bar, GraphCsr, IndexedInMemoryArgs, IndexedInMemoryData};
use crate::data::IndexedMinibatchData;
use crate::decoder::masked_etm::{EmbeddedNbTopicDecoder, MaskedDenseTarget};
use crate::decoder::query_decoder::{QueryDecoder, QueryInput};
use crate::encoder::indexed::IndexedEmbeddingEncoder;
use crate::value_transform::anscombe_lite;
use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use log::{info, warn};
use matrix_util::rand_util::mix_seed;
use nalgebra::DMatrix;
use rand::RngExt;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    pub enc_context_size: usize,
    pub stop: &'a AtomicBool,
    /// Per-gene weights used to *score* candidates during the encoder's
    /// top-K shortlist selection. Stored values remain raw counts.
    pub shortlist_weights: &'a [f32],
    /// Per-gene Anscombe baseline (length = D_full). When supplied, the
    /// loader gathers it at each sample's encoder top-K positions; the
    /// encoder subtracts it from Anscombe-stabilized values before pooling.
    pub feature_mean: &'a [f32],
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
}

/// Per-minibatch mask-rate schedule (any-order / absorbing-diffusion style).
#[derive(Clone, Copy, Debug)]
pub enum MaskSchedule {
    /// Constant mask fraction (the `mask_fraction` arg).
    Fixed,
    /// Sample the mask rate uniformly in `[lo, hi]` each minibatch.
    Uniform { lo: f64, hi: f64 },
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
/// KL (the masked objective alone prevents collapse); they differ only in the
/// simplex parameterization. `Gaussian` is the true variational bottleneck.
///
/// This is a pure identity tag — a `Copy`, round-trippable value used by the
/// train dispatch, the inference dispatch, and model persistence alike. The
/// Gaussian KL weight is a train-only hyperparameter and lives on
/// [`MaskedTrainOpts::kl_weight`], not in the tag, so the inference/persistence
/// sites never fabricate a placeholder weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentHead {
    /// Deterministic simplex `log_softmax(z)` — exchangeable topics. The legacy
    /// masked-topic default.
    Softmax,
    /// Deterministic **stick-breaking** simplex — ordered, exchangeability-
    /// broken topics with a self-pruning tail. Same no-KL objective as
    /// `Softmax`, only the final simplex map differs.
    StickBreaking,
    /// Reparameterized **Gaussian** latent `z` (no simplex projection) plus a
    /// `kl_weight · KL(z ‖ N(0, I))` term. `exp(z)` drives the NB head's
    /// per-topic intensities, so the decoder is reused unchanged.
    Gaussian,
}

pub struct MaskedTrainOpts {
    pub mask_schedule: MaskSchedule,
    /// Per-gene likelihood for the masked imputation loss.
    pub likelihood: MaskedLikelihood,
    /// Latent head: simplex (softmax / stick-breaking, deterministic no-KL) or
    /// Gaussian (reparameterized + KL). See [`LatentHead`].
    pub latent: LatentHead,
    /// KL weight `β` for the Gaussian latent. Ignored unless `latent == Gaussian`.
    pub kl_weight: f64,
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
    /// Seed for the trainer's own stochastic draws: the per-step context mask
    /// (and its rate under [`MaskSchedule::Uniform`]), and
    /// [`MaskedTrainOpts::poisson_thin`]'s per-epoch draw, and the query set.
    /// Each is keyed on a disjoint sub-stream of this seed, so all are
    /// reproducible independently of the thread count.
    pub seed: u64,
    /// `Some`: the query decoder is on — masked and out-of-context genes read
    /// the visible slots and add a per-gene log-residual to the mixture rate
    /// (see [`crate::decoder::query_decoder`]). Requires a [`QueryDecoder`]
    /// handed to [`train_masked`]. `None`: today's heads, byte for byte.
    pub query: Option<QueryOpts>,
}

/// Options of the query-decoder term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryOpts {
    /// Out-of-context genes drawn uniformly per row as extra queries, zeros
    /// included, on top of the masked context genes.
    pub extra: usize,
    /// Weight of `mean r²` in the loss: the mixture explains first, the
    /// residual takes the remainder.
    pub penalty: f64,
}

impl Default for MaskedTrainOpts {
    fn default() -> Self {
        Self {
            mask_schedule: MaskSchedule::Fixed,
            likelihood: MaskedLikelihood::Nb,
            latent: LatentHead::Softmax,
            kl_weight: 1.0,
            poisson_thin: false,
            seed: 42,
            query: None,
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
////////////////////////////////////////////////////////

/// Seed for one training step's stochastic choices, keyed on
/// `(seed, epoch, level, minibatch)`.
///
/// Drawn host-side because there is no seedable device RNG: `Device::set_seed`
/// errors on the CPU backend, so `Tensor::rand` is OS-seeded and a run cannot
/// be replayed or bisected. Keying on the step rather than a running counter
/// also makes the draw independent of how many steps preceded it.
///
/// The `"mask"` name puts this in a different sub-stream from the Poisson
/// thinning draw, which keys `(epoch, level)` off the bare seed.
#[must_use]
pub fn step_seed(seed: u64, epoch: usize, level: usize, minibatch: usize) -> u64 {
    let salt = ((epoch as u64) << 40) | ((level as u64) << 32) | (minibatch as u64 & 0xFFFF_FFFF);
    mix_seed(matrix_util::rand_util::name_seed(seed, "mask"), salt)
}

/// The mask rate for one step: constant, or a seeded draw from `[lo, hi]`.
#[must_use]
pub fn mask_rate(schedule: MaskSchedule, mask_fraction: f64, step_seed: u64) -> f64 {
    match schedule {
        MaskSchedule::Fixed => mask_fraction,
        MaskSchedule::Uniform { lo, hi } => {
            use rand::{rngs::SmallRng, SeedableRng};
            let mut rng = SmallRng::seed_from_u64(step_seed);
            lo + rng.random::<f64>() * (hi - lo)
        }
    }
}

/// Split a packed `[N, K]` context into `(visible, masked)` indicator tensors.
///
/// Only real slots (`value > 0`) are split; a pad is in neither, so it neither
/// contaminates the pool nor gets scored. Drawn on the host from `step_seed`,
/// then handed to the caller for upload with the rest of the minibatch.
pub fn draw_context_mask(
    values: &Tensor,
    rate: f64,
    step_seed: u64,
) -> candle_core::Result<(Tensor, Tensor)> {
    use rand::{rngs::SmallRng, SeedableRng};
    let (n, k) = values.dims2()?;
    let host: Vec<f32> = values
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut vis = vec![0f32; n * k];
    let mut msk = vec![0f32; n * k];
    // Seeded per row, not per worker: the row index pins each row's draw
    // whatever the batch layout, and rows keep independent streams.
    for r in 0..n {
        let mut rng = SmallRng::seed_from_u64(mix_seed(step_seed, r as u64));
        for c in 0..k {
            let slot = r * k + c;
            if host[slot] > 0.0 {
                if rng.random::<f64>() < rate {
                    msk[slot] = 1.0;
                } else {
                    vis[slot] = 1.0;
                }
            }
        }
    }
    let dev = values.device();
    Ok((
        Tensor::from_vec(vis, (n, k), dev)?,
        Tensor::from_vec(msk, (n, k), dev)?,
    ))
}

/// `[N, D]` scored-position mask: 1 everywhere the encoder could not see.
///
/// The canonical masked-training prediction space. The encoder's budget is the
/// `[N, K]` context; what the decoder answers for is every *other* gene,
/// zero-count genes included, so the scored set does not inherit the context's
/// selection bias toward abundant genes.
///
/// Pads carry index 0 with a zero visible flag, so they scatter nothing and
/// gene 0 stays scored for a row that did not really see it.
pub fn target_mask_nd(
    indices: &Tensor,
    visible: &Tensor,
    n_features: usize,
) -> candle_core::Result<Tensor> {
    let n = indices.dim(0)?;
    let zeros = Tensor::zeros((n, n_features), visible.dtype(), visible.device())?;
    zeros.scatter_add(indices, visible, 1)?.affine(-1.0, 1.0)
}

/// One step's query genes per row: `ids [N, Q]` u32 and `weight [N, Q]`, 1 on a
/// real query and 0 on a pad (which carries gene 0 and must not score).
pub struct QuerySet {
    pub ids: Tensor,
    pub weight: Tensor,
}

/// Draw the step's query set: every masked real context gene, plus `extra`
/// genes per row drawn uniformly from the axis outside the row's whole context
/// (so a query is never a visible gene and never drawn twice). Rows are padded
/// to the widest row. Seeded per row on a sub-stream of `step_seed` disjoint
/// from the context mask's.
pub fn draw_query_set(
    indices: &Tensor,
    values: &Tensor,
    visible: &Tensor,
    n_features: usize,
    extra: usize,
    step_seed: u64,
) -> candle_core::Result<QuerySet> {
    use rand::{rngs::SmallRng, SeedableRng};
    let dev = indices.device();
    let cpu = Device::Cpu;
    let idx = indices.to_device(&cpu)?.to_vec2::<u32>()?;
    let val = values.to_device(&cpu)?.to_vec2::<f32>()?;
    let vis = visible.to_device(&cpu)?.to_vec2::<f32>()?;
    let n = idx.len();
    let salt = mix_seed(step_seed, u64::MAX);
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(n);
    for r in 0..n {
        let mut rng = SmallRng::seed_from_u64(mix_seed(salt, r as u64));
        let context: HashSet<u32> = idx[r]
            .iter()
            .zip(&val[r])
            .filter(|(_, &v)| v > 0.0)
            .map(|(&g, _)| g)
            .collect();
        let mut q: Vec<u32> = idx[r]
            .iter()
            .zip(&val[r])
            .zip(&vis[r])
            .filter(|((_, &v), &s)| v > 0.0 && s == 0.0)
            .map(|((&g, _), _)| g)
            .collect();
        let mut drawn: HashSet<u32> = HashSet::with_capacity(extra);
        let room = n_features.saturating_sub(context.len());
        let want = extra.min(room);
        let mut tries = 0usize;
        while drawn.len() < want && tries < 64 * want.max(1) {
            tries += 1;
            let g = rng.random_range(0..n_features) as u32;
            if !context.contains(&g) {
                drawn.insert(g);
            }
        }
        let mut extras: Vec<u32> = drawn.into_iter().collect();
        extras.sort_unstable();
        q.extend(extras);
        rows.push(q);
    }
    let width = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut ids = vec![0u32; n * width];
    let mut weight = vec![0f32; n * width];
    for (r, q) in rows.iter().enumerate() {
        for (j, &g) in q.iter().enumerate() {
            ids[r * width + j] = g;
            weight[r * width + j] = 1.0;
        }
    }
    Ok(QuerySet {
        ids: Tensor::from_vec(ids, (n, width), dev)?,
        weight: Tensor::from_vec(weight, (n, width), dev)?,
    })
}

/// Scatter the weighted per-query residual onto the gene axis: `[N, D]` with
/// `r` at each real query's gene and 0 everywhere else, pads included.
///
/// Done as a one-dimensional `index_add` over the flattened `[N·D]` table with
/// `row·D + gene` offsets rather than a row-wise `scatter_add`: candle's
/// `scatter_add` backward assumes the index tensor has as many columns as the
/// target, which a `[N, Q]` query set never does, whereas `index_add` gathers
/// the gradient back to each query exactly.
pub fn residual_nd(
    ids: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    n_features: usize,
) -> candle_core::Result<Tensor> {
    let (n, q) = ids.dims2()?;
    let dev = residual.device();
    let offsets = Tensor::arange(0u32, n as u32, dev)?
        .affine(n_features as f64, 0.0)?
        .unsqueeze(1)?; // [N, 1]
    let flat_ids = ids.broadcast_add(&offsets)?.reshape(n * q)?; // [N·Q]
    let src = (residual * weight)?.reshape(n * q)?;
    Tensor::zeros(n * n_features, residual.dtype(), dev)?
        .index_add(&flat_ids, &src, 0)?
        .reshape((n, n_features))
}

/// One level's decoder targets, resident on device as `[P, D]`.
///
/// These are the **batch-free** rows (`mu_adjusted` where a batch-aware
/// collapse ran), which the loader used to select a top-K from and then
/// discard. The library is the whole row's total, so the NB mean is on the
/// scale of the counts being scored rather than the encoder context's share
/// of them.
pub struct LevelTarget {
    values_pd: Tensor,
    row_lib_p1: Tensor,
}

impl LevelTarget {
    /// Upload a level's `[P, D]` target rows and precompute `Σ_g y_pg + 1`.
    pub fn from_mat(rows: &Mat, dev: &Device) -> anyhow::Result<Self> {
        let values_pd = crate::data::loader_util::upload_to_device(rows, dev)?;
        let row_lib_p1 = (values_pd.sum_keepdim(1)? + 1.0)?;
        Ok(Self {
            values_pd,
            row_lib_p1,
        })
    }

    /// `(values [N, D], lib [N, 1])` for the minibatch's source rows.
    pub fn rows(&self, row_ids: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        Ok((
            self.values_pd.index_select(row_ids, 0)?,
            self.row_lib_p1.index_select(row_ids, 0)?,
        ))
    }

    /// Per-row library `[P, 1]`.
    pub fn row_lib(&self) -> &Tensor {
        &self.row_lib_p1
    }
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
/// `[N, K]` (`log θ` for the simplex heads, `z` for Gaussian) and — only for
/// `Gaussian` — the per-cell KL `[N]`.
///
/// Single source of truth for the head → encoder-forward dispatch shared by
/// [`train_masked`] and senna's encoder-only inference. The encoder itself
/// stays head-agnostic (three plain forwards, no `LatentHead` dependency).
/// Callers own their post-processing: the trainer smooths the simplex heads
/// (`kl.is_none()`) and adds the KL term; inference discards the KL.
pub fn masked_encode(
    encoder: &IndexedEmbeddingEncoder,
    head: LatentHead,
    input: &MaskedEncoderInput,
    train: bool,
) -> candle_core::Result<(Tensor, Option<Tensor>)> {
    match head {
        LatentHead::Gaussian => {
            let (z, kl) = encoder.forward_indexed_masked_gaussian(
                input.indices,
                input.values,
                input.values_null,
                input.values_mean,
                input.visible_mask,
                train,
            )?;
            Ok((z, Some(kl)))
        }
        LatentHead::StickBreaking => Ok((
            encoder.forward_indexed_masked_stick(
                input.indices,
                input.values,
                input.values_null,
                input.values_mean,
                input.visible_mask,
                train,
            )?,
            None,
        )),
        LatentHead::Softmax => Ok((
            encoder.forward_indexed_masked(
                input.indices,
                input.values,
                input.values_null,
                input.values_mean,
                input.visible_mask,
                train,
            )?,
            None,
        )),
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
/// differ from masked-topic in exactly one respect — the reparameterized sample
/// and its KL.
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
        .map(|&(mixed, batch, _target)| {
            let mut loader = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
                input: mixed,
                input_null: batch,
                input_context_size: config.enc_context_size,
                input_shortlist_weights: config.shortlist_weights,
                input_mean: Some(config.feature_mean),
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

/// What one minibatch's forward reports back to the epoch loop.
struct MaskedMinibatchLoss {
    loss: Tensor,
    /// Masked log-likelihood sum and its normalizer.
    metric_sum: f32,
    metric_count: f32,
    /// `Some((sum, count))` when the Gaussian head's KL is active.
    kl_sums: Option<(f32, f32)>,
    /// `Some((Σ r², #queries))` when the query decoder is active: how much the
    /// residual is carrying, per query, this step.
    query_r2: Option<(f32, f32)>,
}

/// One minibatch's full forward loss, shared by the epoch loop and the
/// GPU memory probe so the probe measures exactly the forward a real
/// step retains. Returns the loss plus the report scalars the loop
/// accumulates (masked llik sum/count, and the KL sum/count when the
/// Gaussian head is active).
#[allow(clippy::too_many_arguments)]
fn masked_minibatch_loss(
    encoder: &IndexedEmbeddingEncoder,
    decoder: &EmbeddedNbTopicDecoder,
    query_decoder: Option<&QueryDecoder>,
    config: &IndexedTrainConfig,
    opts: &MaskedTrainOpts,
    mask_fraction: f64,
    mb: &IndexedMinibatchData,
    target: &LevelTarget,
    step_seed: u64,
) -> anyhow::Result<MaskedMinibatchLoss> {
    // Visible/masked split over the row's real (value>0) top-K. Pads
    // (value==0) are neither visible (no ρ₀ contamination) nor scored.
    // `masked` is only a bookkeeping complement here: what the decoder
    // answers for is every gene OUTSIDE the context (see `target_mask_nd`).
    let rate = mask_rate(opts.mask_schedule, mask_fraction, step_seed);
    let (visible, _masked) = draw_context_mask(&mb.input_values, rate, step_seed)?;

    // Masked-VAE: reparameterized Gaussian `z` (no softmax) + KL.
    // Masked-topic: deterministic simplex `log θ` (softmax or
    // stick-breaking), no KL. In all cases the NB head reads this as
    // its per-topic intensity log — `exp(z)` for the VAE, `θ` for
    // the topic models.
    let (raw_z, kl_opt) = masked_encode(
        encoder,
        opts.latent,
        &MaskedEncoderInput {
            indices: &mb.input_indices,
            values: &mb.input_values,
            values_null: mb.input_values_null.as_ref(),
            values_mean: mb.input_values_mean.as_ref(),
            visible_mask: &visible,
        },
        true,
    )?;
    // Every head reaches the decoder as a smoothed log-simplex; the
    // Gaussian `z` is projected with `log_softmax` first. See
    // [`decoder_log_theta`].
    let log_z = decoder_log_theta(raw_z, opts.latent, config.topic_smoothing)?;

    // full_kd (α·ρᵀ [K,D]): the per-topic log-partition of the dense heads,
    // the one [K, D] product per step.
    let full_kd = decoder.full_logits_kd()?;

    // Query decoder: the masked context genes and a few genes outside the
    // context each read the visible slots and add a log-residual to their own
    // rate. The pool above and this read see the same slots through the same
    // gate; only the question differs.
    let query = match (query_decoder, opts.query) {
        (Some(qd), Some(qo)) => {
            let gate = anscombe_lite(
                &mb.input_values,
                mb.input_values_null.as_ref(),
                mb.input_values_mean.as_ref(),
            )?;
            let qs = draw_query_set(
                &mb.input_indices,
                &mb.input_values,
                &visible,
                decoder.dim_obs(),
                qo.extra,
                step_seed,
            )?;
            let read = qd.forward(
                encoder.feature_embeddings(),
                &QueryInput {
                    indices: &mb.input_indices,
                    gate: &gate,
                    visible: &visible,
                    query_ids: &qs.ids,
                },
            )?;
            let r_nd = residual_nd(&qs.ids, &read.residual, &qs.weight, decoder.dim_obs())?;
            let n_q = qs.weight.sum_all()?.to_scalar::<f32>()?.max(1.0);
            let r2_sum = (read.residual.sqr()? * &qs.weight)?.sum_all()?;
            let penalty = r2_sum.affine(qo.penalty / f64::from(n_q), 0.0)?;
            let r2 = (r2_sum.to_scalar::<f32>()?, n_q);
            Some((r_nd, penalty, r2))
        }
        _ => None,
    };

    let (mut loss, batch_metric, batch_count) = {
        let (values_nd, lib_n1) = target.rows(&mb.row_ids)?;
        let mask_nd = target_mask_nd(&mb.input_indices, &visible, decoder.dim_obs())?;
        // `residual: None` — these rows are the batch-FREE targets, so β is
        // fit to composition the collapse already corrected. The per-row
        // offset belongs to cell-level scoring, where counts are mixed.
        let dense = MaskedDenseTarget {
            values: &values_nd,
            residual: None,
            lib: &lib_n1,
            mask: &mask_nd,
            log_residual: query.as_ref().map(|(r, _, _)| r),
        };
        let llik = match opts.likelihood {
            MaskedLikelihood::Nb => decoder.impute_dense_nb(&log_z, &dense, &full_kd)?,
            MaskedLikelihood::Multinomial => {
                decoder.impute_dense_multinomial(&log_z, &dense, &full_kd)?
            }
        };
        let m = llik.sum_all()?.to_scalar::<f32>()?;
        let c = mask_nd.sum_all()?.to_scalar::<f32>()?;
        (llik.mean_all()?.neg()?, m, c)
    };
    // Masked-VAE KL bottleneck: β · mean_N KL(z ‖ N(0, I)). `kl_opt`
    // is `Some` only on the Gaussian head, so no head re-check needed.
    let mut kl_sums: Option<(f32, f32)> = None;
    if let Some(kl) = kl_opt {
        kl_sums = Some((kl.sum_all()?.to_scalar::<f32>()?, kl.dim(0)? as f32));
        loss = (loss + kl.mean_all()?.affine(opts.kl_weight, 0.0)?)?;
    }
    let mut query_r2 = None;
    if let Some((_, penalty, r2)) = query {
        loss = (loss + penalty)?;
        query_r2 = Some(r2);
    }
    if config.feature_embedding_l2 > 0.0 && config.frozen_feature_var.is_none() {
        let rho_l2 = encoder
            .feature_embeddings()
            .sqr()?
            .mean_all()?
            .affine(f64::from(config.feature_embedding_l2), 0.0)?;
        loss = (loss + rho_l2)?;
    }
    Ok(MaskedMinibatchLoss {
        loss,
        metric_sum: batch_metric,
        metric_count: batch_count,
        kl_sums,
        query_r2,
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
pub fn train_masked(
    level_data: &[LevelData],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[EmbeddedNbTopicDecoder],
    query_decoder: Option<&QueryDecoder>,
    config: &IndexedTrainConfig,
    mask_fraction: f64,
    opts: &MaskedTrainOpts,
) -> anyhow::Result<TrainScores> {
    anyhow::ensure!(
        opts.query.is_none() || query_decoder.is_some(),
        "query options set but no query decoder supplied"
    );
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
        .chain(config.frozen_feature_var)
        .collect();
    let adam_vars: Vec<Var> = crate::frozen_features::trainable_vars(config.parameters, &frozen);
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
    // The decoder's targets: the batch-free rows, whole, on device. The loader
    // holds only the encoder's context; the scored set is this trainer's own.
    let mut level_targets = level_data
        .iter()
        .map(|&(_, _, target)| LevelTarget::from_mat(target, config.dev))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // On CUDA, optionally shrink the minibatch size to fit free device
    // memory. The probe runs the exact forward a training step retains
    // (loss held un-backwarded); `auto_chunk_size` reserves half the
    // measured budget for backward's gradient copies.
    let minibatch_size = match (config.gpu_mem_fraction, data_loaders.first()) {
        (Some(frac), Some(loader)) => {
            let cap = config.minibatch_size;
            crate::device::auto_chunk_size(config.dev, cap, 16.min(cap), frac, |n| {
                // Cycled, not truncated: training bootstrap-pads every
                // batch to the full minibatch size, so the probe must
                // measure `n` real rows even when the level holds fewer.
                let mb = loader
                    .minibatch_cycled(n, config.dev)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let fwd = masked_minibatch_loss(
                    encoder,
                    &decoders[0],
                    query_decoder,
                    config,
                    opts,
                    mask_fraction,
                    &mb,
                    &level_targets[0],
                    step_seed(opts.seed, 0, 0, 0),
                )
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                Ok(fwd.loss)
            })
            .unwrap_or(cap)
        }
        _ => config.minibatch_size,
    };

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
            data_loaders = build_indexed_loaders(&refs, config)?;
            level_targets = refs
                .iter()
                .map(|&(_, _, target)| LevelTarget::from_mat(target, config.dev))
                .collect::<anyhow::Result<Vec<_>>>()?;
        }
        for loader in data_loaders.iter_mut() {
            shuffle_and_precompute(loader, minibatch_size)?;
        }

        let mut metric_tot = 0f32;
        let mut metric_cnt = 0f32;
        let mut kl_tot = 0f32;
        let mut kl_cnt = 0f32;
        let mut r2_tot = 0f32;
        let mut r2_cnt = 0f32;
        let mut skipped_steps = 0usize;

        for (level, loader) in data_loaders.iter().enumerate() {
            let decoder = &decoders[level];

            for b in 0..loader.num_minibatch() {
                let mb = loader.minibatch_cached(b).to_device(config.dev)?;

                let fwd = masked_minibatch_loss(
                    encoder,
                    decoder,
                    query_decoder,
                    config,
                    opts,
                    mask_fraction,
                    &mb,
                    &level_targets[level],
                    step_seed(opts.seed, epoch, level, b),
                )?;
                if let Some((ks, kc)) = fwd.kl_sums {
                    kl_tot += ks;
                    kl_cnt += kc;
                }
                if let Some((rs, rc)) = fwd.query_r2 {
                    r2_tot += rs;
                    r2_cnt += rc;
                }
                let loss = fwd.loss;
                let grads = loss.backward()?;
                if !clip_and_step_dense(&mut adam, grads, f64::from(config.grad_clip))? {
                    skipped_steps += 1;
                }

                metric_tot += fwd.metric_sum;
                metric_cnt += fwd.metric_count;

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
        if log::log_enabled!(log::Level::Info) {
            let kl_msg = if matches!(opts.latent, LatentHead::Gaussian) {
                format!(" kl/cell={per_kl:.4}")
            } else {
                String::new()
            };
            // Root mean square of the residual per query: how much the query
            // decoder is carrying beyond the mixture. Near zero on data without
            // co-expression beyond the topics; the number to watch on real data.
            let r_msg = if r2_cnt > 0.0 {
                format!(" query rms(r)={:.4}", (r2_tot / r2_cnt).sqrt())
            } else {
                String::new()
            };
            // Per scored gene, and the scored set is now every gene the encoder
            // did not see — so this is not comparable to a run that scored only
            // the context's masked share.
            info!("[epoch {epoch}] masked llik/gene={per_metric:.4}{kl_msg}{r_msg}");
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

#[cfg(test)]
#[path = "masked_topic_tests.rs"]
mod masked_topic_tests;
