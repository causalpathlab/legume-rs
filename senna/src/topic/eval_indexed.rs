use super::common::{expand_delta_for_block, process_blocks};
use crate::embed_common::*;

use candle_core::{Device, Tensor};
use candle_util::data::csc_columns_to_indexed_samples;
use candle_util::decoder::masked_etm::ModuleTarget;
use candle_util::decoder::EmbeddedNbTopicDecoder;
use candle_util::fast_index::scatter_add_cols;
use candle_util::traits::*;
use candle_util::vae::masked_topic::{
    decoder_log_theta, dense_module_targets, masked_encode, masked_encode_dense,
    DenseModuleTargets, LatentHead, MaskedDenseInput, MaskedEncoderInput, MaskedLikelihood,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Packed top-K representation consumed by the masked encoder.
///
/// Holds the per-cell `(indices, values)` the encoder reads; a single
/// host pass over the sparse columns is enough. Optional `values_mean`
/// carries the per-feature count-rate divisor gathered at the same
/// per-cell positions. Clone is cheap — Tensor is Arc-buffered.
#[derive(Clone)]
pub(crate) struct IndexedPack {
    /// [N, K] u32 — per-cell top-K feature ids
    pub indices: Tensor,
    /// [N, K] f32 — per-cell values in `indices` order
    pub values: Tensor,
    /// [N, K] f32 — per-gene mean expression rate `μ_d` gathered at
    /// `indices` (encoder side; multiplicative count-rate divisor).
    pub values_mean: Option<Tensor>,
}

/// Optional per-gene context for the packers. Each slice is length `D`
/// (training-D order); the packer gathers it at the per-cell top-K
/// positions to produce the `[N, K]` tensor that flows into the encoder.
#[derive(Clone, Copy, Default)]
pub(crate) struct PerGeneContext<'a> {
    pub feature_mean: Option<&'a [f32]>,
}

/// Build an [`IndexedPack`] directly from a sparse `[D, N]` CSC matrix —
/// columns are cells. Skips the dense `[N, D]` materialization the
/// `dense_*` helpers need; only the stored nonzeros are visited.
/// `gene_remap` (`Some(new_to_train)`) maps held-out → training gene
/// indices for `predict` on a differing gene set; the top-K is built over
/// the remapped, training-space ids. `None` when the CSC is already on the
/// training axis (fit-time eval).
pub(crate) fn csc_to_indexed(
    x_dn: &nalgebra_sparse::CscMatrix<f32>,
    context_size: usize,
    shortlist_weights: &[f32],
    gene_remap: Option<&[Option<usize>]>,
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let k = context_size.min(x_dn.nrows());
    let samples = csc_columns_to_indexed_samples(x_dn, shortlist_weights, context_size, gene_remap);
    let all_top_k: Vec<(Vec<u32>, Vec<f32>)> =
        samples.into_iter().map(|s| (s.indices, s.values)).collect();
    pack_top_k_to_indexed(&all_top_k, k, ctx, dev)
}

/// Pack per-cell top-K `(indices, values)` into the `[N, K]` tensors of
/// an [`IndexedPack`].
fn pack_top_k_to_indexed(
    all_top_k: &[(Vec<u32>, Vec<f32>)],
    k: usize,
    ctx: PerGeneContext<'_>,
    dev: &Device,
) -> anyhow::Result<IndexedPack> {
    let n_batch = all_top_k.len();

    // Pack per-cell (indices, values) into [N, K]. Short rows (when D < K)
    // get padded with (idx=0, val=0.0); the matching zero value makes the
    // gather + weighted-sum a no-op.
    let mut idx_buf = vec![0u32; n_batch * k];
    let mut val_buf = vec![0.0f32; n_batch * k];
    let mut base_buf = ctx.feature_mean.map(|_| vec![0.0f32; n_batch * k]);
    for (row, (indices, values)) in all_top_k.iter().enumerate() {
        let off = row * k;
        let take = indices.len().min(k);
        idx_buf[off..off + take].copy_from_slice(&indices[..take]);
        val_buf[off..off + take].copy_from_slice(&values[..take]);
        if let (Some(buf), Some(b)) = (base_buf.as_mut(), ctx.feature_mean) {
            for (kk, &feat) in indices[..take].iter().enumerate() {
                buf[off + kk] = b[feat as usize];
            }
        }
    }

    let indices =
        Tensor::from_vec(idx_buf, (n_batch, k), dev)?.to_dtype(candle_core::DType::U32)?;
    let values = Tensor::from_vec(val_buf, (n_batch, k), dev)?;
    let values_mean = base_buf
        .map(|buf| Tensor::from_vec(buf, (n_batch, k), dev))
        .transpose()?;

    Ok(IndexedPack {
        indices,
        values,
        values_mean,
    })
}

/// Gather a per-cell null `[N, K] f32` from a dense `[N, D]` null tensor at
/// the encoder's `indices [N, K]`.
///
/// The earlier dense-scatter version walked every union slot for every
/// cell; the packed version pulls only the K positions that the encoder
/// will actually consume, so cost goes from O(N·S) to O(N·K).
pub(crate) fn gather_null_at_indices(
    x0_nd: &Tensor,
    indices: &Tensor,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let x0_rows: Vec<Vec<f32>> = x0_nd.to_vec2()?;
    let idx_vec: Vec<Vec<u32>> = indices.to_vec2()?;
    let n = idx_vec.len();
    let k = if idx_vec.is_empty() {
        0
    } else {
        idx_vec[0].len()
    };
    let mut buf = vec![0.0f32; n * k];
    for (i, idx_row) in idx_vec.iter().enumerate() {
        let null_row = &x0_rows[i];
        let off = i * k;
        for (kk, &feat) in idx_row.iter().enumerate() {
            buf[off + kk] = null_row[feat as usize];
        }
    }
    Ok(Tensor::from_vec(buf, (n, k), dev)?)
}

/// How a masked checkpoint's encoder reads a block of cells.
///
/// Decided by the model's own metadata, never by what happens to be on disk:
/// `enc_context_size` is `Some(k)` only for a model trained with a context
/// window, and such a model must keep scoring through the window it was fitted
/// under or it is a different model. Everything this build trains is `Dense`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MaskedRead<'a> {
    /// Window-free: the encoder reads every gene of the block, zeros included.
    Dense,
    /// An OLD model: each cell's top-K, ranked by the shortlist weights it was
    /// trained with.
    Windowed {
        context_size: usize,
        shortlist_weights: &'a [f32],
    },
}

impl<'a> MaskedRead<'a> {
    /// Resolve from a checkpoint's recorded window and its shortlist, if any.
    pub(crate) fn resolve(
        enc_context_size: Option<usize>,
        shortlist_weights: Option<&'a [f32]>,
    ) -> anyhow::Result<Self> {
        match enc_context_size {
            None => Ok(Self::Dense),
            Some(context_size) => {
                let shortlist_weights = shortlist_weights.ok_or_else(|| {
                    anyhow::anyhow!(
                        "this model records a context window of {context_size}, so it must be \
                         scored through the top-K it was trained under — but its \
                         shortlist_weights.parquet is missing. Reading it without them would \
                         rank a different context."
                    )
                })?;
                Ok(Self::Windowed {
                    context_size,
                    shortlist_weights,
                })
            }
        }
    }
}

/// A sparse `[D_query, n]` block as the dense `[n, D_train]` rows the
/// window-free encoder reads.
///
/// `gene_remap` (`Some(new_to_train)`) sends held-out gene ids to training ids;
/// several query genes can land on one training gene, so the fill ADDS.
pub(crate) fn csc_block_to_dense(
    x_dn: &nalgebra_sparse::CscMatrix<f32>,
    n_train_features: usize,
    gene_remap: Option<&[Option<usize>]>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let n = x_dn.ncols();
    let mut buf = vec![0f32; n * n_train_features];
    for j in 0..n {
        let col = x_dn.col(j);
        for (&r, &v) in col.row_indices().iter().zip(col.values().iter()) {
            let Some(g) = (match gene_remap {
                Some(map) => map.get(r).copied().flatten(),
                None => Some(r),
            }) else {
                continue;
            };
            buf[j * n_train_features + g] += v;
        }
    }
    Ok(Tensor::from_vec(buf, (n, n_train_features), dev)?)
}

/// Config for [`evaluate_latent_masked`] — encoder-only inference for the
/// masked-imputation embedded topic model (no decoder / no refinement).
pub(crate) struct EvaluateLatentMaskedConfig<'a> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    /// Window-free, or the window an OLD model recorded. See [`MaskedRead`].
    pub read: MaskedRead<'a>,
    pub feature_mean: &'a [f32],
    /// Latent head to run at inference — must match the head the model was
    /// trained with. The eval path (all genes visible, no decoder) is identical
    /// across heads; only the final simplex/latent map differs. The Gaussian
    /// arm returns the posterior mean `z` (train=false → reparam returns mean).
    pub head: LatentHead,
}

/// Encoder-only latent inference for the masked-imputation topic model.
///
/// Calls the deterministic masked-encoder forward with **all real genes
/// visible** (`visible = value>0`, no masking at inference) — matching the
/// training-time pooling (pads excluded) — and uses no decoder.
pub(crate) fn evaluate_latent_masked(
    data_vec: &SparseIoVec,
    encoder: &candle_util::encoder::IndexedEmbeddingEncoder,
    config: &EvaluateLatentMaskedConfig,
    delta: Option<&Tensor>,
    gene_remap: Option<&[Option<usize>]>,
) -> anyhow::Result<Mat> {
    evaluate_latent_masked_blocks(
        data_vec.num_columns(),
        encoder,
        config,
        gene_remap,
        |(lb, ub)| {
            let x_dn = data_vec.read_columns_csc(lb..ub)?;
            let x0_nd = delta
                .map(|delta_bm| {
                    expand_delta_for_block(
                        data_vec,
                        delta_bm,
                        config.adj_method,
                        lb,
                        ub,
                        config.dev,
                    )
                })
                .transpose()?;
            Ok((x_dn, x0_nd))
        },
    )
}

/// [`evaluate_latent_masked`] on **dense rate rows** — the pseudobulk matrix the
/// model was trained on, `[D × P]` with an optional per-row null `[P × D]`.
///
/// This is the diagnostic that separates a collapsed latent from a saturated
/// encoder. Training rows are pseudobulk rates; inference rows are single-cell
/// counts. If the same encoder is multi-topic on the rows it trained on and
/// one-hot on cells, the model is fine and the *input* is out of distribution
/// (see `--poisson-thin`); if it is one-hot on both, training collapsed and the
/// anchor penalty is the lever.
pub(crate) fn evaluate_latent_masked_rows(
    x_dp: &Mat,
    null_pd: Option<&Mat>,
    encoder: &candle_util::encoder::IndexedEmbeddingEncoder,
    config: &EvaluateLatentMaskedConfig,
) -> anyhow::Result<Mat> {
    if let Some(n) = null_pd {
        anyhow::ensure!(
            n.nrows() == x_dp.ncols() && n.ncols() == x_dp.nrows(),
            "row null is {}×{}, expected {}×{} (rows × genes)",
            n.nrows(),
            n.ncols(),
            x_dp.ncols(),
            x_dp.nrows()
        );
    }
    evaluate_latent_masked_blocks(x_dp.ncols(), encoder, config, None, |(lb, ub)| {
        let block = x_dp.columns(lb, ub - lb).into_owned();
        let x_dn = nalgebra_sparse::CscMatrix::from(&block);
        let x0_nd = null_pd
            .map(|n| n.rows(lb, ub - lb).into_owned().to_tensor(config.dev))
            .transpose()?;
        Ok((x_dn, x0_nd))
    })
}

/// The shared block loop: `read` yields each block's `[D, n]` counts and its
/// optional `[n, D]` null; everything after that is identical for every source.
fn evaluate_latent_masked_blocks<R>(
    ntot: usize,
    encoder: &candle_util::encoder::IndexedEmbeddingEncoder,
    config: &EvaluateLatentMaskedConfig,
    gene_remap: Option<&[Option<usize>]>,
    read: R,
) -> anyhow::Result<Mat>
where
    R: Fn((usize, usize)) -> anyhow::Result<(nalgebra_sparse::CscMatrix<f32>, Option<Tensor>)>
        + Send
        + Sync,
{
    let kk = IndexedEncoderT::dim_latent(encoder);
    let d_train = config.feature_mean.len();
    let mean_1d = Tensor::from_vec(config.feature_mean.to_vec(), (1, d_train), config.dev)?;

    process_blocks(ntot, kk, config.minibatch_size, config.dev, |block| {
        let (lb, _ub) = block;
        let (x_dn, x0_nd) = read(block)?;

        // Encoder-only inference: `train = false`; the latent alone is
        // written out. Nothing is masked at inference — window-free, that means
        // every gene is visible, zeros included, which is the distribution the
        // dense encoder trained on.
        let latent_nk = match config.read {
            MaskedRead::Dense => {
                let x_nd = csc_block_to_dense(&x_dn, d_train, gene_remap, config.dev)?;
                let visible = Tensor::ones(x_nd.shape(), candle_core::DType::F32, config.dev)?;
                masked_encode_dense(
                    encoder,
                    config.head,
                    &MaskedDenseInput {
                        x_nd: &x_nd,
                        x0_nd: x0_nd.as_ref(),
                        mean_1d: Some(&mean_1d),
                        visible_nd: &visible,
                    },
                    false,
                )?
            }
            MaskedRead::Windowed {
                context_size,
                shortlist_weights,
            } => {
                let ctx = PerGeneContext {
                    feature_mean: Some(config.feature_mean),
                };
                let enc_pack = csc_to_indexed(
                    &x_dn,
                    context_size,
                    shortlist_weights,
                    gene_remap,
                    ctx,
                    config.dev,
                )?;
                let enc_values_null = x0_nd
                    .as_ref()
                    .map(|x0| gather_null_at_indices(x0, &enc_pack.indices, config.dev))
                    .transpose()?;
                let visible = enc_pack.values.gt(0.0)?.to_dtype(candle_core::DType::F32)?;
                masked_encode(
                    encoder,
                    config.head,
                    &MaskedEncoderInput {
                        indices: &enc_pack.indices,
                        values: &enc_pack.values,
                        values_null: enc_values_null.as_ref(),
                        values_mean: enc_pack.values_mean.as_ref(),
                        visible_mask: &visible,
                    },
                    false,
                )?
            }
        };
        let z_nk = latent_nk.to_device(&candle_core::Device::Cpu)?;
        Ok((lb, Mat::from_tensor(&z_nk)?))
    })
}

/// Config for [`evaluate_holdout_imputation`].
pub(crate) struct HoldoutEvalConfig<'a> {
    pub dev: &'a Device,
    pub adj_method: &'a AdjMethod,
    pub minibatch_size: usize,
    /// Window-free, or the window an OLD model recorded. See [`MaskedRead`].
    pub read: MaskedRead<'a>,
    pub feature_mean: &'a [f32],
    /// Latent head the model was trained with (matches the persisted model).
    pub head: LatentHead,
    /// Per-gene likelihood — must match training for a comparable number.
    pub likelihood: MaskedLikelihood,
    /// Topic smoothing applied to the simplex heads, mirroring training.
    pub topic_smoothing: f64,
    /// Fraction of each cell's observed genes to hold out and score.
    pub mask_fraction: f64,
    /// Seed for the hold-out mask. A fixed seed masks the **same** per-cell
    /// positions across heads, so the comparison is apples-to-apples.
    pub seed: u64,
}

/// Held-out masked-imputation likelihood: for each cell, hide `mask_fraction`
/// of its observed genes, encode from the **visible** genes, impute the hidden
/// ones with the trained NB/multinomial ETM decoder, and return the mean
/// log-likelihood per held-out gene.
///
/// This is the honest generalization metric the per-epoch training likelihood
/// can't provide: the scored positions are never encoder inputs on their own
/// forward pass, and individual cells enter training only via their pseudobulk
/// aggregate — so a model that merely memorizes the training pseudobulks cannot
/// score well here. Runs one no-gradient pass; the decoder's `β = α·ρᵀ` (and
/// its log-partition) are fixed, so they are computed once.
pub(crate) fn evaluate_holdout_imputation(
    data_vec: &SparseIoVec,
    encoder: &candle_util::encoder::IndexedEmbeddingEncoder,
    decoder: &EmbeddedNbTopicDecoder,
    config: &HoldoutEvalConfig,
    delta: Option<&Tensor>,
) -> anyhow::Result<f32> {
    let ntot = data_vec.num_columns();
    let full_kd = decoder.full_logits_kd()?;
    let d_train = config.feature_mean.len();
    let mean_1d = Tensor::from_vec(config.feature_mean.to_vec(), (1, d_train), config.dev)?;

    let mut llik_sum = 0f64;
    let mut mask_cnt = 0f64;
    for (lb, ub) in create_jobs(ntot, 0, Some(config.minibatch_size)) {
        let x_dn = data_vec.read_columns_csc(lb..ub)?;
        let x0_nd = delta
            .map(|delta_bm| {
                expand_delta_for_block(data_vec, delta_bm, config.adj_method, lb, ub, config.dev)
            })
            .transpose()?;
        let n = ub - lb;
        let map = decoder.coarsening();
        let n_obs = decoder.dim_obs();

        // `(raw latent, module targets)`. The two reads differ only in which
        // genes the encoder was allowed to look at; the scored set is the
        // complement of that, derived from the same mask either way.
        let (raw_z, t) = match config.read {
            MaskedRead::Dense => {
                // Hold out a fraction of each cell's OBSERVED genes. Holding
                // out zeros as well would move the metric without adding
                // information: the score asks whether the model can put back
                // what was taken away, and a zero that stays zero is not an
                // imputation.
                let x_nd = csc_block_to_dense(&x_dn, d_train, None, config.dev)?;
                let observed: Vec<f32> = x_nd.flatten_all()?.to_vec1()?;
                let mut rng = StdRng::seed_from_u64(config.seed ^ (lb as u64));
                let mut vis = vec![1f32; n * d_train];
                for (slot, &v) in observed.iter().enumerate() {
                    if v > 0.0 && rng.random::<f64>() < config.mask_fraction {
                        vis[slot] = 0.0;
                    }
                }
                let visible_nd = Tensor::from_vec(vis, (n, d_train), config.dev)?;
                let raw_z = masked_encode_dense(
                    encoder,
                    config.head,
                    &MaskedDenseInput {
                        x_nd: &x_nd,
                        x0_nd: x0_nd.as_ref(),
                        mean_1d: Some(&mean_1d),
                        visible_nd: &visible_nd,
                    },
                    false,
                )?;
                // The same aggregation the trainer uses, so the held-out number
                // is on the training trace's scale.
                let t = dense_module_targets(map, &x_nd, &visible_nd)?;
                (raw_z, t)
            }
            MaskedRead::Windowed {
                context_size,
                shortlist_weights,
            } => {
                let ctx = PerGeneContext {
                    feature_mean: Some(config.feature_mean),
                };
                let enc_pack = csc_to_indexed(
                    &x_dn,
                    context_size,
                    shortlist_weights,
                    None,
                    ctx,
                    config.dev,
                )?;
                let values_null = x0_nd
                    .as_ref()
                    .map(|x0| gather_null_at_indices(x0, &enc_pack.indices, config.dev))
                    .transpose()?;
                // Seeded hold-out mask over the cell's real (value>0) top-K
                // positions. Host-side RNG keyed by `seed ^ lb` so the masked
                // set is deterministic and identical across heads.
                let (n_rows, k) = enc_pack.values.dims2()?;
                let values_host: Vec<f32> = enc_pack.values.flatten_all()?.to_vec1()?;
                let mut rng = StdRng::seed_from_u64(config.seed ^ (lb as u64));
                let mut mask_buf = vec![0f32; n_rows * k];
                for (slot, &v) in values_host.iter().enumerate() {
                    if v > 0.0 && rng.random::<f64>() < config.mask_fraction {
                        mask_buf[slot] = 1.0;
                    }
                }
                let masked = Tensor::from_vec(mask_buf, (n_rows, k), config.dev)?;
                let real = enc_pack.values.gt(0.0)?.to_dtype(candle_core::DType::F32)?;
                let visible = (&real - &masked)?;
                let raw_z = masked_encode(
                    encoder,
                    config.head,
                    &MaskedEncoderInput {
                        indices: &enc_pack.indices,
                        values: &enc_pack.values,
                        values_null: values_null.as_ref(),
                        values_mean: enc_pack.values_mean.as_ref(),
                        visible_mask: &visible,
                    },
                    false,
                )?;
                // The module view of the block, straight from the sparse columns.
                let f2c = map.host_fine_to_coarse();
                let mut dense = vec![0f32; n * n_obs];
                let mut lib = vec![1f32; n];
                for (j, col) in (lb..ub).enumerate() {
                    let c = x_dn.col(col - lb);
                    for (&r, &v) in c.row_indices().iter().zip(c.values().iter()) {
                        dense[j * n_obs + f2c[r]] += v;
                        lib[j] += v;
                    }
                }
                let m_ctx = map.groups_of(&enc_pack.indices)?;
                let share_ctx = map.log_share_at(&enc_pack.indices)?.exp()?;
                let t = DenseModuleTargets {
                    values_nm: Tensor::from_vec(dense, (n, n_obs), config.dev)?,
                    visible_counts_nm: scatter_add_cols(
                        &m_ctx,
                        &(&enc_pack.values * &visible)?,
                        n_obs,
                    )?,
                    visible_share_nm: scatter_add_cols(&m_ctx, &(share_ctx * &visible)?, n_obs)?,
                    lib_n1: Tensor::from_vec(lib, (n, 1), config.dev)?,
                };
                (raw_z, t)
            }
        };
        // Must match the training-time decoder coupling exactly, or the
        // held-out number is not comparable to the training trace.
        let log_z = decoder_log_theta(raw_z, config.head, config.topic_smoothing)?;

        // Score what the encoder did not see, matching the training law: every
        // module's unseen counts (every gene's, under the identity map) against
        // the module rate scaled by the unseen share, the full library, and
        // `residual` because these are cells, whose counts still carry the batch
        // effect.
        let residual_nm = x0_nd
            .as_ref()
            .map(|x0| map.aggregate_columns(&x0.broadcast_mul(&map.log_share_1d().exp()?)?))
            .transpose()?;
        let target = ModuleTarget {
            values: &t.values_nm,
            visible_counts: &t.visible_counts_nm,
            visible_share: &t.visible_share_nm,
            residual: residual_nm.as_ref(),
            lib: &t.lib_n1,
        };
        let (llik, units) = match config.likelihood {
            MaskedLikelihood::Nb => decoder.score_unseen_modules_nb(&log_z, &target, &full_kd)?,
            MaskedLikelihood::Multinomial => {
                decoder.score_unseen_modules_multinomial(&log_z, &target, &full_kd)?
            }
        };
        llik_sum += f64::from(llik.sum_all()?.to_scalar::<f32>()?);
        mask_cnt += f64::from(units.sum_all()?.to_scalar::<f32>()?);
    }

    Ok(if mask_cnt > 0.0 {
        (llik_sum / mask_cnt) as f32
    } else {
        f32::NAN
    })
}

#[cfg(test)]
#[path = "eval_indexed_tests.rs"]
mod eval_indexed_tests;
