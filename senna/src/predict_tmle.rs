//! Iterative TMLE refinement of held-out per-batch δ.
//!
//! Each iteration runs the frozen encoder over all held-out cells with the
//! current δ as the input null, computes per-cell θ̂, then refits δ via the
//! Poisson / NB-Fisher-weighted MLE:
//!
//! ```text
//! pred[d,j]   = lib[j] · Σ_k θ̂[j,k] · exp(β[d,k])
//! w[d,j]      = φ[d] / (pred[d,j] + φ[d])     (NB; w = 1 if no φ)
//! δ[d,b]      = Σ_{j∈b} w[d,j]·x[d,j] / Σ_{j∈b} w[d,j]·pred[d,j]
//! ```
//!
//! With the encoder frozen, this is the single-fluctuation TMLE step
//! (multinomial / NB working submodel) applied to the held-out distribution;
//! iterating to a fixed point gives full TMLE for δ.

use crate::embed_common::*;
use crate::topic::common::expand_delta_for_block;
use crate::topic::eval::GeneRemap;
use crate::topic::eval_indexed::dense_to_indexed;
use crate::topic::predict_common::{nb_fisher_weight, solve_delta_from_sums, DeltaSums};

use crate::logging::new_progress_bar;
use candle_core::{Device, Tensor};
use candle_util::candle_encoder_indexed::IndexedEmbeddingEncoder;
use candle_util::candle_encoder_softmax::LogSoftmaxEncoder;
use candle_util::candle_indexed_model_traits::IndexedEncoderT;
use candle_util::candle_model_traits::EncoderModuleT;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use indicatif::ParallelProgressIterator;
use log::info;
use rayon::prelude::*;

/// Build per-cell input null `x0_nd` (encoder dim) from the current δ at full D.
fn build_delta_tensor_at_encoder_dim(
    delta_db_full: &Mat,
    coarsening: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let mut db = delta_db_full.clone();
    if let Some(fc) = coarsening {
        db = fc.aggregate_rows_ds(&db);
    }
    Ok(db.to_tensor(dev)?.transpose(0, 1)?.contiguous()?)
}

/// Scatter held-out CSC rows into training-D order. Used for the TMLE update,
/// which works at full D regardless of encoder coarsening.
fn remap_csc_to_dense_full(
    csc: &nalgebra_sparse::CscMatrix<f32>,
    gene_remap: Option<&GeneRemap>,
    d_train: usize,
) -> Mat {
    let ncols = csc.ncols();
    let mut out = Mat::zeros(d_train, ncols);
    if let Some(remap) = gene_remap {
        for j in 0..ncols {
            let col = csc.col(j);
            for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
                if let Some(row_train) = remap.new_to_train[row_new] {
                    out[(row_train, j)] = val;
                }
            }
        }
    } else {
        for j in 0..ncols {
            let col = csc.col(j);
            for (&row, &val) in col.row_indices().iter().zip(col.values().iter()) {
                out[(row, j)] = val;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn accumulate_block_dense(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &LogSoftmaxEncoder,
    delta_tensor_coarse: Option<&Tensor>,
    gene_remap: Option<&GeneRemap>,
    coarsening: Option<&FeatureCoarsening>,
    exp_beta_dk: &Mat,
    phi: Option<&[f32]>,
    n_batches: usize,
    dev: &Device,
    adj_method: &AdjMethod,
) -> anyhow::Result<DeltaSums> {
    let (lb, ub) = block;
    let d_train = exp_beta_dk.nrows();

    // Held-out data scattered to training-D
    let csc = data_vec.read_columns_csc(lb..ub)?;
    let x_dn_full = remap_csc_to_dense_full(&csc, gene_remap, d_train);
    let n_block = x_dn_full.ncols();

    // Encoder input (coarsened if applicable)
    let x_at_enc = if let Some(fc) = coarsening {
        fc.aggregate_rows_ds(&x_dn_full)
    } else {
        x_dn_full.clone()
    };
    let x_enc_nd = x_at_enc.to_tensor(dev)?.transpose(0, 1)?.contiguous()?;

    // x0 from current δ (already at encoder dim)
    let x0_nd = delta_tensor_coarse
        .map(|delta_bm| expand_delta_for_block(data_vec, delta_bm, adj_method, lb, ub, dev))
        .transpose()?;

    // Encode → θ̂  [N_block, K]
    let (log_z_nk, _) = encoder.forward_t(&x_enc_nd, x0_nd.as_ref(), false)?;
    let theta_nk_t = log_z_nk.exp()?.to_device(&Device::Cpu)?;
    let theta_nk: Mat = Mat::from_tensor(&theta_nk_t)?; // [N_block, K]

    // Predicted gene proportions per cell at full D: pred[d,j] = Σ_k θ̂[j,k]·exp(β[d,k])
    // Compute as exp_beta_dk · θ_nkᵀ → [D_full, N_block]
    let pred_dn: Mat = exp_beta_dk * theta_nk.transpose();

    // Per-cell library at training-D — sum CSC values directly (O(nnz)) rather
    // than reducing the dense matrix column-by-column.
    let lib_per_cell: Vec<f32> = (0..n_block)
        .map(|j| csc.col(j).values().iter().sum())
        .collect();

    // Accumulate per-batch (NB-Fisher-weighted at δ=1 working hypothesis)
    let batch_ids = data_vec.get_batch_membership(lb..ub);
    let mut sums = DeltaSums::zeros(d_train, n_batches);
    for j in 0..n_block {
        let b = batch_ids[j];
        let lib_j = lib_per_cell[j];
        for d in 0..d_train {
            let mu = lib_j * pred_dn[(d, j)];
            let w = nb_fisher_weight(phi, mu, d);
            sums.pb_obs[(d, b)] += w * x_dn_full[(d, j)];
            sums.pb_pred[(d, b)] += w * mu;
        }
    }
    Ok(sums)
}

/// Run iterative TMLE δ-refinement on the dense path.
/// Returns the converged `[D_full, n_batches]` δ matrix.
#[allow(clippy::too_many_arguments)]
pub fn iterate_delta_dense(
    n_iters: usize,
    initial_delta: Mat,
    data_vec: &SparseIoVec,
    encoder: &LogSoftmaxEncoder,
    gene_remap: Option<&GeneRemap>,
    coarsening: Option<&FeatureCoarsening>,
    beta_dk_full: &Mat,
    phi: Option<&[f32]>,
    minibatch_size: usize,
    dev: &Device,
    adj_method: &AdjMethod,
) -> anyhow::Result<Mat> {
    let mut delta = initial_delta;
    let n_batches = delta.ncols();
    let exp_beta_dk = beta_dk_full.map(f32::exp);
    let ntot = data_vec.num_columns();

    for iter in 0..n_iters {
        let delta_tensor_coarse = build_delta_tensor_at_encoder_dim(&delta, coarsening, dev)?;
        let jobs = create_jobs(ntot, 0, Some(minibatch_size));
        let njobs = jobs.len() as u64;

        let chunk_sums: Vec<DeltaSums> = jobs
            .par_iter()
            .progress_with(new_progress_bar(njobs))
            .map(|&block| {
                accumulate_block_dense(
                    block,
                    data_vec,
                    encoder,
                    Some(&delta_tensor_coarse),
                    gene_remap,
                    coarsening,
                    &exp_beta_dk,
                    phi,
                    n_batches,
                    dev,
                    adj_method,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut total = DeltaSums::zeros(beta_dk_full.nrows(), n_batches);
        for s in &chunk_sums {
            total.merge_into(s);
        }
        let delta_new = solve_delta_from_sums(&total);

        let max_change = (0..delta.nrows())
            .flat_map(|d| (0..n_batches).map(move |b| (d, b)))
            .map(|(d, b)| (delta_new[(d, b)] - delta[(d, b)]).abs())
            .fold(0.0_f32, f32::max);
        info!(
            "delta-iter {}/{n_iters}: max |Δδ| = {max_change:.4}, mean δ = {:.3}",
            iter + 1,
            delta_new.iter().sum::<f32>() / (delta_new.nrows() * n_batches) as f32,
        );
        delta = delta_new;
    }
    Ok(delta)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_block_indexed(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: &IndexedEmbeddingEncoder,
    delta_tensor: Option<&Tensor>, // [B, D_full]
    gene_remap: Option<&GeneRemap>,
    exp_beta_dk: &Mat,
    phi: Option<&[f32]>,
    shortlist_weights: &[f32],
    enc_context_size: usize,
    n_batches: usize,
    dev: &Device,
    adj_method: &AdjMethod,
) -> anyhow::Result<DeltaSums> {
    let (lb, ub) = block;
    let d_train = exp_beta_dk.nrows();

    let csc = data_vec.read_columns_csc(lb..ub)?;
    let x_dn_full = remap_csc_to_dense_full(&csc, gene_remap, d_train);
    let n_block = x_dn_full.ncols();
    let x_nd_t = x_dn_full.to_tensor(dev)?.transpose(0, 1)?.contiguous()?;

    let x0_nd = delta_tensor
        .map(|delta_bm| expand_delta_for_block(data_vec, delta_bm, adj_method, lb, ub, dev))
        .transpose()?;

    // Indexed shortlist for encoder
    let (enc_union, enc_indexed_x) =
        dense_to_indexed(&x_nd_t, enc_context_size, shortlist_weights, dev)?;

    let enc_indexed_x_null = if let Some(x0) = &x0_nd {
        let union_vec: Vec<u32> = enc_union.to_vec1()?;
        let s = union_vec.len();
        let x0_vec: Vec<Vec<f32>> = x0.to_vec2()?;
        let mut x0_data = vec![0.0f32; n_block * s];
        for (rr, x0_row) in x0_vec.iter().enumerate() {
            for (col, &feat_idx) in union_vec.iter().enumerate() {
                x0_data[rr * s + col] = x0_row[feat_idx as usize];
            }
        }
        Some(Tensor::from_vec(x0_data, (n_block, s), dev)?)
    } else {
        None
    };

    let (log_z_nk, _) = encoder.forward_indexed_t(
        &enc_union,
        &enc_indexed_x,
        enc_indexed_x_null.as_ref(),
        false,
    )?;
    let theta_nk_t = log_z_nk.exp()?.to_device(&Device::Cpu)?;
    let theta_nk: Mat = Mat::from_tensor(&theta_nk_t)?;

    let pred_dn: Mat = exp_beta_dk * theta_nk.transpose();

    // Per-cell library at training-D — sum CSC values directly (O(nnz)).
    let lib_per_cell: Vec<f32> = (0..n_block)
        .map(|j| csc.col(j).values().iter().sum())
        .collect();

    let batch_ids = data_vec.get_batch_membership(lb..ub);
    let mut sums = DeltaSums::zeros(d_train, n_batches);
    for j in 0..n_block {
        let b = batch_ids[j];
        let lib_j = lib_per_cell[j];
        for d in 0..d_train {
            let mu = lib_j * pred_dn[(d, j)];
            let w = nb_fisher_weight(phi, mu, d);
            sums.pb_obs[(d, b)] += w * x_dn_full[(d, j)];
            sums.pb_pred[(d, b)] += w * mu;
        }
    }
    Ok(sums)
}

#[allow(clippy::too_many_arguments)]
pub fn iterate_delta_indexed(
    n_iters: usize,
    initial_delta: Mat,
    data_vec: &SparseIoVec,
    encoder: &IndexedEmbeddingEncoder,
    gene_remap: Option<&GeneRemap>,
    beta_dk_full: &Mat,
    phi: Option<&[f32]>,
    shortlist_weights: &[f32],
    enc_context_size: usize,
    minibatch_size: usize,
    dev: &Device,
    adj_method: &AdjMethod,
) -> anyhow::Result<Mat> {
    let mut delta = initial_delta;
    let n_batches = delta.ncols();
    let exp_beta_dk = beta_dk_full.map(f32::exp);
    let ntot = data_vec.num_columns();

    for iter in 0..n_iters {
        // For indexed: encoder consumes x0 at full D, gathered later at union positions.
        let delta_tensor: Tensor = delta
            .clone()
            .to_tensor(dev)?
            .transpose(0, 1)?
            .contiguous()?;

        let jobs = create_jobs(ntot, 0, Some(minibatch_size));
        let njobs = jobs.len() as u64;

        let chunk_sums: Vec<DeltaSums> = jobs
            .par_iter()
            .progress_with(new_progress_bar(njobs))
            .map(|&block| {
                accumulate_block_indexed(
                    block,
                    data_vec,
                    encoder,
                    Some(&delta_tensor),
                    gene_remap,
                    &exp_beta_dk,
                    phi,
                    shortlist_weights,
                    enc_context_size,
                    n_batches,
                    dev,
                    adj_method,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut total = DeltaSums::zeros(beta_dk_full.nrows(), n_batches);
        for s in &chunk_sums {
            total.merge_into(s);
        }
        let delta_new = solve_delta_from_sums(&total);

        let max_change = (0..delta.nrows())
            .flat_map(|d| (0..n_batches).map(move |b| (d, b)))
            .map(|(d, b)| (delta_new[(d, b)] - delta[(d, b)]).abs())
            .fold(0.0_f32, f32::max);
        info!(
            "delta-iter {}/{n_iters}: max |Δδ| = {max_change:.4}, mean δ = {:.3}",
            iter + 1,
            delta_new.iter().sum::<f32>() / (delta_new.nrows() * n_batches) as f32,
        );
        delta = delta_new;
    }
    Ok(delta)
}
