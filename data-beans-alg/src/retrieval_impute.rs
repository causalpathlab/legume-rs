//! Retrieval-based imputation: pool a sparse reference's full-feature
//! counts onto query cells matched in a shared latent space.
//!
//! The caller owns the latent semantics — both matrices must already be
//! mapped onto whatever space makes their rows comparable under L2
//! (softmaxed topic proportions, L2-normalized embeddings, …). This
//! module only does the model-agnostic part:
//!
//! 1. Build an approximate kNN index over the reference rows.
//! 2. For each query row, find the K nearest reference rows and convert
//!    their L2 distances to softmax weights with temperature τ.
//! 3. Stream the reference backend's columns chunk by chunk and
//!    accumulate each query row's weighted average of its neighbours'
//!    counts into a dense `[N_query, G_ref]` matrix.
//!
//! The information-theoretic point of retrieval over a decoder readout:
//! the imputed expression inherits the reference's full-rank gene–gene
//! covariance through the retrieved cells, rather than being confined to
//! the rank of the model's dictionary.

use data_beans::sparse_io_vector::SparseIoVec;
use log::{info, warn};
use matrix_util::knn_match::ColumnDict;
use rayon::prelude::*;

type Mat = nalgebra::DMatrix<f32>;

pub struct RetrievalImputeConfig {
    /// Reference nearest neighbours pooled per query row.
    pub knn: usize,
    /// Softmax temperature on kNN distances (lower = sharper weights).
    pub temperature: f32,
    /// Reference columns per streamed read.
    pub chunk: usize,
}

/// Weighted-kNN imputation of `ref_data`'s features onto the query rows.
///
/// `query_latent` is `[N_query, K]` and `ref_latent` is `[N_ref, K]`, in
/// the same (caller-established) matching space; `ref_data` holds the
/// reference counts with one column per `ref_latent` row. Returns the
/// dense imputed `[N_query, G_ref]` matrix.
///
/// A query row with zero norm carries no matching evidence (an
/// unassignable cell, an empty propensity row); it is skipped — its
/// imputed row stays zero — and the count of such rows is logged.
pub fn retrieval_impute(
    query_latent: &Mat,
    ref_latent: &Mat,
    ref_data: &SparseIoVec,
    cfg: &RetrievalImputeConfig,
) -> anyhow::Result<Mat> {
    let (n_query, k_query) = (query_latent.nrows(), query_latent.ncols());
    let (n_ref, k_ref) = (ref_latent.nrows(), ref_latent.ncols());
    anyhow::ensure!(
        k_query == k_ref,
        "latent dimension mismatch: query K={k_query} vs reference K={k_ref}"
    );
    anyhow::ensure!(n_ref > 0, "reference latent has no rows");
    anyhow::ensure!(
        ref_data.num_columns() == n_ref,
        "reference data has {} cells but reference latent has {n_ref}; \
         the data files don't match the latent",
        ref_data.num_columns(),
    );

    info!(
        "Building kNN index over {n_ref} reference rows (k={}, τ={})",
        cfg.knn, cfg.temperature
    );
    // Transpose once so reference rows become columns; `from_dvector_views`
    // then borrows column views directly — no per-row owned copy.
    let ref_latent_t = ref_latent.transpose();
    let ref_dict = ColumnDict::<u32>::from_dvector_views(
        ref_latent_t.column_iter().collect(),
        (0..n_ref as u32).collect(),
    );

    let knn = cfg.knn.min(n_ref);
    let neighbours: Vec<Option<(Vec<u32>, Vec<f32>)>> = (0..n_query)
        .into_par_iter()
        .map(|i| {
            let query: Vec<f32> = query_latent.row(i).iter().copied().collect();
            if query.iter().all(|&x| x == 0.0) {
                return Ok(None);
            }
            ref_dict
                .search_by_query_data(&query, knn)
                .map(Some)
                .map_err(|e| anyhow::anyhow!("kNN search for query row {i}: {e}"))
        })
        .collect::<anyhow::Result<_>>()?;
    let n_skipped = neighbours.iter().filter(|n| n.is_none()).count();
    if n_skipped > 0 {
        warn!(
            "{n_skipped} of {n_query} query rows have a zero latent and were skipped; \
             their imputed rows stay zero"
        );
    }

    let g_ref = ref_data.num_rows();

    // Heads-up before the big allocation: dense [N_query, G_ref] f32 grows
    // fast. Streaming-sparse output is a future optimization; for now we
    // materialize and let the caller write whole.
    let bytes_est = (n_query * g_ref).saturating_mul(4);
    if bytes_est > (1 << 30) {
        warn!(
            "imputed dense matrix will allocate ~{} MB ({n_query} × {g_ref} f32). \
             Consider reducing the query size or the reference feature set if \
             memory is tight.",
            bytes_est >> 20,
        );
    }

    // Pre-compute softmax weights per query row.
    let weights: Vec<Vec<f32>> = neighbours
        .par_iter()
        .map(|nbr| match nbr {
            Some((_, d)) => dist_to_softmax_weights(d, cfg.temperature),
            None => Vec::new(),
        })
        .collect();

    // Invert the (query row → neighbours) map so column-streaming the
    // reference can do `consumers[ref_id]` lookups in O(1) and skip whole
    // chunks where no consumer touches any of the chunk's cells.
    let mut cell_to_consumers: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_ref];
    for (query_id, nbr) in neighbours.iter().enumerate() {
        if let Some((nbrs, _)) = nbr {
            for (k, &c) in nbrs.iter().enumerate() {
                cell_to_consumers[c as usize].push((query_id as u32, weights[query_id][k]));
            }
        }
    }

    let mut imputed = Mat::zeros(n_query, g_ref);
    let chunk_size = cfg.chunk.max(64);
    let mut col_lb = 0;
    while col_lb < n_ref {
        let col_ub = (col_lb + chunk_size).min(n_ref);
        // Skip the read entirely if no consumer touches this chunk —
        // common when N_query × knn ≪ N_ref.
        if cell_to_consumers[col_lb..col_ub]
            .iter()
            .all(std::vec::Vec::is_empty)
        {
            col_lb = col_ub;
            continue;
        }
        let csc = ref_data.read_columns_csc(col_lb..col_ub)?;
        for c_local in 0..csc.ncols() {
            let consumers = &cell_to_consumers[col_lb + c_local];
            if consumers.is_empty() {
                continue;
            }
            let col = csc.col(c_local);
            for (&row_id, &v) in col.row_indices().iter().zip(col.values().iter()) {
                for &(query_id, w) in consumers {
                    imputed[(query_id as usize, row_id)] += w * v;
                }
            }
        }
        col_lb = col_ub;
    }

    Ok(imputed)
}

/// Convert kNN L2 distances to a positive-weight simplex via
/// `w_k ∝ exp(-d_k² / (2 τ²))`, normalised to sum to 1.
#[must_use]
pub fn dist_to_softmax_weights(distances: &[f32], temperature: f32) -> Vec<f32> {
    if distances.is_empty() {
        return Vec::new();
    }
    let tau = temperature.max(1e-6);
    let scale = 1.0 / (2.0 * tau * tau);
    let mut max = f32::NEG_INFINITY;
    let mut out: Vec<f32> = distances
        .iter()
        .map(|d| {
            let v = -d * d * scale;
            if v > max {
                max = v;
            }
            v
        })
        .collect();
    let mut sum = 0.0f32;
    for x in &mut out {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(1e-12);
    for x in &mut out {
        *x *= inv;
    }
    out
}

#[cfg(test)]
#[path = "retrieval_impute_tests.rs"]
mod tests;
