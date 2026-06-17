//! `senna impute` — post-hoc kNN imputation against a reference dataset.
//!
//! Pipeline:
//! 1. Project the new sparse data through the trained masked-topic
//!    encoder (delegates to `senna predict`) to produce `θ_new [N_new, K]`.
//! 2. Load the reference dataset's training-time latent
//!    `θ_ref [N_ref, K]` (a parquet written by the original
//!    `masked-topic` run).
//! 3. Build an HNSW kNN index over `θ_ref` (L2) and, for each new cell,
//!    find the K nearest reference cells.
//! 4. Convert kNN L2 distances to softmax weights with temperature τ
//!    (smaller τ ⇒ sharper, fewer-neighbor-effective).
//! 5. For each new cell, weighted-average the reference cells'
//!    full-feature counts to produce the imputed full-feature row.
//! 6. Write `{out}.imputed.parquet` (`N_new` × `n_ref_features`).
//!
//! Information-theoretic note (per the rag-augmentation memory): the
//! imputed expression is a function of `θ_new` through the reference's
//! distribution-preserving retrieval. It is the genuine RAG payoff
//! (residual full-rank covariance β can't carry), not a deterministic
//! β·θ readout.

use crate::embed_common::*;
use crate::predict::{predict_model, PredictArgs};
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use clap::Args;
use data_beans::sparse_io_vector::SparseIoVec;
use log::info;
use matrix_util::knn_match::{ColumnDict, VecPoint};
use matrix_util::traits::IoOps;
use rayon::prelude::*;

#[derive(Args, Debug)]
pub struct ImputeArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "New (typically sparse-panel) data files (.zarr or .h5) to impute"
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(long, required = true, help = "Trained masked-topic model prefix")]
    pub model: Box<str>,

    #[arg(short, long, required = true, help = "Output file prefix")]
    pub out: Box<str>,

    #[arg(
        long,
        required = true,
        help = "Reference latent parquet (e.g. {train_out}.latent.parquet)"
    )]
    pub reference_latent: Box<str>,

    #[arg(
        long,
        value_delimiter = ',',
        required = true,
        help = "Reference data files used to compute --reference-latent"
    )]
    pub reference_data: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Per-file batch labels for --reference-data"
    )]
    pub reference_batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        help = "Per-file batch labels for --data-files"
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 25,
        help = "Number of reference nearest neighbours to pool per new cell"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature on kNN distances (lower = sharper neighbour weights)"
    )]
    pub knn_temperature: f32,

    #[arg(long, default_value_t = 500, help = "Predict / read minibatch size")]
    pub minibatch_size: usize,

    #[arg(long, help = "Cells per delta-estimation block (auto by default)")]
    pub block_size: Option<usize>,

    #[arg(long, help = "Load all columns into memory before evaluation")]
    pub preload_data: bool,

    #[arg(short, long, help = "Verbose logging")]
    pub verbose: bool,
}

pub fn impute_model(args: &ImputeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    // 1. Run senna-predict on the new data → writes {out}.predict_tmp.latent.parquet
    let predict_prefix: Box<str> = format!("{}.predict_tmp", args.out).into();
    info!("Step 1/4: projecting new data through encoder (predict → {predict_prefix})");
    let predict_args = PredictArgs {
        data_files: args.data_files.clone(),
        model: args.model.clone(),
        out: predict_prefix.clone(),
        batch_files: args.batch_files.clone(),
        minibatch_size: args.minibatch_size,
        block_size: args.block_size,
        preload_data: args.preload_data,
        refine_steps: 0,
        refine_lr: 0.01,
        refine_reg: 1.0,
        decoder_only: false,
        delta_iters: 3,
        verbose: args.verbose,
        residual_out: None,
        residual_include_delta: false,
        residual_threshold: 0.0,
        feature_name_kind: crate::masked_topic::FeatureNameKindArg::Exact,
        feature_name_suffix_delim: None,
        keep_feature_suffix: None,
    };
    predict_model(&predict_args)?;

    // 2. Load log-θ_new and θ_ref, exponentiate to the simplex.
    let theta_new_path = format!("{predict_prefix}.latent.parquet");
    info!("Step 2/4: loading projected latent and reference latent");
    let theta_new_parq =
        nalgebra::DMatrix::<f32>::from_parquet_with_row_names(&theta_new_path, Some(0))?;
    let theta_ref_parq =
        nalgebra::DMatrix::<f32>::from_parquet_with_row_names(&args.reference_latent, Some(0))?;
    let new_cell_names: Vec<Box<str>> = theta_new_parq.rows;
    let mut theta_new = theta_new_parq.mat;
    let mut theta_ref = theta_ref_parq.mat;
    let (n_new, k_new) = (theta_new.nrows(), theta_new.ncols());
    let (n_ref, k_ref) = (theta_ref.nrows(), theta_ref.ncols());
    anyhow::ensure!(
        k_new == k_ref,
        "topic dimension mismatch: theta_new K={k_new} vs reference K={k_ref}"
    );
    info!("  θ_new: {n_new} cells × {k_new} topics; θ_ref: {n_ref} cells × {k_ref} topics");

    // Latent parquets are log-softmax; exp them back to the simplex.
    // L2 on the simplex correlates with cosine for the cell-cell matching
    // the existing L2-backed `ColumnDict` supports.
    theta_new.apply(|x| *x = x.exp());
    theta_ref.apply(|x| *x = x.exp());

    // 3. Build HNSW over θ_ref and find K nearest reference cells per new cell.
    info!(
        "Step 3/4: building kNN index over reference (k={}, τ={})",
        args.knn, args.knn_temperature
    );
    // Transpose once so reference cells become columns; `from_dvector_views`
    // then borrows column views directly — no per-row owned DVector copy.
    let theta_ref_t = theta_ref.transpose();
    let ref_dict = ColumnDict::<u32>::from_dvector_views(
        theta_ref_t.column_iter().collect(),
        (0..n_ref as u32).collect(),
    );

    let knn = args.knn.min(n_ref);
    let neighbours: Vec<(Vec<u32>, Vec<f32>)> = (0..n_new)
        .into_par_iter()
        .map(|i| {
            let row = theta_new.row(i);
            let vp = VecPoint {
                data: row.iter().copied().collect(),
            };
            ref_dict.search_by_query_data(&vp, knn).expect("kNN search")
        })
        .collect();

    // 4. Open reference data + impute. We stream reference columns
    // chunk by chunk and accumulate weighted sums into a dense
    // [N_new, G_ref] output matrix.
    info!("Step 4/4: opening reference data and imputing");
    let ref_loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.reference_data.clone(),
        batch_files: args.reference_batch_files.clone(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let ref_data: SparseIoVec = ref_loaded.data;
    anyhow::ensure!(
        ref_data.num_columns() == n_ref,
        "reference data has {} cells but reference latent has {}; \
         data files don't match the latent file",
        ref_data.num_columns(),
        n_ref,
    );
    let g_ref = ref_data.num_rows();
    let ref_gene_names = ref_data.row_names()?;

    // Heads-up before the big allocation: dense [N_new, G_ref] f32 grows
    // fast (10k cells × 20k genes ≈ 800 MB). Streaming-sparse output is a
    // future optimization; for now we materialize and write whole.
    let bytes_est = (n_new * g_ref).saturating_mul(4);
    if bytes_est > (1 << 30) {
        log::warn!(
            "imputed dense matrix will allocate ~{} MB ({} × {} f32). \
             Consider reducing N_new or G_ref if memory is tight.",
            bytes_est >> 20,
            n_new,
            g_ref,
        );
    }

    // Pre-compute softmax weights per new cell.
    let weights: Vec<Vec<f32>> = neighbours
        .par_iter()
        .map(|(_, d)| dist_to_softmax_weights(d, args.knn_temperature))
        .collect();

    // Invert the (new_cell → neighbours) map so column-streaming reference
    // data can do `consumers[ref_id]` lookups in O(1) and skip whole CSC
    // chunks where no consumer touches any of the chunk's cells.
    let mut cell_to_consumers: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_ref];
    for (new_id, (nbrs, _)) in neighbours.iter().enumerate() {
        for (k, &c) in nbrs.iter().enumerate() {
            cell_to_consumers[c as usize].push((new_id as u32, weights[new_id][k]));
        }
    }

    let mut imputed = nalgebra::DMatrix::<f32>::zeros(n_new, g_ref);
    let chunk_size = args.minibatch_size.max(64);
    let mut col_lb = 0;
    while col_lb < n_ref {
        let col_ub = (col_lb + chunk_size).min(n_ref);
        // Skip the read entirely if no consumer touches this chunk —
        // common when N_new × knn ≪ N_ref.
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
                for &(new_id, w) in consumers {
                    imputed[(new_id as usize, row_id)] += w * v;
                }
            }
        }
        col_lb = col_ub;
    }

    let imputed_path = format!("{}.imputed.parquet", args.out);
    imputed.to_parquet_with_names(
        &imputed_path,
        (Some(&new_cell_names), Some("cell")),
        Some(&ref_gene_names),
    )?;
    info!("Wrote imputed {n_new}× {g_ref} matrix to {imputed_path}");

    Ok(())
}

/// Convert kNN L2 distances to a positive-weight simplex via
/// `w_k ∝ exp(-d_k² / (2 τ²))`, normalised to sum to 1.
fn dist_to_softmax_weights(distances: &[f32], temperature: f32) -> Vec<f32> {
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
