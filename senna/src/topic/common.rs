use crate::embed_common::*;
use crate::hvg::{select_hvg_streaming, HvgSelection};
use crate::logging::new_progress_bar;
use crate::senna_input::{read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch};

use candle_core::{Device, Tensor};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Run a block-processing closure over `ntot` items in blocks of `block_size`,
/// dispatching in parallel on CPU or sequentially on GPU.
///
/// The closure receives `(lb, ub)` and must return `(lb, Mat)`.
/// Results are reassembled into a single `ntot × kk` matrix.
pub(crate) fn process_blocks<F>(
    ntot: usize,
    kk: usize,
    block_size: usize,
    dev: &Device,
    eval_block: F,
) -> anyhow::Result<Mat>
where
    F: Fn((usize, usize)) -> anyhow::Result<(usize, Mat)> + Send + Sync,
{
    let jobs = create_jobs(ntot, 0, Some(block_size));
    let njobs = jobs.len() as u64;

    let mut chunks: Vec<(usize, Mat)> = if dev.is_cpu() {
        jobs.par_iter()
            .progress_count(njobs)
            .map(|&block| eval_block(block))
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        let pb = new_progress_bar(njobs);
        let result = jobs
            .iter()
            .map(|&block| {
                let r = eval_block(block);
                pb.inc(1);
                r
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        pb.finish_and_clear();
        result
    };

    chunks.sort_by_key(|&(lb, _)| lb);

    let mut ret = Mat::zeros(ntot, kk);
    let mut lb = 0;
    for (_, z) in chunks {
        let ub = lb + z.nrows();
        ret.rows_range_mut(lb..ub).copy_from(&z);
        lb = ub;
    }
    Ok(ret)
}

/// Expand a precomputed delta tensor `[B, D]` to `[N, D]` using per-sample
/// batch or group membership indices for the block `lb..ub`.
pub(crate) fn expand_delta_for_block(
    data_vec: &SparseIoVec,
    delta_bd: &Tensor,
    adj_method: &AdjMethod,
    lb: usize,
    ub: usize,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let membership: Vec<u32> = match adj_method {
        AdjMethod::Batch => data_vec
            .get_batch_membership(lb..ub)
            .into_iter()
            .map(|x| x as u32)
            .collect(),
        AdjMethod::Residual => data_vec
            .get_group_membership(lb..ub)?
            .into_iter()
            .map(|x| x as u32)
            .collect(),
    };
    let indices = Tensor::from_iter(membership.into_iter(), dev)?;
    Ok(delta_bd.index_select(&indices, 0)?)
}

/// Posterior-mean PB matrix `[D, n_pb]`, preferring the batch-adjusted
/// estimate when available. Anchor selection and ambient-profile
/// estimation both want the cleanest cell-type signal — the batch-
/// adjusted posterior strips per-batch effects out of the mean.
pub fn preferred_posterior_mean(collapsed: &CollapsedOut) -> &Mat {
    collapsed.mu_adjusted.as_ref().map_or_else(
        || collapsed.mu_observed.posterior_mean(),
        matrix_param::traits::Inference::posterior_mean,
    )
}

/// Posterior log-mean PB matrix `[D, n_pb]`, preferring batch-adjusted.
/// For Gamma(α, β), returns E[log X] = ψ(α) - log(β).
pub fn preferred_posterior_log_mean(collapsed: &CollapsedOut) -> &Mat {
    collapsed.mu_adjusted.as_ref().map_or_else(
        || collapsed.mu_observed.posterior_log_mean(),
        matrix_param::traits::Inference::posterior_log_mean,
    )
}

/// Compute per-level epoch allocation for progressive training.
///
/// Coarser levels (lower index) get more epochs: `w[i] = num_levels - i`.
pub(crate) fn compute_level_epochs(total_epochs: usize, num_levels: usize) -> Vec<usize> {
    let total_weight: usize = (1..=num_levels).sum();
    (0..num_levels)
        .map(|i| {
            let w = num_levels - i;
            (total_epochs * w / total_weight).max(1)
        })
        .collect()
}

/// Divide each column of `data` by the corresponding delta value, clamped to `min_val`.
///
/// `delta_row` is `[1, D]` (each column j divided by `delta_row[(0, j)]`).
pub(crate) fn apply_column_delta(data: &Mat, delta_row: &Mat, min_val: f32) -> Mat {
    let mut corrected = data.clone();
    for j in 0..corrected.ncols() {
        let d = delta_row[(0, j)].max(min_val);
        for i in 0..corrected.nrows() {
            corrected[(i, j)] /= d;
        }
    }
    corrected
}

/// Draw `(mixed_nd, batch_nd, target_nd)` from the collapsed posteriors
/// (one sample per Gamma matrix).
pub(crate) fn sample_collapsed_data(
    collapsed: &CollapsedOut,
) -> anyhow::Result<(Mat, Option<Mat>, Mat)> {
    let mixed_nd = collapsed.mu_observed.posterior_sample()?.transpose();

    let batch_nd = collapsed.mu_residual.as_ref().map(|x| {
        let ret: Mat = x.posterior_sample().unwrap();
        ret.transpose()
    });

    let target_nd = if let Some(adj) = &collapsed.mu_adjusted {
        adj.posterior_sample()?.transpose()
    } else {
        mixed_nd.clone()
    };

    Ok((mixed_nd, batch_nd, target_nd))
}

/// Result of loading and collapsing input data for topic model training.
pub struct PreparedData {
    pub data_vec: SparseIoVec,
    pub collapsed_levels: Vec<CollapsedOut>,
    /// Random-projection feature matrix used for PB partitioning
    /// (`proj_dim × n_cells`). Kept alongside the collapsed output so
    /// downstream steps (e.g. viz cell placement) can reuse it without
    /// recomputing.
    pub proj_kn: Mat,
}

/// Result of the read + batch + HVG + project pipeline. Shared by
/// `load_and_collapse` (multilevel collapse downstream) and `senna svd`
/// (single-level collapse downstream).
pub struct ProjectedData {
    pub data_vec: SparseIoVec,
    pub batch_membership: Vec<Box<str>>,
    /// `proj_dim × n_cells` random-projection sketch (post-batch-correction
    /// when batch labels are present). Same shape and semantics as
    /// `PreparedData::proj_kn`.
    pub proj_kn: Mat,
    /// HVG selection used to weight the basis, if any. `None` when HVG
    /// is disabled or when warm-starting from a saved projection.
    pub selected_features: Option<HvgSelection>,
}

/// Args for [`load_and_project`] — the read + batch + HVG + project
/// portion shared by topic, indexed-topic, and svd routines.
pub struct LoadProjectArgs<'a> {
    pub data_files: &'a [Box<str>],
    pub batch_files: &'a Option<Vec<Box<str>>>,
    pub preload: bool,
    pub warm_start_proj_file: Option<&'a str>,
    pub proj_dim: usize,
    pub block_size: Option<usize>,
    pub max_features: usize,
    pub feature_list_file: Option<&'a str>,
    pub ignore_batch: bool,
}

/// Read sparse files, resolve batch membership, optionally pick HVGs,
/// then run the random projection (or load a warm-start projection).
/// Used by `load_and_collapse` and by `senna svd` so the pre-collapse
/// pipeline is identical across routines.
pub fn load_and_project(args: &LoadProjectArgs) -> anyhow::Result<ProjectedData> {
    let SparseDataWithBatch {
        data: data_vec,
        batch: mut batch_membership,
        ..
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.to_vec(),
        batch_files: args.batch_files.clone(),
        preload: args.preload,
    })?;
    if args.ignore_batch {
        info!("--ignore-batch: collapsing all cells to a single batch");
        crate::senna_input::collapse_to_single_batch(&mut batch_membership);
    }

    let mut selected_features: Option<HvgSelection> = None;

    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file {
        use matrix_util::common_io::file_ext;
        let ext = file_ext(proj_file)?;

        let MatWithNames {
            rows: cell_names,
            cols: _,
            mat: proj_nk,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_row_names(proj_file, Some(0))?,
            _ => Mat::read_data_with_names(proj_file, &['\t', ',', ' '], Some(0), Some(0))?,
        };

        if data_vec.column_names()? != cell_names {
            return Err(anyhow::anyhow!(
                "warm start projection rows don't match with the data"
            ));
        }

        proj_nk.transpose()
    } else {
        // HVG-weighted projection: down-weight uninformative genes so the
        // random sketch (and hence the PB partitioning + cached cell_proj)
        // reflects variable biology. Collapsing still reads all genes.
        let hvg_enabled = args.max_features > 0 || args.feature_list_file.is_some();
        if hvg_enabled {
            selected_features = Some(select_hvg_streaming(
                &data_vec,
                (args.max_features > 0).then_some(args.max_features),
                args.feature_list_file,
                args.block_size,
            )?);
        }

        let proj_out = if let Some(sel) = selected_features.as_ref() {
            let weights = sel.row_weights(data_vec.num_rows());
            data_vec.project_columns_weighted(
                args.proj_dim,
                args.block_size,
                Some(&batch_membership),
                &weights,
            )?
        } else {
            data_vec.project_columns_with_batch_correction(
                args.proj_dim,
                args.block_size,
                Some(&batch_membership),
            )?
        };

        proj_out.proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    Ok(ProjectedData {
        data_vec,
        batch_membership,
        proj_kn,
        selected_features,
    })
}

pub struct LoadCollapseArgs<'a> {
    pub data_files: &'a [Box<str>],
    pub batch_files: &'a Option<Vec<Box<str>>>,
    pub preload: bool,
    pub warm_start_proj_file: Option<&'a str>,
    pub proj_dim: usize,
    pub sort_dim: usize,
    pub knn_cells: usize,
    pub num_levels: usize,
    pub iter_opt: usize,
    pub block_size: Option<usize>,
    pub out: &'a str,
    /// Keep top N HVGs (via binned residual variance) for the random
    /// projection; 0 disables. Collapsing still reads all genes.
    pub max_features: usize,
    /// Optional pre-computed feature list (overrides `max_features`).
    pub feature_list_file: Option<&'a str>,
    /// Opt-in BBKNN + Poisson DC-SBM refinement of the multilevel
    /// partition. `None` keeps the legacy hash-only behavior.
    pub refine: Option<data_beans_alg::refine_multilevel::RefineParams>,
    /// Treat all cells as a single batch — no per-batch δ estimation.
    pub ignore_batch: bool,
}

/// Load sparse data, project, multi-level collapse, and write delta output.
///
/// Shared pipeline for topic and indexed-topic models. The pre-collapse
/// portion (read + batch + HVG + project) is delegated to
/// [`load_and_project`] so `senna svd` can share the same code.
pub fn load_and_collapse(args: &LoadCollapseArgs) -> anyhow::Result<PreparedData> {
    let ProjectedData {
        mut data_vec,
        batch_membership,
        proj_kn,
        selected_features: _,
    } = load_and_project(&LoadProjectArgs {
        data_files: args.data_files,
        batch_files: args.batch_files,
        preload: args.preload,
        warm_start_proj_file: args.warm_start_proj_file,
        proj_dim: args.proj_dim,
        block_size: args.block_size,
        max_features: args.max_features,
        feature_list_file: args.feature_list_file,
        ignore_batch: args.ignore_batch,
    })?;

    info!("Multi-level collapsing with super-cells ...");
    let mut collapsed_levels: Vec<CollapsedOut> = data_vec.collapse_columns_multilevel_vec(
        &proj_kn,
        &batch_membership,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
            refine: args.refine.clone(),
        },
    )?;
    collapsed_levels.reverse();

    // 4. Write delta output from finest level
    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();
    if let Some(batch_db) = finest_collapsed.delta.as_ref() {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_melted_parquet(
            &outfile,
            (Some(&gene_names), Some("gene")),
            (batch_names.as_deref(), Some("batch")),
        )?;
    }

    Ok(PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
    })
}

/// Create a candle compute device from the CLI device enum.
pub(crate) fn create_device(
    device: &ComputeDevice,
    device_no: usize,
) -> candle_core::Result<Device> {
    match device {
        ComputeDevice::Metal => Device::new_metal(device_no),
        ComputeDevice::Cuda => Device::new_cuda(device_no),
        ComputeDevice::Cpu => Ok(Device::Cpu),
    }
}

/// Replace every non-CPU Var in the `VarMap` with a CPU copy.
///
/// After this call, a fresh encoder/decoder built from the `VarMap`
/// will operate on CPU. The old model structs still hold Metal/CUDA
/// Vars and must NOT be reused — rebuild them from the updated `VarMap`.
pub(crate) fn move_varmap_to_cpu(parameters: &candle_nn::VarMap) -> anyhow::Result<()> {
    use candle_core::Var;
    let mut data = parameters.data().lock().expect("VarMap lock");
    for (_name, var) in data.iter_mut() {
        if !var.device().is_cpu() {
            let cpu_tensor = var.to_device(&Device::Cpu)?;
            *var = Var::from_tensor(&cpu_tensor)?;
        }
    }
    Ok(())
}

/// Set up a graceful stop flag for SIGINT/SIGTERM.
/// First Ctrl+C sets the flag for graceful exit after the current
/// minibatch. Second Ctrl+C forces an immediate abort.
pub(crate) fn setup_stop_handler() -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            if stop.load(Ordering::SeqCst) {
                eprintln!("\nSecond interrupt — aborting immediately");
                std::process::exit(1);
            }
            // Use eprintln, not info! — log macros hold internal locks
            // and will deadlock if the main thread is mid-log.
            eprintln!("\nInterrupt received — stopping after current minibatch (Ctrl+C again to force)...");
            stop.store(true, Ordering::SeqCst);
        })
        .expect("failed to set signal handler");
    }
    stop
}
