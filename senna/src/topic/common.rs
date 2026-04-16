use crate::embed_common::*;
use crate::logging::new_progress_bar;
use crate::senna_input::*;

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
    let jobs = create_jobs(ntot, Some(block_size));
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

/// Sample collapsed data for one jitter interval.
///
/// Returns (mixed_nd, batch_nd, target_nd) — all at full D.
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
pub(crate) struct PreparedData {
    pub data_vec: SparseIoVec,
    pub collapsed_levels: Vec<CollapsedOut>,
    /// Per-gene mask for anchor selection: true = informative, false = outlier.
    pub gene_filter_mask: Vec<bool>,
}

pub(crate) struct LoadCollapseArgs<'a> {
    pub data_files: &'a [Box<str>],
    pub batch_files: &'a Option<Vec<Box<str>>>,
    pub preload: bool,
    pub warm_start_proj_file: Option<&'a str>,
    pub proj_dim: usize,
    pub sort_dim: usize,
    pub knn_cells: usize,
    pub num_levels: usize,
    pub iter_opt: usize,
    pub block_size: usize,
    pub out: &'a str,
}

/// Load sparse data, project, multi-level collapse, and write delta output.
///
/// Shared pipeline for topic and indexed-topic models.
pub(crate) fn load_and_collapse(args: &LoadCollapseArgs) -> anyhow::Result<PreparedData> {
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
        ..
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.to_vec(),
        batch_files: args.batch_files.clone(),
        preload: args.preload,
    })?;

    // 1.5 Filter hyper-variable outlier genes (Ig/TCR) for anchor selection.
    // Compute per-gene CV from sparse cell data, k-means into 2-3 clusters,
    // keep the majority. The mask is stored and passed to anchor_prior only;
    // the full gene set flows through SVD, collapsing, and training.
    let gene_filter_mask = {
        let row_stat = data_beans::qc::collect_row_stat_across_vec(&data_vec, args.block_size)?;
        let (_, _, mu, sd) = row_stat.to_vecs();
        let cv: Vec<f32> = mu
            .iter()
            .zip(sd.iter())
            .map(|(&m, &s)| if m.abs() > 1e-8 { s / m.abs() } else { 0.0 })
            .collect();
        super::anchor_prior::kmeans_cv_filter(&cv)
    };

    // 2. Take projection results by warm start or projecting it again
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file {
        use matrix_util::common_io::*;
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
        let proj_out = data_vec.project_columns_with_batch_correction(
            args.proj_dim,
            Some(args.block_size),
            Some(&batch_membership),
        )?;

        proj_out.proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    // 3. Multi-level collapsing (pseudobulk)
    info!("Multi-level collapsing with super-cells ...");
    let mut collapsed_levels: Vec<CollapsedOut> = data_vec.collapse_columns_multilevel_vec(
        &proj_kn,
        &batch_membership,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: args.num_levels,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
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
        gene_filter_mask,
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
        _ => Ok(Device::Cpu),
    }
}

/// Collect trainable Vars for Adam, excluding the BGM decoder's fixed
/// `.bgm_profile` and `.pi_topic` (both frozen; π is set data-driven at init).
pub(crate) fn trainable_vars(parameters: &candle_nn::VarMap) -> Vec<candle_core::Var> {
    let data = parameters.data().lock().expect("VarMap lock");
    data.iter()
        .filter_map(|(name, var)| {
            if name.ends_with(".bgm_profile") {
                None
            } else {
                Some(var.clone())
            }
        })
        .collect()
}

/// Move all parameters in a VarMap to CPU.
///
/// This enables multi-threaded rayon inference in `process_blocks()`.
///
/// `Var::set` requires the source tensor to live on the same device as
/// the existing Var (it copies into the Var's storage), so for non-CPU
/// Vars we have to drop the old Var and insert a freshly-constructed
/// CPU Var under the same name.
pub(crate) fn move_varmap_to_cpu(parameters: &candle_nn::VarMap) -> anyhow::Result<()> {
    let mut data = parameters.data().lock().expect("VarMap lock");
    let names_to_move: Vec<String> = data
        .iter()
        .filter(|(_, v)| !v.device().is_cpu())
        .map(|(k, _)| k.clone())
        .collect();
    for name in names_to_move {
        let old = data.remove(&name).expect("var present");
        let cpu_tensor = old.as_tensor().to_device(&Device::Cpu)?;
        let cpu_var = candle_core::Var::from_tensor(&cpu_tensor)?;
        data.insert(name, cpu_var);
    }
    Ok(())
}

/// Set up a graceful stop flag for SIGINT/SIGTERM.
pub(crate) fn setup_stop_handler() -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            info!("Interrupt received — stopping training early and saving results...");
            stop.store(true, Ordering::SeqCst);
        })
        .expect("failed to set signal handler");
    }
    stop
}
