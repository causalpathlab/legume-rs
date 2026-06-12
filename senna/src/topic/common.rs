use crate::embed_common::*;
use crate::hvg::{select_hvg_streaming, HvgSelection};
use crate::logging::new_progress_bar;
use crate::senna_input::{read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch};

use candle_core::{Device, Tensor};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

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

    let prog_bar = new_progress_bar(njobs);
    let mut chunks: Vec<(usize, Mat)> = if dev.is_cpu() {
        jobs.par_iter()
            .progress_with(prog_bar.clone())
            .map(|&block| eval_block(block))
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        jobs.iter()
            .map(|&block| {
                let r = eval_block(block);
                prog_bar.inc(1);
                r
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    prog_bar.finish_and_clear();

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

/// Per-cell batch (`Batch`) or pseudobulk-group (`Residual`) id for the
/// block `lb..ub`, selecting the membership axis by `adj_method`.
pub(crate) fn block_membership(
    data_vec: &SparseIoVec,
    adj_method: &AdjMethod,
    lb: usize,
    ub: usize,
) -> anyhow::Result<Vec<usize>> {
    Ok(match adj_method {
        AdjMethod::Batch => data_vec.get_batch_membership(lb..ub),
        AdjMethod::Residual => data_vec.get_group_membership(lb..ub)?,
    })
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
    let membership = block_membership(data_vec, adj_method, lb, ub)?;
    let indices = Tensor::from_iter(membership.into_iter().map(|x| x as u32), dev)?;
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

/// Build per-level feature coarsenings via the multilevel pipeline:
/// shared KNN + bottom-up union-find init + top-down DC-Poisson refinement.
/// Used by every senna fit pipeline that exposes `--max-coarse-features`.
/// Defaults: `knn_k = 16`, `feature_weighting = None`.
pub fn coarsen_features_multilevel(
    sketch_ds: &Mat,
    level_targets: &[usize],
    dc_poisson: data_beans_alg::dc_poisson::RefineParams,
) -> anyhow::Result<Vec<FeatureCoarsening>> {
    let knn = FeatureKnnContext::from_sketch(sketch_ds, 16)?;
    let init = compute_multilevel_feature_coarsening(sketch_ds, level_targets, &knn)?;
    let params = MultilevelRefineParams {
        dc_poisson: data_beans_alg::dc_poisson::RefineParams {
            feature_weighting: data_beans_alg::dc_poisson::FeatureWeighting::None,
            ..dc_poisson
        },
    };
    let refined = refine_multilevel_feature_coarsening(sketch_ds, init, &knn, &params)?;
    Ok(refined.levels)
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

/// Per-gene mean rate `μ_d` from a `[D, n_pb]` pseudobulk posterior mean —
/// the row-mean across pseudobulks. This is the divisive gene-mean correction
/// fed to `anscombe_residual` (full-`D`; callers coarsen afterward if needed).
pub(crate) fn pseudobulk_feature_mean(mu_dp: &Mat) -> Vec<f32> {
    let n_pb = mu_dp.ncols().max(1) as f32;
    (0..mu_dp.nrows())
        .map(|d| mu_dp.row(d).iter().sum::<f32>() / n_pb)
        .collect()
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
    /// Per-level cell → pb membership, finest-last (parallel to
    /// `collapsed_levels`). `Some` when `LoadCollapseArgs.want_hierarchy`
    /// was set or when a `prebuilt_partition` was supplied — needed by
    /// the writer that serializes `{out}.cell_to_pb.parquet` for
    /// downstream `--from` chains.
    pub cell_to_pb_per_level: Option<Vec<Vec<usize>>>,
    /// Near-empty output keep-mask from cell QC (post-`mask_columns`
    /// column order). `None` when no QC ran. Applied at the per-cell
    /// output writers via `Mat::select_rows`.
    pub output_keep_idx: Option<Vec<usize>>,
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
    /// is disabled.
    pub selected_features: Option<HvgSelection>,
    /// Near-empty output keep-mask from cell QC (post-`mask_columns`
    /// column order). `None` when no QC ran. Applied at the per-cell
    /// output writers via `Mat::select_rows`.
    pub output_keep_idx: Option<Vec<usize>>,
}

/// Args for [`load_and_project`] — the read + batch + HVG + project
/// portion shared by topic, masked-topic, and svd routines.
pub struct LoadProjectArgs<'a> {
    pub data_files: &'a [Box<str>],
    pub batch_files: &'a Option<Vec<Box<str>>>,
    pub preload: bool,
    pub proj_dim: usize,
    pub block_size: Option<usize>,
    pub max_features: usize,
    pub feature_list_file: Option<&'a str>,
    pub ignore_batch: bool,
    /// Optional shared cell QC. `None` = no QC (current behavior). When
    /// `Some`, applied inside `read_data_on_shared_rows` before
    /// projection: MAD outliers dropped via `mask_columns`, batch filtered
    /// in lockstep, near-empty output mask returned in `output_keep_idx`.
    pub qc: Option<data_beans::qc_lib::QcConfig>,
    /// Block size for the QC streaming stat passes (`None` = default).
    pub qc_block_size: Option<usize>,
    /// Optional per-cell QC report TSV path.
    pub qc_report_out: Option<&'a str>,
    /// Optional row-subset hook. Called once with the loaded data's row
    /// names; returns `keep[d]` per feature. Applied via
    /// `SparseIoVec::mask_rows` before projection. Use this to restrict
    /// the model to features covered by a feature network / curated list.
    pub feature_mask_fn: Option<&'a FeatureMaskFn>,
    /// Row-alignment strategy when multiple `data_files` are passed.
    /// Default Union — keep every row from any backend (single-
    /// modality cohorts unchanged because all backends share the same
    /// row set). Switch to Intersect for strict "common rows only".
    pub row_alignment: data_beans::sparse_io_vector::RowAlignment,
    /// Cell-axis alignment strategy. Default Disjoint preserves the
    /// historical concatenate-cells-with-`@<basename>` semantics.
    /// Set to Union to glue cells across files by raw barcode — the
    /// `senna masked-topic --multiome` path.
    pub column_alignment: data_beans::sparse_io_vector::ColumnAlignment,
    /// Per-name canonicalization rule applied to row names across
    /// backends. Default Exact = strict string match. Gene picks the
    /// last token after a delimiter so `ENSG000_TGFB1` and `TGFB1`
    /// resolve to the same row. Locus normalizes `chr1:1000-2000`,
    /// `1:1000-2000`, etc. LocusOverlap additionally merges overlapping
    /// intervals on the same chromosome into one cluster.
    pub feature_kind: Option<auxiliary_data::feature_names::FeatureNameKind>,
}

/// Callback that, given the loaded data's row names, returns a boolean
/// keep-mask of the same length. Used by [`LoadProjectArgs::feature_mask_fn`]
/// and [`LoadCollapseArgs::feature_mask_fn`] to physically subset rows
/// (via `SparseIoVec::mask_rows`) before projection / collapse / training.
pub type FeatureMaskFn = dyn Fn(&[Box<str>]) -> anyhow::Result<Vec<bool>>;

/// Read sparse files, resolve batch membership, optionally pick HVGs,
/// then run the random projection. Used by `load_and_collapse` and by
/// `senna svd` so the pre-collapse pipeline is identical across routines.
pub fn load_and_project(args: &LoadProjectArgs) -> anyhow::Result<ProjectedData> {
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: mut batch_membership,
        output_keep_idx,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.to_vec(),
        batch_files: args.batch_files.clone(),
        preload: args.preload,
        row_alignment: args.row_alignment,
        column_alignment: args.column_alignment,
        feature_kind: args.feature_kind.clone(),
        qc: args.qc.clone(),
        qc_block_size: args.qc_block_size,
        qc_report_out: args.qc_report_out.map(Box::<str>::from),
        per_file_feature_suffix: None,
        per_file_barcode_suffix: None,
    })?;
    if args.ignore_batch {
        info!("--ignore-batch: collapsing all cells to a single batch");
        crate::senna_input::collapse_to_single_batch(&mut batch_membership);
    }

    // Optional row-subset (e.g. restrict to features covered by a feature
    // network). Applied before projection so all downstream stages
    // (projection, collapse, training, inference) see the smaller axis.
    if let Some(mask_fn) = args.feature_mask_fn {
        let row_names = data_vec.row_names()?;
        let keep = mask_fn(&row_names)?;
        anyhow::ensure!(
            keep.len() == row_names.len(),
            "feature_mask_fn returned {} bools but data has {} rows",
            keep.len(),
            row_names.len(),
        );
        let n_keep = keep.iter().filter(|&&k| k).count();
        anyhow::ensure!(
            n_keep > 0,
            "feature_mask_fn dropped every feature — check name resolution"
        );
        data_vec.mask_rows(&keep)?;
    }

    let mut selected_features: Option<HvgSelection> = None;

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

    let proj_kn = if let Some(sel) = selected_features.as_ref() {
        let weights = sel.row_weights(data_vec.num_rows());
        data_vec
            .project_columns_weighted(
                args.proj_dim,
                args.block_size,
                Some(&batch_membership),
                &weights,
            )?
            .proj
    } else {
        data_vec
            .project_columns_with_batch_correction(
                args.proj_dim,
                args.block_size,
                Some(&batch_membership),
            )?
            .proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    Ok(ProjectedData {
        data_vec,
        batch_membership,
        proj_kn,
        selected_features,
        output_keep_idx,
    })
}

pub struct LoadCollapseArgs<'a> {
    pub data_files: &'a [Box<str>],
    pub batch_files: &'a Option<Vec<Box<str>>>,
    pub preload: bool,
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
    /// Optional shared cell QC — see [`LoadProjectArgs::qc`].
    pub qc: Option<data_beans::qc_lib::QcConfig>,
    /// Block size for the QC streaming stat passes (`None` = default).
    pub qc_block_size: Option<usize>,
    /// Optional per-cell QC report TSV path.
    pub qc_report_out: Option<&'a str>,
    /// Optional row-subset hook — see [`LoadProjectArgs::feature_mask_fn`].
    pub feature_mask_fn: Option<&'a FeatureMaskFn>,
    /// Row-alignment strategy — see [`LoadProjectArgs::row_alignment`].
    pub row_alignment: data_beans::sparse_io_vector::RowAlignment,
    /// Column-alignment strategy — see [`LoadProjectArgs::column_alignment`].
    pub column_alignment: data_beans::sparse_io_vector::ColumnAlignment,
    /// Per-name canonicalization — see [`LoadProjectArgs::feature_kind`].
    pub feature_kind: Option<auxiliary_data::feature_names::FeatureNameKind>,
    /// Retain the per-level cell → pb membership hierarchy. When `true`,
    /// `load_and_collapse` routes through
    /// [`collapse_columns_multilevel_with_hierarchy`] (which requires
    /// `refine = Some(..)`) and populates `PreparedData.cell_to_pb_per_level`.
    /// Default `false` keeps the legacy `masked-topic` behavior.
    pub want_hierarchy: bool,
    /// Optional pre-built `cell_to_pb_per_level` membership (finest-
    /// last) paired with the source's `cell_names`, inherited from a
    /// prior `senna {topic, masked-topic, ce-topic}` run via `--from`.
    /// `load_and_collapse` aligns it to `data_vec.column_names()` by
    /// name and then routes through
    /// `collapse_columns_multilevel_with_partition`, skipping the
    /// BBKNN + Poisson DC-SBM refinement step (still aggregates
    /// counts + re-fits per-PB Gamma posteriors). `num_levels` must
    /// equal `partition.len()` or `load_and_collapse` bails. When
    /// `Some`, the loader auto-sets `want_hierarchy = true`.
    pub prebuilt_partition: Option<crate::run_manifest::InheritedPartition>,
}

/// Load sparse data, project, multi-level collapse, and write delta output.
///
/// Shared pipeline for topic and masked-topic models. The pre-collapse
/// portion (read + batch + HVG + project) is delegated to
/// [`load_and_project`] so `senna svd` can share the same code.
pub fn load_and_collapse(args: &LoadCollapseArgs) -> anyhow::Result<PreparedData> {
    let ProjectedData {
        mut data_vec,
        batch_membership,
        proj_kn,
        selected_features: _,
        output_keep_idx,
    } = load_and_project(&LoadProjectArgs {
        data_files: args.data_files,
        batch_files: args.batch_files,
        preload: args.preload,
        proj_dim: args.proj_dim,
        block_size: args.block_size,
        max_features: args.max_features,
        feature_list_file: args.feature_list_file,
        ignore_batch: args.ignore_batch,
        qc: args.qc.clone(),
        qc_block_size: args.qc_block_size,
        qc_report_out: args.qc_report_out,
        feature_mask_fn: args.feature_mask_fn,
        row_alignment: args.row_alignment,
        column_alignment: args.column_alignment,
        feature_kind: args.feature_kind.clone(),
    })?;

    info!("Multi-level collapsing with pb-samples ...");
    let ml_params = MultilevelParams {
        knn_pb_samples: args.knn_cells,
        num_levels: args.num_levels,
        sort_dim: args.sort_dim,
        num_opt_iter: args.iter_opt,
        refine: args.refine.clone(),
        output_calibration: matrix_param::traits::CalibrateTarget::All,
    };

    // Both `collapse_columns_multilevel_vec` and the with-hierarchy /
    // with-partition variants return levels finest-first; `reverse()`
    // makes them finest-last. `cell_to_pb_per_level` is parallel to
    // `levels`, so it gets the same reversal to stay aligned with
    // `collapsed_levels`. When `prebuilt_partition` is supplied we
    // route through `collapse_columns_multilevel_with_partition` which
    // skips the BBKNN + Poisson DC-SBM refinement.
    let want_hierarchy = args.want_hierarchy || args.prebuilt_partition.is_some();
    let (mut collapsed_levels, cell_to_pb_per_level): (Vec<CollapsedOut>, Option<Vec<Vec<usize>>>) =
        if let Some((partition_src, cell_names_src)) = args.prebuilt_partition.clone() {
            let data_cell_names = data_vec.column_names()?;
            // Align by cell name (handles row-order differences /
            // bails on cell-set mismatch). The aligned partition is
            // returned finest-last; data-beans-alg expects finest-
            // first, so reverse before the call.
            let aligned_finest_last =
                crate::run_manifest::InheritedFromManifest::align_cell_to_pb_to_cells(
                    partition_src,
                    &cell_names_src,
                    &data_cell_names,
                )?;
            let mut partition_finest_first = aligned_finest_last;
            partition_finest_first.reverse();
            info!(
                "Inheriting cell→pb membership for {} levels (skipping BBKNN + DC-SBM refinement)",
                partition_finest_first.len()
            );
            let MultilevelCollapseOut {
                levels,
                mut cell_to_pb_per_level,
            } = data_beans_alg::collapse_data::collapse_columns_multilevel_with_partition(
                &mut data_vec,
                &proj_kn,
                &batch_membership,
                &ml_params,
                &partition_finest_first,
            )?;
            cell_to_pb_per_level.reverse();
            (levels, Some(cell_to_pb_per_level))
        } else if want_hierarchy {
            let MultilevelCollapseOut {
                levels,
                mut cell_to_pb_per_level,
            } = collapse_columns_multilevel_with_hierarchy(
                &mut data_vec,
                &proj_kn,
                &batch_membership,
                &ml_params,
            )?;
            cell_to_pb_per_level.reverse();
            (levels, Some(cell_to_pb_per_level))
        } else {
            (
                data_vec.collapse_columns_multilevel_vec(
                    &proj_kn,
                    &batch_membership,
                    &ml_params,
                )?,
                None,
            )
        };
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
        cell_to_pb_per_level,
        output_keep_idx,
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

/// Set up a graceful stop flag for SIGINT/SIGTERM. Re-exported from
/// `graph-embedding-util` so senna's topic models share the same
/// handler (and behavior — first Ctrl+C → graceful, second → abort)
/// as `senna gbe`.
pub(crate) use graph_embedding_util::setup_stop_handler;

////////////////////////////////////////////////////////////////////////
// Feature-network setup (used by masked-topic)
////////////////////////////////////////////////////////////////////////

use matrix_util::pair_graph::FeaturePairGraph;

/// QC pipeline + alias-matching options shared between the row-mask
/// callback (data-axis restriction) and the post-load graph parse
/// (encoder GCN adjacency). Same numbers used both times.
#[derive(Clone, Copy)]
pub struct FeatureNetworkOpts {
    pub prefix_match: bool,
    pub delim: Option<char>,
    pub min_shared_neighbors: usize,
    pub max_degree: usize,
    pub min_degree: usize,
}

fn apply_qc_pipeline(graph: &mut FeaturePairGraph, opts: &FeatureNetworkOpts) {
    graph.prune_by_shared_neighbors(opts.min_shared_neighbors);
    graph.cap_per_node_degree(opts.max_degree);
    graph.prune_by_min_degree(opts.min_degree);
}

/// Handle returned by [`setup_feature_network`]. Carries the optional
/// row-mask callback for feature-network restriction (present only when
/// restriction is on).
pub struct FeatureNetworkHandle {
    pub mask_fn: Option<Box<FeatureMaskFn>>,
}

/// Build the row-mask callback for feature-network restriction.
///
/// When `restrict_path` is `Some`, the callback parses the edge list
/// against the data axis, applies the QC pipeline, and emits a `keep`
/// mask of features with at least one surviving edge. When `restrict_path`
/// is `None`, the handle has no mask_fn.
pub fn setup_feature_network(
    restrict_path: Option<&str>,
    opts: FeatureNetworkOpts,
) -> FeatureNetworkHandle {
    let mask_fn: Option<Box<FeatureMaskFn>> = restrict_path.map(|p| {
        let path: String = p.to_string();
        let f: Box<FeatureMaskFn> = Box::new(move |row_names| {
            let mut graph = FeaturePairGraph::from_edge_list(
                &path,
                row_names.to_vec(),
                opts.prefix_match,
                opts.delim,
            )?;
            apply_qc_pipeline(&mut graph, &opts);
            let keep: Vec<bool> = graph.feature_degrees().iter().map(|&d| d > 0).collect();
            let n_keep = keep.iter().filter(|&&k| k).count();
            info!(
                "feature-network restriction: keeping {} / {} features with ≥1 edge",
                n_keep,
                row_names.len(),
            );
            Ok(keep)
        });
        f
    });
    FeatureNetworkHandle { mask_fn }
}

/// Resolve the encoder embedding dim `H` shared by the topic-family fits.
///
/// Precedence: a pre-trained ρ's column count pins `H`; an explicit
/// `--embedding-dim` from the CLI is used when no pre-trained ρ is
/// present; otherwise the default `2 × K` is used. An explicit value
/// that disagrees with the pre-trained `H` is a hard error.
///
/// Bails when `H < K` (β = softmax(α·ρᵀ) is rank ≤ H, so K independent
/// topics need H ≥ K); warns when `H < 2K` (β-rank limit, topics may
/// collapse). The warn-threshold matches the default so the warning
/// only fires when the user explicitly under-specified.
pub fn resolve_embedding_dim(
    cli_embedding_dim: usize,
    pretrained_h: Option<usize>,
    k: usize,
) -> anyhow::Result<usize> {
    let h = match (pretrained_h, cli_embedding_dim) {
        (Some(ph), 0) => {
            info!("--embedding-dim auto-set to pre-trained ρ width H = {ph}");
            ph
        }
        (Some(ph), explicit) => {
            anyhow::ensure!(
                explicit == ph,
                "--embedding-dim ({explicit}) disagrees with the pre-trained ρ H ({ph}). \
                 Either omit --embedding-dim (it will be inferred) or pass {ph} to match."
            );
            explicit
        }
        (None, 0) => {
            let auto = 2 * k;
            info!("--embedding-dim not set; defaulting to 2 × K = {auto}");
            auto
        }
        (None, explicit) => explicit,
    };
    anyhow::ensure!(
        h >= k,
        "--embedding-dim ({h}) < --n-latent-topics ({k}). β = softmax(α·ρᵀ) is rank ≤ H, \
         so at most {h} linearly independent topics can be represented — pass \
         --embedding-dim >= {k} (recommended {} for headroom), or omit it for the 2K default.",
        k * 2,
    );
    if h < 2 * k {
        log::warn!(
            "--embedding-dim ({h}) is at the β-rank limit for --n-latent-topics ({k}); \
             topics may collapse during training. Recommend --embedding-dim >= {} for headroom.",
            k * 2,
        );
    }
    Ok(h)
}
