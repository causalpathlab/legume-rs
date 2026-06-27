#![allow(dead_code)]

use data_beans::sparse_data_visitors::*;
use data_beans::sparse_io_stack::SparseIoStack;
use data_beans::sparse_io_vector::SparseIoVec;
use log::{info, warn};
use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::Inference;
use matrix_param::traits::*;
use matrix_util::knn_match::ColumnDict;
use matrix_util::traits::*;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::ops::AddAssign;
use std::sync::{Arc, Mutex};

use crate::random_projection::binary_sort_columns;

use rustc_hash::FxHashMap as HashMap;
type CscMat = nalgebra_sparse::CscMatrix<f32>;

/// Sparse pb-sample gene profile: `rows[pbsamp] = Vec<(gene_idx, sum)>` sorted
/// by `gene_idx`, as produced by `collect_pb_sample_gene_sums` and
/// consumed by `collect_matched_stat_coarse` and the refinement pass.
pub type GeneSums = Vec<Vec<(usize, f32)>>;

pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

/// Per-level collapse output plus the partition hierarchy across levels.
///
/// Returned by [`collapse_columns_multilevel_with_hierarchy`] for
/// consumers (e.g. `graph-embedding-util`'s nested chain sampler) that
/// need parent/child maps between pb-samples at adjacent levels.
/// `cell_to_pb_per_level` is finest-first, parallel to `levels`:
/// `cell_to_pb_per_level[k][c] = pb_id at level k for cell c`.
mod pb_samples;
use pb_samples::{
    build_pb_sample_layout, build_pb_sample_to_cells, build_pb_samples,
    collect_pb_sample_gene_sums, per_batch_sc_neighbors,
};
pub use pb_samples::{PbSampleCollection, PbSampleLayout};
// Shared cross-batch pb-sample matching, reused by `refine_multilevel`.
pub(crate) use pb_samples::bbknn_match_one_pbsamp;
mod refine;
use refine::{
    compute_fine_to_coarse_mapping, compute_level_sort_dims, fine_to_coarse_from_refined,
    pad_numeric_labels, refine_and_collect_single_layer, refine_and_collect_stack,
    RefineCollectCtx,
};
mod stats;
use stats::{
    collect_basic_stat_visitor, collect_batch_stat_visitor, collect_matched_stat_coarse,
    collect_matched_stat_visitor, merge_stat, optimize, KnnParams, DEFAULT_NUM_LEVELS,
};
pub use stats::{resample_and_optimize, CollapsedOut, CollapsedStat};

pub struct MultilevelCollapseOut {
    pub levels: Vec<CollapsedOut>,
    pub cell_to_pb_per_level: Vec<Vec<usize>>,
}

/// Configuration for multi-level collapsing.
pub struct MultilevelParams {
    pub knn_pb_samples: usize,
    pub num_levels: usize,
    pub sort_dim: usize,
    pub num_opt_iter: usize,
    /// Opt-in BBKNN + Poisson DC-SBM refinement on top of the hash
    /// partition. `None` preserves legacy behavior.
    pub refine: Option<crate::refine_multilevel::RefineParams>,
    /// Which posterior planes the *output* `CollapsedOut` should carry.
    /// `MeanOnly` skips the sd / log_mean / log_sd allocations entirely —
    /// a big memory win for consumers that only read `posterior_mean()`
    /// (e.g. bge). Use `All` when the caller exports log-scale dictionaries.
    pub output_calibration: matrix_param::traits::CalibrateTarget,
}

impl MultilevelParams {
    pub fn new(proj_dim: usize) -> Self {
        Self {
            knn_pb_samples: DEFAULT_KNN,
            num_levels: DEFAULT_NUM_LEVELS,
            sort_dim: proj_dim.min(12),
            num_opt_iter: DEFAULT_OPT_ITER,
            refine: Some(crate::refine_multilevel::RefineParams::default()),
            output_calibration: matrix_param::traits::CalibrateTarget::All,
        }
    }
}

pub struct EmptyArg {}

#[cfg(debug_assertions)]
use log::debug;

/// Given a feature/projection matrix (factor x cells), we assign each
/// cell to a sample and return pseudobulk (collapsed) matrices
///
/// (1) Register batches if needed (2) collapse columns/cells into samples
///
pub trait CollapsingOps {
    ///
    /// Collapse columns/cells into samples as allocated by
    /// `assign_columns_to_samples`
    ///
    /// # Arguments
    /// * `cells_per_group` - number of cells per sample (None: no down sampling)
    /// * `knn_batches` - number of nearest neighbour batches
    /// * `knn_cells` - number of nearest neighbors for building HNSW (default: 10)
    /// * `reference` - reference batch for counterfactual inference
    /// * `num_opt_iter` - number of optimization iterations (default: 100)
    ///
    fn collapse_columns(
        &self,
        knn_batches: Option<usize>,
        knn_cells: Option<usize>,
        reference_batch_names: Option<&[Box<str>]>,
        num_opt_iter: Option<usize>,
    ) -> anyhow::Result<CollapsedOut>;

    /// Register batch information and build a `HnswMap` object for
    /// each batch for fast nearest neighbor search within each batch
    /// and store them in the `SparseIoVec`
    ///
    /// # Arguments
    /// * `proj_kn` - random projection matrix
    /// * `col_to_batch` - map: cell -> batch
    fn build_hnsw_per_batch<T>(
        &mut self,
        proj_kn: &nalgebra::DMatrix<f32>,
        col_to_batch: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    fn collect_basic_stat(&self, stat: &mut CollapsedStat) -> anyhow::Result<()>;

    fn collect_batch_stat(&self, stat: &mut CollapsedStat) -> anyhow::Result<()>;

    fn collect_matched_stat(
        &self,
        knn_batches: usize,
        knn_cols: usize,
        reference_indices: Option<&[usize]>,
        stat: &mut CollapsedStat,
    ) -> anyhow::Result<()>;
}

impl CollapsingOps for SparseIoVec {
    fn build_hnsw_per_batch<T>(
        &mut self,
        proj_kn: &nalgebra::DMatrix<f32>,
        col_to_batch: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        info!("creating batch-specific HNSW maps ...");
        self.register_batches_dmatrix(proj_kn, col_to_batch)?;

        info!(
            "found {} columns across {} batches",
            self.num_columns(),
            self.num_batches()
        );

        Ok(())
    }

    fn collapse_columns(
        &self,
        knn_batches: Option<usize>,
        knn_cells: Option<usize>,
        reference_batch_names: Option<&[Box<str>]>,
        num_opt_iter: Option<usize>,
    ) -> anyhow::Result<CollapsedOut> {
        let group_to_cols = self.take_grouped_columns().ok_or(anyhow::anyhow!(
            "The columns were not assigned before. Call `assign_columns_to_groups`"
        ))?;

        let num_features = self.num_rows();
        let num_groups = group_to_cols.len();
        let num_batches = self.num_batches();

        let mut stat = CollapsedStat::new(num_features, num_groups, num_batches);
        info!("basic statistics across {} groups", num_groups);
        self.collect_basic_stat(&mut stat)?;

        if num_batches > 1 {
            info!(
                "batch-specific statistics across {} batches over {} samples",
                num_batches, num_groups
            );

            let batch_name_map = self
                .batch_name_map()
                .ok_or(anyhow::anyhow!("unable to read batch names"))?;

            let reference_indices = reference_batch_names.map(|x| {
                x.iter()
                    .filter_map(|b| batch_name_map.get(b))
                    .copied()
                    .collect::<Vec<_>>()
            });

            if let Some(r) = reference_indices.as_ref() {
                if r.is_empty() {
                    let ref_names = reference_batch_names
                        .unwrap()
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(",");

                    let bat_names = self
                        .batch_names()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(",");

                    warn!("{} vs. {}", ref_names, bat_names);

                    return Err(anyhow::anyhow!("no reference batch names matched!"));
                }
            }

            self.collect_batch_stat(&mut stat)?;

            info!(
                "counterfactual inference across {} batches over {} samples",
                num_batches, num_groups,
            );

            let knn_batches = knn_batches.unwrap_or(2);
            let knn_cells = knn_cells.unwrap_or(DEFAULT_KNN);

            self.collect_matched_stat(
                knn_batches,
                knn_cells,
                reference_indices.as_deref(),
                &mut stat,
            )?;
        } // if num_batches > 1

        /////////////////////////////
        // Resolve mean parameters //
        /////////////////////////////

        info!("optimizing the collapsed parameters...");
        let (a0, b0) = (1_f32, 1_f32);
        optimize(
            &stat,
            (a0, b0),
            num_opt_iter.unwrap_or(DEFAULT_OPT_ITER),
            "Optimizing",
            CalibrateTarget::All,
        )
    }

    fn collect_basic_stat(&self, stat: &mut CollapsedStat) -> anyhow::Result<()> {
        self.visit_columns_by_group(&collect_basic_stat_visitor, &EmptyArg {}, stat)
    }

    fn collect_batch_stat(&self, stat: &mut CollapsedStat) -> anyhow::Result<()> {
        self.visit_columns_by_group(&collect_batch_stat_visitor, &EmptyArg {}, stat)
    }

    fn collect_matched_stat(
        &self,
        knn_batches: usize,
        knn_cells: usize,
        reference_indices: Option<&[usize]>,
        stat: &mut CollapsedStat,
    ) -> anyhow::Result<()> {
        self.visit_columns_by_group(
            &collect_matched_stat_visitor,
            &KnnParams {
                knn_batches,
                knn_cells,
                reference_indices,
            },
            stat,
        )
    }
}

pub trait MultilevelCollapsingOps {
    type LevelOutput;

    fn collapse_columns_multilevel<T>(
        &mut self,
        proj_kn: &DMatrix<f32>,
        batch_membership: &[T],
        params: &MultilevelParams,
    ) -> anyhow::Result<Self::LevelOutput>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    fn collapse_columns_multilevel_vec<T>(
        &mut self,
        proj_kn: &DMatrix<f32>,
        batch_membership: &[T],
        params: &MultilevelParams,
    ) -> anyhow::Result<Vec<Self::LevelOutput>>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;
}

/// Same as `SparseIoVec::collapse_columns_multilevel_vec`, but also returns
/// the per-level cell → pb mapping needed for hierarchical / nested
/// chain sampling in downstream consumers (currently
/// `graph-embedding-util`). Requires `MultilevelParams.refine` to be
/// `Some(..)` — the legacy non-refinement path doesn't currently
/// surface the per-level partition structure. Rather than silently
/// returning an empty hierarchy, we error so callers can't be misled
/// into walking an empty tree.
pub fn collapse_columns_multilevel_with_hierarchy<T>(
    data_vec: &mut SparseIoVec,
    proj_kn: &DMatrix<f32>,
    batch_membership: &[T],
    params: &MultilevelParams,
) -> anyhow::Result<MultilevelCollapseOut>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    let sort_dim = params.sort_dim;
    let knn = params.knn_pb_samples;
    let opt_iter = params.num_opt_iter;

    data_vec.register_batch_membership(batch_membership);
    let num_features = data_vec.num_rows();
    let num_batches = data_vec.num_batches();
    if num_batches >= 2 {
        data_vec.build_hnsw_per_batch(proj_kn, batch_membership)?;
    }

    let level_dims = compute_level_sort_dims(sort_dim, params.num_levels);
    let finest_dim = level_dims[0];
    let nn = proj_kn.ncols();
    let kk = proj_kn.nrows().min(finest_dim).min(nn);
    let fine_codes = binary_sort_columns(proj_kn, kk)?;
    data_vec.assign_groups(&fine_codes, None);

    let group_to_cols = data_vec
        .take_grouped_columns()
        .ok_or_else(|| anyhow::anyhow!("columns not assigned"))?
        .clone();

    let refine_params = params.refine.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "collapse_columns_multilevel_with_hierarchy requires \
             MultilevelParams.refine = Some(..); the legacy non-refinement path \
             doesn't surface per-level cell→pb mappings"
        )
    })?;

    let ctx = RefineCollectCtx {
        fine_codes: &fine_codes,
        group_to_cols_finest: &group_to_cols,
        level_dims: &level_dims,
        num_features,
        num_batches,
        knn,
        opt_iter,
        refine_params,
        output_calibration: params.output_calibration,
    };
    refine_and_collect_single_layer(data_vec, proj_kn, &ctx)
}

/// Variant of [`collapse_columns_multilevel_with_hierarchy`] that **skips
/// the BBKNN + Poisson DC-SBM refinement**, instead synthesising the
/// per-pb-sample group assignment from a caller-supplied
/// `cell_to_pb_per_level` (finest-first; the same shape returned in
/// [`MultilevelCollapseOut.cell_to_pb_per_level`] by the refining
/// entry point). Typical use: `senna {topic, itopic, ce-topic} --from`
/// inheriting a prior run's partition.
///
/// Each level's pb-sample → group label is decided by majority vote
/// across the cells in that pb-sample. The downstream `optimize` step
/// still runs per level, so the returned `CollapsedOut` posteriors
/// reflect this run's batch model and priors — only the expensive
/// clustering work is bypassed.
pub fn collapse_columns_multilevel_with_partition<T>(
    data_vec: &mut SparseIoVec,
    proj_kn: &DMatrix<f32>,
    batch_membership: &[T],
    params: &MultilevelParams,
    cell_to_pb_per_level: &[Vec<usize>],
) -> anyhow::Result<MultilevelCollapseOut>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    let knn = params.knn_pb_samples;
    let opt_iter = params.num_opt_iter;

    data_vec.register_batch_membership(batch_membership);
    let num_features = data_vec.num_rows();
    let num_batches = data_vec.num_batches();
    if num_batches >= 2 {
        data_vec.build_hnsw_per_batch(proj_kn, batch_membership)?;
    }

    let level_dims = compute_level_sort_dims(params.sort_dim, params.num_levels);
    anyhow::ensure!(
        cell_to_pb_per_level.len() == level_dims.len(),
        "inherited cell_to_pb has {} levels but --num-levels is {}; \
         pass --num-levels to match the source run",
        cell_to_pb_per_level.len(),
        level_dims.len(),
    );
    let finest_dim = level_dims[0];
    let nn = proj_kn.ncols();
    let kk = proj_kn.nrows().min(finest_dim).min(nn);
    let fine_codes = binary_sort_columns(proj_kn, kk)?;
    data_vec.assign_groups(&fine_codes, None);
    let group_to_cols = data_vec
        .take_grouped_columns()
        .ok_or_else(|| anyhow::anyhow!("columns not assigned"))?
        .clone();

    // pb-samples are still built locally — they're needed for the
    // cross-batch matched-stat path on multi-batch data. Refinement is
    // what we skip; pb-sample construction is cheap.
    let pb_samples = build_pb_samples(data_vec, proj_kn, num_features)?;
    let num_pb = pb_samples.layout.cell_counts.len();
    let ncols = proj_kn.ncols();
    let col_to_batch: Vec<usize> = data_vec.get_batch_membership(0..ncols);
    let pb_sample_to_cells =
        build_pb_sample_to_cells(&pb_samples.layout, &group_to_cols, &col_to_batch);

    // Synthesize a RefinedAssignment from the inherited cell→pb
    // membership via per-pb-sample modal vote at each level. Modal vote
    // lets us tolerate small misalignments between this run's
    // hash-partition pb-samples and the source's; in practice pb-samps
    // are tiny so most have a unanimous inherited label.
    let num_levels = level_dims.len();
    let mut pbsamp_to_group: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    let mut num_groups_per_level: Vec<usize> = Vec::with_capacity(num_levels);
    for (lvl_idx, lvl) in cell_to_pb_per_level.iter().enumerate() {
        anyhow::ensure!(
            lvl.len() == ncols,
            "inherited cell_to_pb level {} has {} cells, data has {}",
            lvl_idx,
            lvl.len(),
            ncols
        );
        let mut p2g: Vec<usize> = Vec::with_capacity(num_pb);
        for cells in &pb_sample_to_cells {
            p2g.push(modal_group(cells, lvl));
        }
        let (compact, k) = crate::refine_multilevel::compact_labels(&p2g);
        num_groups_per_level.push(k);
        pbsamp_to_group.push(compact);
    }
    let refined = crate::refine_multilevel::RefinedAssignment {
        pbsamp_to_group,
        num_groups_per_level,
    };

    info!(
        "Inherited partition: {} cells, {} pb-samples, finest k={} (skipped BBKNN + DC-SBM refinement)",
        ncols, num_pb, refined.num_groups_per_level[0]
    );

    // From here the path matches refine_and_collect_single_layer's
    // post-refinement tail: assign groups, collect stats, fit Gamma
    // posteriors level-by-level via merge_stat.
    let k_finest = refined.num_groups_per_level[0];
    let mut cell_to_group_finest = vec![0usize; ncols];
    for (pbsamp, cells) in pb_sample_to_cells.iter().enumerate() {
        let g = refined.pbsamp_to_group[0][pbsamp];
        for &c in cells {
            cell_to_group_finest[c] = g;
        }
    }
    let finest_str = pad_numeric_labels(&cell_to_group_finest, k_finest);
    data_vec.assign_groups(&finest_str, None);
    debug_assert_eq!(data_vec.num_groups(), k_finest);

    let mut fine_stat = CollapsedStat::new(num_features, k_finest, num_batches);
    info!("Collecting basic stats over {} groups ...", k_finest);
    data_vec.collect_basic_stat(&mut fine_stat)?;
    if num_batches >= 2 {
        info!(
            "Collecting per-batch stats over {} groups × {} batches ...",
            k_finest, num_batches
        );
        data_vec.collect_batch_stat(&mut fine_stat)?;
        let batch_knn = data_vec
            .batch_knn_lookup()
            .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
        info!(
            "Collecting cross-batch matched stats (knn={}) over {} pb-samples ...",
            knn, num_pb
        );
        collect_matched_stat_coarse(
            &pb_samples.layout,
            &pb_samples.gene_sums,
            &refined.pbsamp_to_group[0],
            batch_knn.as_slice(),
            knn,
            &mut fine_stat,
        )?;
    }

    let mut results: Vec<CollapsedOut> = Vec::with_capacity(num_levels);
    info!(
        "Level 1/{}: inherited k={} (finest; {} cells)",
        num_levels, k_finest, ncols
    );
    let finest_out = optimize(
        &fine_stat,
        (1.0, 1.0),
        opt_iter,
        &format!("Inherit L1/{}", num_levels),
        CalibrateTarget::All,
    )?;
    results.push(finest_out);

    let mut prev_stat = fine_stat;
    for level in 1..num_levels {
        let k_prev = refined.num_groups_per_level[level - 1];
        let k_level = refined.num_groups_per_level[level];
        let fine_to_coarse = fine_to_coarse_from_refined(
            &refined.pbsamp_to_group[level - 1],
            &refined.pbsamp_to_group[level],
            k_prev,
        );
        let coarse_stat = merge_stat(&prev_stat, &fine_to_coarse, k_level);
        info!(
            "Level {}/{}: inherited k={} (merged from {})",
            level + 1,
            num_levels,
            k_level,
            k_prev
        );
        let level_opt_iter = (opt_iter / 2).max(10);
        let out = optimize(
            &coarse_stat,
            (1.0, 1.0),
            level_opt_iter,
            &format!("Inherit L{}/{}", level + 1, num_levels),
            CalibrateTarget::All,
        )?;
        results.push(out);
        prev_stat = coarse_stat;
    }

    // Re-derive cell_to_pb_per_level from the post-modal-vote groups so
    // the returned struct is self-consistent — small drift vs the
    // inherited values is expected when pb-samples spanned multiple
    // source groups (handled by majority).
    let mut cell_to_pb_per_level_out: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    for level in 0..num_levels {
        let mut c2g = vec![0usize; ncols];
        for (pbsamp, cells) in pb_sample_to_cells.iter().enumerate() {
            let g = refined.pbsamp_to_group[level][pbsamp];
            for &c in cells {
                c2g[c] = g;
            }
        }
        cell_to_pb_per_level_out.push(c2g);
    }

    Ok(MultilevelCollapseOut {
        levels: results,
        cell_to_pb_per_level: cell_to_pb_per_level_out,
    })
}

/// Modal `lvl[c]` over `c ∈ cells`. Returns 0 when `cells` is empty.
/// Fast small-Vec path for the common case where pb-samples contain
/// just a handful of cells.
fn modal_group(cells: &[usize], lvl: &[usize]) -> usize {
    match cells {
        [] => 0,
        [c] => lvl[*c],
        _ => {
            use rustc_hash::FxHashMap;
            let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
            for &c in cells {
                *counts.entry(lvl[c]).or_insert(0) += 1;
            }
            counts
                .into_iter()
                .max_by_key(|&(_, n)| n)
                .map(|(g, _)| g)
                .unwrap_or(0)
        }
    }
}

impl MultilevelCollapsingOps for SparseIoVec {
    type LevelOutput = CollapsedOut;

    fn collapse_columns_multilevel<T>(
        &mut self,
        proj_kn: &DMatrix<f32>,
        batch_membership: &[T],
        params: &MultilevelParams,
    ) -> anyhow::Result<CollapsedOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let mut results =
            self.collapse_columns_multilevel_vec(proj_kn, batch_membership, params)?;
        if results.is_empty() {
            return Err(anyhow::anyhow!("no levels processed"));
        }
        Ok(results.remove(0))
    }

    fn collapse_columns_multilevel_vec<T>(
        &mut self,
        proj_kn: &DMatrix<f32>,
        batch_membership: &[T],
        params: &MultilevelParams,
    ) -> anyhow::Result<Vec<CollapsedOut>>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let sort_dim = params.sort_dim;
        let knn = params.knn_pb_samples;
        let opt_iter = params.num_opt_iter;

        self.register_batch_membership(batch_membership);
        let num_features = self.num_rows();
        let num_batches = self.num_batches();
        if num_batches >= 2 {
            self.build_hnsw_per_batch(proj_kn, batch_membership)?;
        }

        // Level dims: [finest, ..., coarsest]
        let level_dims = compute_level_sort_dims(sort_dim, params.num_levels);

        info!(
            "Multi-level collapsing (fine→coarse): {} levels, sort_dims={:?}, {} batches",
            level_dims.len(),
            level_dims,
            num_batches,
        );

        // Compute binary codes at finest resolution once
        let finest_dim = level_dims[0];
        let nn = proj_kn.ncols();
        let kk = proj_kn.nrows().min(finest_dim).min(nn);
        let fine_codes = binary_sort_columns(proj_kn, kk)?;

        // Partition at finest level
        self.assign_groups(&fine_codes, None);

        let group_to_cols = self
            .take_grouped_columns()
            .ok_or(anyhow::anyhow!("columns not assigned"))?
            .clone();
        let num_groups = group_to_cols.len();

        //////////////////////////////////////////////////////////////////
        // Opt-in refinement path: BBKNN + Poisson DC-SBM over pb-samples
        //////////////////////////////////////////////////////////////////

        if let Some(refine_params) = params.refine.as_ref() {
            let ctx = RefineCollectCtx {
                fine_codes: &fine_codes,
                group_to_cols_finest: &group_to_cols,
                level_dims: &level_dims,
                num_features,
                num_batches,
                knn,
                opt_iter,
                refine_params,
                output_calibration: params.output_calibration,
            };
            return refine_and_collect_single_layer(self, proj_kn, &ctx).map(|out| out.levels);
        }

        // Collect statistics at finest level
        let mut fine_stat = CollapsedStat::new(num_features, num_groups, num_batches);

        info!(
            "Level 1/{}: sort_dim={}, {} groups (finest)",
            level_dims.len(),
            finest_dim,
            num_groups
        );
        self.collect_basic_stat(&mut fine_stat)?;

        // Batch correction: pb-sample matching across batches
        if num_batches >= 2 {
            self.collect_batch_stat(&mut fine_stat)?;

            info!("Building pb-samples ...");
            let pb_samples = build_pb_samples(self, proj_kn, num_features)?;
            info!(
                "Built {} pb-samples, matching with knn={} ...",
                pb_samples.layout.cell_counts.len(),
                knn
            );
            let batch_knn = self
                .batch_knn_lookup()
                .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
            collect_matched_stat_coarse(
                &pb_samples.layout,
                &pb_samples.gene_sums,
                &pb_samples.layout.pb_sample_to_group,
                batch_knn.as_slice(),
                knn,
                &mut fine_stat,
            )?;
        }

        // Optimize finest level
        let result = optimize(
            &fine_stat,
            (1.0, 1.0),
            opt_iter,
            &format!("Coarsen L1/{}", level_dims.len()),
            CalibrateTarget::All,
        )?;
        let mut results = vec![result];

        // Agglomeratively merge for coarser levels
        let mut prev_stat = fine_stat;
        let mut prev_group_to_cols = group_to_cols.clone();

        for (level, &level_sort_dim) in level_dims.iter().enumerate().skip(1) {
            let level_opt_iter = (opt_iter / 2).max(10);

            let (fine_to_coarse, num_coarse) =
                compute_fine_to_coarse_mapping(&prev_group_to_cols, &fine_codes, level_sort_dim);

            info!(
                "Level {}/{}: sort_dim={}, {} groups (merged from {})",
                level + 1,
                level_dims.len(),
                level_sort_dim,
                num_coarse,
                prev_stat.num_samples(),
            );

            let coarse_stat = merge_stat(&prev_stat, &fine_to_coarse, num_coarse);

            let coarse_result = optimize(
                &coarse_stat,
                (1.0, 1.0),
                level_opt_iter,
                &format!("Coarsen L{}/{}", level + 1, level_dims.len()),
                CalibrateTarget::All,
            )?;
            results.push(coarse_result);

            let mut coarse_group_to_cols = vec![vec![]; num_coarse];
            for (fine_g, &coarse_g) in fine_to_coarse.iter().enumerate() {
                coarse_group_to_cols[coarse_g].extend_from_slice(&prev_group_to_cols[fine_g]);
            }

            prev_stat = coarse_stat;
            prev_group_to_cols = coarse_group_to_cols;
        }

        Ok(results)
    }
}

impl MultilevelCollapsingOps for SparseIoStack {
    type LevelOutput = Vec<CollapsedOut>;

    fn collapse_columns_multilevel<T>(
        &mut self,
        proj_kn: &DMatrix<f32>,
        batch_membership: &[T],
        params: &MultilevelParams,
    ) -> anyhow::Result<Vec<CollapsedOut>>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let mut results =
            self.collapse_columns_multilevel_vec(proj_kn, batch_membership, params)?;
        if results.is_empty() {
            return Err(anyhow::anyhow!("no levels processed"));
        }
        Ok(results.remove(0))
    }

    fn collapse_columns_multilevel_vec<T>(
        &mut self,
        proj_kn: &DMatrix<f32>,
        batch_membership: &[T],
        params: &MultilevelParams,
    ) -> anyhow::Result<Vec<Vec<CollapsedOut>>>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let num_layers = self.num_types();
        if num_layers == 0 {
            return Err(anyhow::anyhow!("empty SparseIoStack"));
        }

        let sort_dim = params.sort_dim;
        let knn = params.knn_pb_samples;
        let opt_iter = params.num_opt_iter;

        self.register_batch_membership(batch_membership);

        // Use first layer for num_batches (all layers share the same columns)
        let num_batches = self.stack[0].num_batches();
        if num_batches >= 2 {
            for layer in self.stack.iter_mut() {
                layer.build_hnsw_per_batch(proj_kn, batch_membership)?;
            }
        }

        // Build col_to_batch from the first layer (shared across all layers)
        let ncols = proj_kn.ncols();
        let col_to_batch: Vec<usize> = self.stack[0].get_batch_membership(0..ncols);

        // Opt-in refinement path for the stack.
        if let Some(refine_params) = params.refine.as_ref() {
            let level_dims = compute_level_sort_dims(sort_dim, params.num_levels);
            let finest_dim = level_dims[0];
            let kk = proj_kn.nrows().min(finest_dim).min(ncols);
            let fine_codes = binary_sort_columns(proj_kn, kk)?;
            for layer in self.stack.iter_mut() {
                layer.assign_groups(&fine_codes, None);
            }
            let group_to_cols = self.stack[0]
                .take_grouped_columns()
                .ok_or(anyhow::anyhow!("columns not assigned"))?
                .clone();
            let num_features = self.stack[0].num_rows();
            let ctx = RefineCollectCtx {
                fine_codes: &fine_codes,
                group_to_cols_finest: &group_to_cols,
                level_dims: &level_dims,
                num_features,
                num_batches,
                knn,
                opt_iter,
                refine_params,
                output_calibration: params.output_calibration,
            };
            return refine_and_collect_stack(self, proj_kn, &ctx);
        }

        if num_batches < 2 {
            // No batch effects — multi-level collapsing without batch correction
            let level_dims = compute_level_sort_dims(sort_dim, params.num_levels);

            info!(
                "Multi-level stack collapsing (no batch): {} levels, sort_dims={:?}",
                level_dims.len(),
                level_dims,
            );

            // Partition at finest level
            let finest_dim = level_dims[0];
            let kk = proj_kn.nrows().min(finest_dim).min(ncols);
            let fine_codes = binary_sort_columns(proj_kn, kk)?;

            for layer in self.stack.iter_mut() {
                layer.assign_groups(&fine_codes, None);
            }

            let group_to_cols = self.stack[0]
                .take_grouped_columns()
                .ok_or(anyhow::anyhow!("columns not assigned"))?;

            // Finest level stats
            let mut fine_stats: Vec<CollapsedStat> = Vec::with_capacity(num_layers);
            let mut layer_results = Vec::with_capacity(num_layers);
            for (d, layer) in self.stack.iter().enumerate() {
                let num_features = layer.num_rows();
                let num_groups = group_to_cols.len();
                let mut stat = CollapsedStat::new(num_features, num_groups, 0);
                layer.collect_basic_stat(&mut stat)?;
                layer_results.push(optimize(
                    &stat,
                    (1.0, 1.0),
                    opt_iter,
                    &format!(
                        "Coarsen L1/{} layer {}/{}",
                        level_dims.len(),
                        d + 1,
                        num_layers
                    ),
                    CalibrateTarget::All,
                )?);
                fine_stats.push(stat);
            }

            let mut results = vec![layer_results];

            // Agglomeratively merge for coarser levels
            let mut prev_stats = fine_stats;
            let mut prev_group_to_cols = group_to_cols.clone();

            for (level, &level_sort_dim) in level_dims.iter().enumerate().skip(1) {
                let level_opt_iter = (opt_iter / 2).max(10);
                let (fine_to_coarse, num_coarse) = compute_fine_to_coarse_mapping(
                    &prev_group_to_cols,
                    &fine_codes,
                    level_sort_dim,
                );

                let mut layer_results = Vec::with_capacity(num_layers);
                let mut coarse_stats = Vec::with_capacity(num_layers);
                for (d, prev_stat) in prev_stats.iter().enumerate() {
                    let coarse_stat = merge_stat(prev_stat, &fine_to_coarse, num_coarse);
                    layer_results.push(optimize(
                        &coarse_stat,
                        (1.0, 1.0),
                        level_opt_iter,
                        &format!(
                            "Coarsen L{}/{} layer {}/{}",
                            level + 1,
                            level_dims.len(),
                            d + 1,
                            num_layers
                        ),
                        CalibrateTarget::All,
                    )?);
                    coarse_stats.push(coarse_stat);
                }
                results.push(layer_results);

                let mut coarse_group_to_cols = vec![vec![]; num_coarse];
                for (fine_g, &coarse_g) in fine_to_coarse.iter().enumerate() {
                    coarse_group_to_cols[coarse_g].extend_from_slice(&prev_group_to_cols[fine_g]);
                }
                prev_stats = coarse_stats;
                prev_group_to_cols = coarse_group_to_cols;
            }

            return Ok(results);
        }

        // Level dims: [finest, ..., coarsest]
        let level_dims = compute_level_sort_dims(sort_dim, params.num_levels);

        info!(
            "Multi-level stack collapsing (fine→coarse): {} levels, sort_dims={:?}, {} batches, {} layers",
            level_dims.len(), level_dims, num_batches, num_layers
        );

        // Compute binary codes at finest resolution once
        let finest_dim = level_dims[0];
        let kk = proj_kn.nrows().min(finest_dim).min(ncols);
        let fine_codes = binary_sort_columns(proj_kn, kk)?;

        // Partition all layers at finest level using binary codes
        for layer in self.stack.iter_mut() {
            layer.assign_groups(&fine_codes, None);
        }

        // Get group_to_cols from first layer (all layers share the same grouping)
        let group_to_cols = self.stack[0]
            .take_grouped_columns()
            .ok_or(anyhow::anyhow!("columns not assigned"))?;
        let num_groups = group_to_cols.len();

        // Build shared pb-sample layout ONCE at finest level
        info!(
            "Level 1/{}: sort_dim={}, {} groups (finest — full computation)",
            level_dims.len(),
            finest_dim,
            num_groups
        );
        let layout = build_pb_sample_layout(group_to_cols, &col_to_batch, proj_kn)?;
        let num_pb = layout.cell_counts.len();
        info!("Built {} pb-samples, matching with knn={} ...", num_pb, knn);

        // Collect per-layer stats at finest level (reads all cells ONCE)
        let mut fine_stats: Vec<CollapsedStat> = Vec::with_capacity(num_layers);
        let mut layer_results = Vec::with_capacity(num_layers);

        for (d, layer) in self.stack.iter().enumerate() {
            let num_features = layer.num_rows();
            let mut stat = CollapsedStat::new(num_features, num_groups, num_batches);

            info!("Layer {}/{}: collecting stats ...", d + 1, num_layers);
            layer.collect_basic_stat(&mut stat)?;
            layer.collect_batch_stat(&mut stat)?;

            // Collect layer-specific gene sums
            let gene_sums = collect_pb_sample_gene_sums(
                layer,
                group_to_cols,
                &col_to_batch,
                &layout.bg_to_pbsamp,
                num_pb,
            )?;

            // Match across batches using shared layout + layer gene sums
            let batch_knn = layer
                .batch_knn_lookup()
                .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
            collect_matched_stat_coarse(
                &layout,
                &gene_sums,
                &layout.pb_sample_to_group,
                batch_knn.as_slice(),
                knn,
                &mut stat,
            )?;

            // Optimize finest-level parameters
            layer_results.push(optimize(
                &stat,
                (1.0, 1.0),
                opt_iter,
                &format!(
                    "Coarsen L1/{} layer {}/{}",
                    level_dims.len(),
                    d + 1,
                    num_layers
                ),
                CalibrateTarget::All,
            )?);
            fine_stats.push(stat);
        }

        let mut results = vec![layer_results];

        // Agglomeratively merge for coarser levels
        let mut prev_stats = fine_stats;
        let mut prev_group_to_cols = group_to_cols.clone();

        for (level, &level_sort_dim) in level_dims.iter().enumerate().skip(1) {
            let level_opt_iter = (opt_iter / 2).max(10);

            // Compute merge mapping (shared across layers)
            let (fine_to_coarse, num_coarse) =
                compute_fine_to_coarse_mapping(&prev_group_to_cols, &fine_codes, level_sort_dim);

            info!(
                "Level {}/{}: sort_dim={}, {} groups (merged from {})",
                level + 1,
                level_dims.len(),
                level_sort_dim,
                num_coarse,
                prev_stats[0].num_samples(),
            );

            // Merge and optimize each layer
            let mut layer_results = Vec::with_capacity(num_layers);
            let mut coarse_stats = Vec::with_capacity(num_layers);

            for (d, prev_stat) in prev_stats.iter().enumerate() {
                let coarse_stat = merge_stat(prev_stat, &fine_to_coarse, num_coarse);

                layer_results.push(optimize(
                    &coarse_stat,
                    (1.0, 1.0),
                    level_opt_iter,
                    &format!(
                        "Coarsen L{}/{} layer {}/{}",
                        level + 1,
                        level_dims.len(),
                        d + 1,
                        num_layers
                    ),
                    CalibrateTarget::All,
                )?);
                coarse_stats.push(coarse_stat);
            }

            results.push(layer_results);

            // Build coarse group_to_cols for next iteration
            let mut coarse_group_to_cols = vec![vec![]; num_coarse];
            for (fine_g, &coarse_g) in fine_to_coarse.iter().enumerate() {
                coarse_group_to_cols[coarse_g].extend_from_slice(&prev_group_to_cols[fine_g]);
            }

            prev_stats = coarse_stats;
            prev_group_to_cols = coarse_group_to_cols;
        }

        Ok(results)
    }
}
