#![allow(dead_code)]

use data_beans::sparse_data_visitors::*;
use data_beans::sparse_io_stack::SparseIoStack;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
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

/// Sparse super-cell gene profile: `rows[sc] = Vec<(gene_idx, sum)>` sorted
/// by `gene_idx`, as produced by `collect_super_cell_gene_sums` and
/// consumed by `collect_matched_stat_coarse` and the refinement pass.
pub type GeneSums = Vec<Vec<(usize, f32)>>;

pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

/// Configuration for multi-level collapsing.
pub struct MultilevelParams {
    pub knn_super_cells: usize,
    pub num_levels: usize,
    pub sort_dim: usize,
    pub num_opt_iter: usize,
    /// Opt-in BBKNN + Poisson DC-SBM refinement on top of the hash
    /// partition. `None` preserves legacy behavior.
    pub refine: Option<crate::refine_multilevel::RefineParams>,
}

impl MultilevelParams {
    pub fn new(proj_dim: usize) -> Self {
        Self {
            knn_super_cells: DEFAULT_KNN,
            num_levels: DEFAULT_NUM_LEVELS,
            sort_dim: proj_dim.min(12),
            num_opt_iter: DEFAULT_OPT_ITER,
            refine: Some(crate::refine_multilevel::RefineParams::default()),
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
        optimize(&stat, (a0, b0), num_opt_iter.unwrap_or(DEFAULT_OPT_ITER))
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

struct KnnParams<'a> {
    knn_batches: usize,
    knn_cells: usize,
    reference_indices: Option<&'a [usize]>,
}

fn collect_matched_stat_visitor(
    sample: usize,
    cells: &[usize],
    data_vec: &SparseIoVec,
    knn_params: &KnnParams,
    arc_stat: Arc<Mutex<&mut CollapsedStat>>,
) -> anyhow::Result<()> {
    let knn_batches = knn_params.knn_batches;
    let knn_cells = knn_params.knn_cells;

    let (y0_matched, source_columns, euclidean_distances) = match knn_params.reference_indices {
        Some(target_indices) => data_vec.read_matched_columns_csc(
            cells.iter().cloned(),
            target_indices,
            knn_cells,
            true,
        )?,
        None => {
            let (mat, src, _matched, dist) = data_vec.read_neighbouring_columns_csc(
                cells.iter().cloned(),
                knn_batches,
                knn_cells,
                true,
                None,
            )?;
            (mat, src, dist)
        }
    };

    let mut y1 = data_vec.read_columns_csc(cells.iter().cloned())?;

    let y1_pos: HashMap<_, _> = cells
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();

    let neg_distance_triplets = source_columns
        .iter()
        .zip(euclidean_distances.iter())
        .enumerate()
        .map(|(t, (&s, &d))| (t, y1_pos[&s], -d))
        .collect::<Vec<_>>();

    ////////////////////////////////////////////////////////
    // zhat[g,j]  =  sum_k w[j,k] * z[g,k] / sum_k w[j,k] //
    // zsum[g,s]  =  sum_j zhat[g,j]                      //
    ////////////////////////////////////////////////////////

    // Normalize distance for each source cell and take a
    // weighted average of the matched vectors using this
    // weight vector
    let ww = CscMat::from_nonzero_triplets(
        y0_matched.ncols(),
        y1.ncols(),
        neg_distance_triplets.as_ref(),
    )?
    .normalize_exp_logits_columns();

    let y1_hat = &y0_matched * ww;
    y1.adjust_by_division_inplace(&y1_hat);

    let mut stat = arc_stat.lock().expect("lock stat");

    for y_j in y1_hat.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.imputed_sum_ds[(gene, sample)] += y;
        }
    }

    for y_j in y1.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.residual_sum_ds[(gene, sample)] += y;
        }
    }

    Ok(())
}

fn collect_basic_stat_visitor(
    sample: usize,
    cells: &[usize],
    data_vec: &SparseIoVec,
    _: &EmptyArg,
    arc_stat: Arc<Mutex<&mut CollapsedStat>>,
) -> anyhow::Result<()> {
    let yy = data_vec.read_columns_csc(cells.iter().cloned())?;

    let mut stat = arc_stat.lock().expect("lock stat");

    for y_j in yy.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.observed_sum_ds[(gene, sample)] += y;
        }
        stat.size_s[sample] += 1_f32; // each column is a sample
    }
    Ok(())
}

fn collect_batch_stat_visitor(
    sample: usize,
    cells_in_sample: &[usize],
    data_vec: &SparseIoVec,
    _: &EmptyArg,
    arc_stat: Arc<Mutex<&mut CollapsedStat>>,
) -> anyhow::Result<()> {
    let yy = data_vec.read_columns_csc(cells_in_sample.iter().cloned())?;

    // cells_in_sample: sample s -> cell j
    // batches: cell j -> batch b
    let batches = data_vec.get_batch_membership(cells_in_sample.iter().cloned());

    let mut stat = arc_stat.lock().expect("lock stat");

    yy.col_iter().zip(batches.iter()).for_each(|(y_j, &b)| {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.observed_sum_db[(gene, b)] += y;
        }
        stat.n_bs[(b, sample)] += 1_f32;
    });
    Ok(())
}

/// Optimize the mean parameters for three Gamma distributions
///
fn optimize(
    stat: &CollapsedStat,
    hyper: (f32, f32),
    num_iter: usize,
) -> anyhow::Result<CollapsedOut> {
    let (a0, b0) = hyper;
    let num_genes = stat.num_genes();
    let num_samples = stat.num_samples();
    let num_batches = stat.num_batches();
    let mut mu_param = GammaMatrix::new((num_genes, num_samples), a0, b0);

    if num_batches > 1 {
        // temporary denominator
        let mut denom_ds = nalgebra::DMatrix::<f32>::zeros(num_genes, num_samples);

        // parameters
        let mut mu_adj_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut mu_resid_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut gamma_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut delta_param = GammaMatrix::new((num_genes, num_batches), a0, b0);

        ////////////////////////////////////
        // E[y_resid] = E[μ_resid]        //
        //       E[y] = E[μ_resid] * E[μ] //
        //   E[y_hat] = E[γ] * E[μ]       //
        //   E[y_bat] = E[δ] * E[μ]       //
        ////////////////////////////////////

        //            residual_sum_ds
        // μ_resid = -----------------
        //            1_d * size_s'

        {
            for s in 0..num_samples {
                denom_ds.column_mut(s).add_scalar_mut(stat.size_s[s]);
            }
            mu_resid_param.update_stat(&stat.residual_sum_ds, &denom_ds);
            mu_resid_param.calibrate();
        };

        let pb = ProgressBar::new(num_iter as u64).with_style(
            ProgressStyle::with_template("Optimizing {bar:40} {pos}/{len} iterations ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );
        (0..num_iter)
            .progress_with(pb.clone())
            .for_each(|_opt_iter| {
                #[cfg(debug_assertions)]
                {
                    debug!("iteration: {}", &_opt_iter);
                }

                let resid_ds = mu_resid_param.posterior_mean();
                let gamma_ds = gamma_param.posterior_mean();

                //      observed_ds + imputed_sum_ds
                // μ = ---------------------------------
                //      (μ_resid + γ) .* (1_d * size_s')

                denom_ds.copy_from(&(resid_ds + gamma_ds));
                for s in 0..num_samples {
                    denom_ds.column_mut(s).scale_mut(stat.size_s[s]);
                }

                mu_adj_param
                    .update_stat(&(&stat.observed_sum_ds + &stat.imputed_sum_ds), &denom_ds);
                mu_adj_param.calibrate_with(CalibrateTarget::MeanOnly);

                let mu_ds = mu_adj_param.posterior_mean();

                //      imputed_sum_ds
                // γ = ---------------------
                //      μ .* (1_d * size_s')

                denom_ds.copy_from(mu_ds);
                for s in 0..num_samples {
                    denom_ds.column_mut(s).scale_mut(stat.size_s[s]);
                }
                gamma_param.update_stat(&stat.imputed_sum_ds, &denom_ds);
                gamma_param.calibrate_with(CalibrateTarget::MeanOnly);
            });
        pb.finish_and_clear();

        // Full calibration after loop for output/export
        mu_adj_param.calibrate();
        gamma_param.calibrate();

        //      observed_db
        // δ = ---------------------
        //      μ * size_bs'
        {
            let mu_ds = mu_adj_param.posterior_mean();
            delta_param.update_stat(&stat.observed_sum_db, &(mu_ds * &stat.n_bs.transpose()));
            delta_param.calibrate();
        }

        // Take the observed mean
        {
            let mut denom_ds = DMatrix::<f32>::zeros(num_genes, num_samples);
            for s in 0..num_samples {
                denom_ds.column_mut(s).add_scalar_mut(stat.size_s[s]);
            }
            mu_param.update_stat(&stat.observed_sum_ds, &denom_ds);
            mu_param.calibrate();
        };

        Ok(CollapsedOut {
            mu_observed: mu_param,
            mu_adjusted: Some(mu_adj_param),
            mu_residual: Some(mu_resid_param),
            gamma: Some(gamma_param),
            delta: Some(delta_param),
        })
    } else {
        let mut denom_ds = DMatrix::<f32>::zeros(num_genes, num_samples);
        for s in 0..num_samples {
            denom_ds.column_mut(s).add_scalar_mut(stat.size_s[s]);
        }
        mu_param.update_stat(&stat.observed_sum_ds, &denom_ds);
        mu_param.calibrate();
        Ok(CollapsedOut {
            mu_observed: mu_param,
            mu_adjusted: None,
            mu_residual: None,
            gamma: None,
            delta: None,
        })
    }
}

/// output struct to make the model parameters more accessible
#[derive(Debug)]
pub struct CollapsedOut {
    pub mu_observed: GammaMatrix,
    pub mu_adjusted: Option<GammaMatrix>,
    pub mu_residual: Option<GammaMatrix>,
    pub gamma: Option<GammaMatrix>,
    pub delta: Option<GammaMatrix>,
}

/// a struct to hold the sufficient statistics for the model
#[derive(Debug, Clone)]
pub struct CollapsedStat {
    pub observed_sum_ds: nalgebra::DMatrix<f32>, // observed sum within each sample
    pub imputed_sum_ds: nalgebra::DMatrix<f32>,  // counterfactual sum within each sample
    pub residual_sum_ds: nalgebra::DMatrix<f32>, // residual sum within each sample
    pub size_s: nalgebra::DVector<f32>,          // sample s size
    pub observed_sum_db: nalgebra::DMatrix<f32>, // divergence numerator
    pub n_bs: nalgebra::DMatrix<f32>,            // batch-specific sample size
}

impl CollapsedStat {
    pub fn new(ngene: usize, nsample: usize, nbatch: usize) -> Self {
        Self {
            observed_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            imputed_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            residual_sum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            size_s: nalgebra::DVector::<f32>::zeros(nsample),
            observed_sum_db: nalgebra::DMatrix::<f32>::zeros(ngene, nbatch),
            n_bs: nalgebra::DMatrix::<f32>::zeros(nbatch, nsample),
        }
    }

    pub fn num_genes(&self) -> usize {
        self.observed_sum_ds.nrows()
    }

    pub fn num_samples(&self) -> usize {
        self.observed_sum_ds.ncols()
    }

    pub fn num_batches(&self) -> usize {
        self.observed_sum_db.ncols()
    }

    pub fn clear(&mut self) {
        self.observed_sum_ds.fill(0_f32);
        self.imputed_sum_ds.fill(0_f32);
        self.residual_sum_ds.fill(0_f32);
        self.observed_sum_db.fill(0_f32);
        self.size_s.fill(0_f32);
        self.n_bs.fill(0_f32);
    }

    /// Select a subset of sample columns (groups) by index.
    pub fn select_columns(&self, indices: &[usize]) -> Self {
        let n_new = indices.len();
        let ng = self.num_genes();
        let nb = self.num_batches();
        let mut out = Self::new(ng, n_new, nb);
        for (new_col, &old_col) in indices.iter().enumerate() {
            out.observed_sum_ds
                .column_mut(new_col)
                .copy_from(&self.observed_sum_ds.column(old_col));
            out.imputed_sum_ds
                .column_mut(new_col)
                .copy_from(&self.imputed_sum_ds.column(old_col));
            out.residual_sum_ds
                .column_mut(new_col)
                .copy_from(&self.residual_sum_ds.column(old_col));
            out.size_s[new_col] = self.size_s[old_col];
            for b in 0..nb {
                out.n_bs[(b, new_col)] = self.n_bs[(b, old_col)];
            }
        }
        out.observed_sum_db.copy_from(&self.observed_sum_db);
        out
    }
}

/// Resample from over-resolved sufficient statistics: randomly select
/// ~half the groups, then optimise to produce a fresh `CollapsedOut`.
pub fn resample_and_optimize(
    stat: &CollapsedStat,
    rng: &mut impl rand::Rng,
    opt_iter: usize,
) -> anyhow::Result<CollapsedOut> {
    use rand::seq::SliceRandom;
    let n = stat.num_samples();
    let target = n / 2;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices.truncate(target);
    indices.sort_unstable();
    let sub_stat = stat.select_columns(&indices);
    optimize(&sub_stat, (1.0, 1.0), opt_iter)
}

/////////////////////////////////////////////////////////////
// Multi-level (METIS-inspired) collapsing for batch effects
/////////////////////////////////////////////////////////////

const DEFAULT_NUM_LEVELS: usize = 2;
const DEFAULT_COARSEST_SORT_DIM: usize = 7;

/// Shared layout for super-cells (batch × group intersections).
/// Reusable across multiple layers in a `SparseIoStack`.
///
/// Cross-batch neighbor queries go through `SparseIoVec::batch_knn_lookup`
/// (the per-batch HNSW over cells), not a super-cell-level index: the
/// centroid of a super-cell is used as the query point, results are
/// deduped to super-cells via `cell_to_sc`.
pub struct SuperCellLayout {
    /// Centroid matrix: proj_dim x num_super_cells
    pub centroids: DMatrix<f32>,
    /// Number of cells in each super-cell
    pub cell_counts: Vec<f32>,
    /// Batch index for each super-cell
    pub super_cell_to_batch: Vec<usize>,
    /// Sample/group index for each super-cell
    pub super_cell_to_group: Vec<usize>,
    /// Maps (batch, group) → super-cell index
    pub bg_to_sc: HashMap<(usize, usize), usize>,
    /// Global cell index → owning super-cell index.
    pub cell_to_sc: Vec<usize>,
}

/// Pre-aggregated super-cell data for fast cross-batch matching.
/// Each super-cell is the intersection of a (batch, sample) pair.
pub struct SuperCellCollection {
    pub layout: SuperCellLayout,
    /// Sparse gene sums per super-cell: Vec of (gene_index, sum)
    pub gene_sums: Vec<Vec<(usize, f32)>>,
    /// Number of genes
    pub num_genes: usize,
}

/// Intermediate per-batch accumulator used during super-cell construction.
struct BatchAccumulator {
    centroid_sum: Vec<f32>,
    gene_sum: HashMap<usize, f32>,
    count: usize,
}

/// A single super-cell produced from a (batch, group) intersection.
struct SuperCellData {
    centroid: Vec<f32>,
    gene_sums: Vec<(usize, f32)>,
    cell_count: f32,
    batch: usize,
    group: usize,
}

/// Build the shared super-cell layout from (batch, group) intersections.
///
/// For each non-empty (batch, group) block:
/// - Centroid = mean of projection vectors
/// - Cell count = number of cells in block
///
/// This only uses `proj_kn` (no CSC reads), so it can be shared across layers.
fn build_super_cell_layout(
    group_to_cols: &[Vec<usize>],
    col_to_batch: &[usize],
    proj_kn: &DMatrix<f32>,
) -> anyhow::Result<SuperCellLayout> {
    let proj_dim = proj_kn.nrows();

    /// Intermediate per-batch accumulator for centroid computation.
    struct CentroidAccum {
        centroid_sum: Vec<f32>,
        count: usize,
    }

    // Collect centroid data per group in parallel
    type CentroidTuple = (usize, usize, Vec<f32>, f32);
    let per_group_results: Vec<Vec<CentroidTuple>> = group_to_cols
        .par_iter()
        .enumerate()
        .map(|(group, cells)| {
            let mut batch_data: HashMap<usize, CentroidAccum> = HashMap::default();

            for &glob_idx in cells {
                let batch = col_to_batch[glob_idx];
                let acc = batch_data.entry(batch).or_insert_with(|| CentroidAccum {
                    centroid_sum: vec![0f32; proj_dim],
                    count: 0,
                });
                for d in 0..proj_dim {
                    acc.centroid_sum[d] += proj_kn[(d, glob_idx)];
                }
                acc.count += 1;
            }

            batch_data
                .into_iter()
                .filter(|(_, acc)| acc.count > 0)
                .map(|(batch, acc)| {
                    let inv_count = 1.0 / acc.count as f32;
                    let centroid: Vec<f32> =
                        acc.centroid_sum.iter().map(|v| v * inv_count).collect();
                    (batch, group, centroid, acc.count as f32)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Flatten into a single list
    let all_sc: Vec<_> = per_group_results.into_iter().flatten().collect();
    let num_sc = all_sc.len();

    if num_sc == 0 {
        return Err(anyhow::anyhow!("no super-cells built"));
    }

    // Build centroid matrix and metadata
    let mut centroids = DMatrix::<f32>::zeros(proj_dim, num_sc);
    let mut cell_counts = Vec::with_capacity(num_sc);
    let mut sc_to_batch = Vec::with_capacity(num_sc);
    let mut sc_to_group = Vec::with_capacity(num_sc);
    let mut bg_to_sc = HashMap::default();

    for (i, (batch, group, centroid, count)) in all_sc.into_iter().enumerate() {
        for (d, &v) in centroid.iter().enumerate() {
            centroids[(d, i)] = v;
        }
        cell_counts.push(count);
        sc_to_batch.push(batch);
        sc_to_group.push(group);
        bg_to_sc.insert((batch, group), i);
    }

    // Cell → super-cell inversion (each cell belongs to exactly one sc via
    // (batch, group)).
    let ncols = col_to_batch.len();
    let mut cell_to_sc = vec![usize::MAX; ncols];
    for (group, cells) in group_to_cols.iter().enumerate() {
        for &c in cells {
            let b = col_to_batch[c];
            if let Some(&sc) = bg_to_sc.get(&(b, group)) {
                cell_to_sc[c] = sc;
            }
        }
    }

    Ok(SuperCellLayout {
        centroids,
        cell_counts,
        super_cell_to_batch: sc_to_batch,
        super_cell_to_group: sc_to_group,
        bg_to_sc,
        cell_to_sc,
    })
}

/// Collect gene sums for each super-cell from a single `SparseIoVec` layer.
///
/// Uses the `bg_to_sc` mapping from the layout to accumulate gene expression
/// per super-cell, parallelized over groups.
fn collect_super_cell_gene_sums(
    data_vec: &SparseIoVec,
    group_to_cols: &[Vec<usize>],
    col_to_batch: &[usize],
    bg_to_sc: &HashMap<(usize, usize), usize>,
    num_sc: usize,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    use indicatif::ParallelProgressIterator;
    let pb = ProgressBar::new(group_to_cols.len() as u64).with_style(
        ProgressStyle::with_template("Super-cell gene sums {bar:40} {pos}/{len} groups ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    let gene_sum_maps: Vec<(usize, HashMap<usize, f32>)> = group_to_cols
        .par_iter()
        .enumerate()
        .progress_with(pb.clone())
        .flat_map(|(group, cells)| {
            let yy = data_vec
                .read_columns_csc(cells.iter().cloned())
                .expect("read_columns_csc");

            let mut batch_gene_sums: HashMap<usize, HashMap<usize, f32>> = HashMap::default();

            for (local_idx, y_j) in yy.col_iter().enumerate() {
                let batch = col_to_batch[cells[local_idx]];
                let gene_map = batch_gene_sums.entry(batch).or_default();
                for (&gene, &val) in y_j.row_indices().iter().zip(y_j.values().iter()) {
                    *gene_map.entry(gene).or_default() += val;
                }
            }

            batch_gene_sums
                .into_iter()
                .filter_map(|(batch, gene_map)| {
                    bg_to_sc
                        .get(&(batch, group))
                        .map(|&sc_idx| (sc_idx, gene_map))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut gene_sums: Vec<Vec<(usize, f32)>> = vec![vec![]; num_sc];
    for (sc_idx, gene_map) in gene_sum_maps {
        let mut sorted: Vec<(usize, f32)> = gene_map.into_iter().collect();
        sorted.sort_unstable_by_key(|&(g, _)| g);
        gene_sums[sc_idx] = sorted;
    }

    Ok(gene_sums)
}

/// Build super-cells (layout + gene sums) from a single `SparseIoVec`.
fn build_super_cells(
    data_vec: &SparseIoVec,
    proj_kn: &DMatrix<f32>,
    num_genes: usize,
) -> anyhow::Result<SuperCellCollection> {
    let group_to_cols = data_vec
        .take_grouped_columns()
        .ok_or(anyhow::anyhow!("columns not assigned to groups"))?;
    let col_to_batch: Vec<usize> = (0..proj_kn.ncols())
        .map(|c| data_vec.get_batch_membership(std::iter::once(c))[0])
        .collect();

    let layout = build_super_cell_layout(group_to_cols, &col_to_batch, proj_kn)?;
    let num_sc = layout.cell_counts.len();
    let gene_sums = collect_super_cell_gene_sums(
        data_vec,
        group_to_cols,
        &col_to_batch,
        &layout.bg_to_sc,
        num_sc,
    )?;

    Ok(SuperCellCollection {
        layout,
        gene_sums,
        num_genes,
    })
}

/// Per-super-cell, for each non-own batch return up to `knn` distinct
/// super-cells whose member cells are closest to `sc`'s centroid.
///
/// Queries `SparseIoVec::batch_knn_lookup` (per-batch HNSW over cells),
/// then dedups hits to super-cells via `layout.cell_to_sc`. Distances come
/// back as the minimum over collapsed cells for each super-cell.
///
/// Returns: `result[sc] = Vec<(other_sc, distance)>` flattened across all
/// non-own batches.
fn per_batch_sc_neighbors(
    layout: &SuperCellLayout,
    batch_knn_lookup: &[ColumnDict<usize>],
    knn: usize,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    use indicatif::ParallelProgressIterator;
    use matrix_util::knn_match::MakeVecPoint;
    let num_sc = layout.cell_counts.len();
    // Oversample cells per batch so dedup-to-sc still yields ~knn uniques.
    let cell_oversample = (knn * 4 + 1).max(knn);

    let pb = ProgressBar::new(num_sc as u64).with_style(
        ProgressStyle::with_template("BBKNN match {bar:40} {pos}/{len} super-cells ({eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    let result: anyhow::Result<Vec<Vec<(usize, f32)>>> = (0..num_sc)
        .into_par_iter()
        .progress_with(pb.clone())
        .map(|sc| -> anyhow::Result<Vec<(usize, f32)>> {
            let sc_batch = layout.super_cell_to_batch[sc];
            let centroid = layout.centroids.column(sc).to_vp();
            let mut all_hits: Vec<(usize, f32)> = Vec::new();
            for (b, bknn) in batch_knn_lookup.iter().enumerate() {
                if b == sc_batch {
                    continue;
                }
                let (cell_ids, dists) = bknn.search_by_query_data(&centroid, cell_oversample)?;
                let mut best: HashMap<usize, f32> = HashMap::default();
                for (&c, &d) in cell_ids.iter().zip(dists.iter()) {
                    let other_sc = layout.cell_to_sc[c];
                    if other_sc == usize::MAX || other_sc == sc {
                        continue;
                    }
                    best.entry(other_sc)
                        .and_modify(|old| {
                            if d < *old {
                                *old = d;
                            }
                        })
                        .or_insert(d);
                    if best.len() >= knn {
                        break;
                    }
                }
                let mut per_batch: Vec<(usize, f32)> = best.into_iter().collect();
                per_batch
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                per_batch.truncate(knn);
                all_hits.extend(per_batch);
            }
            Ok(all_hits)
        })
        .collect();
    pb.finish_and_clear();
    result
}

/// Match super-cells across batches and accumulate counterfactual
/// statistics into `stat.imputed_sum_ds` and `stat.residual_sum_ds`.
///
/// `sc_to_group` is the per-super-cell group assignment to use when writing
/// into stat columns; callers pass `&layout.super_cell_to_group` for the
/// hash-partition mapping, or a refined mapping from
/// `refine_multilevel::refine_assignments`.
///
/// `knn` is now the per-other-batch neighbour count: each super-cell draws
/// up to `knn` distinct foreign super-cells from **each** non-own batch, so
/// the total match set is up to `knn · (num_batches − 1)`.
fn collect_matched_stat_coarse(
    layout: &SuperCellLayout,
    gene_sums: &[Vec<(usize, f32)>],
    sc_to_group: &[usize],
    batch_knn_lookup: &[ColumnDict<usize>],
    knn: usize,
    stat: &mut CollapsedStat,
) -> anyhow::Result<()> {
    let num_sc = layout.cell_counts.len();
    debug_assert_eq!(sc_to_group.len(), num_sc);

    let neighbors_per_sc = per_batch_sc_neighbors(layout, batch_knn_lookup, knn)?;

    for sc_idx in 0..num_sc {
        let sc_group = sc_to_group[sc_idx];
        let sc_count = layout.cell_counts[sc_idx];

        if sc_count < 1.0 {
            continue;
        }

        let filtered: Vec<(usize, f32)> = neighbors_per_sc[sc_idx].clone();

        if filtered.is_empty() {
            continue;
        }

        // Softmax weights from negative distances
        let max_neg_d = filtered
            .iter()
            .map(|(_, d)| -d)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = filtered
            .iter()
            .map(|(_, d)| (-d - max_neg_d).exp())
            .collect();
        let w_sum: f32 = weights.iter().sum();
        if w_sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= w_sum);
        }

        // Counterfactual: weighted average of matched super-cells'
        // per-cell gene expression
        // y_hat[g] = sum_k w[k] * gene_sums[k][g] / cell_counts[k]
        let mut y_hat: HashMap<usize, f32> = HashMap::default();
        for ((matched_sc, _), &w) in filtered.iter().zip(weights.iter()) {
            let matched_count = layout.cell_counts[*matched_sc];
            if matched_count < 1.0 {
                continue;
            }
            let inv_count = 1.0 / matched_count;
            for &(gene, val) in &gene_sums[*matched_sc] {
                *y_hat.entry(gene).or_default() += w * val * inv_count;
            }
        }

        // Accumulate imputed_sum_ds[g, s] += cell_counts[sc] * y_hat[g]
        for (&gene, &y) in &y_hat {
            stat.imputed_sum_ds[(gene, sc_group)] += sc_count * y;
        }

        // Accumulate residual_sum_ds[g, s] += y_obs[g] / y_hat[g]
        // where y_obs[g] = gene_sums[sc][g] / cell_counts[sc]
        // -> residual_sum_ds[g, s] += gene_sums[sc][g] / (cell_counts[sc] * y_hat[g])
        //    then × cell_counts[sc] to match original scaling
        // = gene_sums[sc][g] / y_hat[g]
        for &(gene, val) in &gene_sums[sc_idx] {
            if let Some(&y_h) = y_hat.get(&gene) {
                if y_h > 0.0 {
                    stat.residual_sum_ds[(gene, sc_group)] += val / y_h;
                }
            }
        }
    }

    Ok(())
}

/// Format a per-cell group-index vector as fixed-width zero-padded strings
/// so that `SparseIoVec::assign_groups`' lexicographic key sort agrees with
/// numeric order. `k` is the number of distinct groups (`group ∈ 0..k`).
fn pad_numeric_labels(cell_to_group: &[usize], k: usize) -> Vec<String> {
    let width = {
        let mut w = 1usize;
        let mut n = k.max(1) - 1;
        while n >= 10 {
            w += 1;
            n /= 10;
        }
        w
    };
    cell_to_group
        .iter()
        .map(|g| format!("{:0width$}", g, width = width))
        .collect()
}

/// Derive a fine→coarse group mapping from two consecutive refined levels.
///
/// The refinement pass enforces hierarchy (sibling-constrained moves), so
/// all super-cells sharing a level-`fine` group also share the same
/// level-`coarse` group. This picks the first super-cell of each fine group
/// to read the coarse label.
fn fine_to_coarse_from_refined(
    sc_to_fine: &[usize],
    sc_to_coarse: &[usize],
    num_fine: usize,
) -> Vec<usize> {
    let mut mapping = vec![usize::MAX; num_fine];
    for sc in 0..sc_to_fine.len() {
        let f = sc_to_fine[sc];
        if mapping[f] == usize::MAX {
            mapping[f] = sc_to_coarse[sc];
        } else {
            debug_assert_eq!(
                mapping[f], sc_to_coarse[sc],
                "refinement broke hierarchy at fine group {}",
                f
            );
        }
    }
    mapping
}

/// Per-level initial super-cell → group, derived from the finest binary
/// hash codes by bit-masking each level's sort dim and compacting labels to
/// `0..k_level`. Each super-cell's finest hash code is read from any of its
/// member cells (all cells in a super-cell share the same finest group).
fn initial_per_level_from_hash(
    fine_codes: &[usize],
    super_cell_to_cells: &[Vec<usize>],
    level_dims: &[usize],
) -> Vec<Vec<usize>> {
    let num_sc = super_cell_to_cells.len();
    level_dims
        .iter()
        .map(|&d| {
            let mask = if d >= usize::BITS as usize {
                usize::MAX
            } else {
                (1_usize << d).wrapping_sub(1)
            };
            let codes: Vec<usize> = (0..num_sc)
                .map(|sc| fine_codes[super_cell_to_cells[sc][0]] & mask)
                .collect();
            crate::refine_multilevel::compact_labels(&codes).0
        })
        .collect()
}

/// Run refinement when `allow_refine`, else return the compacted initial
/// mapping unchanged (single-batch → no BBKNN candidates, nothing to refine).
fn refine_or_identity(
    allow_refine: bool,
    inputs: &crate::refine_multilevel::RefineInputs<'_>,
    refine_params: &crate::refine_multilevel::RefineParams,
) -> anyhow::Result<crate::refine_multilevel::RefinedAssignment> {
    if allow_refine {
        crate::refine_multilevel::refine_assignments(inputs, refine_params)
    } else {
        let mut sc_to_group: Vec<Vec<usize>> =
            Vec::with_capacity(inputs.initial_sc_to_group_per_level.len());
        let mut num_groups_per_level =
            Vec::with_capacity(inputs.initial_sc_to_group_per_level.len());
        for lvl in inputs.initial_sc_to_group_per_level {
            let (compact, k) = crate::refine_multilevel::compact_labels(lvl);
            num_groups_per_level.push(k);
            sc_to_group.push(compact);
        }
        Ok(crate::refine_multilevel::RefinedAssignment {
            sc_to_group,
            num_groups_per_level,
        })
    }
}

/// Build per-super-cell cell lists from the finest hash partition and batch
/// membership: sc → list of raw cell indices contained in that super-cell.
fn build_super_cell_to_cells(
    layout: &SuperCellLayout,
    group_to_cols_finest: &[Vec<usize>],
    col_to_batch: &[usize],
) -> Vec<Vec<usize>> {
    let num_sc = layout.cell_counts.len();
    let mut out: Vec<Vec<usize>> = vec![vec![]; num_sc];
    for (group, cells) in group_to_cols_finest.iter().enumerate() {
        for &c in cells {
            let batch = col_to_batch[c];
            if let Some(&sc) = layout.bg_to_sc.get(&(batch, group)) {
                out[sc].push(c);
            }
        }
    }
    out
}

/// Shared inputs to both the `SparseIoVec` and `SparseIoStack` refinement
/// helpers. Keeps call-site signatures compact; every field is derivable
/// from `MultilevelParams` + the finest-level hash partition.
#[derive(Clone, Copy)]
struct RefineCollectCtx<'a> {
    fine_codes: &'a [usize],
    group_to_cols_finest: &'a [Vec<usize>],
    level_dims: &'a [usize],
    num_features: usize,
    num_batches: usize,
    knn: usize,
    opt_iter: usize,
    refine_params: &'a crate::refine_multilevel::RefineParams,
}

/// Refinement integration path for `SparseIoVec`.
///
/// Walks each level of the hash-initialized hierarchy, runs
fn collapse_level_progress_bar(num_levels: usize) -> ProgressBar {
    ProgressBar::new(num_levels as u64).with_style(
        ProgressStyle::with_template("Collapse levels {bar:40} {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("##-"),
    )
}

/// `refine_multilevel::refine_assignments` over super-cells, then rebuilds
/// `CollapsedStat` per level from the refined cell → group assignment and
/// emits `CollapsedOut` with identical shape to the legacy path.
fn refine_and_collect_single_layer(
    data_vec: &mut SparseIoVec,
    proj_kn: &DMatrix<f32>,
    ctx: &RefineCollectCtx<'_>,
) -> anyhow::Result<Vec<CollapsedOut>> {
    let RefineCollectCtx {
        fine_codes,
        group_to_cols_finest,
        level_dims,
        num_features,
        num_batches,
        knn,
        opt_iter,
        refine_params,
    } = *ctx;
    info!(
        "Multi-level refinement path (BBKNN + DC-SBM): {} levels",
        level_dims.len()
    );

    // 1. Build super-cells (layout + gene sums) from the finest partition.
    let super_cells = build_super_cells(data_vec, proj_kn, num_features)?;
    let num_sc = super_cells.layout.cell_counts.len();
    let ncells_dbg = proj_kn.ncols();
    info!(
        "Built {} super-cells from {} cells (ratio {:.2}; knn={})",
        num_sc,
        ncells_dbg,
        num_sc as f32 / ncells_dbg.max(1) as f32,
        knn
    );
    if num_sc as f32 > 0.8 * ncells_dbg as f32 {
        warn!(
            "super-cell count ({}) is close to cell count ({}) — hash partition is too fine \
             (many 1-cell super-cells). Consider lowering --sort-dim.",
            num_sc, ncells_dbg
        );
    }

    // 2. sc → cells and col → batch.
    let ncols = proj_kn.ncols();
    let col_to_batch: Vec<usize> = data_vec.get_batch_membership(0..ncols);
    let super_cell_to_cells =
        build_super_cell_to_cells(&super_cells.layout, group_to_cols_finest, &col_to_batch);

    let initial_per_level =
        initial_per_level_from_hash(fine_codes, &super_cell_to_cells, level_dims);
    let empty: [ColumnDict<usize>; 0] = [];
    let batch_knn: &[ColumnDict<usize>] = if num_batches >= 2 {
        data_vec
            .batch_knn_lookup()
            .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?
            .as_slice()
    } else {
        &empty
    };
    let inputs = crate::refine_multilevel::RefineInputs {
        layout: &super_cells.layout,
        gene_sums: &super_cells.gene_sums,
        num_genes: num_features,
        super_cell_to_cells: &super_cell_to_cells,
        batch_knn_lookup: batch_knn,
        k_per_batch: knn,
        initial_sc_to_group_per_level: &initial_per_level,
    };
    let refined = refine_or_identity(num_batches >= 2, &inputs, refine_params)?;

    // 5. Build finest CollapsedStat once from a full data pass, then derive
    //    coarser levels by `merge_stat` on column-aggregated sums — avoids
    //    re-reading all cells at every level (matches legacy merge descent).
    let num_levels = level_dims.len();
    let k_finest = refined.num_groups_per_level[0];
    let mut cell_to_group_finest = vec![0usize; ncols];
    for (sc, cells) in super_cell_to_cells.iter().enumerate() {
        let g = refined.sc_to_group[0][sc];
        for &c in cells {
            cell_to_group_finest[c] = g;
        }
    }
    let finest_str = pad_numeric_labels(&cell_to_group_finest, k_finest);
    data_vec.assign_groups(&finest_str, None);
    debug_assert_eq!(data_vec.num_groups(), k_finest);

    let mut fine_stat = CollapsedStat::new(num_features, k_finest, num_batches);
    data_vec.collect_basic_stat(&mut fine_stat)?;
    if num_batches >= 2 {
        data_vec.collect_batch_stat(&mut fine_stat)?;
        let batch_knn = data_vec
            .batch_knn_lookup()
            .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
        collect_matched_stat_coarse(
            &super_cells.layout,
            &super_cells.gene_sums,
            &refined.sc_to_group[0],
            batch_knn.as_slice(),
            knn,
            &mut fine_stat,
        )?;
    }

    info!(
        "Level 1/{}: refined k={} (finest; {} cells read)",
        num_levels, k_finest, ncols
    );
    let level_pb = collapse_level_progress_bar(num_levels);
    let mut results: Vec<CollapsedOut> = Vec::with_capacity(num_levels);
    let finest_out = optimize(&fine_stat, (1.0, 1.0), opt_iter)?;
    results.push(finest_out);
    level_pb.inc(1);

    let mut prev_stat = fine_stat;
    for level in 1..num_levels {
        let k_prev = refined.num_groups_per_level[level - 1];
        let k_level = refined.num_groups_per_level[level];
        let fine_to_coarse = fine_to_coarse_from_refined(
            &refined.sc_to_group[level - 1],
            &refined.sc_to_group[level],
            k_prev,
        );
        let coarse_stat = merge_stat(&prev_stat, &fine_to_coarse, k_level);
        info!(
            "Level {}/{}: refined k={} (merged from {})",
            level + 1,
            num_levels,
            k_level,
            k_prev
        );
        let level_opt_iter = (opt_iter / 2).max(10);
        let out = optimize(&coarse_stat, (1.0, 1.0), level_opt_iter)?;
        results.push(out);
        prev_stat = coarse_stat;
        level_pb.inc(1);
    }
    level_pb.finish_and_clear();

    Ok(results)
}

/// Refinement integration path for `SparseIoStack`.
///
/// Shares one `RefinedAssignment` across all layers (first-layer-owns the
/// grouping decision, matching the existing stack convention at line 1337 of
/// the legacy path). Per level × layer we rebuild `CollapsedStat` and emit
/// `CollapsedOut`.
fn refine_and_collect_stack(
    stack: &mut SparseIoStack,
    proj_kn: &DMatrix<f32>,
    ctx: &RefineCollectCtx<'_>,
) -> anyhow::Result<Vec<Vec<CollapsedOut>>> {
    let RefineCollectCtx {
        fine_codes,
        group_to_cols_finest,
        level_dims,
        num_features: _,
        num_batches,
        knn,
        opt_iter,
        refine_params,
    } = *ctx;
    let num_layers = stack.num_types();
    info!(
        "Multi-level stack refinement (BBKNN + DC-SBM): {} layers × {} levels",
        num_layers,
        level_dims.len()
    );

    let ncols = proj_kn.ncols();
    let col_to_batch: Vec<usize> = stack.stack[0].get_batch_membership(0..ncols);

    // Build shared super-cell layout from layer[0]'s row count and the shared
    // projection. The layout only uses `proj_kn` + grouping, no raw reads.
    let layout = build_super_cell_layout(group_to_cols_finest, &col_to_batch, proj_kn)?;
    let num_sc = layout.cell_counts.len();

    // Gene sums for layer[0] drive the refinement (first-layer-owns).
    let owner_num_features = stack.stack[0].num_rows();
    let gene_sums_owner = collect_super_cell_gene_sums(
        &stack.stack[0],
        group_to_cols_finest,
        &col_to_batch,
        &layout.bg_to_sc,
        num_sc,
    )?;

    let super_cell_to_cells =
        build_super_cell_to_cells(&layout, group_to_cols_finest, &col_to_batch);

    let initial_per_level =
        initial_per_level_from_hash(fine_codes, &super_cell_to_cells, level_dims);
    let empty: [ColumnDict<usize>; 0] = [];
    let batch_knn: &[ColumnDict<usize>] = if num_batches >= 2 {
        stack.stack[0]
            .batch_knn_lookup()
            .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?
            .as_slice()
    } else {
        &empty
    };
    let inputs = crate::refine_multilevel::RefineInputs {
        layout: &layout,
        gene_sums: &gene_sums_owner,
        num_genes: owner_num_features,
        super_cell_to_cells: &super_cell_to_cells,
        batch_knn_lookup: batch_knn,
        k_per_batch: knn,
        initial_sc_to_group_per_level: &initial_per_level,
    };
    let refined = refine_or_identity(num_batches >= 2, &inputs, refine_params)?;

    // Per-layer gene_sums for the remaining layers (layer 0 reuses `gene_sums_owner`).
    let mut per_layer_gene_sums: Vec<GeneSums> = Vec::with_capacity(num_layers);
    for (d, layer) in stack.stack.iter().enumerate() {
        if d == 0 {
            per_layer_gene_sums.push(gene_sums_owner.clone());
        } else {
            per_layer_gene_sums.push(collect_super_cell_gene_sums(
                layer,
                group_to_cols_finest,
                &col_to_batch,
                &layout.bg_to_sc,
                num_sc,
            )?);
        }
    }

    // Finest CollapsedStat per layer via a single data pass, then descend
    //    into coarser levels by `merge_stat` on column aggregates.
    let num_levels = level_dims.len();
    let k_finest = refined.num_groups_per_level[0];
    let mut cell_to_group_finest = vec![0usize; ncols];
    for (sc, cells) in super_cell_to_cells.iter().enumerate() {
        let g = refined.sc_to_group[0][sc];
        for &c in cells {
            cell_to_group_finest[c] = g;
        }
    }
    let finest_str = pad_numeric_labels(&cell_to_group_finest, k_finest);
    for layer in stack.stack.iter_mut() {
        layer.assign_groups(&finest_str, None);
    }

    let mut fine_stats: Vec<CollapsedStat> = Vec::with_capacity(num_layers);
    let mut finest_layer_results = Vec::with_capacity(num_layers);
    for (d, layer) in stack.stack.iter().enumerate() {
        let num_features = layer.num_rows();
        let mut stat = CollapsedStat::new(num_features, k_finest, num_batches);
        layer.collect_basic_stat(&mut stat)?;
        if num_batches >= 2 {
            layer.collect_batch_stat(&mut stat)?;
            let batch_knn = layer
                .batch_knn_lookup()
                .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
            collect_matched_stat_coarse(
                &layout,
                &per_layer_gene_sums[d],
                &refined.sc_to_group[0],
                batch_knn.as_slice(),
                knn,
                &mut stat,
            )?;
        }
        let out = optimize(&stat, (1.0, 1.0), opt_iter)?;
        finest_layer_results.push(out);
        fine_stats.push(stat);
    }
    info!(
        "Level 1/{}: refined k={} (finest; {} layers × {} cells)",
        num_levels, k_finest, num_layers, ncols
    );
    let level_pb = collapse_level_progress_bar(num_levels);
    let mut results: Vec<Vec<CollapsedOut>> = Vec::with_capacity(num_levels);
    results.push(finest_layer_results);
    level_pb.inc(1);

    let mut prev_stats = fine_stats;
    for level in 1..num_levels {
        let k_prev = refined.num_groups_per_level[level - 1];
        let k_level = refined.num_groups_per_level[level];
        let fine_to_coarse = fine_to_coarse_from_refined(
            &refined.sc_to_group[level - 1],
            &refined.sc_to_group[level],
            k_prev,
        );
        let level_opt_iter = (opt_iter / 2).max(10);
        let mut layer_results = Vec::with_capacity(num_layers);
        let mut coarse_stats = Vec::with_capacity(num_layers);
        for prev_stat in prev_stats.iter() {
            let coarse_stat = merge_stat(prev_stat, &fine_to_coarse, k_level);
            let out = optimize(&coarse_stat, (1.0, 1.0), level_opt_iter)?;
            layer_results.push(out);
            coarse_stats.push(coarse_stat);
        }
        info!(
            "Level {}/{}: refined k={} (merged from {}, {} layers)",
            level + 1,
            num_levels,
            k_level,
            k_prev,
            num_layers
        );
        results.push(layer_results);
        prev_stats = coarse_stats;
        level_pb.inc(1);
    }
    level_pb.finish_and_clear();

    Ok(results)
}

/// Compute sort dimensions for each level, linearly spaced from
/// finest to coarsest (fine→coarse). Duplicate dimensions are
/// removed so that extra levels don't repeat the same partitioning.
fn compute_level_sort_dims(finest_sort_dim: usize, num_levels: usize) -> Vec<usize> {
    if num_levels <= 1 {
        return vec![finest_sort_dim];
    }
    let coarsest = DEFAULT_COARSEST_SORT_DIM.min(finest_sort_dim);
    let mut dims = Vec::with_capacity(num_levels);
    for level in 0..num_levels {
        // t goes from 0 (finest) to 1 (coarsest)
        let t = level as f32 / (num_levels - 1) as f32;
        let dim = finest_sort_dim as f32 - t * (finest_sort_dim - coarsest) as f32;
        let dim = dim.round() as usize;
        if dims.last() != Some(&dim) {
            dims.push(dim);
        }
    }
    dims
}

/// Compute the mapping from fine group indices to coarse group indices.
///
/// Each fine group's binary code is masked to `coarse_dim` bits to
/// produce its coarse code. Unique coarse codes are assigned
/// consecutive indices.
fn compute_fine_to_coarse_mapping(
    group_to_cols: &[Vec<usize>],
    fine_codes: &[usize],
    coarse_dim: usize,
) -> (Vec<usize>, usize) {
    let coarse_mask = (1_usize << coarse_dim) - 1;

    // For each fine group, look up binary code from any member column
    let coarse_codes: Vec<usize> = group_to_cols
        .iter()
        .map(|cols| fine_codes[cols[0]] & coarse_mask)
        .collect();

    // Unique coarse codes → consecutive indices
    let mut unique_coarse: Vec<usize> = coarse_codes.to_vec();
    unique_coarse.sort_unstable();
    unique_coarse.dedup();
    let num_coarse = unique_coarse.len();

    let coarse_to_idx: HashMap<usize, usize> = unique_coarse
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect();

    let fine_to_coarse: Vec<usize> = coarse_codes.iter().map(|c| coarse_to_idx[c]).collect();

    (fine_to_coarse, num_coarse)
}

/// Agglomerate fine-level statistics into coarser groups.
///
/// Sums columns of all group-indexed matrices (`observed_sum_ds`,
/// `imputed_sum_ds`, `residual_sum_ds`, `size_s`, `n_bs`) according
/// to the merge mapping. `observed_sum_db` is copied unchanged
/// (batch-marginal).
fn merge_stat(
    fine_stat: &CollapsedStat,
    fine_to_coarse: &[usize],
    num_coarse_groups: usize,
) -> CollapsedStat {
    let num_genes = fine_stat.num_genes();
    let num_batches = fine_stat.num_batches();
    let mut coarse = CollapsedStat::new(num_genes, num_coarse_groups, num_batches);

    for (fine_g, &coarse_g) in fine_to_coarse.iter().enumerate() {
        coarse
            .observed_sum_ds
            .column_mut(coarse_g)
            .add_assign(&fine_stat.observed_sum_ds.column(fine_g));
        coarse
            .imputed_sum_ds
            .column_mut(coarse_g)
            .add_assign(&fine_stat.imputed_sum_ds.column(fine_g));
        coarse
            .residual_sum_ds
            .column_mut(coarse_g)
            .add_assign(&fine_stat.residual_sum_ds.column(fine_g));
        coarse.size_s[coarse_g] += fine_stat.size_s[fine_g];
        for b in 0..num_batches {
            coarse.n_bs[(b, coarse_g)] += fine_stat.n_bs[(b, fine_g)];
        }
    }

    coarse.observed_sum_db.copy_from(&fine_stat.observed_sum_db);
    coarse
}

/// Multi-level collapsing trait.
///
/// `LevelOutput` is `CollapsedOut` for `SparseIoVec` (single layer)
/// and `Vec<CollapsedOut>` for `SparseIoStack` (one per layer).
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
        let knn = params.knn_super_cells;
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
        // Opt-in refinement path: BBKNN + Poisson DC-SBM over super-cells
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
            };
            return refine_and_collect_single_layer(self, proj_kn, &ctx);
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

        // Batch correction: super-cell matching across batches
        if num_batches >= 2 {
            self.collect_batch_stat(&mut fine_stat)?;

            info!("Building super-cells ...");
            let super_cells = build_super_cells(self, proj_kn, num_features)?;
            info!(
                "Built {} super-cells, matching with knn={} ...",
                super_cells.layout.cell_counts.len(),
                knn
            );
            let batch_knn = self
                .batch_knn_lookup()
                .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
            collect_matched_stat_coarse(
                &super_cells.layout,
                &super_cells.gene_sums,
                &super_cells.layout.super_cell_to_group,
                batch_knn.as_slice(),
                knn,
                &mut fine_stat,
            )?;
        }

        // Optimize finest level
        info!("Optimizing parameters ...");
        let result = optimize(&fine_stat, (1.0, 1.0), opt_iter)?;
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

            info!("Optimizing parameters ...");
            let coarse_result = optimize(&coarse_stat, (1.0, 1.0), level_opt_iter)?;
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
        let knn = params.knn_super_cells;
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
            for layer in self.stack.iter() {
                let num_features = layer.num_rows();
                let num_groups = group_to_cols.len();
                let mut stat = CollapsedStat::new(num_features, num_groups, 0);
                layer.collect_basic_stat(&mut stat)?;
                layer_results.push(optimize(&stat, (1.0, 1.0), opt_iter)?);
                fine_stats.push(stat);
            }

            let mut results = vec![layer_results];

            // Agglomeratively merge for coarser levels
            let mut prev_stats = fine_stats;
            let mut prev_group_to_cols = group_to_cols.clone();

            for &level_sort_dim in level_dims.iter().skip(1) {
                let level_opt_iter = (opt_iter / 2).max(10);
                let (fine_to_coarse, num_coarse) = compute_fine_to_coarse_mapping(
                    &prev_group_to_cols,
                    &fine_codes,
                    level_sort_dim,
                );

                let mut layer_results = Vec::with_capacity(num_layers);
                let mut coarse_stats = Vec::with_capacity(num_layers);
                for prev_stat in prev_stats.iter() {
                    let coarse_stat = merge_stat(prev_stat, &fine_to_coarse, num_coarse);
                    layer_results.push(optimize(&coarse_stat, (1.0, 1.0), level_opt_iter)?);
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

        // Build shared super-cell layout ONCE at finest level
        info!(
            "Level 1/{}: sort_dim={}, {} groups (finest — full computation)",
            level_dims.len(),
            finest_dim,
            num_groups
        );
        let layout = build_super_cell_layout(group_to_cols, &col_to_batch, proj_kn)?;
        let num_sc = layout.cell_counts.len();
        info!(
            "Built {} super-cells, matching with knn={} ...",
            num_sc, knn
        );

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
            let gene_sums = collect_super_cell_gene_sums(
                layer,
                group_to_cols,
                &col_to_batch,
                &layout.bg_to_sc,
                num_sc,
            )?;

            // Match across batches using shared layout + layer gene sums
            let batch_knn = layer
                .batch_knn_lookup()
                .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
            collect_matched_stat_coarse(
                &layout,
                &gene_sums,
                &layout.super_cell_to_group,
                batch_knn.as_slice(),
                knn,
                &mut stat,
            )?;

            // Optimize finest-level parameters
            layer_results.push(optimize(&stat, (1.0, 1.0), opt_iter)?);
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

                info!("Layer {}/{}: optimizing ...", d + 1, num_layers);
                layer_results.push(optimize(&coarse_stat, (1.0, 1.0), level_opt_iter)?);
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
