#![allow(dead_code)]

use crate::normalization::NormalizeDistance;
use data_beans::sparse_data_visitors::*;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ProgressIterator;
use log::{info, warn};
use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::Inference;
use matrix_param::traits::*;
use matrix_util::traits::*;
use matrix_util::utils::partition_by_membership;
use std::sync::{Arc, Mutex};

pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

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

    fn collect_basic_stat(
        &self,
        sample_to_cols: &[Vec<usize>],
        stat: &mut CollapsedStat,
    ) -> anyhow::Result<()>;

    fn collect_batch_stat(
        &self,
        sample_to_cols: &[Vec<usize>],
        stat: &mut CollapsedStat,
    ) -> anyhow::Result<()>;

    fn collect_matched_stat(
        &self,
        sample_to_cells: &[Vec<usize>],
        knn_batches: usize,
        knn_cells: usize,
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
        let kk = proj_kn.nrows();

        info!("SVD on the projection matrix with k = {} ...", kk);
        let (_, _, q_nk) = proj_kn.rsvd(kk)?;

        let mut proj_kn = q_nk.transpose();

        let (lb, ub) = (-4., 4.);
        info!(
            "Clamping values within [{}, {}] after standardization",
            lb, ub
        );
        proj_kn.scale_columns_inplace();
        proj_kn.iter_mut().for_each(|x| {
            *x = x.clamp(lb, ub);
        });
        proj_kn.scale_columns_inplace();

        info!("creating batch-specific HNSW maps ...");
        self.register_batches_dmatrix(&proj_kn, col_to_batch)?;

        info!(
            "partitioned {} columns to {} batches",
            self.num_columns()?,
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

        let num_features = self.num_rows()?;
        let num_groups = group_to_cols.len();
        let num_batches = self.num_batches();

        let mut stat = CollapsedStat::new(num_features, num_groups, num_batches);
        info!("basic statistics across {} groups", num_groups);
        self.collect_basic_stat(group_to_cols, &mut stat)?;

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
                    .filter_map(|b| batch_name_map.get(b)).copied()
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

            self.collect_batch_stat(group_to_cols, &mut stat)?;

            info!(
                "counterfactual inference across {} batches over {} samples",
                num_batches, num_groups,
            );

            let knn_batches = knn_batches.unwrap_or(2);
            let knn_cells = knn_cells.unwrap_or(DEFAULT_KNN);

            self.collect_matched_stat(
                group_to_cols,
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

    fn collect_basic_stat(
        &self,
        sample_to_cells: &[Vec<usize>],
        stat: &mut CollapsedStat,
    ) -> anyhow::Result<()> {
        self.visit_columns_by_group(
            sample_to_cells,
            &collect_basic_stat_visitor,
            &EmptyArg {},
            stat,
        )
    }

    fn collect_batch_stat(
        &self,
        sample_to_cells: &[Vec<usize>],
        stat: &mut CollapsedStat,
    ) -> anyhow::Result<()> {
        self.visit_columns_by_group(
            sample_to_cells,
            &collect_batch_stat_visitor,
            &EmptyArg {},
            stat,
        )
    }

    fn collect_matched_stat(
        &self,
        sample_to_cells: &[Vec<usize>],
        knn_batches: usize,
        knn_cells: usize,
        reference_indices: Option<&[usize]>,
        stat: &mut CollapsedStat,
    ) -> anyhow::Result<()> {
        self.visit_columns_by_group(
            sample_to_cells,
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
        None => data_vec.read_neighbouring_columns_csc(
            cells.iter().cloned(),
            knn_batches,
            knn_cells,
            true,
            None,
        )?,
    };

    // Normalize distance for each source cell and take a
    // weighted average of the matched vectors using this
    // weight vector
    let norm_target = 2_f32.ln();
    let source_column_groups = partition_by_membership(&source_columns, None);

    ////////////////////////////////////////////////////////
    // zhat[g,j]  =  sum_k w[j,k] * z[g,k] / sum_k w[j,k] //
    // zsum[g,s]  =  sum_j zhat[g,j]                      //
    ////////////////////////////////////////////////////////

    let mut stat = arc_stat.lock().expect("lock stat");

    for (_, y0_pos) in source_column_groups.iter() {
        let weights = y0_pos
            .iter()
            .map(|&cell| euclidean_distances[cell])
            .normalized_exp(norm_target);

        let denom = weights.iter().sum::<f32>();

        y0_pos.iter().zip(weights.iter()).for_each(|(&k, &w)| {
            let y0 = y0_matched.get_col(k).expect("k missing");
            let y0_rows = y0.row_indices();
            let y0_vals = y0.values();
            y0_rows.iter().zip(y0_vals.iter()).for_each(|(&gene, &z)| {
                stat.zsum_ds[(gene, sample)] += z * w / denom;
            });
        });
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
            stat.ysum_ds[(gene, sample)] += y;
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
            stat.ysum_db[(gene, b)] += y;
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

        let mut mu_adj_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut mu_resid_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut gamma_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
        let mut delta_param = GammaMatrix::new((num_genes, num_batches), a0, b0);

        (0..num_iter).progress().for_each(|_opt_iter| {
            #[cfg(debug_assertions)]
            {
                debug!("iteration: {}", &_opt_iter);
            }

            // shared component (mu_ds)
            //
            // y_sum_ds + z_sum_ds
            // -----------------------------------------
            // sum_b delta_db * n_bs + gamma_ds .* size_s

            let gamma_ds = gamma_param.posterior_mean();
            let delta_db = delta_param.posterior_mean();

            denom_ds.copy_from(gamma_ds);
            denom_ds.row_iter_mut().for_each(|mut row| {
                row.component_mul_assign(&stat.size_s.transpose());
            });
            denom_ds += delta_db * &stat.n_bs;

            mu_adj_param.update_stat(&(&stat.ysum_ds + &stat.zsum_ds), &denom_ds);
            mu_adj_param.calibrate();

            let mu_ds = mu_adj_param.posterior_mean();

            // z-specific component (gamma_ds)
            //
            // z_sum_ds
            // -----------------------------------
            // mu_ds .* size_s

            denom_ds.copy_from(mu_ds);
            denom_ds.row_iter_mut().for_each(|mut row| {
                row.component_mul_assign(&stat.size_s.transpose());
            });

            gamma_param.update_stat(&stat.zsum_ds, &denom_ds);
            gamma_param.calibrate();

            // batch-specific effect (delta_db)
            //
            // y_sum_db
            // ---------------------
            // sum_s mu_ds * n_bs

            delta_param.update_stat(&stat.ysum_db, &(mu_ds * &stat.n_bs.transpose()));
            delta_param.calibrate();
        });

        // Just take the residuals of ysum
        //
        // y_sum_ds
        // -----------------------
        // mu_ds .* (1_d * size_s')
        {
            denom_ds =
                nalgebra::DVector::<f32>::from_element(num_genes, 1_f32) * stat.size_s.transpose();

            mu_resid_param.update_stat(
                &stat.ysum_ds,
                &denom_ds.component_mul(mu_adj_param.posterior_mean()),
            );
            mu_resid_param.calibrate();
        };

        // Take the observed mean
        {
            let denom_ds: nalgebra::DMatrix<f32> =
                nalgebra::DVector::<f32>::from_element(num_genes, 1_f32) * stat.size_s.transpose();
            mu_param.update_stat(&stat.ysum_ds, &denom_ds);
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
        let denom_ds: nalgebra::DMatrix<f32> =
            nalgebra::DVector::<f32>::from_element(num_genes, 1_f32) * stat.size_s.transpose();
        mu_param.update_stat(&stat.ysum_ds, &denom_ds);
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
pub struct CollapsedStat {
    pub ysum_ds: nalgebra::DMatrix<f32>, // observed sum within each sample
    pub zsum_ds: nalgebra::DMatrix<f32>, // counterfactual sum within each sample
    pub size_s: nalgebra::DVector<f32>,  // sample s size
    pub ysum_db: nalgebra::DMatrix<f32>, // divergence numerator
    pub n_bs: nalgebra::DMatrix<f32>,    // batch-specific sample size
}

impl CollapsedStat {
    pub fn new(ngene: usize, nsample: usize, nbatch: usize) -> Self {
        Self {
            ysum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            zsum_ds: nalgebra::DMatrix::<f32>::zeros(ngene, nsample),
            size_s: nalgebra::DVector::<f32>::zeros(nsample),
            ysum_db: nalgebra::DMatrix::<f32>::zeros(ngene, nbatch),
            n_bs: nalgebra::DMatrix::<f32>::zeros(nbatch, nsample),
        }
    }

    pub fn num_genes(&self) -> usize {
        self.ysum_ds.nrows()
    }

    pub fn num_samples(&self) -> usize {
        self.ysum_ds.ncols()
    }

    pub fn num_batches(&self) -> usize {
        self.ysum_db.ncols()
    }

    pub fn clear(&mut self) {
        self.ysum_ds.fill(0_f32);
        self.zsum_ds.fill(0_f32);
        self.ysum_db.fill(0_f32);
        self.size_s.fill(0_f32);
        self.n_bs.fill(0_f32);
    }
}
