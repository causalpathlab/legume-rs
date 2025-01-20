use crate::common::*;
use crate::normalization::*;
use asap_data::sparse_io_vector::SparseIoVec;
use matrix_param::traits::Inference;
use rayon::prelude::*;

use indicatif::ParallelProgressIterator;
use indicatif::ProgressIterator;
use matrix_util::traits::MatOps;
use std::sync::{Arc, Mutex};

use matrix_util::dmatrix_util::*;
use matrix_util::traits::*;

use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::*;

use std::collections::HashMap;

pub const DEFAULT_KNN: usize = 10;
pub const DEFAULT_OPT_ITER: usize = 100;

#[allow(dead_code)]
/// Given a feature/projection matrix (factor x cells), we assign each
/// cell to a sample and return pseudobulk (collapsed) matrices
///
/// (1) Register batches if needed (2) collapse columns/cells into samples
///
pub trait CollapsingOps {
    /// Collapse columns/cells into samples as allocated by
    /// `cell_to_sample` (`Vec<usize>`)
    ///
    /// # Arguments
    /// * `cells_per_group` - number of cells per sample (None: no down sampling)
    /// * `knn` - number of nearest neighbors for building HNSW graph (default: 10)
    /// * `num_opt_iter` - number of optimization iterations (default: 100)
    ///
    fn collapse_columns_as_assigned(
        &self,
        cells_per_group: Option<usize>,
        knn: Option<usize>,
        num_opt_iter: Option<usize>,
    ) -> anyhow::Result<CollapsingOut>;

    /// Register batch information and build a `HnswMap` object for
    /// each batch for fast nearest neighbor search within each batch
    /// and store them in the `SparseIoVec`
    ///
    /// # Arguments
    /// * `proj_kn` - random projection matrix
    /// * `cell_to_batch` - map: cell -> batch
    fn register_batches(&mut self, proj_kn: &Mat, cell_to_batch: &Vec<usize>)
        -> anyhow::Result<()>;

    fn collect_basic_stat(
        &self,
        sample_to_cells: &HashMap<usize, Vec<usize>>,
        stat: &mut CollapsingStat,
    );

    fn collect_batch_stat(
        &self,
        sample_to_cells: &HashMap<usize, Vec<usize>>,
        stat: &mut CollapsingStat,
    );

    fn collect_matched_stat(
        &self,
        sample_to_cells: &HashMap<usize, Vec<usize>>,
        knn: usize,
        stat: &mut CollapsingStat,
    );
}

impl CollapsingOps for SparseIoVec {
    /// Register batch information and build a `HnswMap` object for
    /// each batch for fast nearest neighbor search within each batch
    /// and store them in the `SparseIoVec`
    ///
    /// # Arguments
    /// * `proj_kn` - random projection matrix
    /// * `cell_to_batch` - map: cell -> batch
    fn register_batches(
        &mut self,
        proj_kn: &Mat,
        cell_to_batch: &Vec<usize>,
    ) -> anyhow::Result<()> {
        self.register_batches_dmatrix(proj_kn, cell_to_batch.clone())
    }

    ///
    /// Collapse columns/cells into samples as allocated by `assign_columns_to_samples`
    ///
    /// # Arguments
    /// * `cells_per_group` - number of cells per group (None: no down sampling)
    /// * `knn` - number of nearest neighbors for building HNSW graph (default: 10)
    /// * `num_opt_iter` - number of optimization iterations (default: 100)
    ///
    fn collapse_columns_as_assigned(
        &self,
        cells_per_group: Option<usize>,
        knn: Option<usize>,
        num_opt_iter: Option<usize>,
    ) -> anyhow::Result<CollapsingOut> {
        // Down sampling if needed

        let cell_to_sample: &Vec<usize> = self.take_groups().ok_or(anyhow::anyhow!(
            "The columns were not assigned before. Call `partition_columns`"
        ))?;

        let sample_to_cells = partition_by_membership(cell_to_sample, cells_per_group);

        let num_genes = self.num_rows()?;
        let num_samples = sample_to_cells.len();
        let num_batches = self.num_batches();

        let mut stat = CollapsingStat::new(num_genes, num_samples, num_batches);
        // (&mut stat).clear();
        // let arc_stat = Arc::new(Mutex::new(&mut stat));

        println!("basic statistics across {} samples", num_samples);
        self.collect_basic_stat(&sample_to_cells, &mut stat);

        if num_batches > 1 {
            println!(
                "batch-specific statistics across {} batches over {} samples",
                num_batches, num_samples
            );

            self.collect_batch_stat(&sample_to_cells, &mut stat);

            println!(
                "counterfactual inference across {} batches over {} samples",
                num_batches, num_samples,
            );

            let knn = knn.unwrap_or(DEFAULT_KNN);
            self.collect_matched_stat(&sample_to_cells, knn, &mut stat);
        } // if num_batches > 1

        /////////////////////////////
        // Resolve mean parameters //
        /////////////////////////////

        let (a0, b0) = (1_f32, 1_f32);
        optimize(&stat, (a0, b0), num_opt_iter.unwrap_or(DEFAULT_OPT_ITER))
    }

    fn collect_basic_stat(
        &self,
        sample_to_cells: &HashMap<usize, Vec<usize>>,
        stat: &mut CollapsingStat,
    ) {
        let num_samples = sample_to_cells.len();
        let num_jobs = num_samples as u64;
        let arc_stat = Arc::new(Mutex::new(stat));

        // ysum(g,s) = sum_j C(j,s) * Y(g,j)
        // size(s) = sum_j C(j,s)
        sample_to_cells
            .par_iter()
            .progress_count(num_jobs)
            .for_each(|(&sample, cells)| {
                let mut stat = arc_stat.lock().expect("failed to lock stat");
                let mut yy = self
                    .read_cells_csc(cells.iter().cloned())
                    .expect("failed to read cells");
                yy.normalize_columns_inplace();

                for y_j in yy.col_iter() {
                    let rows = y_j.row_indices();
                    let vals = y_j.values();
                    for (&gene, &y) in rows.iter().zip(vals.iter()) {
                        stat.size_s[sample] += 1_f32;
                        stat.ysum_ds[(gene, sample)] += y;
                    }
                }
            });
    }

    fn collect_batch_stat(
        &self,
        sample_to_cells: &HashMap<usize, Vec<usize>>,
        stat: &mut CollapsingStat,
    ) {
        let num_samples = sample_to_cells.len();
        let num_jobs = num_samples as u64;
        let arc_stat = Arc::new(Mutex::new(stat));

        // ysum(g,b) = sum_j sum_s C(j,s) * Y(g,j) * I(b,s)
        // n(b,s) = sum_j C(j,s) * I(b,s)
        sample_to_cells
            .par_iter()
            .progress_count(num_jobs)
            .for_each(|(&sample, cells)| {
                let mut stat = arc_stat.lock().expect("failed to lock stat");

                let batches = self.get_batch_membership(cells.iter().cloned());

                // 1. read source cells
                let mut yy = self
                    .read_cells_csc(cells.iter().cloned())
                    .expect("failed to read cells");
                yy.normalize_columns_inplace();

                yy.col_iter().zip(batches.iter()).for_each(|(y_j, &b)| {
                    let rows = y_j.row_indices();
                    let vals = y_j.values();
                    for (&gene, &y) in rows.iter().zip(vals.iter()) {
                        stat.ysum_db[(gene, b)] += y;
                    }
                    stat.n_bs[(b, sample)] += 1_f32;
                });
            });
    }

    fn collect_matched_stat(
        &self,
        sample_to_cells: &HashMap<usize, Vec<usize>>,
        knn: usize,
        stat: &mut CollapsingStat,
    ) {
        let num_genes = self.num_rows().expect("failed to get num genes");
        let num_samples = sample_to_cells.len();
        let num_batches = self.num_batches();
        let num_jobs = num_samples as u64;
        let arc_stat = Arc::new(Mutex::new(stat));

        // zsum(g,s) = sum_j C(j,s) * Z(g,j)
        sample_to_cells
            .par_iter()
            .progress_count(num_jobs)
            .for_each(|(&sample, cells)| {
                let mut stat = arc_stat.lock().expect("failed to lock stat");

                let positions: HashMap<_, _> =
                    cells.iter().enumerate().map(|(i, c)| (c, i)).collect();
                let mut matched_triplets: Vec<(usize, usize, f32)> = vec![];
                let mut source_columns: Vec<usize> = vec![];
                let mut mean_square_distances = vec![];
                let mut tot_ncells_matched = 0;

                let mut yy = self
                    .read_cells_dmatrix(cells.iter().cloned())
                    .expect("failed to read cells");
                yy.normalize_columns_inplace();

                for target_batch in 0..num_batches {
                    // match cells between source and target batches
                    let (_, ncol, triplets, source_cells_in_target) = self
                        .collect_matched_cells_triplets(
                            cells.iter().cloned(),
                            target_batch,
                            knn,
                            true,
                        )
                        .expect("failed to read matched cells");

                    matched_triplets.extend(
                        triplets
                            .iter()
                            .map(|(i, j, z_ij)| (*i, *j + tot_ncells_matched, *z_ij)),
                    );

                    // matched cells within this batch
                    let mut zz = CscMatrix::<f32>::from_nonzero_triplets(num_genes, ncol, triplets)
                        .expect("failed to build z matrix");
                    zz.normalize_columns_inplace();

                    let src_pos_in_target: Vec<usize> = source_cells_in_target
                        .iter()
                        .map(|c| {
                            *positions
                                .get(&c)
                                .expect("failed to identify the source position")
                        })
                        .collect();

                    let denom = zz.nrows() as f32;

                    mean_square_distances.extend(
                        // for each column of the matched matrix
                        // MSE(j,k) = sum_g (y[g),j] - z[g,k])^2 / sum_g 1
                        zz.col_iter()
                            .zip(src_pos_in_target.iter())
                            .map(|(z_j, &j)| {
                                // source column/cell
                                let y_j = yy.column(j);
                                // matched/target column/cell
                                let z_rows = z_j.row_indices();
                                let z_vals = z_j.values();

                                let y_tot = y_j.map(|x| x * x).sum();
                                // to avoid double counting
                                let y_overlap =
                                    z_rows.iter().map(|&i| y_j[i] * y_j[i]).sum::<f32>();
                                let delta_overlap = z_rows
                                    .iter()
                                    .zip(z_vals.iter())
                                    .map(|(&i, &z)| (z - y_j[i]) * (z - y_j[i]))
                                    .sum::<f32>();
                                (y_tot - y_overlap + delta_overlap) / denom
                            }),
                    );

                    source_columns.extend(src_pos_in_target);
                    tot_ncells_matched += ncol;
                } // for each target batch of step 2.

                ////////////////////////////////////////////////////
                // a full set of y vectors needed for this sample //
                ////////////////////////////////////////////////////

                let mut zz_full = CscMatrix::from_nonzero_triplets(
                    num_genes,
                    tot_ncells_matched,
                    matched_triplets,
                )
                .expect("failed to build y matrix");
                zz_full.normalize_columns_inplace();

                // 3. normalize distance for each source cell and
                // take a weighted average of the matched vectors
                // using this weight vector
                let norm_target = 2_f32.ln();
                let source_column_groups = partition_by_membership(&source_columns, None);

                ////////////////////////////////////////////////////////
                // zhat[g,j]  =  sum_k w[j,k] * z[g,k] / sum_k w[j,k] //
                // zsum[g,s]  =  sum_j zhat[g,j]                      //
                ////////////////////////////////////////////////////////

                for (_, z_pos) in source_column_groups.iter() {
                    let weights = z_pos
                        .iter()
                        .map(|&cell| mean_square_distances[cell])
                        .normalized_exp(norm_target);

                    let denom = weights.iter().sum::<f32>();

                    z_pos.iter().zip(weights.iter()).for_each(|(&z_pos, &w)| {
                        let z = zz_full.get_col(z_pos).unwrap();
                        let z_rows = z.row_indices();
                        let z_vals = z.values();
                        z_rows.iter().zip(z_vals.iter()).for_each(|(&gene, &z)| {
                            stat.zsum_ds[(gene, sample)] += z * w / denom;
                        });
                    });
                }
            }); // for each sample
    }
}

/// Optimize the mean parameters for three Gamma distributions
///
fn optimize(
    stat: &CollapsingStat,
    hyper: (f32, f32),
    num_iter: usize,
) -> anyhow::Result<CollapsingOut> {
    let (a0, b0) = hyper;
    let num_genes = stat.num_genes();
    let num_samples = stat.num_samples();
    let num_batches = stat.num_batches();
    let mut mu_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
    let mut gamma_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
    let mut delta_param = GammaMatrix::new((num_genes, num_batches), a0, b0);

    if num_batches > 1 {
        // temporary denominator
        let mut denom_ds = Mat::zeros(num_genes, num_samples);

        (0..num_iter).progress().for_each(|iter| {
            #[cfg(debug_assertions)]
            {
                println!("iteration: {}", &iter);
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

            mu_param.update_stat(&(&stat.ysum_ds + &stat.zsum_ds), &denom_ds);
            mu_param.calibrate();

            let mu_ds = mu_param.posterior_mean();

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

        Ok(CollapsingOut {
            mu: mu_param,
            gamma: Some(gamma_param),
            delta: Some(delta_param),
        })
    } else {
        let denom_ds: Mat = DVec::from_element(num_genes, 1_f32) * stat.size_s.transpose();
        mu_param.update_stat(&stat.ysum_ds, &denom_ds);
        mu_param.calibrate();
        Ok(CollapsingOut {
            mu: mu_param,
            gamma: None,
            delta: None,
        })
    }
}

/// output struct to make the model parameters more accessible
#[allow(dead_code)]
pub struct CollapsingOut {
    pub mu: GammaMatrix,
    pub gamma: Option<GammaMatrix>,
    pub delta: Option<GammaMatrix>,
}

/// a struct to hold the sufficient statistics for the model
#[allow(dead_code)]
pub struct CollapsingStat {
    pub ysum_ds: Mat, // observed sum within each sample
    pub zsum_ds: Mat, // counterfactual sum within each sample
    pub size_s: DVec, // sample s size
    pub ysum_db: Mat, // divergence numerator
    pub n_bs: Mat,    // batch-specific sample size
}

impl CollapsingStat {
    pub fn new(ngene: usize, nsample: usize, nbatch: usize) -> Self {
        Self {
            ysum_ds: Mat::zeros(ngene, nsample),
            zsum_ds: Mat::zeros(ngene, nsample),
            size_s: DVec::zeros(nsample),
            ysum_db: Mat::zeros(ngene, nbatch),
            n_bs: Mat::zeros(nbatch, nsample),
        }
    }
    #[allow(dead_code)]
    pub fn num_genes(&self) -> usize {
        self.ysum_ds.nrows()
    }

    #[allow(dead_code)]
    pub fn num_samples(&self) -> usize {
        self.ysum_ds.ncols()
    }

    #[allow(dead_code)]
    pub fn num_batches(&self) -> usize {
        self.ysum_db.ncols()
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.ysum_ds.fill(0_f32);
        self.zsum_ds.fill(0_f32);
        self.ysum_db.fill(0_f32);
        self.size_s.fill(0_f32);
        self.n_bs.fill(0_f32);
    }
}
