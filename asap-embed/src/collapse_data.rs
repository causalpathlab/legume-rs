use crate::common::*;
use crate::normalization::*;
use asap_data::sparse_io_vector::SparseIoVec;

use indicatif::ParallelProgressIterator;
use matrix_util::traits::MatOps;
use std::sync::{Arc, Mutex};

use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::dmatrix_util::*;
use matrix_util::traits::*;

use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::*;

use std::collections::HashMap;

pub const DEFAULT_KNN: usize = 10;

/// Assign each column/cell to random sample by binary encoding
/// # Arguments
/// * `proj_kn` - random projection matrix
/// # Returns
/// * `Vec<usize>` - sorted column indices
#[allow(dead_code)]
pub fn cells_to_samples_by_proj(proj_kn: &Mat) -> anyhow::Result<Vec<usize>> {
    let kk = proj_kn.nrows();
    let nn = proj_kn.ncols();
    let (_, _, mut q_nk) = proj_kn.rsvd(kk)?;
    q_nk.scale_columns_inplace();

    let mut binary_codes = DVector::<usize>::zeros(nn);
    for k in 0..kk {
        let binary_shift = |x: f32| -> usize {
            if x > 0.0 {
                1 << k
            } else {
                0
            }
        };
        binary_codes += q_nk.column(k).map(binary_shift);
    }

    Ok(binary_codes.data.as_vec().clone())
}

#[allow(dead_code)]
pub trait CollapsingOps {
    /// Collapse columns/cells into samples as allocated by
    /// `cell_to_sample` (`Vec<usize>`)
    ///
    /// # Arguments
    /// * `cell_to_sample` - map: cell -> sample
    /// * `cells_per_sample` - number of cells per sample (None: no down sampling)
    /// * `knn` - number of nearest neighbors for building HNSW graph (default: 10)
    ///
    fn collapse_columns(
        &self,
        cell_to_sample: &Vec<usize>,
        cells_per_sample: Option<usize>,
        knn: Option<usize>,
    ) -> anyhow::Result<()>;

    /// Register batch information and build a `HnswMap` object for
    /// each batch for fast nearest neighbor search within each batch
    /// and store them in the `SparseIoVec`
    ///
    /// # Arguments
    /// * `proj_kn` - random projection matrix
    /// * `cell_to_batch` - map: cell -> batch
    fn register_batches(&mut self, proj_kn: &Mat, cell_to_batch: &Vec<usize>)
        -> anyhow::Result<()>;
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

    fn collapse_columns(
        &self,
        cell_to_sample: &Vec<usize>,
        cells_per_sample: Option<usize>,
        knn: Option<usize>,
    ) -> anyhow::Result<()> {
        // Down sampling if needed
        println!("Partitioning {} cells into samples", cell_to_sample.len());
        let sample_to_cells = partition_by_membership(cell_to_sample, cells_per_sample);

        let num_genes = self.num_rows()?;
        let num_samples = sample_to_cells.len();
        let num_batches = self.num_batches();

        let mut stat = Stat::new(num_genes, num_samples, num_batches);
        let arc_stat = Arc::new(Mutex::new(&mut stat));

        /////////////////////////////////
        // accumulate basic statistics //
        /////////////////////////////////

        let num_jobs = num_samples as u64;

        println!("collect basic statistics across {} samples", num_samples);

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

        if num_batches > 1 {
            ///////////////////////////////////////
            // collect counterfactual statistics //
            ///////////////////////////////////////

            println!(
                "collect counterfactual statistics across {} batches over {} samples",
                num_samples, num_batches
            );

            let knn = knn.unwrap_or(DEFAULT_KNN);

            sample_to_cells
                .par_iter()
                .progress_count(num_jobs)
                .for_each(|(&sample, cells)| {
                    let mut stat = arc_stat.lock().expect("failed to lock stat");

                    let batches = self.get_batch_membership(cells.iter().cloned());

                    for b in batches {
                        stat.n_bs[(b, sample)] += 1_f32;
                    }

                    // 1. read source cells -- use dense fo
                    let mut yy = self
                        .read_cells_dmatrix(cells.iter().cloned())
                        .expect("failed to read cells");
                    yy.normalize_columns_inplace();

                    let positions: HashMap<_, _> =
                        cells.iter().enumerate().map(|(i, c)| (c, i)).collect();

                    // 2. read matched cells and distance from their source
                    let mut matched_triplets: Vec<(usize, usize, f32)> = vec![];
                    let mut source_columns: Vec<usize> = vec![];
                    let mut mean_square_distances = vec![];
                    let mut tot_ncells_matched = 0;

                    for target_batch in 0..num_batches {
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
                        let mut zz = nalgebra_sparse::CscMatrix::<f32>::from_nonzero_triplets(
                            num_genes, ncol, triplets,
                        )
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
                                        .map(|(&i, &y)| (y - y_j[i]) * (y - y_j[i]))
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

                    let mut zz_full = nalgebra_sparse::CscMatrix::<f32>::from_nonzero_triplets(
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
        } // if num_batches > 1

        /////////////////////////////
        // Resolve mean parameters //
        /////////////////////////////

        let (a0, b0) = (1_f32, 1_f32);

        if num_batches > 1 {
            ////////////////////////////////////////////////////////////////////
            // optimize three types of gamma parameters: mu, gamma, and delta //
            ////////////////////////////////////////////////////////////////////

            let mu_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
            let gamma_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
            let delta_param = GammaMatrix::new((num_genes, num_batches), a0, b0);

            stat.gamma_ds.fill(1.0);
            stat.delta_db.fill(1.0);

            {
                // shared component (mu_ds)
                //
                // y_sum_ds + z_sum_ds
                // -----------------------------------------
                // sum_b delta_db * n_bs + gamma_ds * size_s
                //

                // z-specific component (gamma_ds)
                //
                // z_sum_ds
                // -----------------------------------
                // mu_ds * size_s

		// delta_db
		//
		// sum_s z_sum_ds * prob_bs
		// ---------------------
		// sum_s mu_ds * n_bs
            }
        } else {
            ////////////////////////////////////////////////////////////
            // pseudobulk estimation without considering batch effect //
            ////////////////////////////////////////////////////////////

            let mut mu_param = GammaMatrix::new((num_genes, num_samples), a0, b0);
            let denom_ds: Mat = DVec::from_element(num_genes, 1_f32) * stat.size_s.transpose();
            mu_param.update_stat(&stat.ysum_ds, &denom_ds);
            mu_param.calibrate();
        }

        Ok(())
    }
}

struct Stat {
    pub mu_ds: Mat,          // observed mean
    pub ysum_ds: Mat,        // observed sum within each sample
    pub gamma_ds: Mat,       // residual mean
    pub zsum_ds: Mat,        // counterfactual sum within each sample
    pub size_s: DVec,        // sample s size
    pub delta_db: Mat,       // divergence mean
    pub delta_num_db: Mat,   // divergence numerator
    pub delta_denom_db: Mat, // divergence denominator
    pub n_bs: Mat,           // batch-specific sample size
    pub prob_bs: Mat,        // P(a cell in a batch in a sample)
}

impl Stat {
    pub fn new(ngene: usize, nsample: usize, nbatch: usize) -> Self {
        Self {
            mu_ds: Mat::zeros(ngene, nsample),
            ysum_ds: Mat::zeros(ngene, nsample),
            gamma_ds: Mat::zeros(ngene, nsample),
            zsum_ds: Mat::zeros(ngene, nsample),
            size_s: DVec::zeros(nsample),
            delta_db: Mat::zeros(ngene, nbatch),
            delta_num_db: Mat::zeros(ngene, nbatch),
            delta_denom_db: Mat::zeros(ngene, nbatch),
            n_bs: Mat::zeros(nbatch, nsample),
            prob_bs: Mat::zeros(nbatch, nsample),
        }
    }
}
