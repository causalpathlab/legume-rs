use crate::common::*;
use asap_data::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::dmatrix_util::*;
use matrix_util::traits::*;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
pub struct RandColProjOut {
    pub basis: Mat,
    pub proj: Mat,
}

#[allow(dead_code)]
pub struct RandRowProjOut {
    pub basis: Mat,
    pub proj: Mat,
}

#[allow(dead_code)]
pub trait RandProjOps {
    ///
    /// Create K x ncol projection by concatenating data across
    /// columns and returns the random basis matrix `D` x `target_dim`
    /// and the projection result `target_dim` x `N`
    ///
    /// # Arguments
    /// * `target_dim`: target dimensionality
    /// * `block_size`: block size for parallel computation
    ///
    fn project_columns(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<RandColProjOut>;

    /// Create `target_dim` x `nrows`/`D` projection by concatenating data
    /// across rows and returns the random basis matrix `ncols`/`N` x
    /// `target_dim` and the projection result `target_dim` x `D`
    ///
    /// # Arguments
    /// * `target_dim`: target dimensionality
    /// * `block_size`: block size for parallel computation
    ///
    fn project_rows(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<RandRowProjOut>;

    /// Assign each column/cell to random sample by binary encoding
    ///
    /// # Arguments
    ///
    /// * `proj_kn` - random projection matrix (feature x column/cell)
    ///
    /// * `num_features` - number of top features (if None, use all of them)
    ///
    fn assign_columns_to_samples(
        &mut self,
        proj_kn: &Mat,
        num_features: Option<usize>,
    ) -> anyhow::Result<usize>;
}

impl RandProjOps for SparseIoVec {
    fn project_columns(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<RandColProjOut> {
        let nrows = self.num_rows()?;
        let ncols = self.num_columns()?;
        let block_size = block_size.unwrap_or(DEFAULT_BLOCK_SIZE);
        let jobs = create_jobs(block_size, ncols);

        let mut proj_kn = Mat::zeros(target_dim, ncols);
        let arc_proj_kn = Arc::new(Mutex::new(&mut proj_kn));
        let basis_dk = Mat::rnorm(nrows, target_dim);

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .for_each(|&(lb, ub)| {
                let mut xx_dm = self
                    .read_columns_csc(lb..ub)
                    .expect("failed to retrieve data");

                xx_dm.normalize_columns_inplace();
                let mut chunk = (xx_dm.transpose() * &basis_dk).transpose();
                chunk.scale_columns_inplace();

                {
                    arc_proj_kn
                        .lock()
                        .expect("failed to lock proj")
                        .columns_range_mut(lb..ub)
                        .copy_from(&chunk);
                }
            });

        Ok(RandColProjOut {
            basis: basis_dk,
            proj: proj_kn,
        })
    }

    fn project_rows(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<RandRowProjOut> {
        let nrows = self.num_rows()?;
        let ncols = self.num_columns()?;
        let block_size = block_size.unwrap_or(DEFAULT_BLOCK_SIZE);
        let jobs = create_jobs(block_size, ncols);

        let mut proj_kd = Mat::zeros(target_dim, nrows);

        let arc_proj_kd = Arc::new(Mutex::new(&mut proj_kd));

        let basis_nk = Mat::rnorm(ncols, target_dim);

        let denom = jobs.len() as f32;

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .for_each(|&(lb, ub)| {
                let basis = basis_nk.rows_range(lb..ub);

                let mut xx = self
                    .read_columns_csc(lb..ub)
                    .expect("failed to retrieve data");

                xx.normalize_columns_inplace();

                let chunk: Mat = (xx * &basis / denom).transpose();

                {
                    arc_proj_kd
                        .lock()
                        .expect("failed to lock proj")
                        .column_iter_mut()
                        .zip(chunk.column_iter())
                        .for_each(|(mut proj_col, chunk_col)| {
                            proj_col += chunk_col;
                        });
                }
            });

        proj_kd.scale_columns_inplace();

        Ok(RandRowProjOut {
            basis: basis_nk,
            proj: proj_kd,
        })
    }

    /// Assign each column/cell to a sample by sorting through binary
    /// encoding
    ///
    /// # Arguments
    ///
    /// * `proj_kn` - random projection matrix (feature x column/cell)
    ///
    /// * `num_features` - number of top features (if None, use all of them)
    ///
    fn assign_columns_to_samples(
        &mut self,
        proj_kn: &Mat,
        num_sorting_features: Option<usize>,
    ) -> anyhow::Result<usize> {
        let nn = proj_kn.ncols();
        if nn != self.num_columns()? {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        let target_kk = num_sorting_features.unwrap_or(proj_kn.nrows());
        let kk = proj_kn.nrows().min(target_kk);

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

        let max_group = binary_codes.max();
        self.assign_groups(binary_codes.data.as_vec().clone());
        Ok(max_group + 1)
    }
}
