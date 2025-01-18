use crate::common::*;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::SparseIoVec;

use indicatif::ParallelProgressIterator;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
pub trait RandomProjectionOps {
    ///
    /// Create K x ncol projection by concatenating data across
    /// columns and returns the random basis matrix `D` x `target_dim`
    /// and the projection result `target_dim` x `N`
    ///
    /// # Arguments
    /// * `target_dim`: target dimensionality
    /// * `block_size`: block size for parallel computation
    ///
    fn project_cbind(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<(Mat, Mat)>;
}

impl RandomProjectionOps for SparseIoVec {
    fn project_cbind(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<(Mat, Mat)> {
        let nrows = self.num_rows()?;
        let ncols = self.num_columns()?;
        let block_size = block_size.unwrap_or(DEFAULT_BLOCK_SIZE);
        let jobs = create_jobs(block_size, ncols);

        let mut proj_kn = Mat::zeros(target_dim, ncols);
        let arc_proj_kn = Arc::new(Mutex::new(&mut proj_kn));
        let basis_dk = Mat::rnorm(nrows, target_dim);

        jobs.par_iter()
            .progress_count(ncols as u64)
            .for_each(|&(lb, ub)| {
                let mut xx_dm = self
                    .read_cells_csc(lb..ub)
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

        Ok((basis_dk, proj_kn))
    }
}
