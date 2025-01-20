use crate::common::*;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::SparseIoVec;

use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::dmatrix_util::*;

use indicatif::ParallelProgressIterator;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
pub struct RandProjOut {
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
    ) -> anyhow::Result<RandProjOut>;

    /// Assign each column/cell to some group.
    ///
    /// # Arguments
    /// * `proj_kn` - random projection matrix
    /// * `num_features` - number of features if `proj_kn` provided
    ///
    fn assign_columns_to_samples(
        &mut self,
        proj_kn: Option<&Mat>,
        num_features: Option<usize>,
    ) -> anyhow::Result<()>;
}

impl RandProjOps for SparseIoVec {
    /// Assign each column/cell to random sample by binary encoding
    /// # Arguments
    /// * `proj_kn` - random projection matrix
    /// * `num_features` - number of features if `proj_kn` provided
    fn assign_columns_to_samples(
        &mut self,
        proj_kn: Option<&Mat>,
        num_features: Option<usize>,
    ) -> anyhow::Result<()> {
        let proj_kn = match proj_kn {
            Some(x) => x,
            None => match num_features {
                Some(kk) => &Mat::rnorm(kk, self.num_columns()?),
                None => {
                    return Err(anyhow::anyhow!(
                        "either `proj_kn` or  `num_features` should be provided"
                    ))
                }
            },
        };

        let kk = proj_kn.nrows();
        let nn = proj_kn.ncols();

        if nn != self.num_columns()? {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

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

        self.assign_groups(binary_codes.data.as_vec().clone());
        Ok(())
    }

    fn project_columns(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<RandProjOut> {
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

        Ok(RandProjOut {
            basis: basis_dk,
            proj: proj_kn,
        })
    }
}
