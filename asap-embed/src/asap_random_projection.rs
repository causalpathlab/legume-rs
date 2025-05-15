#![allow(dead_code)]

use crate::asap_embed_common::*;
use asap_data::sparse_data_visitors::*;
use asap_data::sparse_io_vector::SparseIoVec;
use std::sync::{Arc, Mutex};

use log::{info, warn};
use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::traits::*;
use matrix_util::utils::*;
use nalgebra::DVector;

pub struct RandColProjOut {
    pub basis: Mat,
    pub proj: Mat,
}

pub struct RandRowProjOut {
    pub basis: Mat,
    pub proj: Mat,
}

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

    ///
    /// Create K x ncol projection by concatenating data across
    /// columns and returns the random basis matrix `D` x `target_dim`
    /// and the projection result `target_dim` x `N`
    ///
    /// # Arguments
    /// * `target_dim`: target dimensionality
    /// * `block_size`: block size for parallel computation
    ///
    fn project_columns_with_batch_correction<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&Vec<T>>,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

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

/// column-wise visitor for random projection
fn visit_rand_proj_columnwise(
    job: (usize, usize),
    data_vec: &SparseIoVec,
    basis_dk: &Mat,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) {
    let (lb, ub) = job;
    let mut xx_dm = data_vec.read_columns_csc(lb..ub).unwrap();
    xx_dm.normalize_columns_inplace();
    let chunk = (xx_dm.transpose() * basis_dk).transpose();

    arc_proj_kn
        .lock()
        .unwrap()
        .columns_range_mut(lb..ub)
        .copy_from(&chunk);
}

impl RandProjOps for SparseIoVec {
    fn project_columns(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<RandColProjOut> {
        self.project_columns_with_batch_correction::<usize>(target_dim, block_size, None)
    }

    fn project_columns_with_batch_correction<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&Vec<T>>,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let nrows = self.num_rows()?;
        let ncols = self.num_columns()?;

        let mut proj_kn = Mat::zeros(target_dim, ncols);

        let basis_dk = Mat::rnorm(nrows, target_dim);

        self.visit_columns_by_jobs(
            &visit_rand_proj_columnwise,
            &basis_dk,
            &mut proj_kn,
            block_size,
        )?;

        if let Some(col_to_batch) = batch_membership {
            info!("adjusting batch biases ...");

            if col_to_batch.len() == ncols {
                let batches = partition_by_membership(col_to_batch, None);
                for (_, cols) in batches.iter() {
                    // columnwise processing is faster
                    let x_t_rows: Vec<_> = cols
                        .iter()
                        .map(|&j| proj_kn.column(j).transpose().into_owned())
                        .collect();

                    let mut x_t: Mat = Mat::from_rows(&x_t_rows);
                    x_t.centre_columns_inplace();
                    let xx: Mat = x_t.transpose();

                    cols.iter().zip(xx.column_iter()).for_each(|(&j, x_j)| {
                        proj_kn.column_mut(j).copy_from(&x_j);
                    });
                }
            } else {
                warn!(
                    "The batch membership size {} mismatches with the number of columns {} ...",
                    col_to_batch.len(),
                    ncols
                );
            }
        }

        proj_kn.scale_columns_inplace();

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

        Ok(RandColProjOut {
            basis: basis_dk,
            proj: proj_kn,
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
        let kk = proj_kn.nrows().min(target_kk).min(nn);

        let binary_codes = binary_sort_columns(proj_kn, kk)?;
        let max_group = *binary_codes
            .iter()
            .max()
            .ok_or(anyhow::anyhow!("unable to determine max element"))?;
        self.assign_groups(binary_codes);
        Ok(max_group + 1)
    }
}

/// Binarize the projection matrix and assign columns to some groups
///
/// # Arguments
/// * `proj_kn` - random projection matrix (feature x column/cell)
/// * `kk` - number of features
pub fn binary_sort_columns(proj_kn: &Mat, kk: usize) -> anyhow::Result<Vec<usize>> {
    // SVD to spread out the points
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
