use asap_data::sparse_io::*;

use std::sync::Arc;
use std::sync::Mutex;
pub type Data = dyn SparseIo<IndexIter = Vec<usize>>;

const DEFAULT_BLOCK_SIZE: usize = 500;

pub struct RandProjVec<'a> {
    data_vec: &'a Vec<Arc<Data>>,
    block_size: usize,
    num_batch: usize,
    rand_basis_kd: Option<DMatrix<f32>>,
    rand_proj_kn: Option<DMatrix<f32>>,
    batch_glob_index: HashMap<usize, Vec<usize>>,
    batch_membership: Vec<usize>,
}

#[allow(dead_code)]
impl<'a> RandProjVec<'a> {
    ///
    /// Create a new instance of `RandProjVec`
    ///
    /// # Arguments
    /// * `_vec`: a vector of data sets to be concatenated (Arc<Data>). Each element contains a atomic reference to the data sets, so it can be shared across threads and cloned.
    /// * `_blk`: block size for parallel computation
    ///
    pub fn new(_vec: &'a Vec<Arc<Data>>, _blk: Option<usize>) -> anyhow::Result<Self> {
        let nb = _vec.len();
        let blk = _blk.unwrap_or(DEFAULT_BLOCK_SIZE);
        if nb < 1 {
            anyhow::bail!("no data set in the vector");
        }

        Ok(Self {
            data_vec: &_vec,
            block_size: blk,
            num_batch: nb,
            rand_basis_kd: None,
            rand_proj_kn: None,
            batch_glob_index: HashMap::new(),
            batch_membership: Vec::new(),
        })
    }

    ///
    /// Step 1: Sample random projection basis matrix
    /// # Arguments
    /// * `dim`: target dimensionality
    /// # Returns
    /// * a `kk x nrow` matrix
    ///
    pub fn step1_sample_basis_cbind(&mut self, dim: usize) -> anyhow::Result<()> {
        let first_data = &self.data_vec[0];

        let nrow = first_data.num_rows().ok_or_else(|| {
            anyhow::anyhow!(
                "can't figure out #rows in the data: {}",
                first_data.get_backend_file_name()
            )
        })?;

        let rand_basis_kd = Self::sample_basis_to_reduce_rows(nrow, dim)?;

        for (ii, data) in self.data_vec.iter().enumerate().skip(1) {
            let data_nrow = data.num_rows().ok_or_else(|| {
                anyhow::anyhow!(
                    "can't figure out #rows in the data: {}",
                    data.get_backend_file_name()
                )
            })?;
            if data_nrow != nrow {
                anyhow::bail!(
                    "#rows in the data #{} don't match with the first one ({})",
                    ii,
                    nrow
                );
            }
        }

        self.rand_basis_kd = Some(rand_basis_kd);

        Ok(())
    }

    ///
    /// Step 2: Create K x ncol projection by concatenating data across columns
    /// # Arguments
    /// * `rand_basis_kd`: random basis matrix from step 1
    ///
    pub fn step2_proj_cbind(&mut self) -> anyhow::Result<()> {
        if let Some(rand_basis_kd) = &self.rand_basis_kd {
            let num_batch = self.num_batch;
            let kk = rand_basis_kd.nrows();
            let nrow = rand_basis_kd.ncols();
            let ncol = self.data_vec.iter().fold(0, |ncols, data| -> usize {
                ncols + data.num_columns().expect("failed to figure out # cols")
            });

            // the results of random projection
            let mut rand_proj_kn: DMatrix<f32> = DMatrix::zeros(kk, ncol);
            let arc_rand_proj_kn = Arc::new(Mutex::new(&mut rand_proj_kn));

            // batch to global membership and vice versa
            let mut batch_glob_index: HashMap<usize, Vec<usize>> = HashMap::new();
            for b in 0..num_batch {
                batch_glob_index.insert(b, Vec::new());
            }
            let mut batch_membership: Vec<usize> = vec![0; ncol];
            let mut offset = 0;
            self.batch_glob_index.clear();
            self.batch_membership.clear();

            for (batch, data_batch) in self.data_vec.iter().enumerate() {
                anyhow::ensure!(data_batch.num_rows().unwrap() == nrow);
                let ncol_batch = data_batch.num_columns().unwrap();
                let nblock = (ncol_batch + self.block_size - 1) / self.block_size;

                /////////////////////////////////
                // populate projection results //
                /////////////////////////////////

                (0..nblock)
                    .into_par_iter()
                    .map(|block| {
                        let lb: usize = block * self.block_size;
                        let ub: usize = ((block + 1) * self.block_size).min(ncol_batch);
                        (lb, ub)
                    })
                    .for_each(|(lb, ub)| {
                        // This could be inefficient since we are populating a dense matrix
                        let xx_dm = data_batch
                            .read_columns_dmatrix((lb..ub).collect())
                            .expect("failed to read columns");

                        let yy_dm = Self::normalize_columns(&xx_dm);
                        let proj_block = rand_basis_kd * &yy_dm;

                        let (lb_glob, ub_glob) = (lb + offset, ub + offset);

                        {
                            let mut proj_kn = arc_rand_proj_kn.lock().expect("failed to lock proj");
                            proj_kn
                                .columns_range_mut(lb_glob..ub_glob)
                                .copy_from(&proj_block);
                        }
                    });

                ////////////////////////////
                // sorted out the indexes //
                ////////////////////////////

                (0..nblock).into_iter().for_each(|block| {
                    let lb: usize = block * self.block_size;
                    let ub: usize = ((block + 1) * self.block_size).min(ncol_batch);
                    let (lb_glob, ub_glob) = (lb + offset, ub + offset);
                    let globs = batch_glob_index.get_mut(&batch).expect("batch glob index");
                    for j in lb_glob..ub_glob {
                        batch_membership[j] = batch;
                        globs.push(j);
                    }
                });

                offset += ncol_batch;
            }

            self.rand_proj_kn = Some(rand_proj_kn);
            self.batch_glob_index.extend(batch_glob_index);
            self.batch_membership.extend(batch_membership);
            Ok(())
        } else {
            Err(anyhow::anyhow!("random basis matrix is not available"))
        }
    }

    // A helper function to normalize column-wise
    fn normalize_columns(xx_dm: &DMatrix<f32>) -> DMatrix<f32> {
        let mut yy_dm = xx_dm.clone();
        for mut xx_j in yy_dm.column_iter_mut() {
            let denom = xx_j.sum().max(1.0);
            xx_j /= denom;
        }
        yy_dm
    }

    // A helper function to sample a random basis matrix
    fn sample_basis_to_reduce_rows(nrow: usize, target_dim: usize) -> anyhow::Result<DMatrix<f32>> {
        let kk = target_dim.min(nrow);
        use rand::{thread_rng, Rng};
        use rand_distr::StandardNormal;

        let rvec: Vec<f32> = (0..(nrow * kk))
            .into_par_iter()
            .map_init(thread_rng, |rng, _| rng.sample(StandardNormal))
            .collect();

        Ok(DMatrix::from_vec(kk, nrow, rvec))
    }
}

// #[allow(dead_code)]
// pub fn collapse_columns_cbind(
//     data_vec: &Vec<Box<&Data>>,
//     target_dim: usize,
//     block_size: Option<usize>,
// ) -> anyhow::Result<()> {

//     // if num_batch < 1 {
//     //     anyhow::bail!("no data set in the vector");
//     // }

//     ///////////////////////////////////////////////////
//     // Step 1: Sample random projection basis matrix //
//     ///////////////////////////////////////////////////

//     // // We figure out the number of rows using the first data set
//     // let step1 = |dim: usize| -> anyhow::Result<Array2<f32>> {
//     //     let first_data = &data_vec[0];
//     //     let nrow = first_data.num_rows().expect("#rows");
//     //     let rand_basis_kd = sample_basis_to_reduce_rows(nrow, dim)?;
//     //     Ok(rand_basis_kd)
//     // };

//     // let rand_basis_kd = step1(target_dim)?;

//     // Gather dimensionality info

//     // let kk = rand_basis_kd.nrows();
//     // let nrow = rand_basis_kd.ncols();
//     // let ncol = data_vec.into_iter().fold(0, |ncols, data| -> usize {
//     //     ncols + data.num_columns().expect("failed to figure out # cols")
//     // });

//     ////////////////////////////////////////
//     // Step 2: Create K x ncol projection //
//     ////////////////////////////////////////

//     // let step2 =
//     //     |rand_basis_kd: &Array2<f32>| -> anyhow::Result<(Array2<f32>, HashMap<_, _>, Vec<_>)> {
//     //         let num_batch = data_vec.len();
//     //         // the results of random projection
//     //         let mut rand_proj_kn: Array2<f32> = Array2::zeros((kk, ncol));
//     //         let arc_rand_proj_kn = Arc::new(Mutex::new(&mut rand_proj_kn));

//     //         // batch to global membership and vice versa
//     //         let mut batch_glob_index: HashMap<usize, Vec<usize>> = HashMap::new();
//     //         for b in 0..num_batch {
//     //             batch_glob_index.insert(b, Vec::new());
//     //         }
//     //         let mut batch_membership: Vec<usize> = vec![0; ncol];
//     //         let mut offset = 0;

//     //         for (batch, data_batch) in data_vec.into_iter().enumerate() {
//     //             anyhow::ensure!(data_batch.num_rows().unwrap() == nrow);
//     //             let ncol_batch = data_batch.num_columns().unwrap();
//     //             let nblock = (ncol_batch + block_size - 1) / block_size;

//     //             /////////////////////////////////
//     //             // populate projection results //
//     //             /////////////////////////////////

//     //             (0..nblock)
//     //                 .into_par_iter()
//     //                 .map(|block| {
//     //                     let lb: usize = block * block_size;
//     //                     let ub: usize = ((block + 1) * block_size).min(ncol_batch);
//     //                     (lb, ub)
//     //                 })
//     //                 .for_each(|(lb, ub)| {
//     //                     let mut proj_km = arc_rand_proj_kn.lock().expect("failed to lock proj");
//     //                     // This could be inefficient since we are populating a dense matrix
//     //                     let xx_dm = data_batch.read_columns_ndarray((lb..ub).collect()).unwrap();
//     //                     // normalized columns by the column-wise sum values
//     //                     let denom = xx_dm.sum_axis(Axis(0)).mapv(|x| 1.0 / x.max(1.0));
//     //                     let yy_dm = xx_dm.dot(&Array2::from_diag(&denom));
//     //                     let proj = rand_basis_kd.dot(&yy_dm);

//     //                     let (lb_glob, ub_glob) = (lb + offset, ub + offset);
//     //                     let target = proj_km.slice_mut(s![.., lb_glob..ub_glob]);
//     //                     proj.assign_to(target);
//     //                 });

//     //             ////////////////////////////
//     //             // sorted out the indexes //
//     //             ////////////////////////////

//     //             (0..nblock).into_iter().for_each(|block| {
//     //                 let lb: usize = block * block_size;
//     //                 let ub: usize = ((block + 1) * block_size).min(ncol_batch);
//     //                 let (lb_glob, ub_glob) = (lb + offset, ub + offset);
//     //                 let globs = batch_glob_index.get_mut(&batch).expect("batch glob index");
//     //                 for j in lb_glob..ub_glob {
//     //                     batch_membership[j] = batch;
//     //                     globs.push(j);
//     //                 }
//     //             });

//     //             offset += ncol_batch;
//     //         }
//     //         Ok((rand_proj_kn, batch_glob_index, batch_membership))
//     //     };

//     // let (rand_proj_kn, batch_glob_index, batch_membership) = step2(&rand_basis_kd)?;

//     //////////////////////////////////////////////////////////
//     // Step 3: Assign columns to unified pseudobulk samples //
//     //////////////////////////////////////////////////////////

//     // fn step1(data_vec: &Vec<ArcData>, check_dim: usize) -> anyhow::Result<()> {
//     //     for arc_data in data_vec.iter() {
//     //         let data = arc_data.clone().lock()?;
//     //     }

//     //     Ok(())
//     // }

//     // let arc_data = Arc::new(Mutex::new(first_data.deref()));

//     Ok(())
// }

// // #[allow(dead_code)]
// // pub fn collapse_columns(
// //     data: &Data,
// //     target_dim: usize,
// //     block_size: Option<usize>,
// // ) -> anyhow::Result<()> {
// //     let block_size = block_size.unwrap_or(100);

// //     if let (Some(ncol), Some(nrow)) = (data.num_columns(), data.num_rows()) {
// //         // 1. Sample a random matrix to project
// //         let rand_basis_kd = sample_basis_to_reduce_rows(nrow, target_dim)?;
// //         let kk = rand_basis_kd.nrows();

// //         // let rand_basis_kd = scale_columns(Array2::random((kk, nrow), StandardNormal))?;

// //         // 2. Visit each column to store up the RP matrix
// //         let nblock = (ncol + block_size - 1) / block_size;
// //         let arc_data = Arc::new(Mutex::new(data));

// //         let mut rand_proj_kn: Array2<f32> = Array2::zeros((kk, ncol));
// //         let arc_rand_proj_kn = Arc::new(Mutex::new(&mut rand_proj_kn));

// //         (0..nblock)
// //             .into_par_iter()
// //             .map(|b| {
// //                 let lb: usize = b * block_size;
// //                 let ub: usize = ((b + 1) * block_size).min(ncol);
// //                 (lb, ub)
// //             })
// //             .for_each(|(lb, ub)| {
// //                 let data_b = arc_data.lock().expect("failed to lock data");
// //                 let mut proj_km = arc_rand_proj_kn.lock().expect("failed to lock proj");
// //                 // This could be inefficient since we are populating a dense matrix
// //                 let xx_dm = data_b.read_columns_ndarray((lb..ub).collect()).unwrap();
// //                 let target = proj_km.slice_mut(s![.., lb..ub]);
// //                 (rand_basis_kd.dot(&xx_dm)).assign_to(target);
// //             });

// //         let binary_code_kn: Array2<u32> = Array2::zeros((kk, ncol));

// //         let proj_km = arc_rand_proj_kn
// //             .lock()
// //             .expect("failed to lock proj")
// //             .clone();

// //         // let svd = TruncatedSvd::new(scale_columns(proj_km)?, TruncatedOrder::Largest)
// //         //     .decompose(kk)?
// //         //     .values_vectors();

// //         // let (svd_u, svd_d, svd_vt) = svd;

// //         // dbg!(&u.shape());
// //         // dbg!(&d.shape());
// //         // dbg!(&vt.shape());

// //         // let (u, s, vt) = proj_km.svd(true, true)?;
// //         // let vt: Array2<f32> = vt.unwrap();
// //         // let zz = svd_obj.values_vectors();
// //         // dbg!(zz.0);

// //         //     let mut _assignment: Vec<(usize, usize)> = vt
// //         //         .axis_iter(Axis(1))
// //         //         .into_par_iter()
// //         //         .enumerate()
// //         //         .map(|(col_idx, v_j)| {
// //         //             let idx = v_j.iter().fold(0, |acc, &v_ij| {
// //         //                 if v_ij > 0.0 {
// //         //                     acc // + (1 << col_idx)
// //         //                 } else {
// //         //                     acc
// //         //                 }
// //         //             });
// //         //             (col_idx, idx)
// //         //         })
// //         //         .collect();

// //         //     _assignment.sort_by_key(|&(i, _)| i);
// //         // }

// //         // 3. Random binary sorting
// //     }

// //     Ok(())
// // }
