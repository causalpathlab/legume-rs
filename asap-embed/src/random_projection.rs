use asap_data::ndarray_util::*;
use asap_data::sparse_io::*;
// use ndarray::parallel::prelude::*;
// use ndarray::prelude::*;
// use ndarray_linalg::transpose_data;

// use ndarray_linalg::svd::SVD; -- could be less efficient
use ndarray_linalg::TruncatedOrder;
use ndarray_linalg::TruncatedSvd;
use ndarray_rand::RandomExt;
// use ndarray_rand::rand_distr::Gamma;
// use ndarray_rand::rand_distr::Poisson;
// use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand_distr::StandardNormal;

use std::sync::Arc;
use std::sync::Mutex;
type Data = dyn SparseIo<IndexIter = Vec<usize>>;
type ArcData = Arc<Mutex<dyn SparseIo<IndexIter = Vec<usize>>>>;

#[allow(dead_code)]
pub fn sample_basis_to_reduce_rows(nrow: usize, target_dim: usize) -> anyhow::Result<Array2<f32>> {
    let kk = target_dim.min(nrow);
    Ok(scale_columns(Array2::random((kk, nrow), StandardNormal))?)
}

#[allow(dead_code)]
pub fn collapse_columns_cbind(
    data_vec: &Vec<Box<&Data>>,
    target_dim: usize,
    block_size: Option<usize>,
) -> anyhow::Result<()> {
    let block_size = block_size.unwrap_or(100);
    let num_batch = data_vec.len();

    if num_batch < 1 {
        anyhow::bail!("no data set in the vector");
    }

    ///////////////////////////////////////////////////
    // Step 1: Sample random projection basis matrix //
    ///////////////////////////////////////////////////

    // We figure out the number of rows using the first data set
    let step1 = |dim: usize| -> anyhow::Result<Array2<f32>> {
        let first_data = &data_vec[0];
        let nrow = first_data.num_rows().expect("#rows");
        let rand_basis_kd = sample_basis_to_reduce_rows(nrow, dim)?;
        Ok(rand_basis_kd)
    };

    let rand_basis_kd = step1(target_dim)?;

    // Gather dimensionality info

    let kk = rand_basis_kd.nrows();
    let nrow = rand_basis_kd.ncols();
    let ncol = data_vec.into_iter().fold(0, |ncols, data| -> usize {
        ncols + data.num_columns().expect("failed to figure out # cols")
    });

    ////////////////////////////////////////
    // Step 2: Create K x ncol projection //
    ////////////////////////////////////////

    let step2 =
        |rand_basis_kd: &Array2<f32>| -> anyhow::Result<(Array2<f32>, HashMap<_, _>, Vec<_>)> {
            let num_batch = data_vec.len();
            // the results of random projection
            let mut rand_proj_kn: Array2<f32> = Array2::zeros((kk, ncol));
            let arc_rand_proj_kn = Arc::new(Mutex::new(&mut rand_proj_kn));

            // batch to global membership and vice versa
            let mut batch_glob_index: HashMap<usize, Vec<usize>> = HashMap::new();
            for b in 0..num_batch {
                batch_glob_index.insert(b, Vec::new());
            }
            let mut batch_membership: Vec<usize> = vec![0; ncol];
            let mut offset = 0;

            for (batch, data_batch) in data_vec.into_iter().enumerate() {
                anyhow::ensure!(data_batch.num_rows().unwrap() == nrow);
                let ncol_batch = data_batch.num_columns().unwrap();
                let nblock = (ncol_batch + block_size - 1) / block_size;

                /////////////////////////////////
                // populate projection results //
                /////////////////////////////////

                (0..nblock)
                    .into_par_iter()
                    .map(|block| {
                        let lb: usize = block * block_size;
                        let ub: usize = ((block + 1) * block_size).min(ncol_batch);
                        (lb, ub)
                    })
                    .for_each(|(lb, ub)| {
                        let mut proj_km = arc_rand_proj_kn.lock().expect("failed to lock proj");
                        // This could be inefficient since we are populating a dense matrix
                        let xx_dm = data_batch.read_columns((lb..ub).collect()).unwrap();
                        // normalized columns by the column-wise sum values
                        let denom = xx_dm.sum_axis(Axis(0)).mapv(|x| 1.0 / x.max(1.0));
                        let yy_dm = xx_dm.dot(&Array2::from_diag(&denom));
                        let proj = rand_basis_kd.dot(&yy_dm);

                        let (lb_glob, ub_glob) = (lb + offset, ub + offset);
                        let target = proj_km.slice_mut(s![.., lb_glob..ub_glob]);
                        proj.assign_to(target);
                    });

                ////////////////////////////
                // sorted out the indexes //
                ////////////////////////////

                (0..nblock).into_iter().for_each(|block| {
                    let lb: usize = block * block_size;
                    let ub: usize = ((block + 1) * block_size).min(ncol_batch);
                    let (lb_glob, ub_glob) = (lb + offset, ub + offset);
                    let globs = batch_glob_index.get_mut(&batch).expect("batch glob index");
                    for j in lb_glob..ub_glob {
                        batch_membership[j] = batch;
                        globs.push(j);
                    }
                });

                offset += ncol_batch;
            }
            Ok((rand_proj_kn, batch_glob_index, batch_membership))
        };

    let (rand_proj_kn, batch_glob_index, batch_membership) = step2(&rand_basis_kd)?;

    //////////////////////////////////////////////////////////
    // Step 3: Assign columns to unified pseudobulk samples //
    //////////////////////////////////////////////////////////

    // fn step1(data_vec: &Vec<ArcData>, check_dim: usize) -> anyhow::Result<()> {
    //     for arc_data in data_vec.iter() {
    //         let data = arc_data.clone().lock()?;
    //     }

    //     Ok(())
    // }

    // let arc_data = Arc::new(Mutex::new(first_data.deref()));

    Ok(())
}

#[allow(dead_code)]
pub fn collapse_columns(
    data: &Data,
    target_dim: usize,
    block_size: Option<usize>,
) -> anyhow::Result<()> {
    let block_size = block_size.unwrap_or(100);

    if let (Some(ncol), Some(nrow)) = (data.num_columns(), data.num_rows()) {
        // 1. Sample a random matrix to project
        let rand_basis_kd = sample_basis_to_reduce_rows(nrow, target_dim)?;
        let kk = rand_basis_kd.nrows();

        // let rand_basis_kd = scale_columns(Array2::random((kk, nrow), StandardNormal))?;

        // 2. Visit each column to store up the RP matrix
        let nblock = (ncol + block_size - 1) / block_size;
        let arc_data = Arc::new(Mutex::new(data));

        let mut rand_proj_kn: Array2<f32> = Array2::zeros((kk, ncol));
        let arc_rand_proj_kn = Arc::new(Mutex::new(&mut rand_proj_kn));

        (0..nblock)
            .into_par_iter()
            .map(|b| {
                let lb: usize = b * block_size;
                let ub: usize = ((b + 1) * block_size).min(ncol);
                (lb, ub)
            })
            .for_each(|(lb, ub)| {
                let data_b = arc_data.lock().expect("failed to lock data");
                let mut proj_km = arc_rand_proj_kn.lock().expect("failed to lock proj");
                // This could be inefficient since we are populating a dense matrix
                let xx_dm = data_b.read_columns((lb..ub).collect()).unwrap();
                let target = proj_km.slice_mut(s![.., lb..ub]);
                (rand_basis_kd.dot(&xx_dm)).assign_to(target);
            });

        let binary_code_kn: Array2<u32> = Array2::zeros((kk, ncol));

        let proj_km = arc_rand_proj_kn
            .lock()
            .expect("failed to lock proj")
            .clone();

        let svd = TruncatedSvd::new(scale_columns(proj_km)?, TruncatedOrder::Largest)
            .decompose(kk)?
            .values_vectors();

        let (svd_u, svd_d, svd_vt) = svd;

        // dbg!(&u.shape());
        // dbg!(&d.shape());
        // dbg!(&vt.shape());

        // let (u, s, vt) = proj_km.svd(true, true)?;
        // let vt: Array2<f32> = vt.unwrap();
        // let zz = svd_obj.values_vectors();
        // dbg!(zz.0);

        //     let mut _assignment: Vec<(usize, usize)> = vt
        //         .axis_iter(Axis(1))
        //         .into_par_iter()
        //         .enumerate()
        //         .map(|(col_idx, v_j)| {
        //             let idx = v_j.iter().fold(0, |acc, &v_ij| {
        //                 if v_ij > 0.0 {
        //                     acc // + (1 << col_idx)
        //                 } else {
        //                     acc
        //                 }
        //             });
        //             (col_idx, idx)
        //         })
        //         .collect();

        //     _assignment.sort_by_key(|&(i, _)| i);
        // }

        // 3. Random binary sorting
    }

    Ok(())
}
