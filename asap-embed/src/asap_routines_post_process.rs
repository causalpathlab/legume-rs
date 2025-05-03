use crate::asap_embed_common::*;
use crate::asap_normalization::*;
use crate::asap_visitors::VisitColumnsOps;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;

use matrix_util::traits::*;

use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;

use std::sync::{Arc, Mutex};

use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

fn visit_data_to_adjust(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    delta_db: &Mat,
    triplets: Arc<Mutex<&mut Vec<(u64, u64, f32)>>>,
) {
    let (lb, ub) = job;
    let batches = full_data_vec.get_batch_membership(lb..ub);

    let mut x_dn = full_data_vec
        .read_columns_csc(lb..ub)
        .expect("read columns");

    x_dn.adjust_by_division(&delta_db, &batches);

    let new_triplets = x_dn
        .triplet_iter()
        .filter_map(|(i, j, &x_ij)| {
            let x_ij = x_ij.round();
            if x_ij < 1_f32 {
                None
            } else {
                Some((i as u64, (j + lb) as u64, x_ij))
            }
        })
        .collect::<Vec<_>>();

    let mut triplets = triplets.lock().expect("lock triplets");
    triplets.extend(new_triplets);
}

/// Adjust the original data by eliminating batch effects `delta_db`
/// (`d x b`) from each column. We will directly call
/// `get_batch_membership` in `data_vec`.
///
/// # Arguments
/// * `data_vec` - sparse data vector
/// * `delta_db` - row/feature by batch average effect matrix
///
/// # Returns
/// * `triplets` - we can feed this vector to create a new backend
pub fn triplets_adjusted_by_batch(
    data_vec: &SparseIoVec,
    delta_db: &Mat,
) -> anyhow::Result<Vec<(u64, u64, f32)>> {
    let mut triplets = vec![];
    let ntot = data_vec.num_columns()?;
    let block_size = DEFAULT_BLOCK_SIZE;
    let jobs = create_jobs(ntot, block_size);
    data_vec.visit_columns_by_jobs(jobs, &visit_data_to_adjust, &delta_db, &mut triplets)?;
    Ok(triplets)
}

/// Evaluate latent representation with the trained encoder network
///
/// #Arguments
/// * `data_vec` - full data vector
/// * `encoder` - encoder network
/// * `train_config` - training configuration
/// * `delta_db` - batch effect matrix (feature x batch)
pub fn evaluate_latent_by_encoder<Enc>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    train_config: &TrainConfig,
    delta_db: Option<&Mat>,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
{
    let dev = &train_config.device;
    let ntot = data_vec.num_columns()?;
    let kk = encoder.dim_latent();

    let block_size = train_config.batch_size;

    let jobs = create_jobs(ntot, block_size);
    let njobs = jobs.len() as u64;
    let arc_enc = Arc::new(Mutex::new(encoder));

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&(lb, ub)| {
            let enc = arc_enc.lock().expect("enc lock");

            let z_nk = match delta_db {
                Some(delta_db) => {
                    let x_dn = data_vec.read_columns_dmatrix(lb..ub).expect("read columns");
                    let (d, n) = x_dn.shape();

                    // Note: x_dn.as_slice() will take values in the column-major order
                    // However, Tensor::from_slice will take them in the row-major order
                    let x_nd =
                        Tensor::from_slice(x_dn.as_slice(), (n, d), dev).expect("tensor x_nd");

                    let b = delta_db.ncols();
                    let delta_bd = Tensor::from_slice(delta_db.as_slice(), (b, d), dev)
                        .expect("tensor delta_bd");

                    let batches = data_vec
                        .get_batch_membership(lb..ub)
                        .into_iter()
                        .map(|x| x as u32);

                    let batches = Tensor::from_iter(batches.clone(), dev).unwrap();
                    let x0_nd = delta_bd.index_select(&batches, 0).expect("expand delta_bd");
                    let (z_nk, _) = enc
                        .forward_with_null_t(&x_nd, &x0_nd, false)
                        .expect("forward");

                    z_nk
                }
                None => {
                    // simple forward pass without batch adjustment
                    let x_dn = data_vec.read_columns_dmatrix(lb..ub).expect("read columns");
                    let (d, n) = x_dn.shape();

                    // Note: x_dn.as_slice() will take values in the column-major order
                    // However, Tensor::from_slice will take them in the row-major order
                    let x_nd =
                        Tensor::from_slice(x_dn.as_slice(), (n, d), dev).expect("tensor x_nd");

                    let (z_nk, _) = enc.forward_t(&x_nd, false).expect("forward");
                    z_nk
                }
            };
            let z_nk = z_nk.to_device(&candle_core::Device::Cpu).expect("to cpu");
            (lb, Mat::from_tensor(&z_nk).expect("to mat"))
        })
        .collect::<Vec<_>>();

    chunks.sort_by_key(|&(lb, _)| lb);
    let chunks = chunks.into_iter().map(|(_, z_nk)| z_nk).collect::<Vec<_>>();

    let mut ret = Mat::zeros(ntot, kk);
    {
        let mut lb = 0;
        for z in chunks {
            let ub = lb + z.nrows();
            ret.rows_range_mut(lb..ub).copy_from(&z);
            lb = ub;
        }
    }
    Ok(ret)
}
