use crate::asap_embed_common::*;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;

use matrix_util::traits::*;

use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;

use std::sync::{Arc, Mutex};

use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

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
