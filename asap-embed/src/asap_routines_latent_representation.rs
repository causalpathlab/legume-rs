use crate::asap_embed_common::*;

use log::info;

use matrix_param::traits::Inference;
use matrix_param::traits::ParamIo;
use matrix_util::common_io::{extension, read_lines};
use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::traits::*;

use crate::asap_collapse_data::CollapsingOps;
use crate::asap_random_projection::RandProjOps;
use crate::asap_routines_post_process::*;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_encoder::*;
use candle_util::candle_model_topic::*;
use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;

use clap::{Parser, ValueEnum};
use std::sync::{Arc, Mutex};

use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

/// Nystrom projection for fast latent representation
///
/// # Arguments
/// * `xx_dn` - feature x sample matrix
/// * `delta_db` - feature x batch batch effect matrix
/// * `full_data_vec` - full sparse data vector
/// * `rank` - matrix factorization rank
/// * `block_size` - online learning block size
///
/// # Returns
/// * (`z_nk`, `u_dk`, `vec![]`)
/// * `z_nk` - sample x factor latent representation matrix
/// * `u_dk` - feature x factor dictionary matrix
///
pub fn do_nystrom_proj(
    log_xx_dn: Mat,
    delta_db: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    rank: usize,
    block_size: Option<usize>,
) -> anyhow::Result<(Mat, Mat, Option<Mat>, Option<Vec<f32>>)> {
    let mut log_xx_dn = log_xx_dn.clone();

    log_xx_dn.scale_columns_inplace();

    let (u_dk, _, _) = log_xx_dn.rsvd(rank)?;

    let eps = 1e-8;
    let xdepth = 1e4;

    info!(
        "Constructed {} x {} projection matrix",
        u_dk.nrows(),
        u_dk.ncols()
    );

    let ntot = full_data_vec.num_columns()?;
    let kk = rank;
    let block_size = block_size.unwrap_or(100);

    let jobs = create_jobs(ntot, block_size);
    let njobs = jobs.len() as u64;

    info!("Visiting {} blocks ...", njobs);

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&(lb, ub)| {
            let mut x_dn = full_data_vec
                .read_columns_dmatrix(lb..ub)
                .expect("read columns");

            x_dn.column_iter_mut().for_each(|mut x_j| {
                let xsum = x_j.sum();
                if xsum > 0.0 {
                    x_j.scale_mut(xdepth / xsum);
                }
            });

            if let Some(delta_db) = delta_db {
                let batches = full_data_vec.get_batch_membership(lb..ub);

                // This will (a) estimate scaling parameter for each
                // cell and (b) adjust the raw expression data with
                // delta
                x_dn.column_iter_mut()
                    .zip(batches)
                    .for_each(|(mut x_j, b)| {
                        let d_j = delta_db.column(b);
                        let xsum = x_j.sum();
                        let dsum = d_j.sum();
                        let scale = if dsum > 0.0 { xsum / dsum } else { 1.0 };
                        x_j.zip_apply(&d_j, |x_ij, d_ij| *x_ij /= d_ij * scale + eps);
                    });
            }

            let mut log_x_dn = x_dn.map(|x| (x + 0.5).log2());
            log_x_dn.scale_columns_inplace();

            let z_nk = log_x_dn.transpose() * &u_dk;
            (lb, z_nk)
        })
        .collect::<Vec<_>>();

    chunks.sort_by_key(|&(lb, _)| lb);
    let chunks = chunks.into_iter().map(|(_, z_nk)| z_nk).collect::<Vec<_>>();

    info!("Done {} projections ...", njobs);

    let mut z_nk = Mat::zeros(ntot, kk);

    {
        let mut lb = 0;
        for z in chunks {
            let ub = lb + z.nrows();
            z_nk.rows_range_mut(lb..ub).copy_from(&z);
            lb = ub;
        }
    }

    Ok((z_nk, u_dk, None, None))
}

/// Train an autoencoder model on collapsed data and evaluate latent
/// mapping by the fixed encoder model.
///
/// # Arguments
/// * `data_nd` - sample x feature collapsed data matrix
/// * `delta_db` - feature x batch batch effect matrix
/// * `full_data_vec` - full sparse data vector
/// * `encoder` - encoder model
/// * `decoder` - decoder model
/// * `log_likelihood_func` - log-likelihood function
/// * `train_config` - training configuration
///
/// # Returns
/// * (`z_nk`, `beta_dk`, `llik`)
/// * `z_nk` - sample x factor latent representation matrix
/// * `beta_dk` - feature x factor dictionary matrix
/// * `llik` - log-likelihood trace vector
///
pub fn train_encoder_decoder<Enc, Dec, LLikFn>(
    raw_data_nd: &Mat,
    adj_data_nd: Option<&Mat>,
    residual_data_nd: Option<&Mat>,
    delta_db: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
) -> anyhow::Result<(Mat, Mat, Option<Mat>, Option<Vec<f32>>)>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModule + Send + Sync + 'static,
    LLikFn: Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor> + Sync + Send,
{
    let kk = encoder.dim_latent();

    let mut x_std_nd = match adj_data_nd {
        Some(adj_data_nd) => adj_data_nd,
        None => raw_data_nd,
    }
    .map(|x| (x + 0.5).log2())
    .clone();

    x_std_nd.scale_columns_inplace();
    let (mut z_nk_init, _, _) = x_std_nd.rsvd(kk)?;
    z_nk_init.scale_columns_inplace();

    let kk_init = z_nk_init.ncols();
    if kk_init < kk {
        info!("zero-padding {} columns...", kk - kk_init);
        z_nk_init = z_nk_init.insert_columns(kk_init, kk - kk_init, 0.);
    }

    let mut vae = Vae::build(encoder, decoder, parameters);

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    if train_config.num_pretrain_epochs > 0 {
        info!(
            "Start pre-training with z {} x {} ...",
            z_nk_init.nrows(),
            z_nk_init.ncols()
        );

        let mut data_loader = match adj_data_nd {
            Some(data_nd) => InMemoryData::from_with_output(data_nd, &z_nk_init)?,
            None => InMemoryData::from_with_output(raw_data_nd, &z_nk_init)?,
        };

        let _llik = vae.pretrain_encoder(
            &mut data_loader,
            &loss_func::gaussian_likelihood,
            &train_config,
        )?;
    }

    let mut data_loader = if let (Some(target_nd), Some(null_nd)) = (adj_data_nd, residual_data_nd)
    {
        InMemoryData::from_with_aux_output(raw_data_nd, null_nd, target_nd)?
    } else {
        InMemoryData::from(raw_data_nd)?
    };

    let llik_trace =
        vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, &train_config)?;

    info!("Done with training {} epochs", train_config.num_epochs);

    let z_nk = evaluate_latent_by_encoder(&full_data_vec, encoder, &train_config, delta_db, false)?;

    let z_direct_nk =
        evaluate_latent_by_encoder(&full_data_vec, encoder, &train_config, delta_db, true)?;

    info!("Finished encoding latent states for all");

    let beta_dk = decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?;
    let beta_dk = Mat::from_tensor(&beta_dk)?;

    Ok((z_nk, beta_dk, Some(z_direct_nk), Some(llik_trace)))
}

/// Evaluate latent representation with the trained encoder network
///
/// #Arguments
/// * `data_vec` - full data vector
/// * `encoder` - encoder network
/// * `train_config` - training configuration
/// * `delta_db` - batch effect matrix (feature x batch)
fn evaluate_latent_by_encoder<Enc>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    train_config: &TrainConfig,
    delta_db: Option<&Mat>,
    direct_adjustment: bool,
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
                    let (z_nk, _) = if direct_adjustment {
                        let mut x_dn = data_vec.read_columns_dmatrix(lb..ub).expect("read columns");
                        let (d, n) = x_dn.shape();

                        let batches = data_vec.get_batch_membership(lb..ub);

                        // This will (a) estimate scaling parameter for each
                        // cell and (b) adjust the raw expression data with
                        // delta
                        let eps = 1e-4;
                        x_dn.column_iter_mut()
                            .zip(batches)
                            .for_each(|(mut x_j, b)| {
                                let d_j = delta_db.column(b);
                                let xsum = x_j.sum();
                                let dsum = d_j.sum();
                                let scale = if dsum > 0.0 { xsum / dsum } else { 1.0 };
                                x_j.zip_apply(&d_j, |x_ij, d_ij| *x_ij /= d_ij * scale + eps);
                            });

                        // Note: x_dn.as_slice() will take values in the column-major order
                        // However, Tensor::from_slice will take them in the row-major order
                        let x_nd =
                            Tensor::from_slice(x_dn.as_slice(), (n, d), dev).expect("tensor x_nd");

                        enc.forward_t(&x_nd, false).expect("forward")
                    } else {
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
                        enc.forward_with_null_t(&x_nd, &x0_nd, false)
                            .expect("forward")
                    };
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
