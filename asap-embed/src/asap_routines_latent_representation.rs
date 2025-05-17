use crate::embed_common::*;
use asap_alg::normalization::*;
use asap_data::sparse_data_visitors::VisitColumnsOps;

use log::{info, warn};

use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::traits::*;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;

use std::sync::{Arc, Mutex};

fn visit_nystrom_proj_columnwise(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    proj_basis: &NystromParam,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) {
    let (lb, ub) = job;
    let u_dk = proj_basis.dictionary_dk;
    let delta_db = proj_basis.batch_db;

    let mut x_dn = full_data_vec
        .read_columns_csc(lb..ub)
        .expect("read columns");

    x_dn.normalize_columns_inplace();

    if let Some(delta_db) = delta_db {
        let batches = full_data_vec.get_batch_membership(lb..ub);
        x_dn.adjust_by_division(&delta_db, &batches);
    }

    x_dn.values_mut().iter_mut().for_each(|x| {
        *x = (*x + 1.).ln();
    });

    x_dn.scale_columns_inplace();

    let chunk = (x_dn.transpose() * u_dk).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in nystrom");

    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
}

struct NystromParam<'a> {
    dictionary_dk: &'a Mat,
    batch_db: Option<&'a Mat>,
}

pub struct NystromOut {
    pub dictionary_dk: Mat,
    pub latent_nk: Mat,
}

/// Nystrom projection for fast latent representation
///
/// # Arguments
/// * `xx_dn` - feature x sample matrix
/// * `delta_db` - feature x batch batch effect matrix
/// * `full_data_vec` - full sparse data vector
/// * `rank` - matrix factorization rank
/// * `block_size` - online learning block size
///
///
pub fn do_nystrom_proj(
    log_xx_dn: Mat,
    delta_db: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    rank: usize,
    block_size: Option<usize>,
) -> anyhow::Result<NystromOut> {
    let mut log_xx_dn = log_xx_dn.clone();

    log_xx_dn.scale_columns_inplace();

    let (u_dk, _, _) = log_xx_dn.rsvd(rank)?;

    info!(
        "Constructed {} x {} projection matrix",
        u_dk.nrows(),
        u_dk.ncols()
    );

    let ntot = full_data_vec.num_columns()?;
    let kk = rank;

    let nystrom_param = NystromParam {
        dictionary_dk: &u_dk,
        batch_db: delta_db,
    };

    let mut proj_kn = Mat::zeros(kk, ntot);

    full_data_vec.visit_columns_by_jobs(
        &visit_nystrom_proj_columnwise,
        &nystrom_param,
        &mut proj_kn,
        block_size,
    )?;

    let z_nk = proj_kn.transpose();

    Ok(NystromOut {
        dictionary_dk: u_dk,
        latent_nk: z_nk,
    })
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
    raw_data_nm: &Mat,
    adj_data_nd: Option<&Mat>,
    residual_data_nm: Option<&Mat>,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
) -> anyhow::Result<Vec<f32>>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModule + Send + Sync + 'static,
    LLikFn: Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor> + Sync + Send,
{
    let kk = encoder.dim_latent();

    let mut x_std_nd = match adj_data_nd {
        Some(adj_data_nd) => adj_data_nd,
        None => raw_data_nm,
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
            None => InMemoryData::from_with_output(raw_data_nm, &z_nk_init)?,
        };

        let _llik = vae.pretrain_encoder(
            &mut data_loader,
            &loss_func::gaussian_likelihood,
            &train_config,
        )?;
    }

    let mut data_loader = match (adj_data_nd, residual_data_nm) {
        (Some(target_nd), Some(null_nm)) => {
            info!(
                "data loader with [{}, {}] -> [{}]",
                raw_data_nm.ncols(),
                null_nm.ncols(),
                target_nd.ncols()
            );
            InMemoryData::from_with_aux_output(raw_data_nm, null_nm, target_nd)?
        }
        (Some(target_nd), None) => {
            info!(
                "data loader with [{}] -> [{}]",
                raw_data_nm.ncols(),
                target_nd.ncols()
            );
            InMemoryData::from_with_output(raw_data_nm, target_nd)?
        }
        _ => {
            warn!("The dimension of dictionary may ");
            InMemoryData::from(raw_data_nm)?
        }
    };

    let llik_trace =
        vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, &train_config)?;

    info!("Done with training {} epochs", train_config.num_epochs);

    Ok(llik_trace)
}
