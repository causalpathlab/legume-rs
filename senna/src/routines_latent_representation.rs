use crate::embed_common::*;

use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans_alg::normalization::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;
use indicatif::{ProgressBar, ProgressDrawTarget};

use matrix_param::dmatrix_gamma::GammaMatrix;

fn nystrom_proj_visitor(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    proj_basis: &NystromParam,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let u_dk = proj_basis.dictionary_dk;
    let delta_db = proj_basis.batch_db;

    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    x_dn.normalize_columns_inplace();

    if let Some(delta_db) = delta_db {
        let batches = full_data_vec.get_batch_membership(lb..ub);
        x_dn.adjust_by_division_of_selected_inplace(delta_db, &batches);
    }

    x_dn.values_mut().iter_mut().for_each(|x| {
        *x = (*x + 1.).ln();
    });

    x_dn.scale_columns_inplace();

    let chunk = (x_dn.transpose() * u_dk).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in nystrom");

    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
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

    full_data_vec.visit_columns_by_block(
        &nystrom_proj_visitor,
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
    input_data_nm: &Mat,
    null_data_nm: Option<&Mat>,
    adjusted_data_nd: &Mat,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
) -> anyhow::Result<Vec<f32>>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModuleT + Send + Sync + 'static,
    LLikFn: Fn(&candle_core::Tensor, &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>
        + Sync
        + Send,
{
    let mut vae = Vae::build(encoder, decoder, parameters);

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    let mut data_loader = match null_data_nm {
        Some(null_nm) => {
            info!(
                "data loader with [{}, {}] -> [{}]",
                input_data_nm.ncols(),
                null_nm.ncols(),
                adjusted_data_nd.ncols()
            );

            InMemoryData::from(DataLoaderArgs {
                input: input_data_nm,
                input_null: Some(null_nm),
                output: Some(adjusted_data_nd),
                output_null: None,
            })?
        }
        _ => {
            info!(
                "data loader with [{}] -> [{}]",
                input_data_nm.ncols(),
                adjusted_data_nd.ncols()
            );

            InMemoryData::from(DataLoaderArgs {
                input: input_data_nm,
                input_null: None,
                output: Some(adjusted_data_nd),
                output_null: None,
            })?
        }
    };

    let llik_trace =
        vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, train_config)?;

    info!(
        "Done with training over {} epochs using {} samples",
        train_config.num_epochs,
        data_loader.num_data()
    );

    Ok(llik_trace)
}

/// Train an autoencoder model on collapsed data and evaluate latent
/// mapping by the fixed encoder model.
///
/// # Arguments
/// * `mixed_dn` - raw data, feature x sample
/// * `batch_dn` - batch effect, feature x sample
/// * `clean_dn` - clean feature x sample
/// * `aggregate_dm` - row aggregator for input
/// * `encoder` - encoder model: `m -> k`
/// * `decoder` - decoder model: `k -> d`
/// * `log_likelihood_func` - log-likelihood function
/// * `train_config` - training configuration
/// * `jitter` - jitter interval
///
/// # Returns
/// * (`z_nk`, `beta_dk`, `llik`)
/// * `z_nk` - sample x factor latent representation matrix
/// * `beta_dk` - feature x factor dictionary matrix
/// * `llik` - log-likelihood trace vector
///
pub fn train_encoder_decoder_stochastic<Enc, Dec, LLikFn>(
    mixed_dn: &GammaMatrix,
    batch_dn: Option<&GammaMatrix>,
    clean_dn: Option<&GammaMatrix>,
    aggregate_dm: &Mat,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
    jitter: usize,
) -> anyhow::Result<Vec<f32>>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModuleT + Send + Sync + 'static,
    LLikFn: Fn(&candle_core::Tensor, &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>
        + Sync
        + Send,
{
    let mut vae = Vae::build(encoder, decoder, parameters);

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    let mut llik_trace = vec![];

    let sub_train_config = TrainConfig {
        learning_rate: train_config.learning_rate, // override
        batch_size: train_config.batch_size,       // train
        num_epochs: jitter,                        // config
        num_pretrain_epochs: 0,                    //
        device: train_config.device.clone(),       //
        verbose: false,
        show_progress: false,
    };

    let pb = ProgressBar::new(train_config.num_epochs as u64);

    if !train_config.show_progress || train_config.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    for epoch in (0..train_config.num_epochs).step_by(jitter) {
        let input_nm = mixed_dn.posterior_sample()?.transpose() * aggregate_dm;

        let batch_nm = match batch_dn {
            Some(x) => Some(x.posterior_sample()?.transpose() * aggregate_dm),
            _ => None,
        };

        let output_nd = match clean_dn {
            Some(x) => x.posterior_sample()?.transpose(),
            _ => mixed_dn.posterior_sample()?.transpose(),
        };

        let mut data_loader = match batch_nm {
            Some(batch_nm) => InMemoryData::from(DataLoaderArgs {
                input: &input_nm,
                input_null: Some(&batch_nm),
                output: Some(&output_nd),
                output_null: None,
            })?,
            _ => InMemoryData::from(DataLoaderArgs {
                input: &input_nm,
                input_null: None,
                output: Some(&output_nd),
                output_null: None,
            })?,
        };

        let llik =
            vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, &sub_train_config)?;

        llik_trace.extend(llik);
        pb.inc(jitter as u64);

        if train_config.verbose {
            info!(
                "[{}] log-likelihood: {}",
                epoch + 1,
                llik_trace.last().ok_or(anyhow::anyhow!("llik"))?
            );
        }
    }
    pb.finish_and_clear();

    Ok(llik_trace)
}
