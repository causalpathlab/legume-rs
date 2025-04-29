mod asap_collapse_data;
mod asap_embed_common;
mod asap_normalization;
mod asap_random_projection;

use log::info;

use matrix_param::traits::Inference;
use matrix_param::traits::ParamIo;
use matrix_util::common_io::{extension, read_lines};
use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::traits::*;

use asap_embed_common::*;

use asap_collapse_data::CollapsingOps;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use asap_random_projection::RandProjOps;

use candle_util::candle_data_loader::*;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_encoder::*;
use candle_util::candle_model_poisson::*;
use candle_util::candle_model_topic::*;
use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use indicatif::ParallelProgressIterator;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum DecoderModel {
    Topic,
    Poisson,
    Proj,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "ASAP embedding", version, about, long_about, term_width = 80)]
/// A quick embedding utility
///
/// This command will embed high-dimensional data (where each data
/// point is a column vector) into a lower-dimensional space in three
/// steps: (1) approximate collapsing to reduce sample size, (2)
/// training an embedding model, and (3) recover latent states by
/// revisiting the data.
///
struct EmbedArgs {
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// in the given list should be identical. We can convert `.mtx`
    /// to `.zarr` or `.h5` using `asap-data build` command.
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 30)]
    proj_dim: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// Batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Reference batch name (comma-separated names)
    #[arg(short = 'r', long, value_delimiter(','))]
    reference_batch: Option<Vec<Box<str>>>,

    /// #k-nearest neighbours within each batch
    #[arg(long, short = 'n', default_value_t = 10)]
    knn: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 100)]
    iter_opt: usize,

    /// Block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// Number of latent topics
    #[arg(short = 'k', long, default_value_t = 10)]
    latent_topics: usize,

    /// Encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// Intensity levels for frequency embedding
    #[arg(long, default_value_t = 5000)]
    vocab_size: usize,

    /// Intensity embedding dimension
    #[arg(long, default_value_t = 5)]
    vocab_emb: usize,

    /// # pre-training epochs with an encoder-only model
    #[arg(long, default_value_t = 0)]
    pretrain_epochs: usize,

    /// # training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// Candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

    /// Choose a decoder model
    #[arg(long, short = 'm', value_enum, default_value = "topic")]
    decoder_model: DecoderModel,

    /// Save intermediate projection results
    #[arg(long)]
    save_intermediate: bool,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = EmbedArgs::parse();

    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let train_config = TrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.minibatch_size,
        num_epochs: args.epochs,
        num_pretrain_epochs: args.pretrain_epochs,
        device: dev.clone(),
        verbose: args.verbose,
    };
    info!("Reading data files...");

    let (mut data_vec, batch_membership) = read_data_vec_membership(args.clone())?;
    if let Some(reference_vec) = args.reference_batch.as_ref() {
        info!("Reference batches: {:?}", reference_vec);
    }

    let dd = data_vec.num_rows()?;
    let kk = args.latent_topics;

    info!("Estimate {} topics", kk);

    if args.decoder_model != DecoderModel::Proj {
        info!("Encoder layers: {:?}", args.encoder_layers);
    }

    /////////////////////////////////////
    // 1. Randomly project the columns //
    /////////////////////////////////////

    let proj_dim = args.proj_dim.max(kk);

    info!("Random projection of data onto {} dims", proj_dim);
    let proj_out = data_vec.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size.clone()),
        Some(&batch_membership),
    )?;

    let proj_kn = proj_out.proj;
    info!("Assigning {} columns to samples...", proj_kn.ncols());

    if args.save_intermediate {
        proj_kn
            .transpose()
            .to_tsv(&(args.out.to_string() + ".rp.gz"))?;
    }

    let nsamp = data_vec.assign_columns_to_samples(&proj_kn, Some(args.sort_dim))?;
    info!("at most {} samples are assigned", nsamp);

    //////////////////////////////////
    // 2. Register batch membership //
    //////////////////////////////////

    if args.batch_files.is_some() && batch_membership.len() > 0 {
        info!("Registering batch information");
        data_vec.register_batches(&proj_kn, &batch_membership)?;
    }

    ///////////////////////////
    // 3. Collapsing columns //
    ///////////////////////////

    info!("Collapsing columns... into {} samples", nsamp);
    let collapse_out = data_vec.collapse_columns(
        args.down_sample,
        args.reference_batch.clone(),
        Some(args.knn),
        Some(args.iter_opt),
    )?;

    /////////////////////////////////////////////////////////
    // 4. Train embedded topic model on the collapsed data //
    /////////////////////////////////////////////////////////

    let parameters = candle_nn::VarMap::new();
    let dev = &train_config.device;
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

    let kk = args.latent_topics;

    let raw_dn = &collapse_out.mu_observed;
    let delta_db = collapse_out.delta.as_ref();
    let adj_dn = collapse_out.mu_adjusted.as_ref();
    let resid_dn = collapse_out.mu_residual.as_ref();

    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;

    let (latent_nk, dictionary_dk, latent_direct_nk, log_likelihood_trace) =
        match args.decoder_model {
            DecoderModel::Topic => {
                let enc = LogSoftmaxEncoder::new(
                    dd,
                    kk,
                    n_vocab,
                    d_vocab_emb,
                    &args.encoder_layers,
                    param_builder.clone(),
                )?;
                let dec = TopicDecoder::new(dd, kk, param_builder.clone())?;
                run_asap_embedding(
                    &raw_dn.posterior_mean().transpose().clone(),
                    adj_dn.map(|x| x.posterior_mean().transpose()).as_ref(),
                    resid_dn.map(|x| x.posterior_mean().transpose()).as_ref(),
                    delta_db.map(|x| x.posterior_mean()),
                    &data_vec,
                    &enc,
                    &dec,
                    &parameters,
                    &loss_func::topic_likelihood,
                    &train_config,
                )?
            }

            DecoderModel::Poisson => {
                let enc = LogSoftmaxEncoder::new(
                    dd,
                    kk,
                    n_vocab,
                    d_vocab_emb,
                    &args.encoder_layers,
                    param_builder.clone(),
                )?;
                let dec = PoissonDecoder::new(dd, kk, param_builder.clone())?;
                run_asap_embedding(
                    &raw_dn.posterior_mean().transpose().clone(),
                    adj_dn.map(|x| x.posterior_mean().transpose()).as_ref(),
                    resid_dn.map(|x| x.posterior_mean().transpose()).as_ref(),
                    delta_db.map(|x| x.posterior_mean()),
                    &data_vec,
                    &enc,
                    &dec,
                    &parameters,
                    &loss_func::topic_likelihood,
                    &train_config,
                )?
            }
            _ => {
                let x_dn = match collapse_out.mu_adjusted.as_ref() {
                    Some(adj) => adj,
                    None => &collapse_out.mu_observed,
                };

                run_nystrom_proj(
                    x_dn.posterior_log_mean().clone(),
                    delta_db.map(|x| x.posterior_mean()),
                    &data_vec,
                    args.latent_topics,
                    Some(args.block_size.clone()),
                )?
            }
        };

    if let Some(log_likelihood_trace) = log_likelihood_trace {
        Mat::from_row_iterator(
            log_likelihood_trace.len(),
            1,
            log_likelihood_trace.into_iter(),
        )
        .to_tsv(&(args.out.to_string() + ".llik.gz"))?;
    }

    latent_nk.to_tsv(&(args.out.to_string() + ".latent.gz"))?;

    if let Some(latent_nk) = latent_direct_nk {
        latent_nk.to_tsv(&(args.out.to_string() + ".latent.direct.gz"))?;
    }

    dictionary_dk.to_tsv(&(args.out.to_string() + ".dictionary.gz"))?;

    if let Some(param) = delta_db {
        param.write_tsv(&(args.out.to_string() + ".delta.gz"))?;
    }

    info!("done");
    Ok(())
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
/// # Returns
/// * (`z_nk`, `u_dk`, `vec![]`)
/// * `z_nk` - sample x factor latent representation matrix
/// * `u_dk` - feature x factor dictionary matrix
///
fn run_nystrom_proj(
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
fn run_asap_embedding<Enc, Dec, LLikFn>(
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

//////////////////////////////////////////
// read data files and batch membership //
//////////////////////////////////////////

fn read_data_vec_membership(args: EmbedArgs) -> anyhow::Result<(SparseIoVec, Vec<Box<str>>)> {
    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    let mut data_vec = SparseIoVec::new();
    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        match extension(&data_file)?.as_ref() {
            "zarr" => {
                assert_eq!(backend, SparseIoBackend::Zarr);
            }
            "h5" => {
                assert_eq!(backend, SparseIoBackend::HDF5);
            }
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", data_file)),
        };

        let data = open_sparse_matrix(&data_file, &backend)?;
        data_vec.push(Arc::from(data))?;
    }

    // check if row names are the same
    let row_names = data_vec[0].row_names()?;

    for j in 1..data_vec.len() {
        let row_names_j = data_vec[j].row_names()?;
        if row_names != row_names_j {
            return Err(anyhow::anyhow!("Row names are not the same"));
        }
    }

    // check batch membership
    let mut batch_membership = vec![];

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# batch files != # of data files"));
        }

        for batch_file in batch_files.iter() {
            info!("Reading batch file: {}", batch_file);
            for s in read_lines(&batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
        }
    } else {
        for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
            batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
        }
    }

    if batch_membership.len() != data_vec.num_columns()? {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()?
        ));
    }

    Ok((data_vec, batch_membership))
}
