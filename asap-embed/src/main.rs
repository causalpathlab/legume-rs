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
use candle_util::candle_loss_functions::*;
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
    #[arg(long, short, default_value_t = 30)]
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
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![32,128,1024,128])]
    encoder_layers: Vec<usize>,

    /// # pre-training epochs
    #[arg(long, default_value_t = 100)]
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

    /// Decodeer model: topic (softmax) model, poisson MF, or nystrom
    /// projection
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
    info!("Reference batches: {:?}", args.reference_batch);

    let dd = data_vec.num_rows()?;
    let kk = args.latent_topics;

    info!("Estimate {} topics", kk);

    if args.decoder_model != DecoderModel::Proj {
        info!("Encoder layers: {:?}", args.encoder_layers);
    }

    /////////////////////////////////////
    // 1. Randomly project the columns //
    /////////////////////////////////////

    info!("Random projection of data onto {} dims", args.proj_dim);
    let proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
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

    let x_nd = collapse_out.mu.posterior_mean().transpose().clone();

    info!(
        "Collapsed: {} x {} -> {} x {}",
        data_vec.num_columns()?,
        data_vec.num_rows()?,
        x_nd.nrows(),
        x_nd.ncols()
    );

    let kk = args.latent_topics;

    let delta_bd = collapse_out.delta.as_ref();

    let (z_nk, beta_dk, llik) = match args.decoder_model {
        DecoderModel::Poisson => {
            let enc = LogSoftmaxEncoder::new(dd, kk, &args.encoder_layers, param_builder.clone())?;
            let dec = PoissonDecoder::new(dd, kk, param_builder.clone())?;
            run_asap_embedding(
                x_nd,
                delta_bd.map(|x| x.posterior_mean()),
                &data_vec,
                &enc,
                &dec,
                &parameters,
                &poisson_likelihood,
                &train_config,
            )?
        }

        DecoderModel::Topic => {
            let enc = LogSoftmaxEncoder::new(dd, kk, &args.encoder_layers, param_builder.clone())?;
            let dec = TopicDecoder::new(dd, kk, param_builder.clone())?;
            run_asap_embedding(
                x_nd,
                delta_bd.map(|x| x.posterior_mean()),
                &data_vec,
                &enc,
                &dec,
                &parameters,
                &topic_likelihood,
                &train_config,
            )?
        }

        _ => run_nystrom_proj(
            collapse_out.mu.posterior_log_mean().clone(),
            delta_bd.map(|x| x.posterior_mean()),
            &data_vec,
            args.latent_topics,
            Some(args.block_size.clone()),
        )?,
    };

    if llik.len() > 0 {
        Mat::from_row_iterator(llik.len(), 1, llik.into_iter())
            .to_tsv(&(args.out.to_string() + ".llik.gz"))?;
    }

    z_nk.to_tsv(&(args.out.to_string() + ".latent.gz"))?;

    beta_dk.to_tsv(&(args.out.to_string() + ".dictionary.gz"))?;

    if let Some(param) = delta_bd {
        param.write_tsv(&(args.out.to_string() + ".delta.gz"))?;
    }

    info!("done");
    Ok(())
}

fn run_nystrom_proj(
    xx_dn: Mat,
    delta: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    rank: usize,
    block_size: Option<usize>,
) -> anyhow::Result<(Mat, Mat, Vec<f32>)> {
    let mut xx_dn = xx_dn.clone();
    xx_dn.scale_columns_inplace();
    let (uu_dk, _, _) = xx_dn.rsvd(rank)?;

    let eps = 1e-8;
    let xdepth = 1e4;

    // let mut proj_dk = uu_dk.clone();
    // let safe_dd = dd.map(|x| x + eps);
    // proj_dk.row_iter_mut().for_each(|mut u| {
    //     u.component_div_assign(&safe_dd.transpose());
    // });

    info!(
        "Constructed {} x {} projection matrix",
        uu_dk.nrows(),
        uu_dk.ncols()
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

            if let Some(delta) = delta {
                let batches = full_data_vec.get_batch_membership(lb..ub);

                x_dn.column_iter_mut()
                    .zip(batches)
                    .for_each(|(mut x_j, b)| {
                        let d_j = delta.column(b);
                        let xsum = x_j.sum();
                        let dsum = d_j.sum();
                        let scale = if dsum > 0.0 { xsum / dsum } else { 1.0 };
                        x_j.zip_apply(&d_j, |x_ij, d_ij| *x_ij /= d_ij * scale + eps);
                    });
            }

            let mut log_x_dn = x_dn.map(|x| (x + 0.5).log2());
            log_x_dn.centre_columns_inplace();

            let z_nk = log_x_dn.transpose() * &uu_dk;
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

    Ok((z_nk, uu_dk, vec![]))
}

fn run_asap_embedding<Enc, Dec, LLikFn>(
    data_nd: Mat,
    batch_adjust: Option<&Mat>,
    full_data_vec: &SparseIoVec,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
) -> anyhow::Result<(Mat, Mat, Vec<f32>)>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModule,
    LLikFn: Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
{
    let kk = encoder.dim_latent();

    let mut x_std_nd = data_nd.clone();
    x_std_nd.scale_columns_inplace();
    let (mut z_nk_init, _, _) = x_std_nd.rsvd(kk)?;
    z_nk_init.scale_columns_inplace();

    let kk_init = z_nk_init.ncols();
    if kk_init < kk {
        info!("zero-padding {} columns...", kk - kk_init);
        z_nk_init = z_nk_init.insert_columns(kk_init, kk - kk_init, 0.);
    }

    let mut data_loader = InMemoryData::from_with_aux(&data_nd, &z_nk_init)?;

    let mut vae = Vae::build(encoder, decoder, parameters);

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    info!(
        "Start pre-training with z {} x {} ...",
        z_nk_init.nrows(),
        z_nk_init.ncols()
    );

    let _llik = vae.pretrain_encoder(&mut data_loader, &gaussian_likelihood, &train_config)?;

    info!("Start full log-likelihood training ...");

    let llik_trace =
        vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, &train_config)?;

    info!("Done with training {} epochs", train_config.num_epochs);

    let z_nk = estimate_latent_by_encoder(&full_data_vec, encoder, &train_config, batch_adjust)?;

    info!("Done encoding latent states for all");

    let beta_dk = decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?;
    let beta_dk = Mat::from_tensor(&beta_dk)?;

    Ok((z_nk, beta_dk, llik_trace))
}

fn estimate_latent_by_encoder<Enc>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    train_config: &TrainConfig,
    delta: Option<&Mat>,
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
    let eps = 1e-4;

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&(lb, ub)| {
            let mut x_dn = data_vec.read_columns_dmatrix(lb..ub).expect("read columns");

            if let Some(delta) = delta {
                let batches = data_vec.get_batch_membership(lb..ub);

                x_dn.column_iter_mut()
                    .zip(batches)
                    .for_each(|(mut x_j, b)| {
                        let d_j = delta.column(b);
                        let xsum = x_j.sum();
                        let dsum = d_j.sum();
                        let scale = if dsum > 0.0 { xsum / dsum } else { 1.0 };
                        x_j.zip_apply(&d_j, |x_ij, d_ij| *x_ij /= d_ij * scale + eps);
                    });
            }

            let (d, n) = x_dn.shape();
            let x_nd = Tensor::from_slice(x_dn.as_slice(), (n, d), dev).expect("tensor x_nd");

            let enc = arc_enc.lock().expect("lock");
            let (z_nk, _) = enc.forward_t(&x_nd, false).expect("forward");
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
