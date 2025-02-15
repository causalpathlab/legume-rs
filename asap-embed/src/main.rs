mod asap_collapse_data;
mod asap_common;
mod asap_normalization;
mod asap_random_projection;
mod candle_aux_layers;
mod candle_data_loader;
mod candle_inference;
mod candle_loss_functions;
mod candle_model_decoder;
mod candle_model_encoder;

use asap_collapse_data::CollapsingOps;
use asap_common::*;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use asap_random_projection::RandProjOps;
use candle_data_loader::InMemoryData;
use candle_inference::*;
use candle_loss_functions::topic_likelihood;
use candle_model_decoder::TopicDecoder;
use candle_model_encoder::EncoderModuleT;
use candle_model_encoder::NonNegEncoder;
use clap::{Parser, ValueEnum};
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::Inference;
use matrix_param::traits::ParamIo;
use matrix_util::common_io::{extension, read_lines};
use matrix_util::traits::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about=None)]
///
/// Embedding high-dimensional data (where each data point is a column
/// vector) into a lower-dimensional space in three steps: (1)
/// approximate collapsing to reduce sample size, (2) training an
/// embedding model, and (3) recover latent states by revisiting the
/// data.
///
struct EmbedArgs {
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// should be identical. We can convert `.mtx` to `.zarr` or `.h5`
    /// using `asap-data build`
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Random projection dimension to project the data.
    #[arg(long, short, default_value_t = 30)]
    proj_dim: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short, default_value_t = 10)]
    sort_dim: usize,

    /// Batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// Reference batch name (comma-separated names)
    #[arg(long, value_delimiter(','))]
    reference_batch: Option<Vec<Box<str>>>,

    /// #k-nearest neighbours within each batch
    #[arg(long, short, default_value_t = 10)]
    knn: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long)]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 100)]
    iter_opt: usize,

    /// Block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// Number of latent topics
    #[arg(short, long, default_value_t = 10)]
    latent_topics: usize,

    /// Encoder layers
    #[arg(long, value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    #[arg(long, default_value_t = 1000)]
    epochs: usize,

    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// Candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

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

    let train_config = candle_inference::TrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.minibatch_size,
        num_epochs: args.epochs,
        device: dev.clone(),
        verbose: args.verbose,
    };

    info!("Reading data files...");
    let (mut data_vec, batch_membership) = read_data_vec_membership(args.clone())?;
    info!("Reference batches: {:?}", args.reference_batch);

    let dd = data_vec.num_rows()?;
    let kk = args.latent_topics;

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    info!("Estimate {} topics", kk);
    info!("Encoder layers: {:?}", args.encoder_layers);
    let enc = NonNegEncoder::new(dd, kk, &args.encoder_layers, param_builder.clone())?;
    let dec = TopicDecoder::new(dd, kk, param_builder.clone())?;

    let mut vae = candle_inference::Vae::build(&enc, &dec, &parameters);

    // let nround = args.nround;
    // let mut llik_out = vec![];
    // for rr in 0..nround {
    // 1. Randomly project the columns
    info!("Random projection of data onto {} dims", args.proj_dim);
    let proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
        Some(args.block_size.clone()),
        Some(&batch_membership),
    )?;

    let proj_kn = proj_out.proj;
    info!("Assigning {} columns to samples...", proj_kn.ncols());

    let nsamp = data_vec.assign_columns_to_samples(&proj_kn, Some(args.sort_dim))?;
    info!("at most {} samples are assigned", nsamp);

    // 2. Register batch membership
    if args.batch_files.is_some() && batch_membership.len() > 0 {
        info!("Registering batch information");
        data_vec.register_batches(&proj_kn, &batch_membership)?;
    }

    // 3. Collapsing columns
    info!("Collapsing columns... into {} samples", nsamp);
    let collapse_out = data_vec.collapse_columns(
        args.down_sample,
        args.reference_batch.clone(),
        Some(args.knn),
        Some(args.iter_opt),
    )?;

    let delta = collapse_out.delta.as_ref();
    if let Some(param) = delta {
        param.write_tsv(&(args.out.to_string() + ".delta"))?;
    }

    // 4. Train embedded topic model on the foreground data x1
    let x_nd = collapse_out.mu.posterior_mean().transpose().clone();
    let mut data_loader = InMemoryData::from(&x_nd)?;

    info!("Start training ...");
    let llik = vae.train(&mut data_loader, &topic_likelihood, &train_config)?;
    info!("Done with training {} epochs", train_config.num_epochs);

    // 5. Revisit the data to recover latent states
    info!("Encoding latent states for all...");
    for var in parameters.all_vars() {
        var.to_device(&dev)?;
    }
    let z_nk = estimate_latent(&data_vec, &enc, &train_config, delta)?;

    z_nk.to_tsv(&(args.out.to_string() + ".logit_latent.gz"))?;

    dec.dictionary()
        .log_weight()?
        .to_tsv(&(args.out.to_string() + ".logit_dict.gz"))?;

    let llik_out: Mat = Mat::from_row_iterator(llik.len(), 1, llik.into_iter());
    llik_out.to_tsv(&(args.out.to_string() + ".llik.gz"))?;

    info!("done");
    Ok(())
}

//////////////////////////////////////////////////////////
// Just evaluate latent states based on the encoder net //
//////////////////////////////////////////////////////////

fn estimate_latent(
    data_vec: &SparseIoVec,
    encoder: &NonNegEncoder,
    train_config: &candle_inference::TrainConfig,
    delta: Option<&GammaMatrix>,
) -> anyhow::Result<Tensor> {
    let dev = &train_config.device;
    let ntot = data_vec.num_columns()?;
    let block_size = train_config.batch_size;

    let jobs = create_jobs(ntot, block_size);
    let njobs = jobs.len() as u64;

    let arc_enc = Arc::new(Mutex::new(encoder));

    let delta = delta.map(|x| x.posterior_mean());

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
            (
                lb,
                z_nk.to_device(&candle_core::Device::Cpu).expect("to cpu"),
            )
        })
        .collect::<Vec<_>>();

    chunks.sort_by_key(|&(lb, _)| lb);
    let chunks = chunks.into_iter().map(|(_, z_nk)| z_nk).collect::<Vec<_>>();
    let out = Tensor::cat(&chunks, 0)?;
    Ok(out)
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
