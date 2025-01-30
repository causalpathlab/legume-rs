mod candle_data_loader;
mod candle_inference;
mod candle_loss_functions;
mod candle_model_decoder;
mod candle_model_encoder;
mod collapse_data;
mod common;
mod normalization;
mod random_projection;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use asap_embed::candle_data_loader::InMemoryData;
use candle_inference::*;
use clap::{Parser, ValueEnum};
use collapse_data::CollapsingOps;
use log::info;
use matrix_param::traits::Inference;
use matrix_util::common_io::{extension, read_lines};
use matrix_util::traits::*;
use random_projection::RandProjOps;
use std::sync::Arc;

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
    #[arg(long, short, required = true)]
    proj_dim: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short, default_value_t = 10)]
    sort_dim: usize,

    /// Batch membership files. Each bach file should correspond to
    /// each data file.
    #[arg(long, short)]
    batch_files: Option<Vec<Box<str>>>,

    /// Reference batch name
    #[arg(long, short)]
    reference_batch: Option<Box<str>>,

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
    #[arg(long, default_value_t = 10)]
    latent_topics: usize,

    /// ETM encoder layers
    #[arg(long, default_values_t = vec![128,16])]
    etm_layers: Vec<usize>,

    /// Candle device
    #[arg(value_enum, default_value = "cpu")]
    device: ComputeDevice,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = EmbedArgs::parse();

    info!("Reading data files...");
    let (mut data_vec, batch_membership) = read_data_vec_membership(args.clone())?;

    // 1. Randomly project the columns
    info!("Random projection of data onto {} dims", args.proj_dim);
    let proj_res = data_vec.project_columns(args.proj_dim, Some(args.block_size.clone()))?;
    proj_res
        .basis
        .to_tsv(&(args.out.to_string() + ".basis.gz"))?;

    let proj_kn = proj_res.proj;

    proj_kn
        .transpose()
        .to_tsv(&(args.out.to_string() + ".proj.gz"))?;

    info!("Assigning {} columns to samples...", proj_kn.ncols());

    let nsamp = data_vec.assign_columns_to_samples(&proj_kn, Some(args.sort_dim))?;
    info!("at most {} samples are assigned", nsamp);

    // 2. Register batch membership
    info!("Registering batch-specific information");
    data_vec.register_batches(&proj_kn, &batch_membership)?;

    // 3. Collapsing columns
    info!("Collapsing columns... into {} samples", nsamp);
    let collapse_out = data_vec.collapse_columns(
        args.down_sample,
        args.reference_batch.clone(),
        Some(args.knn),
        Some(args.iter_opt),
    )?;

    // 4. Train embedded topic model
    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let train_cofig = candle_inference::TrainingConfig {
        learning_rate: 1e-3,
        batch_size: 10,
        num_epochs: 100,
        device: dev,
        verbose: true,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder = candle_nn::VarBuilder::from_varmap(
        &parameters,
        candle_core::DType::F32,
        &train_cofig.device,
    );

    {
        let x_nd = collapse_out.mu.posterior_mean().transpose();
        let (nn, dd) = x_nd.shape();
        let data = data_loader(&x_nd)?;
        let kk = args.latent_topics;
        let layers = args.etm_layers;
        let enc = candle_model_encoder::NonNegEncoder::new(dd, kk, &layers, param_builder.clone())?;
        let dec = candle_model_decoder::ETMDecoder::new(dd, kk, param_builder.clone())?;
        let vae = candle_inference::Vae::build(enc, dec, &parameters);
    }

    // info!("writing down the results...");
    // ret.mu.write_tsv(&(args.out.to_string() + ".mu"))?;

    // if let Some(delta) = &ret.delta {
    //     delta.write_tsv(&(args.out.to_string() + ".delta"))?;
    // }

    // if let Some(gamma) = &ret.gamma {
    //     gamma.write_tsv(&(args.out.to_string() + ".gamma"))?;
    // }

    info!("done");
    Ok(())
}

fn data_loader(data_dn: &DMatrix<f32>) -> candle_core::Result<InMemoryData> {
    InMemoryData::from_dmatrix(&data_dn)
}

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
