mod asap_routines_latent_representation;
mod asap_routines_post_process;
mod embed_common;

use asap_alg::random_projection::*;
use embed_common::*;

use matrix_param::io::ParamIo;
use matrix_param::traits::{Inference, TwoStatParam};
use matrix_util::common_io::{self, extension, read_lines, remove_file, write_types};
use matrix_util::dmatrix_util::row_membership_matrix;
use matrix_util::traits::*;

use asap_routines_latent_representation::*;
use asap_routines_post_process::*;

use asap_alg::collapse_data::CollapsingOps;
use asap_alg::random_projection::RandProjOps;

use candle_util::candle_decoder_topic::*;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::DecoderModuleT;

use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum DecoderModel {
    Topic,
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
#[command(name = "Embedding ASAP", version, about, long_about, term_width = 80)]
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
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// Use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = false)]
    ignore_batch_effects: bool,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 15)]
    iter_opt: usize,

    /// block_size (# columns) for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 'k', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// targeted number of row feature modules
    #[arg(short = 'r', long, default_value_t = 1000)]
    n_row_modules: usize,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 100)]
    vocab_size: usize,

    /// intensity embedding dimension
    #[arg(long, default_value_t = 3)]
    vocab_emb: usize,

    /// # training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

    /// choose a decoder model
    #[arg(long = "model", short = 'm', value_enum, default_value = "topic")]
    decoder_model: DecoderModel,

    /// save intermediate projection results
    #[arg(long)]
    save_intermediate: bool,

    /// save batch-adjusted data
    #[arg(long)]
    save_adjusted: bool,

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
        num_pretrain_epochs: 0,
        device: dev.clone(),
        verbose: args.verbose,
    };
    info!("Reading data files...");

    let (mut data_vec, batch_membership) = read_data_vec_membership(args.clone())?;

    if args.decoder_model != DecoderModel::Proj {
        info!("Encoder layers: {:?}", args.encoder_layers);
    }

    /////////////////////////////////////
    // 1. Randomly project the columns //
    /////////////////////////////////////

    let proj_dim = args.proj_dim.max(args.n_latent_topics);

    info!("Random projection of data onto {} dims", proj_dim);
    let proj_out = data_vec.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(&batch_membership),
    )?;

    let proj_kn = proj_out.proj;
    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    if args.save_intermediate {
        let cell_names = data_vec.column_names()?;
        proj_kn.transpose().to_parquet(
            Some(&cell_names),
            None,
            &(args.out.to_string() + ".randproj.parquet"),
        )?;
    }

    let nsamp =
        data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    //////////////////////////////////
    // 2. Register batch membership //
    //////////////////////////////////

    if !args.ignore_batch_effects {
        if !batch_membership.is_empty() {
            info!("Registering batch information");
            data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
        }
    }

    ///////////////////////////
    // 3. Collapsing columns //
    ///////////////////////////

    info!("Collapsing columns into {} pseudobulk samples ...", nsamp);
    let collapse_out = data_vec.collapse_columns(
        Some(args.knn_batches),
        Some(args.knn_cells),
        Some(args.iter_opt),
    )?;

    let batch_db = collapse_out.delta.as_ref();

    if let Some(delta_db) = batch_db.map(|x| x.posterior_mean()) {
        if args.save_adjusted {
            info!("Generating batch-adjusted data...");

            let triplets = triplets_adjusted_by_batch(&data_vec, delta_db)?;

            let mtx_shape = (
                data_vec.num_rows()?,
                data_vec.num_columns()?,
                triplets.len(),
            );

            let backend_file = args.out.to_string() + ".adjusted.zarr";
            let backend = SparseIoBackend::Zarr;
            remove_file(&backend_file)?;

            let mut adjusted_data = create_sparse_from_triplets(
                triplets,
                mtx_shape,
                Some(&backend_file),
                Some(&backend),
            )?;

            adjusted_data.register_row_names_vec(&data_vec.row_names()?);
            adjusted_data.register_column_names_vec(&data_vec.column_names()?);

            info!("Batch-adjusted backend: {}", backend_file);
        }
    }

    let rows = data_vec.row_names()?;

    if args.save_intermediate {
        info!("Saving intermediate results...");
        collapse_out.mu_observed.to_parquet(
            Some(&rows),
            None,
            &(args.out.to_string() + ".collapsed.observed.parquet"),
        )?;
        info!("Wrote {}", args.out.to_string() + ".collapsed.observed");
        if let Some(param) = collapse_out.mu_adjusted.as_ref() {
            param.to_parquet(
                Some(&rows),
                None,
                &(args.out.to_string() + ".collapsed.adjusted.parquet"),
            )?;
            info!("Wrote {}", args.out.to_string() + ".collapsed.adjusted");
        }
        if let Some(param) = collapse_out.mu_residual.as_ref() {
            param.to_parquet(
                Some(&rows),
                None,
                &(args.out.to_string() + ".collapsed.residual.parquet"),
            )?;
            info!("Wrote {}", args.out.to_string() + ".collapsed.residual");
        }
    }

    if let Some(batch_db) = batch_db {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet(Some(&gene_names), batch_names.as_deref(), &outfile)?;
    }

    /////////////////////////////////////////////////////////
    // 4. Train embedded topic model on the collapsed data //
    /////////////////////////////////////////////////////////

    if args.decoder_model == DecoderModel::Proj {
        let x_dn = match collapse_out.mu_adjusted.as_ref() {
            Some(adj) => adj,
            None => &collapse_out.mu_observed,
        };

        let nystrom_out = do_nystrom_proj(
            x_dn.posterior_log_mean().clone(),
            batch_db.map(|x| x.posterior_mean()),
            &data_vec,
            args.n_latent_topics,
            Some(args.block_size),
        )?;

        let cell_names = data_vec.column_names()?;
        let gene_names = data_vec.row_names()?;

        nystrom_out.latent_nk.to_parquet(
            Some(&cell_names),
            None,
            &(args.out.to_string() + ".latent.parquet"),
        )?;

        nystrom_out.dictionary_dk.to_parquet(
            Some(&gene_names),
            None,
            &(args.out.to_string() + ".dictionary.parquet"),
        )?;

        info!("Done");
        return Ok(());
    }

    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;

    let aggregate_rows = if collapse_out.mu_observed.nrows() > args.n_row_modules {
        let log_x_nd = match collapse_out.mu_adjusted.as_ref() {
            Some(x) => x.posterior_log_mean().transpose().clone(),
            _ => collapse_out
                .mu_observed
                .posterior_log_mean()
                .transpose()
                .clone(),
        };
        let kk = (args.n_row_modules as f32).log2().ceil() as usize + 1;
        info!(
            "reduce data features: {} -> {}",
            log_x_nd.ncols(),
            args.n_row_modules
        );
        row_membership_matrix(binary_sort_columns(&log_x_nd, kk)?)?
    } else {
        let d = collapse_out.mu_observed.nrows();
        Mat::identity(d, d)
    };

    let parameters = candle_nn::VarMap::new();
    let dev = &train_config.device;
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, dev);

    let mixed_dn = &collapse_out.mu_observed;
    let clean_dn = collapse_out.mu_adjusted.as_ref();
    let batch_dn = collapse_out.mu_residual.as_ref();

    // encoder input can be modularized
    let input_nm = mixed_dn.posterior_mean().transpose().clone() * &aggregate_rows;
    let batch_nm = batch_dn.map(|x| x.posterior_mean().transpose().clone() * &aggregate_rows);

    // output decoder should maintain the original dimension
    let output_nd = match clean_dn {
        Some(x) => x.posterior_mean().transpose().clone(),
        _ => mixed_dn.posterior_mean().transpose().clone(),
    };

    ///////////////////////////////////////////////////
    // training variational autoencoder architecture //
    ///////////////////////////////////////////////////

    let n_features_encoder = input_nm.ncols();
    let n_features_decoder = output_nd.ncols();

    let encoder = LogSoftmaxEncoder::new(
        n_features_encoder,
        n_topics,
        n_vocab,
        d_vocab_emb,
        &args.encoder_layers,
        param_builder.clone(),
    )?;

    let log_likelihood = match args.decoder_model {
        DecoderModel::Topic => {
            info!(
                "input: {} -> encoder -> decoder -> output: {}",
                n_features_encoder, n_features_decoder
            );
            let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

            let llik = train_encoder_decoder(
                &input_nm,
                &output_nd,
                batch_nm.as_ref(),
                &encoder,
                &decoder,
                &parameters,
                &loss_func::topic_likelihood,
                &train_config,
            )?;

            let gene_names = data_vec.row_names()?;

            decoder
                .get_dictionary()?
                .to_device(&candle_core::Device::Cpu)?
                .to_parquet(
                    Some(&gene_names),
                    None,
                    &(args.out.to_string() + ".dictionary.parquet"),
                )?;

            llik
        }

        _ => {
            unimplemented!("This decoder model is not yet implemented");
        }
    };

    write_types::<f32>(&log_likelihood, &(args.out.to_string() + ".llik.gz"))?;

    let delta_db = batch_db.map(|x| x.posterior_mean());
    let z_nk = evaluate_latent_by_encoder(
        &data_vec,
        &encoder,
        &aggregate_rows,
        &train_config,
        delta_db,
    )?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    info!("done");
    Ok(())
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

        match extension(data_file)?.as_ref() {
            "zarr" => {
                assert_eq!(backend, SparseIoBackend::Zarr);
            }
            "h5" => {
                assert_eq!(backend, SparseIoBackend::HDF5);
            }
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", data_file)),
        };

        let data = open_sparse_matrix(data_file, &backend)?;
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
    let mut batch_membership = Vec::with_capacity(data_vec.len());

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# batch files != # of data files"));
        }

        for batch_file in batch_files.iter() {
            info!("Reading batch file: {}", batch_file);
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
        }
    } else {
        for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
            let data_file = args.data_files[id].clone();
            let (_dir, base, _ext) = common_io::dir_base_ext(&data_file)?;
            batch_membership.extend(vec![base; nn]);
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
