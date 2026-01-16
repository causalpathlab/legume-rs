use crate::embed_common::*;
use crate::senna_input::*;

use candle_core::Device;
use candle_nn::AdamW;
use candle_nn::Optimizer;
use candle_util::candle_data_loader::*;
use candle_util::candle_decoder_topic::*;
use candle_util::candle_loss_functions::topic_likelihood;
use candle_util::candle_model_traits::*;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

use candle_util::candle_encoder_softmax_iaf::*;
use candle_util::candle_model_traits::DecoderModuleT;
use indicatif::{ProgressBar, ProgressDrawTarget};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum AdjMethod {
    Batch,
    Residual,
}

#[derive(Args, Debug)]
pub struct TopicArgs {
    #[arg(
        required = true,
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for results.\n\
		     Specify the output file or prefix for generated files:\n\
		     - {out}.delta.parquet\n\
		     - {out}.dictionary.parquet\n\
		     - {out}.latent.parquet\n"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step."
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection",
        long_help = "Use top {d} components of projection.\n\
		     Number of samples will be less than `2^{d}+1`."
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Ignore batch adjustment",
        long_help = "Ignore batch adjustment.\n\
		     Disables batch effect correction during processing."
    )]
    ignore_batch_effects: bool,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm start projection file",
        long_help = "Warm start from the previous projection (cell x k).\n\
		     Provide a file to initialize the projection."
    )]
    warm_start_proj_file: Option<Box<str>>,

    #[arg(
        short = 'c',
        long,
        default_value_t = 1e4,
        help = "Column sum normalization scale",
        long_help = "Column sum normalization scale (affects decoder only).\n\
		     Adjusts normalization of columns in the decoder."
    )]
    column_sum_norm: f32,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of k-nearest neighbour batches",
        long_help = "Number of k-nearest neighbour batches.\n\
		     Controls the number of batches considered \n\
		     for nearest neighbour search."
    )]
    knn_batches: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch",
        long_help = "Number of k-nearest neighbours within each batch.\n\
		     Controls the number of cells considered \n\
		     for nearest neighbour search within each batch."
    )]
    knn_cells: usize,

    #[arg(
        long,
        value_delimiter(','),
        help = "Reference batch names",
        long_help = "Reference batch names (comma-separated).\n\
		     Specify batches to be used as reference during adjustment."
    )]
    reference_batches: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 30,
        help = "Optimization iterations",
        long_help = "Number of optimization iterations.\n\
		     Controls the number of steps for model optimization."
    )]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing",
        long_help = "Block size (number of columns) for parallel processing.\n\
		     Controls the granularity of parallel computation."
    )]
    block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics",
        long_help = "Number of latent topics.\n\
		     Controls the dimensionality of the latent topic space."
    )]
    n_latent_topics: usize,

    #[arg(
        short = 'f',
        long,
        help = "Number of feature modules",
        long_help = "Number of modules of the features in the encoder model.\n\
		     If not specified, encoder_layers[0] will be used. \n\
		     Giving the number of features modules smaller than that of features,\n\
		     we can expedite model training while not loosing too much of accuracy,\n\
		     as many features are redundant and frequently dropped out.\n"
    )]
    feature_modules: Option<usize>,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder layers",
        long_help = "Encoder layers (comma-separated).\n\
		     Specify the size of each layer in the encoder model.\n\
		     Example: 128,1024,128"
    )]
    encoder_layers: Vec<usize>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of inverse autoregressive flow transformations",
        long_help = "Number of inverse autoregressive flow transformations.\n\
		     Controls the number of flow steps in the model."
    )]
    iaf_trans: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Intensity levels for frequency embedding",
        long_help = "Intensity levels for frequency embedding.\n\
		     Controls the vocabulary size for intensity embedding."
    )]
    vocab_size: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Intensity embedding dimension",
        long_help = "Intensity embedding dimension.\n\
		     Controls the size of the embedding for intensity levels."
    )]
    vocab_emb: usize,

    #[arg(
        long,
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs",
        long_help = "Number of training epochs.\n\
		     Controls how many times the model is trained over the data."
    )]
    epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Data jitter interval",
        long_help = "Data jitter interval.\n\
		     Controls the interval for adding jitter to the collapsed data\n\
		     by posterior resampling during VAE training."
    )]
    jitter_interval: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Minibatch size",
        long_help = "Minibatch size for training.\n\
		     Controls the number of samples per training batch."
    )]
    minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        help = "Learning rate",
        long_help = "Learning rate for optimization.\n\
		     Controls the step size for parameter updates."
    )]
    learning_rate: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Candle device",
        long_help = "Candle device to use for computation.\n\
		     Options: cpu, cuda, metal."
    )]
    device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "A device for cuda",
        long_help = "For cuda or meta, we may want to choose a different device."
    )]
    device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Adjustment method",
        long_help = "Adjust by batch or residual.\n\
		     Choose the method for batch adjustment."
    )]
    adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all columns data",
        long_help = "Preload all the columns data into memory.\n\
		     Improves performance for large datasets."
    )]
    preload_data: bool,

    #[arg(
        long,
        short,
        help = "Verbosity",
        long_help = "Enable verbose output.\n\
		     Prints additional information during execution."
    )]
    verbose: bool,
}

pub fn fit_topic_model(args: &TopicArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let reference = args.reference_batches.as_deref();

    // 1. Read the data with batch membership
    let SparseDataWithBatch {
        data: mut data_vec,
        batch: batch_membership,
        nbatch,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;

    // 2. Take projection results by warm start or projecting it again
    let proj_kn = if let Some(proj_file) = args.warm_start_proj_file.as_deref() {
        use matrix_util::common_io::*;
        let ext = file_ext(proj_file)?;

        let MatWithNames {
            rows: cell_names,
            cols: _,
            mat: proj_nk,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_row_names(&proj_file, Some(0))?,
            _ => Mat::read_data_with_names(&proj_file, &['\t', ',', ' '], Some(0), Some(0))?,
        };

        if data_vec.column_names()? != cell_names {
            return Err(anyhow::anyhow!(
                "warm start projection rows don't match with the data"
            ));
        }

        proj_nk.transpose()
    } else {
        let proj_dim = args.proj_dim.max(args.n_latent_topics);

        let proj_out = data_vec.project_columns_with_batch_correction(
            proj_dim,
            Some(args.block_size),
            Some(&batch_membership),
        )?;

        proj_out.proj
    };

    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    // 3. Batch-adjusted collapsing (pseudobulk)
    // assign pseudobulk samples by proj_kn
    let nsamp = data_vec.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), None)?;

    if !args.ignore_batch_effects && nbatch > 1 {
        info!("Registering batch information");
        data_vec.build_hnsw_per_batch(&proj_kn, &batch_membership)?;
    }

    info!("Collapsing columns into {} pseudobulk samples ...", nsamp);
    let collapsed = data_vec.collapse_columns(
        Some(args.knn_batches),
        Some(args.knn_cells),
        reference,
        Some(args.iter_opt),
    )?;

    let batch_db = collapsed.delta.as_ref();

    if let Some(batch_db) = batch_db {
        let outfile = args.out.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet(Some(&gene_names), batch_names.as_deref(), &outfile)?;
    }

    // 4. Train a topic model on the collapsed data
    let n_topics = args.n_latent_topics;
    let n_vocab = args.vocab_size;
    let d_vocab_emb = args.vocab_emb;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let n_features_decoder = data_vec.num_rows();
    let n_features_encoder = data_vec.num_rows();

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(args.device_no)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(args.device_no)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let encoder = LogSoftmaxIAFEncoder::new(
        LogSoftmaxIAFEncoderArgs {
            n_features: n_features_encoder,
            n_topics,
            n_modules,
            n_vocab,
            d_vocab_emb,
            layers: &args.encoder_layers,
            n_transforms: args.iaf_trans,
        },
        param_builder.clone(),
    )?;

    let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    let scores = train_encoder_decoder(&collapsed, &encoder, &decoder, &parameters, &args)?;

    info!("Writing down the model parameters");

    let gene_names = data_vec.row_names()?;

    decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?
        .to_parquet(
            Some(&gene_names),
            None,
            &(args.out.to_string() + ".dictionary.parquet"),
        )?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    // encoder
    //     .feature_module_membership()?
    //     .to_device(&candle_core::Device::Cpu)?
    //     .to_parquet(
    //         Some(&gene_names),
    //         None,
    //         &(args.out.to_string() + ".feature_module.parquet"),
    //     )?;

    /////////////////////////////////////////////////////
    // evaluate latent states while adjusting the bias //
    /////////////////////////////////////////////////////

    info!("Writing down the latent states");

    let z_nk = evaluate_latent_by_encoder(&data_vec, &encoder, &collapsed, &args)?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    info!("Done");
    Ok(())
}

///////////////////////
// training routines //
///////////////////////

struct TrainScores {
    llik: Vec<f32>,
    kl: Vec<f32>,
}

impl TrainScores {
    fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let mat = Mat::from_columns(&[
            DVec::from_vec(self.llik.clone()),
            DVec::from_vec(self.kl.clone()),
        ]);

        let score_types = vec![
            "log_likelihood".to_string().into_boxed_str(),
            "kl_divergence".to_string().into_boxed_str(),
        ];

        let epochs: Vec<Box<str>> = (0..mat.nrows())
            .into_iter()
            .map(|x| (x + 1).to_string().into_boxed_str())
            .collect();

        mat.to_parquet(Some(&epochs), Some(&score_types), file_path)
    }
}

fn train_encoder_decoder<Enc, Dec>(
    collapsed: &CollapsedOut,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    args: &TopicArgs,
) -> anyhow::Result<TrainScores>
where
    Enc: EncoderModuleT,
    Dec: DecoderModuleT,
{
    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let mut adam = AdamW::new_lr(parameters.all_vars(), args.learning_rate as f64)?;

    let pb = ProgressBar::new(args.epochs as u64);

    if args.verbose {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    let mut llik_trace = Vec::with_capacity(args.epochs);
    let mut kl_trace = Vec::with_capacity(args.epochs);

    info!("Start training VAE...");

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {
        let mixed_nd = collapsed
            .mu_observed
            .posterior_sample()?
            .sum_to_one_columns()
            .scale(args.column_sum_norm)
            .transpose();

        let clean_nd = collapsed.mu_adjusted.as_ref().map(|x| {
            let mut ret: Mat = x.posterior_sample().unwrap();
            ret.sum_to_one_columns_inplace();
            ret.scale_mut(args.column_sum_norm);
            ret.transpose()
        });

        let batch_nd = collapsed.mu_residual.as_ref().map(|x| {
            let ret: Mat = x.posterior_sample().unwrap();
            ret.transpose()
        });

        let mut data_loader = InMemoryData::from(InMemoryArgs {
            input: &mixed_nd,
            input_null: batch_nd.as_ref(),
            output: clean_nd.as_ref(),
            output_null: None,
        })?;

        data_loader.shuffle_minibatch(args.minibatch_size)?;

        for jitter in 0..args.jitter_interval {
            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;

            for b in 0..data_loader.num_minibatch() {
                let mb = data_loader.minibatch_shuffled(b, &dev)?;
                let (z_nk, kl) = encoder.forward_t(&mb.input, mb.input_null.as_ref(), true)?;
                let y_nd = mb.output.unwrap_or(mb.input);
                let (_, llik) = decoder.forward_with_llik(&z_nk, &y_nd, &topic_likelihood)?;

                let loss = (&kl - &llik)?.mean_all()?;
                adam.backward_step(&loss)?;

                let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                let kl_val = kl.sum_all()?.to_scalar::<f32>()?;
                llik_tot += llik_val;
                kl_tot += kl_val;
            }

            kl_trace.push(kl_tot / data_loader.num_minibatch() as f32);
            llik_trace.push(llik_tot / data_loader.num_minibatch() as f32);

            pb.inc(1);

            if args.verbose {
                info!("[{}][{}] {} {}", epoch, jitter, llik_tot, kl_tot);
            }
        }
    }
    pb.finish_and_clear();

    info!("done model training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}

fn evaluate_latent_by_encoder<Enc>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    collapsed: &CollapsedOut,
    args: &TopicArgs,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync,
{
    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let ntot = data_vec.num_columns();
    let kk = encoder.dim_latent();

    let block_size = args.minibatch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;
    let arc_enc = Arc::new(encoder);

    let delta = match args.adj_method {
        AdjMethod::Batch => collapsed.delta.as_ref(),
        AdjMethod::Residual => collapsed.mu_residual.as_ref(),
    }
    .map(|x| x.posterior_mean().clone())
    .map(|delta_db| {
        delta_db
            .to_tensor(&dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
    });

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&block| match args.adj_method {
            AdjMethod::Residual => {
                evaluate_with_residuals(block, data_vec, arc_enc.clone(), &dev, delta.as_ref())
            }
            AdjMethod::Batch => {
                evalulate_with_batch(block, data_vec, arc_enc.clone(), &dev, delta.as_ref())
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    chunks.par_sort_by_key(|&(lb, _)| lb);
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

fn evalulate_with_batch<Enc>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: Arc<&Enc>,
    dev: &Device,
    delta_bd: Option<&Tensor>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: EncoderModuleT,
{
    let (lb, ub) = block;
    let x0_nd = delta_bd.map(|delta_bm| {
        let batches = data_vec
            .get_batch_membership(lb..ub)
            .into_iter()
            .map(|x| x as u32);
        let batches = Tensor::from_iter(batches, dev).unwrap();
        delta_bm.index_select(&batches, 0).expect("expand delta")
    });

    let x_nd = data_vec
        .read_columns_tensor(lb..ub)?
        .to_device(dev)?
        .transpose(0, 1)?;

    let (z_nk, _) = encoder.forward_t(&x_nd, x0_nd.as_ref(), false)?;
    let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

fn evaluate_with_residuals<Enc>(
    block: (usize, usize),
    data_vec: &SparseIoVec,
    encoder: Arc<&Enc>,
    dev: &Device,
    delta_bp: Option<&Tensor>,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: EncoderModuleT,
{
    let (lb, ub) = block;
    let x0_nd = delta_bp.map(|delta_bm| {
        let groups = data_vec
            .get_group_membership(lb..ub)
            .expect("failed to get group membership")
            .into_iter()
            .map(|x| x as u32);
        let groups = Tensor::from_iter(groups, dev).unwrap();
        delta_bm.index_select(&groups, 0).expect("expand delta")
    });

    let x_nd = data_vec
        .read_columns_tensor(lb..ub)?
        .to_device(dev)?
        .transpose(0, 1)?;

    let (z_nk, _) = encoder.forward_t(&x_nd, x0_nd.as_ref(), false)?;
    let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}
