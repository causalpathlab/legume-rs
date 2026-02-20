use crate::embed_common::*;
use crate::routines_post_process::*;
use crate::routines_pre_process::*;

use candle_util::candle_model_traits::EncoderModuleT;
use dashmap::DashMap as HashMap;
use data_beans_alg::normalization::NormalizeDistance;
use matrix_util::common_io::extension;
use matrix_util::dmatrix_util::concatenate_horizontal;

use candle_util::candle_data_loader::*;
use candle_util::candle_decoder_topic::*;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_inference::TrainConfig;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_traits::DecoderModuleT;
use candle_util::candle_vae_inference::*;
use indicatif::ProgressBar;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Args, Debug)]
pub struct DeconvArgs {
    /// single-cell data (`.zarr` or `.h5`)
    #[arg(short = 's', long, value_delimiter = ',', required = true)]
    sc_data_files: Vec<Box<str>>,

    /// bulk data files (`.parquet`, `.tsv.gz`, `.csv.gz`)
    /// where the first column corresponds to gene names
    #[arg(short = 'x', long, required = true, value_delimiter = ',')]
    bulk_data_files: Vec<Box<str>>,

    /// random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 10)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// #k-nearest neighbours within each bulk
    #[arg(long, default_value_t = 50)]
    knn_bulk: usize,

    /// reference batch names
    #[arg(long, value_delimiter(','))]
    reference_batches: Option<Vec<Box<str>>>,

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
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// number of modules of the features in the encoder model.
    /// If not specified, `encoder_layers[0]` will be used.
    #[arg(short = 'm', long)]
    feature_modules: Option<usize>,

    /// to reduce row features (#gene modules ~ 2^r)
    #[arg(short = 'r', long, default_value_t = 10)]
    n_row_proj_dim: usize,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// # training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// data jitter interval
    #[arg(long, short = 'j', default_value_t = 5)]
    jitter_interval: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

}

struct BulkDataOut {
    genes: Vec<Box<str>>,
    samples: Vec<Box<str>>,
    data: Mat,
}

/// a helper function to read bulk data with the matching row names
/// * `bulk_data_files` - bulk file names
/// * `sc_data` - opened `SparseIoVec`
///
fn read_bulk_data_consistent_with_sc(
    bulk_data_files: &[Box<str>],
    sc_data: &SparseIoVec,
) -> anyhow::Result<BulkDataOut> {
    let genes = sc_data.row_names()?;
    let gene_to_position: HashMap<Box<str>, usize> = genes
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();

    let ngenes = gene_to_position.len();
    info!("use {} genes as common features", ngenes);

    let mut samples = vec![];
    let mut bulk_data_vec = vec![];

    for bulk_file in bulk_data_files {
        let MatWithNames {
            rows: raw_genes,
            cols: raw_samples,
            mat: raw_ds,
        } = match extension(bulk_file.as_ref())?.as_ref() {
            "parquet" => Mat::from_parquet(bulk_file.as_ref())?,
            _ => Mat::read_data(bulk_file.as_ref(), &['\t', ','], None, Some(0), None, None)?,
        };

        let ncols = raw_samples.len();

        let mut padded_ds = Mat::zeros(ngenes, ncols);
        for (i, g) in raw_genes.iter().enumerate() {
            if let Some(r) = gene_to_position.get(g) {
                padded_ds.row_mut(*r.value()).copy_from(&raw_ds.row(i));
            }
        }

        samples.extend(raw_samples);
        bulk_data_vec.push(padded_ds);
    }
    let bulk_data = concatenate_horizontal(&bulk_data_vec)?;

    info!(
        "Read bulk data {} genes x {} samples",
        ngenes,
        samples.len()
    );
    Ok(BulkDataOut {
        genes,
        samples,
        data: bulk_data,
    })
}

/// a master function to perform the deconvolution of bulk data files
///
pub fn fit_deconv(args: &DeconvArgs) -> anyhow::Result<()> {
    info!("Opening single-cell data files...");
    // 1. Read sc data with batch membership
    let SparseDataWithBatch {
        data: mut sc_data,
        batch: sc_batch,
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.sc_data_files.clone(),
        batch_files: args.batch_files.clone(),
    })?;

    // 2. Read bulk data
    info!("Reading in bulk data files...");
    let BulkDataOut {
        genes: bulk_genes,
        samples: bulk_samples,
        data: bulk_data,
    } = read_bulk_data_consistent_with_sc(&args.bulk_data_files, &sc_data)?;

    if bulk_genes != sc_data.row_names()? {
        return Err(anyhow::anyhow!("bulk and sc data gene names should match"));
    }

    info!("Finding a shared basis matrix for random projection...");
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let rand_proj = sc_data.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(sc_batch.as_ref()),
    )?;

    let proj_kn = rand_proj.proj.scale_columns();

    let nsamp =
        sc_data.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    info!("Registering batch information");
    sc_data.build_hnsw_per_batch(&proj_kn, &sc_batch)?;

    // 3. Batch-adjusted collapsing (pseudobulk)
    info!("Constructing PB in the sc data... into {} samples", nsamp);
    let collapsed_sc = sc_data.collapse_columns(
        Some(args.knn_batches),
        Some(args.knn_cells),
        args.reference_batches.as_deref(),
        Some(args.iter_opt),
    )?;

    let bias_db = collapsed_sc
        .delta
        .as_ref()
        .map(|x| x.posterior_mean().clone());
    let bulk_dm =
        sc_data.project_onto_bulk(&bulk_data, &rand_proj, bias_db.as_ref(), args.knn_cells)?;

    // 4. Train topic model on the collapsed data
    let n_topics = args.n_latent_topics;
    let n_modules = args.feature_modules.unwrap_or(args.encoder_layers[0]);

    let n_features_decoder = sc_data.num_rows();
    let n_features_encoder = sc_data.num_rows();

    let dev = match args.device {
        ComputeDevice::Metal => candle_core::Device::new_metal(0)?,
        ComputeDevice::Cuda => candle_core::Device::new_cuda(0)?,
        _ => candle_core::Device::Cpu,
    };

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: n_features_encoder,
            n_topics,
            n_modules,
            layers: &args.encoder_layers,
            use_sparsemax: false,
            temperature: 1.0,
        },
        param_builder.clone(),
    )?;

    let decoder = TopicDecoder::new(n_features_decoder, n_topics, param_builder.clone())?;

    info!(
        "input: {} -> encoder -> decoder -> output: {}",
        n_features_encoder, n_features_decoder
    );

    let mut train_config = TrainConfig {
        learning_rate: args.learning_rate,
        batch_size: args.minibatch_size,
        num_epochs: args.epochs,
        num_pretrain_epochs: 0,
        device: dev.clone(),
        verbose: log::log_enabled!(log::Level::Info),
        show_progress: true,
    };

    info!("Set up training data");

    let pb = ProgressBar::new(train_config.num_epochs as u64);

    let mut vae = Vae::build(&encoder, &decoder, &parameters);
    let mut llik_sc = Vec::with_capacity(train_config.num_epochs);
    let mut llik_bulk = Vec::with_capacity(train_config.num_epochs);

    for epoch in (0..args.epochs).step_by(args.jitter_interval) {
        // training with stochastic sc data

        let mixed_nd = collapsed_sc.mu_observed.posterior_sample()?.transpose();
        let clean_nd = collapsed_sc.mu_adjusted.as_ref().map(|x| {
            let ret: Mat = x.posterior_sample().unwrap();
            ret.transpose()
        });

        let batch_nd = collapsed_sc.mu_residual.as_ref().map(|x| {
            let ret: Mat = x.posterior_sample().unwrap();
            ret.transpose()
        });

        let mut data_loader = InMemoryData::from(InMemoryArgs {
            input: &mixed_nd,
            input_null: batch_nd.as_ref(),
            output: clean_nd.as_ref(),
            output_null: None,
        })?;

        data_loader.shuffle_minibatch(args.block_size)?;

        train_config.verbose = false;
        train_config.show_progress = true;
        train_config.num_epochs = args.jitter_interval;

        let llik = vae.train_encoder_decoder(
            &mut data_loader,
            &loss_func::topic_likelihood,
            &train_config,
        )?;

        llik_sc.extend(llik);

        // training with bulk data
        let mut bulk_data_loader = InMemoryData::from(InMemoryArgs {
            input: &bulk_dm.transpose(),
            input_null: None,
            output: None,
            output_null: None,
        })?;

        train_config.verbose = false;
        train_config.show_progress = true;
        train_config.num_epochs = args.jitter_interval;

        let llik = vae.train_encoder_decoder(
            &mut bulk_data_loader,
            &loss_func::topic_likelihood,
            &train_config,
        )?;

        llik_bulk.extend(llik);

        pb.inc(args.jitter_interval as u64);

        info!(
            "[{}] log-likelihood: {} {}",
            epoch + args.jitter_interval,
            llik_sc.last().ok_or(anyhow::anyhow!("No SC loss values recorded"))?,
            llik_bulk.last().ok_or(anyhow::anyhow!("No bulk loss values recorded"))?,
        );
    }

    pb.finish_and_clear();

    info!("Finished {} epochs", train_config.num_epochs);

    info!("Writing down the model parameters");

    let gene_names = sc_data.row_names()?;

    decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?
        .to_parquet(
            Some(&gene_names),
            None,
            &(args.out.to_string() + ".dictionary.parquet"),
        )?;

    write_types::<f32>(&llik_sc, &(args.out.to_string() + ".log_likelihood.gz"))?;

    write_types::<f32>(
        &llik_bulk,
        &(args.out.to_string() + ".log_likelihood_bulk.gz"),
    )?;

    encoder
        .feature_module_membership()?
        .to_device(&candle_core::Device::Cpu)?
        .to_parquet(
            Some(&gene_names),
            None,
            &(args.out.to_string() + ".feature_module.parquet"),
        )?;

    /////////////////////////////////////////////////////
    // evaluate latent states while adjusting the bias //
    /////////////////////////////////////////////////////

    info!("Writing down the latent states");

    let z_nk = evaluate_latent_by_encoder(&sc_data, &encoder, &train_config, bias_db.as_ref())?;

    let cell_names = sc_data.column_names()?;

    z_nk.to_parquet(
        Some(&cell_names),
        None,
        &(args.out.to_string() + ".latent.parquet"),
    )?;

    let x_md = bulk_dm.transpose().to_tensor(&dev)?;
    let (z_mk, _) = encoder.forward_t(&x_md, None, false)?;
    let z_mk = Mat::from_tensor(&z_mk)?;

    z_mk.to_parquet(
        Some(&bulk_samples),
        None,
        &(args.out.to_string() + ".deconv.parquet"),
    )?;

    info!("Done");
    Ok(())
}

trait ProjectOntoBulk {
    fn project_onto_bulk(
        &self,
        target: &Mat,
        rand_proj: &RandColProjOut,
        bias_db: Option<&Mat>,
        knn: usize,
    ) -> anyhow::Result<Mat>;
}

impl ProjectOntoBulk for SparseIoVec {
    fn project_onto_bulk(
        &self,
        target: &Mat,
        rand_proj: &RandColProjOut,
        bias_db: Option<&Mat>,
        knn: usize,
    ) -> anyhow::Result<Mat> {
        let basis_dk = &rand_proj.basis;
        let ln_x = target.map(|x| x.ln_1p()).normalize_columns();
        let bulk_km = (basis_dk.transpose() * &ln_x).scale_columns();
        let norm_target = 2_f32.ln();

        let mut imputed_dm = Mat::zeros(target.nrows(), target.ncols());

        for (i, query) in bulk_km.column_iter().enumerate() {
            let (mut y, cells, distances) = self.query_columns_by_data_csc(query, knn)?;

            let weights = distances.into_iter().normalized_exp(norm_target);
            let denom = weights.iter().sum::<f32>().max(1e-8);
            let weights = DVec::from_vec(weights).unscale(denom);

            if let Some(denom_db) = bias_db {
                let batches = self.get_batch_membership(cells.into_iter());
                y.adjust_by_division_of_selected_inplace(denom_db, &batches);
            }
            imputed_dm.column_mut(i).copy_from(&(y * weights));
        }

        Ok(imputed_dm)
    }
}
