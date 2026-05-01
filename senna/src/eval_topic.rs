use crate::embed_common::*;
use crate::topic::eval::{
    build_gene_remap, evaluate_latent_with_gene_remap, EvaluateLatentConfig, GeneRemap,
};
use crate::topic::model_metadata::{load_coarsening, load_dictionary, TopicModelMetadata};

use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use candle_util::candle_decoder_nb_mixture::{
    NbMixtureTopicDecoder, DECODER_NAME as NBMIXTURE_NAME,
};
use candle_util::candle_decoder_topic::*;
use candle_util::candle_decoder_vmf_topic::VmfTopicDecoder;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_model_traits::*;
use candle_util::candle_topic_refinement::TopicRefinementConfig;
use data_beans::sparse_io_vector::SparseIoVec;
use log::info;

type Mat = nalgebra::DMatrix<f32>;

#[derive(Args, Debug)]
pub struct EvalTopicArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5)",
        long_help = "Sparse backends to embed with the pre-trained model.\n\
                     Gene sets may differ from training; missing genes are padded\n\
                     and batch delta is re-estimated from the frozen dictionary."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "Trained model prefix",
        long_help = "Prefix passed to `senna topic -o`. Loads:\n  \
                     {model}.dictionary.parquet   frozen gene × topic dictionary\n  \
                     {model}.metadata.json        model architecture metadata\n  \
                     {model}.safetensors          encoder+decoder weights\n  \
                     {model}.coarsening.json      feature coarsening (if used)"
    )]
    pub(crate) model: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file prefix",
        long_help = "Writes {out}.latent.parquet (cell × topic log-softmax proportions)."
    )]
    pub(crate) out: Box<str>,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell in the same order as its\n\
                     matching data file. Example: batch1.tsv,batch2.tsv"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = 500, help = "Evaluation minibatch size")]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        help = "Cells per delta-estimation block (omit for auto-scaling by feature count)"
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(long, help = "Load all columns into memory before evaluation")]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-cell refinement steps at inference (0 = off)",
        long_help = "Gradient steps that optimize topic logits against the frozen\n\
                     decoder likelihood, anchored to the encoder output by L2."
    )]
    pub(crate) refine_steps: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Learning rate for inference-time refinement"
    )]
    pub(crate) refine_lr: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 anchor strength for inference-time refinement"
    )]
    pub(crate) refine_reg: f64,

    #[arg(short, long, help = "Verbose logging")]
    pub(crate) verbose: bool,
}

pub fn eval_topic_model(args: &EvalTopicArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let metadata = TopicModelMetadata::load(&args.model)?;
    info!(
        "Model: {} topics, {} features (encoder {}), decoders: {:?}",
        metadata.n_topics,
        metadata.n_features_full,
        metadata.n_features_encoder,
        metadata.decoder_types,
    );

    let (training_genes, beta_dk) = load_dictionary(&args.model)?;
    let coarsening = if metadata.has_coarsening {
        load_coarsening(&args.model)?
    } else {
        None
    };

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;
    let mut data_vec = loaded.data;
    data_vec.register_batch_membership(&loaded.batch);
    info!(
        "New data: {} features × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );

    let new_genes = data_vec.row_names()?;
    let gene_remap = build_gene_remap(&training_genes, &new_genes);

    let min_overlap = (training_genes.len() as f32 * 0.1) as usize;
    anyhow::ensure!(
        gene_remap.n_mapped >= min_overlap,
        "Too few genes overlap: {}/{} mapped (need at least {})",
        gene_remap.n_mapped,
        training_genes.len(),
        min_overlap,
    );

    // Skip remap when genes match training exactly (same set, same order)
    let needs_remap = gene_remap
        .new_to_train
        .iter()
        .enumerate()
        .any(|(i, opt)| *opt != Some(i))
        || new_genes.len() != training_genes.len();

    let gene_remap_opt = if needs_remap {
        info!(
            "Gene remapping enabled ({} → {} features)",
            new_genes.len(),
            training_genes.len()
        );
        Some(gene_remap)
    } else {
        info!("Genes match training — no remapping needed");
        None
    };

    let delta_db = crate::topic::predict_common::estimate_delta(
        &data_vec,
        &beta_dk,
        metadata.theta_mean.as_deref(),
        gene_remap_opt.as_ref(),
        args.block_size,
    )?;

    let cpu_dev = candle_core::Device::Cpu;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);

    let encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: metadata.n_features_encoder,
            n_topics: metadata.n_topics,
            layers: &metadata.encoder_hidden,
        },
        &parameters,
        vb.clone(),
    )?;

    let refine_config = if args.refine_steps > 0 {
        Some(TopicRefinementConfig {
            num_steps: args.refine_steps,
            learning_rate: args.refine_lr,
            regularization: args.refine_reg,
        })
    } else {
        None
    };
    let adj_method = AdjMethod::Batch;

    let eval_inputs = EvalInputs {
        metadata: &metadata,
        parameters: &mut parameters,
        vb: &vb,
        model_prefix: &args.model,
        encoder: &encoder,
        data_vec: &data_vec,
        delta_db: delta_db.as_ref(),
        gene_remap: gene_remap_opt.as_ref(),
        coarsening: coarsening.as_ref(),
        cpu_dev: &cpu_dev,
        adj_method: &adj_method,
        minibatch_size: args.minibatch_size,
        refine_config: refine_config.as_ref(),
    };

    // Dispatch on trained decoder type so the right concrete type is
    // registered in the VarMap (safetensors keys must match) and used for
    // refinement. Multi-decoder models use the first decoder type —
    // refinement against the primary dictionary.
    let decoder_name = metadata
        .decoder_types
        .first()
        .map_or("multinom", std::convert::AsRef::as_ref);
    info!("Evaluating latent states on CPU (decoder: {decoder_name})");
    let z_nk = match decoder_name {
        "multinom" => eval_with_decoder::<MultinomTopicDecoder>(eval_inputs)?,
        "nb" => eval_with_decoder::<NbTopicDecoder>(eval_inputs)?,
        name if name == NBMIXTURE_NAME => eval_with_decoder::<NbMixtureTopicDecoder>(eval_inputs)?,
        "vmf" => eval_with_decoder::<VmfTopicDecoder>(eval_inputs)?,
        other => anyhow::bail!("unsupported decoder type in metadata: {other}"),
    };

    let cell_names = data_vec.column_names()?;
    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&axis_id_names("T", z_nk.ncols())),
    )?;
    info!("Done — wrote {}.latent.parquet", args.out);
    Ok(())
}

/// Estimate per-batch delta from new data pseudobulk and trained dictionary.
/// Shared inputs for the decoder-dispatch path.
struct EvalInputs<'a> {
    metadata: &'a TopicModelMetadata,
    parameters: &'a mut candle_nn::VarMap,
    vb: &'a candle_nn::VarBuilder<'a>,
    model_prefix: &'a str,
    encoder: &'a LogSoftmaxEncoder,
    data_vec: &'a SparseIoVec,
    delta_db: Option<&'a Mat>,
    gene_remap: Option<&'a GeneRemap>,
    coarsening: Option<&'a FeatureCoarsening>,
    cpu_dev: &'a candle_core::Device,
    adj_method: &'a AdjMethod,
    minibatch_size: usize,
    refine_config: Option<&'a TopicRefinementConfig>,
}

/// Register decoder params, load safetensors, and run latent evaluation
/// with the concrete decoder type matching the trained model.
fn eval_with_decoder<Dec>(inputs: EvalInputs<'_>) -> anyhow::Result<Mat>
where
    Dec: DecoderModuleT + NewDecoder + Send + Sync,
{
    let EvalInputs {
        metadata,
        parameters,
        vb,
        model_prefix,
        encoder,
        data_vec,
        delta_db,
        gene_remap,
        coarsening,
        cpu_dev,
        adj_method,
        minibatch_size,
        refine_config,
    } = inputs;

    // Register decoders at every level so safetensors keys match the
    // training layout. Refinement only uses the finest.
    let mut decoders: Vec<Dec> = Vec::with_capacity(metadata.level_decoder_dims.len());
    for (i, &d_l) in metadata.level_decoder_dims.iter().enumerate() {
        decoders.push(Dec::new(d_l, metadata.n_topics, vb.pp(format!("dec_{i}")))?);
    }

    let safetensors_path = format!("{model_prefix}.safetensors");
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    let decoder_ref = if refine_config.is_some() {
        decoders.last()
    } else {
        None
    };

    let eval_config = EvaluateLatentConfig {
        dev: cpu_dev,
        adj_method,
        minibatch_size,
        feature_coarsening: coarsening,
        decoder: decoder_ref,
        refine_config,
    };

    evaluate_latent_with_gene_remap(data_vec, encoder, delta_db, gene_remap, &eval_config)
}
