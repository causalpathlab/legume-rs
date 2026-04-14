use crate::embed_common::*;
use crate::topic::eval::*;
use crate::topic::model_metadata::*;

use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use candle_util::candle_decoder_topic::TopicDecoder;
use candle_util::candle_encoder_softmax::*;
use candle_util::candle_topic_refinement::*;
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
        default_value_t = 100,
        help = "Column block size for delta-estimation streaming"
    )]
    pub(crate) block_size: usize,

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

    #[arg(long, default_value_t = 0.01, help = "Learning rate for inference-time refinement")]
    pub(crate) refine_lr: f64,

    #[arg(long, default_value_t = 1.0, help = "L2 anchor strength for inference-time refinement")]
    pub(crate) refine_reg: f64,

    #[arg(short, long, help = "Verbose logging")]
    pub(crate) verbose: bool,
}

pub fn eval_topic_model(args: &EvalTopicArgs) -> anyhow::Result<()> {
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

    let delta_db = estimate_delta(&data_vec, &beta_dk, &gene_remap_opt, args.block_size)?;

    let cpu_dev = candle_core::Device::Cpu;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);

    let encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: metadata.n_features_encoder,
            n_topics: metadata.n_topics,
            layers: &metadata.encoder_hidden,
        },
        vb.clone(),
    )?;

    // Register decoder params so safetensors keys match
    let decoder = if args.refine_steps > 0 {
        let d = *metadata
            .level_decoder_dims
            .last()
            .unwrap_or(&metadata.n_features_encoder);
        Some(TopicDecoder::new(d, metadata.n_topics, vb.pp("dec_0"))?)
    } else {
        for (i, &d_l) in metadata.level_decoder_dims.iter().enumerate() {
            let _ = TopicDecoder::new(d_l, metadata.n_topics, vb.pp(format!("dec_{i}")))?;
        }
        None
    };

    let safetensors_path = format!("{}.safetensors", &args.model);
    info!("Loading weights from {}", safetensors_path);
    parameters.load(&safetensors_path)?;

    let refine_config = if args.refine_steps > 0 {
        Some(TopicRefinementConfig {
            num_steps: args.refine_steps,
            learning_rate: args.refine_lr,
            regularization: args.refine_reg,
        })
    } else {
        None
    };

    // Always use Batch adjustment — group membership from collapsing is unavailable
    let adj_method = AdjMethod::Batch;

    let eval_config = EvaluateLatentConfig {
        dev: &cpu_dev,
        adj_method: &adj_method,
        minibatch_size: args.minibatch_size,
        feature_coarsening: coarsening.as_ref(),
        decoder: decoder.as_ref(),
        refine_config: refine_config.as_ref(),
    };

    info!("Evaluating latent states on CPU");
    let z_nk = evaluate_latent_with_gene_remap(
        &data_vec,
        &encoder,
        delta_db.as_ref(),
        gene_remap_opt.as_ref(),
        &eval_config,
    )?;

    let cell_names = data_vec.column_names()?;
    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;
    info!("Done — wrote {}.latent.parquet", args.out);
    Ok(())
}

/// Estimate per-batch delta from new data pseudobulk and trained dictionary.
///
/// `delta[d,b] = (pb[d,b] / lib_b) / predicted[d]` where predicted is the
/// marginal gene proportion from the dictionary.
fn estimate_delta(
    data_vec: &SparseIoVec,
    beta_dk: &Mat,
    gene_remap: &Option<GeneRemap>,
    block_size: usize,
) -> anyhow::Result<Option<Mat>> {
    let n_batches = data_vec.num_batches();
    if n_batches <= 1 {
        info!("Single batch or no batches — skipping delta estimation");
        return Ok(None);
    }

    let d_train = beta_dk.nrows();
    let k = beta_dk.ncols();

    // Predicted marginal gene proportions from dictionary
    let exp_beta = beta_dk.map(|v| v.exp());
    let mut predicted = nalgebra::DVector::<f32>::zeros(d_train);
    for d in 0..d_train {
        predicted[d] = exp_beta.row(d).sum() / k as f32;
    }
    let pred_sum: f32 = predicted.iter().sum();
    if pred_sum > 0.0 {
        predicted /= pred_sum;
    }

    // Stream pseudobulk per batch in block order
    let d_new = data_vec.num_rows();
    let ntot = data_vec.num_columns();
    let mut pb_new = Mat::zeros(d_new, n_batches);

    for lb in (0..ntot).step_by(block_size) {
        let ub = (lb + block_size).min(ntot);
        let csc = data_vec.read_columns_csc(lb..ub)?;
        let batch_ids = data_vec.get_batch_membership(lb..ub);
        for (local_j, &batch_id) in batch_ids.iter().enumerate() {
            let col = csc.col(local_j);
            for (&row, &val) in col.row_indices().iter().zip(col.values().iter()) {
                pb_new[(row, batch_id)] += val;
            }
        }
    }

    // Remap to training gene order
    let mut pb_train = Mat::zeros(d_train, n_batches);
    if let Some(remap) = gene_remap {
        for (new_idx, opt_train) in remap.new_to_train.iter().enumerate() {
            if let Some(&train_idx) = opt_train.as_ref() {
                pb_train.row_mut(train_idx).copy_from(&pb_new.row(new_idx));
            }
        }
    } else {
        pb_train.copy_from(&pb_new);
    }

    // delta = observed_proportion / predicted_proportion
    let mut delta_db = Mat::zeros(d_train, n_batches);
    for b in 0..n_batches {
        let lib: f32 = pb_train.column(b).sum();
        if lib <= 0.0 {
            delta_db.column_mut(b).fill(1.0);
            continue;
        }
        for d in 0..d_train {
            let obs_prop = pb_train[(d, b)] / lib;
            let pred = predicted[d].max(1e-10);
            delta_db[(d, b)] = (obs_prop / pred).clamp(0.01, 100.0);
        }
    }

    info!("Estimated delta: {} genes × {} batches", d_train, n_batches);
    Ok(Some(delta_db))
}
