//! Unified prediction subcommand for the dense and indexed topic models.
//!
//! Loads a model trained by `senna topic` or `senna indexed-topic`, applies it
//! to a held-out backend file, and writes:
//!   - `{out}.latent.parquet`     [N × K] log θ (per-cell topic proportions)
//!   - `{out}.predictive.parquet` [N × 3] per-cell `[llik, total, llik_per_count]`
//!
//! Three latent-inference modes (mutually exclusive):
//!   - `Encoder` (default): forward pass through the trained encoder only.
//!   - `EncoderRefine` (`--refine-steps > 0`): encoder warm-start, then
//!     decoder gradient on θ anchored to encoder by L2.
//!   - `DecoderOnly` (`--decoder-only`): skip encoder; init θ uniform
//!     `log(1/K)`; optimize purely against the frozen decoder. Useful when
//!     the test feature set is too divergent for the encoder.

use crate::embed_common::*;
use crate::topic::eval::{build_gene_remap, GeneRemap};
use crate::topic::eval_indexed::{
    dense_to_indexed, dense_to_indexed_pair, refine_indexed_topic_proportions,
};
use crate::topic::model_metadata::{
    load_coarsening, load_dictionary, load_shortlist_weights, TopicModelMetadata,
};
use crate::topic::predict_common::{
    decoder_only_inference_dense, decoder_only_inference_indexed, estimate_delta,
    predictive_llik_dense, predictive_llik_indexed, LatentMode,
};

use crate::logging::new_progress_bar;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use candle_core::{Device, Tensor};
use candle_util::candle_decoder_indexed_topic::IndexedTopicDecoder;
use candle_util::candle_decoder_nb_mixture::{
    NbMixtureTopicDecoder, DECODER_NAME as NBMIXTURE_NAME,
};
use candle_util::candle_decoder_topic::{MultinomTopicDecoder, NbTopicDecoder};
use candle_util::candle_encoder_indexed::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
use candle_util::candle_encoder_softmax::{LogSoftmaxEncoder, LogSoftmaxEncoderArgs};
use candle_util::candle_indexed_model_traits::IndexedEncoderT;
use candle_util::candle_model_traits::{DecoderModuleT, EncoderModuleT, NewDecoder};
use candle_util::candle_topic_refinement::{refine_topic_proportions, TopicRefinementConfig};
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use indicatif::ParallelProgressIterator;
use log::info;
use rayon::prelude::*;

#[derive(Args, Debug)]
pub struct PredictArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Held-out data files (.zarr or .h5)",
        long_help = "Sparse backends to score with the pre-trained model.\n\
                     Gene sets may differ from training; missing genes are padded\n\
                     and per-batch delta is re-estimated from the frozen dictionary."
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "Trained model prefix (output of `senna topic -o` or `senna indexed-topic -o`)",
        long_help = "Loads:\n  \
                     {model}.dictionary.parquet      gene × topic dictionary\n  \
                     {model}.model.json              model architecture metadata\n  \
                     {model}.safetensors             encoder + decoder weights\n  \
                     {model}.coarsening.json         (dense only) feature coarsening\n  \
                     {model}.shortlist_weights.parquet (indexed only) NB-Fisher weights"
    )]
    pub(crate) model: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file prefix",
        long_help = "Writes:\n  \
                     {out}.latent.parquet      [N × K] log θ\n  \
                     {out}.predictive.parquet  per-cell [llik, total, llik_per_count]"
    )]
    pub(crate) out: Box<str>,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = 500, help = "Evaluation minibatch size")]
    pub(crate) minibatch_size: usize,

    #[arg(long, help = "Cells per delta-estimation block (auto by default)")]
    pub(crate) block_size: Option<usize>,

    #[arg(long, help = "Load all columns into memory before evaluation")]
    pub(crate) preload_data: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Decoder-side gradient steps on θ at inference (0 = encoder forward only)",
        long_help = "If --decoder-only is set, this controls iterations of\n\
                     uniform-init optimization. Otherwise, controls per-cell\n\
                     refinement steps anchored to the encoder output."
    )]
    pub(crate) refine_steps: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Learning rate for refinement / decoder-only"
    )]
    pub(crate) refine_lr: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 anchor strength for refinement"
    )]
    pub(crate) refine_reg: f64,

    #[arg(
        long,
        help = "Skip the encoder; init θ uniform and optimize purely against the frozen decoder",
        long_help = "Useful when the held-out feature set is too divergent for the\n\
                     trained encoder. Uses --refine-steps and --refine-lr (defaults\n\
                     bumped to 100 / 0.05 if --refine-steps was left at 0)."
    )]
    pub(crate) decoder_only: bool,

    #[arg(
        long,
        default_value_t = 3,
        help = "Iterative TMLE rounds for held-out batch δ (0 = legacy single-pass plug-in)",
        long_help = "Per iteration: encode all cells with current δ → θ̂; refit\n\
                     δ as Σ_obs / Σ_pred per batch (NB-Fisher-weighted for nb /\n\
                     nbmixture decoders, using the saved {model}.dispersion.parquet\n\
                     when present). Default 3 typically converges; 0 reverts to the\n\
                     legacy 1/K-marginal plug-in."
    )]
    pub(crate) delta_iters: usize,

    #[arg(short, long, help = "Verbose logging")]
    pub(crate) verbose: bool,
}

pub fn predict_model(args: &PredictArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let metadata = TopicModelMetadata::load(&args.model)?;
    info!(
        "Loaded model metadata: type={}, K={}, D_full={}, D_enc={}",
        metadata.model_type,
        metadata.n_topics,
        metadata.n_features_full,
        metadata.n_features_encoder,
    );

    // Residual-mode x0 needs per-pseudobulk-group mu_residual, which only
    // exists after a multilevel collapse pass over the held-out data — and
    // the predict path skips that pass. We feed Batch-style x0 instead, so
    // the encoder sees a different-distribution null than it saw at training.
    // Warn loudly: θ̂ may be biased on held-out for residual-trained models.
    if metadata.adj_method.as_ref() == "residual" {
        log::warn!(
            "model was trained with --adj-method residual; predict only supports \
             batch-style x0. θ̂ may be biased — retrain with --adj-method batch \
             for clean held-out semantics."
        );
    }

    use crate::topic::model_metadata::{MODEL_TYPE_INDEXED, MODEL_TYPE_TOPIC};
    match metadata.model_type.as_ref() {
        MODEL_TYPE_TOPIC => predict_dense(args, &metadata),
        MODEL_TYPE_INDEXED => predict_indexed(args, &metadata),
        other => anyhow::bail!(
            "predict: unsupported model_type '{other}' (expected '{MODEL_TYPE_TOPIC}' or '{MODEL_TYPE_INDEXED}')",
        ),
    }
}

fn resolve_mode(args: &PredictArgs) -> LatentMode {
    if args.decoder_only {
        LatentMode::DecoderOnly
    } else if args.refine_steps > 0 {
        LatentMode::EncoderRefine
    } else {
        LatentMode::Encoder
    }
}

fn build_remap(
    training_genes: &[Box<str>],
    new_genes: &[Box<str>],
) -> anyhow::Result<Option<GeneRemap>> {
    let gene_remap = build_gene_remap(training_genes, new_genes);
    let min_overlap = (training_genes.len() as f32 * 0.1) as usize;
    anyhow::ensure!(
        gene_remap.n_mapped >= min_overlap,
        "Too few genes overlap: {}/{} mapped (need at least {})",
        gene_remap.n_mapped,
        training_genes.len(),
        min_overlap,
    );

    let needs_remap = gene_remap
        .new_to_train
        .iter()
        .enumerate()
        .any(|(i, opt)| *opt != Some(i))
        || new_genes.len() != training_genes.len();

    if needs_remap {
        info!(
            "Gene remapping enabled ({} → {} features)",
            new_genes.len(),
            training_genes.len()
        );
        Ok(Some(gene_remap))
    } else {
        info!("Genes match training — no remapping needed");
        Ok(None)
    }
}

/// Aggregate a `[D_full]` per-gene mean to `[D_coarse]` so the divisor
/// is on the same per-fine-gene rate scale as the encoder expects.
///
/// The encoder's input `y_coarse` and batch null `x0_coarse` are both
/// **sum**-coarsened (`aggregate_columns_nd`), giving `y ≈ G·batch·<μ>·bio`
/// and `x0 ≈ G·batch`. For `clean = y / (x0·μ_coarse)` to recover `bio`,
/// `μ_coarse` must be at the per-fine-gene rate `<μ>`, not the summed
/// `Σμ`. So we sum-coarsen first (matching the call sites used for
/// data and batch null) then divide each coarse cell by its group size.
pub(crate) fn aggregate_feature_mean_to_coarse(
    full: &[f32],
    coarsening: Option<&FeatureCoarsening>,
) -> Vec<f32> {
    let n_full = full.len();
    let mu_1d = nalgebra::DMatrix::<f32>::from_row_slice(1, n_full, full);
    match coarsening {
        Some(fc) => {
            let mu_summed = fc.aggregate_columns_nd(&mu_1d);
            mu_summed
                .row(0)
                .iter()
                .zip(fc.coarse_to_fine.iter())
                .map(|(&s, fines)| s / fines.len().max(1) as f32)
                .collect()
        }
        None => mu_1d.row(0).iter().copied().collect(),
    }
}

////////////////////////
// Dense prediction //
////////////////////////

fn predict_dense(args: &PredictArgs, metadata: &TopicModelMetadata) -> anyhow::Result<()> {
    let (training_genes, beta_dk) = load_dictionary(&args.model)?;
    let coarsening = if metadata.has_coarsening {
        load_coarsening(&args.model)?
    } else {
        None
    };

    // Reload `μ_d` at D_full and aggregate via the (optional) coarsening
    // matrix to the encoder's D_coarse. Saved by `senna topic` at
    // training time; absent for older models, where the encoder falls
    // back to live per-feature batch centering inside `anscombe_residual`.
    let feature_mean_enc: Option<Vec<f32>> =
        match crate::topic::model_metadata::load_feature_mean(&args.model) {
            Ok((_, full)) => Some(aggregate_feature_mean_to_coarse(&full, coarsening.as_ref())),
            Err(_) => None,
        };

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;
    let mut data_vec = loaded.data;
    data_vec.register_batch_membership(&loaded.batch);
    info!(
        "Held-out data: {} features × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );

    let new_genes = data_vec.row_names()?;
    let gene_remap = build_remap(&training_genes, &new_genes)?;

    let delta_db = estimate_delta(
        &data_vec,
        &beta_dk,
        metadata.theta_mean.as_deref(),
        gene_remap.as_ref(),
        args.block_size,
    )?;

    let cpu_dev = Device::Cpu;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);

    let encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features: metadata.n_features_encoder,
            n_topics: metadata.n_topics,
            layers: &metadata.encoder_hidden,
            feature_mean: feature_mean_enc.as_deref(),
        },
        &parameters,
        vb.clone(),
    )?;

    let mode = resolve_mode(args);
    let refine_config = TopicRefinementConfig {
        num_steps: if args.decoder_only && args.refine_steps == 0 {
            100
        } else {
            args.refine_steps
        },
        learning_rate: if args.decoder_only && args.refine_lr <= 0.01 {
            0.05
        } else {
            args.refine_lr
        },
        regularization: args.refine_reg,
    };
    info!("Latent inference mode: {mode:?}");

    let decoder_name = metadata
        .decoder_types
        .first()
        .map_or("multinom", std::convert::AsRef::as_ref);
    info!("Predicting with decoder: {decoder_name}");

    let inputs = DensePredictInputs {
        metadata,
        parameters: &mut parameters,
        vb: &vb,
        model_prefix: &args.model,
        encoder: &encoder,
        data_vec: &data_vec,
        delta_db,
        gene_remap: gene_remap.as_ref(),
        coarsening: coarsening.as_ref(),
        beta_dk: &beta_dk,
        delta_iters: args.delta_iters,
        cpu_dev: &cpu_dev,
        adj_method: AdjMethod::Batch,
        minibatch_size: args.minibatch_size,
        mode,
        refine_config: &refine_config,
    };

    let (z_nk, llik, total) = match decoder_name {
        "multinom" => predict_dense_with_decoder::<MultinomTopicDecoder>(inputs)?,
        "nb" => predict_dense_with_decoder::<NbTopicDecoder>(inputs)?,
        name if name == NBMIXTURE_NAME => {
            predict_dense_with_decoder::<NbMixtureTopicDecoder>(inputs)?
        }
        other => anyhow::bail!("unsupported decoder type in metadata: {other}"),
    };

    write_outputs(args, &data_vec, &z_nk, &llik, &total)
}

struct DensePredictInputs<'a> {
    metadata: &'a TopicModelMetadata,
    parameters: &'a mut candle_nn::VarMap,
    vb: &'a candle_nn::VarBuilder<'a>,
    model_prefix: &'a str,
    encoder: &'a LogSoftmaxEncoder,
    data_vec: &'a SparseIoVec,
    delta_db: Option<Mat>,
    gene_remap: Option<&'a GeneRemap>,
    coarsening: Option<&'a FeatureCoarsening>,
    beta_dk: &'a Mat,
    delta_iters: usize,
    cpu_dev: &'a Device,
    adj_method: AdjMethod,
    minibatch_size: usize,
    mode: LatentMode,
    refine_config: &'a TopicRefinementConfig,
}

fn predict_dense_with_decoder<Dec>(
    inputs: DensePredictInputs<'_>,
) -> anyhow::Result<(Mat, Vec<f32>, Vec<f32>)>
where
    Dec: DecoderModuleT + NewDecoder + Send + Sync,
{
    let DensePredictInputs {
        metadata,
        parameters,
        vb,
        model_prefix,
        encoder,
        data_vec,
        delta_db,
        gene_remap,
        coarsening,
        beta_dk,
        delta_iters,
        cpu_dev,
        adj_method,
        minibatch_size,
        mode,
        refine_config,
    } = inputs;

    // Register decoders at every level so safetensors keys match training.
    // Predict only uses the finest-level decoder.
    let mut decoders: Vec<Dec> = Vec::with_capacity(metadata.level_decoder_dims.len());
    for (i, &d_l) in metadata.level_decoder_dims.iter().enumerate() {
        decoders.push(Dec::new(d_l, metadata.n_topics, vb.pp(format!("dec_{i}")))?);
    }

    let safetensors_path = format!("{model_prefix}.safetensors");
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;

    // Attach finest-level NB-Fisher weights to the finest decoder so
    // predictive llik uses the same loss as training. Older models
    // without saved coarse weights fall back to the unweighted form.
    let coarse_path = format!("{model_prefix}.fisher_weights_coarse.parquet");
    if std::path::Path::new(&coarse_path).exists() {
        let (_, coarse_w) = data_beans_alg::gene_weighting::load_per_gene_weights(&coarse_path)?;
        if let Some(finest) = decoders.last_mut() {
            finest.attach_feature_weights(&coarse_w, cpu_dev)?;
        }
    }
    let decoder = decoders.last().expect("at least one decoder level");

    // Iterative TMLE δ refinement: replaces the single-pass plug-in with a
    // per-cell θ̂-aware obs/pred update (NB-Fisher-weighted when φ is saved).
    // `delta_iters == 0` falls through with the plug-in δ unchanged.
    let delta_db = if delta_iters > 0 {
        if let Some(initial) = delta_db {
            let phi_opt = crate::topic::model_metadata::load_dispersion(model_prefix)?;
            let phi_for_iter: Option<&[f32]> = match metadata
                .decoder_types
                .first()
                .map(std::convert::AsRef::as_ref)
            {
                Some("nb") | Some("nbmixture") => phi_opt.as_deref(),
                _ => None,
            };
            let refined = crate::predict_tmle::iterate_delta_dense(
                delta_iters,
                initial,
                data_vec,
                encoder,
                gene_remap,
                coarsening,
                beta_dk,
                phi_for_iter,
                minibatch_size,
                cpu_dev,
                &adj_method,
            )?;
            Some(refined)
        } else {
            None
        }
    } else {
        delta_db
    };

    // Delta tensor at encoder D (coarsened if applicable). Note: dense
    // refinement uses the encoder's input dim, which equals D_finest here.
    let delta_tensor = delta_db
        .as_ref()
        .map(|db| -> anyhow::Result<Tensor> {
            let mut db = db.clone();
            if let Some(fc) = coarsening {
                db = fc.aggregate_rows_ds(&db);
            }
            let t = db.to_tensor(cpu_dev)?.transpose(0, 1)?.contiguous()?;
            Ok(t)
        })
        .transpose()?;

    let ntot = data_vec.num_columns();
    let kk = metadata.n_topics;

    let jobs = create_jobs(ntot, 0, Some(minibatch_size));
    let njobs = jobs.len() as u64;

    let chunks: Vec<(usize, Mat, Vec<f32>, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .map(
            |&(lb, ub)| -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)> {
                predict_block_dense::<Dec>(PredictBlockDenseArgs {
                    lb,
                    ub,
                    data_vec,
                    encoder,
                    decoder,
                    delta_tensor: delta_tensor.as_ref(),
                    gene_remap,
                    coarsening,
                    dev: cpu_dev,
                    adj_method: &adj_method,
                    mode,
                    refine_config,
                })
            },
        )
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut chunks = chunks;
    chunks.sort_by_key(|c| c.0);

    let mut z_nk = Mat::zeros(ntot, kk);
    let mut llik = Vec::with_capacity(ntot);
    let mut total = Vec::with_capacity(ntot);
    let mut row = 0;
    for (_, z_block, lk, tot) in chunks {
        let n = z_block.nrows();
        z_nk.rows_range_mut(row..row + n).copy_from(&z_block);
        llik.extend(lk);
        total.extend(tot);
        row += n;
    }
    Ok((z_nk, llik, total))
}

struct PredictBlockDenseArgs<'a, Dec> {
    lb: usize,
    ub: usize,
    data_vec: &'a SparseIoVec,
    encoder: &'a LogSoftmaxEncoder,
    decoder: &'a Dec,
    delta_tensor: Option<&'a Tensor>,
    gene_remap: Option<&'a GeneRemap>,
    coarsening: Option<&'a FeatureCoarsening>,
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    mode: LatentMode,
    refine_config: &'a TopicRefinementConfig,
}

fn predict_block_dense<Dec>(
    a: PredictBlockDenseArgs<'_, Dec>,
) -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)>
where
    Dec: DecoderModuleT,
{
    use crate::topic::common::expand_delta_for_block;

    let PredictBlockDenseArgs {
        lb,
        ub,
        data_vec,
        encoder,
        decoder,
        delta_tensor,
        gene_remap,
        coarsening,
        dev,
        adj_method,
        mode,
        refine_config,
    } = a;

    let x0_nd = delta_tensor
        .map(|delta_bm| expand_delta_for_block(data_vec, delta_bm, adj_method, lb, ub, dev))
        .transpose()?;

    let csc = data_vec.read_columns_csc(lb..ub)?;
    let x_at_dec = remap_and_coarsen_dense(&csc, gene_remap, coarsening, dev)?;

    let log_z_nk = match mode {
        LatentMode::Encoder => {
            let (log_z, _) = encoder.forward_t(&x_at_dec, x0_nd.as_ref(), false)?;
            log_z
        }
        LatentMode::EncoderRefine => {
            let (log_z, _) = encoder.forward_t(&x_at_dec, x0_nd.as_ref(), false)?;
            refine_topic_proportions(&log_z, &x_at_dec, decoder, refine_config)?
        }
        LatentMode::DecoderOnly => decoder_only_inference_dense(
            decoder,
            &x_at_dec,
            decoder.dim_latent(),
            refine_config.learning_rate,
            refine_config.num_steps,
            dev,
        )?,
    };

    let llik_t = predictive_llik_dense(decoder, &log_z_nk, &x_at_dec)?;
    let llik: Vec<f32> = llik_t.to_device(&Device::Cpu)?.to_vec1()?;

    let total: Vec<f32> = {
        let summed = x_at_dec.sum(1)?.to_device(&Device::Cpu)?;
        summed.to_vec1()?
    };

    let z_cpu = log_z_nk.to_device(&Device::Cpu)?;
    let z_mat = Mat::from_tensor(&z_cpu)?;
    Ok((lb, z_mat, llik, total))
}

/// Scatter CSC rows from new-data order to training gene order, then optionally
/// coarsen, returning a `[N, D_dec]` tensor on the requested device.
fn remap_and_coarsen_dense(
    csc: &nalgebra_sparse::CscMatrix<f32>,
    gene_remap: Option<&GeneRemap>,
    coarsening: Option<&FeatureCoarsening>,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let nd = if let Some(remap) = gene_remap {
        let ncols = csc.ncols();
        let mut out = Mat::zeros(remap.d_train, ncols);
        for j in 0..ncols {
            let col = csc.col(j);
            for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
                if let Some(row_train) = remap.new_to_train[row_new] {
                    out[(row_train, j)] = val;
                }
            }
        }
        if let Some(fc) = coarsening {
            fc.aggregate_rows_ds(&out)
                .to_tensor(dev)?
                .transpose(0, 1)?
                .contiguous()?
        } else {
            out.to_tensor(dev)?.transpose(0, 1)?.contiguous()?
        }
    } else if let Some(fc) = coarsening {
        fc.aggregate_sparse_csc(csc)
            .to_tensor(dev)?
            .transpose(0, 1)?
            .contiguous()?
    } else {
        csc.to_tensor(dev)?.transpose(0, 1)?.contiguous()?
    };
    Ok(nd)
}

//////////////////////////
// Indexed prediction //
//////////////////////////

fn predict_indexed(args: &PredictArgs, metadata: &TopicModelMetadata) -> anyhow::Result<()> {
    let embedding_dim = metadata
        .embedding_dim
        .ok_or_else(|| anyhow::anyhow!("indexed model metadata missing embedding_dim"))?;
    let enc_context_size = metadata
        .enc_context_size
        .ok_or_else(|| anyhow::anyhow!("indexed model metadata missing enc_context_size"))?;
    let dec_context_size = metadata
        .dec_context_size
        .ok_or_else(|| anyhow::anyhow!("indexed model metadata missing dec_context_size"))?;

    let (training_genes, beta_dk) = load_dictionary(&args.model)?;
    let (sw_genes, shortlist_weights) = load_shortlist_weights(&args.model)?;
    anyhow::ensure!(
        sw_genes.len() == training_genes.len(),
        "shortlist_weights gene count ({}) != dictionary gene count ({})",
        sw_genes.len(),
        training_genes.len(),
    );
    let (fb_genes, feature_mean) = crate::topic::model_metadata::load_feature_mean(&args.model)?;
    anyhow::ensure!(
        fb_genes.len() == training_genes.len(),
        "feature_mean gene count ({}) != dictionary gene count ({})",
        fb_genes.len(),
        training_genes.len(),
    );
    // The decoder's NB-Fisher loss weights are the same NB-Fisher per-gene
    // weights used for selection — the data on disk is one vector.
    let feature_fisher_weights = shortlist_weights.clone();

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
    })?;
    let mut data_vec = loaded.data;
    data_vec.register_batch_membership(&loaded.batch);
    info!(
        "Held-out data: {} features × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );

    let new_genes = data_vec.row_names()?;
    let gene_remap = build_remap(&training_genes, &new_genes)?;

    let delta_db = estimate_delta(
        &data_vec,
        &beta_dk,
        metadata.theta_mean.as_deref(),
        gene_remap.as_ref(),
        args.block_size,
    )?;

    let cpu_dev = Device::Cpu;
    let mut parameters = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);

    let n_features_full = metadata.n_features_full;
    let encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics: metadata.n_topics,
            embedding_dim,
            layers: &metadata.encoder_hidden,
        },
        &parameters,
        vb.pp("enc"),
    )?;

    // Register decoders at every level so safetensors keys match training.
    // Predict only uses the finest-level decoder.
    let num_levels = metadata.level_decoder_dims.len().max(1);
    let mut decoders: Vec<IndexedTopicDecoder> = Vec::with_capacity(num_levels);
    for i in 0..num_levels {
        decoders.push(IndexedTopicDecoder::new(
            n_features_full,
            metadata.n_topics,
            vb.pp(format!("dec_{i}")),
        )?);
    }

    let safetensors_path = format!("{}.safetensors", args.model);
    info!("Loading weights from {safetensors_path}");
    parameters.load(&safetensors_path)?;
    let decoder = decoders.last().expect("at least one decoder level");

    let mode = resolve_mode(args);
    let refine_config = TopicRefinementConfig {
        num_steps: if args.decoder_only && args.refine_steps == 0 {
            100
        } else {
            args.refine_steps
        },
        learning_rate: if args.decoder_only && args.refine_lr <= 0.01 {
            0.05
        } else {
            args.refine_lr
        },
        regularization: args.refine_reg,
    };
    info!("Latent inference mode: {mode:?}");

    let adj_method = AdjMethod::Batch;

    // Iterative TMLE δ refinement (indexed). Encoder + dictionary are now
    // loaded; iterate δ against them. Indexed decoder is currently always
    // multinomial in metadata, so no φ weighting needed.
    let delta_db = if args.delta_iters > 0 {
        if let Some(initial) = delta_db {
            let phi_for_iter: Option<&[f32]> = None;
            let refined = crate::predict_tmle::iterate_delta_indexed(
                args.delta_iters,
                initial,
                &data_vec,
                &encoder,
                gene_remap.as_ref(),
                &beta_dk,
                phi_for_iter,
                &shortlist_weights,
                &feature_mean,
                enc_context_size,
                args.minibatch_size,
                &cpu_dev,
                &adj_method,
            )?;
            Some(refined)
        } else {
            None
        }
    } else {
        delta_db
    };

    let delta_tensor = delta_db
        .as_ref()
        .map(|db| -> anyhow::Result<Tensor> {
            let t = db.to_tensor(&cpu_dev)?.transpose(0, 1)?.contiguous()?;
            Ok(t)
        })
        .transpose()?;

    let ntot = data_vec.num_columns();
    let kk = metadata.n_topics;

    let jobs = create_jobs(ntot, 0, Some(args.minibatch_size));
    let njobs = jobs.len() as u64;

    let chunks: Vec<(usize, Mat, Vec<f32>, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(new_progress_bar(njobs))
        .map(
            |&(lb, ub)| -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)> {
                predict_block_indexed(PredictBlockIndexedArgs {
                    lb,
                    ub,
                    data_vec: &data_vec,
                    encoder: &encoder,
                    decoder,
                    delta_tensor: delta_tensor.as_ref(),
                    gene_remap: gene_remap.as_ref(),
                    shortlist_weights: &shortlist_weights,
                    feature_mean: &feature_mean,
                    feature_fisher_weights: &feature_fisher_weights,
                    enc_context_size,
                    dec_context_size,
                    dev: &cpu_dev,
                    adj_method: &adj_method,
                    mode,
                    refine_config: &refine_config,
                    n_topics: metadata.n_topics,
                })
            },
        )
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut chunks = chunks;
    chunks.sort_by_key(|c| c.0);

    let mut z_nk = Mat::zeros(ntot, kk);
    let mut llik = Vec::with_capacity(ntot);
    let mut total = Vec::with_capacity(ntot);
    let mut row = 0;
    for (_, z_block, lk, tot) in chunks {
        let n = z_block.nrows();
        z_nk.rows_range_mut(row..row + n).copy_from(&z_block);
        llik.extend(lk);
        total.extend(tot);
        row += n;
    }

    write_outputs(args, &data_vec, &z_nk, &llik, &total)
}

struct PredictBlockIndexedArgs<'a> {
    lb: usize,
    ub: usize,
    data_vec: &'a SparseIoVec,
    encoder: &'a IndexedEmbeddingEncoder,
    decoder: &'a IndexedTopicDecoder,
    delta_tensor: Option<&'a Tensor>,
    gene_remap: Option<&'a GeneRemap>,
    shortlist_weights: &'a [f32],
    feature_mean: &'a [f32],
    feature_fisher_weights: &'a [f32],
    enc_context_size: usize,
    dec_context_size: usize,
    dev: &'a Device,
    adj_method: &'a AdjMethod,
    mode: LatentMode,
    refine_config: &'a TopicRefinementConfig,
    n_topics: usize,
}

fn predict_block_indexed(
    a: PredictBlockIndexedArgs<'_>,
) -> anyhow::Result<(usize, Mat, Vec<f32>, Vec<f32>)> {
    use crate::topic::common::expand_delta_for_block;

    let PredictBlockIndexedArgs {
        lb,
        ub,
        data_vec,
        encoder,
        decoder,
        delta_tensor,
        gene_remap,
        shortlist_weights,
        feature_mean,
        feature_fisher_weights,
        enc_context_size,
        dec_context_size,
        dev,
        adj_method,
        mode,
        refine_config,
        n_topics,
    } = a;
    let ctx = crate::topic::eval_indexed::PerGeneContext {
        feature_mean: Some(feature_mean),
        feature_fisher_weights: Some(feature_fisher_weights),
    };

    // Read held-out CSC and scatter to training gene order at full D.
    let csc = data_vec.read_columns_csc(lb..ub)?;
    let x_dn_train = if let Some(remap) = gene_remap {
        let ncols = csc.ncols();
        let mut out = Mat::zeros(remap.d_train, ncols);
        for j in 0..ncols {
            let col = csc.col(j);
            for (&row_new, &val) in col.row_indices().iter().zip(col.values().iter()) {
                if let Some(row_train) = remap.new_to_train[row_new] {
                    out[(row_train, j)] = val;
                }
            }
        }
        out
    } else {
        // No remap: csc is already at training D.
        let dim = csc.nrows();
        let ncols = csc.ncols();
        let mut out = Mat::zeros(dim, ncols);
        for j in 0..ncols {
            let col = csc.col(j);
            for (&row, &val) in col.row_indices().iter().zip(col.values().iter()) {
                out[(row, j)] = val;
            }
        }
        out
    };
    let x_nd = x_dn_train.to_tensor(dev)?.transpose(0, 1)?.contiguous()?;

    // Per-cell batch correction at full D
    let x0_nd = delta_tensor
        .map(|delta_bm| expand_delta_for_block(data_vec, delta_bm, adj_method, lb, ub, dev))
        .transpose()?;

    // Packed top-K representation: encoder window and decoder window.
    // When the two contexts match, the decoder pack is identical to the
    // encoder pack — clone (cheap; Tensor is Arc-buffered) instead of
    // re-walking the dense rows.
    let (enc_pack, dec_pack) = if dec_context_size != enc_context_size {
        dense_to_indexed_pair(
            &x_nd,
            enc_context_size,
            dec_context_size,
            shortlist_weights,
            ctx,
            dev,
        )?
    } else {
        let enc = dense_to_indexed(&x_nd, enc_context_size, shortlist_weights, ctx, dev)?;
        let dec = enc.clone();
        (enc, dec)
    };

    // Encoder-side null gathered at the encoder's per-cell ids — O(N·K).
    let enc_values_null = match x0_nd.as_ref() {
        Some(x0) => Some(crate::topic::eval_indexed::gather_null_at_indices(
            x0,
            &enc_pack.indices,
            dev,
        )?),
        None => None,
    };

    // Decoder-side log_q_s = uniform (zeros) at inference time.
    let s_dec = dec_pack.union_indices.dim(0)?;
    let dec_log_q_s = Tensor::zeros((1, s_dec), candle_core::DType::F32, dev)?;

    let log_z_nk = match mode {
        LatentMode::Encoder => {
            let (log_z, _) = encoder.forward_indexed_t(
                &enc_pack.indices,
                &enc_pack.values,
                enc_values_null.as_ref(),
                enc_pack.values_mean.as_ref(),
                false,
            )?;
            log_z
        }
        LatentMode::EncoderRefine => {
            let (log_z, _) = encoder.forward_indexed_t(
                &enc_pack.indices,
                &enc_pack.values,
                enc_values_null.as_ref(),
                enc_pack.values_mean.as_ref(),
                false,
            )?;
            refine_indexed_topic_proportions(
                &log_z,
                &dec_pack.union_indices,
                &dec_pack.scatter_pos,
                &dec_pack.values,
                dec_pack.values_weight.as_ref(),
                &dec_log_q_s,
                decoder,
                refine_config,
            )?
        }
        LatentMode::DecoderOnly => decoder_only_inference_indexed(
            decoder,
            &crate::topic::predict_common::IndexedDecoderInput {
                union_indices: &dec_pack.union_indices,
                scatter_pos: &dec_pack.scatter_pos,
                values: &dec_pack.values,
                values_weight: dec_pack.values_weight.as_ref(),
                log_q_s: &dec_log_q_s,
            },
            n_topics,
            refine_config.learning_rate,
            refine_config.num_steps,
            dev,
        )?,
    };

    // Predictive lik at the decoder shortlist (K_dec features per cell).
    let llik_t = predictive_llik_indexed(
        decoder,
        &log_z_nk,
        &dec_pack.union_indices,
        &dec_pack.scatter_pos,
        &dec_pack.values,
        dec_pack.values_weight.as_ref(),
        &dec_log_q_s,
    )?;
    let llik: Vec<f32> = llik_t.to_device(&Device::Cpu)?.to_vec1()?;

    // Total counts at the decoder shortlist (denominator for llik_per_count).
    let total: Vec<f32> = {
        let summed = dec_pack.values.sum(1)?.to_device(&Device::Cpu)?;
        summed.to_vec1()?
    };

    let z_cpu = log_z_nk.to_device(&Device::Cpu)?;
    let z_mat = Mat::from_tensor(&z_cpu)?;
    Ok((lb, z_mat, llik, total))
}

/////////////////////
// Output writers //
/////////////////////

fn write_outputs(
    args: &PredictArgs,
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    llik: &[f32],
    total: &[f32],
) -> anyhow::Result<()> {
    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&axis_id_names("T", z_nk.ncols())),
    )?;
    info!("Wrote {}.latent.parquet", args.out);

    // Per-cell predictive scores: [llik, total, llik_per_count]
    let n = llik.len();
    let mut pred = Mat::zeros(n, 3);
    for i in 0..n {
        pred[(i, 0)] = llik[i];
        pred[(i, 1)] = total[i];
        pred[(i, 2)] = if total[i] > 0.0 {
            llik[i] / total[i]
        } else {
            0.0
        };
    }
    let pred_cols: Vec<Box<str>> = vec!["llik".into(), "total".into(), "llik_per_count".into()];
    pred.to_parquet_with_names(
        &(args.out.to_string() + ".predictive.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&pred_cols),
    )?;
    info!("Wrote {}.predictive.parquet", args.out);
    Ok(())
}
