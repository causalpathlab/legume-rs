//! `senna vae` — scVI-style **Gaussian VAE** (continuous factor model).
//!
//! Sibling of `senna topic`. Same data pipeline (collapse → multilevel
//! pseudobulks → dense VAE), but the latent is an **unconstrained Gaussian**
//! `z` from a [`GaussianEncoder`] (no simplex projection), paired with a
//! [`GaussianNbDecoder`] (`π = softmax_d(z·W) → μ = library·π`, NB). Outputs are
//! continuous **factors** (cell × factor) and **loadings** (gene × factor), not
//! topic proportions + a topic-gene dictionary.
//!
//! The dense [`candle_util::vae::topic::train_mixed`] loop is latent-agnostic
//! when `topic_smoothing = 0` (the simplex smoothing becomes a no-op and the raw
//! `z` flows straight to the decoder's own NB likelihood), so this path reuses it
//! verbatim. The topic-specific machinery (anchor prior, NB-Fisher weighting,
//! ambient mixture, empirical dictionary, feature coarsening) does not apply to a
//! continuous-factor model and is intentionally omitted.

use crate::embed_common::*;
use crate::topic::common::{
    create_device, load_and_collapse, move_varmap_to_cpu, sample_collapsed_data,
    setup_stop_handler, LoadCollapseArgs, PreparedData,
};
use crate::topic::eval::{evaluate_latent_by_encoder, EvaluateLatentConfig};

use candle_util::decoder::GaussianNbDecoder;
use candle_util::encoder::{GaussianEncoder, GaussianEncoderArgs};

#[derive(Args, Debug)]
pub struct VaeArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5; optional when --from is given)"
    )]
    pub(crate) data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain data + batch + cell→pb partition from a prior \
                `senna {topic, vae, masked-topic}` run's manifest"
    )]
    pub(crate) from: Option<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet       gene × factor loadings\n  \
                     {out}.latent.parquet           cell × factor scores (Gaussian z)\n  \
                     {out}.log_likelihood.parquet   training loss trace\n  \
                     {out}.safetensors              encoder+decoder weights\n  \
                     {out}.model.json               model metadata (for `senna predict`)\n  \
                     {out}.feature_mean.parquet     per-gene mean rate μ_d\n  \
                     {out}.cell_proj.parquet        cached random projection\n  \
                     {out}.senna.json               run manifest"
    )]
    pub(crate) out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file"
    )]
    pub(crate) batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    pub(crate) collapse: crate::refine_weighting::CollapseArgs,

    #[arg(
        long = "init-from",
        help = "Initialize weights from a previously trained `senna vae` model",
        long_help = "Initialize encoder + decoder weights from a previously trained\n\
                     `senna vae` model."
    )]
    pub(crate) init_from: Option<Box<str>>,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    pub(crate) block_size: Option<usize>,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent factors (K)"
    )]
    pub(crate) n_latent: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder hidden layer sizes (comma-separated)"
    )]
    pub(crate) encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    pub(crate) epochs: usize,

    #[arg(long, default_value_t = 100, help = "Training minibatch size")]
    pub(crate) minibatch_size: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        alias = "lr",
        help = "Adam learning rate"
    )]
    pub(crate) learning_rate: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off)"
    )]
    pub(crate) grad_clip: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device (cpu|cuda|metal)"
    )]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    pub(crate) device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Batch adjustment (batch|residual)"
    )]
    pub(crate) adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training"
    )]
    pub(crate) preload_data: bool,

    #[command(flatten)]
    pub(crate) hvg: crate::hvg::HvgCliArgs,

    #[command(flatten)]
    pub(crate) qc: QcArgs,
}

pub fn fit_vae_model(args: &VaeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let inherited = args
        .from
        .as_deref()
        .map(crate::run_manifest::inherit_from)
        .transpose()?;
    if let Some(inh) = inherited.as_ref() {
        info!(
            "--from: inheriting data + batch{} from a '{}' manifest",
            if inh.cell_to_pb_path.is_some() {
                " + cell→pb partition"
            } else {
                ""
            },
            inh.source_kind
        );
    }
    let data_files = crate::run_manifest::InheritedFromManifest::resolve_data(
        inherited.as_ref(),
        &args.data_files,
    )?;
    let batch_files = crate::run_manifest::InheritedFromManifest::resolve_batch(
        inherited.as_ref(),
        args.batch_files.as_deref(),
    );
    let prebuilt_partition = inherited
        .as_ref()
        .map(super::run_manifest::InheritedFromManifest::load_cell_to_pb)
        .transpose()?
        .flatten();

    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
        cell_to_pb_per_level,
        output_keep_idx,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &data_files,
        batch_files: &batch_files,
        preload: args.preload_data,
        proj_dim: args.collapse.proj_dim.max(args.n_latent),
        sort_dim: args.collapse.sort_dim,
        knn_cells: args.collapse.knn_cells,
        num_levels: args.collapse.num_levels,
        iter_opt: args.collapse.iter_opt,
        block_size: args.block_size,
        out: &args.out,
        max_features: args.hvg.n_hvg,
        feature_list_file: args.hvg.feature_list_file.as_deref(),
        refine: Some(args.collapse.pb_refine.to_params()),
        ignore_batch: args.collapse.ignore_batch,
        qc: args.qc.to_config(),
        qc_block_size: args.block_size,
        qc_report_out: args.qc.qc_report.as_deref(),
        feature_mask_fn: None,
        row_alignment: data_beans::sparse_io_vector::RowAlignment::default(),
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::default(),
        feature_kind: None,
        want_hierarchy: true,
        prebuilt_partition,
    })?;

    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();
    let n_features = data_vec.num_rows();
    let num_levels = collapsed_levels.len();
    let n_latent = args.n_latent;

    // Per-gene mean rate μ_d from the finest-level pseudobulk posterior — the
    // divisive gene-mean correction inside `anscombe_residual`. Full-D
    // (no coarsening); same null the topic encoder gets.
    let feature_mean = crate::topic::common::pseudobulk_feature_mean(
        finest_collapsed.mu_observed.posterior_mean(),
    );

    let dev = create_device(&args.device, args.device_no)?;
    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    info!(
        "input: {n_features} -> Gaussian encoder -> {n_latent} factors -> NB decoder (dim {n_features}), {num_levels} level(s)"
    );

    let gene_names = data_vec.row_names()?;
    let stop = setup_stop_handler();

    let mut encoder = build_encoder(
        n_features,
        n_latent,
        &args.encoder_layers,
        &feature_mean,
        &parameters,
        param_builder.clone(),
    )?;

    // One full-D NB decoder per pseudobulk level, sharing the encoder.
    let decoders: Vec<GaussianNbDecoder> = (0..num_levels)
        .map(|i| GaussianNbDecoder::new(n_features, n_latent, param_builder.pp(format!("dec_{i}"))))
        .collect::<candle_core::Result<Vec<_>>>()?;

    if let Some(prefix) = args.init_from.as_deref() {
        use crate::topic::warm_start::{warm_start_load, WarmStartCheck};
        warm_start_load(
            &parameters,
            prefix,
            &WarmStartCheck {
                model_type_expected: crate::topic::model_metadata::MODEL_TYPE_VAE,
                n_topics: n_latent,
                n_features_full: n_features,
                n_features_encoder: n_features,
                encoder_hidden: &args.encoder_layers,
                level_decoder_dims: &vec![n_features; num_levels],
                embedding_dim: None,
            },
        )?;
    }

    // Per-level (encoder-input, batch-null, decoder-target) triples — full-D,
    // no coarsening. The encoder reads `mixed` (μ observed), the decoder
    // reconstructs `target` (μ adjusted); `batch` is the per-cell null.
    let level_data: Vec<(Mat, Option<Mat>, Mat)> = collapsed_levels
        .iter()
        .map(sample_collapsed_data)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let level_refs: Vec<candle_util::vae::topic::LevelData> = level_data
        .iter()
        .map(|(a, b, c)| (a, b.as_ref(), c))
        .collect();

    let train_cfg = candle_util::vae::topic::TrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.epochs,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        // 0 ⇒ `smooth_topics` is a no-op: the raw Gaussian `z` reaches the
        // decoder unmodified (simplex smoothing would corrupt it).
        topic_smoothing: 0.0,
        grad_clip: args.grad_clip,
        stop: &stop,
        loss_hook: None,
    };
    let scores =
        candle_util::vae::topic::train_mixed(&level_refs, &mut encoder, &decoders, &train_cfg)?;
    TrainScores {
        llik: scores.llik,
        kl: scores.kl,
    }
    .to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    // Persist weights + per-gene mean, then move to CPU for threaded eval.
    info!("Writing model parameters");
    crate::topic::model_metadata::save_parameters(&parameters, &args.out)?;
    crate::topic::model_metadata::save_feature_mean(&feature_mean, &gene_names, &args.out)?;

    // Gene × factor loadings (the decoder weight `[D, K]`). Full-D, no
    // coarsening to expand.
    let finest_decoder = decoders.last().unwrap();
    crate::topic::decoder_output::write_dictionary_expanded(
        finest_decoder,
        None,
        n_features,
        &gene_names,
        &args.out,
    )?;

    let metadata = crate::topic::model_metadata::TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_VAE.into(),
        decoder_types: vec!["gauss_nb".into()],
        decoder_weights: vec![1.0],
        n_features_encoder: n_features,
        n_features_full: n_features,
        n_topics: n_latent,
        encoder_hidden: args.encoder_layers.clone(),
        num_levels,
        level_decoder_dims: vec![n_features; num_levels],
        adj_method: args.adj_method.as_str().into(),
        has_coarsening: false,
        embedding_dim: None,
        enc_context_size: None,
        dec_context_size: None,
        theta_mean: None,
        n_train_cells: Some(data_vec.num_columns()),
    };
    metadata.save(&args.out)?;

    // Encoder-only latent evaluation over all cells (CPU, multi-threaded).
    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(&parameters)?;
    let cpu_vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);
    let cpu_encoder = build_encoder(
        n_features,
        n_latent,
        &args.encoder_layers,
        &feature_mean,
        &parameters,
        cpu_vb.clone(),
    )?;

    let eval_config: EvaluateLatentConfig<GaussianNbDecoder> = EvaluateLatentConfig {
        dev: &cpu_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        feature_coarsening: None,
        decoder: None,
        refine_config: None,
    };
    let z_nk = evaluate_latent_by_encoder(&data_vec, &cpu_encoder, finest_collapsed, &eval_config)?;

    let cell_names = data_vec.column_names()?;
    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names, output_keep_idx.as_deref())?;

    crate::postprocess::viz_prep::write_cell_proj(
        &args.out,
        &proj_kn,
        &cell_names,
        output_keep_idx.as_deref(),
    )?;
    let has_cell_to_pb = if let Some(ref c2p) = cell_to_pb_per_level {
        crate::postprocess::viz_prep::write_cell_to_pb(
            &args.out,
            c2p,
            &cell_names,
            output_keep_idx.as_deref(),
        )?;
        true
    } else {
        false
    };

    let input: Vec<String> = data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = batch_files
        .as_deref()
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::Vae,
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: true,
        has_cell_proj: true,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: None,
        cell_embedding_suffix: None,
        default_colour_by: "cluster",
        has_latent: true,
        has_cell_to_pb,
    })?;

    info!("Done");
    Ok(())
}

/// Build a [`GaussianEncoder`] on the given `VarBuilder`. Shared by the
/// training-device and the CPU-rebuild (post-`move_varmap_to_cpu`) sites, which
/// differ only in the device backing `vb`.
fn build_encoder(
    n_features: usize,
    n_latent: usize,
    layers: &[usize],
    feature_mean: &[f32],
    parameters: &candle_nn::VarMap,
    vb: candle_nn::VarBuilder,
) -> anyhow::Result<GaussianEncoder> {
    Ok(GaussianEncoder::new(
        GaussianEncoderArgs {
            n_features,
            n_latent,
            layers,
            feature_mean: Some(feature_mean),
        },
        parameters,
        vb,
    )?)
}
