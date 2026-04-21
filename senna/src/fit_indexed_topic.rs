use crate::embed_common::*;
use crate::topic::common::{
    create_device, load_and_collapse, move_varmap_to_cpu, setup_stop_handler, LoadCollapseArgs,
    PreparedData,
};
use crate::topic::eval_indexed::{evaluate_latent_by_indexed_encoder, EvaluateLatentConfig};
use crate::topic::train_indexed::{
    estimate_bulk_delta, evaluate_bulk_samples, train_mixed, train_progressive,
    write_indexed_dictionary, BulkEvalConfig, IndexedTrainConfig,
};

use candle_util::candle_decoder_indexed_topic::IndexedTopicDecoder;
use candle_util::candle_encoder_indexed::*;
use matrix_param::dmatrix_gamma::GammaMatrix;

#[derive(Args, Debug)]
pub struct IndexedTopicArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed (comma- or space-separated)\n\
                     and are concatenated column-wise on a shared feature set."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet       gene × topic loadings (log-prob)\n  \
                     {out}.latent.parquet           cell × topic log-softmax proportions\n  \
                     {out}.delta.parquet            per-batch effects (if --batch-files)\n  \
                     {out}.log_likelihood.parquet   training loss trace\n  \
                     {out}.safetensors              encoder+decoder weights\n  \
                     {out}.cell_proj.parquet        cached random projection (consumed by `senna layout`)\n  \
                     {out}.senna.json               run manifest consumed by `senna viz --from` and `senna plot --from`\n\n\
                     With -x bulk files: {out}.bulk_latent.parquet additionally."
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Target rank of the initial random sketch used to seed\n\
                     batch correction and multi-level pseudobulk collapsing."
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Partition depth: ≤ 2^d + 1 pseudobulk groups",
        long_help = "Binary-tree partitioning over the top d projection components.\n\
                     Produces at most 2^d + 1 pseudobulk leaves."
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell in the same order as its\n\
                     matching data file. Example: batch1.tsv,batch2.tsv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm-start projection file (cell × k)",
        long_help = "Skip random projection and use this matrix instead.\n\
                     Rows must match the concatenated cell order of the inputs."
    )]
    warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "In-batch k-NN for super-cell merging",
        long_help = "Number of within-batch nearest neighbours used when\n\
                     aggregating cells into pseudobulk super-cells."
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Multi-level coarsening levels",
        long_help = "Hierarchical pseudobulk refinement passes. Level sort dims are\n\
                     linearly spaced from 4 to --sort-dim. Set to 1 to disable."
    )]
    num_levels: usize,

    #[arg(
        long,
        value_enum,
        default_value = "mixed",
        help = "Multi-level training schedule (mixed|progressive)",
        long_help = "mixed       — every level trained simultaneously each epoch.\n\
                     progressive — coarse→fine, more epochs for coarser levels."
    )]
    level_schedule: LevelSchedule,

    #[arg(
        long,
        default_value_t = 30,
        help = "Batch-correction optimizer iterations",
        long_help = "Coordinate-descent steps when fitting the per-batch delta."
    )]
    iter_opt: usize,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    block_size: Option<usize>,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics (K)"
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder hidden layer sizes (comma-separated)",
        long_help = "Example: 128,1024,128 (input → 128 → 1024 → 128 → topics)."
    )]
    encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    epochs: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Posterior resampling interval (epochs)",
        long_help = "How often to jitter the collapsed targets by posterior resampling\n\
                     during VAE training."
    )]
    jitter_interval: usize,

    #[arg(
        long = "no-oversample-pb",
        default_value_t = false,
        help = "Disable PB augmentation (default: 2× groups, subsample each jitter)"
    )]
    no_oversample_pb: bool,

    #[arg(long, default_value_t = 100, help = "Training minibatch size")]
    minibatch_size: usize,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 0.05,
        help = "Adam learning rate"
    )]
    learning_rate: f32,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device (cpu|cuda|metal)"
    )]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "CUDA/Metal device index")]
    device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Batch adjustment (batch|residual)",
        long_help = "batch    — subtract per-batch pseudobulk mean.\n\
                     residual — divide by fitted delta per pseudobulk group."
    )]
    adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training"
    )]
    preload_data: bool,

    #[command(flatten)]
    hvg: crate::hvg::HvgCliArgs,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Uniform smoothing of topic proportions during training",
        long_help = "z_smooth = (1-α) z + α/K. Keeps every topic on the gradient path\n\
                     and prevents dead topics. Typical: 0.01–0.2. Set 0 to disable."
    )]
    topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 512,
        help = "Encoder context window (top-K features per cell)",
        long_help = "Each cell keeps its top-K features by value; minibatches use the\n\
                     union of selected indices. Smaller K = faster decoder."
    )]
    context_size: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Per-feature embedding dimension",
        long_help = "Features are pooled via [N,S] × [S,H] instead of the dense\n\
                     [N,D] × [D,M] used by the standard encoder."
    )]
    embedding_dim: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-cell refinement steps at inference (0 = off)",
        long_help = "Gradient steps that optimize topic logits against the frozen\n\
                     decoder likelihood, anchored to the encoder output by L2."
    )]
    refine_steps: usize,

    #[arg(long, default_value_t = 0.01, help = "Refinement learning rate")]
    refine_lr: f64,

    #[arg(long, default_value_t = 1.0, help = "Refinement L2 regularization")]
    refine_reg: f64,

    #[arg(
        long,
        help = "Decoder context window (default: --context-size)",
        long_help = "Top-K features per sample used on the decoder side.\n\
                     Defaults to the encoder's --context-size when not set."
    )]
    decoder_context_size: Option<usize>,

    #[arg(
        short = 'x',
        long,
        value_delimiter = ',',
        help = "Bulk expression files for joint deconvolution",
        long_help = "Accepts .parquet or .tsv.gz; rows are aligned to the SC gene set.\n\
                     Bulk samples are embedded using the trained encoder/decoder."
    )]
    bulk_data_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'm',
        long,
        help = "Marker TSV (gene<TAB>celltype) — labels anchors"
    )]
    markers: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Min z-score margin to assign a celltype to an anchor"
    )]
    anchor_margin: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cross-entropy penalty λ on β toward anchor prior (0 = off)"
    )]
    anchor_penalty: f32,

    #[command(flatten)]
    cnv: CnvArgs,
}

pub fn fit_indexed_topic_model(args: &IndexedTopicArgs) -> anyhow::Result<()> {
    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &args.data_files,
        batch_files: &args.batch_files,
        preload: args.preload_data,
        warm_start_proj_file: args.warm_start_proj_file.as_deref(),
        proj_dim: args.proj_dim.max(args.n_latent_topics),
        sort_dim: args.sort_dim,
        knn_cells: args.knn_cells,
        num_levels: args.num_levels,
        iter_opt: args.iter_opt,
        block_size: args.block_size,
        out: &args.out,
        oversample: !args.no_oversample_pb,
        max_features: args.hvg.n_hvg,
        feature_list_file: args.hvg.feature_list_file.as_deref(),
        refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
    })?;

    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();

    // 4. No feature coarsening for indexed model — both encoder and decoder
    //    use indexed top-K lookup, so D_full is efficient.
    //    Levels differ only in sample coarsening (N).
    let n_features_full = data_vec.num_rows();
    let num_levels = collapsed_levels.len();

    // 5. Train indexed topic model on collapsed data
    let n_topics = args.n_latent_topics;

    let dev = create_device(&args.device, args.device_no)?;

    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let dec_context_size = args.decoder_context_size.unwrap_or(args.context_size);

    let base_encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: args.embedding_dim,
            layers: &args.encoder_layers,
        },
        &parameters,
        param_builder.pp("enc"),
    )?;

    // Per-level decoders: all at D_full, levels differ in N (sample coarsening)
    let decoders: Vec<IndexedTopicDecoder> = (0..num_levels)
        .map(|i| {
            IndexedTopicDecoder::new(
                n_features_full,
                n_topics,
                param_builder.pp(format!("dec_{i}")),
            )
            .expect("decoder creation")
        })
        .collect();

    info!(
        "input: {} -> indexed encoder (emb={}, ctx={}) -> {} decoders (D={}, ctx={})",
        n_features_full,
        args.embedding_dim,
        args.context_size,
        num_levels,
        n_features_full,
        dec_context_size,
    );

    let gene_names = data_vec.row_names()?;

    // Data-driven anchor β prior. Indexed decoders all run at D_full, so
    // there's no finest-level coarsening to adapt anchor selection to.
    let markers = args
        .markers
        .as_deref()
        .map(|p| crate::marker_support::load_marker_info(p, &gene_names))
        .transpose()?;
    let anchor_prior = crate::topic::anchor_prior::AnchorPrior::from_pseudobulk(
        finest_collapsed,
        n_topics,
        markers.as_ref(),
        args.anchor_margin,
        None,
    )?;
    anchor_prior.write_side_outputs(&args.out, &gene_names, markers.as_ref())?;

    // Per-level anchor tensors. Indexed decoders all run at D_full, so
    // no feature coarsening applies — one `None` per level.
    let level_coarsenings_none: Vec<Option<FeatureCoarsening>> =
        (0..num_levels).map(|_| None).collect();
    let anchor_tensors = anchor_prior.per_level_device_tensors(&level_coarsenings_none, &dev)?;
    // β init from anchor prior disabled — random init + anchor penalty
    // during training avoids locking the dictionary too early.

    // Read bulk data aligned to SC genes
    let bulk = args
        .bulk_data_files
        .as_ref()
        .map(|files| read_bulk_data_aligned(files, &gene_names))
        .transpose()?;

    // Compute per-level bulk delta
    let bulk_deltas: Option<Vec<GammaMatrix>> = bulk.as_ref().map(|b| {
        collapsed_levels
            .iter()
            .map(|collapsed| estimate_bulk_delta(&b.data, collapsed))
            .collect::<Vec<_>>()
    });

    let stop = setup_stop_handler();

    // Bulk data at full D
    let bulk_nd_full: Option<Mat> = bulk.as_ref().map(|b| b.data.transpose());

    let train_config = IndexedTrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.epochs,
        jitter_interval: args.jitter_interval,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        topic_smoothing: args.topic_smoothing,
        enc_context_size: args.context_size,
        dec_context_size,
        sort_dim_budget: 1usize << args.sort_dim,
        stop: &stop,
        anchor_prior_per_level: Some(&anchor_tensors),
        anchor_penalty: args.anchor_penalty,
    };

    let bulk_with_deltas: Option<(&Mat, &[GammaMatrix])> = match (&bulk_nd_full, &bulk_deltas) {
        (Some(full), Some(deltas)) => Some((full, deltas)),
        _ => None,
    };

    let scores = match args.level_schedule {
        LevelSchedule::Progressive => train_progressive(
            &collapsed_levels,
            &base_encoder,
            &decoders,
            &train_config,
            bulk_with_deltas,
        )?,
        LevelSchedule::Mixed => train_mixed(
            &collapsed_levels,
            &base_encoder,
            &decoders,
            &train_config,
            bulk_with_deltas,
        )?,
    };

    info!("Writing down the model parameters");

    // Use finest-level decoder for output
    let finest_decoder = decoders.last().unwrap();
    write_indexed_dictionary(finest_decoder, &gene_names, &args.out)?;

    // Move VarMap to CPU, then rebuild encoder + finest-level decoder from the
    // CPU Vars. The original structs still hold CUDA/Metal tensors internally
    // and will trigger "device mismatch" errors if reused on CPU indices.
    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(&parameters)?;

    let cpu_vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);
    let cpu_encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: args.embedding_dim,
            layers: &args.encoder_layers,
        },
        &parameters,
        cpu_vb.pp("enc"),
    )?;
    let finest_dec_idx = decoders.len().saturating_sub(1);
    let cpu_finest_decoder = IndexedTopicDecoder::new(
        n_features_full,
        n_topics,
        cpu_vb.pp(format!("dec_{finest_dec_idx}")),
    )?;

    let refine_config = if args.refine_steps > 0 {
        Some(
            candle_util::candle_topic_refinement::TopicRefinementConfig {
                num_steps: args.refine_steps,
                learning_rate: args.refine_lr,
                regularization: args.refine_reg,
            },
        )
    } else {
        None
    };

    info!("Writing down the latent states");
    let eval_config = EvaluateLatentConfig {
        dev: &cpu_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        enc_context_size: args.context_size,
        dec_context_size,
        decoder: &cpu_finest_decoder,
        refine_config: refine_config.as_ref(),
    };
    let z_nk = evaluate_latent_by_indexed_encoder(
        &data_vec,
        &cpu_encoder,
        finest_collapsed,
        &eval_config,
    )?;

    // Evaluate bulk with the CPU encoder/decoder for consistency.
    if let (Some(bulk), Some(bulk_deltas)) = (&bulk, &bulk_deltas) {
        let bulk_config = BulkEvalConfig {
            dev: &cpu_dev,
            enc_context_size: args.context_size,
            dec_context_size,
            refine_config: refine_config.as_ref(),
            decoder: &cpu_finest_decoder,
            gene_names: &gene_names,
            out_prefix: &args.out,
        };
        evaluate_bulk_samples(bulk, bulk_deltas, &cpu_encoder, &bulk_config)?;
    }

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    let cell_names = data_vec.column_names()?;

    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        None,
    )?;

    // CNV detection using topic proportions
    let gene_names = data_vec.row_names()?;
    let cnv_positions = crate::cnv_pseudobulk::load_gene_positions(&args.cnv, &gene_names)?;

    if let Some(positions) = cnv_positions {
        if let Some(batch_labels) = crate::cnv_pseudobulk::reconstruct_batch_labels(&data_vec) {
            let topic_probs = z_nk.map(f32::exp);
            let cnv_config = crate::cnv_pseudobulk::build_cnv_config(&args.cnv);

            let cnv_result = crate::cnv_pseudobulk::detect_cnv_topic_informed(
                data_vec,
                &topic_probs,
                &batch_labels,
                &positions,
                &cnv_config,
            )?;

            crate::cnv_pseudobulk::write_cnv_results(&cnv_result, &args.out, &gene_names)?;
        } else {
            info!("CNV detection: skipped (no batch information)");
        }
    }

    crate::postprocess::viz_prep::write_cell_proj(&args.out, &proj_kn, &cell_names)?;

    let input: Vec<String> = args.data_files.iter().map(|s| s.to_string()).collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: "itopic",
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_markers: args.markers.is_some(),
        has_model: true,
        has_cell_proj: true,
        default_colour_by: "topic",
    })?;

    info!("Done");
    Ok(())
}
