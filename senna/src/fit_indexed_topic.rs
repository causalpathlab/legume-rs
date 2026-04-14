use crate::embed_common::*;
use crate::topic::common::*;
use crate::topic::eval_indexed::*;
use crate::topic::train_indexed::*;

use candle_util::candle_decoder_indexed_topic::*;
use candle_util::candle_encoder_indexed::*;
use matrix_param::dmatrix_gamma::GammaMatrix;

#[derive(Args, Debug)]
pub struct IndexedTopicArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Data files",
        long_help = "Data files to be processed.\n\
		     Each file should be specified as a path.\n\
		     Multiple files can be provided (space or comma separated)."
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
		     - {out}.latent.parquet (log-softmax topic proportions)\n"
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension"
    )]
    proj_dim: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 10,
        help = "Top {d} components of projection"
    )]
    sort_dim: usize,

    #[arg(long, short, value_delimiter(','), help = "Batch membership files")]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(short = 'w', long = "warm-start", help = "Warm start projection file")]
    warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours within each batch"
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of multi-level coarsening levels"
    )]
    num_levels: usize,

    #[arg(
        long,
        value_enum,
        default_value = "mixed",
        help = "Multi-level training schedule"
    )]
    level_schedule: LevelSchedule,

    #[arg(long, default_value_t = 30, help = "Optimization iterations")]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing"
    )]
    block_size: usize,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of latent topics"
    )]
    n_latent_topics: usize,

    #[arg(
        long,
        short = 'e',
        value_delimiter(','),
        default_values_t = vec![128, 1024, 128],
        help = "Encoder layers (comma-separated)"
    )]
    encoder_layers: Vec<usize>,

    #[arg(
        long,
        short = 'i',
        default_value_t = 1000,
        help = "Number of training epochs"
    )]
    epochs: usize,

    #[arg(long, short = 'j', default_value_t = 5, help = "Data jitter interval")]
    jitter_interval: usize,

    #[arg(long, default_value_t = 100, help = "Minibatch size")]
    minibatch_size: usize,

    #[arg(long, alias = "lr", default_value_t = 0.05, help = "Learning rate")]
    learning_rate: f32,

    #[arg(long, value_enum, default_value = "cpu", help = "Candle device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "A device for cuda")]
    device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value = "residual",
        help = "Adjustment method"
    )]
    adj_method: AdjMethod,

    #[arg(long, default_value_t = false, help = "Preload all columns data")]
    preload_data: bool,

    #[arg(long, default_value_t = 0.01, help = "Topic smoothing during training")]
    topic_smoothing: f64,

    // Indexed-specific args
    #[arg(
        long,
        default_value_t = 512,
        help = "Top-K features per sample (context window size)",
        long_help = "Number of top features to keep per sample by value.\n\
                     Each sample selects its top-K features; minibatches use\n\
                     the union of selected indices. Smaller K = faster decoder."
    )]
    context_size: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Feature embedding dimension",
        long_help = "Dimensionality of per-feature embeddings.\n\
                     Features are aggregated via [N, S] × [S, H] matmul\n\
                     instead of dense [N, D] × [D, M] in the standard encoder."
    )]
    embedding_dim: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Epochs of VCD training before switching to SGVB",
        long_help = "Number of initial epochs using variational contrastive divergence (VCD).\n\
		     VCD refines encoder samples via elliptical slice sampling (ESS),\n\
		     then switches to standard SGVB for remaining epochs.\n\
		     Set to 0 to use SGVB only (default)."
    )]
    vcd_epochs: usize,

    #[arg(
        long,
        default_value_t = 5,
        alias = "ess-steps",
        help = "ESS steps per minibatch during VCD epochs"
    )]
    vcd_ess_steps: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Max shrink iterations per ESS step"
    )]
    ess_max_shrink: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-cell refinement steps at inference",
        long_help = "Number of gradient steps for per-cell topic refinement at inference time.\n\
		     Optimizes topic logits against the frozen decoder likelihood,\n\
		     anchored to the encoder output via L2 regularization.\n\
		     Set to 0 to disable (default)."
    )]
    refine_steps: usize,

    #[arg(long, default_value_t = 0.01, help = "Learning rate for refinement")]
    refine_lr: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 regularization strength for refinement"
    )]
    refine_reg: f64,

    #[arg(
        long,
        help = "Decoder context window size (defaults to --context-size)",
        long_help = "Top-K features per sample for the decoder side.\n\
                     Defaults to the encoder's --context-size when not set."
    )]
    decoder_context_size: Option<usize>,

    #[arg(
        short = 'x',
        long,
        value_delimiter = ',',
        help = "Bulk data files for joint deconvolution (.parquet, .tsv.gz)"
    )]
    bulk_data_files: Option<Vec<Box<str>>>,

    // CNV detection args
    #[command(flatten)]
    cnv: CnvArgs,
}

pub fn fit_indexed_topic_model(args: &IndexedTopicArgs) -> anyhow::Result<()> {
    let PreparedData {
        data_vec,
        collapsed_levels,
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

    // Read bulk data aligned to SC genes
    let bulk = args
        .bulk_data_files
        .as_ref()
        .map(|files| read_bulk_data_aligned(files, &gene_names))
        .transpose()?;

    // Compute per-level bulk delta
    let bulk_deltas: Option<Vec<GammaMatrix>> = bulk
        .as_ref()
        .map(|b| {
            collapsed_levels
                .iter()
                .map(|collapsed| estimate_bulk_delta(&b.data, collapsed))
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .transpose()?;

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
        vcd_epochs: args.vcd_epochs,
        vcd_ess_steps: args.vcd_ess_steps,
        ess_max_shrink: args.ess_max_shrink,
        stop: &stop,
    };

    let bulk_with_deltas: Option<(&Mat, &[GammaMatrix])> = match (&bulk_nd_full, &bulk_deltas) {
        (Some(full), Some(deltas)) => Some((full, deltas)),
        _ => None,
    };

    if args.vcd_epochs > 0 && matches!(args.level_schedule, LevelSchedule::Progressive) {
        log::warn!("--vcd-epochs is only supported with --level-schedule mixed; ignoring VCD");
    }

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

    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(&parameters)?;

    info!("Writing down the latent states");
    let eval_config = EvaluateLatentConfig {
        dev: &cpu_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        enc_context_size: args.context_size,
        dec_context_size,
        decoder: finest_decoder,
        refine_config: None,
    };
    let z_nk = evaluate_latent_by_indexed_encoder(
        &data_vec,
        &base_encoder,
        finest_collapsed,
        &eval_config,
    )?;

    // Evaluate bulk with standard encoder/decoder
    if let (Some(bulk), Some(bulk_deltas)) = (&bulk, &bulk_deltas) {
        let bulk_config = BulkEvalConfig {
            dev: &cpu_dev,
            enc_context_size: args.context_size,
            dec_context_size,
            refine_config: None,
            decoder: finest_decoder,
            gene_names: &gene_names,
            out_prefix: &args.out,
        };
        evaluate_bulk_samples(bulk, bulk_deltas, &base_encoder, &bulk_config)?;
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
            let topic_probs = z_nk.map(|x| x.exp());
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

    info!("Done");
    Ok(())
}
