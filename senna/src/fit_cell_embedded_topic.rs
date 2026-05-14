//! `senna cell-embedded-topic` — hierarchical cell→PB pooling topic model.
//!
//! Same model *type* as `senna indexed-topic` (shared ρ `[D, H]` table,
//! ETM-factorized decoder, multi-level PB training, lazy-ρ optimizer). The
//! difference is structural and lives in the encoder + loader: a PB sample
//! is treated as a *pool of cells*, and that pooling is moved **into the
//! encoder** as a two-level gene→cell→PB `EmbeddingBag`. Because single
//! cells are the genuinely sparse atoms, the per-minibatch touched ρ-row
//! set shrinks to single-digit % of D and the lazy-ρ optimizer pays off
//! end-to-end.
//!
//! v1 trains and writes the model + training trace only. Latent inference
//! and `senna predict` are **not** wired yet — see the `warn!` at the end
//! of [`fit_cell_embedded_topic_model`].

use crate::embed_common::*;
use crate::fit_indexed_topic::{remap_graph_to_subset, FeatureNameKindArg};
use crate::topic::common::{
    create_device, load_and_collapse, setup_stop_handler, LoadCollapseArgs, PreparedData,
};
use crate::topic::model_metadata::MODEL_TYPE_CELL_EMBEDDED;
use crate::topic::train_cell_embedded::{
    extract_cell_samples, train_mixed_cell, CellEmbeddedTrainConfig,
};
use crate::topic::train_indexed::{write_feature_embedding, write_indexed_dictionary};

use candle_util::candle_decoder_embedded_topic::EmbeddedTopicDecoder;
use candle_util::candle_encoder_cell_embedded::{CellEmbeddedEncoder, CellEmbeddedEncoderArgs};
use candle_util::candle_value_transform::ValueEmbeddingConfig;
use log::warn;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct CellEmbeddedTopicArgs {
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
                     {out}.feature_embedding.parquet ρ at full gene resolution\n  \
                     {out}.delta.parquet            per-batch effects (if --batch-files)\n  \
                     {out}.log_likelihood.parquet   training loss trace\n  \
                     {out}.safetensors              encoder+decoder weights\n\n\
                     Note: latent / predict are not wired in this revision."
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
        help = "Partition depth: ≤ 2^d + 1 pseudobulk groups"
    )]
    sort_dim: usize,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        help = "Skip per-batch correction; treat all cells as a single batch"
    )]
    ignore_batch: bool,

    #[arg(
        short = 'w',
        long = "warm-start",
        help = "Warm-start projection file (cell × k)"
    )]
    warm_start_proj_file: Option<Box<str>>,

    #[arg(
        long = "init-from",
        help = "Initialize encoder + decoder weights from a previously trained \
                cell-embedded-topic model (matching {prefix}.model.json + \
                {prefix}.safetensors)."
    )]
    init_from: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10,
        help = "In-batch k-NN for pb-sample merging"
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Multi-level coarsening levels"
    )]
    num_levels: usize,

    #[arg(
        long,
        default_value_t = 30,
        help = "Batch-correction optimizer iterations"
    )]
    iter_opt: usize,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    block_size: Option<usize>,

    #[command(flatten)]
    pb_refine: crate::refine_weighting::PbRefineArgs,

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
        long_help = "Example: 128,1024,128. The [fg, bg] pool concat (2H) feeds\n\
                     the first layer; final layer → topics."
    )]
    encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 100, help = "Training minibatch size (pseudobulk samples)")]
    minibatch_size: usize,

    #[arg(long, alias = "lr", default_value_t = 0.05, help = "Adam learning rate")]
    learning_rate: f32,

    #[arg(
        long,
        default_value_t = 5.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off; typical 0.5–5.0)"
    )]
    grad_clip: f32,

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
        help = "Batch adjustment (batch|residual)"
    )]
    adj_method: AdjMethod,

    #[arg(
        long,
        default_value_t = false,
        help = "Load all columns into memory before training"
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Treat input files as modalities of the same cells, glued by raw barcode."
    )]
    multiome: bool,

    #[command(flatten)]
    hvg: crate::hvg::HvgCliArgs,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Uniform smoothing of topic proportions during training"
    )]
    topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 512,
        help = "Encoder foreground context window (top-K features per cell)",
        long_help = "Each member cell keeps its top-K features by value. Cells are\n\
                     genuinely sparse, so this is the parameter that drives the\n\
                     lazy-ρ optimizer's touched-row sparsity."
    )]
    fg_context_size: usize,

    #[arg(
        long,
        help = "Encoder background context window (top-K μ_residual features per PB; \
                default: --fg-context-size)"
    )]
    bg_context_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-feature embedding dimension H (0 = auto = 2 × n-latent-topics)"
    )]
    embedding_dim: usize,

    #[arg(
        long,
        help = "Intensity-embedding vocabulary size (value bins per scale; \
                default: --embedding-dim). A resolution knob, not a perf one \
                — the per-bin width is H and the lookup cost is independent \
                of vocab size."
    )]
    value_vocab_size: Option<usize>,

    #[command(flatten)]
    amort_refine: crate::refine_weighting::AmortRefineArgs,

    #[arg(
        long,
        help = "Decoder context window (default: --fg-context-size)"
    )]
    decoder_context_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Use dense AdamW over the full ρ [D,H] table every minibatch \
                instead of the lazy touched-row-only optimizer."
    )]
    dense_rho_update: bool,

    #[arg(
        long,
        help = "Optional feature-feature edge list (TSV/CSV) — activates a \
                Ball-Karrer-Newman degree-corrected Poisson graph likelihood on ρ."
    )]
    feature_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Also try forward/reverse prefix matching when resolving \
                feature-network names against the gene axis"
    )]
    feature_network_prefix_match: bool,

    #[arg(
        long,
        help = "Alias-splitting delimiter for feature-network name resolution."
    )]
    feature_network_delim: Option<char>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Relative weight λ_G of the graph Poisson log-likelihood. \
                Ignored without --feature-network."
    )]
    graph_loss_weight: f32,

    #[arg(
        long,
        default_value_t = 0,
        help = "Run the graph Poisson likelihood only for the first N epochs, \
                then drop it (0 = never drop). Ignored without --feature-network."
    )]
    graph_warmup_epochs: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Shared-neighbor edge augmentation threshold (0 disables)."
    )]
    feature_network_snn_min_shared: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Restrict the model to features covered by --feature-network."
    )]
    feature_network_restrict: bool,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Per-name canonicalization across input backends"
    )]
    feature_name_kind: FeatureNameKindArg,

    #[command(flatten)]
    cnv: CnvArgs,
}

pub fn fit_cell_embedded_topic_model(args: &CellEmbeddedTopicArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let k = args.n_latent_topics;
    let h = if args.embedding_dim == 0 {
        let auto = 2 * k;
        info!("--embedding-dim not set; defaulting to 2 × K = {auto}");
        auto
    } else {
        args.embedding_dim
    };
    if h < k {
        anyhow::bail!(
            "--embedding-dim ({h}) < --n-latent-topics ({k}). β = softmax(α·ρᵀ) is rank ≤ H, \
             so pass --embedding-dim >= {k} (recommended: {} for headroom).",
            k * 2,
        );
    }
    if h < 2 * k {
        warn!(
            "--embedding-dim ({h}) is at the β-rank limit for --n-latent-topics ({k}); \
             recommend --embedding-dim >= {} for headroom.",
            k * 2,
        );
    }

    let (effective_multiome, effective_hvg_n, effective_hvg_list) =
        crate::hvg::resolve_multiome_with_hvg(args.multiome, args.data_files.len(), &args.hvg);

    // --feature-network-restrict: parse the network once inside the
    // row-mask callback, then remap to the post-mask axis. Identical
    // machinery to `fit_indexed_topic`.
    let restrict_path: Option<&str> = if args.feature_network_restrict {
        args.feature_network.as_deref()
    } else {
        None
    };
    let net_prefix = args.feature_network_prefix_match;
    let net_delim = args.feature_network_delim;
    let net_snn = args.feature_network_snn_min_shared;
    use matrix_util::pair_graph::FeaturePairGraph;
    use std::cell::RefCell;
    use std::rc::Rc;
    let cached_graph: Rc<RefCell<Option<FeaturePairGraph>>> = Rc::new(RefCell::new(None));
    let cached_keep: Rc<RefCell<Option<Vec<bool>>>> = Rc::new(RefCell::new(None));
    let mask_fn_box: Option<Box<crate::topic::common::FeatureMaskFn>> = restrict_path.map(|p| {
        let path: String = p.to_string();
        let cached_graph = Rc::clone(&cached_graph);
        let cached_keep = Rc::clone(&cached_keep);
        let f: Box<crate::topic::common::FeatureMaskFn> = Box::new(move |row_names| {
            let mut graph =
                FeaturePairGraph::from_edge_list(&path, row_names.to_vec(), net_prefix, net_delim)?;
            graph.augment_with_snn(net_snn);
            let keep: Vec<bool> = graph.feature_degrees().iter().map(|&d| d > 0).collect();
            let n_keep = keep.iter().filter(|&&k| k).count();
            info!(
                "--feature-network-restrict: keeping {} / {} features with ≥1 edge",
                n_keep,
                row_names.len()
            );
            *cached_keep.borrow_mut() = Some(keep.clone());
            *cached_graph.borrow_mut() = Some(graph);
            Ok(keep)
        });
        f
    });
    let feature_mask_fn: Option<&crate::topic::common::FeatureMaskFn> = mask_fn_box.as_deref();

    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn: _,
        cell_to_pb_per_level,
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
        max_features: effective_hvg_n,
        feature_list_file: effective_hvg_list,
        feature_mask_fn,
        row_alignment: data_beans::sparse_io_vector::RowAlignment::default(),
        column_alignment: if effective_multiome {
            data_beans::sparse_io_vector::ColumnAlignment::Union
        } else {
            data_beans::sparse_io_vector::ColumnAlignment::Disjoint
        },
        feature_kind: args.feature_name_kind.clone().into(),
        refine: Some(args.pb_refine.to_params()),
        ignore_batch: args.ignore_batch,
        want_hierarchy: true,
    })?;

    let cell_to_pb_per_level = cell_to_pb_per_level.ok_or_else(|| {
        anyhow::anyhow!("load_and_collapse did not return the cell→pb hierarchy")
    })?;

    let n_features_full = data_vec.num_rows();
    let num_levels = collapsed_levels.len();
    let n_topics = args.n_latent_topics;

    let dev = create_device(&args.device, args.device_no)?;
    let parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let fg_context_size = args.fg_context_size;
    let bg_context_size = args.bg_context_size.unwrap_or(fg_context_size);
    let dec_context_size = args.decoder_context_size.unwrap_or(fg_context_size);

    // Cell-embedded encoder: two value-weighted ρ pools (FG member cells +
    // BG PB residual) concatenated into the latent head. The value
    // transform is the learned dual-binned intensity-embedding gate
    // (Anscombe retired); vocab size defaults to the embedding dim H.
    let value_embedding = ValueEmbeddingConfig {
        n_vocab: args.value_vocab_size.unwrap_or(h),
    };
    info!(
        "intensity-embedding value transform: vocab={} (per-bin width H={})",
        value_embedding.n_vocab, h
    );
    let base_encoder = CellEmbeddedEncoder::new(
        CellEmbeddedEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
            value_embedding,
        },
        &parameters,
        param_builder.pp("enc"),
    )?;

    // Resolve the ρ `Var` backing the encoder's feature-embedding table —
    // matched by tensor id — so the lazy optimizer can step it sparsely.
    let rho_tid = base_encoder.feature_embeddings().id();
    let rho_var = parameters
        .all_vars()
        .into_iter()
        .find(|v| v.id() == rho_tid)
        .ok_or_else(|| anyhow::anyhow!("ρ feature-embedding Var not found in VarMap"))?;

    // Per-level decoders share the encoder's ρ (ETM tying); each learns
    // only its own topic embeddings α_{level} [K, H].
    let shared_rho = base_encoder.feature_embeddings().clone();
    let decoders: Vec<EmbeddedTopicDecoder> = (0..num_levels)
        .map(|i| {
            EmbeddedTopicDecoder::new(
                n_topics,
                shared_rho.clone(),
                param_builder.pp(format!("dec_{i}")),
            )
            .expect("decoder creation")
        })
        .collect();

    if let Some(prefix) = args.init_from.as_deref() {
        use crate::topic::warm_start::{warm_start_load, WarmStartCheck};
        warm_start_load(
            &parameters,
            prefix,
            &WarmStartCheck {
                model_type_expected: MODEL_TYPE_CELL_EMBEDDED,
                n_topics,
                n_features_full,
                n_features_encoder: n_features_full,
                encoder_hidden: &args.encoder_layers,
                level_decoder_dims: &vec![n_features_full; num_levels],
                embedding_dim: Some(h),
                value_embedding: Some(value_embedding.n_vocab),
            },
        )?;
    }

    info!(
        "input: {} -> cell-embedded encoder (emb={}, fg_ctx={}, bg_ctx={}) -> {} decoders (D={}, ctx={})",
        n_features_full, h, fg_context_size, bg_context_size, num_levels, n_features_full, dec_context_size,
    );

    let gene_names = data_vec.row_names()?;
    let stop = setup_stop_handler();

    info!("Computing NB-Fisher weights for shortlist scoring");
    let shortlist_weights: Vec<f32> =
        crate::empirical_dict::compute_nb_fisher_weights(&data_vec, args.block_size)?;
    // Decoder + FG/BG Fisher loss weights are the same NB-Fisher per-gene
    // quantity already computed for shortlist selection.
    let feature_fisher_weights = shortlist_weights.clone();

    // Optional Ball-Karrer-Newman feature-graph likelihood on ρ_graph.
    // Reuses the pre-mask graph cached by the row-mask callback when
    // --feature-network-restrict is set (one parse total instead of two).
    let cached_pre_mask = cached_graph.borrow_mut().take();
    let cached_pre_mask_keep = cached_keep.borrow_mut().take();
    let graph_for_bkn: Option<FeaturePairGraph> = match (cached_pre_mask, cached_pre_mask_keep) {
        (Some(pre_graph), Some(keep)) => Some(remap_graph_to_subset(pre_graph, &keep)),
        _ => args
            .feature_network
            .as_deref()
            .map(|path| -> anyhow::Result<_> {
                let mut g = FeaturePairGraph::from_edge_list(
                    path,
                    gene_names.to_vec(),
                    args.feature_network_prefix_match,
                    args.feature_network_delim,
                )?;
                g.augment_with_snn(args.feature_network_snn_min_shared);
                Ok(g)
            })
            .transpose()?,
    };
    let graph_cfg: Option<crate::topic::graph_likelihood::PoissonGraphConfig> = graph_for_bkn
        .as_ref()
        .map(|g| {
            crate::topic::graph_likelihood::PoissonGraphConfig::build(
                g,
                n_features_full,
                h,
                args.graph_loss_weight,
                &param_builder,
                &dev,
            )
        })
        .transpose()?;

    // Extract the genuinely sparse single-cell atoms once; the per-cell
    // top-K samples + library-size factors are level-independent and
    // shared across every PB level via `Arc`.
    info!("Extracting per-cell sparse top-K samples + library-size factors");
    let (cell_samples, cell_size_factor) = extract_cell_samples(
        &data_vec,
        &shortlist_weights,
        fg_context_size,
        args.block_size,
    )?;
    let cell_samples = Arc::new(cell_samples);
    let cell_size_factor = Arc::new(cell_size_factor);

    let train_config = CellEmbeddedTrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.epochs,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        topic_smoothing: args.topic_smoothing,
        fg_context_size,
        bg_context_size,
        dec_context_size,
        stop: &stop,
        shortlist_weights: &shortlist_weights,
        feature_fisher_weights: &feature_fisher_weights,
        grad_clip: args.grad_clip,
        graph_warmup_epochs: args.graph_warmup_epochs,
        rho_var: &rho_var,
        lazy_rho: !args.dense_rho_update,
    };

    let scores = train_mixed_cell(
        &collapsed_levels,
        &cell_to_pb_per_level,
        cell_samples,
        cell_size_factor,
        &base_encoder,
        &decoders,
        &train_config,
        graph_cfg.as_ref(),
    )?;

    info!("Writing down the model parameters");

    let finest_decoder = decoders.last().unwrap();
    write_indexed_dictionary(finest_decoder, &gene_names, &args.out)?;
    write_feature_embedding(base_encoder.feature_embeddings(), &gene_names, &args.out)?;

    use crate::topic::model_metadata::{save_parameters, save_shortlist_weights, TopicModelMetadata};
    save_parameters(&parameters, &args.out)?;
    let metadata = TopicModelMetadata {
        model_type: MODEL_TYPE_CELL_EMBEDDED.into(),
        decoder_types: vec!["multinom".into()],
        decoder_weights: vec![1.0],
        n_features_encoder: n_features_full,
        n_features_full,
        n_topics,
        encoder_hidden: args.encoder_layers.clone(),
        num_levels,
        level_decoder_dims: vec![n_features_full; num_levels],
        adj_method: args.adj_method.as_str().into(),
        has_coarsening: false,
        embedding_dim: Some(h),
        enc_context_size: Some(fg_context_size),
        dec_context_size: Some(dec_context_size),
        value_vocab_size: Some(value_embedding.n_vocab),
        theta_mean: None,
    };
    metadata.save(&args.out)?;
    save_shortlist_weights(&shortlist_weights, &gene_names, &args.out)?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    // TODO(cell-embedded-topic): wire latent inference + `senna predict`.
    // The cell-level eval forward pass would re-run `extract_cell_samples`
    // + `CellGroupedInMemoryData` over the finest level (no shuffle) and
    // call `encoder.forward_cells_t(.., false)` per minibatch, then write
    // {out}.latent.parquet / {out}.senna.json as `indexed-topic` does.
    warn!(
        "cell-embedded-topic v1: trained model + trace written; latent inference \
         and `senna predict` are NOT wired yet (no {{out}}.latent.parquet / .senna.json)"
    );

    info!("Done");
    Ok(())
}
