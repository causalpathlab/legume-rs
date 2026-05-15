use crate::embed_common::*;
use crate::topic::common::{
    create_device, load_and_collapse, move_varmap_to_cpu, setup_stop_handler, LoadCollapseArgs,
    PreparedData,
};
use crate::topic::eval_indexed::{evaluate_latent_by_indexed_encoder, EvaluateLatentConfig};
use crate::topic::train_indexed::{
    estimate_bulk_delta, evaluate_bulk_samples, train_mixed, write_feature_embedding,
    write_indexed_dictionary, BulkEvalConfig, IndexedTrainConfig,
};

use candle_util::candle_decoder_embedded_topic::EmbeddedTopicDecoder;
use candle_util::candle_encoder_indexed::*;
use candle_util::candle_value_transform::ValueEmbeddingConfig;
use log::warn;
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
                     {out}.senna.json               run manifest consumed by `senna layout --from` and `senna plot --from`\n\n\
                     With -x bulk files: {out}.bulk_latent.parquet additionally."
    )]
    out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file",
        long_help = "Each file lists a batch label per cell in the same order as its\n\
                     matching data file. Example: batch1.tsv,batch2.tsv"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    collapse: crate::refine_weighting::CollapseArgs,

    #[arg(
        long = "init-from",
        help = "Initialize encoder + decoder weights from a previously trained model",
        long_help = "Path prefix of a model saved by `senna indexed-topic`\n\
                     (matching {prefix}.model.json + {prefix}.safetensors).\n\
                     Architecture must match: same K, encoder layers,\n\
                     embedding_dim, and n_features_full. Cross-gene-set\n\
                     warm-start is not supported — train on the same gene set."
    )]
    init_from: Option<Box<str>>,

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

    #[arg(
        long,
        default_value_t = false,
        help = "Treat input files as modalities of the same cells, glued by raw barcode.",
        long_help = "Patchy multi-modal (multiome) load. Each file keeps its own \
                     feature space (no cross-file barcode suffixing); cells are \
                     unioned across files by raw barcode — a cell observed only in \
                     RNA contributes triplets just to the RNA row block, ATAC-only \
                     cells just to the ATAC block, and shared cells get both. \
                     Disables `@<basename>` suffixing on cell names. Maps to \
                     `ColumnAlignment::Union` in the loader.\n\
                     \n\
                     With --multiome, batch resolution is constrained: a single \
                     --batch-files file is allowed (one label per unified cell), or \
                     embedded `@batch` tags in raw column names that AGREE across \
                     modalities. The default `@<filename>` fallback is disabled — a \
                     cell can come from multiple files and cannot carry two labels."
    )]
    multiome: bool,

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
        default_value_t = 0,
        help = "Per-feature embedding dimension H (0 = auto = 2 × n-latent-topics)",
        long_help = "Dimension H of the per-gene embedding ρ ∈ ℝ^{D×H}. ρ is shared\n\
                     between encoder (value-weighted pool over each cell's top-K\n\
                     features) and decoder (β_kd = log_softmax_d(α_k · ρ_dᵀ), with\n\
                     α ∈ ℝ^{K×H} as topic embeddings). β is rank ≤ H, so H must be\n\
                     ≥ K (--n-latent-topics) for K independent topics to be\n\
                     representable. Default 0 resolves to 2K for headroom. Set\n\
                     explicitly to override; H < K errors at startup."
    )]
    embedding_dim: usize,

    #[arg(
        long,
        default_value_t = 16,
        help = "Intensity-embedding bin count (log1p-scale value bins). \
                A resolution knob, not a perf one — the per-bin width is H and \
                the lookup cost is independent of bin count."
    )]
    n_value_bins: usize,

    #[command(flatten)]
    amort_refine: crate::refine_weighting::AmortRefineArgs,

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
        long,
        help = "Optional feature-feature edge list (TSV/CSV) — wires a \
                graph-attention layer into the indexed encoder. Per-cell \
                sub-adjacency on the measured top-K acts as an additive \
                edge bias in attention so the encoder can pool signal \
                across functionally related genes. Edges may be intra- \
                or cross-modal (gene-gene PPI, peak-gene ABC, ATAC- \
                derived regulatory links). Edge names are resolved \
                against the loaded gene axis."
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
        help = "Alias-splitting delimiter for feature-network name resolution. \
                When set (e.g. '_'), each row name is registered under its \
                full form AND every split component, so a row like \
                `ENSG00000105329_TGFB1` matches network edges that name \
                *either* `ENSG00000105329` *or* `TGFB1` (via matrix-util's \
                GeneIndexResolver — both aliases point to the same row)"
    )]
    feature_network_delim: Option<char>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Shared-neighbor edge augmentation threshold (0 disables). \
                Adds a synthetic edge between any pair that share ≥ N \
                neighbors in the raw network. Densifies incomplete graphs \
                before the GAT sees them — same machinery as pinto's \
                `--snn-min-shared`. Set 2–3 for sparse networks; \
                BioGRID-scale dense PPIs usually don't need it."
    )]
    feature_network_snn_min_shared: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Restrict the model to features covered by --feature-network. \
                Features with zero edges in the network (after alias \
                resolution + optional SNN augmentation) are physically \
                dropped from the data axis before projection, collapse, \
                and training. Useful when the network defines the in-scope \
                feature set (e.g. PPI-restricted topics)."
    )]
    feature_network_restrict: bool,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Per-name canonicalization across input backends",
        long_help = "How row names align across `--data-files`. \
                     `auto` — sniff sampled row names and pick: \
                     locus-overlap if ≥50% parse as `chr:start-end`, \
                     gene if ≥50% contain `_`, exact otherwise (default). \
                     `exact` — strict string match. \
                     `gene` — also register each `_`-split component as an \
                     alias (so `ENSG000_TGFB1` and `TGFB1` resolve to the \
                     same row). \
                     `locus` — normalize `chr1:1000-2000`, `1:1000-2000`, \
                     etc. to a canonical form. \
                     `locus-overlap` — same as `locus` plus cluster any \
                     intervals that overlap on the same chromosome \
                     (useful for cross-dataset ATAC peak sets called \
                     independently)."
    )]
    feature_name_kind: FeatureNameKindArg,

    #[command(flatten)]
    cnv: CnvArgs,
}

#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum FeatureNameKindArg {
    #[default]
    Auto,
    Exact,
    Gene,
    Locus,
    LocusOverlap,
    Mixed,
}

impl From<FeatureNameKindArg> for Option<auxiliary_data::feature_names::FeatureNameKind> {
    fn from(arg: FeatureNameKindArg) -> Self {
        use auxiliary_data::feature_names::FeatureNameKind;
        match arg {
            FeatureNameKindArg::Auto => None,
            FeatureNameKindArg::Exact => Some(FeatureNameKind::Exact),
            FeatureNameKindArg::Gene => Some(FeatureNameKind::Gene { delim: '_' }),
            FeatureNameKindArg::Locus => Some(FeatureNameKind::Locus {
                merge_overlapping: false,
            }),
            FeatureNameKindArg::LocusOverlap => Some(FeatureNameKind::Locus {
                merge_overlapping: true,
            }),
            FeatureNameKindArg::Mixed => Some(FeatureNameKind::Mixed),
        }
    }
}

/// Renumber a `FeaturePairGraph`'s edges + names from a pre-mask axis to
/// the post-mask one defined by `keep`. Used to avoid re-parsing the same
/// edge list twice when `--feature-network-restrict` is set.
pub(crate) fn remap_graph_to_subset(
    pre: matrix_util::pair_graph::FeaturePairGraph,
    keep: &[bool],
) -> matrix_util::pair_graph::FeaturePairGraph {
    debug_assert_eq!(keep.len(), pre.n_features);
    let mut old_to_new: Vec<Option<usize>> = vec![None; keep.len()];
    let mut next = 0usize;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            old_to_new[i] = Some(next);
            next += 1;
        }
    }
    let feature_names: Vec<Box<str>> = pre
        .feature_names
        .into_iter()
        .zip(keep.iter())
        .filter_map(|(n, &k)| k.then_some(n))
        .collect();
    let feature_edges: Vec<(usize, usize)> = pre
        .feature_edges
        .into_iter()
        .map(|(u, v)| (old_to_new[u].expect("kept"), old_to_new[v].expect("kept")))
        .collect();
    matrix_util::pair_graph::FeaturePairGraph {
        feature_names,
        n_features: next,
        feature_edges,
    }
}

pub fn fit_indexed_topic_model(args: &IndexedTopicArgs) -> anyhow::Result<()> {
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
             so at most {h} linearly independent topics can be represented — the remaining \
             {} would collapse onto linear combinations. Pass --embedding-dim >= {k} \
             (recommended: {} for headroom), or omit --embedding-dim to use the 2K default.",
            k - h,
            k * 2,
        );
    }
    if h < 2 * k {
        warn!(
            "--embedding-dim ({h}) is at the β-rank limit for --n-latent-topics ({k}); \
             topics may collapse during training. Recommend --embedding-dim >= {} for headroom.",
            k * 2,
        );
    }

    let (effective_multiome, effective_hvg_n, effective_hvg_list) =
        crate::hvg::resolve_multiome_with_hvg(args.multiome, args.data_files.len(), &args.hvg);

    // When --feature-network-restrict is set, parse the network once
    // inside the row-mask callback. The cached graph carries pre-mask
    // indices; after mask_rows physically drops dropped-degree features,
    // we remap to the post-mask axis (rather than parse the file again).
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
        // Own the path so the closure satisfies `'static`.
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
                "--feature-network-restrict: keeping {} / {} features with ≥1 \
                     edge in the network",
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
        proj_kn,
        cell_to_pb_per_level: _,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &args.data_files,
        batch_files: &args.batch_files,
        preload: args.preload_data,
        proj_dim: args.collapse.proj_dim.max(args.n_latent_topics),
        sort_dim: args.collapse.sort_dim,
        knn_cells: args.collapse.knn_cells,
        num_levels: args.collapse.num_levels,
        iter_opt: args.collapse.iter_opt,
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
        refine: Some(args.collapse.pb_refine.to_params()),
        ignore_batch: args.collapse.ignore_batch,
        want_hierarchy: false,
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

    let mut parameters = candle_nn::VarMap::new();
    let param_builder =
        candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &dev);

    let dec_context_size = args.decoder_context_size.unwrap_or(args.context_size);

    // Value transform: the learned dual-binned intensity-embedding gate
    // (Anscombe retired). `n_value_bins` = the number of log1p-scale
    // value bins.
    let value_embedding = ValueEmbeddingConfig {
        n_value_bins: args.n_value_bins,
    };
    info!(
        "intensity-embedding value transform: n_value_bins={} (per-bin width H={})",
        value_embedding.n_value_bins, h
    );

    // Gene names — needed both for feature-graph resolution (below) and
    // for output artifacts further down.
    let gene_names = data_vec.row_names()?;

    // Feature-feature graph: parsed once and (when --feature-network was
    // supplied) plumbed into the indexed encoder's GAT block as additive
    // attention bias. Reuses the pre-mask graph cached by the row-mask
    // callback when --feature-network-restrict is set (one parse total).
    let cached_pre_mask = cached_graph.borrow_mut().take();
    let cached_pre_mask_keep = cached_keep.borrow_mut().take();
    let feature_graph: Option<FeaturePairGraph> = match (cached_pre_mask, cached_pre_mask_keep) {
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

    // Convert FeaturePairGraph → GraphCsr at the encoder's feature axis.
    // The CSR is shared (Arc) across every level's data loader and is
    // also persisted into the model artifact in safetensors form.
    let feature_graph_csr: Option<
        std::sync::Arc<candle_util::candle_indexed_data_loader::GraphCsr>,
    > = feature_graph.as_ref().map(|g| {
        let edges: Vec<(usize, usize)> = g.feature_edges.clone();
        std::sync::Arc::new(
            candle_util::candle_indexed_data_loader::GraphCsr::from_edges(
                n_features_full,
                &edges,
                None,
            ),
        )
    });
    if let Some(ref g) = feature_graph_csr {
        info!(
            "feature-network: {} undirected entries (directed CSR entries: {}); \
             GAT enabled on indexed encoder",
            g.n_directed_entries() / 2,
            g.n_directed_entries(),
        );
    }

    let base_encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
            value_embedding,
            use_gat: feature_graph_csr.is_some(),
        },
        &parameters,
        param_builder.pp("enc"),
    )?;

    // Per-level decoders: all at D_full, levels differ in N (sample coarsening).
    // ETM-factorized — each decoder shares the encoder's feature embeddings ρ,
    // and learns only its own topic embeddings α_{level} [K, H].
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
                model_type_expected: crate::topic::model_metadata::MODEL_TYPE_INDEXED,
                n_topics,
                n_features_full,
                n_features_encoder: n_features_full,
                encoder_hidden: &args.encoder_layers,
                level_decoder_dims: &vec![n_features_full; num_levels],
                embedding_dim: Some(h),
                value_embedding: Some(value_embedding.n_value_bins),
            },
        )?;
    }

    info!(
        "input: {} -> indexed encoder (emb={}, ctx={}) -> {} decoders (D={}, ctx={})",
        n_features_full, h, args.context_size, num_levels, n_features_full, dec_context_size,
    );

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

    info!("Computing NB-Fisher weights for shortlist scoring");
    let shortlist_weights: Vec<f32> =
        crate::empirical_dict::compute_nb_fisher_weights(&data_vec, args.block_size)?;

    // Per-gene mean expression rate `μ_d` from the finest-level pseudobulk
    // posterior. The encoder composes it with the per-cell batch null as a
    // multiplicative count-rate divisor before Anscombe — joint correction
    // for batch effect × gene-typical-rate, leaving the cell's biological
    // deviation. Stored as the raw mean; Anscombe is applied inside the
    // encoder via `anscombe_lite`.
    let feature_mean: Vec<f32> = {
        let mu = finest_collapsed.mu_observed.posterior_mean();
        let n_pb = mu.ncols().max(1) as f32;
        (0..n_features_full)
            .map(|d| mu.row(d).iter().sum::<f32>() / n_pb)
            .collect()
    };
    // Decoder Fisher loss weights are the same NB-Fisher per-gene quantity
    // already computed for shortlist selection.
    let feature_fisher_weights = shortlist_weights.clone();

    let train_config = IndexedTrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.epochs,
        minibatch_size: args.minibatch_size,
        learning_rate: args.learning_rate,
        topic_smoothing: args.topic_smoothing,
        enc_context_size: args.context_size,
        dec_context_size,
        stop: &stop,
        shortlist_weights: &shortlist_weights,
        feature_mean: &feature_mean,
        feature_fisher_weights: &feature_fisher_weights,
        grad_clip: args.grad_clip,
        feature_graph: feature_graph_csr.clone(),
    };

    let bulk_with_deltas: Option<(&Mat, &[GammaMatrix])> = match (&bulk_nd_full, &bulk_deltas) {
        (Some(full), Some(deltas)) => Some((full, deltas)),
        _ => None,
    };

    let scores = train_mixed(
        &collapsed_levels,
        &base_encoder,
        &decoders,
        &train_config,
        bulk_with_deltas,
    )?;

    info!("Writing down the model parameters");

    // Use finest-level decoder for output
    let finest_decoder = decoders.last().unwrap();
    write_indexed_dictionary(finest_decoder, &gene_names, &args.out)?;
    write_feature_embedding(base_encoder.feature_embeddings(), &gene_names, &args.out)?;

    // Persist trainable weights, model architecture, and shortlist weights
    // so `senna predict` (and `--init-from` re-runs) can rebuild this model.
    use crate::topic::model_metadata::{
        save_feature_graph_into_varmap, save_feature_mean, save_parameters, save_shortlist_weights,
        TopicModelMetadata,
    };

    // Bake the feature graph into the VarMap (non-trainable) before the
    // safetensors snapshot is written. Adam has already stopped, so adding
    // Vars at this point doesn't affect optimization.
    let n_graph_edges: Option<usize> = match feature_graph_csr.as_ref() {
        Some(g) => {
            save_feature_graph_into_varmap(&mut parameters, &dev, g)?;
            Some(g.col_idx.len())
        }
        None => None,
    };
    save_parameters(&parameters, &args.out)?;
    let mut metadata = TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_INDEXED.into(),
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
        enc_context_size: Some(args.context_size),
        dec_context_size: Some(dec_context_size),
        n_value_bins: Some(value_embedding.n_value_bins),
        theta_mean: None,
        n_graph_edges,
    };
    metadata.save(&args.out)?;
    save_shortlist_weights(&shortlist_weights, &gene_names, &args.out)?;
    save_feature_mean(&feature_mean, &gene_names, &args.out)?;

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
            embedding_dim: h,
            layers: &args.encoder_layers,
            value_embedding,
            use_gat: feature_graph_csr.is_some(),
        },
        &parameters,
        cpu_vb.pp("enc"),
    )?;
    let finest_dec_idx = decoders.len().saturating_sub(1);
    let cpu_finest_decoder = EmbeddedTopicDecoder::new(
        n_topics,
        cpu_encoder.feature_embeddings().clone(),
        cpu_vb.pp(format!("dec_{finest_dec_idx}")),
    )?;

    let refine_config = args.amort_refine.to_config();

    info!("Writing down the latent states");
    let eval_config = EvaluateLatentConfig {
        dev: &cpu_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        enc_context_size: args.context_size,
        dec_context_size,
        decoder: &cpu_finest_decoder,
        refine_config: refine_config.as_ref(),
        shortlist_weights: &shortlist_weights,
        feature_mean: &feature_mean,
        feature_fisher_weights: &feature_fisher_weights,
    };
    let z_nk = evaluate_latent_by_indexed_encoder(
        &data_vec,
        &cpu_encoder,
        finest_collapsed,
        &eval_config,
    )?;

    metadata.populate_theta_mean_and_save(&z_nk, &args.out)?;

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
            shortlist_weights: &shortlist_weights,
            feature_mean: &feature_mean,
            feature_fisher_weights: &feature_fisher_weights,
        };
        evaluate_bulk_samples(bulk, bulk_deltas, &cpu_encoder, &bulk_config)?;
    }

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    let cell_names = data_vec.column_names()?;

    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names)?;

    // pb_latent omitted: indexed encoder's PB-level forward pass isn't
    // wired here; annotate reconstructs θ_PB from pb_gene · β.
    {
        let pb_gene_gp: Mat = finest_collapsed.mu_observed.posterior_mean().clone();
        crate::output_helpers::save_pb_gene(&args.out, &pb_gene_gp, &gene_names)?;
    }

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
        kind: crate::run_manifest::RunKind::Itopic,
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: true,
        has_cell_proj: true,
        pb_gene_suffix: Some("pb_gene.parquet"),
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        default_colour_by: "cluster",
    })?;

    info!("Done");
    Ok(())
}
