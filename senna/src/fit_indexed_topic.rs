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

use candle_util::decoder::EmbeddedTopicDecoder;
use candle_util::encoder::*;
use matrix_param::dmatrix_gamma::GammaMatrix;

#[derive(Args, Debug)]
pub struct IndexedTopicArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files (.zarr or .h5; optional when --from is given)",
        long_help = "Sparse backends produced by `data-beans from-mtx`.\n\
                     Multiple files may be passed (comma- or space-separated)\n\
                     and are concatenated column-wise on a shared feature set.\n\
                     When `--from <run.senna.json>` is provided and this list\n\
                     is empty, the data paths come from the source manifest."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Chain inputs + ρ warm-start from a prior `senna bge / fne / topic` \
                run's manifest",
        long_help = "Read a `{run}.senna.json` manifest and pre-fill `data_files`, \
                     `--batch-files`, and `--freeze-feature-embedding` from it. \
                     Explicit CLI flags override the manifest values. SVD-family \
                     manifests are rejected (no feature embedding to inherit). \
                     Typical use: bge → itopic warm-start.\n\
                     \n\
                     senna bge   data.zarr.zip -b batch.gz -o run-bge ...\n\
                     senna itopic --from run-bge.senna.json -o run-topic ..."
    )]
    from: Option<Box<str>>,

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
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off; typical 0.5–5.0)"
    )]
    grad_clip: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 penalty λ on the feature embedding matrix ρ ∈ ℝ^{D×H}: \
                adds λ · mean(ρ²) to the per-minibatch loss (mean-normalized, \
                so λ stays scale-invariant across D·H). Shrinks β dynamic \
                range (β = log_softmax(α·ρᵀ)) and can speed ETM convergence \
                on high-D gene sets. Default 1.0 (mild shrinkage). Set 0.0 \
                to disable. Typical: 0.1–10.0."
    )]
    feature_embedding_l2: f32,

    #[arg(
        long,
        help = "Reuse a pre-trained per-gene embedding ρ from a prior senna run. \
                Loads `{prefix}.feature_embedding.parquet` (topic / cell-embedded-\
                topic layout) or `{prefix}.dictionary.parquet` (gbe layout). \
                Gene names are strict-intersected against this dataset's gene \
                axis under the `--feature-name-kind` rule; unmatched genes are \
                dropped from training. The encoder/decoder ρ stays fixed; \
                everything else (α, FC, BN, GCN, value-embedding, decoder \
                topic embeddings) trains as usual. Incompatible with \
                `--feature-network` (SGC nudge has no learnable target); \
                forces `--feature-embedding-l2 0` (frozen ρ doesn't need \
                shrinkage)."
    )]
    freeze_feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        conflicts_with = "freeze_feature_embedding",
        help = "Warm-start ρ from a prior senna run (typically `senna bge`). \
                Same layout resolution as `--freeze-feature-embedding` \
                (gbe `{prefix}.dictionary.parquet` or topic \
                `{prefix}.feature_embedding.parquet`), same strict gene-name \
                intersection. The difference is that AdamW continues to \
                update ρ during training — this just gives a biology-aware \
                starting point instead of random Kaiming-normal init. \
                Pairs well with `senna bge` pre-training: cheap NCE-based \
                gene embedding that's robust to batch effects, used as the \
                warm-start here. Mutually exclusive with \
                `--freeze-feature-embedding`."
    )]
    init_feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cross-entropy penalty λ on β toward the anchor prior — \
                anchors topic indices to Gram-Schmidt anchor gene sets \
                derived from the finest-level pseudobulks. Breaks the \
                K-way permutation symmetry of the ETM-factorized β and \
                is the load-bearing anti-mode-collapse force for this \
                model. 0 disables; default 1.0."
    )]
    anchor_penalty: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay applied uniformly to every \
                parameter (encoder ρ + α + FC + BN + GCN γ). \
                Per-step post-update shrinkage; doesn't enter the backward \
                graph. Default 0.0 (off, i.e. plain Adam despite the name). \
                Typical: 1e-5 to 1e-4."
    )]
    weight_decay: f32,

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
                γ-gated GCN diffusion block into the indexed encoder. \
                Per-cell sub-adjacency on the measured top-K (pre- \
                normalised at cache build) lets the encoder pool signal \
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
        default_value_t = 1,
        help = "Shared-neighbor edge QC: drop edges (u,v) where the \
                endpoints share fewer than N neighbors in the feature \
                network. Default 1 drops edges with zero corroboration \
                — standard PPI/topological-overlap denoising. Set 0 to \
                keep every parsed edge."
    )]
    feature_network_min_shared_neighbors: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-node degree cap on the feature network (0 = off). \
                After shared-neighbor QC, for each feature with degree \
                > N, rank its neighbors by shared-neighbor count and \
                keep the top N (union-symmetric: an edge survives iff \
                either endpoint kept it). Caps PPI hubs whose degree \
                would otherwise blow up per-cell sub-adjacency."
    )]
    feature_network_max_degree: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Iterative k-core pruning threshold on the feature \
                network (default 0 = off). Drops every feature whose \
                degree falls below N until the surviving subgraph is \
                N-degenerate."
    )]
    feature_network_min_degree: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable feature-network feature restriction. By default \
                when --feature-network is supplied, features with zero \
                edges after the QC pipeline (shared-neighbor prune → \
                hub cap → k-core) are dropped from the data axis before \
                projection, collapse, and training. Pass to keep the \
                full feature axis (graph diffusion as a soft prior)."
    )]
    no_feature_network_restrict: bool,

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

pub fn fit_indexed_topic_model(args: &IndexedTopicArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let k = args.n_latent_topics;

    // --from chain-inheritance: explicit CLI flags win per-field; with no
    // explicit init/freeze flag, default to freezing the inherited ρ.
    let inherited = args
        .from
        .as_deref()
        .map(crate::run_manifest::inherit_from)
        .transpose()?;
    if let Some(inh) = inherited.as_ref() {
        info!(
            "--from: inheriting data + batch + ρ-prefix from a '{}' manifest",
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

    let init_feature_embedding: Option<Box<str>> = args.init_feature_embedding.clone();
    let freeze_feature_embedding: Option<Box<str>> =
        args.freeze_feature_embedding.clone().or_else(|| {
            match (init_feature_embedding.is_none(), inherited.as_ref()) {
                (true, Some(inh)) => Some(inh.feature_embedding_prefix.clone()),
                _ => None,
            }
        });
    let pretrained_prefix = freeze_feature_embedding
        .as_deref()
        .or(init_feature_embedding.as_deref());
    let freeze_rho = freeze_feature_embedding.is_some();
    // Resolved before H so the pre-trained dictionary's column count
    // pins the encoder dim.
    let pretrained_spec: Option<crate::topic::freeze::FrozenFeatureSpec> = match pretrained_prefix {
        None => None,
        Some(prefix) => {
            if freeze_rho {
                anyhow::ensure!(
                    args.feature_network.is_none(),
                    "--freeze-feature-embedding is incompatible with --feature-network \
                     (SGC γ-gated graph diffusion has no learnable target when ρ is frozen)"
                );
            }
            let kind: Option<auxiliary_data::feature_names::FeatureNameKind> =
                args.feature_name_kind.clone().into();
            // Pre-train inputs are gene-keyed; row names aren't available yet.
            let kind =
                kind.unwrap_or(auxiliary_data::feature_names::FeatureNameKind::Gene { delim: '_' });
            Some(crate::topic::freeze::FrozenFeatureSpec::resolve_from_prefix(prefix, kind)?)
        }
    };

    let pretrained_h: Option<usize> = pretrained_spec
        .as_ref()
        .map(|s| s.dictionary_h())
        .transpose()?;
    let h = crate::topic::common::resolve_embedding_dim(args.embedding_dim, pretrained_h, k)?;

    let (effective_multiome, effective_hvg_n, effective_hvg_list) =
        crate::hvg::resolve_multiome_with_hvg(args.multiome, data_files.len(), &args.hvg);

    let net_opts = crate::topic::common::FeatureNetworkOpts {
        prefix_match: args.feature_network_prefix_match,
        delim: args.feature_network_delim,
        min_shared_neighbors: args.feature_network_min_shared_neighbors,
        max_degree: args.feature_network_max_degree,
        min_degree: args.feature_network_min_degree,
    };
    let restrict_path = if args.no_feature_network_restrict {
        None
    } else {
        args.feature_network.as_deref()
    };

    let feature_network = crate::topic::common::setup_feature_network(restrict_path, net_opts);
    let pretrained_mask_holder = pretrained_spec.as_ref().map(|s| s.mask_fn());
    let feature_mask_fn: Option<&crate::topic::common::FeatureMaskFn> = match (
        pretrained_mask_holder.as_deref(),
        feature_network.mask_fn.as_deref(),
    ) {
        (Some(fm), _) => Some(fm),
        (None, Some(nm)) => Some(nm),
        (None, None) => None,
    };

    let PreparedData {
        data_vec,
        collapsed_levels,
        proj_kn,
        cell_to_pb_per_level: _,
    } = load_and_collapse(&LoadCollapseArgs {
        data_files: &data_files,
        batch_files: &batch_files,
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

    // Gene names — needed both for feature-graph resolution (below) and
    // for output artifacts further down.
    let gene_names = data_vec.row_names()?;

    let feature_graph: Option<matrix_util::pair_graph::FeaturePairGraph> = feature_network
        .into_feature_graph(args.feature_network.as_deref(), &gene_names, net_opts)?;

    // Convert FeaturePairGraph → GraphCsr at the encoder's feature axis.
    // The CSR is shared (Arc) across every level's data loader and is
    // also persisted into the model artifact in safetensors form.
    let feature_graph_csr: Option<std::sync::Arc<candle_util::data::GraphCsr>> =
        feature_graph.as_ref().map(|g| {
            let edges: Vec<(usize, usize)> = g.feature_edges.clone();
            std::sync::Arc::new(candle_util::data::GraphCsr::from_edges(
                n_features_full,
                &edges,
                None,
            ))
        });
    if let Some(ref g) = feature_graph_csr {
        info!(
            "feature-network: {} undirected entries (directed CSR entries: {}); \
             GCN block enabled on indexed encoder",
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
            use_gcn: feature_graph_csr.is_some(),
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

    // Overwrite ρ in place with the pre-trained values BEFORE warm-start
    // from a prior topic checkpoint. The encoder/decoder both hold a
    // Tensor reference into the same Var, so a single `var.set(...)`
    // updates everywhere. The Var stays in the VarMap (round-trips
    // through safetensors); for freeze mode the optimizer excludes it
    // via `trainable_vars` (see `train_indexed.rs`), for init mode it
    // keeps updating.
    if let Some(spec) = pretrained_spec.as_ref() {
        anyhow::ensure!(
            args.init_from.is_none(),
            "ρ pre-training is incompatible with --init-from \
             (warm-start would overwrite the pre-trained ρ from a different checkpoint)"
        );
        let host = spec.materialize(&gene_names)?;
        anyhow::ensure!(
            host.h == h,
            "pre-trained feature embedding has H={} but --embedding-dim={}",
            host.h,
            h
        );
        anyhow::ensure!(
            host.e_feat.nrows() == n_features_full,
            "pre-trained ρ has {} rows but the post-mask gene axis has {} — \
             loader and feature-mask disagree (this is a bug, please report)",
            host.e_feat.nrows(),
            n_features_full
        );
        candle_util::frozen_features::overwrite_var_2d(
            &parameters,
            "enc.feature.embeddings",
            &host.e_feat,
            &dev,
        )?;
        if freeze_rho {
            info!(
                "Freeze mode: ρ seeded from {} (D={}, H={}); encoder/decoders share frozen ρ, \
                 only α + FC + BN (+ GCN γ) train",
                spec.dictionary_path, n_features_full, h
            );
        } else {
            info!(
                "Warm-start: ρ initialised from {} (D={}, H={}); AdamW continues to update it \
                 alongside α + FC + BN (+ GCN γ)",
                spec.dictionary_path, n_features_full, h
            );
        }
    }

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

    // Anchor-prior CE penalty: Gram-Schmidt anchors selected on the
    // finest-level pseudobulks give each topic k a per-gene prior simplex
    // (rows of `[K, D]`), which we cross-entropy against
    // `β_kd = log_softmax_d(α · ρᵀ)` at every minibatch. This is the
    // only K-way symmetry-breaking force on the ETM-factorized β —
    // without it the model collapses onto one or two dominant topics.
    let anchor_tensors: Option<Vec<candle_core::Tensor>> = if args.anchor_penalty > 0.0 {
        info!("Building anchor prior (K={n_topics}) from finest pseudobulks");
        let anchor_prior = crate::topic::anchor_prior::AnchorPrior::from_pseudobulk(
            finest_collapsed,
            n_topics,
            None,
        )?;
        // Indexed decoders all run at D_full — no per-level feature coarsening.
        let level_coarsenings_none: Vec<
            Option<data_beans_alg::feature_coarsening::FeatureCoarsening>,
        > = (0..num_levels).map(|_| None).collect();
        Some(anchor_prior.per_level_device_tensors(&level_coarsenings_none, &dev)?)
    } else {
        None
    };

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
        feature_embedding_l2: args.feature_embedding_l2,
        weight_decay: args.weight_decay,
        frozen_feature_var: if freeze_rho {
            Some("enc.feature.embeddings")
        } else {
            None
        },
        anchor_prior_per_level: anchor_tensors.as_deref(),
        anchor_penalty: args.anchor_penalty,
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
            use_gcn: feature_graph_csr.is_some(),
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

    let input: Vec<String> = data_files.iter().map(|s| s.to_string()).collect();
    let batch: Vec<String> = batch_files
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
        has_latent: true,
    })?;

    info!("Done");
    Ok(())
}
