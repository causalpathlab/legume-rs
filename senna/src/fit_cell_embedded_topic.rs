//! `senna cell-embedded-topic` — hierarchical cell→PB pooling topic model.
//!
//! Same model *type* as `senna indexed-topic` (shared ρ `[D, H]` table,
//! ETM-factorized decoder, multi-level PB training). The difference is
//! structural and lives in the encoder + loader: a PB sample is treated
//! as a *pool of cells*, and that pooling is moved **into the encoder**
//! as a two-level gene→cell→PB `EmbeddingBag`.
//!
//! A PB at a coarse level can pool hundreds of cells, so the FG pool draws
//! an `S = min(--fg-cells-per-pb, |PB|)`-cell random subsample per PB per
//! minibatch (rescaled by `|PB|/S` to stay an unbiased estimate of the
//! full-PB sum). That bounds the per-minibatch member-cell count `M` — so
//! encoder memory stays flat at coarse PB levels — and the fresh
//! per-epoch subsample is the within-PB SGD stochasticity.
//!
//! Trains, runs per-cell latent inference, and writes the full
//! topic-family artifact set (latent / pb_gene / cell_proj / manifest /
//! CNV); `senna predict` applies the trained model to held-out data.

use crate::embed_common::*;
use crate::fit_indexed_topic::FeatureNameKindArg;
use crate::topic::common::{
    create_device, load_and_collapse, move_varmap_to_cpu, setup_stop_handler, LoadCollapseArgs,
    PreparedData,
};
use crate::topic::eval_cell_embedded::{
    evaluate_latent_by_cell_embedded_encoder, EvaluateCellLatentConfig,
};
use crate::topic::model_metadata::MODEL_TYPE_CELL_EMBEDDED;
use crate::topic::train_cell_embedded::{
    extract_cell_samples, train_mixed_cell, CellEmbeddedTrainConfig,
};
use crate::topic::train_indexed::{write_feature_embedding, write_indexed_dictionary};

use candle_util::decoder::EmbeddedTopicDecoder;
use candle_util::encoder::{CellEmbeddedEncoder, CellEmbeddedEncoderArgs};
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct CellEmbeddedTopicArgs {
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
                     manifests are rejected (no feature embedding to inherit)."
    )]
    from: Option<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Prefix for generated files:\n  \
                     {out}.dictionary.parquet       gene × topic loadings (log-prob)\n  \
                     {out}.feature_embedding.parquet ρ at full gene resolution\n  \
                     {out}.latent.parquet           cell × topic log-softmax proportions\n  \
                     {out}.pb_gene.parquet          pseudobulk × gene posterior mean\n  \
                     {out}.log_likelihood.parquet   training loss trace\n  \
                     {out}.safetensors              encoder+decoder weights\n  \
                     {out}.model.json               model architecture metadata\n  \
                     {out}.cell_proj.parquet        cached random projection (consumed by `senna layout`)\n  \
                     {out}.senna.json               run manifest consumed by `senna layout --from` and `senna plot --from`\n  \
                     {out}.shortlist_weights.parquet NB-Fisher per-gene weights (consumed by `senna predict`)"
    )]
    out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files, one per data file"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    collapse: crate::refine_weighting::CollapseArgs,

    #[arg(
        long = "init-from",
        help = "Initialize encoder + decoder weights from a previously trained \
                cell-embedded-topic model (matching {prefix}.model.json + \
                {prefix}.safetensors)."
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
        long_help = "Example: 128,1024,128. The [fg, bg] pool concat (2H) feeds\n\
                     the first layer; final layer → topics."
    )]
    encoder_layers: Vec<usize>,

    #[arg(long, short = 'i', default_value_t = 1000, help = "Training epochs")]
    epochs: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Training minibatch size (pseudobulk samples)"
    )]
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
        long_help = "Each member cell keeps its top-K features by value — the\n\
                     per-cell sparse atom the FG pool sums over."
    )]
    fg_context_size: usize,

    #[arg(
        long,
        default_value_t = 16,
        help = "Member cells sampled per PB per minibatch (0 = use all)",
        long_help = "Each minibatch draws an S = min(cap, |PB|)-cell random\n\
                     subsample of every pseudobulk's member cells; the FG pool\n\
                     is then rescaled by |PB|/S to stay an unbiased estimate of\n\
                     the full-PB sum. This caps encoder memory and keeps the\n\
                     touched-ρ row set sparse at coarse PB levels, where a\n\
                     single PB can pool hundreds of cells. A fresh subsample is\n\
                     drawn every epoch — that resampling is the within-PB SGD\n\
                     stochasticity. 0 disables the cap (pool all members)."
    )]
    fg_cells_per_pb: usize,

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

    #[command(flatten)]
    amort_refine: crate::refine_weighting::AmortRefineArgs,

    #[arg(long, help = "Decoder context window (default: --fg-context-size)")]
    decoder_context_size: Option<usize>,

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
        default_value_t = 1,
        help = "Shared-neighbor edge QC threshold. Default 1 drops edges \
                with zero corroboration. Set 0 to keep every parsed edge."
    )]
    feature_network_min_shared_neighbors: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-node degree cap (0 = off). Ranks neighbors by \
                shared-neighbor count; union-symmetric."
    )]
    feature_network_max_degree: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Iterative k-core pruning threshold (0 = off)."
    )]
    feature_network_min_degree: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable feature-network feature restriction. By default, \
                when --feature-network is supplied, features with zero edges \
                after the QC pipeline are dropped from the data axis."
    )]
    no_feature_network_restrict: bool,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Per-name canonicalization across input backends"
    )]
    feature_name_kind: FeatureNameKindArg,

    #[arg(
        long,
        help = "Reuse a pre-trained per-gene embedding ρ from a prior senna run. \
                Loads `{prefix}.feature_embedding.parquet` (topic / cell-embedded-\
                topic) or `{prefix}.dictionary.parquet` (gbe). Strict-intersects \
                gene names under `--feature-name-kind`; unmatched genes are \
                dropped from training. Encoder/decoder ρ stays fixed; the cell- \
                embedded FC stack, α, and value-embedding train as usual. \
                Incompatible with `--feature-network`."
    )]
    freeze_feature_embedding: Option<Box<str>>,

    #[command(flatten)]
    cnv: CnvArgs,
}

pub fn fit_cell_embedded_topic_model(args: &CellEmbeddedTopicArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let k = args.n_latent_topics;

    // --from chain-inheritance: explicit CLI wins; otherwise inherit
    // data/batch/ρ-prefix from the source manifest (default-freeze).
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
    let freeze_feature_embedding: Option<Box<str>> =
        args.freeze_feature_embedding.clone().or_else(|| {
            inherited
                .as_ref()
                .map(|i| i.feature_embedding_prefix.clone())
        });

    // Resolved before H so the pre-trained dictionary's column count
    // pins the encoder dim.
    let frozen_spec: Option<crate::topic::freeze::FrozenFeatureSpec> =
        match freeze_feature_embedding.as_deref() {
            None => None,
            Some(prefix) => {
                anyhow::ensure!(
                    args.feature_network.is_none(),
                    "--freeze-feature-embedding is incompatible with --feature-network"
                );
                let kind = args.feature_name_kind.resolve_or_gene();
                Some(crate::topic::freeze::FrozenFeatureSpec::resolve_from_prefix(prefix, kind)?)
            }
        };

    let pretrained_h: Option<usize> = frozen_spec.as_ref().map(|s| s.dictionary_h()).transpose()?;
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
    let freeze_mask_holder = frozen_spec.as_ref().map(|s| s.mask_fn());
    let feature_mask_fn: Option<&crate::topic::common::FeatureMaskFn> = match (
        freeze_mask_holder.as_deref(),
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
        cell_to_pb_per_level,
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
        want_hierarchy: true,
    })?;

    let cell_to_pb_per_level = cell_to_pb_per_level
        .ok_or_else(|| anyhow::anyhow!("load_and_collapse did not return the cell→pb hierarchy"))?;

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
    // BG PB residual) concatenated into the latent head. Value transform
    // is the fixed Anscombe scalar.
    let base_encoder = CellEmbeddedEncoder::new(
        CellEmbeddedEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
        },
        &parameters,
        param_builder.pp("enc"),
    )?;

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

    let gene_names_for_freeze = data_vec.row_names()?;
    if let Some(spec) = frozen_spec.as_ref() {
        anyhow::ensure!(
            args.init_from.is_none(),
            "--freeze-feature-embedding is incompatible with --init-from"
        );
        let host = spec.materialize(&gene_names_for_freeze)?;
        anyhow::ensure!(
            host.h == h,
            "frozen feature embedding has H={} but --embedding-dim={}",
            host.h,
            h
        );
        anyhow::ensure!(
            host.e_feat.nrows() == n_features_full,
            "frozen ρ has {} rows but the post-mask gene axis has {} (bug)",
            host.e_feat.nrows(),
            n_features_full
        );
        candle_util::frozen_features::overwrite_var_2d(
            &parameters,
            "enc.feature.embeddings",
            &host.e_feat,
            &dev,
        )?;
        info!(
            "Freeze mode: ρ seeded from {} (D={}, H={}); FC/α/value-embedding train, ρ stays fixed",
            spec.dictionary_path, n_features_full, h
        );
    }

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

    // Parsed for downstream restrict-mode consumption; cell-embedded
    // doesn't yet wire the graph into training (GCN integration deferred).
    let _feature_graph: Option<matrix_util::pair_graph::FeaturePairGraph> = feature_network
        .into_feature_graph(args.feature_network.as_deref(), &gene_names, net_opts)?;

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
        fg_cells_per_pb: args.fg_cells_per_pb,
        stop: &stop,
        shortlist_weights: &shortlist_weights,
        feature_fisher_weights: &feature_fisher_weights,
        grad_clip: args.grad_clip,
        frozen_feature_var: frozen_spec.as_ref().map(|_| "enc.feature.embeddings"),
    };

    let scores = train_mixed_cell(
        &collapsed_levels,
        &cell_to_pb_per_level,
        cell_samples,
        cell_size_factor,
        &base_encoder,
        &decoders,
        &train_config,
    )?;

    info!("Writing down the model parameters");

    let finest_decoder = decoders.last().unwrap();
    write_indexed_dictionary(finest_decoder, &gene_names, &args.out)?;
    write_feature_embedding(base_encoder.feature_embeddings(), &gene_names, &args.out)?;

    use crate::topic::model_metadata::{
        save_parameters, save_shortlist_weights, TopicModelMetadata,
    };
    save_parameters(&parameters, &args.out)?;
    let mut metadata = TopicModelMetadata {
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
        theta_mean: None,
        n_graph_edges: None,
    };
    metadata.save(&args.out)?;
    save_shortlist_weights(&shortlist_weights, &gene_names, &args.out)?;

    scores.to_parquet(&format!("{}.log_likelihood.parquet", &args.out))?;

    // Move the VarMap to CPU, then rebuild the encoder + finest-level
    // decoder from the CPU Vars for multi-threaded per-cell inference.
    // The training structs still hold CUDA/Metal tensors and must not be
    // reused on CPU block indices.
    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    move_varmap_to_cpu(&parameters)?;

    let cpu_vb = candle_nn::VarBuilder::from_varmap(&parameters, candle_core::DType::F32, &cpu_dev);
    let cpu_encoder = CellEmbeddedEncoder::new(
        CellEmbeddedEncoderArgs {
            n_features: n_features_full,
            n_topics,
            embedding_dim: h,
            layers: &args.encoder_layers,
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

    let finest_collapsed: &CollapsedOut = collapsed_levels.last().unwrap();
    let refine_config = args.amort_refine.to_config();

    info!("Writing down the latent states");
    let eval_config = EvaluateCellLatentConfig {
        dev: &cpu_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        fg_context_size,
        bg_context_size,
        dec_context_size,
        decoder: &cpu_finest_decoder,
        refine_config: refine_config.as_ref(),
        shortlist_weights: &shortlist_weights,
        feature_fisher_weights: &feature_fisher_weights,
    };
    let z_nk = evaluate_latent_by_cell_embedded_encoder(
        &data_vec,
        &cpu_encoder,
        finest_collapsed,
        &eval_config,
    )?;

    // Re-save metadata with θ̄_train populated — initial δ guess for any
    // downstream re-embedding, better than uniform 1/K.
    metadata.populate_theta_mean_and_save(&z_nk, &args.out)?;

    let cell_names = data_vec.column_names()?;
    crate::output_helpers::save_latent(&args.out, &z_nk, &cell_names)?;

    // pb_latent omitted: the cell-embedded encoder's PB-level forward pass
    // isn't wired here; `annotate` reconstructs θ_PB from pb_gene · β.
    {
        let pb_gene_gp: Mat = finest_collapsed.mu_observed.posterior_mean().clone();
        crate::output_helpers::save_pb_gene(&args.out, &pb_gene_gp, &gene_names)?;
    }

    // CNV detection using topic proportions as cell-type membership.
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
        kind: crate::run_manifest::RunKind::CellEmbeddedTopic,
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
