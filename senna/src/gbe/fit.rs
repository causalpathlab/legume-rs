//! `senna gbe` entry point.

use crate::embed_common::*;
use crate::gbe::coarsen::{build_cell_coarsenings, CellCoarseningArgs};
use crate::gbe::data::load_unified_data;
use crate::gbe::eval::{save_outputs, OutputContext};
use crate::gbe::feature_network::FeatureNetworkSmoother;
use crate::gbe::loss::build_per_file_samplers;
use crate::gbe::model::{BiasInit, JointEmbedModel, ModelArgs};
use crate::gbe::training::{train, TrainingContext, TrainingParams};
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::gene_weighting::compute_nb_fisher_weights;
use matrix_util::pair_graph::FeaturePairGraph;

#[derive(Args, Debug)]
pub struct GbeArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Sparse count matrices (zarr/h5), comma-separated. Each \
                file contributes its rows to the unified feature axis; \
                cells unify by barcode across files."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, default_value_t = 64, help = "Embedding dimension H")]
    embedding_dim: usize,

    #[arg(long, default_value_t = 8, help = "Number of coarsening seeds")]
    num_coarsen_seeds: usize,

    #[arg(
        long,
        default_value_t = 200,
        help = "Target super-cell blocks (cell axis)"
    )]
    super_cells: usize,

    #[arg(long, default_value_t = 32, help = "Sketch dim for coarsening RP")]
    sketch_dim: usize,

    #[arg(long, default_value_t = 200, help = "Training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 100, help = "Batches per epoch")]
    batches_per_epoch: usize,

    #[arg(long, default_value_t = 1024, help = "Positive edges per batch")]
    batch_size: usize,

    #[arg(long, default_value_t = 16, help = "Negative samples per positive")]
    num_negatives: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "AdamW learning rate",
        alias = "lr"
    )]
    learning_rate: f64,

    #[arg(long, default_value_t = 1, help = "Random seed (base)")]
    seed: u64,

    #[arg(
        long,
        help = "Optional feature-feature edge list (TSV/CSV; e.g. \
                BioGRID, STRING, synthetic-lethality). Activates SGC \
                smoothing of E_feat through the K-hop normalized \
                adjacency."
    )]
    feature_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching when resolving feature-network names"
    )]
    feature_network_prefix_match: bool,

    #[arg(
        long,
        help = "Optional name-stripping delimiter for feature-network resolution \
                (e.g. '.' to match `TP53.1` → `TP53`)"
    )]
    feature_network_delim: Option<char>,

    #[arg(
        long,
        default_value_t = 2,
        help = "SGC propagation hops K (default 2 — useful for sparse synthetic-\
                lethality / regulatory networks). K=2 already reaches all \
                shared-neighbor pairs, so no separate SNN augmentation is needed."
    )]
    feature_network_k: usize,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "SGC neighbor-mix coefficient α ∈ [0, 1]; smaller = gentler nudge"
    )]
    feature_network_alpha: f32,

    #[arg(
        long,
        default_value_t = 5,
        help = "Re-propagate the frozen network residual every N epochs"
    )]
    feature_network_refresh: usize,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    device_no: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix; produces {out}.latent.parquet, \
                     {out}.dictionary.parquet, {out}.cell_bias.parquet, \
                     {out}.feature_bias.parquet, {out}.senna.json"
    )]
    out: Box<str>,
}

pub fn fit_gbe(args: &GbeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let dev = match args.device {
        ComputeDevice::Cpu => Device::Cpu,
        ComputeDevice::Cuda => Device::new_cuda(args.device_no)?,
        ComputeDevice::Metal => Device::new_metal(args.device_no)?,
    };

    let unified = load_unified_data(&args.data_files, args.batch_files.as_deref())?;

    let n_cells = unified.n_cells();
    let n_features = unified.n_features();

    info!(
        "Building cell coarsenings (K={} seeds, target ~{} super-cells)",
        args.num_coarsen_seeds, args.super_cells
    );

    let cell_axis = build_cell_coarsenings(CellCoarseningArgs {
        triplets: &unified.triplets,
        n_cells,
        n_features,
        target_blocks: args.super_cells,
        sketch_dim: args.sketch_dim,
        n_seeds: args.num_coarsen_seeds,
        base_seed: args.seed.wrapping_add(0xC347),
    })?;

    info!("Avg coarse blocks: cells {:.0}", cell_axis.avg_n_coarse());

    info!("Computing NB-Fisher weights per file...");
    let mut feat_weights: Vec<f32> = Vec::with_capacity(n_features);
    for (i, data) in unified.per_file_data.iter().enumerate() {
        let w = compute_nb_fisher_weights(data, None)?;
        info!(
            "  file {}: {} features, mean Fisher weight {:.3}",
            i,
            w.len(),
            w.iter().sum::<f32>() / w.len().max(1) as f32
        );
        feat_weights.extend(w);
    }

    let zeros_feat = vec![0f32; n_features];
    let zeros_cell = vec![0f32; n_cells];

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_util::candle_core::DType::F32, &dev);
    let model = JointEmbedModel::new(
        ModelArgs {
            n_features,
            n_cells,
            embedding_dim: args.embedding_dim,
        },
        &BiasInit {
            b_feat: &zeros_feat,
            b_cell: &zeros_cell,
        },
        &varmap,
        vs,
        &dev,
    )?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        },
    )?;

    info!("Building per-file edge samplers...");
    let alpha_neg = 0.75_f32;
    let file_samplers = build_per_file_samplers(
        &unified.triplets,
        &unified.file_triplet_ranges,
        &unified.file_feature_ranges,
        &feat_weights,
        alpha_neg,
    );

    let mut smoother = if let Some(path) = args.feature_network.as_deref() {
        info!("Loading feature network from {}...", path);
        let graph = FeaturePairGraph::from_edge_list(
            path,
            unified.feature_names.to_vec(),
            args.feature_network_prefix_match,
            args.feature_network_delim,
        )?;
        if graph.num_edges() == 0 {
            anyhow::bail!(
                "Feature network has 0 matched edges — check name resolution \
                 (--feature-network-delim / --feature-network-prefix-match)."
            );
        }
        info!(
            "SGC smoothing: K={}, α={}, refresh={} epochs over {} edges",
            args.feature_network_k,
            args.feature_network_alpha,
            args.feature_network_refresh,
            graph.num_edges()
        );
        Some(FeatureNetworkSmoother::new(
            &graph,
            n_features,
            args.embedding_dim,
            args.feature_network_alpha,
            args.feature_network_k,
            args.feature_network_refresh,
        )?)
    } else {
        None
    };

    let train_ctx = TrainingContext {
        unified: &unified,
        cell_axis: &cell_axis,
        feat_weights: &feat_weights,
        file_samplers: &file_samplers,
        dev: &dev,
    };
    let train_params = TrainingParams {
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch_size: args.batch_size,
        num_negatives: args.num_negatives,
        seed: args.seed,
    };
    train(
        &model,
        &mut opt,
        &train_ctx,
        &train_params,
        smoother.as_mut(),
    )?;

    save_outputs(
        &model,
        &OutputContext {
            feature_names: &unified.feature_names,
            barcodes: &unified.barcodes,
        },
        &args.out,
    )?;

    let input: Vec<String> = args.data_files.iter().map(|s| s.to_string()).collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::Gbe,
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        default_colour_by: "cluster",
    })?;

    info!(
        "Done — outputs at {}.{{latent,dictionary,*_bias}}.parquet",
        args.out
    );

    Ok(())
}
