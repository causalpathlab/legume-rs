//! `senna gbe` — thin clap + run-manifest wrapper around the
//! `graph-embedding-util` engine.
//!
//! All algorithmic work lives in `graph_embedding_util`. This file
//! exists only to translate `GbeArgs` → `FitConfig`, resolve the
//! optional feature-network edge file against the unified feature
//! axis, and write senna's run manifest after training.

use crate::embed_common::*;
use graph_embedding_util as ge;

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

    let unified = ge::load_unified_data(&args.data_files, args.batch_files.as_deref())?;

    let feature_network = args
        .feature_network
        .as_deref()
        .map(|path| {
            ge::load_feature_network(ge::FeatureNetworkArgs {
                path,
                feature_names: &unified.feature_names,
                prefix_match: args.feature_network_prefix_match,
                delim: args.feature_network_delim,
                k_hops: args.feature_network_k,
                alpha: args.feature_network_alpha,
                refresh_epochs: args.feature_network_refresh,
            })
        })
        .transpose()?;

    let config = ge::FitConfig {
        embedding_dim: args.embedding_dim,
        num_coarsen_seeds: args.num_coarsen_seeds,
        super_cells: args.super_cells,
        sketch_dim: args.sketch_dim,
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch_size: args.batch_size,
        num_negatives: args.num_negatives,
        learning_rate: args.learning_rate,
        seed: args.seed,
        device: args.device.to_device(args.device_no)?,
        feature_network,
        stop: None,
    };

    let out = ge::fit(&unified, config)?;

    ge::save_outputs(
        &out.model,
        &ge::OutputContext {
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
