//! `senna gbe` — thin clap + run-manifest wrapper around the
//! `graph-embedding-util` engine.
//!
//! All algorithmic work lives in `graph_embedding_util`. This file
//! exists only to translate `GbeArgs` → `FitConfig`, resolve the
//! optional feature-network edge file against the unified feature
//! axis, and write senna's run manifest after training.

use crate::embed_common::*;
use data_beans_alg::hvg::{select_hvg_streaming, HvgCliArgs};
use graph_embedding_util as ge;
use rustc_hash::FxHashMap;
use std::io::BufRead;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub(crate) enum CompositeModeArg {
    /// Per step, sample a coordinated bottom-up chain — one real
    /// (cell, feature) edge whose pb ancestors at each level come from
    /// the cell→pb_per_level map. All axes share the same positive
    /// feature and negatives per chain. Lowest variance per step;
    /// per-step compute close to `sum`. Default.
    #[default]
    Chain,
    /// Per step, sum NCE losses across every axis. Higher variance and
    /// O(n_axes) per-step cost than `chain`, but axes draw independent
    /// minibatches (no shared-positive correlation).
    Sum,
    /// Per step, pick one axis weighted by λ. Same expected gradient as
    /// `sum`, ~n_axes× faster epochs, higher per-step variance — needs
    /// proportionally more epochs to reach the same loss.
    Sample,
}

impl From<CompositeModeArg> for ge::CompositeMode {
    fn from(value: CompositeModeArg) -> Self {
        match value {
            CompositeModeArg::Sum => ge::CompositeMode::Sum,
            CompositeModeArg::Sample => ge::CompositeMode::Sample,
            CompositeModeArg::Chain => ge::CompositeMode::Chain,
        }
    }
}

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
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[command(flatten)]
    hvg: HvgCliArgs,

    #[arg(long, default_value_t = 16, help = "Embedding dimension H")]
    embedding_dim: usize,

    #[arg(long, default_value_t = 8, help = "Number of coarsening seeds")]
    num_coarsen_seeds: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Cap on the number of genes trained (0 = keep all). When > 0 and \
                less than the feature axis, keeps the top-N genes by NB-Fisher \
                weight and drops the rest before the multilevel collapse. Shrinks \
                E_feat, triplets, and per-batch samplers proportionally — the main \
                large-data speed knob."
    )]
    max_features: usize,

    #[arg(
        long = "composite-mode",
        value_enum,
        default_value_t = CompositeModeArg::Chain,
        help = "How to mix per-axis NCE losses each step. `chain` (default) samples a \
                coordinated bottom-up chain (one real cell-feature edge whose pb ancestors \
                at each level come from the cell→pb_per_level map). All axes share the \
                same positive feature and negatives per chain — lowest variance per step. \
                `sum` runs every axis with independent minibatches per step. `sample` picks \
                one axis per step weighted by λ (~n_axes× faster epochs, more epochs needed \
                to converge)."
    )]
    composite_mode: CompositeModeArg,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable BBKNN + DC-Poisson refinement of the multi-level pseudobulk \
                partition. Default: enabled (parity with senna topic / svd)."
    )]
    no_refine: bool,

    #[arg(long, default_value_t = 20, help = "Gibbs sweeps per refinement level")]
    refine_gibbs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Greedy sweeps per refinement level"
    )]
    refine_greedy: usize,

    #[arg(
        long = "refine-weighting",
        value_enum,
        default_value_t = crate::refine_weighting::WeightingArg::NbFisherInfo,
        help = crate::refine_weighting::WEIGHTING_HELP,
    )]
    refine_weighting: crate::refine_weighting::WeightingArg,

    #[arg(long, default_value_t = 42, help = "Seed for refinement Gibbs sampler")]
    refine_seed: u64,

    #[arg(
        long,
        default_value_t = 200,
        help = "Target super-cell blocks (cell axis)"
    )]
    super_cells: usize,

    #[arg(long, default_value_t = 10, help = "Sketch dim for coarsening RP")]
    sketch_dim: usize,

    #[arg(short = 'i', long, default_value_t = 200, help = "Training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 100, help = "Batches per epoch")]
    batches_per_epoch: usize,

    #[arg(long, default_value_t = 1024, help = "Positive edges per batch")]
    batch_size: usize,

    #[arg(long, default_value_t = 4, help = "Negative samples per positive")]
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

    #[arg(
        long,
        default_value_t = '_',
        help = "Delimiter for fuzzy gene-name matching across input files. The last \
                token after splitting on this char is used as the canonical row name, \
                so `ENSG00000000003_TSPAN6` (file A) and `TSPAN6` (file B) merge into \
                a single row. Pass an empty string (currently unsupported by clap; \
                set --feature-name-kind to override) to fall back to exact matching."
    )]
    feature_name_delim: char,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable fuzzy gene-name matching (use exact row-name match across files)"
    )]
    feature_name_exact: bool,

    #[arg(
        long,
        help = "Optional cell-cell edge list (whitespace-separated, two cell-barcode \
                columns per line; lines starting with `#` are ignored). When provided, \
                activates the cell-cell NCE term — positives are these edges, negatives \
                are within-batch random non-neighbor cells. Use a precomputed graph \
                (e.g. pinto's spatial KNN, exported to TSV)."
    )]
    cell_cell_edges: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cell-cell loss weight λ; final loss = L_bipartite + λ · L_cell_cell. \
                Default 1.0 weights cell-cell equal to cell-feature. Set to 0 to disable. \
                Ignored if --cell-cell-edges is not provided."
    )]
    cell_cell_lambda: f32,

    #[arg(
        long,
        default_value_t = 4,
        help = "Negative cells per positive cell-pair (cell-cell loss only)"
    )]
    cell_cell_negatives: usize,

    #[arg(
        long,
        help = "Cells per parallel block for the streaming NB-Fisher pass and \
                column-block I/O. Omit for auto-scaling (clamps to 100 for large \
                feature counts — slow on rotational disks). Pass 1024+ when you \
                have RAM, especially without --preload-data."
    )]
    block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory before any pass over \
                cells. Faster when data fits in RAM; required on slow disks."
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Always recompute NB-Fisher weights and overwrite the cache. By \
                default `{out}.fisher_weights.parquet` is loaded if it exists \
                with matching gene names, otherwise computed and written."
    )]
    no_fisher_cache: bool,

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

    let feature_kind = if args.feature_name_exact {
        ge::FeatureNameKind::Exact
    } else {
        ge::FeatureNameKind::Gene {
            delim: args.feature_name_delim,
        }
    };
    let mut unified = ge::load_unified_data(
        &args.data_files,
        args.batch_files.as_deref(),
        feature_kind,
        args.preload_data,
    )?;

    // HVG → projection weights (no longer subsets the feature axis).
    // Mirrors senna topic: HVG down-weights uninformative genes for the
    // random projection / pb sketching only; collapse + supergene
    // coarsening + training read all genes. Caller passes the weights
    // through `FitConfig.hvg_weights`.
    let hvg_enabled = args.hvg.n_hvg > 0 || args.hvg.feature_list_file.is_some();
    let hvg_weights: Option<Vec<f32>> = if hvg_enabled {
        let hvg = select_hvg_streaming(
            &unified.per_file_data[0],
            (args.hvg.n_hvg > 0).then_some(args.hvg.n_hvg),
            args.hvg.feature_list_file.as_deref(),
            args.block_size,
        )?;
        Some(hvg.row_weights(unified.n_features()))
    } else {
        None
    };

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

    let cell_cell = args
        .cell_cell_edges
        .as_deref()
        .map(|path| {
            let edges = load_cell_cell_edges(path, &unified.barcodes)?;
            anyhow::Ok(ge::CellCellConfig {
                edges,
                lambda: args.cell_cell_lambda,
                n_negatives: args.cell_cell_negatives,
            })
        })
        .transpose()?;

    let refine = if args.no_refine {
        None
    } else {
        Some(ge::RefineParams {
            num_gibbs: args.refine_gibbs,
            num_greedy: args.refine_greedy,
            feature_weighting: args.refine_weighting.into(),
            seed: args.refine_seed,
            ..ge::RefineParams::default()
        })
    };

    let config = ge::FitConfig {
        embedding_dim: args.embedding_dim,
        num_coarsen_seeds: args.num_coarsen_seeds,
        max_features: args.max_features,
        hvg_weights,
        composite_mode: args.composite_mode.into(),
        refine,
        super_cells: args.super_cells,
        sketch_dim: args.sketch_dim,
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch_size: args.batch_size,
        num_negatives: args.num_negatives,
        learning_rate: args.learning_rate,
        seed: args.seed,
        device: args.device.to_device(args.device_no)?,
        block_size: args.block_size,
        fisher_weights_cache: if args.no_fisher_cache {
            None
        } else {
            Some(format!("{}.fisher_weights.parquet", args.out).into_boxed_str())
        },
        feature_network,
        cell_cell,
        stop: None,
    };

    let out = ge::fit(&mut unified, config)?;

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

/// Read a whitespace-separated two-column edge list of cell barcodes,
/// resolve each barcode to its global cell id via `barcodes`. Edges
/// with either endpoint unmatched are skipped (warned on count).
/// Self-loops (i == j) are dropped silently.
fn load_cell_cell_edges(path: &str, barcodes: &[Box<str>]) -> anyhow::Result<Vec<(u32, u32)>> {
    info!("Loading cell-cell edge list from {}...", path);
    let bc_to_id: FxHashMap<&str, u32> = barcodes
        .iter()
        .enumerate()
        .map(|(i, b)| (b.as_ref(), i as u32))
        .collect();

    let file =
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("failed to open {}: {}", path, e))?;
    let reader = std::io::BufReader::new(file);

    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut unmatched = 0usize;
    let mut malformed = 0usize;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut toks = trimmed.split_whitespace();
        let (Some(a), Some(b)) = (toks.next(), toks.next()) else {
            malformed += 1;
            continue;
        };
        match (bc_to_id.get(a), bc_to_id.get(b)) {
            (Some(&i), Some(&j)) if i != j => {
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                edges.push((lo, hi));
            }
            (Some(_), Some(_)) => {} // self-loop, drop silently
            _ => unmatched += 1,
        }
    }

    if malformed > 0 {
        log::warn!(
            "{} malformed line(s) in {} (need ≥2 whitespace-separated tokens)",
            malformed,
            path
        );
    }
    if unmatched > 0 {
        log::warn!(
            "{} edge(s) had at least one barcode not present in the data — skipped",
            unmatched
        );
    }
    if edges.is_empty() {
        anyhow::bail!(
            "cell-cell edge file {} produced 0 usable edges (after barcode resolution)",
            path
        );
    }
    info!("Cell-cell edges loaded: {} retained", edges.len());
    Ok(edges)
}
