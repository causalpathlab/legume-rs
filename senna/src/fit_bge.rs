//! `senna bge` (Bipartite Graph Embedding) — thin clap + run-manifest
//! wrapper around the `graph-embedding-util` engine.
//!
//! Previously `senna gbe`; renamed to clarify that the bipartite (cell ×
//! feature) graph is *built* internally from expression counts. The
//! sibling `senna fne` (Feature Network Embedding) is the graph-input
//! companion that consumes an explicit feature-feature edge list.
//! `gbe` remains a clap alias for one release cycle.
//!
//! All algorithmic work lives in `graph_embedding_util`. This file
//! exists only to translate `BgeArgs` → `FitConfig`, resolve the
//! optional feature-network edge file against the unified feature
//! axis, and write senna's run manifest after training.

use crate::embed_common::*;
use auxiliary_data::frozen_features::{
    load_frozen_feature_host, FrozenFeatureHost, FrozenLoadArgs,
};
use data_beans_alg::hvg::{select_hvg_streaming, HvgCliArgs};
use graph_embedding_util as ge;
use rustc_hash::FxHashMap;
use std::io::BufRead;
use std::path::Path;

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
pub struct BgeArgs {
    #[arg(
        value_delimiter = ',',
        help = "Sparse count matrices (zarr/h5), comma-separated (single-modality). \
                For multiome input use --multiome instead.",
        long_help = "Single-modality input: one or more files sharing a feature axis, \
                     cells unified by barcode. For multiome (distinct feature spaces \
                     per modality, glued by barcode) pass the files to --multiome \
                     instead. Exactly one of [positional files | --multiome] is required."
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

    #[command(flatten)]
    collapse: crate::refine_weighting::CollapseArgs,

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
        long = "skip-etm",
        default_value_t = false,
        help = "Skip the default ETM resolution and emit only the raw bge embeddings \
                (latent = cell embedding Z, dictionary = ρ). By default bge resolves \
                ETM topics from the cell embedding via archetypal analysis and writes \
                a topic-model layout (latent = log θ, dictionary = β) plus \
                {cell,feature}_embedding.parquet."
    )]
    skip_etm: bool,

    #[arg(
        long = "num-topics",
        help = "Number of ETM topics K for --resolve-etm. Omit to auto-select \
                via an archetypal RSS-elbow sweep over 2..=--max-k."
    )]
    num_topics: Option<usize>,

    #[arg(
        long = "max-k",
        default_value_t = 30,
        help = "Upper K for the --resolve-etm auto-sweep (when --num-topics is unset)."
    )]
    max_k: usize,

    #[arg(
        long = "aa-iters",
        default_value_t = 50,
        help = "Archetypal-analysis alternating iterations for --resolve-etm."
    )]
    aa_iters: usize,

    #[arg(
        long = "aa-subsample",
        help = "Cap on cells used to fit archetypes for --resolve-etm \
                (θ is still assigned for every cell)."
    )]
    aa_subsample: Option<usize>,

    #[arg(
        long = "bridge-weight",
        default_value_t = 1.0,
        help = "Up-weight matched (multi-modality) cells in the cell-axis sampler by \
                this factor so they anchor cross-modal alignment (--multiome only; \
                1.0 = off)."
    )]
    bridge_weight: f32,

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

    #[arg(short = 'i', long, default_value_t = 200, help = "Training epochs")]
    epochs: usize,

    /// Batches per epoch. **Omit for auto** — one weighted pass per
    /// epoch over the largest axis (`ceil(max_axis_units / batch_size)`).
    /// Pass a value to force a fixed step budget per epoch (historical
    /// default: 100).
    #[arg(
        long,
        help = "Batches per epoch (default: auto = one pass over largest axis)"
    )]
    batches_per_epoch: Option<usize>,

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

    #[arg(
        long,
        default_value_t = 1.0,
        help = "L2 penalty λ on the shared feature embedding E_feat ∈ ℝ^{D×H}: \
                adds λ · mean(E_feat²) to the per-step composite loss \
                (mean-normalized, so λ stays scale-invariant across D·H). \
                Default 1.0 (mild shrinkage). Set 0.0 to disable. \
                Typical: 0.1–10.0."
    )]
    feature_embedding_l2: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay applied uniformly to every \
                parameter (E_feat, b_feat, per-axis heads). Per-step \
                post-update shrinkage; doesn't enter the backward graph. \
                Default 0.0 (off — plain Adam despite the optimizer name)."
    )]
    weight_decay: f64,

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
        help = "Reuse a pre-trained per-gene embedding from a prior senna run. \
                Loads `{prefix}.dictionary.parquet` (+ `{prefix}.feature_bias.parquet` \
                if present) — or `{prefix}.feature_embedding.parquet` from a topic run, \
                in which case bias defaults to zero. Gene names are strict-intersected \
                with this dataset's unified feature axis under the fuzzy \
                `feature_name_delim` / `feature_name_exact` rules; unmatched features \
                are dropped from training. Cell-side embeddings still train; E_feat / \
                b_feat stay frozen. Incompatible with `--feature-network`; forces \
                `--max-features 0` and disables HVG."
    )]
    freeze_feature_embedding: Option<Box<str>>,

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
        value_delimiter = ',',
        num_args = 1..,
        help = "Multiome modality files (comma-separated), one per modality, glued by raw barcode.",
        long_help = "Multiome load: pass one sparse file per modality, e.g. \
                     `--multiome rna.zarr,atac.zarr`. Each file keeps its own feature \
                     space (no cross-file barcode suffixing); cells are unioned by raw \
                     barcode — a cell seen only in RNA contributes triplets just to the \
                     RNA block, ATAC-only cells just to the ATAC block, shared cells get \
                     both. File ORDER defines modality order (used for per-modality \
                     rebalancing / outputs). Maps to `ColumnAlignment::Union` and \
                     replaces the positional data files.\n\
                     \n\
                     Batch resolution is constrained: a single --batch-files file is \
                     allowed (one label per unified cell), or embedded `@batch` tags \
                     that agree across modalities."
    )]
    multiome: Option<Vec<Box<str>>>,

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

pub fn fit_bge(args: &BgeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    // Input files: positional (single-modality) OR --multiome modality files.
    // File order under --multiome defines modality order.
    let is_multiome = args.multiome.as_ref().is_some_and(|v| !v.is_empty());

    // Multiome mixes gene rows (RNA) and locus rows (ATAC peaks) on one axis,
    // so canonicalize per-name via `Mixed` (genes → gene rule, `chrX:s-e` →
    // locus rule). This also makes feature-network edges (e.g. gene↔peak
    // cis-links) resolve against the same canonical names. `--feature-name-exact`
    // still forces verbatim matching.
    let feature_kind = if args.feature_name_exact {
        ge::FeatureNameKind::Exact
    } else if is_multiome {
        ge::FeatureNameKind::Mixed
    } else {
        ge::FeatureNameKind::Gene {
            delim: args.feature_name_delim,
        }
    };

    let data_files: &[Box<str>] = if is_multiome {
        args.multiome.as_deref().unwrap()
    } else {
        anyhow::ensure!(
            !args.data_files.is_empty(),
            "no input files: pass single-modality files positionally, or modality \
             files via `--multiome rna,atac`"
        );
        &args.data_files
    };

    let (effective_multiome, effective_hvg_n, effective_hvg_list) =
        crate::hvg::resolve_multiome_with_hvg(is_multiome, data_files.len(), &args.hvg);
    let column_alignment = if effective_multiome {
        data_beans::sparse_io_vector::ColumnAlignment::Union
    } else {
        data_beans::sparse_io_vector::ColumnAlignment::Disjoint
    };

    // `--ignore-batch` drops the batch labels entirely, so the projection
    // and multilevel collapse run as if every cell shared one batch.
    let batch_files = if args.collapse.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };

    let mut unified = ge::load_unified_data(
        data_files,
        batch_files,
        feature_kind.clone(),
        args.preload_data,
        column_alignment,
    )?;

    // ---- Optional frozen feature side ----
    //
    // Loaded BEFORE HVG / max_features / feature_network so they all see
    // the post-intersection axis. The prior gene set is the selection,
    // so we force off the other gene-selection knobs when frozen.
    let frozen_feature_host = if let Some(prefix) = args.freeze_feature_embedding.as_deref() {
        if args.feature_network.is_some() {
            anyhow::bail!(
                "--freeze-feature-embedding is incompatible with --feature-network \
                 (SGC smoothing of a frozen table has no learnable parameter to nudge)"
            );
        }
        if args.max_features > 0 {
            log::warn!(
                "--freeze-feature-embedding overrides --max-features={} → 0 \
                 (the frozen gene set is the selection)",
                args.max_features
            );
        }
        if effective_hvg_n > 0 || effective_hvg_list.is_some() {
            log::warn!(
                "--freeze-feature-embedding disables HVG selection \
                 (the frozen gene set is the selection)"
            );
        }
        Some(load_frozen_feature_host_for_bge(
            prefix,
            &unified.feature_names,
            feature_kind.clone(),
        )?)
    } else {
        None
    };
    if let Some(host) = frozen_feature_host.as_ref() {
        unified.subset_features(&host.keep_target_indices);
    }
    let freeze_active = frozen_feature_host.is_some();

    // HVG → projection weights (no longer subsets the feature axis).
    // Mirrors senna topic: HVG down-weights uninformative genes for the
    // random projection / pb sketching only; collapse + supergene
    // coarsening + training read all genes. Caller passes the weights
    // through `FitConfig.hvg_weights`.
    let hvg_enabled = !freeze_active && (effective_hvg_n > 0 || effective_hvg_list.is_some());
    let hvg_weights: Option<Vec<f32>> = if hvg_enabled {
        let hvg = select_hvg_streaming(
            &unified.per_file_data[0],
            (effective_hvg_n > 0).then_some(effective_hvg_n),
            effective_hvg_list,
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
                feature_kind: feature_kind.clone(),
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
                pb_levels: None,
                lambda_per_level: None,
            })
        })
        .transpose()?;

    // `--no-refine` is gbe-specific (the other subcommands always refine);
    // otherwise the shared `--pb-refine-*` flags drive RefineParams.
    let refine = if args.no_refine {
        None
    } else {
        Some(args.collapse.pb_refine.to_params())
    };

    // Up-weight matched (multi-modality) cells in the cell-axis sampler so
    // they anchor the cross-modal alignment. No-op outside --multiome
    // (single-modality cells all have one modality bit set).
    let cell_weight_mult: Option<Vec<f32>> =
        if is_multiome && (args.bridge_weight - 1.0).abs() > f32::EPSILON {
            let mult: Vec<f32> = unified
                .cell_modality
                .iter()
                .map(|&m| {
                    if m.count_ones() >= 2 {
                        args.bridge_weight
                    } else {
                        1.0
                    }
                })
                .collect();
            let n_matched = mult
                .iter()
                .filter(|&&w| (w - 1.0).abs() > f32::EPSILON)
                .count();
            info!(
                "--bridge-weight {}: up-weighting {} matched cells in the cell-axis sampler",
                args.bridge_weight, n_matched
            );
            Some(mult)
        } else {
            None
        };

    let config = ge::FitConfig {
        embedding_dim: args.embedding_dim,
        num_levels: args.collapse.num_levels,
        sort_dim: args.collapse.sort_dim,
        knn_pb_samples: args.collapse.knn_cells,
        num_opt_iter: args.collapse.iter_opt,
        proj_dim: args.collapse.proj_dim,
        max_features: if freeze_active { 0 } else { args.max_features },
        hvg_weights,
        composite_mode: args.composite_mode.into(),
        refine,
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch_size: args.batch_size,
        num_negatives: args.num_negatives,
        learning_rate: args.learning_rate,
        // gbe no longer exposes a --seed knob; pin the sampling RNG.
        seed: 1,
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
        feature_embedding_l2: args.feature_embedding_l2,
        weight_decay: args.weight_decay,
        frozen_feature_host,
        cell_weight_mult,
    };

    let out = ge::fit(&mut unified, config)?;

    // Output layout depends on --resolve-etm:
    //   off → bge embeddings (latent = cell embedding Z, dictionary = ρ).
    //   on  → ETM topic-model layout (latent = log θ, dictionary = β); the raw
    //         embeddings are preserved as {cell,feature}_embedding.parquet so a
    //         `--from` chain still finds ρ, while `senna plot` / `plot-topic`
    //         pick up the resolved topics directly.
    let resolve_etm = !args.skip_etm;
    if resolve_etm {
        resolve_etm_topics(&out.model, &unified.feature_names, &unified.barcodes, args)?;
    } else {
        ge::save_outputs(
            &out.model,
            &ge::OutputContext {
                feature_names: &unified.feature_names,
                barcodes: &unified.barcodes,
            },
            &args.out,
        )?;
    }

    let input: Vec<String> = data_files.iter().map(|s| s.to_string()).collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::Bge,
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        // With --resolve-etm the dictionary is β (gene × topic) and ρ moves to
        // feature_embedding.parquet; otherwise the dictionary IS ρ.
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: if resolve_etm {
            Some("feature_embedding.parquet")
        } else {
            None
        },
        default_colour_by: if resolve_etm { "topic" } else { "cluster" },
        has_latent: true,
        has_cell_to_pb: false,
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

/// Probe a senna-run prefix for a frozen feature side. Tries:
///   1. `{prefix}.dictionary.parquet` + `{prefix}.feature_bias.parquet` (bge layout).
///   2. `{prefix}.feature_embedding.parquet` + optional `{prefix}.feature_bias.parquet`
///      (topic / fne layout — fne writes a learned bias too).
///
/// Strict-intersects gene names against `target_feature_names` under `kind`.
fn load_frozen_feature_host_for_bge(
    prefix: &str,
    target_feature_names: &[Box<str>],
    kind: ge::FeatureNameKind,
) -> anyhow::Result<FrozenFeatureHost> {
    let bge_dict = format!("{prefix}.dictionary.parquet");
    let bias_file = format!("{prefix}.feature_bias.parquet");
    let topic_dict = format!("{prefix}.feature_embedding.parquet");

    let (dict_path, bias_path) = if Path::new(&bge_dict).exists() {
        let bias = Path::new(&bias_file).exists().then_some(bias_file.clone());
        if bias.is_none() {
            log::warn!(
                "{} found but {} missing — loading dictionary only, bias defaults to zero",
                bge_dict,
                bias_file
            );
        }
        (bge_dict, bias)
    } else if Path::new(&topic_dict).exists() {
        let bias = Path::new(&bias_file).exists().then_some(bias_file.clone());
        if bias.is_some() {
            info!(
                "Frozen feature side: loading {} + {} (fne-style with learned bias)",
                topic_dict, bias_file
            );
        } else {
            info!(
                "Frozen feature side: loading topic-style {} (bias = 0)",
                topic_dict
            );
        }
        (topic_dict, bias)
    } else {
        anyhow::bail!(
            "--freeze-feature-embedding {prefix}: no {prefix}.dictionary.parquet or \
             {prefix}.feature_embedding.parquet — pass a prior senna bge / topic / \
             fne output prefix"
        );
    };

    let host = load_frozen_feature_host(FrozenLoadArgs {
        dictionary_path: &dict_path,
        bias_path: bias_path.as_deref(),
        target_feature_names,
        name_kind: kind,
    })?;
    Ok(host)
}

/////////////////////////////////////////////////////////////////////
// ETM resolution from the bge cell embedding (--resolve-etm)        //
/////////////////////////////////////////////////////////////////////

/// Resolve the ETM topic side from a finished bge run, with no further
/// training, and write a topic-model-shaped output layout so that
/// `senna {plot, plot-topic, clustering, annotate} --from` consume the
/// topics directly (matching the `senna topic` / `itopic` conventions:
/// `latent` = log θ, `dictionary` = β).
///
/// Archetypal analysis on the cell embedding `Z [N,H]` yields archetypes
/// `α [K,H]` (= topic embeddings) and per-cell simplex weights `θ [N,K]`
/// (= topic proportions); the dictionary is `β = log_softmax_d(ρ·αᵀ)`,
/// the same factorization the ETM decoder uses. Writes:
///   - `{out}.latent.parquet`           log θ [N,K]   (topic proportions)
///   - `{out}.dictionary.parquet`       β    [D,K]   (each topic column a gene simplex)
///   - `{out}.topic_embedding.parquet`  α    [K,H]   (for a later `itopic` finetune)
///   - `{out}.cell_embedding.parquet`   Z    [N,H]   (raw bge cell embedding)
///   - `{out}.feature_embedding.parquet`ρ    [D,H]   (raw bge feature embedding)
///   - `{out}.{cell,feature}_bias.parquet`            (as `ge::save_outputs`)
fn resolve_etm_topics(
    model: &ge::JointEmbedModel,
    feature_names: &[Box<str>],
    barcodes: &[Box<str>],
    args: &BgeArgs,
) -> anyhow::Result<()> {
    use matrix_util::archetypal::{archetypal_analysis, select_archetype_k, AaArgs};

    let cpu = candle_core::Device::Cpu;
    let z = Mat::from_tensor(&model.e_cell.to_device(&cpu)?)?; // [N, H]
    let rho = Mat::from_tensor(&model.e_feat.to_device(&cpu)?)?; // [D, H]
    let h = z.ncols();
    anyhow::ensure!(
        rho.ncols() == h,
        "resolve-etm: cell embedding H={} != feature embedding H={}",
        h,
        rho.ncols()
    );

    let base = AaArgs {
        k: 2,
        max_iter: args.aa_iters,
        fw_iters: 30,
        tol: 1e-4,
        seed: 1,
        subsample: args.aa_subsample,
    };

    let (k, res) = match args.num_topics {
        Some(k) => {
            anyhow::ensure!(k >= 2, "resolve-etm: --num-topics must be ≥ 2");
            info!("resolve-etm: archetypal analysis with fixed K={k}");
            (k, archetypal_analysis(&z, &AaArgs { k, ..base }))
        }
        None => {
            let krange: Vec<usize> = (2..=args.max_k.max(2)).collect();
            info!(
                "resolve-etm: auto-selecting K via archetypal RSS-elbow over 2..={}",
                args.max_k.max(2)
            );
            select_archetype_k(&z, &krange, &base)
        }
    };
    info!("resolve-etm: K={k}, reconstruction RSS={:.4}", res.rss);

    // β = log_softmax_d(ρ · αᵀ): [D, K], each topic column a simplex over genes.
    let beta_dk = (&rho * res.alpha.transpose()).log_softmax_columns();
    // log θ on the simplex.
    let log_theta = res.theta.map(|x| (x + 1e-8).ln());

    let topic_names = axis_id_names("T", k);
    let h_names = axis_id_names("h", h);
    let out = &args.out;

    // Topic-model layout — latent = log θ, dictionary = β.
    log_theta.to_parquet_with_names(
        &format!("{out}.latent.parquet"),
        (Some(barcodes), Some("cell")),
        Some(&topic_names),
    )?;
    beta_dk.to_parquet_with_names(
        &format!("{out}.dictionary.parquet"),
        (Some(feature_names), Some("gene")),
        Some(&topic_names),
    )?;
    // Resolved topic embeddings α (warm-start for a later `itopic` finetune).
    res.alpha.to_parquet_with_names(
        &format!("{out}.topic_embedding.parquet"),
        (Some(&topic_names), Some("topic")),
        Some(&h_names),
    )?;
    // Raw bge embeddings preserved under non-conflicting names.
    z.to_parquet_with_names(
        &format!("{out}.cell_embedding.parquet"),
        (Some(barcodes), Some("cell")),
        Some(&h_names),
    )?;
    rho.to_parquet_with_names(
        &format!("{out}.feature_embedding.parquet"),
        (Some(feature_names), Some("feature")),
        Some(&h_names),
    )?;
    // Bias terms — reuse the canonical writer `ge::save_outputs` uses.
    ge::eval::save_bias(
        &format!("{out}.cell_bias.parquet"),
        &model.b_cell,
        barcodes,
        "cell",
    )?;
    ge::eval::save_bias(
        &format!("{out}.feature_bias.parquet"),
        &model.b_feat,
        feature_names,
        "feature",
    )?;

    info!(
        "resolve-etm: wrote topic-model layout (latent=log θ, dictionary=β) + \
         {{cell,feature}}_embedding.parquet to {out}.*"
    );
    Ok(())
}
