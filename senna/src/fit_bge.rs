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
use data_beans_alg::hvg::{select_hvg_streaming, HvgCliArgs};
use graph_embedding_util as ge;
use rustc_hash::FxHashMap;
use std::io::BufRead;

/// One parsed `--multiome` file entry: `(optional modality label, file path)`.
/// The label (or, when `None`, the within-group position) namespaces that
/// file's features as `{name}/{modality}`.
type MultiomeFile = (Option<Box<str>>, Box<str>);

#[derive(Args, Debug)]
pub struct BgeArgs {
    #[arg(
        value_delimiter = ',',
        help = "Sparse count matrices (zarr/h5), comma-separated. Use --multiome\n\
                for multi-modality input.",
        long_help = "Single-modality input: one or more files sharing a feature axis,\n\
                     cells unified by barcode. For multiome (distinct feature spaces\n\
                     per modality, glued by barcode) pass the files to --multiome\n\
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

    #[arg(
        long,
        value_delimiter = ',',
        help = "Per-cell condition label file(s) for the feature gate\n\
                (same format as --batch-files; omit → condition = batch).",
        long_help = "Drives a per-condition multiplicative gate on the gene\n\
                     embedding: e_feat(f | condition s) = E_feat[f] ⊙\n\
                     exp(Σ_k z[f,k]·δ̄[k,s,:]), where δ̄ is δ mean-centered\n\
                     across conditions (Σ_s δ̄ = 0). No condition is a\n\
                     reference — every condition deviates symmetrically and\n\
                     the baseline E_feat is the average-condition embedding;\n\
                     δ captures how each condition deviates each gene.\n\
                     Mirrors faba gem's modality gate with conditions in\n\
                     place of modalities. When omitted, condition = batch."
    )]
    condition_files: Option<Vec<Box<str>>>,

    #[arg(
        long = "num-programs",
        default_value_t = 8,
        help = "K: latent programs in the per-condition feature gate.",
        long_help = "K: number of latent programs in the per-condition feature\n\
                     gate (low-rank deviation). Only adds capacity when conditions\n\
                     diverge; identity at init."
    )]
    num_programs: usize,

    #[arg(
        long = "z-l2",
        default_value_t = 1e-4,
        help = "L2 penalty on gate program loadings z (mean-normalized). 0 disables."
    )]
    z_l2: f32,

    #[arg(
        long = "delta-l2",
        default_value_t = 1e-4,
        help = "L2 penalty on gate deviation directions δ (mean-normalized). 0 disables."
    )]
    delta_l2: f32,

    #[command(flatten)]
    hvg: HvgCliArgs,

    #[arg(long, default_value_t = 16, help = "Embedding dimension H")]
    embedding_dim: usize,

    #[command(flatten)]
    collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    qc: QcArgs,

    #[arg(
        long,
        default_value_t = 0,
        help = "Cap on genes trained (0 = keep all); main large-data speed knob.",
        long_help = "Cap on the number of genes trained (0 = keep all). When > 0\n\
                     and less than the feature axis, keeps the top-N genes by\n\
                     NB-Fisher weight and drops the rest before the multilevel\n\
                     collapse. Shrinks E_feat, triplets, and per-batch samplers\n\
                     proportionally — the main large-data speed knob."
    )]
    max_features: usize,

    #[arg(
        long = "phase1-cells-per-pb",
        default_value_t = 0,
        help = "Phase-1 cell-axis mode (k); 0 = pure-pb (fastest), phase 2 always\n\
                projects every cell.",
        long_help = "Phase-1 cell-axis mode (k). Controls what shapes the feature\n\
                     dictionary in phase 1; phase 2 ALWAYS analytically projects\n\
                     every cell, so the per-cell embedding output is unaffected.\n\
                     k=0 (default) → suppress the cell axis entirely (pure-pb:\n\
                     E_feat from pb aggregates only — fastest). 1≤k<n_cells →\n\
                     keep ≤k cells per pb-sample at each collapse level (union),\n\
                     cutting the phase-1 step budget while preserving rare-cell\n\
                     coverage. k≥n_cells → all cells (legacy; slowest). NOTE: an\n\
                     optional --cell-cell term rides the cell axis and is dropped\n\
                     when k=0."
    )]
    phase1_cells_per_pb: usize,

    #[arg(
        long = "skip-etm",
        default_value_t = false,
        help = "Skip ETM resolution; emit raw bge embeddings (Z and ρ) only.",
        long_help = "Skip the default ETM resolution and emit only the raw bge\n\
                     embeddings (latent = cell embedding Z, dictionary = ρ). \n\
		     By default bge resolves ETM topics from the cell embedding via\n\
                     anchor analysis and writes a topic-model layout\n\
                     (latent = log θ, dictionary = β)."
    )]
    skip_etm: bool,

    #[arg(
        long = "num-topics",
        help = "ETM topics K (omit to auto-select via SPA-anchor residual-elbow sweep)."
    )]
    num_topics: Option<usize>,

    #[arg(
        long = "bridge-weight",
        default_value_t = 1.0,
        help = "Up-weight matched cells in the cell-axis sampler (--multiome only;\n\
                1.0 = off).",
        long_help = "Up-weight matched (multi-modality) cells in the cell-axis\n\
                     sampler by this factor so they anchor cross-modal alignment\n\
                     (--multiome only; 1.0 = off)."
    )]
    bridge_weight: f32,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable BBKNN + DC-Poisson refinement of the multi-level pseudobulk\n\
                partition. Default: enabled."
    )]
    no_refine: bool,

    #[arg(short = 'i', long, default_value_t = 1000, help = "Training epochs")]
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
        help = "L2 penalty λ on E_feat (mean-normalized). Default 1.0; 0.0 = off.",
        long_help = "L2 penalty λ on the shared feature embedding E_feat ∈ ℝ^{D×H}:\n\
                     adds λ · mean(E_feat²) to the per-step composite loss\n\
                     (mean-normalized, so λ stays scale-invariant across D·H).\n\
                     Default 1.0 (mild shrinkage). Set 0.0 to disable.\n\
                     Typical: 0.1–10.0."
    )]
    feature_embedding_l2: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay (all params). Default 0.0 = off.",
        long_help = "AdamW decoupled weight decay applied uniformly to every\n\
                     parameter (E_feat, b_feat, per-axis heads). Per-step\n\
                     post-update shrinkage; doesn't enter the backward graph.\n\
                     Default 0.0 (off — plain Adam despite the optimizer name)."
    )]
    weight_decay: f64,

    #[arg(
        long,
        help = "Optional feature-feature edge list (TSV/CSV; activates SGC\n\
                smoothing of E_feat through the K-hop normalized adjacency)."
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
        help = "Name-stripping delimiter for feature-network resolution\n\
                (e.g. '.' to match `TP53.1` → `TP53`)"
    )]
    feature_network_delim: Option<char>,

    #[arg(
        long,
        default_value_t = 2,
        help = "SGC propagation hops K (default 2; K=2 covers all shared-neighbor\n\
                pairs).",
        long_help = "SGC propagation hops K (default 2 — useful for sparse\n\
                     synthetic-lethality / regulatory networks). K=2 already\n\
                     reaches all shared-neighbor pairs, so no separate SNN\n\
                     augmentation is needed."
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
        help = "Delimiter for fuzzy gene-name suffix matching across input files.",
        long_help = "Delimiter for fuzzy gene-name matching across input files.\n\
                     The last token after splitting on this char is used as the\n\
                     canonical row name, so `ENSG00000000003_TSPAN6` (file A)\n\
                     and `TSPAN6` (file B) merge into a single row. Pass an\n\
                     empty string (currently unsupported by clap; set\n\
                     --feature-name-kind to override) to fall back to exact\n\
                     matching."
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
        help = "Optional cell-cell edge list (TSV); activates the cell-cell NCE term.",
        long_help = "Optional cell-cell edge list (whitespace-separated, two\n\
                     cell-barcode columns per line; lines starting with `#` are\n\
                     ignored). When provided, activates the cell-cell NCE term —\n\
                     positives are these edges, negatives are within-batch random\n\
                     non-neighbor cells. Use a precomputed graph (e.g. pinto's\n\
                     spatial KNN, exported to TSV)."
    )]
    cell_cell_edges: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Cell-cell loss weight λ (1.0 = equal to cell-feature; 0 = disable).",
        long_help = "Cell-cell loss weight λ; \n\
		     final loss = L_bipartite + λ · L_cell_cell.\n\
                     Default 1.0 weights cell-cell equal to\n\
                     cell-feature. Set to 0 to disable. Ignored if\n\
                     --cell-cell-edges is not provided."
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
        help = "Cells per block for streaming NB-Fisher / column I/O (omit for auto).",
        long_help = "Cells per parallel block for the streaming NB-Fisher pass\n\
                     and column-block I/O. Omit for auto-scaling (clamps to 100\n\
                     for large feature counts — slow on rotational disks). Pass\n\
                     1024+ when you have RAM, especially without --preload-data."
    )]
    block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory. Faster when data fits\n\
                in RAM; required on slow disks."
    )]
    preload_data: bool,

    #[arg(
        long,
        value_name = "FILE[,FILE...]",
        help = "Multiome modality files (comma-separated); repeat for multiple samples.",
        long_help = "Multiome load: pass files for one sample (group) per flag,\n\
                     comma-separated, e.g. `--multiome rna.zarr,atac.zarr`. Cells\n\
                     are the shared axis; each modality keeps its own features.\n\
                     Repeat the flag for each additional sample/group:\n\
                     \n\
                       --multiome rna1.zarr,atac1.zarr \\\n\
                       --multiome rna2.zarr,atac2.zarr\n\
                     \n\
                     Cell (barcode) identity: within a group, equal barcodes are\n\
                     the same cell (Union merge across modalities; a cell present\n\
                     in only some files is fine — patchy multiome). ACROSS groups\n\
                     barcodes must be disjoint — a shared barcode would merge\n\
                     cells from different samples (validated; error on collision).\n\
                     \n\
                     Feature (modality) identity: features are namespaced\n\
                     `{name}/{modality}` so the SAME modality across samples\n\
                     merges (shared gene panel) while DIFFERENT modalities stay\n\
                     on separate rows — even when names collide (e.g. spliced vs\n\
                     unspliced `TSPAN6`). The modality tag defaults to the\n\
                     within-group file position (m0, m1, …); override it with a\n\
                     `label=` prefix, e.g.\n\
                       --multiome spliced=spliced.zarr,unspliced=unspliced.zarr\n\
                     File ORDER within a group defines modality order, so the\n\
                     positional default lines up across groups.\n\
                     \n\
                     Batch identity: each group becomes its own batch when\n\
                     --batch-files is omitted (modality-presence auto-batch).\n\
                     Pass a single --batch-files (one label per unified cell) to\n\
                     set batches explicitly. Replaces the positional data files.\n\
                     \n\
                     Note: comma-separate files within one group with no spaces\n\
                     (e.g. rna.zarr,atac.zarr); use a separate --multiome flag\n\
                     for each additional group."
    )]
    multiome: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Always recompute NB-Fisher weights, overwriting any existing cache.",
        long_help = "Always recompute NB-Fisher weights and overwrite the cache.\n\
                     By default `{out}.fisher_weights.parquet` is loaded if it\n\
                     exists with matching gene names, otherwise computed and\n\
                     written."
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
        long_help = "Output prefix; produces {out}.latent.parquet,\n\
                     {out}.dictionary.parquet, {out}.feature_bias.parquet,\n\
                     {out}.senna.json"
    )]
    out: Box<str>,
}

pub fn fit_bge(args: &BgeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    // Input files: positional (single-modality) OR --multiome modality groups.
    // Each --multiome occurrence is one group; comma-separated files within it.
    let is_multiome = !args.multiome.is_empty();

    // Parse each --multiome occurrence (one group) into its files, honoring an
    // optional `label=file` prefix that names the modality. The label (or, when
    // omitted, the within-group position `m{pos}`) namespaces that file's
    // features as `{name}/{label}` so distinct modalities stay on separate rows
    // (e.g. spliced vs unspliced `TSPAN6`), while the same modality across
    // samples (same label/position) still merges. Each `(label, file)` pair is
    // (Option<modality label>, file path).
    let multiome_groups: Vec<Vec<MultiomeFile>> = args
        .multiome
        .iter()
        .map(|s| {
            s.split(',')
                .map(|tok| match tok.split_once('=') {
                    Some((label, file)) if !label.is_empty() && !file.is_empty() => {
                        (Some(label.into()), file.into())
                    }
                    _ => (None, tok.into()),
                })
                .collect()
        })
        .collect();

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

    // Flatten groups to get the per-file slice passed to load_unified_data,
    // plus the parallel per-file modality suffix (label, else `m{within-group
    // position}`) used to namespace features as `{name}/{suffix}`.
    let data_files_flat: Vec<Box<str>>;
    let feature_suffix: Option<Vec<Box<str>>>;
    let data_files: &[Box<str>] = if is_multiome {
        data_files_flat = multiome_groups
            .iter()
            .flat_map(|g| g.iter().map(|(_, file)| file.clone()))
            .collect();
        feature_suffix = Some(
            multiome_groups
                .iter()
                .flat_map(|g| {
                    g.iter().enumerate().map(|(pos, (label, _))| {
                        label
                            .clone()
                            .unwrap_or_else(|| format!("m{pos}").into_boxed_str())
                    })
                })
                .collect(),
        );
        if multiome_groups.len() > 1 {
            let counts = multiome_groups
                .iter()
                .map(|g| g.len().to_string())
                .collect::<Vec<_>>()
                .join("+");
            info!(
                "--multiome: {} groups, {} total files ({})",
                multiome_groups.len(),
                data_files_flat.len(),
                counts
            );
        }
        if let Some(suf) = feature_suffix.as_ref() {
            info!(
                "--multiome: namespacing features as {{name}}/{{modality}} (per-file \
                 modality: {})",
                suf.iter()
                    .map(|s| s.as_ref())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        &data_files_flat
    } else {
        feature_suffix = None;
        anyhow::ensure!(
            !args.data_files.is_empty(),
            "no input files: pass single-modality files positionally, or multiome \
             groups via `--multiome rna.zarr,atac.zarr [--multiome rna2.zarr,atac2.zarr ...]`"
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
    // and multilevel collapse run as if every cell shared one batch. The
    // per-condition gate keys off the same notion of "no structure", so we
    // drop condition labels too (condition collapses to a single "all").
    let batch_files = if args.collapse.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };
    let condition_files = if args.collapse.ignore_batch {
        if args.condition_files.is_some() {
            info!("--ignore-batch: dropping condition labels; the feature gate is disabled");
        }
        None
    } else {
        args.condition_files.as_deref()
    };

    let mut unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: data_files.to_vec(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        condition_files: condition_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(feature_kind.clone()),
        preload: args.preload_data,
        column_alignment,
        per_file_feature_suffix: feature_suffix,
        // senna uses disjoint barcodes per group; no per-file barcode suffix.
        ..Default::default()
    })?;

    // Guard barcode identity across groups: disjoint barcodes, so Union
    // loading never merges cells from different samples. No-op for one group.
    if is_multiome {
        let group_sizes: Vec<usize> = multiome_groups.iter().map(Vec::len).collect();
        ge::validate_multiome_groups(&group_sizes, &unified.barcodes, &unified.cell_modality)?;
    }

    if unified.n_conditions() > 1 {
        info!(
            "Per-condition feature gate: {} conditions (symmetric, mean-centered \
             across conditions; no reference), K = {} programs",
            unified.n_conditions(),
            args.num_programs
        );
    }

    // ---- Cell QC ----
    // gem-style (UnifiedData has no `mask_columns`): every cell + edge still
    // informs the joint embedding / feature dictionary, but QC-failed cells
    // (near-empty floor + MAD outliers) are dropped from the archetypal
    // analysis and all per-cell outputs via a write-time `select_rows`.
    // Computed on the full-feature unified count backend, so n_genes is the
    // per-cell detected-feature count across all modalities.
    let qc_keep_idx: Option<Vec<usize>> = if let Some(cfg) = args.qc.to_config() {
        if cfg.feature_min_cells > 0 {
            log::warn!(
                "--qc-feature-min-cells is ignored by bge (cell-only QC; the \
                 dictionary keeps all features)"
            );
        }
        let report =
            data_beans::qc_lib::compute_qc(unified.count_backend(), &cfg, args.block_size)?;
        let keep = report.emit_idx_unmasked();
        info!(
            "QC: {} / {} cells kept for output ({} near-empty, {} MAD-outlier dropped)",
            keep.len(),
            unified.n_cells(),
            report.near_empty.iter().filter(|&&e| e).count(),
            report.n_cells_dropped,
        );
        Some(keep)
    } else {
        None
    };

    // HVG → projection weights (no longer subsets the feature axis).
    // Mirrors senna topic: HVG down-weights uninformative genes for the
    // random projection / pb sketching only; collapse + supergene
    // coarsening + training read all genes. Caller passes the weights
    // through `FitConfig.hvg_weights`.
    let hvg_enabled = effective_hvg_n > 0 || effective_hvg_list.is_some();
    let hvg_weights: Option<Vec<f32>> = if hvg_enabled {
        let hvg = select_hvg_streaming(
            unified.count_backend(),
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
        max_features: args.max_features,
        hvg_weights,
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
        num_programs: args.num_programs,
        z_l2: args.z_l2,
        delta_l2: args.delta_l2,
        weight_decay: args.weight_decay,
        cell_weight_mult,
        phase1_cells_per_pb: args.phase1_cells_per_pb,
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
        resolve_etm_topics(
            &out.model,
            &unified.feature_names,
            &unified.barcodes,
            args,
            qc_keep_idx.as_deref(),
        )?;
    } else {
        ge::save_outputs(
            &out.model,
            &ge::OutputContext {
                feature_names: &unified.feature_names,
                barcodes: &unified.barcodes,
                cell_keep_idx: qc_keep_idx.as_deref(),
            },
            &args.out,
        )?;
    }

    // Inspectable per-condition gate params (z, δ). Baseline E_feat / E_cell
    // / ETM topics above stay condition-free; this is the only artifact that
    // exposes the condition deviations. Skip when there's nothing to show
    // (single condition ⇒ δ ≡ 0).
    if unified.n_conditions() > 1 {
        ge::save_gate(
            &out.model,
            &unified.feature_names,
            &unified.condition_names,
            &args.out,
        )?;
        info!(
            "Wrote per-condition gate: {}.{{gene_program_loadings,program_condition_deviation}}.parquet",
            args.out
        );
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

/////////////////////////////////////////////////////////////////////
// ETM resolution from the bge cell embedding (--resolve-etm)        //
/////////////////////////////////////////////////////////////////////

/// Resolve the ETM topic side from a finished bge run, with no further
/// training, and write a topic-model-shaped output layout so that
/// `senna {plot, plot-topic, clustering, annotate} --from` consume the
/// topics directly (matching the `senna topic` / `masked-topic` conventions:
/// `latent` = log θ, `dictionary` = β).
///
/// Archetypal analysis on the cell embedding `Z [N,H]` yields archetypes
/// `α [K,H]` (= topic embeddings) and per-cell simplex weights `θ [N,K]`
/// (= topic proportions); the dictionary is `β = log_softmax_d(ρ·αᵀ)`,
/// the same factorization the ETM decoder uses. Writes:
///   - `{out}.latent.parquet`           log θ [N,K]   (topic proportions)
///   - `{out}.dictionary.parquet`       β    [D,K]   (each topic column a gene simplex)
///   - `{out}.topic_embedding.parquet`  α    [K,H]   (for a later `masked-topic` finetune)
///   - `{out}.cell_embedding.parquet`   Z    [N,H]   (raw bge cell embedding)
///   - `{out}.feature_embedding.parquet`ρ    [D,H]   (raw bge feature embedding)
///   - `{out}.feature_bias.parquet`     b_feat [D]    (no cell_bias — b_cell is dropped)
fn resolve_etm_topics(
    model: &ge::JointEmbedModel,
    feature_names: &[Box<str>],
    barcodes: &[Box<str>],
    args: &BgeArgs,
    cell_keep_idx: Option<&[usize]>,
) -> anyhow::Result<()> {
    use matrix_util::archetypal::{
        anchor_topics, select_anchor_topics, topic_dictionary, AnchorOpts,
    };

    let cpu = candle_core::Device::Cpu;
    let z_full = Mat::from_tensor(&model.e_cell.to_device(&cpu)?)?; // [N, H]
                                                                    // Drop QC-failed cells from archetype fitting + per-cell outputs. `z`
                                                                    // and `barcodes` are subset by the same `keep` so their rows stay
                                                                    // aligned; the dictionary β (from ρ + archetypes) is per-feature and
                                                                    // unaffected.
    let (z, barcodes): (Mat, Vec<Box<str>>) = match cell_keep_idx {
        Some(keep) => (
            z_full.select_rows(keep.iter()),
            keep.iter().map(|&i| barcodes[i].clone()).collect(),
        ),
        None => (z_full, barcodes.to_vec()),
    };
    let rho = Mat::from_tensor(&model.e_feat.to_device(&cpu)?)?; // [D, H]
    let h = z.ncols();
    anyhow::ensure!(
        rho.ncols() == h,
        "resolve-etm: cell embedding H={} != feature embedding H={}",
        h,
        rho.ncols()
    );

    // Separable-NMF topic recovery (Arora anchors via SPA) on the feature
    // embedding ρ: anchor features are the convex-hull vertices of the
    // feature cloud — near-pure markers, one per topic. Deterministic and
    // single-pass (no subsample, no nonconvex fit, no per-K refit); θ is then
    // assigned to every cell by projecting it onto the anchors (FW_ITERS
    // Frank–Wolfe steps — the simplex projection converges quickly, not a user
    // knob). The MIN_ANCHOR_CELLS guard drops singleton/outlier-feature topics:
    // a topic claimed by < 10 cells is almost surely an artifact at this scale.
    const FW_ITERS: usize = 30;
    const MIN_ANCHOR_CELLS: usize = 10;
    let opts = AnchorOpts {
        fw_iters: FW_ITERS,
        min_anchor_cells: MIN_ANCHOR_CELLS,
    };
    let res = match args.num_topics {
        Some(k) => {
            anyhow::ensure!(k >= 2, "resolve-etm: --num-topics must be ≥ 2");
            info!("resolve-etm: separable-NMF (SPA anchors) with fixed K={k}");
            anchor_topics(&z, &rho, k, opts)
        }
        None => {
            // SPA anchors are nested → the sweep is one pass; residual elbow
            // over 2..=H+1 selects K.
            let upper = (h + 1).max(2);
            let krange: Vec<usize> = (2..=upper).collect();
            info!("resolve-etm: auto-selecting K via SPA residual-elbow over 2..={upper}");
            select_anchor_topics(&z, &rho, &krange, opts).1
        }
    };
    // K reflects any anchors the guard dropped, not the requested count.
    let k = res.anchors.len();
    info!(
        "resolve-etm: K={k}, anchor features=[{}], reconstruction RSS={:.4}",
        res.anchors
            .iter()
            .map(|&d| feature_names[d].as_ref())
            .collect::<Vec<&str>>()
            .join(", "),
        res.rss
    );

    // β = log_softmax_d(ρ · (α − ᾱ)ᵀ): [D, K], each topic column a feature
    // simplex; markers surface as each topic's deviation from the mean anchor.
    // α here are the anchor-feature embeddings.
    let beta_dk = topic_dictionary(&rho, &res.alpha);
    // log θ on the simplex.
    let log_theta = res.theta.map(|x| (x + 1e-8).ln());

    let topic_names = axis_id_names("T", k);
    let h_names = axis_id_names("h", h);
    let out = &args.out;

    // Topic-model layout — latent = log θ, dictionary = β.
    log_theta.to_parquet_with_names(
        &format!("{out}.latent.parquet"),
        (Some(&barcodes), Some("cell")),
        Some(&topic_names),
    )?;
    beta_dk.to_parquet_with_names(
        &format!("{out}.dictionary.parquet"),
        (Some(feature_names), Some("gene")),
        Some(&topic_names),
    )?;
    // Resolved topic embeddings α (warm-start for a later `masked-topic` finetune).
    res.alpha.to_parquet_with_names(
        &format!("{out}.topic_embedding.parquet"),
        (Some(&topic_names), Some("topic")),
        Some(&h_names),
    )?;
    // Raw bge embeddings preserved under non-conflicting names.
    z.to_parquet_with_names(
        &format!("{out}.cell_embedding.parquet"),
        (Some(&barcodes), Some("cell")),
        Some(&h_names),
    )?;
    rho.to_parquet_with_names(
        &format!("{out}.feature_embedding.parquet"),
        (Some(feature_names), Some("feature")),
        Some(&h_names),
    )?;
    // Feature bias only — bge drops the per-cell bias `b_cell`
    // (score = E_feat·E_cell + b_feat), so no `cell_bias.parquet`.
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
