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

mod io;
mod resolve_etm;

use io::write_feature_qc;
use resolve_etm::resolve_etm_topics;

/// One parsed `--multiome` file entry: `(optional modality label, file path)`.
/// The label (or, when `None`, the within-group position) namespaces that
/// file's features as `{name}/{modality}`.
type MultiomeFile = (Option<Box<str>>, Box<str>);

/// NCE training objective for the bge feature/cell embedding (maps to
/// [`ge::loss::NceObjective`]).
#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum NceObjectiveArg {
    /// Per-pair logistic (SGNS) — bge's historical loss; byte-identical runs.
    Logistic,
    /// Sampled-softmax / InfoNCE — the negatives compete with the positive in
    /// one softmax; sharpens separation on dense data (the loss `faba gem` uses).
    Softmax,
}

impl NceObjectiveArg {
    fn to_ge(&self) -> ge::loss::NceObjective {
        match self {
            NceObjectiveArg::Logistic => ge::loss::NceObjective::Logistic,
            NceObjectiveArg::Softmax => ge::loss::NceObjective::Softmax,
        }
    }
}

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

    #[command(flatten)]
    hvg: HvgCliArgs,

    #[arg(
        long,
        default_value_t = 128,
        help = "Embedding dimension H",
        alias = "dim-embedding"
    )]
    embedding_dim: usize,

    #[command(flatten)]
    collapse: crate::refine_weighting::CollapseArgs,

    #[command(flatten)]
    qc: QcArgs,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Drop null features (E_feat never moved off init) at this FDR, then re-fit (default 0.05; 0 = off)",
        long_help = "Empirical-Bayes feature-null QC — ON by default at FDR 0.05.\n\
                     After training, the shared ash (adaptive-shrinkage) null call tests each\n\
                     feature's signed loadings one embedding dimension at a time, against a\n\
                     per-axis null whose scale is estimated empirically from the data\n\
                     (no fixed χ² assumption).\n\
                     A feature is live when it is confidently non-null on some axis\n\
                     (local false-sign rate below this FDR, Bonferroni over the H axes).\n\
                     The per-axis null is elicited by `--n-hvg`: the lowest-norm `n − n_hvg`\n\
                     features stand in as the presumed null that seeds each axis's scale,\n\
                     which the sampler then refines.\n\
                     Null features (untrained / background) are DROPPED and the model re-fits\n\
                     on the live feature axis (two-pass refine);\n\
                     dropped rows still get a projected co-embedding.\n\
                     The live/null flags and norm² are written to {out}.feature_qc.parquet.\n\
                     0 disables. Must be in [0, 1)."
    )]
    feature_null_fdr: f32,

    #[arg(
        long = "phase1-cells-per-pb",
        default_value_t = 0,
        help = "Phase-1 cell-axis mode (k); 0 = pure-pb (fastest),\n\
                phase 2 always projects every cell.",
        long_help = "Phase-1 cell-axis mode (k). Controls what shapes the feature dictionary in phase 1;\n\
                     phase 2 ALWAYS analytically projects every cell,\n\
                     so the per-cell embedding output is unaffected.\n\
                     k=0 (default) → suppress the cell axis entirely\n\
                     (pure-pb: E_feat from pb aggregates only — fastest).\n\
                     1≤k<n_cells → keep ≤k cells per pb-sample at each collapse level (union),\n\
                     cutting the phase-1 step budget while preserving rare-cell coverage.\n\
                     k≥n_cells → all cells (legacy; slowest)."
    )]
    phase1_cells_per_pb: usize,

    #[arg(
        long = "skip-etm",
        default_value_t = false,
        help = "Skip ETM resolution; emit raw bge embeddings (Z and ρ) only.",
        long_help = "Skip the default ETM resolution and emit only the raw bge embeddings\n\
                     (latent = cell embedding Z, dictionary = ρ).\n\
                     By default bge resolves ETM topics from the cell embedding\n\
                     via anchor analysis and writes a topic-model layout\n\
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
        long_help = "Up-weight matched (multi-modality) cells in the cell-axis sampler by this factor\n\
                     so they anchor cross-modal alignment\n\
                     (--multiome only; 1.0 = off)."
    )]
    bridge_weight: f32,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable BBKNN + DC-Poisson refinement of the multi-level pseudobulk partition.\n\
                Default: enabled."
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
        default_value_t = 0.0,
        help = "L2 penalty λ on E_feat (mean-normalized). Default 0 (off).",
        long_help = "L2 penalty λ on the shared feature embedding E_feat ∈ ℝ^{D×H}:\n\
                     adds λ · mean(E_feat²) to the per-step composite loss\n\
                     (mean-normalized, so λ stays scale-invariant across D·H).\n\
                     Default 0 (off): mean-normalization makes the per-element gradient tiny (÷ D·H),\n\
                     so E_feat — self-bounded under the NCE + analytical-projection setup —\n\
                     barely moves with it (toggling it shifts cell-type purity within run-to-run noise).\n\
                     Set > 0 only if E_feat drifts on long/deep runs."
    )]
    feature_embedding_l2: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay (all params). Default 0.0 = off.",
        long_help = "AdamW decoupled weight decay applied uniformly to every parameter\n\
                     (E_feat, b_feat, per-axis heads).\n\
                     Per-step post-update shrinkage; doesn't enter the backward graph.\n\
                     Default 0.0 (off — plain Adam despite the optimizer name)."
    )]
    weight_decay: f64,

    #[arg(
        long = "max-grad-norm",
        default_value_t = 1.0,
        help = "Global-norm gradient clip per AdamW step (0 = off). When > 0, \n\
                gradients are scaled down if their global L2 norm exceeds this, \n\
                bounding embedding inflation on NCE loss spikes."
    )]
    max_grad_norm: f32,

    #[arg(
        long,
        help = "Optional feature-feature edge list (TSV/CSV;\n\
                activates SGC smoothing of E_feat through the K-hop normalized adjacency)."
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
        help = "SGC propagation hops K (default 2;\n\
                K=2 covers all shared-neighbor pairs).",
        long_help = "SGC propagation hops K\n\
                     (default 2 — useful for sparse synthetic-lethality / regulatory networks).\n\
                     K=2 already reaches all shared-neighbor pairs,\n\
                     so no separate SNN augmentation is needed."
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
                     The last token after splitting on this char is used as the canonical row name,\n\
                     so `ENSG00000000003_TSPAN6` (file A) and `TSPAN6` (file B) merge into a single row.\n\
                     Pass an empty string\n\
                     (currently unsupported by clap; set --feature-name-kind to override)\n\
                     to fall back to exact matching."
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
        help = "Cells per block for column I/O / streaming (omit for auto).",
        long_help = "Cells per parallel block for streaming column-block I/O.\n\
                     Omit for auto-scaling (clamps to 100 for large feature counts — slow on rotational disks).\n\
                     Pass 1024+ when you have RAM, especially without --preload-data."
    )]
    block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse column data into memory.\n\
                Faster when data fits in RAM; required on slow disks."
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
                     Feature (modality) identity: features are namespaced `{name}/{modality}`\n\
                     so the SAME modality across samples merges (shared gene panel)\n\
                     while DIFFERENT modalities stay on separate rows —\n\
                     even when names collide (e.g. spliced vs unspliced `TSPAN6`).\n\
                     The modality tag defaults to the within-group file position (m0, m1, …);\n\
                     override it with a `label=` prefix, e.g.\n\
                       --multiome spliced=spliced.zarr,unspliced=unspliced.zarr\n\
                     File ORDER within a group defines modality order,\n\
                     so the positional default lines up across groups.\n\
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
        long = "nce-objective",
        default_value_t = NceObjectiveArg::Softmax,
        value_enum,
        help = "NCE objective: softmax (InfoNCE — negatives compete; sharper on\n\
                dense pseudobulk data; default) or logistic (per-pair SGNS)"
    )]
    nce_objective: NceObjectiveArg,

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
                     {out}.cell_bias.parquet, {out}.senna.json"
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
                "--multiome: namespacing features as {{name}}/{{modality}} \n\
		 (per-file modality: {})",
                suf.iter()
                    .map(std::convert::AsRef::as_ref)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        &data_files_flat
    } else {
        feature_suffix = None;
        anyhow::ensure!(
            !args.data_files.is_empty(),
            "no input files: pass single-modality files positionally, or multiome \n\
             groups via `--multiome rna.zarr,atac.zarr [--multiome rna2.zarr,atac2.zarr ...]`"
        );
        &args.data_files
    };

    let effective_hvg =
        crate::hvg::resolve_multiome_with_hvg(is_multiome, data_files.len(), &args.hvg);
    let effective_multiome = effective_hvg.multiome;
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

    let mut unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: data_files.to_vec(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
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

    /////////////////////////////
    // Cell QC (output filter) //
    /////////////////////////////
    // The `--qc` near-empty floor + MAD-outlier call is an OUTPUT filter: every
    // cell + edge still informs the joint embedding / feature dictionary, but
    // QC-failed cells are dropped from the archetypal analysis and all per-cell
    // outputs via a write-time `select_rows`. (The separate EB empty-droplet
    // call below, when `--cell-null-fdr > 0`, instead masks empties out of the
    // backend and re-fits.) Computed on the full-feature unified count backend,
    // so n_genes is the per-cell detected-feature count across all modalities.
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
    // Full-axis HVG weights (backend-row indexed, identity-aligned to the
    // current feature axis). Subset through `feature_to_backend_row` inside
    // `build_config` so the same vector serves pass 1 (full) and the post-QC
    // pass 2 (null features dropped). The feature network is rebuilt per pass
    // (its graph is aligned to the live feature-name axis), so it lives in the
    // closure rather than here.
    //
    // `--must-train-features` is loaded whether or not HVG is on, because it also
    // exempts its features from the `--feature-null-fdr` drop further down — that
    // gate runs independently of HVG, so a curated panel must survive it too.
    let hvg_enabled = effective_hvg.selection_on();
    let must_train = crate::hvg::load_must_train(
        effective_hvg.must_train_file,
        hvg_enabled || args.feature_null_fdr > 0.0,
    )?;
    let hvg_full: Option<Vec<f32>> = if hvg_enabled {
        let hvg = select_hvg_streaming(
            unified.count_backend(),
            (effective_hvg.n_hvg > 0).then_some(effective_hvg.n_hvg),
            effective_hvg.feature_list_file,
            must_train.as_ref(),
            args.block_size,
        )?;
        Some(hvg.row_weights(unified.n_features()))
    } else {
        None
    };

    // `--no-refine` is gbe-specific (the other subcommands always refine);
    // otherwise the shared `--pb-refine-*` flags drive RefineParams.
    let refine = if args.no_refine {
        None
    } else {
        Some(args.collapse.pb_refine.to_params())
    };

    // Install the Ctrl-C stop handler once and share the flag across both
    // passes — each `ge::fit` would otherwise try to register its own SIGINT
    // handler and the second registration panics (`MultipleHandlers`).

    // Full-backend feature identity, captured before any subsetting. bge never
    // narrows the feature axis by HVG, so this differs from the trained axis only
    // after a `--feature-null-fdr` refit — and those dropped rows are exactly what
    // the post-hoc projection re-embeds.
    let backend_feature_names = unified.count_backend().row_names()?;
    let n_backend_rows = backend_feature_names.len();

    // Assemble a `FitConfig` for the CURRENT feature AND cell axes of `unified`,
    // so the same builder serves pass 1 (full axis), the post-QC feature re-fit
    // (null features dropped) and the cell-empty re-fit (empties dropped): HVG
    // weights subset through `feature_to_backend_row`, the feature network
    // reloads against the live feature names, and the cell-indexed bridge weights
    // resolve against the live barcodes/cell axis. Everything else is
    // axis-independent and cloned in.
    let build_config = |unified: &ge::UnifiedData| -> anyhow::Result<ge::FitConfig> {
        let hvg_weights = hvg_full.as_ref().map(|w| {
            unified
                .feature_to_backend_row
                .iter()
                .map(|&i| w[i])
                .collect::<Vec<f32>>()
        });
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
        // Up-weight matched (multi-modality) cells in the cell-axis sampler so
        // they anchor the cross-modal alignment. No-op outside --multiome.
        let cell_weight_mult: Option<Vec<f32>> =
            if is_multiome && (args.bridge_weight - 1.0).abs() > f32::EPSILON {
                Some(
                    unified
                        .cell_modality
                        .iter()
                        .map(|&m| {
                            if m.count_ones() >= 2 {
                                args.bridge_weight
                            } else {
                                1.0
                            }
                        })
                        .collect(),
                )
            } else {
                None
            };
        Ok(ge::FitConfig {
            embedding_dim: args.embedding_dim,
            num_levels: args.collapse.num_levels,
            sort_dim: args.collapse.sort_dim,
            knn_pb_samples: args.collapse.knn_cells,
            num_opt_iter: args.collapse.iter_opt,
            proj_dim: args.collapse.proj_dim,
            hvg_weights,
            refine: refine.clone(),
            epochs: args.epochs,
            batches_per_epoch: args.batches_per_epoch,
            batch_size: args.batch_size,
            num_negatives: args.num_negatives,
            learning_rate: args.learning_rate,
            // gbe no longer exposes a --seed knob; pin the sampling RNG.
            seed: 1,
            device: args.device.to_device(args.device_no)?,
            block_size: args.block_size,
            feature_network,
            feature_embedding_l2: args.feature_embedding_l2,
            weight_decay: args.weight_decay,
            max_grad_norm: args.max_grad_norm,
            cell_weight_mult,
            phase1_cells_per_pb: args.phase1_cells_per_pb,
            // bge uses a free E_feat (no per-gene β-sharing factorization).
            feat_factor: None,
            // δ_g splice offset is gem-only (needs feat_factor); off for bge.
            delta_l2: 0.0,
            // Lineage-DAG is a gem-only (β-sharing) path; off for bge.
            lineage_dag: false,
            dag_learnable: false,
            lineage_smooth: false,
            // bge never subsets the feature axis (HVG only reweights the random
            // projection), so on pass 1 the trained axis IS the backend and this
            // is an exact no-op. It earns its keep on the `--feature-null-fdr`
            // refit, where it re-embeds the rows that pass dropped. Each backend
            // row is its own "gene" (no β-sharing), and there is no splice track.
            feature_projection: Some(ge::FeatureProjectionConfig {
                ridge: ge::DEFAULT_PROJECTION_RIDGE,
                calib_ridge: ge::DEFAULT_PROJECTION_CALIB_RIDGE,
                backend_row_to_gene: (0..n_backend_rows as u32).collect(),
                backend_unspliced_rows: vec![false; n_backend_rows],
                with_velocity: false,
                // The same null call that dropped these rows also gates their
                // return: a row re-solves to a real direction or it stays at the
                // origin. Restoring coverage must not mean restoring noise.
                null_fdr: args.feature_null_fdr,
            }),
            // bge selects features by the norm² ash null call in `write_feature_qc`
            // (post-fit, below), not the engine's LRT scan. The NCE objective is
            // `--nce-objective` (default softmax = InfoNCE; logistic = per-pair
            // SGNS). The LRT knob stays off (gem-only).
            select_lrt_fdr: None,
            nce_objective: args.nce_objective.to_ge(),
        })
    };

    // Pass 1: fit on the full feature axis.
    let cfg = build_config(&unified)?;
    let mut out = ge::fit(&mut unified, cfg)?;

    // Empirical-Bayes null-feature QC on the trained E_feat (shared with faba
    // gem's gene QC): flags features the model never moved from init. With
    // `--feature-null-fdr > 0` this is a two-pass refine — drop the null
    // features and re-fit on the live axis (a fresh inference, not a reuse of
    // the pass-1 embeddings).
    if args.feature_null_fdr > 0.0 {
        let (n, h) = out.model.e_feat.dims2()?;
        let e_feat: Vec<f32> = out
            .model
            .e_feat
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let null = write_feature_qc(
            &e_feat,
            n,
            h,
            &unified.feature_names,
            args.feature_null_fdr,
            effective_hvg.n_hvg,
            &args.out,
        )?;
        let mut live: Vec<usize> = (0..n).filter(|&i| null.live[i]).collect();
        let raw_live = live.len();

        // Degenerate-fit guards read the RAW null call — BEFORE any
        // `--must-train-features` rescue. Rescuing first would make `live`
        // non-empty, the guard would never fire, and the refit would drop every
        // feature EXCEPT the force-kept ones — the panel would eat the
        // dictionary. Two cases keep the pass-1 output and skip the refit:
        //   (a) all-null — a refit on zero features is invalid;
        //   (b) >90% dropped — either a genuine collapse, OR a feature set with
        //       no empty-feature null population to find (e.g. an HVG-subset gem
        //       run: every row is a trained HVG gene, norm² is unimodal, and the
        //       null call over-flags). Refitting on the handful of survivors
        //       trains a broken run, so keep pass-1 and warn.
        if raw_live == 0 {
            log::warn!(
                "Feature null QC flagged all {n} features as null; keeping pass-1 (no refit)."
            );
        } else if raw_live * 10 < n {
            log::warn!(
                "Feature null QC flagged {} / {n} features null ({:.0}%) — a collapse or a \
                 null-free feature set, not a clean drop. Keeping pass-1 output instead of \
                 refitting on only {raw_live} survivor(s). Inspect {}.feature_qc.parquet before \
                 trusting this run.",
                n - raw_live,
                100.0 * (n - raw_live) as f64 / n as f64,
                args.out
            );
        } else {
            // `--must-train-features` outranks the null call as well as the HVG cut: a
            // curated feature the user named stays in the dictionary even when the model
            // never moved it off init. Still reported as null in the QC parquet.
            if let Some(must_train) = must_train.as_ref() {
                let rescued = crate::hvg::union_indices(
                    &mut live,
                    &must_train.resolve_quiet(&unified.feature_names),
                );
                if rescued > 0 {
                    log::warn!(
                        "--must-train-features: {rescued} feature(s) were called null at FDR {} \
                         but kept anyway — they carry no signal in this data, so their embeddings \
                         stay near their init. See {}.feature_qc.parquet.",
                        args.feature_null_fdr,
                        args.out
                    );
                }
            }

            if live.len() < n {
                info!(
                    "Two-pass refine: dropping {} null features, re-fitting on {} live features.",
                    n - live.len(),
                    live.len()
                );
                unified.subset_features(&live);
                let cfg = build_config(&unified)?;
                out = ge::fit(&mut unified, cfg)?;
            }
        }
    }

    // No per-batch cell QC: it was removed because the per-batch debris cut
    // behaved incoherently across batches (near-identical depth distributions
    // produced 0%-vs-44% drops, guillotining real cells). The upfront `--qc` floor
    // is the only cell filter; bge fits on every cell that passes it.

    // The SIMBA-style co-embedding and the cluster-seeded ETM share ONE Leiden
    // clustering of the QC-kept cell embedding: the co-embed uses its median
    // cluster size as the temperature target, ETM uses the labels as topics —
    // so the embedding is clustered a single time. The co-embed re-embeds every
    // feature onto the cell manifold (gene = softmax-over-cells weighted average
    // of cell embeddings) and OVERRIDES {out}.feature_embedding.parquet (the raw
    // off-manifold ρ is not written). Cells are SIMBA's reference and are
    // unchanged. Post-hoc only — training (pseudobulk efficiency, phase-2
    // projection) is untouched.
    let cpu = candle_core::Device::Cpu;
    let e_feat_cpu = out.model.e_feat.to_device(&cpu)?; // [D, H] raw ρ
    let e_cell_cpu = match qc_keep_idx.as_deref() {
        Some(keep) => {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx_t = candle_core::Tensor::from_vec(idx, keep.len(), &cpu)?;
            out.model.e_cell.to_device(&cpu)?.index_select(&idx_t, 0)?
        }
        None => out.model.e_cell.to_device(&cpu)?,
    };
    let (cell_labels, target_eff) = ge::cell_clusters(&e_cell_cpu, args.num_topics)?;

    // Rows the null QC dropped, re-embedded post-hoc against the frozen pseudobulk
    // side (bge never subsets by HVG, so this is empty without a `--feature-null-fdr`
    // refit). They join the co-embed — which is what `annotate-by-projection` reads —
    // so a dropped gene still gets a manifold coordinate. They deliberately do NOT
    // join `out.model.e_feat`, so ETM keeps factorizing exactly the trained rows.
    let proj = out
        .feature_projection
        .as_ref()
        .filter(|p| !p.gene_ids.is_empty());
    let (coembed_feat, coembed_names) = match proj {
        Some(p) => {
            let h = out.model.embedding_dim;
            let mut flat: Vec<f32> = e_feat_cpu.flatten_all()?.to_vec1()?;
            flat.extend_from_slice(&p.beta);
            let mut names = unified.feature_names.clone();
            names.extend(
                p.gene_ids
                    .iter()
                    .map(|&g| backend_feature_names[g as usize].clone()),
            );
            info!(
                "restoring {} null-dropped feature(s) into the co-embed \
                 (frame calibration on {} trained: cosine {:.3}, norm ratio {:.3}, R² {:.3})",
                p.gene_ids.len(),
                p.calib.n_trained,
                p.calib.mean_cosine,
                p.calib.norm_ratio,
                p.calib.r2,
            );
            (
                candle_core::Tensor::from_vec(flat, (names.len(), h), &cpu)?,
                names,
            )
        }
        None => (e_feat_cpu.clone(), unified.feature_names.clone()),
    };
    ge::write_feature_coembedding(
        &args.out,
        &e_cell_cpu,
        &coembed_feat,
        &coembed_names,
        target_eff,
    )?;

    // Output layout depends on whether ETM was resolved (default on; --skip-etm
    // disables):
    //   skipped  → bge embeddings (latent = cell embedding Z, dictionary = ρ).
    //   resolved → ETM topic-model layout (latent = log θ, dictionary = β); the
    //         co-embedded feature_embedding is written above for both paths.
    let resolve_etm = !args.skip_etm;
    if resolve_etm {
        resolve_etm_topics(
            &out.model,
            &unified.feature_names,
            &unified.barcodes,
            args,
            qc_keep_idx.as_deref(),
            &cell_labels,
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

    let input: Vec<String> = data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::Bge,
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        // With ETM resolved the dictionary is β (gene × topic); otherwise it IS ρ.
        //
        // ρ does NOT go to feature_embedding.parquet — that file is always the SIMBA co-embed (see
        // below, and `write_feature_coembedding` above). ρ lives on the model's own axis, not on
        // the cell manifold, so putting it there would hand `annotate-by-projection` an
        // off-manifold gene table and make its Euclidean nearest-centroid call ill-posed.
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        // The SIMBA co-embed is written as feature_embedding.parquet in BOTH
        // the ETM and --skip-etm paths, so record it unconditionally (else a
        // skip-etm run's annotate-by-projection falls back to the raw-ρ
        // dictionary and ignores the co-embed file on disk).
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        // With ETM resolved `latent` is log θ (topic space); the H-space cell
        // embedding Z is written separately so annotate-by-projection finds it.
        // Without it, `latent` IS Z, so this stays None.
        cell_embedding_suffix: if resolve_etm {
            Some("cell_embedding.parquet")
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
