//! `senna bge` (Bipartite Graph Embedding) — thin clap + run-manifest
//! wrapper around the `graph-embedding-util` engine.
//!
//! Previously `senna gbe`; renamed to clarify that the bipartite (cell ×
//! feature) graph is *built* internally from expression counts. The
//! sibling `senna fne` (Feature Network Embedding) is the graph-input
//! companion that consumes an explicit feature-feature edge list.
//! `gbe` remains a clap alias for one release cycle.
//!
//! All algorithmic work lives in `graph_embedding_util`. This file resolves
//! the multiome layout, loads the data, and weights the HVG selection — the
//! parts of a fit that are bge's own — then hands off to [`driver`], which
//! trains and writes every output. `senna gem` (`gem::run`) shares that same
//! driver over its own (simpler) load.
//!
//! `--feature-network` (SGC smoothing of `E_feat` through a feature-feature
//! edge list) was removed along with its implementation: it saw no practical
//! use, and its six flags dominated bge's surface. `senna topic` /
//! `masked-topic` keep their own, unrelated feature-network restriction.

use crate::embed_common::*;
use data_beans_alg::hvg::select_hvg_streaming;
use graph_embedding_util as ge;

pub(crate) mod args;
pub(crate) mod driver;
mod multiome;
mod resolve_etm;
pub(crate) mod score;
pub(crate) mod transfer;

pub use args::BgeArgs;

/// Resolve the multiome layout, load the data, and weight the HVG selection —
/// everything about this fit that is bge's own — then hand off to the shared
/// [`driver::fit_embed_family`], which trains and writes every output
/// (`senna gem` hands off to the very same function).
pub fn fit_bge(args: &BgeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    anyhow::ensure!(
        args.pb_reference.is_none() || args.multiome.is_empty(),
        "a pb_reference and --multiome do not compose: multiome loads with union column \
         alignment, which gives no guarantee the carried pseudobulks stay contiguous at the \
         end — and their weights are applied by position. Pass --no-pb-reference for this \
         round and let it re-collapse."
    );
    // Multiome layout. `--multiome` declares it; otherwise it is read off the
    // inputs themselves — the feature axes say which files are the same assay,
    // and the barcode lists say which cells are the same cell. Both routes
    // land on one `MultiomePlan`, so nothing below has two shapes to handle.
    let multiome_plan: Option<ge::MultiomePlan> = if args.multiome.is_empty() {
        multiome::auto_plan(
            &args.data_files,
            args.batch_files.as_ref().map_or(0, Vec::len),
            args.pb_reference.is_some(),
        )?
    } else {
        multiome::declared_plan(&args.multiome)?
    };
    if let Some(p) = multiome_plan.as_ref() {
        multiome::log_plan(p, !args.multiome.is_empty());
    }
    let is_multiome = multiome_plan.is_some();

    // Multiome mixes gene rows (RNA) and locus rows (ATAC peaks) on one axis,
    // so canonicalize per-name via `Mixed` (genes → gene rule, `chrX:s-e` →
    // locus rule).
    //
    // The rule is fixed rather than exposed: `--feature-name-delim` /
    // `--feature-name-exact` were CLI knobs whose defaults ('_', fuzzy) were the
    // only settings anyone used, and `_` is the separator every loader in this
    // workspace already writes (`ENSG…_TSPAN6`). `senna gem` and `senna predict`
    // still expose their own overrides where query-vs-reference name bridging is
    // an actual concern.
    let feature_kind = if is_multiome {
        ge::FeatureNameKind::Mixed
    } else {
        ge::FeatureNameKind::Gene { delim: '_' }
    };

    // Under a plan the files are re-ordered so each sample group is contiguous
    // — the order `validate_multiome_groups` reads `group_sizes` against.
    let data_files: Vec<Box<str>> = match multiome_plan.as_ref() {
        Some(p) => p.files.clone(),
        None => {
            anyhow::ensure!(
                !args.data_files.is_empty(),
                "no input files: pass the count matrices positionally. Modality \n\
                 groups are detected from the data; declare them by hand with \n\
                 `--multiome rna.zarr,atac.zarr [--multiome rna2.zarr,atac2.zarr ...]` \n\
                 when the feature axes overlap (spliced vs unspliced, say)."
            );
            args.data_files.clone()
        }
    };
    let feature_suffix: Option<Vec<Box<str>>> = multiome_plan.as_ref().map(|p| p.modality.clone());
    let barcode_suffix: Option<Vec<Option<Box<str>>>> = multiome_plan
        .as_ref()
        .and_then(ge::MultiomePlan::barcode_suffix);

    let effective_hvg =
        crate::hvg::resolve_multiome_with_hvg(is_multiome, data_files.len(), &args.hvg);
    // From the plan, never from the HVG resolver: that one also clears its
    // multiome flag on file count, and a plan that says Union while the load
    // says Disjoint is a manifest whose replay cannot reproduce the run.
    let column_alignment = if is_multiome {
        data_beans::sparse_io_vector::ColumnAlignment::Union
    } else {
        data_beans::sparse_io_vector::ColumnAlignment::Disjoint
    };

    let batch_files = crate::senna_input::effective_batch_files(
        args.collapse.ignore_batch,
        args.batch_files.as_deref(),
    );

    let mut unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: data_files.clone(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(feature_kind.clone()),
        preload: args.preload_data,
        column_alignment,
        per_file_feature_suffix: feature_suffix,
        // Only set when a detected layout has several sample groups: raw 10x
        // barcodes collide across samples, and Union loading would fold two
        // donors' cells into one.
        per_file_barcode_suffix: barcode_suffix,
        ..Default::default()
    })?;

    // Carried pseudobulks: registered exactly like every other family — their
    // cell counts become column multiplicities on the count backend (weights
    // keyed on the PBREF_-tail layout, verified by name), so the collapse and
    // every downstream statistic treat a carried column as the cells it
    // stands for. The CELL axis of phase 1 sees them unweighted, as ~1k
    // prototype profiles among the new cells — a documented approximation;
    // the pb axis carries the properly weighted signal.
    if let Some(r) = args.pb_reference.as_ref() {
        let v = unified.count_backend_mut();
        let names = v.column_names()?;
        let w = crate::pb_reference::weights_for(&r.cell_counts, &names)?;
        v.register_column_multiplicity(&w)?;
        info!(
            "Column multiplicity: {} carried pseudobulks stand for {} cells",
            r.cell_counts.len(),
            r.cells_represented() as usize,
        );
    }

    // Guard barcode identity across groups, so Union loading never merges cells
    // from different samples. A detected layout already tags barcodes by group,
    // which makes this a no-op; a declared one relies on it.
    if let Some(p) = multiome_plan.as_ref() {
        ge::validate_multiome_groups(&p.group_sizes, &unified.barcodes, &unified.cell_modality)?;
    }

    // HVG → projection weights (no longer subsets the feature axis).
    // Mirrors senna topic: HVG down-weights uninformative genes for the
    // random projection / pb sketching only; collapse + supergene
    // coarsening + training read all genes. The driver subsets this full-axis
    // vector through `feature_to_backend_row` for the live feature axis.
    //
    // `--must-train-features` is a curated panel kept in the HVG-weighted set.
    let hvg_enabled = effective_hvg.selection_on();
    let must_train = crate::hvg::load_must_train(effective_hvg.must_train_file, hvg_enabled)?;
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

    let run_multiome = multiome_plan
        .as_ref()
        .map(crate::multiome_layout::RunMultiome::from_plan);

    let (preset_features, carried) = crate::feature_preset::resolve_preset(
        args.feature_embedding.resolve()?,
        &unified.feature_names,
        &feature_kind,
    )?;
    let embedding_dim =
        crate::feature_preset::resolve_dim(args.embedding_dim, preset_features.as_ref())?;

    driver::fit_embed_family(driver::EmbedPlan {
        kind: crate::run_manifest::RunKind::Bge,
        knobs: args.knobs(embedding_dim),
        unified,
        data_files,
        multiome: run_multiome,
        hvg_weights: hvg_full,
        tracks: None,
        offset_l2: 0.0,
        preset_features,
        carried,
        pb_reference: args.pb_reference.as_ref(),
        init_from: args.init_from.as_deref(),
        train_args: crate::run_manifest::record_train_args(args)?,
        after_fit: None,
    })
}
