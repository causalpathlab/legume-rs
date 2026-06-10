//! Entry point for `faba gem` (alias `gem-embedding`).

use anyhow::Context;
use data_beans::qc::collect_column_stat_across_vec;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::{load_unified_data, FeatureNameKind};
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::RunningStatOps;

use faba::gem::args::GemArgs;
use faba::gem::feature_table::FeatureTable;
use faba::gem::manifest::write_outputs;
use faba::gem::model::GemModel;
use faba::gem::pseudobulk::build_pseudobulk;
use faba::gem::region::{load_component_annotations, ComponentAnnotation, RegionMap};
use faba::gem::train::train;

pub fn run_gem_embedding(args: &GemArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    validate_args(args)?;

    // Collect modality files in faba's conventional order: count first,
    // then m6A, A2I, pA. Order seeds modality_id assignment in the
    // FeatureTable (slot 0 is forced to "count" regardless). Each flag
    // accepts a comma-separated list of prefixes; all are stacked under
    // Union (cells merged by barcode, batches resolved from the barcodes'
    // `@batch` tags or a single --batch-files file). Modality is inferred
    // from the row name, not the flag, so multiple files per flag are fine.
    let mut data_files: Vec<Box<str>> = Vec::new();
    data_files.extend(args.genes.iter().cloned());
    if let Some(ps) = args.dartseq.as_ref() {
        data_files.extend(ps.iter().cloned());
    }
    if let Some(ps) = args.atoi.as_ref() {
        data_files.extend(ps.iter().cloned());
    }
    if let Some(ps) = args.apa.as_ref() {
        data_files.extend(ps.iter().cloned());
    }

    let batch_files: Option<&[Box<str>]> = if args.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };

    let feature_kind = if args.feature_name_exact {
        FeatureNameKind::Exact
    } else {
        FeatureNameKind::Gene {
            delim: args.feature_name_delim,
        }
    };
    info!(
        "loading {} modality file(s) under Union column alignment \
         (feature_kind={:?}, preload={})",
        data_files.len(),
        feature_kind,
        args.preload_data
    );
    let mut unified = load_unified_data(
        &data_files,
        batch_files,
        None,
        feature_kind,
        args.preload_data,
        ColumnAlignment::Union,
    )
    .context("load_unified_data")?;

    // `--ignore-batch` semantics: under Union with multiple modality
    // files, `load_unified_data` defaults to auto_modality_batch (one
    // batch per cell-modality-presence pattern). That contradicts
    // --ignore-batch's "one batch" promise — explicitly collapse here.
    if args.ignore_batch {
        let n = unified.n_cells();
        unified.batch_membership = vec![0_u32; n];
        unified.batch_names = vec!["all".into()];
    }

    info!(
        "unified {} features × {} cells × {} batches",
        unified.n_features(),
        unified.n_cells(),
        unified.n_batches()
    );

    // ---- Cell QC: inclusive, modality-agnostic, no data rewritten ----
    // Per-cell detected-feature count over ALL modalities (count +
    // m6A/A2I/APA), via the data-beans column-stat infra. Cells below
    // `--min-cell-nnz` are near-empty — typically a barcode seen in one
    // sparse modality with a single read and no counts; the count-anchored
    // phase-2 projection maps these to ~0. We keep them through pb collapse
    // and feature training (their signal still counts) but drop them from
    // the per-cell outputs (a write-time selection, NOT a squeeze).
    let cell_nnz = collect_column_stat_across_vec(&unified.per_file_data[0], None, None)
        .context("cell QC column statistics")?
        .count_positives();
    let keep_idx: Vec<usize> = (0..unified.n_cells())
        .filter(|&c| (cell_nnz[c] as usize) >= args.min_cell_nnz)
        .collect();
    info!(
        "cell QC: {} / {} cells pass --min-cell-nnz {} ({} dropped as near-empty)",
        keep_idx.len(),
        unified.n_cells(),
        args.min_cell_nnz,
        unified.n_cells() - keep_idx.len()
    );

    // Load component annotations (region binning). Modality labels must
    // match the modifier row names emitted by faba m6a/atoi/apa.
    let region_map = build_region_map(args)?;
    let table = FeatureTable::build(&unified.feature_names, &region_map);
    info!(
        "feature_table: {} genes, {} modalities, {} regions, {} count-comp + {} modifier-comp rows",
        table.n_genes(),
        table.n_modalities(),
        table.n_regions,
        table.count_comp_rows.len(),
        table.modifier_comp_rows.len(),
    );
    anyhow::ensure!(
        table.n_genes() > 0,
        "no genes parsed from feature axis — check row name convention (`gene/modality/detail`)"
    );

    let n_cells = unified.n_cells();
    let pb = build_pseudobulk(&mut unified, &table, args).context("build pseudobulk")?;

    // Persist the per-gene NB-Fisher housekeeping weights (senna
    // convention: `{out}.fisher_weights.parquet`) so the suppression is
    // inspectable. Skipped when the penalty is disabled.
    if args.housekeeping_penalty > 0.0 {
        data_beans_alg::gene_weighting::save_fisher_weights(
            &args.out,
            &pb.gene_fisher_weights,
            &table.gene_names,
        )
        .context("save fisher weights")?;
        info!(
            "wrote {}.fisher_weights.parquet ({} genes)",
            args.out,
            pb.gene_fisher_weights.len()
        );
    }

    // Persist per-gene ubiquity (fraction of cells expressing) — a
    // diagnostic / inverse-propensity signal (breadth complement to the
    // NB-Fisher magnitude weight), inspectable next to the Fisher weights.
    data_beans_alg::gene_weighting::save_per_gene_weights(
        &pb.gene_ubiquity,
        &table.gene_names,
        &format!("{}.ubiquity.parquet", args.out),
    )
    .context("save gene ubiquity")?;
    info!(
        "wrote {}.ubiquity.parquet ({} genes)",
        args.out,
        pb.gene_ubiquity.len()
    );

    let n_pbs_per_level: Vec<usize> = pb.pb_pools_per_level.iter().map(|l| l.n_units).collect();

    let dev = args
        .device
        .to_device(args.device_no)
        .context("candle device init")?;
    info!("compute device = {:?}", dev);
    let mut model = GemModel::new(
        table.n_genes(),
        table.n_modalities(),
        args.n_programs,
        table.n_regions,
        args.embedding_dim,
        n_cells,
        &n_pbs_per_level,
        &dev,
    )
    .context("init model")?;

    let stop = setup_stop_handler();
    train(args, &table, &pb, &mut model, &stop).context("training loop")?;

    // Durable embeddings first, so a force-abort (second Ctrl+C) during the
    // optional topic step never loses the trained model.
    write_outputs(&args.out, &table, &pb, &model, &unified, &keep_idx).context("write outputs")?;

    // Archetype-based topics from the (possibly interrupted) embedding.
    // Runs after the embeddings are on disk; the manifest below then
    // references them. When training was Ctrl+C-stopped this still reports
    // topics, skipping the K-sweep.
    let topics = if args.resolve_topics {
        Some(
            faba::gem::topics::resolve_topics(
                &args.out, &model, &table, &unified, args, &stop, &keep_idx,
            )
            .context("resolve topics")?,
        )
    } else {
        None
    };

    // Manifest last, so it records the resolved-topic artifacts. n_cells
    // reflects the QC-passed cells actually written to the per-cell outputs.
    faba::gem::manifest::write_manifest(&args.out, &model, keep_idx.len(), topics.as_ref())
        .context("write manifest")?;

    info!("done — prefix '{}'", args.out);
    Ok(())
}

/// Load any supplied `*_components.parquet` sidecars and build the
/// transcript-position `RegionMap`. Each sidecar is tagged with the
/// modality label that matches its modifier row names (`m6A`, `A2I`,
/// `pA`). With no sidecars the map is empty and every satellite falls
/// back to region 0 (γ collapses to one per-modality offset).
fn build_region_map(args: &GemArgs) -> anyhow::Result<RegionMap> {
    let sidecars: [(&Option<Box<str>>, &str); 3] = [
        (&args.dartseq_components, "m6A"),
        (&args.atoi_components, "A2I"),
        (&args.apa_components, "pA"),
    ];
    let mut records: Vec<ComponentAnnotation> = Vec::new();
    for (path, modality) in sidecars {
        if let Some(path) = path.as_ref() {
            let recs = load_component_annotations(path, modality)
                .with_context(|| format!("loading {modality} component annotations from {path}"))?;
            info!(
                "region: {} {} component annotations from {}",
                recs.len(),
                modality,
                path
            );
            records.extend(recs);
        }
    }
    Ok(RegionMap::from_records(&records, args.n_regions))
}

/// Argument-level sanity checks before any I/O or training. Surfaces
/// configuration mistakes (zero-dim model, NaN/out-of-range tempering,
/// stratum-fraction overflow) as a clear `anyhow::Error` rather than a
/// candle panic or silently zero-loss training.
fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.embedding_dim
    );
    anyhow::ensure!(
        args.n_programs > 0,
        "--num-programs must be > 0 (got {})",
        args.n_programs
    );
    anyhow::ensure!(
        args.n_regions > 0,
        "--num-regions must be > 0 (got {})",
        args.n_regions
    );
    anyhow::ensure!(
        args.tau.is_finite() && (0.0..=1.0).contains(&args.tau),
        "--tau must be a finite value in [0, 1] (got {})",
        args.tau
    );
    anyhow::ensure!(
        args.tau_modality.is_finite() && (0.0..=1.0).contains(&args.tau_modality),
        "--tau-modality must be a finite value in [0, 1] (got {})",
        args.tau_modality
    );
    anyhow::ensure!(
        args.housekeeping_penalty.is_finite() && args.housekeeping_penalty >= 0.0,
        "--housekeeping-penalty must be a finite value ≥ 0 (got {})",
        args.housekeeping_penalty
    );
    if args.resolve_topics {
        // --no-cell-axis leaves e_cell at its random init (never trained),
        // so resolving topics from it would yield archetypes of noise.
        anyhow::ensure!(
            !args.no_cell_axis,
            "--resolve-topics requires a trained cell embedding, but --no-cell-axis \
             leaves e_cell at its random init; drop one of the two flags"
        );
        if let Some(k) = args.num_topics {
            anyhow::ensure!(k >= 2, "--num-topics must be ≥ 2 (got {})", k);
        } else {
            anyhow::ensure!(
                args.max_k >= 2,
                "--max-k must be ≥ 2 for the topic auto-sweep (got {})",
                args.max_k
            );
        }
        anyhow::ensure!(
            args.aa_iters > 0,
            "--aa-iters must be > 0 (got {})",
            args.aa_iters
        );
    }
    anyhow::ensure!(
        args.f_agg.is_finite() && (0.0..=1.0).contains(&args.f_agg),
        "--f-agg must be in [0, 1] (got {})",
        args.f_agg
    );
    anyhow::ensure!(
        args.f_count.is_finite() && (0.0..=1.0).contains(&args.f_count),
        "--f-count must be in [0, 1] (got {})",
        args.f_count
    );
    anyhow::ensure!(
        args.f_agg + args.f_count <= 1.0,
        "--f-agg + --f-count must be ≤ 1.0 (got {} + {} = {}); modifier stratum gets the remainder",
        args.f_agg,
        args.f_count,
        args.f_agg + args.f_count
    );
    // At least one negative-source must be active or training is a no-op:
    // log_softmax over a single positive column is identically zero.
    anyhow::ensure!(
        args.n_rand + args.n_swap_gene_mode > 0,
        "at least one of --n-rand, --n-swap-gene-mode must be > 0 \
         (else NCE loss collapses to zero and AdamW makes no progress)"
    );
    Ok(())
}
