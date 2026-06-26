//! Entry point for `faba gem` (alias `gem-embedding`).

use anyhow::Context;
use data_beans::qc::{collect_column_stat_across_vec, collect_row_stat_across_vec};
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::data::UnifiedData;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::{load_unified_data, FeatureNameKind, LoadUnifiedArgs};
use log::{info, warn};
use matrix_util::common_io::{basename, mkdir_parent};
use matrix_util::traits::RunningStatOps;
use matrix_util::utils::median;
use rayon::ThreadPoolBuilder;
use rustc_hash::{FxHashMap, FxHashSet};

use faba::gem::args::GemArgs;
use faba::gem::cell_solve::{CellStreamCtx, SatStream};
use faba::gem::feature_table::{
    classify, parse_component_idx, parse_feature_name, BackendRowMap, FeatureTable, RowStratum,
};
use faba::gem::manifest::{write_outputs, CellQcOutputs, OutputCtx};
use faba::gem::model::{GemDims, GemModel};
use faba::gem::pseudobulk::{
    build_pseudobulk, spliced_backend_rows, PseudobulkData, SatelliteData,
};
use faba::gem::region::RegionMap;
use faba::gem::sample_id::{
    file_sample_id, infer_satellite_strip, longest_common_underscore_suffix,
    longest_shared_underscore_prefix, strip_sample_id,
};
use faba::gem::train::train;
use graph_embedding_util::feature_qc::hvg_feature_qc;

/// A loaded satellite-modality backend (m6A / A2I / pA), held **separately**
/// from the genes backend. The collapse never sees it; it only donates
/// `modifier_comp` mass at aggregation time, matched to genes cells by
/// `barcode@sample`. Triplets are materialized once at load.
struct SatelliteBackend {
    /// Modality label for logging (`m6A` / `A2I` / `pA`); the actual modality
    /// id is resolved from row names by the `FeatureTable`.
    label: Box<str>,
    unified: UnifiedData,
}

/// Per-satellite `col → genes cell` map (matched by `barcode@sample`) plus the
/// satellite's row classification against the global table. Owns the `Vec`s
/// that [`SatelliteData`] borrows.
struct SatelliteLink {
    row_map: BackendRowMap,
    col_to_genes_cell: Vec<Option<usize>>,
}

/// Build `col_to_genes_cell` for every satellite against a given set of genes
/// cell barcodes (`barcode@sample`). Recomputed in the refine pass after the
/// genes cells are subset. Logs how many satellite columns matched.
fn link_satellites(
    satellites: &[SatelliteBackend],
    sat_row_maps: Vec<BackendRowMap>,
    genes_barcodes: &[Box<str>],
) -> Vec<SatelliteLink> {
    let bc_to_cell: FxHashMap<&str, usize> = genes_barcodes
        .iter()
        .enumerate()
        .map(|(c, b)| (b.as_ref(), c))
        .collect();
    satellites
        .iter()
        .zip(sat_row_maps)
        .map(|(s, row_map)| {
            let col_to_genes_cell: Vec<Option<usize>> = s
                .unified
                .barcodes
                .iter()
                .map(|b| bc_to_cell.get(b.as_ref()).copied())
                .collect();
            let total = col_to_genes_cell.len();
            let matched = col_to_genes_cell.iter().filter(|c| c.is_some()).count();
            if matched == 0 {
                // Almost always a sample-tag mismatch: the satellite's
                // `barcode@sample` names don't coincide with any genes cell, so
                // the whole modality is silently dropped. Surface it loudly.
                warn!(
                    "satellite {}: 0 / {total} columns matched a genes cell — this modality \n\
                     donates NOTHING to the model. Its `barcode@sample` names don't line up with \n\
                     the genes cells; check that the genes and {} files reduce to the SAME sample \n\
                     id (set the per-flag --*-sample-strip suffixes so both strip to e.g. `rep1_wt`).",
                    s.label, s.label,
                );
            } else {
                info!(
                    "satellite {}: {matched} / {total} columns matched a genes cell ({} unmatched, donate nothing)",
                    s.label,
                    total - matched,
                );
            }
            SatelliteLink {
                row_map,
                col_to_genes_cell,
            }
        })
        .collect()
}

/// Assemble the streaming phase-2 context: the genes backend + row map, and
/// per-satellite `SatStream`s carrying the inverse `genes_cell → sat_cols`
/// index (so a chunk of genes cells can gather its satellite mass). Valid only
/// when cell id == backend column (after the up-front mask+subset, i.e. the
/// non-refine pass).
fn build_cell_stream_ctx<'a>(
    unified: &'a UnifiedData,
    genes_row_map: &'a BackendRowMap,
    sats: &'a [SatelliteBackend],
    sat_links: &'a [SatelliteLink],
    // Model cell id → genes-backend column. `(0..n_cells)` in the main pass;
    // `live_cell_old_ids` in refine (backend keeps its N_old layout). Indexed
    // by model cell id, matching how `col_to_genes_cell` is keyed.
    cell_columns: Vec<usize>,
) -> CellStreamCtx<'a> {
    let n_cells = cell_columns.len();
    let satellites = sats
        .iter()
        .zip(sat_links)
        .map(|(s, link)| {
            let mut inv: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
            for (sat_col, &gc) in link.col_to_genes_cell.iter().enumerate() {
                if let Some(gc) = gc {
                    if gc < n_cells {
                        inv[gc].push(sat_col as u32);
                    }
                }
            }
            SatStream {
                backend: s.unified.count_backend(),
                row_map: &link.row_map,
                feature_to_backend_row: &s.unified.feature_to_backend_row,
                genes_cell_to_sat_cols: inv,
            }
        })
        .collect();
    CellStreamCtx {
        genes_backend: unified.count_backend(),
        genes_row_map,
        feature_to_backend_row: &unified.feature_to_backend_row,
        satellites,
        cell_columns,
    }
}

/// Borrow `SatelliteData` views from the owned backends + links, ready to pass
/// into `build_pseudobulk`.
fn satellite_views<'a>(
    satellites: &'a [SatelliteBackend],
    links: &'a [SatelliteLink],
) -> Vec<SatelliteData<'a>> {
    satellites
        .iter()
        .zip(links)
        .map(|(s, l)| SatelliteData {
            unified: &s.unified,
            row_map: &l.row_map,
            col_to_genes_cell: &l.col_to_genes_cell,
        })
        .collect()
}

/// Load one modality's files into its own `UnifiedData` under Union alignment.
/// `do_tag` tags each file's barcodes with its `@sample` id (basename with the
/// per-flag suffix stripped) so samples stay distinct **and** a cell matches
/// across modalities. `batch_files` is only passed for the genes backend.
fn load_modality(
    files: &[Box<str>],
    strip: &str,
    do_tag: bool,
    batch_files: Option<&[Box<str>]>,
    feature_kind: FeatureNameKind,
    preload: bool,
) -> anyhow::Result<UnifiedData> {
    let mut data_files: Vec<Box<str>> = Vec::with_capacity(files.len());
    let mut sample_ids: Vec<Box<str>> = Vec::with_capacity(files.len());
    for f in files {
        sample_ids.push(file_sample_id(f, strip)?);
        data_files.push(f.clone());
    }
    let per_file_barcode_suffix: Option<Vec<Option<Box<str>>>> = if do_tag {
        Some(sample_ids.into_iter().map(Some).collect())
    } else {
        None
    };
    load_unified_data(LoadUnifiedArgs {
        data_files,
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(feature_kind),
        preload,
        column_alignment: ColumnAlignment::Union,
        per_file_barcode_suffix,
        ..Default::default()
    })
}

pub fn run_gem_embedding(args: &GemArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    validate_args(args)?;

    if !args.refine() && (args.qc.min_cells > 1 || args.qc.feature_qc_enabled()) {
        warn!(
            "gene QC (--min-cells {}, feature QC enabled = {}) ignored under --skip-refine: both \n\
             need the refine pass to measure support / β on real (not ambient) cells; no gene \n\
             QC will run",
            args.qc.min_cells,
            args.qc.feature_qc_enabled()
        );
    }

    let n_threads = if args.runtime.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        args.runtime.threads
    };
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // ignore error if pool already initialised
    info!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    // Separate per-modality backends. The `--genes` files form the primary
    // `unified` (owns the cell axis, `e_cell`, QC, outputs) and the 6
    // gene-sample batches that drive the collapse. Each satellite flag
    // (`--dartseq`/`--atoi`/`--apa`) loads into its own backend and only
    // donates `modifier_comp` mass, matched to genes cells by `barcode@sample`.
    // Modality is inferred from the row name, not the flag.
    let feature_kind = if args.collapse.feature_name_exact {
        FeatureNameKind::Exact
    } else {
        FeatureNameKind::Gene {
            delim: args.collapse.feature_name_delim,
        }
    };

    let batch_files: Option<&[Box<str>]> = if args.collapse.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };

    // Tag barcodes with the per-file `@sample` id whenever there is
    // more than one input file and the caller hasn't supplied
    // explicit batch identity.  The SAME rule applies to genes and
    // satellites so a cell's columns carry identical names across
    // modalities (the basis for matching). With `--batch-files` the
    // caller owns batch identity, so leave barcodes untagged.
    let total_files = args.genes.len()
        + args.dartseq.as_deref().map_or(0, <[_]>::len)
        + args.atoi.as_deref().map_or(0, <[_]>::len)
        + args.apa.as_deref().map_or(0, <[_]>::len);
    let do_tag = batch_files.is_none() && total_files > 1;
    if do_tag {
        info!("tagging barcodes with per-file @sample id (stripped basename) for cross-modality matching");
    }

    // Basenames per modality, computed once and reused for all strip
    // inference below (avoids re-deriving them inside each helper).
    let basenames_of = |files: Option<&[Box<str>]>| -> anyhow::Result<Vec<Box<str>>> {
        files.unwrap_or(&[]).iter().map(|f| basename(f)).collect()
    };
    let genes_bn = basenames_of(Some(&args.genes))?;
    let dartseq_bn = basenames_of(args.dartseq.as_deref())?;
    let atoi_bn = basenames_of(args.atoi.as_deref())?;
    let apa_bn = basenames_of(args.apa.as_deref())?;
    // All satellite basenames, used to recover the genes sample id in the
    // single-genes-file case below.
    let sat_basenames: Vec<Box<str>> = dartseq_bn
        .iter()
        .chain(&atoi_bn)
        .chain(&apa_bn)
        .cloned()
        .collect();

    // Resolve per-modality strip suffixes (only relevant when tagging).
    // genes: ≥2 files → LCS at `_` across genes basenames; exactly 1 file →
    // the `_`-prefix it shares with the satellites (so a single genes file can
    // still be matched). Satellites: LCS first, else per-file LCP against the
    // genes sample-id set, hard-erroring if nothing lines up — that's the
    // silent-modality-drop bug surfaced. Explicit user strips always win.
    let genes_strip: Box<str> = if !args.collapse.genes_sample_strip.is_empty() {
        args.collapse.genes_sample_strip.clone()
    } else if !do_tag {
        "".into()
    } else if genes_bn.len() >= 2 {
        let s = longest_common_underscore_suffix(&genes_bn);
        if !s.is_empty() {
            info!("auto-strip: --genes-sample-strip = {:?}", s.as_ref());
        }
        s
    } else if !sat_basenames.is_empty() {
        // Single genes file + satellites: the sample id is the longest
        // `_`-prefix the genes basename shares with every satellite; the tail
        // after it is the strip (e.g. `rep1_wt_genes` vs `rep1_wt_m6a_mixture`
        // → strip `_genes`). No shared prefix → leave unstripped and let the
        // satellite call raise the clear no-overlap error.
        let genes_bn0 = genes_bn[0].as_ref();
        let s: Box<str> = match longest_shared_underscore_prefix(genes_bn0, &sat_basenames) {
            Some(prefix) if prefix.len() < genes_bn0.len() => genes_bn0[prefix.len()..].into(),
            _ => "".into(),
        };
        if !s.is_empty() {
            info!("auto-strip: --genes-sample-strip = {:?}", s.as_ref());
        }
        s
    } else {
        "".into()
    };
    let genes_ids: FxHashSet<Box<str>> = genes_bn
        .iter()
        .map(|b| strip_sample_id(b.as_ref(), &genes_strip))
        .collect();

    let resolve_sat =
        |bn: &[Box<str>], user_strip: &str, label: &str| -> anyhow::Result<Box<str>> {
            if bn.is_empty() || !do_tag || !user_strip.is_empty() {
                return Ok(user_strip.into());
            }
            let s = infer_satellite_strip(bn, &genes_ids, label)?;
            info!("auto-strip: --{label}-sample-strip = {:?}", s.as_ref());
            Ok(s)
        };
    let dartseq_strip = resolve_sat(&dartseq_bn, &args.collapse.dartseq_sample_strip, "dartseq")?;
    let atoi_strip = resolve_sat(&atoi_bn, &args.collapse.atoi_sample_strip, "atoi")?;
    let apa_strip = resolve_sat(&apa_bn, &args.collapse.apa_sample_strip, "apa")?;

    info!(
        "loading {} genes file(s) (feature_kind={:?}, preload={})",
        args.genes.len(),
        feature_kind,
        args.runtime.preload_data
    );
    let mut unified = load_modality(
        &args.genes,
        &genes_strip,
        do_tag,
        batch_files,
        feature_kind.clone(),
        args.runtime.preload_data,
    )
    .context("load genes backend")?;

    if args.collapse.ignore_batch {
        let n = unified.n_cells();
        unified.batch_membership = vec![0_u32; n];
        unified.batch_names = vec!["all".into()];
    }

    info!(
        "genes backend: {} features × {} cells × {} batches",
        unified.n_features(),
        unified.n_cells(),
        unified.n_batches()
    );

    // Load satellite modalities, each into its own backend (triplets
    // materialized for aggregation). Satellites are never batched/collapsed.
    let satellite_specs: [(&[Box<str>], &str, &str); 3] = [
        (
            args.dartseq.as_deref().unwrap_or(&[]),
            dartseq_strip.as_ref(),
            "m6A",
        ),
        (
            args.atoi.as_deref().unwrap_or(&[]),
            atoi_strip.as_ref(),
            "A2I",
        ),
        (args.apa.as_deref().unwrap_or(&[]), apa_strip.as_ref(), "pA"),
    ];
    let mut satellites: Vec<SatelliteBackend> = Vec::new();
    for (files, strip, label) in satellite_specs {
        if files.is_empty() {
            continue;
        }
        let mut sat = load_modality(
            files,
            strip,
            do_tag,
            None,
            feature_kind.clone(),
            args.runtime.preload_data,
        )
        .with_context(|| format!("load {label} backend"))?;
        sat.materialize_cell_triplets()
            .with_context(|| format!("materialize {label} triplets"))?;
        info!(
            "{} backend: {} features × {} cells ({} triplets)",
            label,
            sat.n_features(),
            sat.n_cells(),
            sat.triplets.len(),
        );
        satellites.push(SatelliteBackend {
            label: label.into(),
            unified: sat,
        });
    }

    // Global feature table spanning genes + satellites (by name, so ids are
    // joint). The genes backend defines the gene namespace; satellites only
    // augment existing genes with modifier modalities.
    //
    // First pass is **genes-only** when refining: the QC β embedding is trained
    // on the core count signal alone, with the sparse epi-modification
    // satellites (m6A / A-to-I / pA) held out until the cells and genes have
    // been cleaned. Full modelling over every modality then runs in pass 2 on
    // the QC-passed subset (see `run_refine_pass`). Without `--refine` the lone
    // pass uses all modalities. `satellites` stays alive either way for pass 2.
    // Resolve `n_regions` once: explicit arg if given, else
    // `max(component_idx) + 1` scanned from satellite row names so each
    // component gets its own γ slot. `RegionMap` then keys per-component
    // (clamped to n_regions − 1).
    let n_regions: usize = match args.model.n_regions {
        Some(n) => n,
        None => {
            let n = infer_n_regions(&satellites, 1);
            info!("auto-regions: --num-regions = {n} (max component_idx + 1)");
            n
        }
    };
    let region_map = RegionMap::new(n_regions);
    let pass1_sats: &[SatelliteBackend] = if args.refine() { &[] } else { &satellites };
    let sat_name_lists: Vec<&[Box<str>]> = pass1_sats
        .iter()
        .map(|s| s.unified.feature_names.as_slice())
        .collect();
    let table = FeatureTable::build_layered(&unified.feature_names, &sat_name_lists, &region_map);
    // In a refine run the satellites are intentionally held out of pass 1, so
    // the modifier-comp count is trivially 0 — don't print a bare "0" that reads
    // as "no modifier rows found". Note the deferral instead; the full count
    // surfaces in the pass-2 table.
    let modifier_summary = if args.refine() && !satellites.is_empty() {
        format!(
            "{} satellite {} deferred to refine pass 2",
            satellites.len(),
            if satellites.len() == 1 {
                "modality"
            } else {
                "modalities"
            },
        )
    } else {
        format!("{} modifier-comp rows", table.modifier_comp_rows.len())
    };
    info!(
        "feature_table: {} genes, {} modalities, {} regions, {} count-comp rows + {}",
        table.n_genes(),
        table.n_modalities(),
        table.n_regions,
        table.count_comp_rows.len(),
        modifier_summary,
    );
    anyhow::ensure!(
        table.n_genes() > 0,
        "no genes parsed from feature axis — check row name convention (`gene/modality/detail`)"
    );

    // Genes-backend row map (feature axis; drives the spliced QC + collapse mask).
    let genes_row_map = table.map_backend_rows(&unified.feature_names, &region_map);

    /////////////////////////////////////////////////////////////
    // Cell QC over the SPLICED features (the collapse driver) //
    /////////////////////////////////////////////////////////////

    // The collapse places each cell from its spliced projection, so a cell with
    // zero spliced counts is a zero vector there — unplaceable. gem's upfront
    // cell gate is intentionally CONSERVATIVE: it drops only genuinely
    // zero-spliced (unplaceable) cells (cutoff = 1). There is no upfront bimodal
    // nnz cut — the real empty↔real decision is the refine pass's
    // embedding-norm empty-call (gem's two-step QC), which uses the model's own
    // learned signal rather than a 1-D nnz heuristic.
    let spliced_rows = spliced_backend_rows(&unified, &table, &genes_row_map);
    let cell_stat =
        collect_column_stat_across_vec(unified.count_backend(), Some(&spliced_rows), None)
            .context("cell QC column statistics (spliced)")?;
    let cell_nnz_full = cell_stat.count_positives();
    let n_cells_full = unified.n_cells();
    let cell_cutoff = 1usize;
    // Single keep predicate, shared by the backend mask and the logical subset
    // below — they MUST select the same cells (and both renumber survivors in
    // ascending order) for "logical cell i ≡ backend column i" to hold.
    let keep_mask: Vec<bool> = (0..n_cells_full)
        .map(|c| (cell_nnz_full[c] as usize) >= cell_cutoff)
        .collect();
    let kept: Vec<usize> = keep_mask
        .iter()
        .enumerate()
        .filter_map(|(c, &k)| k.then_some(c))
        .collect();
    anyhow::ensure!(
        !kept.is_empty(),
        "cell QC dropped every cell at spliced cutoff {cell_cutoff} — no cell has a \
         non-zero spliced count; check the inputs"
    );

    // Subset the genes backend to the called cells UP FRONT — `mask_columns`
    // shrinks the backend columns (the established senna / pinto cell-QC path),
    // `subset_cells` syncs the logical axis. The collapse + training then run on
    // real cells only (efficient; ambient never shapes the model).
    if kept.len() < n_cells_full {
        unified
            .count_backend_mut()
            .mask_columns(&keep_mask)
            .context("subset genes cells (mask_columns)")?;
        unified.subset_cells(&kept);
    }
    // Only surface this line when it actually filtered something — the refine
    // empty-call is what makes the real cut; an "kept N/N (dropped 0)" line is
    // pure noise. The dropped path keeps the conservative reminder.
    let n_dropped = n_cells_full - kept.len();
    if n_dropped > 0 {
        info!(
            "cell QC (spliced non-zeros): kept {} / {} cells at cutoff {} (dropped {}) \
             [conservative; refine empty-call makes the real cut]",
            kept.len(),
            n_cells_full,
            cell_cutoff,
            n_dropped,
        );
    }
    // After the up-front subset every remaining cell is written out.
    let keep_idx: Vec<usize> = (0..unified.n_cells()).collect();

    // Satellite row maps + cell links (against the SUBSET genes cells). Empty
    // in a refine pass-1 (genes-only); the full set is linked in pass 2.
    let sat_row_maps: Vec<BackendRowMap> = pass1_sats
        .iter()
        .map(|s| table.map_backend_rows(&s.unified.feature_names, &region_map))
        .collect();
    let sat_links = link_satellites(pass1_sats, sat_row_maps, &unified.barcodes);
    let sat_views = satellite_views(pass1_sats, &sat_links);

    let n_cells = unified.n_cells();
    // Build the per-cell pool only when the sampler draws the cell axis; the
    // default pure-pb path leaves it out and phase-2 streams from the backend
    // (no ~per-(gene,cell) object → fits 700k cells).
    let build_cell_pools = args.use_phase1_cell_axis();
    let pb = build_pseudobulk(
        &mut unified,
        &table,
        &genes_row_map,
        &sat_views,
        args,
        build_cell_pools,
    )
    .context("build pseudobulk")?;

    // Persist per-gene ubiquity (fraction of cells expressing) — a breadth
    // diagnostic.
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
        .runtime
        .device
        .to_device(args.runtime.device_no)
        .context("candle device init")?;
    info!("compute device = {:?}", dev);
    let mut model = GemModel::new(
        GemDims {
            n_genes: table.n_genes(),
            n_modalities: table.n_modalities(),
            n_programs: args.model.n_programs,
            n_regions: table.n_regions,
            embedding_dim: args.model.embedding_dim,
            n_cells,
        },
        &n_pbs_per_level,
        &dev,
    )
    .context("init model")?;
    // Deterministic init (CPU candle can't seed Init::Randn) so the QC is
    // reproducible run-to-run.
    model
        .seed_init(args.runtime.seed)
        .context("seed model init")?;

    let stop = setup_stop_handler();
    // Streaming phase-2 context (cell id == backend column after the up-front
    // mask+subset). Built only on the pool-free path.
    let cell_stream = (!build_cell_pools).then(|| {
        build_cell_stream_ctx(
            &unified,
            &genes_row_map,
            pass1_sats,
            &sat_links,
            (0..n_cells).collect(),
        )
    });
    let cell_nrms = train(args, &table, &pb, &mut model, &stop, cell_stream.as_ref())
        .context("training loop")?;

    // When --refine is active, pass-1 prior-score parquets go under
    // `{out}.pass1.*` so they survive the pass-2 overwrite.
    let score_prefix_p1: String = if args.refine() {
        format!("{}.pass1", args.out)
    } else {
        args.out.to_string()
    };

    // Will the (overwriting) refine pass run? If so, pass-1 outputs are
    // throwaway — skip the expensive SIMBA co-embedding here (pass 2 regenerates
    // it on the QC-passed survivors; pass 1 still holds pre-empty-call cells).
    let will_refine = args.refine() && !stop.load(std::sync::atomic::Ordering::Relaxed);

    // Durable embeddings first, so a force-abort (second Ctrl+C) during the
    // optional topic step never loses the trained model. Pass-1 has no feature QC
    // (it runs in the refine pass), so every gene co-embeds.
    let feature_keep_p1 = vec![true; table.n_genes()];
    write_outputs(
        OutputCtx {
            prefix: &args.out,
            score_prefix: &score_prefix_p1,
            table: &table,
            pb: &pb,
            model: &model,
            unified: &unified,
            target_clusters: args.qc.num_topics,
            feature_keep: &feature_keep_p1,
            feature_prior_fdr: args.qc.feature_prior_fdr,
        },
        CellQcOutputs {
            keep_idx: &keep_idx,
            cell_nrms: &cell_nrms,
            coembed: !will_refine,
        },
    )
    .context("write outputs")?;

    // Refinement pass: filter dead genes + dead cells identified from pass-1,
    // rebuild everything, retrain.  Skipped when stopped early or disabled.
    if will_refine {
        let ctx = Pass1Context {
            table: &table,
            genes_row_map: &genes_row_map,
            satellites: &satellites,
            region_map: &region_map,
        };
        // Pass 1 was genes-only, so pass 2 (full modelling over every modality)
        // always runs — even when QC filters nothing, the satellites still need
        // to enter the model.
        let out2 = run_refine_pass(args, ctx, &mut unified, &stop).context("refinement pass")?;

        {
            let Pass2Outputs {
                model: model2,
                table: table2,
                pb: pb2,
                keep_idx: keep_idx2,
                cell_nrms: cell_nrms2,
                feature_keep: feature_keep2,
            } = out2;
            write_outputs(
                OutputCtx {
                    prefix: &args.out,
                    score_prefix: &args.out,
                    table: &table2,
                    pb: &pb2,
                    model: &model2,
                    unified: &unified,
                    target_clusters: args.qc.num_topics,
                    feature_keep: &feature_keep2,
                    feature_prior_fdr: args.qc.feature_prior_fdr,
                },
                CellQcOutputs {
                    keep_idx: &keep_idx2,
                    cell_nrms: &cell_nrms2,
                    coembed: true,
                },
            )
            .context("write refined outputs")?;

            let topics2 = if args.resolve_topics() {
                Some(
                    faba::gem::topics::resolve_topics(
                        &args.out, &model2, &table2, &unified, args, &stop, &keep_idx2,
                    )
                    .context("resolve topics (refined)")?,
                )
            } else {
                None
            };

            faba::gem::manifest::write_manifest(
                &args.out,
                &model2,
                keep_idx2.len(),
                topics2.as_ref(),
                !cell_nrms2.is_empty(),
            )
            .context("write manifest (refined)")?;

            info!("done (refined) — prefix '{}'", args.out);
            return Ok(());
        }
    }

    // Single pass (no --refine, or interrupted before the refine pass): write
    // the manifest for the genes+satellites pass-1 results.
    let topics = if args.resolve_topics() {
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
    faba::gem::manifest::write_manifest(
        &args.out,
        &model,
        keep_idx.len(),
        topics.as_ref(),
        !cell_nrms.is_empty(),
    )
    .context("write manifest")?;

    info!("done — prefix '{}'", args.out);
    Ok(())
}

/// Outputs produced by a successful `run_refine_pass` invocation.
struct Pass2Outputs {
    model: GemModel,
    table: FeatureTable,
    pb: faba::gem::pseudobulk::PseudobulkData,
    keep_idx: Vec<usize>,
    cell_nrms: Vec<f32>,
    /// Per-gene (β-row order) keep mask for the SIMBA feature co-embedding: the
    /// empty cluster from feature QC is masked out of the gene *visualization*
    /// only. All-`true` when feature QC is off or finds no empty cluster.
    feature_keep: Vec<bool>,
}

/// Pass-1 (genes-only) scoring inputs bundled for `run_refine_pass`.
struct Pass1Context<'a> {
    /// Pass-1 feature table — used for gene scoring (n_genes).
    table: &'a FeatureTable,

    /// Pass-1 genes-backend row map — maps each genes feature row to its gene
    /// for the dead-gene feature-row filter.
    genes_row_map: &'a BackendRowMap,

    /// Satellite backends (unchanged across passes); re-linked to the subset
    /// genes cells in pass-2.
    satellites: &'a [SatelliteBackend],

    /// Region map (unchanged across passes).
    region_map: &'a faba::gem::region::RegionMap,
}

/// Per-gene cell-support QC for the refine pass. `support[g]` = number of cells
/// (whatever columns are currently in `unified`'s backend) with a non-zero
/// spliced count for gene `g`; `keep_mask[g] = support[g] >= min_cells`. Call
/// AFTER the empty-droplet cell subset so support is counted on REAL cells, not
/// ambient droplets — counting on raw droplets is the confound that sank the
/// old ‖β_g‖² gene QC. `min_cells <= 1` is a no-op (all kept, empty support vec).
fn gene_support_mask(
    unified: &UnifiedData,
    table: &FeatureTable,
    genes_row_map: &BackendRowMap,
    min_cells: usize,
) -> anyhow::Result<(Vec<bool>, Vec<u32>)> {
    let n_genes = table.n_genes();
    if min_cells <= 1 {
        return Ok((vec![true; n_genes], Vec::new()));
    }
    let spliced_rows = spliced_backend_rows(unified, table, genes_row_map);
    let row_nnz = collect_row_stat_across_vec(unified.count_backend(), None)
        .context("per-gene row statistics (spliced support)")?
        .count_positives();
    let mut support = vec![0u32; n_genes];
    for &row in &spliced_rows {
        if let Some(g) = genes_row_map.gene[row] {
            support[g as usize] = support[g as usize].max(row_nnz[row] as u32);
        }
    }
    let mask = support.iter().map(|&s| (s as usize) >= min_cells).collect();
    Ok((mask, support))
}

/// Rebuild the global table + pseudobulk from the current `unified` (already
/// subset to the surviving cells × genes) plus the unchanged satellites, then
/// init and train a fresh full model over every modality. Shared by the refine
/// pass-2 re-fit and the optional feature-null pass-3 re-fit. Returns
/// `(model, table, pb, cell_nrms)`; the caller derives `keep_idx`. `label` only
/// tags log/error context (`"refined"` / `"feature-refined"`).
fn rebuild_and_train(
    args: &GemArgs,
    unified: &mut UnifiedData,
    satellites: &[SatelliteBackend],
    region_map: &RegionMap,
    stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    label: &str,
) -> anyhow::Result<(GemModel, FeatureTable, PseudobulkData, Vec<f32>)> {
    // Global table from the surviving genes + the (unchanged) satellites. Dropped
    // genes are gone from the genes namespace, so `map_backend_rows` drops their
    // satellite modifier rows too (gene → None).
    let sat_name_lists: Vec<&[Box<str>]> = satellites
        .iter()
        .map(|s| s.unified.feature_names.as_slice())
        .collect();
    let table = FeatureTable::build_layered(&unified.feature_names, &sat_name_lists, region_map);
    anyhow::ensure!(
        table.n_genes() > 0,
        "refine: no genes remain after gene filter ({label})"
    );

    // Re-link satellites to the subset genes cells (some matched cells died).
    let genes_row_map = table.map_backend_rows(&unified.feature_names, region_map);
    let sat_row_maps: Vec<BackendRowMap> = satellites
        .iter()
        .map(|s| table.map_backend_rows(&s.unified.feature_names, region_map))
        .collect();
    let sat_links = link_satellites(satellites, sat_row_maps, &unified.barcodes);
    let sat_views = satellite_views(satellites, &sat_links);

    // Same pool-vs-stream choice as the main pass. Cells + features were re-subset
    // in lockstep, so cell id == backend column still holds (no remapping needed).
    let build_cell_pools = args.use_phase1_cell_axis();
    let pb = build_pseudobulk(
        unified,
        &table,
        &genes_row_map,
        &sat_views,
        args,
        build_cell_pools,
    )
    .with_context(|| format!("build pseudobulk ({label})"))?;

    data_beans_alg::gene_weighting::save_per_gene_weights(
        &pb.gene_ubiquity,
        &table.gene_names,
        &format!("{}.ubiquity.parquet", args.out),
    )
    .with_context(|| format!("save gene ubiquity ({label})"))?;

    let n_cells = unified.n_cells();
    let n_pbs: Vec<usize> = pb.pb_pools_per_level.iter().map(|l| l.n_units).collect();
    let dev = args
        .runtime
        .device
        .to_device(args.runtime.device_no)
        .with_context(|| format!("candle device init ({label})"))?;
    let mut model = GemModel::new(
        GemDims {
            n_genes: table.n_genes(),
            n_modalities: table.n_modalities(),
            n_programs: args.model.n_programs,
            n_regions: table.n_regions,
            embedding_dim: args.model.embedding_dim,
            n_cells,
        },
        &n_pbs,
        &dev,
    )
    .with_context(|| format!("init model ({label})"))?;
    model
        .seed_init(args.runtime.seed)
        .with_context(|| format!("seed model init ({label})"))?;

    let cell_stream = (!build_cell_pools).then(|| {
        build_cell_stream_ctx(
            unified,
            &genes_row_map,
            satellites,
            &sat_links,
            (0..n_cells).collect(),
        )
    });
    let cell_nrms = train(args, &table, &pb, &mut model, stop, cell_stream.as_ref())
        .with_context(|| format!("training loop ({label})"))?;

    Ok((model, table, pb, cell_nrms))
}

/// Pass 2 of `--refine`: QC the genes-only pass-1 model — call empty droplets
/// from the pre-L2 cell norm and drop genes below the `--min-cells` support
/// floor — subset `unified` to the surviving cells × genes, and rebuild + train
/// the **full** model over every modality (genes + satellites). Always runs
/// when refining (even if QC filters nothing) because pass 1 was genes-only.
fn run_refine_pass(
    args: &GemArgs,
    ctx: Pass1Context<'_>,
    unified: &mut graph_embedding_util::data::UnifiedData,
    stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<Pass2Outputs> {
    let Pass1Context {
        table: table_p1,
        genes_row_map: genes_row_map_p1,
        satellites,
        region_map,
    } = ctx;
    let n_genes = table_p1.n_genes();

    // Gene Q/C only. There is NO per-batch cell QC here: it was removed because the
    // per-batch debris cut behaved incoherently across batches (near-identical
    // depth distributions produced 0%-vs-44% drops, guillotining real cells). The
    // up-front spliced-nnz≥1 gate is the only cell filter; the refine pass refits
    // the full model on every called cell.

    // gem's only gene QC is the deterministic --min-cells cell-support filter.
    // It is deliberately NOT model-derived: an earlier EB χ²_H null on ‖β_g‖²
    // collapsed the dictionary to a handful of genes on ambient-heavy data (the
    // embedding has gene signal in few genes when most droplets are empty) and
    // corrupted every downstream consumer (annotation, topics). Counting raw cell
    // support sidesteps that confound.
    //
    // Drop genes supported by < --min-cells cells. The mask indexes genes;
    // `genes_row_map_p1` still addresses the full pre-subset feature axis, which
    // `subset_features` collapses below.
    let (live_gene_mask, gene_support) =
        gene_support_mask(unified, table_p1, genes_row_map_p1, args.qc.min_cells)
            .context("per-gene cell-support QC")?;
    if args.qc.min_cells > 1 {
        let (mut dropped, mut kept) = (Vec::new(), Vec::new());
        for (g, &keep) in live_gene_mask.iter().enumerate() {
            if keep {
                kept.push(gene_support[g] as f32);
            } else {
                dropped.push(gene_support[g] as f32);
            }
        }
        info!(
            "refine: gene QC (--min-cells {}): kept {} / {} genes (dropped median support={:.0} vs kept {:.0})",
            args.qc.min_cells,
            kept.len(),
            n_genes,
            median(&dropped),
            median(&kept),
        );
    }
    let live_feature_rows: Vec<usize> = (0..unified.n_features())
        .filter(|&r| {
            genes_row_map_p1.gene[r]
                .map(|g| live_gene_mask[g as usize])
                .unwrap_or(false)
        })
        .collect();

    info!(
        "refine: keeping {} / {} feature rows over {} cells",
        live_feature_rows.len(),
        unified.n_features(),
        unified.n_cells(),
    );

    unified.subset_features(&live_feature_rows);

    // ---- Pass 2: rebuild + train the full model on the QC-passed cells × genes.
    // `mut` so the default feature-QC drop path can re-fit (`--feature-qc-mask` opts out).
    let (mut model2, mut table2, mut pb2, mut cell_nrms2) =
        rebuild_and_train(args, unified, satellites, region_map, stop, "refined")?;

    // ---- Pass 3: HVG feature QC (model-independent, from counts) ----
    // Keep genes that are OVERDISPERSED relative to the fitted NB mean–variance
    // trend (`DispersionTrend::excess > --hvg-min-excess`) with adequate support
    // (`--feature-qc-min-nnz`); drop the low-variability background. This replaces
    // the β-based k-means/elbow QC, which was circular (it QC'd genes using the
    // embedding those genes shape) and brittle (found nothing under logistic).
    //
    // Default = DROP the low-variability genes from the model + re-fit (safe under
    // logistic NCE — no softmax partition to collapse). With `--feature-qc-mask` the
    // genes instead stay in the model and are excluded from the SIMBA gene
    // visualization only (the cell embedding + reconstruction params stay intact).
    // Off with `--skip-feature-qc`; ignored under `--skip-refine`.
    let mut feature_keep = vec![true; table2.n_genes()];
    if args.qc.feature_qc_enabled() && !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let n_genes2 = table2.n_genes();

        // Per-gene cell-support (nnz) + NB dispersion from the spliced rows.
        let genes_row_map2 = table2.map_backend_rows(&unified.feature_names, region_map);
        let spliced_rows = spliced_backend_rows(unified, &table2, &genes_row_map2);
        let row_stat = collect_row_stat_across_vec(unified.count_backend(), None)
            .context("per-gene row stats for feature QC")?;
        let row_nnz = row_stat.count_positives();
        let row_mean = row_stat.mean();
        let row_var = row_stat.variance();
        let (mut gene_nnz, mut gene_mean, mut gene_var) = (
            vec![0f32; n_genes2],
            vec![0f32; n_genes2],
            vec![0f32; n_genes2],
        );
        for &row in &spliced_rows {
            if let Some(g) = genes_row_map2.gene[row] {
                let g = g as usize;
                // Dominant spliced row per gene drives its support + dispersion.
                if row_nnz[row] > gene_nnz[g] {
                    gene_nnz[g] = row_nnz[row];
                    gene_mean[g] = row_mean[row];
                    gene_var[g] = row_var[row];
                }
            }
        }
        let trend = data_beans_alg::nb_dispersion::DispersionTrend::fit(&gene_mean, &gene_var);
        let gene_hvg: Vec<f32> = (0..n_genes2)
            .map(|g| trend.excess(gene_mean[g], gene_var[g]))
            .collect();

        let fqc = hvg_feature_qc(&gene_nnz, &gene_hvg, &args.qc.to_feature_qc_config());
        if fqc.n_dropped() == 0 {
            info!("refine: HVG feature QC — all {n_genes2} genes pass (variable)");
        } else if args.qc.feature_qc_mask {
            // Opt-out: keep the genes in the model, exclude them from the co-embed.
            info!(
                "refine: HVG feature QC — masking {} / {} low-variability genes from co-embedding",
                fqc.n_dropped(),
                n_genes2,
            );
            feature_keep = fqc.keep;
        } else {
            // Default: drop the low-variability genes from the model + re-fit.
            info!(
                "refine: HVG feature QC — DROPPING {} / {} low-variability genes + re-fitting",
                fqc.n_dropped(),
                n_genes2,
            );
            let live_rows: Vec<usize> = (0..unified.n_features())
                .filter(|&r| {
                    genes_row_map2.gene[r]
                        .map(|g| fqc.keep[g as usize])
                        .unwrap_or(false)
                })
                .collect();
            unified.subset_features(&live_rows);
            let (m3, t3, p3, c3) = rebuild_and_train(
                args,
                unified,
                satellites,
                region_map,
                stop,
                "feature-refined",
            )?;
            model2 = m3;
            table2 = t3;
            pb2 = p3;
            cell_nrms2 = c3;
            feature_keep = vec![true; table2.n_genes()];
        }
    }

    // Cell QC masked debris cells before the re-fit, so every cell remaining on the
    // axis is a real survivor — write them all.
    let keep_idx2: Vec<usize> = (0..unified.n_cells()).collect();

    Ok(Pass2Outputs {
        model: model2,
        table: table2,
        pb: pb2,
        keep_idx: keep_idx2,
        cell_nrms: cell_nrms2,
        feature_keep,
    })
}

/// Scan modifier-comp rows of every loaded satellite, return
/// `max(component_idx) + 1`. Falls back to `default_n` if no parseable
/// modifier-comp row is found (e.g. only the genes backend was loaded, or
/// the satellites carry only `chr:pos` site rows).
fn infer_n_regions(satellites: &[SatelliteBackend], default_n: usize) -> usize {
    let mut max_c: u32 = 0;
    let mut any = false;
    for sat in satellites {
        for name in &sat.unified.feature_names {
            let Some(key) = parse_feature_name(name) else {
                continue;
            };
            if !matches!(classify(&key), RowStratum::ModifierComp) {
                continue;
            }
            if let Some(c) = parse_component_idx(&key.detail) {
                max_c = max_c.max(c);
                any = true;
            }
        }
    }
    if any {
        (max_c as usize) + 1
    } else {
        default_n
    }
}

/// Argument-level sanity checks before any I/O or training. Surfaces
/// configuration mistakes (zero-dim model, NaN/out-of-range tempering,
/// stratum-fraction overflow) as a clear `anyhow::Error` rather than a
/// candle panic or silently zero-loss training.
fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.model.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.model.embedding_dim
    );
    anyhow::ensure!(
        args.model.n_programs > 0,
        "--num-programs must be > 0 (got {})",
        args.model.n_programs
    );
    if let Some(n) = args.model.n_regions {
        anyhow::ensure!(n > 0, "--num-regions must be > 0 (got {n})");
    }
    anyhow::ensure!(
        args.train.tau.is_finite() && (0.0..=1.0).contains(&args.train.tau),
        "--tau must be a finite value in [0, 1] (got {})",
        args.train.tau
    );
    anyhow::ensure!(
        args.train.tau_modality.is_finite() && (0.0..=1.0).contains(&args.train.tau_modality),
        "--tau-modality must be a finite value in [0, 1] (got {})",
        args.train.tau_modality
    );
    if args.resolve_topics() {
        // --no-cell-axis leaves e_cell at its random init (never trained),
        // so resolving topics from it would yield archetypes of noise.
        anyhow::ensure!(
            !args.collapse.no_cell_axis,
            "--resolve-topics requires a trained cell embedding, but --no-cell-axis \
             leaves e_cell at its random init; drop one of the two flags"
        );
        if let Some(k) = args.qc.num_topics {
            anyhow::ensure!(k >= 2, "--num-topics must be ≥ 2 (got {})", k);
        }
        // Else: K is auto-swept over 2..=H+1, which is always ≥ 2.
    }
    anyhow::ensure!(
        args.train.f_agg.is_finite() && (0.0..=1.0).contains(&args.train.f_agg),
        "--f-agg must be in [0, 1] (got {})",
        args.train.f_agg
    );
    anyhow::ensure!(
        args.train.f_count.is_finite() && (0.0..=1.0).contains(&args.train.f_count),
        "--f-count must be in [0, 1] (got {})",
        args.train.f_count
    );
    anyhow::ensure!(
        args.train.f_agg + args.train.f_count <= 1.0,
        "--f-agg + --f-count must be ≤ 1.0 (got {} + {} = {}); modifier stratum gets the remainder",
        args.train.f_agg,
        args.train.f_count,
        args.train.f_agg + args.train.f_count
    );
    // At least one negative source must be active, else the logistic NEG loss
    // has no contrast (positive scores run to +∞ with no opposing gradient).
    anyhow::ensure!(
        args.train.n_marginal_neg + args.train.n_swap_gene_mode + args.train.n_swap_modality > 0,
        "at least one of --n-marginal-neg, --n-swap-gene-mode, --n-swap-modality must be > 0 \
         (else the NCE loss has no negatives and AdamW makes no progress)"
    );
    if args.refine() {
        anyhow::ensure!(
            args.qc.gene_null_fdr > 0.0 && args.qc.gene_null_fdr < 1.0,
            "--gene-null-fdr must be in (0, 1) (got {})",
            args.qc.gene_null_fdr
        );
    }
    Ok(())
}
