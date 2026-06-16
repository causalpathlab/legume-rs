//! Entry point for `faba gem` (alias `gem-embedding`).

use anyhow::Context;
use data_beans::qc::collect_column_stat_across_vec;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::data::UnifiedData;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::{load_unified_data, FeatureNameKind, LoadUnifiedArgs};
use log::{info, warn};
use matrix_util::common_io::{basename, mkdir_parent};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::RunningStatOps;
use matrix_util::utils::median;
use rayon::ThreadPoolBuilder;
use rustc_hash::FxHashMap;

use faba::gem::args::GemArgs;
use faba::gem::cell_solve::{CellStreamCtx, SatStream};
use faba::gem::feature_table::{BackendRowMap, FeatureTable};
use faba::gem::manifest::{write_outputs, CellQcOutputs};
use faba::gem::model::GemModel;
use faba::gem::pseudobulk::{build_pseudobulk, spliced_backend_rows, SatelliteData};
use faba::gem::region::{load_component_annotations, ComponentAnnotation, RegionMap};
use faba::gem::train::train;
use graph_embedding_util::null_call::embedding_mixture_empty_call;

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
                    "satellite {}: 0 / {total} columns matched a genes cell — this modality \
                     donates NOTHING to the model. Its `barcode@sample` names don't line up with \
                     the genes cells; check that the genes and {} files reduce to the SAME sample \
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

    let n_threads = if args.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        args.threads
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
    let feature_kind = if args.feature_name_exact {
        FeatureNameKind::Exact
    } else {
        FeatureNameKind::Gene {
            delim: args.feature_name_delim,
        }
    };

    let batch_files: Option<&[Box<str>]> = if args.ignore_batch {
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

    info!(
        "loading {} genes file(s) (feature_kind={:?}, preload={})",
        args.genes.len(),
        feature_kind,
        args.preload_data
    );
    let mut unified = load_modality(
        &args.genes,
        &args.genes_sample_strip,
        do_tag,
        batch_files,
        feature_kind.clone(),
        args.preload_data,
    )
    .context("load genes backend")?;

    if args.ignore_batch {
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
            args.dartseq_sample_strip.as_ref(),
            "m6A",
        ),
        (
            args.atoi.as_deref().unwrap_or(&[]),
            args.atoi_sample_strip.as_ref(),
            "A2I",
        ),
        (
            args.apa.as_deref().unwrap_or(&[]),
            args.apa_sample_strip.as_ref(),
            "pA",
        ),
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
            args.preload_data,
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
    let region_map = build_region_map(args)?;
    let pass1_sats: &[SatelliteBackend] = if args.refine() { &[] } else { &satellites };
    let sat_name_lists: Vec<&[Box<str>]> = pass1_sats
        .iter()
        .map(|s| s.unified.feature_names.as_slice())
        .collect();
    let table = FeatureTable::build_layered(&unified.feature_names, &sat_name_lists, &region_map);
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
    let cell_nnz_full =
        collect_column_stat_across_vec(unified.count_backend(), Some(&spliced_rows), None)
            .context("cell QC column statistics (spliced)")?
            .count_positives();
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
    // Per-cell spliced nnz for the kept cells (re-applied in the refine pass).
    let cell_nnz: Vec<f32> = kept.iter().map(|&c| cell_nnz_full[c]).collect();
    info!(
        "cell QC (spliced non-zeros): kept {} / {} cells at cutoff {} (dropped {}) \
         [conservative; refine empty-call makes the real cut]",
        kept.len(),
        n_cells_full,
        cell_cutoff,
        n_cells_full - kept.len(),
    );
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
    // Deterministic init (CPU candle can't seed Init::Randn) so the QC is
    // reproducible run-to-run.
    model.seed_init(args.seed).context("seed model init")?;

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

    // Durable embeddings first, so a force-abort (second Ctrl+C) during the
    // optional topic step never loses the trained model.
    write_outputs(
        &args.out,
        &score_prefix_p1,
        &table,
        &pb,
        &model,
        &unified,
        CellQcOutputs {
            keep_idx: &keep_idx,
            cell_nrms: &cell_nrms,
        },
    )
    .context("write outputs")?;

    // Refinement pass: filter dead genes + dead cells identified from pass-1,
    // rebuild everything, retrain.  Skipped when stopped early or disabled.
    if args.refine() && !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let ctx = Pass1Context {
            cell_nrms: &cell_nrms,
            cell_nnz: &cell_nnz,
            cell_cutoff,
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
            } = out2;
            write_outputs(
                &args.out,
                &args.out,
                &table2,
                &pb2,
                &model2,
                &unified,
                CellQcOutputs {
                    keep_idx: &keep_idx2,
                    cell_nrms: &cell_nrms2,
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
}

/// Pass-1 (genes-only) scoring inputs bundled for `run_refine_pass`.
struct Pass1Context<'a> {
    /// Pre-L2-normalisation cell norms from phase-2 (one per original cell);
    /// `√emb_sq_norm`. The magnitude the cell projection found — near-zero
    /// means the IRLS fit nothing beyond the gene-bias background, i.e. a
    /// near-empty cell. Empty when phase-2 was skipped. Drives the cell QC.
    cell_nrms: &'a [f32],

    /// Per-cell **spliced** non-zero count (f32, one per cell). Reported
    /// side-by-side with the pre-L2 norm in `{out}.cell_qc.parquet`.
    cell_nnz: &'a [f32],

    /// Effective spliced-nnz cell cutoff from pass-1 (auto cell-calling, else 1),
    /// re-applied to the dead-cell filter so both passes agree.
    cell_cutoff: usize,

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

/// Per-cell QC report written to `{out}.cell_qc.parquet`: the pre-L2 fit
/// norm and the spliced nnz **side by side**, both cutoff values, and how
/// each cutoff fired per cell (`pass_nnz` / `pass_norm` / `kept`). Covers
/// all N_old cells (the full pre-subset axis), so a row exists for every
/// dropped cell too. `pre_l2_norm` is `NaN` when phase-2 was skipped.
#[allow(clippy::too_many_arguments)]
fn write_cell_qc_parquet(
    args: &GemArgs,
    gem_data: &graph_embedding_util::data::UnifiedData,
    cell_nrms: &[f32],
    cell_nnz: &[f32],
    cell_cutoff: usize,
    norm_cut: f32,
    have_nrms: bool,
    kept: &[bool],
) -> anyhow::Result<()> {
    use matrix_util::traits::IoOps;
    let n = gem_data.n_cells();
    let mut m = DMatrix::<f32>::zeros(n, 7);
    for c in 0..n {
        let pass_nnz = (cell_nnz[c] as usize) >= cell_cutoff;
        // The EB call is the cell decision, so `pass_norm` == `kept` (no
        // separate nnz-AND); kept here for the side-by-side QC schema.
        let pass_norm = !have_nrms || kept[c];
        m[(c, 0)] = cell_nnz[c];
        m[(c, 1)] = if have_nrms { cell_nrms[c] } else { f32::NAN };
        m[(c, 2)] = cell_cutoff as f32;
        m[(c, 3)] = norm_cut;
        m[(c, 4)] = pass_nnz as u8 as f32;
        m[(c, 5)] = pass_norm as u8 as f32;
        m[(c, 6)] = kept[c] as u8 as f32;
    }
    let cols: Vec<Box<str>> = [
        "nnz",
        "pre_l2_norm",
        "nnz_cutoff",
        "norm_cutoff",
        "pass_nnz",
        "pass_norm",
        "kept",
    ]
    .iter()
    .map(|s| Box::from(*s))
    .collect();
    let path = format!("{}.cell_qc.parquet", args.out);
    m.to_parquet_with_names(&path, (Some(&gem_data.barcodes), Some("cell")), Some(&cols))
        .with_context(|| format!("writing {path}"))?;
    info!("refine: cell QC → {} ({} cells)", path, n);
    Ok(())
}

/// Pass 2 of `--refine`: QC the genes-only pass-1 model — flag empty gene
/// clusters by low mean ‖β‖, then drop near-empty cells by their pre-L2
/// projection norm — subset `unified` to the survivors, and rebuild + train
/// the **full** model over every modality (genes + satellites). Always runs
/// when refining (even if QC filters nothing) because pass 1 was genes-only.
fn run_refine_pass(
    args: &GemArgs,
    ctx: Pass1Context<'_>,
    unified: &mut graph_embedding_util::data::UnifiedData,
    stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<Pass2Outputs> {
    let Pass1Context {
        cell_nrms,
        cell_nnz,
        cell_cutoff,
        table: table_p1,
        genes_row_map: genes_row_map_p1,
        satellites,
        region_map,
    } = ctx;
    let n_genes = table_p1.n_genes();
    let n_cells_old = unified.n_cells();

    // QC on the genes-only pass-1 model, then full modelling in pass 2 — both
    // calls are keyed on the learned gene embedding β (empirical-Bayes, no
    // k-means, no median×frac heuristic). nnz, the pre-L2 norm, the cutoffs and
    // each cell's pass/fail land side by side in `{out}.cell_qc.parquet`.

    ////////////////////////////////////////////////////////////////////////////
    // step 1. NO gene Q/C — keep the full gene dictionary, matching senna bge. //
    ////////////////////////////////////////////////////////////////////////////

    // gem does cell-only QC (like bge). An earlier EB χ²_H null call on ‖β_g‖²
    // collapsed the dictionary to a handful of genes on ambient-heavy data
    // (the embedding has gene signal in few genes when most droplets are empty),
    // which then corrupted the cell empty-call and every downstream consumer
    // (annotation, topics). Keep all genes; the cell empty-call below is the
    // only refine filter, exactly as bge has no feature QC.
    let live_gene_mask = vec![true; n_genes];

    //////////////////////
    // step 2. cell q/c //
    //////////////////////

    // Independent EB "empty droplet" call on the pre-L2 cell norm. Genes and
    // cells live on different scales (gene β is weight-decayed small; the cell
    // MAP norm scales with depth), so the gene null does NOT transfer — cells
    // get their OWN call. Once the model has trained, empties don't collapse to
    // a ≈0 lower tail; they form their OWN low mode (norm ~0.2) separated from
    // the real bulk by a density valley, so a median+MAD lower-tail fit misses
    // them. Instead fit a BIC-selected 1-D Gaussian mixture on `log(cell_nrms)`,
    // take the lowest mode as empty (the first prominent density valley above
    // it is the cut), and drop by MAP posterior P(empty)≥0.5. No median×frac
    // heuristic, no kmeans. The spliced nnz cutoff still gates depth.
    let have_nrms = cell_nrms.len() == n_cells_old;
    let (cell_keep, norm_cut): (Vec<bool>, f32) = if have_nrms {
        // BIC-selected Gaussian mixture on log(cell_nrms): the empties are a
        // separate low mode (not a tail at ≈0 — once trained, ambient droplets
        // still get a small non-zero norm), so a lower-tail/median fit misses
        // them. The mixture isolates the empty mode (lowest, below the first
        // prominent valley) and drops it by MAP posterior. k is BIC-selected up
        // to QC_MIXTURE_K_MAX (generous — a minority empty mode needs enough
        // components to separate from the broad real distribution).
        let call = embedding_mixture_empty_call(
            cell_nrms,
            graph_embedding_util::null_call::QC_MIXTURE_K_MAX,
            args.gene_null_fdr,
        );
        // Effective drop cut = largest norm still called empty (MAP rule, at the
        // density valley — can sit above the component-labeling antimode).
        let norm_cut = if call.cut.is_finite() {
            call.cut.exp() as f32
        } else {
            0.0
        };
        info!(
            "refine: cell empty call (mixture k={}, dropped norm ≤ {:.3}, π̂_empty={:.2}) — {} / {} cells empty",
            call.k, norm_cut, call.empty_frac, call.n_drop, n_cells_old
        );
        (call.drop.iter().map(|&d| !d).collect(), norm_cut)
    } else {
        (vec![true; n_cells_old], 0.0)
    };
    // `cell_keep` IS the final per-cell decision: the EB empty call alone
    // (matches senna bge — the spliced-nnz cutoff already filtered every cell
    // before pass 1, so AND-ing it here would be redundant).
    let n_kept_cells = cell_keep.iter().filter(|&&v| v).count();
    let n_dead_cells = n_cells_old - n_kept_cells;

    // Paired QC report: nnz beside the pre-L2 norm + how each cutoff fired.
    write_cell_qc_parquet(
        args,
        unified,
        cell_nrms,
        cell_nnz,
        cell_cutoff,
        norm_cut,
        have_nrms,
        &cell_keep,
    )
    .context("write cell_qc.parquet")?;

    if have_nrms {
        // One pass: split nnz into dropped vs kept for the diagnostic medians.
        let (mut dropped_nnz, mut kept_nnz) = (Vec::new(), Vec::new());
        for c in 0..n_cells_old {
            if cell_keep[c] {
                kept_nnz.push(cell_nnz[c]);
            } else {
                dropped_nnz.push(cell_nnz[c]);
            }
        }
        info!(
            "refine: {} / {} cells dead (dropped median nnz={:.0} vs kept {:.0})",
            n_dead_cells,
            n_cells_old,
            median(&dropped_nnz),
            median(&kept_nnz),
        );
    } else {
        info!(
            "refine: {} / {} cells dead (phase-2 skipped — nnz cutoff {} only)",
            n_dead_cells, n_cells_old, cell_cutoff
        );
    }

    // ---- Filter unified (genes backend) ----
    let live_feature_rows: Vec<usize> = (0..unified.n_features())
        .filter(|&r| {
            genes_row_map_p1.gene[r]
                .map(|g| live_gene_mask[g as usize])
                .unwrap_or(false)
        })
        .collect();
    let live_cell_old_ids: Vec<usize> = (0..n_cells_old).filter(|&c| cell_keep[c]).collect();

    info!(
        "refine: keeping {} / {} feature rows, {} / {} cells",
        live_feature_rows.len(),
        unified.n_features(),
        live_cell_old_ids.len(),
        n_cells_old,
    );

    // Workflow step (iii): mask out the empty cells AND the dead genes, then
    // re-fit on the survivors (not output-only — the EB-called empties never
    // re-enter the model). `mask_columns` shrinks the backend cell columns,
    // `subset_cells` syncs the logical axis, `subset_features` drops the
    // dead-gene rows; all three renumber survivors in ascending order, so
    // "cell id == backend column" still holds for the streaming phase-2 pass,
    // exactly as the initial up-front QC established. Guard against a degenerate
    // all-empty call (mask_columns to 0 columns would crash the re-fit) — keep
    // every cell in that case, like the pass-1 fit.
    if live_cell_old_ids.is_empty() {
        warn!(
            "refine: cell empty call flagged ALL {n_cells_old} cells empty — keeping them \
             (no cell mask); inspect {{out}}.cell_qc.parquet and --gene-null-fdr"
        );
    } else if live_cell_old_ids.len() < n_cells_old {
        unified
            .count_backend_mut()
            .mask_columns(&cell_keep)
            .context("subset genes cells in refine (mask_columns)")?;
        unified.subset_cells(&live_cell_old_ids);
    }
    unified.subset_features(&live_feature_rows);

    // ---- Rebuild ----
    // Global table from the surviving genes + the (unchanged) satellites.
    // Dead genes are gone from the genes namespace, so `map_backend_rows`
    // drops their satellite modifier rows too (gene → None).
    let sat_name_lists: Vec<&[Box<str>]> = satellites
        .iter()
        .map(|s| s.unified.feature_names.as_slice())
        .collect();
    let table2 = FeatureTable::build_layered(&unified.feature_names, &sat_name_lists, region_map);
    anyhow::ensure!(
        table2.n_genes() > 0,
        "refine: no genes remain after dead-gene filter"
    );

    // Re-link satellites to the subset genes cells (some matched cells died).
    let genes_row_map2 = table2.map_backend_rows(&unified.feature_names, region_map);
    let sat_row_maps2: Vec<BackendRowMap> = satellites
        .iter()
        .map(|s| table2.map_backend_rows(&s.unified.feature_names, region_map))
        .collect();
    let sat_links2 = link_satellites(satellites, sat_row_maps2, &unified.barcodes);
    let sat_views2 = satellite_views(satellites, &sat_links2);

    // Same pool-vs-stream choice as the main pass. Cells + features were
    // re-subset in lockstep above, so cell id == backend column still holds (no
    // refine remapping needed).
    let build_cell_pools = args.use_phase1_cell_axis();
    let pb2 = build_pseudobulk(
        unified,
        &table2,
        &genes_row_map2,
        &sat_views2,
        args,
        build_cell_pools,
    )
    .context("build pseudobulk (refined)")?;

    // Diagnostic weights for pass-2.
    data_beans_alg::gene_weighting::save_per_gene_weights(
        &pb2.gene_ubiquity,
        &table2.gene_names,
        &format!("{}.ubiquity.parquet", args.out),
    )
    .context("save gene ubiquity (refined)")?;

    let n_cells2 = unified.n_cells();
    let n_pbs2: Vec<usize> = pb2.pb_pools_per_level.iter().map(|l| l.n_units).collect();
    let dev = args
        .device
        .to_device(args.device_no)
        .context("candle device init (refined)")?;
    let mut model2 = GemModel::new(
        table2.n_genes(),
        table2.n_modalities(),
        args.n_programs,
        table2.n_regions,
        args.embedding_dim,
        n_cells2,
        &n_pbs2,
        &dev,
    )
    .context("init model (refined)")?;
    model2
        .seed_init(args.seed)
        .context("seed model init (refined)")?;

    // Cells were re-subset in lockstep with the backend, so cell id == backend
    // column (identity) still holds — stream phase-2 over every kept cell.
    let cell_stream2 = (!build_cell_pools).then(|| {
        build_cell_stream_ctx(
            unified,
            &genes_row_map2,
            satellites,
            &sat_links2,
            (0..n_cells2).collect(),
        )
    });
    let cell_nrms2 = train(
        args,
        &table2,
        &pb2,
        &mut model2,
        stop,
        cell_stream2.as_ref(),
    )
    .context("training loop (refined)")?;

    // Empties were masked out before the re-fit, so every cell remaining on the
    // axis is a real survivor — write them all.
    let keep_idx2: Vec<usize> = (0..n_cells2).collect();

    Ok(Pass2Outputs {
        model: model2,
        table: table2,
        pb: pb2,
        keep_idx: keep_idx2,
        cell_nrms: cell_nrms2,
    })
}

/// Per-file sample id: the file's basename (sparse-data extension stripped)
/// with the per-flag `strip` suffix removed. `rep1_wt_genes.zarr.zip` with
/// `strip = "_genes"` → `rep1_wt`. Empty `strip` (or a non-matching one)
/// keeps the full basename, so two modality files of one sample merge only
/// when their stripped basenames agree.
fn file_sample_id(file: &str, strip: &str) -> anyhow::Result<Box<str>> {
    let base = basename(file)?;
    let sid = if strip.is_empty() {
        base.as_ref()
    } else {
        base.as_ref().strip_suffix(strip).unwrap_or(base.as_ref())
    };
    Ok(sid.into())
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
    if args.resolve_topics() {
        // --no-cell-axis leaves e_cell at its random init (never trained),
        // so resolving topics from it would yield archetypes of noise.
        anyhow::ensure!(
            !args.no_cell_axis,
            "--resolve-topics requires a trained cell embedding, but --no-cell-axis \
             leaves e_cell at its random init; drop one of the two flags"
        );
        if let Some(k) = args.num_topics {
            anyhow::ensure!(k >= 2, "--num-topics must be ≥ 2 (got {})", k);
        }
        // Else: K is auto-swept over 2..=H+1, which is always ≥ 2.
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
    if args.refine() {
        anyhow::ensure!(
            args.gene_null_fdr > 0.0 && args.gene_null_fdr < 1.0,
            "--gene-null-fdr must be in (0, 1) (got {})",
            args.gene_null_fdr
        );
    }
    Ok(())
}
