#![allow(clippy::too_many_arguments)]
//! `pinto plot` — publication-quality spatial scatter for link-community /
//! dsvd / propensity outputs, with optional marker-gene overlays.
//!
//! Shares rasterizer / palette / SVG emission with senna via
//! `plot-utils`. The only pinto-specific pieces live here:
//!
//! - [`discover`] — find `{prefix}.L*.propensity.parquet` siblings
//! - [`load`] — read `coord_pairs.parquet` + propensity +
//!   link_community + gene_topic
//! - [`partition`] — split cells by `left_batch` into cores (≥1)
//! - [`viridis`] — robust-percentile log standardization + viridis LUT
//! - [`markers`] — top-N gene ranking + chunked row extraction from
//!   `data-beans::SparseIoVec`
//! - [`render`] — per-(level, core) `TopicLayer` builders + shared
//!   SVG→PNG/PDF emitter

pub mod args;
pub mod discover;
pub mod interfaces;
pub mod load;
pub mod lr_overlay;
pub mod markers;
pub mod partition;
pub mod render;
pub mod viridis;

pub use args::SrtPlotArgs;

use crate::util::common::*;
use crate::util::input::read_expr_data;
use rayon::prelude::*;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Mutex;

/// Output sub-directory under `{out}.plots/`. Single source of truth so
/// the mkdir loop and the per-plot dispatch stay in sync.
#[derive(Copy, Clone)]
enum PlotKind {
    Community,
    Propensity,
    Mesh,
    Interfaces,
    Markers,
    Lr,
}

impl PlotKind {
    const ALL: &'static [PlotKind] = &[
        PlotKind::Community,
        PlotKind::Propensity,
        PlotKind::Mesh,
        PlotKind::Interfaces,
        PlotKind::Markers,
        PlotKind::Lr,
    ];

    fn subdir(self) -> &'static str {
        match self {
            PlotKind::Community => "community",
            PlotKind::Propensity => "propensity",
            PlotKind::Mesh => "mesh",
            PlotKind::Interfaces => "interfaces",
            PlotKind::Markers => "markers",
            PlotKind::Lr => "lr",
        }
    }
}

use discover::{discover_levels, LevelSelector};
use load::{read_cells_from_coord_pairs, read_gene_topic, read_link_community, read_propensity};
use partition::partition_cells;
use render::{
    build_community_layers, build_marker_by_community_layers, build_marker_heatmap_layers,
    build_mesh_layers, build_propensity_argmax_layers, build_propensity_community_heatmap_layers,
    emit_figure, ColorBook, Frame,
};

/// Entry point: auto-discover outputs, partition, emit figures, write
/// a JSON manifest listing everything produced.
pub fn make_srt_plot(args: &SrtPlotArgs) -> anyhow::Result<()> {
    // Auto-detect if --from is a JSON metadata file or a prefix.
    // When metadata.json is supplied, capture its coord_columns hint so
    // the coord_pairs reader doesn't have to auto-discover.
    let (prefix, coord_columns_hint) = if args.from.ends_with(".json") {
        let meta_path = std::path::Path::new(args.from.as_ref());
        let meta = crate::util::metadata::PintoMetadata::read(meta_path)?;
        (
            meta.prefix.into_boxed_str(),
            meta.outputs.coord_columns.clone(),
        )
    } else {
        (args.from.clone(), None)
    };

    let out_prefix = args.out.as_deref().unwrap_or(&prefix).to_string();

    let selector = LevelSelector::parse(&args.levels);
    let levels = discover_levels(&prefix, &selector)?;
    info!(
        "discovered {} level(s): {}",
        levels.len(),
        levels
            .iter()
            .map(|l| l.tag.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let coord_path = PathBuf::from(format!("{prefix}.coord_pairs.parquet"));
    if !coord_path.exists() {
        anyhow::bail!("{coord_path:?} not found — did you pass the right --from prefix?");
    }
    let cells = read_cells_from_coord_pairs(&coord_path, coord_columns_hint.as_deref())?;
    info!(
        "loaded {} cells; batch column: {}",
        cells.n(),
        if cells.batches.is_some() { "yes" } else { "no" }
    );

    let cores = partition_cells(&cells, args.min_core_cells, args.coord_clip);
    if cores.is_empty() {
        anyhow::bail!(
            "no cores passed --min-core-cells={} filter (try lowering it)",
            args.min_core_cells
        );
    }
    info!(
        "plotting {} core(s): {}",
        cores.len(),
        cores
            .iter()
            .map(|c| format!("{}({})", c.name, c.n()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // One gene_topic (shared across all levels) — fine for now; if a
    // level carries its own, we swap below.
    let final_gene_topic_path = PathBuf::from(format!("{prefix}.gene_topic.parquet"));
    let global_gt = if final_gene_topic_path.exists() {
        Some(read_gene_topic(&final_gene_topic_path)?)
    } else {
        None
    };

    // Open expression data once. Used by markers (--top-markers > 0) AND
    // by the LR-overlay direction inference (per-edge L/R expression).
    let expr_data = match &args.data {
        Some(files) => Some(read_expr_data(files)?),
        None => None,
    };
    // Only the cell-name → data-beans column index map is needed at plot
    // time — gene lookup is done per-backend inside `fetch_gene_rows_aligned`
    // using each backend's row_names().
    let cell_col_index: Option<HashMap<Box<str>, usize>> = match &expr_data {
        Some(d) => Some(
            d.column_names()?
                .into_iter()
                .enumerate()
                .map(|(i, c)| (c, i))
                .collect(),
        ),
        None => None,
    };

    let emitted = Mutex::new(Vec::<PathBuf>::new());

    // Each level is independent — fan out across levels, then serialize
    // cores within a level so per-level logs stay readable.
    // Coord columns in coord_pairs aren't `left_*`/`right_*`-stripped in
    // propensity parquets (pinto prop appends them bare), so build the
    // exclude set from CellTable's known coord column names.
    let excluded: HashSet<Box<str>> = cells.coord_col_names.iter().cloned().collect();

    // Each plot type lands in its own subdir; filenames carry only the
    // level tag (and any kind-specific suffix). Single core → no
    // `core{batch}` infix.
    let plot_dir = PathBuf::from(format!("{}.plots", out_prefix));
    for kind in PlotKind::ALL {
        std::fs::create_dir_all(plot_dir.join(kind.subdir()))?;
    }

    levels
        .par_iter()
        .try_for_each(|level| -> anyhow::Result<()> {
            let prop_path = level.propensity_path(&prefix);
            let (propensity, dominant, entropy_opt, prop_cell_names) =
                read_propensity(&prop_path, &excluded)?;

            // Align propensity rows → global cell index.
            let aligned_dominant = align_dominant(&cells, &prop_cell_names, &dominant);
            let aligned_propensity = align_propensity(&cells, &prop_cell_names, &propensity);
            let aligned_entropy: Option<Vec<f32>> = entropy_opt
                .as_ref()
                .map(|ent| align_entropy(&cells, &prop_cell_names, ent));

            let k = aligned_propensity.ncols();
            let colors = ColorBook::new(args, k);

            // Per-level link_community (skip if absent — dsvd output).
            // Loaded eagerly when present because both the mesh plot and
            // the interfaces sub-mode need the edge list; --no-mesh only
            // turns off the mesh render, not the edge load.
            let lc_path = level.link_community_path(&prefix);
            let lc_pair = if lc_path.exists() {
                Some(read_link_community(&lc_path)?)
            } else {
                None
            };

            // Per-level gene_topic (fall back to global).
            let level_gt_path = level.gene_topic_path(&prefix);
            let gene_topic = if level_gt_path.exists() {
                Some(read_gene_topic(&level_gt_path)?)
            } else {
                global_gt.clone()
            };

            for core in &cores {
                let frame = Frame::new(core, args);
                let kind_path = |kind: PlotKind, suffix: &str| -> PathBuf {
                    let leaf = if suffix.is_empty() {
                        level.tag.clone()
                    } else {
                        format!("{}.{}", level.tag, suffix)
                    };
                    plot_dir.join(kind.subdir()).join(leaf)
                };
                let mut local_emitted: Vec<PathBuf> = Vec::new();

                // 1. Community scatter
                let layers = build_community_layers(
                    &frame,
                    &cells,
                    core,
                    &aligned_dominant,
                    &colors,
                    args.point_shape,
                    args.alpha,
                )?;
                emit_figure(
                    &layers,
                    &frame,
                    args,
                    &kind_path(PlotKind::Community, ""),
                    &mut local_emitted,
                )?;

                // 2. Propensity argmax overlay
                let layers = build_propensity_argmax_layers(
                    &frame,
                    &cells,
                    core,
                    &aligned_dominant,
                    &aligned_propensity,
                    &colors,
                    args.point_shape,
                    args.alpha,
                    args.size_scale,
                )?;
                emit_figure(
                    &layers,
                    &frame,
                    args,
                    &kind_path(PlotKind::Propensity, "argmax"),
                    &mut local_emitted,
                )?;

                // 3. Per-community soft-membership heatmaps
                for kk in 0..k {
                    let layers = build_propensity_community_heatmap_layers(
                        &frame,
                        &cells,
                        core,
                        &aligned_propensity,
                        kk,
                        args.heat_bins,
                        args.alpha,
                        args.point_shape,
                        args.size_scale,
                    )?;
                    emit_figure(
                        &layers,
                        &frame,
                        args,
                        &kind_path(PlotKind::Propensity, &format!("community{kk}")),
                        &mut local_emitted,
                    )?;
                }

                // 4. Mesh (cell-cell edges colored by community) if lc data present
                if !args.no_mesh {
                    if let Some((edges, community)) = &lc_pair {
                        let core_set: HashSet<usize> = core.cell_ixs.iter().copied().collect();
                        let layers = build_mesh_layers(
                            &frame,
                            &cells,
                            &core_set,
                            edges,
                            community,
                            &colors,
                            args.mesh_alpha,
                        )?;
                        emit_figure(
                            &layers,
                            &frame,
                            args,
                            &kind_path(PlotKind::Mesh, ""),
                            &mut local_emitted,
                        )?;
                    }
                }

                // 4b. Interfaces (high-entropy + neighborhoods).
                if args.show_interfaces {
                    match &aligned_entropy {
                        Some(ent) => {
                            let ifc_stub = kind_path(PlotKind::Interfaces, "");
                            let edges_only = lc_pair.as_ref().map(|(e, _)| e.as_slice());
                            let written = interfaces::render_interfaces(
                                args,
                                &frame,
                                &cells,
                                core,
                                ent,
                                &aligned_dominant,
                                edges_only,
                                gene_topic.as_ref(),
                                &ifc_stub,
                            )?;
                            local_emitted.extend(written);
                        }
                        None => {
                            log::warn!(
                                "[{}] propensity parquet has no `entropy` column — \
                                 skipping --show-interfaces; rerun lc/dsvd/prop to populate it",
                                level.tag,
                            );
                        }
                    }
                }

                // 5. Marker genes (requires data + gene_topic)
                if args.top_markers > 0 {
                    let (gt, gene_names) = match &gene_topic {
                        Some(x) => x,
                        None => {
                            info!("[{}] no gene_topic parquet — skipping markers", level.tag);
                            continue;
                        }
                    };
                    let data = match &expr_data {
                        Some(d) => d,
                        None => continue,
                    };
                    let ccol_ix = cell_col_index
                        .as_ref()
                        .expect("cell index built alongside data");

                    emit_marker_figures(
                        args,
                        &frame,
                        &cells,
                        core,
                        &aligned_dominant,
                        gt,
                        gene_names,
                        data,
                        ccol_ix,
                        &colors,
                        &level.tag,
                        &out_prefix,
                        &mut local_emitted,
                    )?;
                }

                info!("[{}] wrote {} files", level.tag, local_emitted.len());
                emitted.lock().expect("emitted lock").extend(local_emitted);
            }

            Ok(())
        })?;

    let mut emitted = emitted.into_inner().expect("emitted lock");

    // Post-pass: LR-activity overlays. Level-independent — one figure per
    // (core, significant pair), runs after the level loop completes.
    if !args.no_lr_overlay {
        if let Some(p) = resolve_lr_json_path(args, &prefix) {
            match lr_overlay::load_lr_json(&p) {
                Ok(lr) => {
                    info!(
                        "Rendering LR overlays from {} ({} strata, {} results)",
                        p.display(),
                        lr.strata.len(),
                        lr.results.len(),
                    );
                    let lr_expr = match (&expr_data, cell_col_index.as_ref()) {
                        (Some(data), Some(ccol)) => {
                            lr_overlay::prefetch_lr_expression(data, ccol, &cells, &lr)?
                        }
                        _ => {
                            log::info!(
                                "LR overlay: --data not provided; arrows use canonical \
                                 (left → right) orientation only"
                            );
                            None
                        }
                    };

                    // Load the "final" propensity once. LR-overlay focal
                    // pool = cells whose top community propensity is
                    // *below* `commit_threshold` (i.e. uncommitted /
                    // boundary cells). Cleaner than an entropy quantile.
                    let (final_aligned_prop, final_dominant): (Option<Mat>, Option<Vec<i64>>) = {
                        let final_path = PathBuf::from(format!("{prefix}.propensity.parquet"));
                        if final_path.exists() {
                            let (prop, dom, _, prop_cell_names) =
                                load::read_propensity(&final_path, &excluded)?;
                            let aligned = align_propensity(&cells, &prop_cell_names, &prop);
                            let aligned_dom = align_dominant(&cells, &prop_cell_names, &dom);
                            (Some(aligned), Some(aligned_dom))
                        } else {
                            (None, None)
                        }
                    };
                    if final_aligned_prop.is_none() {
                        log::info!(
                            "LR overlay: no final propensity parquet; arrows will not be \
                             restricted to boundary cells"
                        );
                    }

                    let commit_threshold = args.lr_commit_threshold;

                    // Load the final-level link_community edges once so we
                    // can expand the boundary belt by one hop (a focal cell
                    // pulls in all its graph neighbors). Wider belt without
                    // touching the commitment threshold.
                    let final_lc_path = PathBuf::from(format!("{prefix}.link_community.parquet"));
                    let final_lc: Option<(Vec<load::EdgePair>, Vec<i64>)> =
                        if final_lc_path.exists() {
                            Some(load::read_link_community(&final_lc_path)?)
                        } else {
                            None
                        };

                    let plot_dir = PathBuf::from(format!("{}.plots", out_prefix));
                    let lr_dir = plot_dir.join(PlotKind::Lr.subdir());
                    std::fs::create_dir_all(&lr_dir)?;
                    let per_core: Vec<Vec<PathBuf>> = cores
                        .par_iter()
                        .map(|core| -> anyhow::Result<Vec<PathBuf>> {
                            let frame = render::Frame::new(core, args);
                            let hulls_by_community: HashMap<
                                i64,
                                Vec<plot_utils::svg_emit::TopicLayer>,
                            > = if args.no_lr_hulls {
                                HashMap::default()
                            } else {
                                match (&final_dominant, final_lc.as_ref()) {
                                    (Some(dom), Some((edges, _))) => render::community_cc_hulls(
                                        &frame,
                                        &cells,
                                        core,
                                        dom,
                                        edges,
                                        args.lr_hull_min_cells,
                                    ),
                                    _ => HashMap::default(),
                                }
                            };
                            let focal_set: Option<HashSet<usize>> =
                                final_aligned_prop.as_ref().map(|prop| {
                                    let mut focal: HashSet<usize> =
                                        interfaces::pick_uncommitted_cells(
                                            core,
                                            prop,
                                            commit_threshold,
                                        )
                                        .into_iter()
                                        .collect();
                                    if let Some((edges, _)) = final_lc.as_ref() {
                                        let core_set: HashSet<usize> =
                                            core.cell_ixs.iter().copied().collect();
                                        let mut neighbors: HashSet<usize> = HashSet::default();
                                        for (l, r) in edges {
                                            let li = match cells.index.get(l) {
                                                Some(&i) if core_set.contains(&i) => i,
                                                _ => continue,
                                            };
                                            let ri = match cells.index.get(r) {
                                                Some(&i) if core_set.contains(&i) => i,
                                                _ => continue,
                                            };
                                            if focal.contains(&li) {
                                                neighbors.insert(ri);
                                            }
                                            if focal.contains(&ri) {
                                                neighbors.insert(li);
                                            }
                                        }
                                        focal.extend(neighbors);
                                    }
                                    focal
                                });
                            lr_overlay::render_lr_overlays_for_core(
                                args,
                                &frame,
                                &cells,
                                core,
                                &lr,
                                lr_expr.as_ref(),
                                focal_set.as_ref(),
                                &hulls_by_community,
                                &lr_dir,
                            )
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    for v in per_core {
                        emitted.extend(v);
                    }
                }
                Err(e) => log::warn!("LR-activity JSON load failed ({}): {e}", p.display()),
            }
        }
    }

    write_manifest(&out_prefix, &emitted)?;
    info!(
        "wrote {} files total → {}.plot.manifest.json",
        emitted.len(),
        out_prefix
    );
    Ok(())
}

/// Align propensity rows (indexed by prop-parquet's row-names) to the
/// global cell table's order. Missing cells get NaN row.
fn align_propensity(cells: &CellTable, prop_cell_names: &[Box<str>], prop: &Mat) -> Mat {
    let n = cells.n();
    let k = prop.ncols();
    let mut out = Mat::from_element(n, k, f32::NAN);
    for (src_row, name) in prop_cell_names.iter().enumerate() {
        if let Some(&dst_row) = cells.index.get(name) {
            for j in 0..k {
                out[(dst_row, j)] = prop[(src_row, j)];
            }
        }
    }
    out
}

fn align_dominant(cells: &CellTable, prop_cell_names: &[Box<str>], dominant: &[i64]) -> Vec<i64> {
    let mut out = vec![-1i64; cells.n()];
    for (src_row, name) in prop_cell_names.iter().enumerate() {
        if let Some(&dst_row) = cells.index.get(name) {
            out[dst_row] = dominant[src_row];
        }
    }
    out
}

fn resolve_lr_json_path(args: &SrtPlotArgs, prefix: &str) -> Option<PathBuf> {
    if let Some(p) = args.lr_activity_json.as_deref() {
        return Some(PathBuf::from(p));
    }
    let meta_path = PathBuf::from(format!("{prefix}.metadata.json"));
    crate::util::metadata::PintoMetadata::read(&meta_path)
        .ok()
        .and_then(|m| m.outputs.lr_activity.map(PathBuf::from))
}

fn align_entropy(cells: &CellTable, prop_cell_names: &[Box<str>], entropy: &[f32]) -> Vec<f32> {
    let mut out = vec![f32::NAN; cells.n()];
    for (src_row, name) in prop_cell_names.iter().enumerate() {
        if let Some(&dst_row) = cells.index.get(name) {
            out[dst_row] = entropy[src_row];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn emit_marker_figures(
    args: &SrtPlotArgs,
    frame: &Frame,
    cells: &CellTable,
    core: &partition::CoreSpec,
    dominant: &[i64],
    gt: &Mat,
    gene_names: &[Box<str>],
    data: &SparseIoVec,
    cell_col_index: &HashMap<Box<str>, usize>,
    colors: &ColorBook,
    level_tag: &str,
    out_prefix: &str,
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    // For each topic, gather its top-N markers up-front.
    let mut plan: Vec<(usize /*k*/, Box<str>)> = Vec::new();
    for k in 0..gt.ncols() {
        for (_, gname) in markers::top_n_markers(gt, gene_names, k, args.top_markers) {
            plan.push((k, gname));
        }
    }
    if plan.is_empty() {
        return Ok(());
    }

    // Unique marker names (different topics may pick the same gene).
    let mut uniq_names: Vec<Box<str>> = plan.iter().map(|(_, g)| g.clone()).collect();
    uniq_names.sort();
    uniq_names.dedup();
    let name_to_local: HashMap<Box<str>, usize> = uniq_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Map core.cell_ixs (CellTable-space) → data-beans column indices
    // via cell-name lookup; cells missing from --data stay None (slot
    // renders at 0.0) so output positions still align with core.cell_ixs.
    let data_col_ixs: Vec<Option<usize>> = core
        .cell_ixs
        .iter()
        .map(|&i| cell_col_index.get(&cells.names[i]).copied())
        .collect();
    let present_count = data_col_ixs.iter().filter(|x| x.is_some()).count();
    if present_count == 0 {
        log::warn!("[{level_tag}] no cells found in --data; skipping markers");
        return Ok(());
    }

    // Single thin (markers × cells) slab per backend via SparseIo::read_rows_*.
    let rows = markers::fetch_gene_rows_aligned(data, &uniq_names, &data_col_ixs)?;

    for (k, gname) in plan {
        let local = name_to_local[&gname];
        let expr = &rows[local];

        // (a) Heatmap (viridis color, fixed size)
        let layers = build_marker_heatmap_layers(
            frame,
            cells,
            core,
            expr,
            args.heat_bins,
            args.alpha,
            args.point_shape,
            args.expr_clip,
        )?;
        let out_stub = marker_out_path(out_prefix, level_tag, k, &gname, "heatmap");
        emit_figure(&layers, frame, args, &out_stub, emitted)?;

        // (b) Community-colored (argmax color, size ∝ expression)
        let layers = build_marker_by_community_layers(
            frame,
            cells,
            core,
            dominant,
            expr,
            colors,
            args.point_shape,
            args.alpha,
            args.size_scale,
            args.expr_clip,
        )?;
        let out_stub = marker_out_path(out_prefix, level_tag, k, &gname, "by-community");
        emit_figure(&layers, frame, args, &out_stub, emitted)?;
    }
    Ok(())
}

fn marker_out_path(
    out_prefix: &str,
    level_tag: &str,
    k: usize,
    gname: &str,
    kind: &str,
) -> PathBuf {
    let safe_gname: String = gname
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            c => c,
        })
        .collect();
    let plot_dir = PathBuf::from(format!("{}.plots", out_prefix));
    plot_dir
        .join(PlotKind::Markers.subdir())
        .join(format!("{level_tag}.topic{k}.{safe_gname}.{kind}"))
}

fn write_manifest(out_prefix: &str, emitted: &[PathBuf]) -> anyhow::Result<()> {
    let json = json!({
        "out_prefix": out_prefix,
        "file_count": emitted.len(),
        "files": emitted.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
    });
    let path = format!("{out_prefix}.plot.manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}

use load::CellTable;
