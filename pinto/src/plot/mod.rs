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
pub mod load;
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
    // Auto-detect if --from is a JSON metadata file or a prefix
    let prefix = if args.from.ends_with(".json") {
        let meta_path = std::path::Path::new(args.from.as_ref());
        let meta = crate::util::metadata::PintoMetadata::read(meta_path)?;
        meta.prefix.into_boxed_str()
    } else {
        args.from.clone()
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
    let cells = read_cells_from_coord_pairs(&coord_path)?;
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

    // Open expression data once (read-only; shared across cores / markers).
    let expr_data = match (&args.data, args.top_markers) {
        (Some(files), n) if n > 0 => Some(read_expr_data(files)?),
        _ => None,
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

    // Create output subdirectories once before parallel iteration
    let plot_dir = PathBuf::from(format!("{}.plots", out_prefix));
    std::fs::create_dir_all(&plot_dir)?;
    let markers_dir = plot_dir.join("markers");
    std::fs::create_dir_all(&markers_dir)?;

    levels
        .par_iter()
        .try_for_each(|level| -> anyhow::Result<()> {
            let prop_path = level.propensity_path(&prefix);
            let (propensity, dominant, prop_cell_names) = read_propensity(&prop_path, &excluded)?;

            // Align propensity rows → global cell index.
            let aligned_dominant = align_dominant(&cells, &prop_cell_names, &dominant);
            let aligned_propensity = align_propensity(&cells, &prop_cell_names, &propensity);

            let k = aligned_propensity.ncols();
            let colors = ColorBook::new(args, k);

            // Per-level link_community (skip if absent — dsvd output).
            let lc_path = level.link_community_path(&prefix);
            let lc_pair = if !args.no_mesh && lc_path.exists() {
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
                let out_stub = |kind: &str| -> PathBuf {
                    plot_dir.join(format!(
                        "{level}.core{batch}.{kind}",
                        level = level.tag,
                        batch = core.name,
                    ))
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
                    &out_stub("community"),
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
                    &out_stub("propensity.argmax"),
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
                        &out_stub(&format!("propensity.community{kk}")),
                        &mut local_emitted,
                    )?;
                }

                // 4. Mesh (cell-cell edges colored by community) if lc data present
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
                    emit_figure(&layers, &frame, args, &out_stub("mesh"), &mut local_emitted)?;
                }

                // 5. Marker genes (requires data + gene_topic)
                if args.top_markers > 0 {
                    let (gt, gene_names) = match &gene_topic {
                        Some(x) => x,
                        None => {
                            info!(
                                "[{}/{}] no gene_topic parquet — skipping markers",
                                level.tag, core.name
                            );
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
                        &core.name,
                        &out_prefix,
                        &mut local_emitted,
                    )?;
                }

                info!(
                    "[{}/{}] wrote {} files",
                    level.tag,
                    core.name,
                    local_emitted.len()
                );
                emitted.lock().expect("emitted lock").extend(local_emitted);
            }

            Ok(())
        })?;

    let emitted = emitted.into_inner().expect("emitted lock");
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
    core_name: &str,
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
        log::warn!("[{level_tag}/{core_name}] no core cells found in --data; skipping markers");
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
        let out_stub = marker_out_path(out_prefix, level_tag, core_name, k, &gname, "heatmap");
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
        let out_stub = marker_out_path(out_prefix, level_tag, core_name, k, &gname, "by-community");
        emit_figure(&layers, frame, args, &out_stub, emitted)?;
    }
    Ok(())
}

fn marker_out_path(
    out_prefix: &str,
    level_tag: &str,
    core_name: &str,
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
    plot_dir.join("markers").join(format!(
        "{level_tag}.core{core_name}.topic{k}.{safe_gname}.{kind}"
    ))
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
