#![allow(clippy::too_many_arguments)]
//! `pinto plot` — publication-quality spatial scatter for link-community /
//! dsvd / propensity outputs, with optional marker-gene overlays.
//!
//! Shares rasterizer / palette / SVG emission with senna via
//! `plot-utils`. The only pinto-specific pieces live here:
//!
//! - [`discover`] — find `{prefix}.L*.propensity.parquet` siblings
//! - [`load`] — read `coord_pairs.parquet` + propensity +
//!   link_community + gene_community
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
    Propensity,
    Mesh,
    Interfaces,
    Markers,
    Lr,
}

impl PlotKind {
    const ALL: &'static [PlotKind] = &[
        PlotKind::Propensity,
        PlotKind::Mesh,
        PlotKind::Interfaces,
        PlotKind::Markers,
        PlotKind::Lr,
    ];

    fn subdir(self) -> &'static str {
        match self {
            PlotKind::Propensity => "propensity",
            PlotKind::Mesh => "mesh",
            PlotKind::Interfaces => "interfaces",
            PlotKind::Markers => "markers",
            PlotKind::Lr => "lr",
        }
    }
}

/// Coarse level classifier driving the per-level emission profile.
///
/// - `Final`: full suite (propensity / per-community heatmaps / mesh /
///   interfaces / markers).
/// - `Intermediate` (`L*`): only the propensity argmax — emitting
///   per-community heatmaps × levels would balloon the output dir.
/// - `Draft`: mesh + propensity argmax. Drafts are for inspection, not
///   publication.
#[derive(Copy, Clone, PartialEq, Eq)]
enum LevelKind {
    Final,
    Intermediate,
    Draft,
}

impl LevelKind {
    fn from_tag(tag: &str) -> Self {
        match tag {
            "final" => LevelKind::Final,
            "draft" => LevelKind::Draft,
            _ => LevelKind::Intermediate,
        }
    }
}

use discover::{discover_levels, LevelSelector};
use load::{
    read_cells_from_coord_pairs, read_gene_community, read_link_community, read_propensity,
};
use partition::partition_cells;
use render::{
    build_marker_heatmap_layers, build_mesh_layers, build_propensity_argmax_layers,
    build_propensity_community_heatmap_layers, emit_figure, ColorBook, Frame,
};

/// Entry point: auto-discover outputs, partition, emit figures, write
/// a JSON manifest listing everything produced.
pub fn make_srt_plot(args: &SrtPlotArgs) -> anyhow::Result<()> {
    let resolved = resolve_from(&args.from)?;
    let prefix = resolved.prefix;
    let coord_columns_hint = resolved.coord_columns;
    let meta_data_files = resolved.data_files;
    let lra_only = resolved.lra_only;
    let lr_json_override = resolved.lr_json_override;

    let out_prefix = args.out.as_deref().unwrap_or(&prefix).to_string();

    // LRA mode focuses on the final-level partition (interfaces + LR
    // overlays only) — intermediate L* levels carry the same info.
    let selector = if lra_only {
        LevelSelector::parse("final")
    } else {
        LevelSelector::parse(&args.levels)
    };
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

    // One gene_community (shared across all levels) — fine for now; if a
    // level carries its own, we swap below.
    let final_gene_community_path = PathBuf::from(format!("{prefix}.gene_community.parquet"));
    let global_gt = if final_gene_community_path.exists() {
        Some(read_gene_community(&final_gene_community_path)?)
    } else {
        None
    };

    // Open expression data once. Used by markers (--top-markers > 0) AND
    // by the LR-overlay direction inference (per-edge L/R expression).
    // Falls back to the data_files recorded in metadata.json when --data
    // is omitted, so `pinto plot --from foo.metadata.json` is enough to
    // get marker plots without re-passing the original data paths.
    let resolved_data_files: Option<Vec<Box<str>>> = match (&args.data, &meta_data_files) {
        (Some(d), _) => Some(d.clone()),
        (None, Some(meta)) if !meta.is_empty() => {
            info!(
                "--data not given; using {} data file(s) from metadata.json",
                meta.len(),
            );
            Some(meta.iter().map(|s| s.clone().into_boxed_str()).collect())
        }
        _ => None,
    };
    let expr_data = match &resolved_data_files {
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
    if lra_only {
        std::fs::create_dir_all(plot_dir.join(PlotKind::Interfaces.subdir()))?;
        std::fs::create_dir_all(plot_dir.join(PlotKind::Lr.subdir()))?;
    } else {
        for kind in PlotKind::ALL {
            std::fs::create_dir_all(plot_dir.join(kind.subdir()))?;
        }
    }

    levels
        .par_iter()
        .try_for_each(|level| -> anyhow::Result<()> {
            let prop_path = level.propensity_path(&prefix);
            let (propensity, dominant, entropy_opt, prop_cell_names) =
                read_propensity(&prop_path, &excluded)?;

            // Align propensity rows → global cell index.
            let mut aligned_dominant = align_dominant(&cells, &prop_cell_names, &dominant);
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

            // Drop sparse communities: cells whose dominant community has
            // fewer than --min-edges-per-community edges get dominant = -1
            // and are skipped by hulls / markers / LR overlays downstream.
            if let Some((_, edge_community)) = lc_pair.as_ref() {
                let max_c = edge_community.iter().copied().max().unwrap_or(-1);
                if max_c >= 0 {
                    let mut counts = vec![0usize; (max_c + 1) as usize];
                    for &c in edge_community {
                        if c >= 0 {
                            counts[c as usize] += 1;
                        }
                    }
                    let n_dropped = counts
                        .iter()
                        .filter(|&&n| 0 < n && n < args.min_edges_per_community)
                        .count();
                    if n_dropped > 0 {
                        info!(
                            "[{}] dropping {} sparse communities (< {} edges)",
                            level.tag, n_dropped, args.min_edges_per_community
                        );
                    }
                    for c_ref in aligned_dominant.iter_mut() {
                        let c = *c_ref;
                        if c >= 0
                            && (c as usize) < counts.len()
                            && counts[c as usize] < args.min_edges_per_community
                        {
                            *c_ref = -1;
                        }
                    }
                }
            }

            // Per-level gene_community (fall back to global).
            let level_gt_path = level.gene_community_path(&prefix);
            let gene_community = if level_gt_path.exists() {
                Some(read_gene_community(&level_gt_path)?)
            } else {
                global_gt.clone()
            };

            let level_kind = LevelKind::from_tag(&level.tag);

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

                // Propensity argmax (every level): color = argmax community, size ∝ propensity.
                if !lra_only {
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
                        &kind_path(PlotKind::Propensity, "argmax.propensity"),
                        &mut local_emitted,
                        true,
                    )?;
                }

                // 3. Per-community soft-membership heatmaps (final only)
                if !lra_only && level_kind == LevelKind::Final {
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
                            &kind_path(PlotKind::Propensity, &format!("C{kk}")),
                            &mut local_emitted,
                            true,
                        )?;
                    }
                }

                // 4. Mesh (cell-cell edges colored by community) — final + draft
                if !lra_only
                    && !args.no_mesh
                    && matches!(level_kind, LevelKind::Final | LevelKind::Draft)
                {
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
                            false,
                        )?;
                    }
                }

                // 4b. Interfaces (high-entropy + neighborhoods) — final only.
                // LRA-only mode forces interfaces on (LR analysis cares about
                // boundary cells; rendering them is the whole point).
                if level_kind == LevelKind::Final && (args.show_interfaces || lra_only) {
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
                                gene_community.as_ref(),
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

                // Marker genes — final + draft only (intermediate skips per LevelKind).
                if !lra_only && level_kind != LevelKind::Intermediate && args.top_markers > 0 {
                    let (gt, gene_names) = match &gene_community {
                        Some(x) => x,
                        None => {
                            log::warn!(
                                "[{}] no gene_community parquet — skipping markers",
                                level.tag,
                            );
                            continue;
                        }
                    };
                    let data = match &expr_data {
                        Some(d) => d,
                        None => {
                            log::warn!(
                                "[{}] --data not supplied — skipping markers \
                                 (pass `--data <expr.h5/.zarr>` to enable)",
                                level.tag,
                            );
                            continue;
                        }
                    };
                    let ccol_ix = cell_col_index
                        .as_ref()
                        .expect("cell index built alongside data");

                    let marker_hulls_by_c: HashMap<i64, Vec<plot_utils::svg_emit::TopicLayer>> =
                        HashMap::default();

                    emit_marker_figures(
                        args,
                        &frame,
                        &cells,
                        core,
                        &aligned_dominant,
                        &colors,
                        gt,
                        gene_names,
                        data,
                        ccol_ix,
                        &marker_hulls_by_c,
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
    // In `lra_only` mode the user passed an `lr_activity.json` directly,
    // so the overlay is the whole point — `--no-lr-overlay` is ignored.
    let render_lr = lra_only || !args.no_lr_overlay;
    let lr_json_path = lr_json_override
        .clone()
        .or_else(|| resolve_lr_json_path(args, &prefix));
    if render_lr {
        if let Some(p) = lr_json_path {
            match lr_overlay::load_lr_json(&p) {
                Ok(lr) => {
                    let n_sig = lr.results.iter().filter(|r| r.significant).count();
                    info!(
                        "Rendering LR overlays from {} ({} strata, {} results, {} significant)",
                        p.display(),
                        lr.strata.len(),
                        lr.results.len(),
                        n_sig,
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
                                        let mut frontier: HashSet<usize> = focal.clone();
                                        for _ in 0..args.lr_belt_hops {
                                            if frontier.is_empty() {
                                                break;
                                            }
                                            let mut next: HashSet<usize> = HashSet::default();
                                            for (l, r) in edges {
                                                let li = match cells.index.get(l) {
                                                    Some(&i) if core_set.contains(&i) => i,
                                                    _ => continue,
                                                };
                                                let ri = match cells.index.get(r) {
                                                    Some(&i) if core_set.contains(&i) => i,
                                                    _ => continue,
                                                };
                                                if frontier.contains(&li) && !focal.contains(&ri) {
                                                    next.insert(ri);
                                                }
                                                if frontier.contains(&ri) && !focal.contains(&li) {
                                                    next.insert(li);
                                                }
                                            }
                                            focal.extend(next.iter().copied());
                                            frontier = next;
                                        }
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
                                final_dominant.as_deref(),
                                &lr_dir,
                            )
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    for v in per_core {
                        emitted.extend(v);
                    }
                    // Read the lr-activity parquet sibling once and share
                    // it across the three summary emitters — each one used
                    // to reopen the file on its own.
                    let lr_parquet = lr_overlay::resolve_lr_parquet_path(&p)
                        .and_then(|pp| lr_overlay::read_lr_activity_parquet(&pp).ok());
                    let lr_parquet_slice = lr_parquet.as_deref();
                    let k_total = final_aligned_prop.as_ref().map(|p| p.ncols());
                    // A community ID beyond the propensity's K is the
                    // tell that lr-activity was run against a different
                    // prefix (e.g. pre-merge `.draft.`) than plot —
                    // colours/labels can't be reconciled by the plot
                    // code in that case, only by the user re-running.
                    if let Some(k) = k_total {
                        let lr_max =
                            lr.results.iter().map(|r| r.community).max().unwrap_or(-1) as i64;
                        if lr_max >= k as i64 {
                            log::warn!(
                                "LR overlay: max community in {} is C{} but final propensity has K={}. \
                                 Likely the lr-activity JSON was computed against a different prefix \
                                 (e.g. pre-merge `.draft.`). Community indices in LR plots will not \
                                 match marker / propensity / mesh plots.",
                                p.display(),
                                lr_max,
                                k,
                            );
                        }
                    }
                    lr_overlay::emit_lr_summary_per_community(
                        args,
                        &lr,
                        lr_parquet_slice,
                        k_total,
                        &lr_dir,
                        &mut emitted,
                    )?;
                    lr_overlay::emit_lr_summary_global(
                        args,
                        &lr,
                        lr_parquet_slice,
                        k_total,
                        &lr_dir,
                        &mut emitted,
                    )?;
                    lr_overlay::emit_lr_bipartite(
                        args,
                        &lr,
                        lr_parquet_slice,
                        k_total,
                        &lr_dir,
                        &mut emitted,
                    )?;
                }
                Err(e) => {
                    log::warn!("LR-activity JSON load failed ({}): {e}", p.display())
                }
            }
        } else if lra_only {
            log::warn!("pinto plot (lra-only): no LR-activity JSON resolved");
        }
    }

    let manifest_suffix = if lra_only { "plot.lra" } else { "plot" };
    write_manifest(&out_prefix, manifest_suffix, &emitted)?;
    info!(
        "wrote {} files total → {}.{}.manifest.json",
        emitted.len(),
        out_prefix,
        manifest_suffix
    );
    Ok(())
}

struct ResolvedFrom {
    prefix: Box<str>,
    coord_columns: Option<Vec<String>>,
    data_files: Option<Vec<String>>,
    lra_only: bool,
    lr_json_override: Option<PathBuf>,
}

fn resolve_from(from: &str) -> anyhow::Result<ResolvedFrom> {
    if !from.ends_with(".json") {
        return Ok(ResolvedFrom {
            prefix: from.to_string().into_boxed_str(),
            coord_columns: None,
            data_files: None,
            lra_only: false,
            lr_json_override: None,
        });
    }
    let from_path = std::path::Path::new(from);
    let raw = std::fs::read_to_string(from_path)
        .map_err(|e| anyhow::anyhow!("reading {from_path:?}: {e}"))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("parsing {from_path:?} as JSON: {e}"))?;
    let cmd = value.get("command").and_then(|c| c.as_str()).unwrap_or("");
    if cmd == "lr-activity" {
        let lc_prefix = value
            .get("lc_prefix")
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow::anyhow!("{from_path:?}: missing `lc_prefix`"))?;
        let upstream_meta_path = value
            .get("input_metadata")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());
        let (coord_columns, data_files) = match upstream_meta_path
            .as_deref()
            .and_then(|p| crate::util::metadata::PintoMetadata::read(std::path::Path::new(p)).ok())
        {
            Some(m) => (m.outputs.coord_columns.clone(), m.data_files.clone()),
            None => (None, None),
        };
        return Ok(ResolvedFrom {
            prefix: lc_prefix.to_string().into_boxed_str(),
            coord_columns,
            data_files,
            lra_only: true,
            lr_json_override: Some(from_path.to_path_buf()),
        });
    }
    let meta = crate::util::metadata::PintoMetadata::read(from_path)?;
    Ok(ResolvedFrom {
        prefix: meta.prefix.into_boxed_str(),
        coord_columns: meta.outputs.coord_columns.clone(),
        data_files: meta.data_files.clone(),
        lra_only: false,
        lr_json_override: None,
    })
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
    colors: &ColorBook,
    gt: &Mat,
    gene_names: &[Box<str>],
    data: &SparseIoVec,
    cell_col_index: &HashMap<Box<str>, usize>,
    hulls_by_community: &HashMap<i64, Vec<plot_utils::svg_emit::TopicLayer>>,
    level_tag: &str,
    out_prefix: &str,
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    // For each community, gather its top-N markers up-front.
    let mut plan: Vec<(usize /*k*/, Box<str>)> = Vec::new();
    for k in 0..gt.ncols() {
        for (_, gname) in markers::top_n_markers(gt, gene_names, k, args.top_markers) {
            plan.push((k, gname));
        }
    }
    if plan.is_empty() {
        return Ok(());
    }

    // Unique marker names (different communities may pick the same gene).
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

        // Skip markers that fire in too few cells — the heatmap is
        // mostly empty and the resulting plot reads as a sparse dust
        // cloud (see image-9 from the 2026-04-27 review). Threshold
        // is on the fraction of *core* cells with non-zero expression.
        let n_nonzero = expr.iter().filter(|v| v.is_finite() && **v > 0.0).count();
        let n_core = core.cell_ixs.len().max(1);
        if (n_nonzero as f32) / (n_core as f32) < args.marker_min_frac {
            log::info!(
                "[{level_tag}] community {k}: skipping {gname} ({n_nonzero}/{n_core} = {:.2}% < --marker-min-frac {:.2}%)",
                100.0 * n_nonzero as f32 / n_core as f32,
                100.0 * args.marker_min_frac,
            );
            continue;
        }

        // (a) Heatmap (viridis color, fixed size)
        let mut layers = build_marker_heatmap_layers(
            frame,
            cells,
            core,
            expr,
            args.heat_bins,
            args.alpha,
            args.point_shape,
            args.expr_clip,
        )?;
        if let Some(h) = hulls_by_community.get(&(k as i64)) {
            layers.extend(h.iter().cloned());
        }
        let out_stub = marker_out_path(out_prefix, level_tag, k, &gname);
        emit_figure(&layers, frame, args, &out_stub, emitted, false)?;
    }

    emit_marker_summary(
        args,
        cells,
        core,
        dominant,
        colors,
        &uniq_names,
        &rows,
        level_tag,
        out_prefix,
        emitted,
    )?;
    Ok(())
}

/// Hinton-style summary: union of top markers (rows) × communities (columns),
/// box area encodes mean expression of that gene in cells dominated by that
/// community. Rows and columns are diagonalized so high-activity blocks line
/// up along the main diagonal.
#[allow(clippy::too_many_arguments)]
fn emit_marker_summary(
    args: &SrtPlotArgs,
    cells: &CellTable,
    core: &partition::CoreSpec,
    dominant: &[i64],
    colors: &ColorBook,
    gene_names: &[Box<str>],
    expr_rows: &[Vec<f32>],
    level_tag: &str,
    out_prefix: &str,
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    let n_genes = gene_names.len();
    let k = colors.k();
    if n_genes == 0 || k == 0 {
        return Ok(());
    }

    // Mean expression per (gene, community), where each cell contributes to
    // its argmax community only. Cells with no community (-1 / out of range)
    // are dropped.
    let mut mean = vec![0.0f32; n_genes * k];
    let mut cnt = vec![0u32; k];
    for (local, &i) in core.cell_ixs.iter().enumerate() {
        let c = dominant.get(i).copied().unwrap_or(-1);
        if c < 0 || (c as usize) >= k {
            continue;
        }
        let c = c as usize;
        cnt[c] += 1;
        for (g, row) in expr_rows.iter().enumerate() {
            if let Some(&v) = row.get(local) {
                if v.is_finite() {
                    mean[g * k + c] += v;
                }
            }
        }
    }
    for c in 0..k {
        if cnt[c] == 0 {
            continue;
        }
        let inv = 1.0 / cnt[c] as f32;
        for g in 0..n_genes {
            mean[g * k + c] *= inv;
        }
    }

    let (row_order, col_order) = plot_utils::diagonalize_order(&mean, n_genes, k);
    let _ = cells; // kept for symmetry with other figure emitters; unused

    let col_labels: Vec<Box<str>> = (0..k).map(|c| format!("C{c}").into_boxed_str()).collect();
    let col_colors: Vec<plot_utils::Rgb> = (0..k).map(|c| colors.color(c)).collect();
    // Publish the C{c} → palette colour mapping in the legend so the user
    // can verify community indices match across propensity / mesh / LR
    // plots. Only communities that have any cells assigned are listed.
    let color_legend: Vec<(Box<str>, plot_utils::Rgb)> = (0..k)
        .filter(|&c| cnt[c] > 0)
        .map(|c| (format!("C{c}").into_boxed_str(), colors.color(c)))
        .collect();

    let opts = plot_utils::HintonOpts {
        row_labels: Some(gene_names),
        col_labels: Some(&col_labels),
        row_order: Some(&row_order),
        col_order: Some(&col_order),
        col_colors: Some(&col_colors),
        cell_colors: None,
        scale: plot_utils::HintonScale::Sqrt,
        cell_px: 18.0,
        font_px: 11.0,
        title: Some(&format!(
            "Marker × community (mean expr) — {} cells, level {}",
            core.n(),
            level_tag,
        )),
        grid_stroke_px: 0.4,
        grid_color: (220, 220, 220),
        color_legend: Some(&color_legend),
    };
    let svg = plot_utils::render_hinton(&mean, n_genes, k, &opts);

    let plot_dir = PathBuf::from(format!("{}.plots", out_prefix));
    let stub = plot_dir
        .join(PlotKind::Markers.subdir())
        .join(format!("{level_tag}.summary"));

    let with_ext = |ext: &str| -> PathBuf { PathBuf::from(format!("{}.{ext}", stub.display())) };

    if args.svg {
        let p = with_ext("svg");
        std::fs::write(&p, svg.as_bytes())?;
        emitted.push(p);
    }
    if !args.no_pdf {
        let p = with_ext("pdf");
        plot_utils::render_pdf(&svg, &p)?;
        emitted.push(p);
    }
    if args.png {
        let p = with_ext("png");
        let size = plot_utils::hinton_size(n_genes, k, &opts);
        plot_utils::render_png(&svg, size.width_px, size.height_px, &p)?;
        emitted.push(p);
    }
    Ok(())
}

fn marker_out_path(out_prefix: &str, level_tag: &str, k: usize, gname: &str) -> PathBuf {
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
        .join(format!("{level_tag}.C{k}.{safe_gname}"))
}

fn write_manifest(out_prefix: &str, suffix: &str, emitted: &[PathBuf]) -> anyhow::Result<()> {
    let json = json!({
        "out_prefix": out_prefix,
        "file_count": emitted.len(),
        "files": emitted.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
    });
    let path = format!("{out_prefix}.{suffix}.manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}

use load::CellTable;
