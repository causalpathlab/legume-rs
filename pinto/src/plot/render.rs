//! Per-(level, core) figure orchestration.
//!
//! All plot variants funnel through one emitter, [`emit_figure`], which
//! renders a set of `TopicLayer`s to SVG + (optionally) PNG + PDF. The
//! per-variant builders below (`build_community_layers`,
//! `build_propensity_argmax_layers`, etc.) each return a
//! `Vec<TopicLayer>` plus optional metadata so the shared emitter
//! handles every I/O concern.

#![allow(clippy::too_many_arguments)]

use super::args::SrtPlotArgs;
use super::load::CellTable;
use super::partition::CoreSpec;
use super::viridis;
use crate::util::common::*;
use plot_utils::hull::Pt;
use plot_utils::palette::{self, Palette, Rgb};
use plot_utils::rasterize::{
    rasterize_group_png, rasterize_segment_layer_png, DataBounds, Extent, PointShape, RadiusSpec,
};
use plot_utils::svg_emit::{emit_svg, SvgOpts, TopicLayer};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const PT_PER_INCH: f32 = 72.0;

/// Compact view of the per-core plot frame — bounds, pixel extent, and
/// scaled radii so every builder starts from identical geometry.
pub struct Frame {
    pub bounds: DataBounds,
    pub extent: Extent,
    pub radius_px_base: f32,
    pub label_font_px: f32,
    pub stroke_px: f32,
}

impl Frame {
    pub fn new(core: &CoreSpec, args: &SrtPlotArgs) -> Self {
        let (extent, bounds) = compute_extent(core, args);
        let radius_px_base = args.point_size * args.dpi as f32 / PT_PER_INCH / 2.0;
        let label_font_px = 10.0 * args.dpi as f32 / PT_PER_INCH;
        let stroke_px = args.mesh_stroke * args.dpi as f32 / PT_PER_INCH;
        Frame {
            bounds,
            extent,
            radius_px_base,
            label_font_px,
            stroke_px,
        }
    }
}

/// Derive Extent + (possibly inflated) DataBounds from core bounds +
/// user width/dpi/max-aspect. Cells outside bounds are kept (they clip
/// in `to_pixel`); we just refuse to let them warp the frame.
fn compute_extent(core: &CoreSpec, args: &SrtPlotArgs) -> (Extent, DataBounds) {
    let dx = (core.bounds.xmax - core.bounds.xmin).max(1e-6);
    let dy = (core.bounds.ymax - core.bounds.ymin).max(1e-6);
    let raw_ar = (dy / dx).clamp(1.0 / args.max_aspect, args.max_aspect);

    // If the clamp bit, inflate the narrower axis symmetrically so the
    // view still covers all cells.
    let mut bounds = core.bounds;
    let actual_ar = dy / dx;
    if actual_ar != raw_ar {
        if actual_ar > raw_ar {
            // y too tall → widen x.
            let want_dx = dy / raw_ar;
            let mid = 0.5 * (bounds.xmax + bounds.xmin);
            bounds.xmin = mid - 0.5 * want_dx;
            bounds.xmax = mid + 0.5 * want_dx;
        } else {
            // y too short → taller y.
            let want_dy = dx * raw_ar;
            let mid = 0.5 * (bounds.ymax + bounds.ymin);
            bounds.ymin = mid - 0.5 * want_dy;
            bounds.ymax = mid + 0.5 * want_dy;
        }
    }

    let width_in = args.width.max(0.5);
    let height_in = width_in * raw_ar;
    let w = (width_in * args.dpi as f32).round() as u32;
    let h = (height_in * args.dpi as f32).round().max(8.0) as u32;
    (Extent { w, h }, bounds)
}

/// Shared "layers + labels → SVG → (PNG, PDF)" emitter. Builders below
/// construct `TopicLayer`s; this handles every file-system concern.
pub fn emit_figure(
    layers: &[TopicLayer],
    frame: &Frame,
    args: &SrtPlotArgs,
    out_base: &Path,
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    // Drop empty layers (builders may return placeholders for vacant bins).
    let kept: Vec<TopicLayer> = layers
        .iter()
        .filter(|l| !l.png.is_empty())
        .cloned()
        .collect();
    if kept.is_empty() {
        return Ok(());
    }
    let svg = emit_svg(
        &kept,
        &SvgOpts {
            width_px: frame.extent.w,
            height_px: frame.extent.h,
            draw_hulls: false,
            draw_labels: !layers.iter().all(|l| l.label.is_empty()),
            label_font_size_px: frame.label_font_px,
            hull_stroke_px: frame.stroke_px.max(1.0),
            hull_fill_alpha: 0.0,
        },
    );

    let with_ext = |ext: &str| -> PathBuf {
        // out_base is a "stub" like `…coreall.community`; filenames may
        // contain dots already, so we append rather than call
        // `with_extension`, which replaces the last `.ext` segment.
        PathBuf::from(format!("{}.{ext}", out_base.display()))
    };

    if !args.no_svg {
        let p = with_ext("svg");
        std::fs::write(&p, svg.as_bytes())?;
        emitted.push(p);
    }

    let png_task = (!args.no_png).then(|| with_ext("png"));
    let pdf_task = (!args.no_pdf).then(|| with_ext("pdf"));

    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => plot_utils::render_png(&svg, frame.extent.w, frame.extent.h, p)
                .map(|()| Some(p.clone())),
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => plot_utils::render_pdf(&svg, p).map(|()| Some(p.clone())),
            None => Ok(None),
        },
    );
    if let Some(p) = png_res? {
        emitted.push(p);
    }
    if let Some(p) = pdf_res? {
        emitted.push(p);
    }
    Ok(())
}

/// Map a set of cell indices into pixel-space points for the given frame.
fn cells_to_pixels(frame: &Frame, cells: &CellTable, cell_ixs: &[usize]) -> Vec<Pt> {
    cell_ixs
        .iter()
        .map(|&i| frame.bounds.to_pixel(cells.coords[i], frame.extent))
        .collect()
}

/// Per-community palette resolved once per core.
pub struct ColorBook {
    palette: Palette,
    k: usize,
}

impl ColorBook {
    pub fn new(args: &SrtPlotArgs, k: usize) -> Self {
        let palette = palette::resolve(&args.palette.clone().unwrap_or(Palette::Auto), k.max(1));
        Self { palette, k }
    }
    pub fn color(&self, i: usize) -> Rgb {
        palette::color(&self.palette, i)
    }
    pub fn k(&self) -> usize {
        self.k
    }
}

// ─── Builders: each returns a Vec<TopicLayer> ready for emit_figure ──

/// Community plot: one layer per community, fixed point size.
pub fn build_community_layers(
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    dominant: &[i64],
    colors: &ColorBook,
    shape: PointShape,
    alpha: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    let mut by_k: Vec<Vec<Pt>> = (0..colors.k()).map(|_| Vec::new()).collect();
    for &i in &core.cell_ixs {
        let k = dominant.get(i).copied().unwrap_or(-1);
        if k < 0 {
            continue;
        }
        let k = k as usize;
        if k >= by_k.len() {
            continue;
        }
        by_k[k].push(frame.bounds.to_pixel(cells.coords[i], frame.extent));
    }

    by_k.into_par_iter()
        .enumerate()
        .map(|(k, pts_px)| {
            if pts_px.is_empty() {
                return Ok(empty_layer(format!("C{k}")));
            }
            let color = colors.color(k);
            let png = rasterize_group_png(
                &pts_px,
                frame.extent,
                RadiusSpec::Scalar(frame.radius_px_base),
                color,
                alpha,
                shape,
            )?;
            Ok(TopicLayer {
                label: String::new(),
                png,
                hull_px: Vec::new(),
                label_xy_px: (f32::NAN, f32::NAN),
                color,
            })
        })
        .collect()
}

/// Propensity (argmax) plot: one layer per community, size ∝ propensity
/// at the cell's argmax topic. All cells rendered.
pub fn build_propensity_argmax_layers(
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    dominant: &[i64],
    propensity: &Mat,
    colors: &ColorBook,
    shape: PointShape,
    alpha: f32,
    size_scale: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    let mut by_k: Vec<(Vec<Pt>, Vec<f32>)> =
        (0..colors.k()).map(|_| (Vec::new(), Vec::new())).collect();

    for &i in &core.cell_ixs {
        let k = dominant.get(i).copied().unwrap_or(-1);
        if k < 0 || (k as usize) >= colors.k() {
            continue;
        }
        let k = k as usize;
        by_k[k]
            .0
            .push(frame.bounds.to_pixel(cells.coords[i], frame.extent));
        by_k[k].1.push(propensity[(i, k)]);
    }

    by_k.into_par_iter()
        .enumerate()
        .map(|(k, (pts_px, props))| {
            if pts_px.is_empty() {
                return Ok(empty_layer(format!("C{k}")));
            }
            let radii = viridis::prop_to_radii(&props, frame.radius_px_base, size_scale);
            let color = colors.color(k);
            let png = rasterize_group_png(
                &pts_px,
                frame.extent,
                RadiusSpec::Per(&radii),
                color,
                alpha,
                shape,
            )?;
            Ok(TopicLayer {
                label: String::new(),
                png,
                hull_px: Vec::new(),
                label_xy_px: (f32::NAN, f32::NAN),
                color,
            })
        })
        .collect()
}

/// Per-community propensity heatmap: ONE community per call. All cells
/// rendered, size ∝ that topic's propensity, color ∝ propensity (viridis
/// bin). Produces one PDF per (level, core, community) via the caller.
pub fn build_propensity_community_heatmap_layers(
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    propensity: &Mat,
    k: usize,
    bins: usize,
    alpha: f32,
    shape: PointShape,
    size_scale: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    let props: Vec<f32> = core.cell_ixs.iter().map(|&i| propensity[(i, k)]).collect();
    let pts_px = cells_to_pixels(frame, cells, &core.cell_ixs);
    let radii = viridis::prop_to_radii(&props, frame.radius_px_base, size_scale);
    bucket_by_viridis_layers(
        &pts_px, &radii, &props, frame, bins, alpha, shape,
        0.0, /* no clip — propensity is already ∈ [0,1] */
    )
}

/// Marker gene heatmap (type 1): color = viridis bin on log-scale
/// expression, point size fixed. Every cell drawn.
pub fn build_marker_heatmap_layers(
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    expr: &[f32],
    bins: usize,
    alpha: f32,
    shape: PointShape,
    expr_clip: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    let pts_px = cells_to_pixels(frame, cells, &core.cell_ixs);
    let radii = vec![frame.radius_px_base; pts_px.len()];
    bucket_by_viridis_layers(&pts_px, &radii, expr, frame, bins, alpha, shape, expr_clip)
}

/// Marker gene by-community (type 2): color = argmax community, size ∝
/// log-scale expression.
pub fn build_marker_by_community_layers(
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    dominant: &[i64],
    expr: &[f32],
    colors: &ColorBook,
    shape: PointShape,
    alpha: f32,
    size_scale: f32,
    expr_clip: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    let pts_px = cells_to_pixels(frame, cells, &core.cell_ixs);
    let radii = viridis::log_expr_to_radii(expr, frame.radius_px_base, size_scale, expr_clip);

    let mut by_k: Vec<(Vec<Pt>, Vec<f32>)> =
        (0..colors.k()).map(|_| (Vec::new(), Vec::new())).collect();
    for (local, &i) in core.cell_ixs.iter().enumerate() {
        let k = dominant.get(i).copied().unwrap_or(-1);
        if k < 0 || (k as usize) >= colors.k() {
            continue;
        }
        let k = k as usize;
        by_k[k].0.push(pts_px[local]);
        by_k[k].1.push(radii[local]);
    }

    by_k.into_par_iter()
        .enumerate()
        .map(|(k, (pts, rs))| {
            if pts.is_empty() {
                return Ok(empty_layer(format!("C{k}")));
            }
            let color = colors.color(k);
            let png = rasterize_group_png(
                &pts,
                frame.extent,
                RadiusSpec::Per(&rs),
                color,
                alpha,
                shape,
            )?;
            Ok(TopicLayer {
                label: String::new(),
                png,
                hull_px: Vec::new(),
                label_xy_px: (f32::NAN, f32::NAN),
                color,
            })
        })
        .collect()
}

/// Mesh plot: cell-cell edges colored by community. Edges with either
/// endpoint outside `core` are dropped so multi-batch runs don't draw
/// stray lines across cores.
pub fn build_mesh_layers(
    frame: &Frame,
    cells: &CellTable,
    core_cell_set: &HashSet<usize>,
    edges: &[(Box<str>, Box<str>)],
    community: &[i64],
    colors: &ColorBook,
    alpha: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    let mut segs_by_k: Vec<Vec<(Pt, Pt)>> = (0..colors.k()).map(|_| Vec::new()).collect();
    for ((l, r), &c) in edges.iter().zip(community) {
        if c < 0 || (c as usize) >= colors.k() {
            continue;
        }
        let li = match cells.index.get(l) {
            Some(&i) => i,
            None => continue,
        };
        let ri = match cells.index.get(r) {
            Some(&i) => i,
            None => continue,
        };
        if !core_cell_set.contains(&li) || !core_cell_set.contains(&ri) {
            continue;
        }
        let p0 = frame.bounds.to_pixel(cells.coords[li], frame.extent);
        let p1 = frame.bounds.to_pixel(cells.coords[ri], frame.extent);
        segs_by_k[c as usize].push((p0, p1));
    }

    segs_by_k
        .into_par_iter()
        .enumerate()
        .map(|(k, segs)| {
            if segs.is_empty() {
                return Ok(empty_layer(format!("C{k}")));
            }
            let color = colors.color(k);
            let png =
                rasterize_segment_layer_png(&segs, frame.extent, frame.stroke_px, color, alpha)?;
            Ok(TopicLayer {
                label: String::new(),
                png,
                hull_px: Vec::new(),
                label_xy_px: (f32::NAN, f32::NAN),
                color,
            })
        })
        .collect()
}

/// Shared impl: given pts + radii + intensity values, bucket into
/// viridis color bins with monotone-increasing alpha so dropout cells
/// read as background. Returns one layer per bin.
fn bucket_by_viridis_layers(
    pts_px: &[Pt],
    radii: &[f32],
    values: &[f32],
    frame: &Frame,
    bins: usize,
    alpha_max: f32,
    shape: PointShape,
    clip: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    let bins = bins.max(2);
    let buckets = viridis::standardize_log_to_bins(values, bins, clip);
    let mut by_bin: Vec<(Vec<Pt>, Vec<f32>)> =
        (0..bins).map(|_| (Vec::new(), Vec::new())).collect();
    for (i, &b) in buckets.iter().enumerate() {
        by_bin[b as usize].0.push(pts_px[i]);
        by_bin[b as usize]
            .1
            .push(*radii.get(i).unwrap_or(&frame.radius_px_base));
    }
    by_bin
        .into_par_iter()
        .enumerate()
        .map(|(b, (pts, rs))| {
            if pts.is_empty() {
                return Ok(empty_layer(format!("bin{b}")));
            }
            let color = viridis::viridis_bin(b, bins);
            // dropout bin 0 fades out; top bin full-alpha.
            let a_lo = 0.15;
            let frac = b as f32 / (bins - 1) as f32;
            let alpha = a_lo + (alpha_max - a_lo).max(0.0) * frac;
            let png = rasterize_group_png(
                &pts,
                frame.extent,
                RadiusSpec::Per(&rs),
                color,
                alpha,
                shape,
            )?;
            Ok(TopicLayer {
                label: String::new(),
                png,
                hull_px: Vec::new(),
                label_xy_px: (f32::NAN, f32::NAN),
                color,
            })
        })
        .collect()
}

fn empty_layer(label: impl Into<String>) -> TopicLayer {
    TopicLayer {
        label: label.into(),
        png: Vec::new(),
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: (0, 0, 0),
    }
}
