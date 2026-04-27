//! Spatial overlay of significant LR-activity pairs.
//!
//! Reads the `{prefix}.lr_activity.json` sidecar emitted by
//! `pinto lr-activity` and renders one PDF per (batch core × significant
//! LR pair). The figure is a faint gray scatter of all core cells with
//! the participating edges (from the pair's stratum) drawn on top in a
//! palette color. Title carries `batch / community / ligand→receptor /
//! z, fwer_wy`.

use super::args::SrtPlotArgs;
use super::load::CellTable;
use super::markers;
use super::partition::{sanitize, CoreSpec};
use super::render::{emit_figure, Frame};
use crate::util::common::*;
use plot_utils::hull::Pt;
use plot_utils::rasterize::{rasterize_arrow_layer_png, rasterize_group_png, RadiusSpec};
use plot_utils::svg_emit::TopicLayer;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Per-cell expression for L and R genes used by significant pairs.
/// Each Vec is indexed by the cell-table position (`cells.names`); cells
/// missing from `--data` get `0.0` so the orientation comparison falls
/// back to the canonical direction for those edges.
pub type LrExpression = HashMap<Box<str>, Vec<f32>>;

/// Walk the LR JSON's significant rows, collect the union of (ligand,
/// receptor) gene names, and pull a single per-cell expression vector
/// per gene aligned to `cells.names`. Returns `None` when no
/// significant pair was found.
pub fn prefetch_lr_expression(
    data: &SparseIoVec,
    cell_col_index: &HashMap<Box<str>, usize>,
    cells: &CellTable,
    lr: &LrJson,
) -> anyhow::Result<Option<LrExpression>> {
    let mut needed: HashSet<Box<str>> = HashSet::default();
    for r in &lr.results {
        if r.significant {
            let l = r.ligand_resolved.as_deref().unwrap_or(r.ligand.as_str());
            let rr = r
                .receptor_resolved
                .as_deref()
                .unwrap_or(r.receptor.as_str());
            needed.insert(l.to_string().into_boxed_str());
            needed.insert(rr.to_string().into_boxed_str());
        }
    }
    if needed.is_empty() {
        return Ok(None);
    }
    let names: Vec<Box<str>> = needed.into_iter().collect();
    let data_col_ixs: Vec<Option<usize>> = cells
        .names
        .iter()
        .map(|n| cell_col_index.get(n).copied())
        .collect();
    let rows = markers::fetch_gene_rows_aligned(data, &names, &data_col_ixs)?;
    Ok(Some(names.into_iter().zip(rows).collect()))
}

#[derive(Deserialize, Debug)]
pub struct LrJson {
    pub strata: Vec<LrStratum>,
    pub results: Vec<LrResult>,
}

#[derive(Deserialize, Debug)]
pub struct LrStratum {
    pub stratum_id: usize,
    #[allow(dead_code)]
    pub batch: String,
    #[allow(dead_code)]
    pub community: i32,
    pub edges: Vec<(Box<str>, Box<str>)>,
}

#[derive(Deserialize, Debug)]
pub struct LrResult {
    pub batch: String,
    pub community: i32,
    pub ligand: String,
    pub receptor: String,
    /// Backend-row-name versions of ligand/receptor (post gene resolution)
    /// for direct lookup against expression `row_names()`. Optional for
    /// backward compatibility with older JSON sidecars.
    #[serde(default)]
    pub ligand_resolved: Option<String>,
    #[serde(default)]
    pub receptor_resolved: Option<String>,
    #[serde(default)]
    pub z: Option<f32>,
    /// Westfall-Young FWER-adjusted p (current); falls back to `q_storey`
    /// / `q_bh` for older sidecars from earlier statistical pipelines.
    #[serde(default, alias = "q_storey", alias = "q_bh")]
    pub fwer_wy: Option<f32>,
    /// Defaults to `true` because new JSON sidecars only carry significant
    /// rows (full table is in `lr_activity.parquet`). Old sidecars still
    /// deserialize: they set this explicitly per row.
    #[serde(default = "default_true")]
    pub significant: bool,
    #[serde(default)]
    pub stratum_id: Option<usize>,
}

fn default_true() -> bool {
    true
}

pub fn load_lr_json(path: &Path) -> anyhow::Result<LrJson> {
    use anyhow::Context;
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?;
    let parsed: LrJson =
        serde_json::from_str(&s).with_context(|| format!("parsing LR JSON {path:?}"))?;
    Ok(parsed)
}

/// Render LR-activity overlays for one core. Returns the list of files
/// written (added to the plot manifest).
///
/// `focal_cells`, when present, restricts the rendered arrows to edges
/// whose source OR destination is a high-entropy "interface" cell, AND
/// renders those focal cells as a dark underlay (size scaled by entropy
/// rank) so each LR PDF inherits the salience of `interfaces.pdf`.
/// `entropy_aligned`, when present (length = `cells.n()`, NaN where
/// missing), drives that underlay's per-cell sizing.
#[allow(clippy::too_many_arguments)]
pub fn render_lr_overlays_for_core(
    args: &SrtPlotArgs,
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    lr: &LrJson,
    lr_expr: Option<&LrExpression>,
    focal_cells: Option<&HashSet<usize>>,
    hulls_by_community: &HashMap<i64, Vec<TopicLayer>>,
    out_dir: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut emitted = Vec::new();
    if args.no_lr_overlay {
        return Ok(emitted);
    }

    let mut by_id: HashMap<usize, &LrStratum> = HashMap::default();
    for s in &lr.strata {
        by_id.insert(s.stratum_id, s);
    }

    // Render every significant result; the per-pair filename carries
    // `B{batch}` to disambiguate cross-batch results in the same dir.
    let mut sig: Vec<&LrResult> = lr
        .results
        .iter()
        .filter(|r| r.significant && r.stratum_id.is_some())
        .collect();
    if sig.is_empty() {
        return Ok(emitted);
    }
    // Global top-N by |z|: 16 strata × per-stratum-cap was exhaustive.
    sig.sort_by(|a, b| {
        b.z.unwrap_or(0.0)
            .abs()
            .partial_cmp(&a.z.unwrap_or(0.0).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cap = args.lr_top_pairs.max(1);
    sig.truncate(cap);

    let core_cell_set: HashSet<usize> = core.cell_ixs.iter().copied().collect();

    // Single background layer: faint hex tiling of every core cell so the
    // tissue shape is visible. No focal-cell underlay — arrows are the
    // only foreground decoration.
    let bg_pts: Vec<Pt> = core
        .cell_ixs
        .iter()
        .map(|&i| frame.bounds.to_pixel(cells.coords[i], frame.extent))
        .collect();
    let bg_png = rasterize_group_png(
        &bg_pts,
        frame.extent,
        RadiusSpec::Scalar(frame.radius_px_base),
        (225, 225, 225),
        0.5,
        args.point_shape,
    )?;
    let bg_layer = TopicLayer {
        label: String::new(),
        png: bg_png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: (225, 225, 225),
    };

    for r in sig.into_iter() {
        let sid = match r.stratum_id {
            Some(s) => s,
            None => continue,
        };
        let stratum = match by_id.get(&sid) {
            Some(s) => *s,
            None => continue,
        };

        // Per-edge ligand-expression and receptor-expression vectors when
        // we have --data; otherwise fall back to canonical (left → right).
        let l_key = r.ligand_resolved.as_deref().unwrap_or(r.ligand.as_str());
        let r_key = r
            .receptor_resolved
            .as_deref()
            .unwrap_or(r.receptor.as_str());
        let l_expr = lr_expr.and_then(|m| m.get(l_key));
        let r_expr = lr_expr.and_then(|m| m.get(r_key));

        // Build full-edge arrows + per-edge coexpression scores.
        let mut segs: Vec<(Pt, Pt)> = Vec::new();
        let mut coexpr: Vec<f32> = Vec::new();
        for (l_name, r_name) in &stratum.edges {
            let li = match cells.index.get(l_name) {
                Some(&i) => i,
                None => continue,
            };
            let ri = match cells.index.get(r_name) {
                Some(&i) => i,
                None => continue,
            };
            if !core_cell_set.contains(&li) || !core_cell_set.contains(&ri) {
                continue;
            }
            if let Some(focal) = focal_cells {
                if !focal.contains(&li) && !focal.contains(&ri) {
                    continue;
                }
            }

            let (src, dst, ce_raw) = match (l_expr, r_expr) {
                (Some(le), Some(re)) => {
                    let l_li = le.get(li).copied().unwrap_or(0.0);
                    let r_ri = re.get(ri).copied().unwrap_or(0.0);
                    let l_ri = le.get(ri).copied().unwrap_or(0.0);
                    let r_li = re.get(li).copied().unwrap_or(0.0);
                    let canon = l_li + r_ri;
                    let rev = l_ri + r_li;
                    if canon == 0.0 && rev == 0.0 {
                        continue;
                    }
                    if rev > canon {
                        (ri, li, (l_ri.max(0.0) * r_li.max(0.0)).sqrt())
                    } else {
                        (li, ri, (l_li.max(0.0) * r_ri.max(0.0)).sqrt())
                    }
                }
                _ => (li, ri, f32::NAN),
            };

            let p_src = frame.bounds.to_pixel(cells.coords[src], frame.extent);
            let p_dst = frame.bounds.to_pixel(cells.coords[dst], frame.extent);
            segs.push((p_src, p_dst));
            coexpr.push(ce_raw);
        }
        if segs.is_empty() {
            continue;
        }
        // Center on the per-pair empirical mean over edges so the
        // diverging color ramp encodes "above / below typical edge of
        // this pair" — Jensen-clean, no marginal-mean artifact.
        let mut s = 0.0f32;
        let mut n_finite = 0u32;
        for &v in &coexpr {
            if v.is_finite() {
                s += v;
                n_finite += 1;
            }
        }
        if n_finite > 0 {
            let mean_ce = s / (n_finite as f32);
            for v in coexpr.iter_mut() {
                if v.is_finite() {
                    *v -= mean_ce;
                }
            }
        }

        // Consistent visual size across edges; per-edge color carries the
        // coexpression magnitude.
        let stroke = 2.5f32;
        let head_len = 11.0f32;
        let arrow_layers = build_coexpr_arrow_layers(
            &segs,
            &coexpr,
            frame.extent,
            stroke,
            head_len,
            args.lr_coexpr_bins,
        )?;

        let stub = out_dir.join(format!(
            "B{}.C{}.{}-{}",
            sanitize(&r.batch),
            r.community,
            sanitize(&r.ligand),
            sanitize(&r.receptor),
        ));
        let label = format!(
            "{}->{}; B={}; C={}; z={:.2}; q={:.3}",
            r.ligand,
            r.receptor,
            r.batch,
            r.community,
            r.z.unwrap_or(f32::NAN),
            r.fwer_wy.unwrap_or(f32::NAN),
        );
        let pair_hulls: &[TopicLayer] = hulls_by_community
            .get(&(r.community as i64))
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let mut layers: Vec<TopicLayer> =
            Vec::with_capacity(1 + arrow_layers.len() + pair_hulls.len());
        layers.push(bg_layer.clone());
        for (i, l) in arrow_layers.into_iter().enumerate() {
            let mut tl = l;
            if i == 0 {
                tl.label = label.clone();
            }
            layers.push(tl);
        }
        layers.extend(pair_hulls.iter().cloned());
        emit_figure(&layers, frame, args, &stub, &mut emitted, false)?;
    }

    Ok(emitted)
}

/// Bin per-edge coexpression scores and rasterize one arrow layer per
/// bin so each gets a distinct color from the red↔blue ramp (high
/// coexpression → red, low → blue). NaN coexpression → bin 0.
fn build_coexpr_arrow_layers(
    segs: &[(Pt, Pt)],
    coexpr: &[f32],
    ext: plot_utils::rasterize::Extent,
    stroke_px: f32,
    head_len_px: f32,
    bins: usize,
) -> anyhow::Result<Vec<TopicLayer>> {
    let bins = bins.max(2);

    // Diverging scale centered at zero — `coexpr` has already been
    // mean-subtracted by the caller, so 0 = "typical edge of this pair".
    // Symmetric scale via robust |.| max so red and blue carry equal weight.
    let mut abs: Vec<f32> = coexpr
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .collect();
    let max_abs = if abs.is_empty() {
        1.0f32
    } else {
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = abs.len();
        let hi_idx = (n - 1).saturating_sub(n / 50);
        abs[hi_idx].max(1e-12)
    };

    let mut by_bin: Vec<Vec<(Pt, Pt)>> = (0..bins).map(|_| Vec::new()).collect();
    let half = bins as f32 * 0.5;
    for (i, &seg) in segs.iter().enumerate() {
        let v = coexpr.get(i).copied().unwrap_or(f32::NAN);
        let b = if !v.is_finite() {
            (bins / 2).min(bins - 1)
        } else {
            let scaled = (v / max_abs).clamp(-1.0, 1.0);
            ((scaled * half + half).clamp(0.0, (bins - 1) as f32 + 0.999_999)) as usize
        };
        by_bin[b.min(bins - 1)].push(seg);
    }

    let mut layers: Vec<TopicLayer> = Vec::new();
    for (b, segs_bin) in by_bin.into_iter().enumerate() {
        if segs_bin.is_empty() {
            continue;
        }
        let color = red_blue_bin(b, bins);
        let png = rasterize_arrow_layer_png(&segs_bin, ext, stroke_px, head_len_px, color, 0.9)?;
        layers.push(TopicLayer {
            label: String::new(),
            png,
            hull_px: Vec::new(),
            label_xy_px: (f32::NAN, f32::NAN),
            color,
        });
    }
    Ok(layers)
}

fn red_blue_bin(bin: usize, bins: usize) -> (u8, u8, u8) {
    let t = (bin as f32) / ((bins.saturating_sub(1)).max(1) as f32);
    let stops: [(f32, (u8, u8, u8)); 3] = [
        (0.0, (40, 70, 180)),
        (0.5, (190, 190, 190)),
        (1.0, (200, 30, 40)),
    ];
    for w in stops.windows(2) {
        let (a_t, a_c) = w[0];
        let (b_t, b_c) = w[1];
        if t >= a_t && t <= b_t {
            let f = (t - a_t) / (b_t - a_t).max(1e-9);
            return (
                lerp_u8(a_c.0, b_c.0, f),
                lerp_u8(a_c.1, b_c.1, f),
                lerp_u8(a_c.2, b_c.2, f),
            );
        }
    }
    stops[stops.len() - 1].1
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}
