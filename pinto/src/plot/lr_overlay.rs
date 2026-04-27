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
            let (l, rr) = r.keys();
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

impl LrResult {
    /// Backend-row-name keys for `(ligand, receptor)`, falling back to
    /// the raw symbols when `*_resolved` is absent (older sidecars).
    pub fn keys(&self) -> (&str, &str) {
        let l = self
            .ligand_resolved
            .as_deref()
            .unwrap_or(self.ligand.as_str());
        let r = self
            .receptor_resolved
            .as_deref()
            .unwrap_or(self.receptor.as_str());
        (l, r)
    }
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
    dominant: Option<&[i64]>,
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

    let core_cell_set: HashSet<usize> = core.cell_ixs.iter().copied().collect();

    // Pooled-across-LR-pairs orientation per edge — see `pooled_orientations`.
    // Only computed when --data is available; otherwise fall back to
    // canonical (left → right) inside the per-edge loop.
    let pooled_orient: HashMap<usize, HashMap<(usize, usize), bool>> = match lr_expr {
        Some(le_map) => pooled_orientations(lr, le_map, cells, &core_cell_set),
        None => HashMap::default(),
    };

    // Render every significant result; the per-pair filename carries
    // `B{batch}` to disambiguate cross-batch results in the same dir.
    // Homotypic pairs (`L == R`, e.g. CADM3-CADM3, PCDHB3-PCDHB3) usually
    // dominate the top-of-list because adhesion molecules co-aggregate
    // — drop them by default so heterotypic signaling stays visible.
    let mut sig: Vec<&LrResult> = lr
        .results
        .iter()
        .filter(|r| {
            r.significant
                && r.stratum_id.is_some()
                && (args.lr_keep_homotypic || r.ligand != r.receptor)
        })
        .collect();
    if sig.is_empty() {
        return Ok(emitted);
    }
    // Top-N by |z| *per stratum* — a stratum is (batch, community),
    // matching the test's natural unit. In single-batch runs this
    // collapses to per-community; in multi-batch runs each (batch,
    // community) gets its own budget so high-effect batches don't crowd
    // out their counterparts in other batches.
    sig.sort_by(|a, b| {
        b.z.unwrap_or(0.0)
            .abs()
            .partial_cmp(&a.z.unwrap_or(0.0).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cap = args.lr_top_pairs.max(1);
    let mut per_stratum: HashMap<usize, usize> = HashMap::default();
    sig.retain(|r| {
        let sid = match r.stratum_id {
            Some(s) => s,
            None => return false,
        };
        let n = per_stratum.entry(sid).or_insert(0);
        if *n < cap {
            *n += 1;
            true
        } else {
            false
        }
    });

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

    let mut empty_strata = 0usize;
    for r in sig.into_iter() {
        let sid = match r.stratum_id {
            Some(s) => s,
            None => continue,
        };
        let stratum = match by_id.get(&sid) {
            Some(s) => *s,
            None => continue,
        };
        if stratum.edges.is_empty() {
            empty_strata += 1;
            continue;
        }

        // Per-edge ligand-expression and receptor-expression vectors when
        // we have --data; otherwise fall back to canonical (left → right).
        let (l_key, r_key) = r.keys();
        let l_expr = lr_expr.and_then(|m| m.get(l_key));
        let r_expr = lr_expr.and_then(|m| m.get(r_key));

        // Build full-edge arrows + per-edge coexpression scores.
        let mut segs: Vec<(Pt, Pt)> = Vec::new();
        let mut seg_cells: Vec<(usize, usize)> = Vec::new();
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
                    if l_li + r_ri == 0.0 && l_ri + r_li == 0.0 {
                        continue;
                    }
                    // Use the pooled per-stratum direction when we have
                    // it; per-pair L+R sum is the fallback when this
                    // stratum had no orientation entry for this edge
                    // (e.g. edge dropped from the orientation pool).
                    let use_canon = match pooled_orient.get(&sid).and_then(|m| m.get(&(li, ri))) {
                        Some(&v) => v,
                        None => (l_li + r_ri) >= (l_ri + r_li),
                    };
                    if use_canon {
                        (li, ri, (l_li.max(0.0) * r_ri.max(0.0)).sqrt())
                    } else {
                        (ri, li, (l_ri.max(0.0) * r_li.max(0.0)).sqrt())
                    }
                }
                _ => (li, ri, f32::NAN),
            };

            let p_src = frame.bounds.to_pixel(cells.coords[src], frame.extent);
            let p_dst = frame.bounds.to_pixel(cells.coords[dst], frame.extent);
            segs.push((p_src, p_dst));
            seg_cells.push((src, dst));
            coexpr.push(ce_raw);
        }
        if segs.len() < args.lr_min_edges {
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

        // Thin shaft + sharper, slightly longer head reads as a clean
        // directional glyph at slide scale. Per-edge color depends on
        // --lr-color-mode.
        let stroke = 1.0f32;
        let head_len = 13.0f32;
        let arrow_layers = match args.lr_color_mode {
            crate::plot::args::LrColorMode::LogRatio => {
                // log((R + 1) / (L + 1)) — receptor over ligand.
                // Positive (red): R ≫ L → ligand-limited regime; signal
                //   scales ~linearly with ligand, the "activating" /
                //   sensitive direction.
                // Negative (blue): L ≫ R → receptor-saturation plateau;
                //   adding ligand does little, signal capped by R.
                let lr_log: Vec<f32> = seg_cells
                    .iter()
                    .map(|&(src, dst)| {
                        let l_at_src = l_expr
                            .and_then(|v| v.get(src).copied())
                            .unwrap_or(0.0)
                            .max(0.0);
                        let r_at_dst = r_expr
                            .and_then(|v| v.get(dst).copied())
                            .unwrap_or(0.0)
                            .max(0.0);
                        ((r_at_dst + 1.0).ln()) - ((l_at_src + 1.0).ln())
                    })
                    .collect();
                build_diverging_arrow_layers(
                    &segs,
                    &lr_log,
                    frame.extent,
                    stroke,
                    head_len,
                    args.lr_coexpr_bins,
                )?
            }
            crate::plot::args::LrColorMode::Coexpr => build_diverging_arrow_layers(
                &segs,
                &coexpr,
                frame.extent,
                stroke,
                head_len,
                args.lr_coexpr_bins,
            )?,
            crate::plot::args::LrColorMode::Direction => build_direction_arrow_layers(
                &segs,
                &seg_cells,
                dominant,
                r.community as i64,
                frame.extent,
                stroke,
                head_len,
            )?,
        };

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
        let mut layers: Vec<TopicLayer> = Vec::with_capacity(1 + arrow_layers.len());
        layers.push(bg_layer.clone());
        for (i, l) in arrow_layers.into_iter().enumerate() {
            let mut tl = l;
            if i == 0 {
                tl.label = label.clone();
            }
            layers.push(tl);
        }
        emit_figure(&layers, frame, args, &stub, &mut emitted, false)?;
    }

    if emitted.is_empty() && empty_strata > 0 {
        log::warn!(
            "lr-overlay: 0 figures rendered. {empty_strata} of the top \
             significant pairs reference strata with no edges in the JSON \
             — re-run `pinto lr-activity` to regenerate the sidecar."
        );
    }

    Ok(emitted)
}

/// Bin per-edge values onto a diverging red↔blue ramp (positive = red,
/// negative = blue). Values are pre-centered by the caller (per-pair
/// mean for `LrColorMode::Coexpr`, or 0 by definition for
/// `LrColorMode::LogRatio`). Symmetric scaling via a 98th-percentile
/// clip on `|values|`; non-finite values land in the mid-bin.
fn build_diverging_arrow_layers(
    segs: &[(Pt, Pt)],
    values: &[f32],
    ext: plot_utils::rasterize::Extent,
    stroke_px: f32,
    head_len_px: f32,
    bins: usize,
) -> anyhow::Result<Vec<TopicLayer>> {
    let bins = bins.max(2);

    let mut abs: Vec<f32> = values
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
        let v = values.get(i).copied().unwrap_or(f32::NAN);
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

/// Color arrows by their relation to the pair's community hull.
/// Each edge falls into exactly one of four classes:
///
/// * **Outgoing** — sender is in the community, receiver is not
///   (signal leaving the community).
/// * **Incoming** — receiver is in the community, sender is not
///   (signal entering the community).
/// * **Internal** — both endpoints sit inside the community
///   (signal staying within the hull).
/// * **External** — neither endpoint is in the community (rare for
///   significant edges, since the stratum's edges are by definition
///   community-incident, but possible when both endpoints are
///   "uncommitted" boundary cells with a different argmax).
///
/// One [`TopicLayer`] per non-empty class; the legend label on each
/// layer is "out" / "in" / "internal" / "ext" so the figure's title
/// row carries a small directional key.
fn build_direction_arrow_layers(
    segs: &[(Pt, Pt)],
    seg_cells: &[(usize, usize)],
    dominant: Option<&[i64]>,
    target_community: i64,
    ext: plot_utils::rasterize::Extent,
    stroke_px: f32,
    head_len_px: f32,
) -> anyhow::Result<Vec<TopicLayer>> {
    // (label, color) buckets; deterministic, slide-friendly palette.
    let buckets: [(&str, (u8, u8, u8)); 4] = [
        ("internal", (90, 90, 90)), // gray — within the hull
        ("out", (215, 70, 50)),     // red-ish — leaving the hull
        ("in", (40, 100, 200)),     // blue-ish — entering the hull
        ("ext", (180, 180, 180)),   // light gray — neither endpoint in hull
    ];
    let mut by_class: [Vec<(Pt, Pt)>; 4] = Default::default();

    for (i, &seg) in segs.iter().enumerate() {
        let (src, dst) = seg_cells[i];
        let src_in = dominant
            .and_then(|d| d.get(src).copied())
            .map(|c| c == target_community)
            .unwrap_or(false);
        let dst_in = dominant
            .and_then(|d| d.get(dst).copied())
            .map(|c| c == target_community)
            .unwrap_or(false);
        let class = match (src_in, dst_in) {
            (true, true) => 0,   // internal
            (true, false) => 1,  // out
            (false, true) => 2,  // in
            (false, false) => 3, // external
        };
        by_class[class].push(seg);
    }

    let mut layers: Vec<TopicLayer> = Vec::new();
    for (class, segs_class) in by_class.into_iter().enumerate() {
        if segs_class.is_empty() {
            continue;
        }
        let (label, color) = buckets[class];
        let png = rasterize_arrow_layer_png(&segs_class, ext, stroke_px, head_len_px, color, 0.9)?;
        layers.push(TopicLayer {
            label: label.to_string(),
            png,
            hull_px: Vec::new(),
            label_xy_px: (f32::NAN, f32::NAN),
            color,
        });
    }
    Ok(layers)
}

/// Per-stratum, per-edge orientation pooled across every significant
/// LR pair in the stratum. For edge (i, j), accumulates
///   Σ_p (log1p(L_p[i]·R_p[j]) - log1p(L_p[j]·R_p[i]))
/// over all significant pairs `p` in the stratum and returns `true`
/// (= canonical, i → j) when the sum is non-negative. The log1p
/// damps single-pair outliers — the part of MI that matters for
/// direction is preserved, but no individual pair can dictate the
/// answer. Edges with no expression support across any pair are
/// omitted from the map (caller falls back to per-pair L+R sum).
fn pooled_orientations(
    lr: &LrJson,
    le_map: &LrExpression,
    cells: &CellTable,
    core_cell_set: &HashSet<usize>,
) -> HashMap<usize, HashMap<(usize, usize), bool>> {
    let mut by_stratum: HashMap<usize, Vec<&LrResult>> = HashMap::default();
    for r in &lr.results {
        if !r.significant {
            continue;
        }
        if let Some(sid) = r.stratum_id {
            by_stratum.entry(sid).or_default().push(r);
        }
    }
    let strata_by_id: HashMap<usize, &LrStratum> =
        lr.strata.iter().map(|s| (s.stratum_id, s)).collect();

    let mut out: HashMap<usize, HashMap<(usize, usize), bool>> = HashMap::default();
    for (sid, results) in by_stratum {
        let stratum = match strata_by_id.get(&sid) {
            Some(s) => *s,
            None => continue,
        };
        // Resolve each pair's (l_vec, r_vec) once per stratum; the inner
        // edge loop then indexes the vectors instead of re-hashing
        // Box<str> keys per (edge × pair).
        let pair_vecs: Vec<(&Vec<f32>, &Vec<f32>)> = results
            .iter()
            .filter_map(|r| {
                let (l_key, r_key) = r.keys();
                Some((le_map.get(l_key)?, le_map.get(r_key)?))
            })
            .collect();
        if pair_vecs.is_empty() {
            continue;
        }
        let mut dirs: HashMap<(usize, usize), bool> = HashMap::default();
        for (l_name, r_name) in &stratum.edges {
            let li = match cells.index.get(l_name) {
                Some(&i) if core_cell_set.contains(&i) => i,
                _ => continue,
            };
            let ri = match cells.index.get(r_name) {
                Some(&i) if core_cell_set.contains(&i) => i,
                _ => continue,
            };
            let mut score = 0.0f64;
            let mut any_signal = false;
            for (l_vec, r_vec) in &pair_vecs {
                let l_at_li = l_vec.get(li).copied().unwrap_or(0.0).max(0.0);
                let r_at_ri = r_vec.get(ri).copied().unwrap_or(0.0).max(0.0);
                let l_at_ri = l_vec.get(ri).copied().unwrap_or(0.0).max(0.0);
                let r_at_li = r_vec.get(li).copied().unwrap_or(0.0).max(0.0);
                let canon = (l_at_li * r_at_ri) as f64;
                let rev = (l_at_ri * r_at_li) as f64;
                if canon > 0.0 || rev > 0.0 {
                    any_signal = true;
                    score += (1.0 + canon).ln() - (1.0 + rev).ln();
                }
            }
            if any_signal {
                dirs.insert((li, ri), score >= 0.0);
            }
        }
        if !dirs.is_empty() {
            out.insert(sid, dirs);
        }
    }
    out
}

/// Bin per-edge coexpression scores and rasterize one arrow layer per
/// bin so each gets a distinct color from the red↔blue ramp (high
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
