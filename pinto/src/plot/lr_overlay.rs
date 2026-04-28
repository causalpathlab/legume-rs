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
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use plot_utils::hull::Pt;
use plot_utils::rasterize::{rasterize_arrow_layer_png, rasterize_group_png, RadiusSpec};
use plot_utils::svg_emit::TopicLayer;
use serde::Deserialize;
use std::fs::File;
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
    /// FWER cutoff used by the writer to mark `significant`. Falls back
    /// to 0.05 for older sidecars that didn't carry this field.
    #[serde(default = "default_fwer_threshold")]
    pub fwer_threshold: f32,
}

fn default_fwer_threshold() -> f32 {
    0.05
}

/// Subset of `pinto lr-activity` parquet columns needed by the
/// summary plots: every tested pair, with effect size + significance
/// inputs. Lets the plot include non-significant pairs in the Hinton
/// matrices for direct comparison.
pub struct LrParquetRow {
    pub community: i32,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    /// One-sided Gaussian z (parametric), drives box area as `|z|`.
    pub z: f32,
    /// Westfall-Young FWER. Significance = `fwer_wy < threshold && z_re > 0`.
    pub fwer_wy: f32,
    pub z_re: f32,
}

impl LrParquetRow {
    pub fn is_significant(&self, threshold: f32) -> bool {
        self.fwer_wy.is_finite()
            && self.fwer_wy < threshold
            && self.z_re.is_finite()
            && self.z_re > 0.0
    }
}

/// Read every row from `{prefix}.lr_activity.parquet`. Used by the
/// summary plots so non-significant pairs can be drawn alongside the
/// significant ones for visual comparison.
pub fn read_lr_activity_parquet(path: &Path) -> anyhow::Result<Vec<LrParquetRow>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let mut field_idx: HashMap<Box<str>, usize> = HashMap::default();
    for (i, f) in schema.get_fields().iter().enumerate() {
        field_idx.insert(f.name().to_string().into_boxed_str(), i);
    }
    let need = |name: &str| -> anyhow::Result<usize> {
        field_idx
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("{path:?}: missing column `{name}`"))
    };
    let i_c = need("community")?;
    let i_l = need("ligand")?;
    let i_r = need("receptor")?;
    let i_z = need("z")?;
    let i_fwer = need("fwer_wy")?;
    let i_zre = need("z_re")?;

    let mut out: Vec<LrParquetRow> = Vec::new();
    for record in reader.get_row_iter(None)? {
        let row = record?;
        out.push(LrParquetRow {
            community: row.get_int(i_c)?,
            ligand: row.get_string(i_l)?.clone().into_boxed_str(),
            receptor: row.get_string(i_r)?.clone().into_boxed_str(),
            z: row.get_float(i_z).unwrap_or(0.0),
            fwer_wy: row.get_float(i_fwer).unwrap_or(1.0),
            z_re: row.get_float(i_zre).unwrap_or(0.0),
        });
    }
    Ok(out)
}

/// Resolve the `lr_activity.parquet` sibling of an `lr_activity.json`.
/// Returns `None` when the sibling is absent.
pub fn resolve_lr_parquet_path(json_path: &Path) -> Option<PathBuf> {
    let p = json_path.with_extension("parquet");
    p.exists().then_some(p)
}

/// Emit the standard svg / pdf / png triplet for a finished SVG. The
/// global `--svg`, `--no-pdf`, `--png` flags decide which outputs land
/// on disk; `stub` is the path *without* extension. Each emitted file
/// is appended to `emitted` for the manifest writer.
fn emit_svg_outputs(
    args: &SrtPlotArgs,
    svg: &str,
    stub: &Path,
    size_px: (u32, u32),
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    let with_ext = |ext: &str| -> PathBuf { PathBuf::from(format!("{}.{ext}", stub.display())) };
    if args.svg {
        let p = with_ext("svg");
        std::fs::write(&p, svg.as_bytes())?;
        emitted.push(p);
    }
    if !args.no_pdf {
        let p = with_ext("pdf");
        plot_utils::render_pdf(svg, &p)?;
        emitted.push(p);
    }
    if args.png {
        let p = with_ext("png");
        plot_utils::render_png(svg, size_px.0, size_px.1, &p)?;
        emitted.push(p);
    }
    Ok(())
}

/// Palette size for the LR plots: the final clustering's total K when
/// known (so colours match marker Hinton / propensity argmax even when
/// K crosses the Paired→Category20 boundary), else just enough to hold
/// every community ID we've seen.
fn palette_size_for(k_total: Option<usize>, k_max_observed: i32) -> usize {
    let observed = (k_max_observed + 1).max(1) as usize;
    k_total.map_or(observed, |k| k.max(observed))
}

/// Mix `rgb` toward near-white by `t` (0 → unchanged, 1 → almost white).
/// Used to fade non-significant cells in the Hinton summaries.
fn fade_to_near_white(rgb: plot_utils::Rgb, t: f32) -> plot_utils::Rgb {
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| -> u8 {
        ((a as f32) * (1.0 - t) + (b as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    (lerp(rgb.0, 245), lerp(rgb.1, 245), lerp(rgb.2, 245))
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

/// Remap "0","1",… numeric batch labels on every `LrResult.batch` and
/// `LrStratum.batch` to the friendly names supplied by the caller (the
/// same map that `plot::load::resolve_batch_name_map` builds for
/// `cells.batches`). Non-numeric or out-of-range labels are left
/// untouched so user-supplied names pass through unchanged.
pub fn remap_lr_batches(lr: &mut LrJson, name_map: &[Box<str>]) {
    let lookup = |label: &str| -> Option<String> {
        label
            .parse::<usize>()
            .ok()
            .and_then(|i| name_map.get(i))
            .map(|n| n.to_string())
    };
    for r in &mut lr.results {
        if let Some(friendly) = lookup(&r.batch) {
            r.batch = friendly;
        }
    }
    for s in &mut lr.strata {
        if let Some(friendly) = lookup(&s.batch) {
            s.batch = friendly;
        }
    }
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
    kept_communities: Option<&HashSet<usize>>,
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

    // Render every significant result that belongs to this core's
    // batch. Per-batch cores own a subdir (`out_dir/{core.name}/`) so
    // the filename can drop the `B{batch}` prefix; single-batch cores
    // (no `batch_label`) keep the legacy flat layout.
    // Homotypic pairs (`L == R`, e.g. CADM3-CADM3, PCDHB3-PCDHB3) usually
    // dominate the top-of-list because adhesion molecules co-aggregate
    // — drop them by default so heterotypic signaling stays visible.
    let core_batch = core.batch_label.as_deref();
    let mut sig: Vec<&LrResult> = lr
        .results
        .iter()
        .filter(|r| {
            r.significant
                && r.stratum_id.is_some()
                && (args.lr_keep_homotypic || r.ligand != r.receptor)
                && match core_batch {
                    Some(b) => r.batch.as_str() == b,
                    None => true,
                }
                && match kept_communities {
                    Some(k) => r.community >= 0 && k.contains(&(r.community as usize)),
                    None => true,
                }
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

        // Per-batch run: filename is already disambiguated by the
        // per-core subdir (`out_dir/{core.name}/`), so the stub just
        // carries `C{c}.{L}-{R}`. Single-batch (no batch_label) keeps
        // the legacy `B{batch}` prefix in case the lr-activity fit
        // stratified by batch even though the plot core didn't.
        let leaf = match core_batch {
            Some(_) => format!(
                "C{}.{}-{}",
                r.community,
                sanitize(&r.ligand),
                sanitize(&r.receptor),
            ),
            None => format!(
                "B{}.C{}.{}-{}",
                sanitize(&r.batch),
                r.community,
                sanitize(&r.ligand),
                sanitize(&r.receptor),
            ),
        };
        let stub = core.subdir_in(out_dir).join(leaf);
        let label = format!(
            "{}->{}; B={}; C{}; z={:.2}; q={:.3}",
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

/// Combined Hinton diagram across all communities: rows = ligands,
/// columns = receptors, box area = `max |z|` for that (L, R) pair
/// across communities, **box color = the community where that max is
/// attained**. The (L, R) axes are picked from the top-N sig pairs
/// (capped by `--lr-summary-pairs`), then the matrix is filled from
/// **all** parquet rows whose (L, R) lands in those axes — non-sig
/// cells are faded toward white so they read as "tested but didn't
/// pass FWER" instead of vanishing. Falls back to JSON-only (no
/// non-sig comparison) when the parquet sibling is absent.
pub fn emit_lr_summary_global(
    args: &SrtPlotArgs,
    lr: &LrJson,
    parquet_rows: Option<&[LrParquetRow]>,
    k_total: Option<usize>,
    out_dir: &Path,
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    let sig: Vec<&LrResult> = lr
        .results
        .iter()
        .filter(|r| r.significant && (args.lr_keep_homotypic || r.ligand != r.receptor))
        .collect();
    if sig.is_empty() {
        return Ok(());
    }

    // (max |z|, community at which that max is attained) per (L, R).
    type PairBest = (f32, i32);

    // Per (ligand, receptor) pair: max |z| across communities (and the
    // community where that max sits, for color). Pairs ranked by that
    // max are then capped at `--lr-summary-pairs`.
    let mut pair_best: HashMap<(&str, &str), PairBest> = HashMap::default();
    let mut k_max: i32 = -1;
    for r in &sig {
        let z = r.z.unwrap_or(0.0).abs();
        pair_best
            .entry((r.ligand.as_str(), r.receptor.as_str()))
            .and_modify(|v| {
                if z > v.0 {
                    *v = (z, r.community);
                }
            })
            .or_insert((z, r.community));
        if r.community > k_max {
            k_max = r.community;
        }
    }
    let mut pairs_ranked: Vec<((&str, &str), PairBest)> = pair_best.into_iter().collect();
    pairs_ranked.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cap = args.lr_summary_pairs.max(1);
    if pairs_ranked.len() > cap {
        pairs_ranked.truncate(cap);
    }
    if pairs_ranked.is_empty() || k_max < 0 {
        return Ok(());
    }

    // Restrict ligand / receptor axes to those involved in the kept pairs,
    // sorted by per-axis max |z| descending.
    let mut l_max: HashMap<&str, f32> = HashMap::default();
    let mut r_max: HashMap<&str, f32> = HashMap::default();
    for ((l, r), (z, _)) in &pairs_ranked {
        l_max
            .entry(*l)
            .and_modify(|v| {
                if *z > *v {
                    *v = *z;
                }
            })
            .or_insert(*z);
        r_max
            .entry(*r)
            .and_modify(|v| {
                if *z > *v {
                    *v = *z;
                }
            })
            .or_insert(*z);
    }
    let mut ligands: Vec<&str> = l_max.keys().copied().collect();
    ligands.sort_by(|a, b| {
        l_max[b]
            .partial_cmp(&l_max[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut receptors: Vec<&str> = r_max.keys().copied().collect();
    receptors.sort_by(|a, b| {
        r_max[b]
            .partial_cmp(&r_max[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_l = ligands.len();
    let n_r = receptors.len();
    if n_l == 0 || n_r == 0 {
        return Ok(());
    }
    let l_idx: HashMap<&str, usize> = ligands.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let r_idx: HashMap<&str, usize> = receptors.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    // Fill matrix: per cell, track max |z| (drives box area) and the
    // community where that max sits (drives color). When the parquet
    // sibling is available, also include non-significant rows so the
    // user can see how non-sig pairs compare to sig ones in the same
    // axes; non-sig cells are faded toward white below.
    let mut mat = vec![0.0f32; n_l * n_r];
    let mut major_c = vec![-1i32; n_l * n_r];
    let mut sig_cell = vec![false; n_l * n_r];
    let mut communities_present: HashSet<i32> = HashSet::default();

    if let Some(rows) = parquet_rows {
        for r in rows {
            let li = match l_idx.get(r.ligand.as_ref()) {
                Some(&i) => i,
                None => continue,
            };
            let ri = match r_idx.get(r.receptor.as_ref()) {
                Some(&i) => i,
                None => continue,
            };
            let z = r.z.abs();
            if !z.is_finite() {
                continue;
            }
            let pos = li * n_r + ri;
            if z > mat[pos] {
                mat[pos] = z;
                major_c[pos] = r.community;
                sig_cell[pos] = r.is_significant(lr.fwer_threshold);
                communities_present.insert(r.community);
            }
        }
    } else {
        // Parquet missing — JSON sig-only (no non-sig comparison).
        for ((l, r), (z, c)) in &pairs_ranked {
            let li = l_idx[l];
            let ri = r_idx[r];
            let pos = li * n_r + ri;
            mat[pos] = *z;
            major_c[pos] = *c;
            sig_cell[pos] = true;
            communities_present.insert(*c);
        }
    }

    let k = palette_size_for(k_total, k_max);
    let colors = super::render::ColorBook::new(args, k);
    let cell_colors: Vec<plot_utils::Rgb> = major_c
        .iter()
        .zip(sig_cell.iter())
        .map(|(&c, &is_sig)| {
            let base = if c < 0 || (c as usize) >= k {
                (200, 200, 200)
            } else {
                colors.color(c as usize)
            };
            if is_sig {
                base
            } else {
                fade_to_near_white(base, 0.65)
            }
        })
        .collect();

    // Color legend: one swatch per community that actually shows up in
    // the kept matrix, sorted by community id for stable ordering.
    let mut legend_communities: Vec<i32> = communities_present.into_iter().collect();
    legend_communities.sort();
    let color_legend: Vec<(Box<str>, plot_utils::Rgb)> = legend_communities
        .iter()
        .filter(|&&c| (c as usize) < k)
        .map(|&c| (format!("C{c}").into_boxed_str(), colors.color(c as usize)))
        .collect();

    let row_labels: Vec<Box<str>> = ligands
        .iter()
        .map(|s| (*s).to_string().into_boxed_str())
        .collect();
    let col_labels: Vec<Box<str>> = receptors
        .iter()
        .map(|s| (*s).to_string().into_boxed_str())
        .collect();
    let (row_order, col_order) = plot_utils::diagonalize_order(&mat, n_l, n_r);

    let n_sig_cells = sig_cell.iter().filter(|&&b| b).count();
    let title = format!(
        "LR-activity summary — top {n_pairs} sig pairs, {n_l}L × {n_r}R, {n_sig_cells} sig cells (faded = not FWER-significant)",
        n_pairs = pairs_ranked.len(),
    );
    let opts = plot_utils::HintonOpts {
        row_labels: Some(&row_labels),
        col_labels: Some(&col_labels),
        row_order: Some(&row_order),
        col_order: Some(&col_order),
        col_colors: None,
        cell_colors: Some(&cell_colors),
        scale: plot_utils::HintonScale::Sqrt,
        cell_px: 18.0,
        font_px: 11.0,
        title: Some(&title),
        grid_stroke_px: 0.4,
        grid_color: (220, 220, 220),
        color_legend: Some(&color_legend),
    };
    let svg = plot_utils::render_hinton(&mat, n_l, n_r, &opts);
    let size = plot_utils::hinton_size(n_l, n_r, &opts);
    emit_svg_outputs(
        args,
        &svg,
        &out_dir.join("summary"),
        (size.width_px, size.height_px),
        emitted,
    )?;
    Ok(())
}

/// Bipartite-graph view of LR pairs: ligands stacked on the left,
/// receptors stacked on the right. Edges scaled by
/// **`-log10(fwer_wy)`** (FWER significance) and colored by the
/// **major community** of each pair. Axes (which ligands / receptors
/// to include) come from the top-N **significant** pairs (cap =
/// `--lr-summary-pairs`); when the parquet sibling is available, the
/// matrix is then filled with **all** pairs (sig + non-sig) whose
/// endpoints land in those axes — non-sig edges are faded to ~30%
/// opacity so they read as "tested but didn't pass FWER" without
/// drowning the headline edges. Both axes sorted by (major community
/// ascending, max |z| descending) to minimize edge crossings.
/// `--lr-keep-homotypic` filter applies. Output: `out_dir/bipartite.pdf`
/// (+ svg/png per the global toggles).
pub fn emit_lr_bipartite(
    args: &SrtPlotArgs,
    lr: &LrJson,
    parquet_rows: Option<&[LrParquetRow]>,
    k_total: Option<usize>,
    out_dir: &Path,
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    let sig: Vec<&LrResult> = lr
        .results
        .iter()
        .filter(|r| r.significant && (args.lr_keep_homotypic || r.ligand != r.receptor))
        .collect();
    if sig.is_empty() {
        return Ok(());
    }

    // Per (L, R): pick the row with the largest |z|; record its
    // community (color) and fwer (edge width).
    type EdgeInfo = (i32, f32, f32);
    let mut best: HashMap<(&str, &str), EdgeInfo> = HashMap::default();
    let mut k_max: i32 = -1;
    for r in &sig {
        let z = r.z.unwrap_or(0.0).abs();
        let fwer = r.fwer_wy.unwrap_or(1.0);
        let entry = best
            .entry((r.ligand.as_str(), r.receptor.as_str()))
            .or_insert((r.community, z, fwer));
        if z > entry.1 {
            *entry = (r.community, z, fwer);
        }
        if r.community > k_max {
            k_max = r.community;
        }
    }
    let mut edges: Vec<((&str, &str), EdgeInfo)> = best.into_iter().collect();
    edges.sort_by(|a, b| {
        b.1 .1
            .partial_cmp(&a.1 .1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cap = args.lr_summary_pairs.max(1);
    if edges.len() > cap {
        edges.truncate(cap);
    }
    if edges.is_empty() || k_max < 0 {
        return Ok(());
    }

    // Major community per ligand / per receptor = community with max |z|
    // across the kept edges involving that gene.
    let mut l_major: HashMap<&str, (i32, f32)> = HashMap::default();
    let mut r_major: HashMap<&str, (i32, f32)> = HashMap::default();
    for ((l, r), (c, z, _)) in &edges {
        let v = l_major.entry(*l).or_insert((*c, *z));
        if *z > v.1 {
            *v = (*c, *z);
        }
        let v = r_major.entry(*r).or_insert((*c, *z));
        if *z > v.1 {
            *v = (*c, *z);
        }
    }

    let mut ligands: Vec<&str> = l_major.keys().copied().collect();
    ligands.sort_by(|a, b| {
        let (ca, za) = l_major[a];
        let (cb, zb) = l_major[b];
        ca.cmp(&cb)
            .then_with(|| zb.partial_cmp(&za).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut receptors: Vec<&str> = r_major.keys().copied().collect();
    receptors.sort_by(|a, b| {
        let (ca, za) = r_major[a];
        let (cb, zb) = r_major[b];
        ca.cmp(&cb)
            .then_with(|| zb.partial_cmp(&za).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Layout — labels are rendered on outside edges, edges in the
    // middle gap. Row height adapts to font; widths are font-derived.
    let font_px = 10.0_f32;
    let row_h = font_px * 1.4;
    let max_label_chars = ligands
        .iter()
        .chain(receptors.iter())
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(8) as f32;
    let label_w = (font_px * 0.62 * max_label_chars).max(60.0);
    let middle_w = 320.0_f32;
    let legend_w = 90.0_f32;
    let pad_top = font_px * 3.0;
    let pad_bot = font_px * 1.5;

    let n_l = ligands.len();
    let n_r = receptors.len();
    let n_max = n_l.max(n_r);
    let total_h = pad_top + n_max as f32 * row_h + pad_bot;
    let total_w = label_w + middle_w + label_w + legend_w + 16.0;

    let left_x = label_w + 6.0;
    let right_x = label_w + middle_w - 6.0;

    let l_y: HashMap<&str, f32> = ligands
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, pad_top + (i as f32 + 0.5) * row_h))
        .collect();
    let r_y: HashMap<&str, f32> = receptors
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, pad_top + (i as f32 + 0.5) * row_h))
        .collect();

    let k = palette_size_for(k_total, k_max);
    let palette = super::render::ColorBook::new(args, k);

    // Edge stroke scale: -log10(fwer_wy), capped to avoid blowups when
    // fwer rounds to 0 (limit-of-detection); empirical max in `edges`
    // anchors the high end of the stroke ramp.
    let logp_of = |fwer: f32| (-(fwer.max(1e-12).log10())).max(0.0);
    let max_logp = edges
        .iter()
        .map(|(_, (_, _, fwer))| logp_of(*fwer))
        .fold(0.0_f32, f32::max)
        .max(1.0);
    let stroke_min = 0.3_f32;
    let stroke_max = 3.0_f32;

    use std::fmt::Write as _;
    let mut svg = String::with_capacity(8192 + edges.len() * 96);
    let _ = writeln!(
        svg,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w:.0} {h:.0}\" \
         width=\"{w:.0}\" height=\"{h:.0}\">",
        w = total_w,
        h = total_h,
    );
    let _ = writeln!(
        svg,
        "  <text x=\"{x:.1}\" y=\"{y:.1}\" font-family=\"Helvetica, Arial, sans-serif\" \
         font-size=\"{fs:.1}\" font-weight=\"bold\">\
         LR-activity bipartite — {n_edges} sig pairs (axes), {n_l} ligands → {n_r} receptors, \
         edge ∝ -log10(FWER); faded = below cutoff</text>",
        x = 6.0,
        y = font_px * 1.4,
        fs = font_px * 1.15,
        n_edges = edges.len(),
    );

    // Build the edge set actually drawn:
    //  - When the parquet sibling is readable, we draw every (L, R) row
    //    whose endpoints land in the current axes — non-sig edges are
    //    rendered faintly (low opacity, thin stroke) so the FWER survivors
    //    pop above a context layer.
    //  - Otherwise we draw only the JSON sig edges already collected in
    //    `edges`, all marked significant.
    type DrawEdge<'a> = (&'a str, &'a str, i32, f32, bool);
    let mut draw_edges: Vec<DrawEdge<'_>> = Vec::new();
    if let Some(rows) = parquet_rows {
        // Pick the row with the largest |z| per (L, R) so we don't draw
        // duplicates across batches; flag significance per kept row.
        let mut best: HashMap<(&str, &str), (i32, f32, f32, bool)> = HashMap::default();
        for r in rows {
            if !args.lr_keep_homotypic && r.ligand.as_ref() == r.receptor.as_ref() {
                continue;
            }
            let l_str = r.ligand.as_ref();
            let r_str = r.receptor.as_ref();
            if !l_y.contains_key(l_str) || !r_y.contains_key(r_str) {
                continue;
            }
            let z = r.z.abs();
            if !z.is_finite() {
                continue;
            }
            let is_sig = r.is_significant(lr.fwer_threshold);
            let entry = best
                .entry((l_str, r_str))
                .or_insert((r.community, z, r.fwer_wy, is_sig));
            if z > entry.1 {
                *entry = (r.community, z, r.fwer_wy, is_sig);
            }
        }
        for ((l, r), (c, _z, fwer, is_sig)) in best {
            draw_edges.push((l, r, c, fwer, is_sig));
        }
    } else {
        for ((l, r), (c, _z, fwer)) in &edges {
            draw_edges.push((l, r, *c, *fwer, true));
        }
    }
    // Paint non-sig first so sig edges land on top.
    draw_edges.sort_by_key(|e| e.4);

    let _ = writeln!(svg, "  <g id=\"edges\" stroke-linecap=\"round\">");
    let mut communities_present: HashSet<i32> = HashSet::default();
    for (l, r, c, fwer, is_sig) in &draw_edges {
        let (Some(&yl), Some(&yr)) = (l_y.get(l), r_y.get(r)) else {
            continue;
        };
        let logp = logp_of(*fwer);
        // Sig: full -log10(p) ramp in [stroke_min, stroke_max], opacity 0.75.
        // Non-sig: half-strength stroke and ~30% opacity so they read as
        // "tested but not significant" rather than as headline signals.
        let (stroke, opacity) = if *is_sig {
            (
                stroke_min + (stroke_max - stroke_min) * (logp / max_logp),
                0.75_f32,
            )
        } else {
            (
                stroke_min + (stroke_max - stroke_min) * 0.5 * (logp / max_logp),
                0.30_f32,
            )
        };
        let color = if (*c as usize) < k {
            palette.color(*c as usize)
        } else {
            (90, 90, 90)
        };
        if *is_sig {
            communities_present.insert(*c);
        }
        let _ = writeln!(
            svg,
            "    <line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
             stroke=\"rgb({r},{g},{b})\" stroke-width=\"{sw:.2}\" opacity=\"{op:.2}\"/>",
            x1 = left_x,
            y1 = yl,
            x2 = right_x,
            y2 = yr,
            r = color.0,
            g = color.1,
            b = color.2,
            sw = stroke,
            op = opacity,
        );
    }
    let _ = writeln!(svg, "  </g>");

    // Endpoint dots colored by each gene's major community.
    let _ = writeln!(svg, "  <g id=\"dots\">");
    for &n in &ligands {
        let (c, _) = l_major[n];
        let color = if (c as usize) < k {
            palette.color(c as usize)
        } else {
            (90, 90, 90)
        };
        let _ = writeln!(
            svg,
            "    <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"2.2\" fill=\"rgb({r},{g},{b})\"/>",
            x = left_x,
            y = l_y[n],
            r = color.0,
            g = color.1,
            b = color.2,
        );
    }
    for &n in &receptors {
        let (c, _) = r_major[n];
        let color = if (c as usize) < k {
            palette.color(c as usize)
        } else {
            (90, 90, 90)
        };
        let _ = writeln!(
            svg,
            "    <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"2.2\" fill=\"rgb({r},{g},{b})\"/>",
            x = right_x,
            y = r_y[n],
            r = color.0,
            g = color.1,
            b = color.2,
        );
    }
    let _ = writeln!(svg, "  </g>");

    // Labels.
    let _ = writeln!(
        svg,
        "  <g id=\"l-labels\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{fs:.1}\" \
         text-anchor=\"end\">",
        fs = font_px,
    );
    for &n in &ligands {
        let _ = writeln!(
            svg,
            "    <text x=\"{x:.1}\" y=\"{y:.1}\">{t}</text>",
            x = left_x - 5.0,
            y = l_y[n] + font_px * 0.35,
            t = escape_svg(n),
        );
    }
    let _ = writeln!(svg, "  </g>");

    let _ = writeln!(
        svg,
        "  <g id=\"r-labels\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{fs:.1}\" \
         text-anchor=\"start\">",
        fs = font_px,
    );
    for &n in &receptors {
        let _ = writeln!(
            svg,
            "    <text x=\"{x:.1}\" y=\"{y:.1}\">{t}</text>",
            x = right_x + 5.0,
            y = r_y[n] + font_px * 0.35,
            t = escape_svg(n),
        );
    }
    let _ = writeln!(svg, "  </g>");

    // Column headers + community legend.
    let _ = writeln!(
        svg,
        "  <g font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{fs:.1}\" \
         font-weight=\"bold\">",
        fs = font_px,
    );
    let _ = writeln!(
        svg,
        "    <text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"middle\">ligand</text>",
        x = left_x - label_w * 0.5,
        y = pad_top - font_px * 0.6,
    );
    let _ = writeln!(
        svg,
        "    <text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"middle\">receptor</text>",
        x = right_x + label_w * 0.5,
        y = pad_top - font_px * 0.6,
    );
    let _ = writeln!(svg, "  </g>");

    let mut legend_communities: Vec<i32> = communities_present.into_iter().collect();
    legend_communities.sort();
    if !legend_communities.is_empty() {
        let lx = total_w - legend_w + 6.0;
        let mut ly = pad_top;
        let _ = writeln!(
            svg,
            "  <g id=\"legend\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"{fs:.1}\">",
            fs = font_px,
        );
        let _ = writeln!(
            svg,
            "    <text x=\"{x:.1}\" y=\"{y:.1}\" font-weight=\"bold\">community</text>",
            x = lx,
            y = ly,
        );
        ly += font_px * 1.6;
        for c in legend_communities {
            if (c as usize) >= k {
                continue;
            }
            let (r, g, b) = palette.color(c as usize);
            let _ = writeln!(
                svg,
                "    <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"10\" height=\"10\" \
                 fill=\"rgb({r},{g},{b})\"/>",
                x = lx,
                y = ly,
            );
            let _ = writeln!(
                svg,
                "    <text x=\"{x:.1}\" y=\"{y:.1}\">C{c}</text>",
                x = lx + 14.0,
                y = ly + 8.5,
            );
            ly += 14.0;
        }
        let _ = writeln!(svg, "  </g>");
    }

    let _ = writeln!(svg, "</svg>");

    emit_svg_outputs(
        args,
        &svg,
        &out_dir.join("bipartite"),
        (total_w.ceil() as u32, total_h.ceil() as u32),
        emitted,
    )?;
    Ok(())
}

fn escape_svg(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// One Hinton diagram per community: rows = ligands, columns =
/// receptors, box area = max `|z|` for that (ligand, receptor) pair
/// within the community across batches. Axes come from significant
/// pairs (matched to the per-pair PDFs); when the parquet sibling is
/// available, **non-significant cells are also drawn** in the same
/// axes (faded color) so you can see which untested-marker pairs sit
/// just below the FWER cutoff. `--lr-keep-homotypic` filter applies.
/// One file per active community at `out_dir/C{c}.summary.pdf`.
pub fn emit_lr_summary_per_community(
    args: &SrtPlotArgs,
    lr: &LrJson,
    parquet_rows: Option<&[LrParquetRow]>,
    k_total: Option<usize>,
    out_dir: &Path,
    emitted: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    let mut by_c: HashMap<i32, Vec<&LrResult>> = HashMap::default();
    for r in &lr.results {
        if !r.significant {
            continue;
        }
        if !args.lr_keep_homotypic && r.ligand == r.receptor {
            continue;
        }
        by_c.entry(r.community).or_default().push(r);
    }
    if by_c.is_empty() {
        return Ok(());
    }

    let mut communities: Vec<i32> = by_c.keys().copied().collect();
    communities.sort();

    // Pre-bucket parquet rows by community so the per-community loop
    // doesn't rescan the full ~25k-row table 9× (one full filter
    // per community).
    let parquet_by_c: Option<HashMap<i32, Vec<&LrParquetRow>>> = parquet_rows.map(|rows| {
        let mut by_c: HashMap<i32, Vec<&LrParquetRow>> = HashMap::default();
        for r in rows {
            by_c.entry(r.community).or_default().push(r);
        }
        by_c
    });

    for c in communities {
        let results = &by_c[&c];

        // Axes: ligands / receptors from sig pairs only — keep the
        // headline cells in scope and don't let non-sig pairs balloon
        // the matrix.
        let mut l_max: HashMap<&str, f32> = HashMap::default();
        let mut r_max: HashMap<&str, f32> = HashMap::default();
        for r in results {
            let z = r.z.unwrap_or(0.0).abs();
            l_max
                .entry(r.ligand.as_str())
                .and_modify(|v| {
                    if z > *v {
                        *v = z;
                    }
                })
                .or_insert(z);
            r_max
                .entry(r.receptor.as_str())
                .and_modify(|v| {
                    if z > *v {
                        *v = z;
                    }
                })
                .or_insert(z);
        }
        let mut ligands: Vec<&str> = l_max.keys().copied().collect();
        ligands.sort_by(|a, b| {
            l_max[b]
                .partial_cmp(&l_max[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut receptors: Vec<&str> = r_max.keys().copied().collect();
        receptors.sort_by(|a, b| {
            r_max[b]
                .partial_cmp(&r_max[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_l = ligands.len();
        let n_r = receptors.len();
        if n_l == 0 || n_r == 0 {
            continue;
        }
        let l_idx: HashMap<&str, usize> =
            ligands.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let r_idx: HashMap<&str, usize> =
            receptors.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        // Fill matrix: parquet (all rows) when available — sig flag per
        // cell controls the box color. Otherwise sig-only from JSON.
        let mut mat = vec![0.0f32; n_l * n_r];
        let mut sig_cell = vec![false; n_l * n_r];
        if let Some(rows_for_c) = parquet_by_c.as_ref().and_then(|m| m.get(&c)) {
            for r in rows_for_c {
                let li = match l_idx.get(r.ligand.as_ref()) {
                    Some(&i) => i,
                    None => continue,
                };
                let ri = match r_idx.get(r.receptor.as_ref()) {
                    Some(&i) => i,
                    None => continue,
                };
                let z = r.z.abs();
                if !z.is_finite() {
                    continue;
                }
                let pos = li * n_r + ri;
                if z > mat[pos] {
                    mat[pos] = z;
                    sig_cell[pos] = r.is_significant(lr.fwer_threshold);
                }
            }
        } else {
            for r in results {
                let li = match l_idx.get(r.ligand.as_str()) {
                    Some(&i) => i,
                    None => continue,
                };
                let ri = match r_idx.get(r.receptor.as_str()) {
                    Some(&i) => i,
                    None => continue,
                };
                let z = r.z.unwrap_or(0.0).abs();
                let pos = li * n_r + ri;
                if z > mat[pos] {
                    mat[pos] = z;
                    sig_cell[pos] = true;
                }
            }
        }

        // FWER-survivors get the community's palette colour; the rest
        // fade toward white so they read as "tested but subordinate".
        let k_max = lr.results.iter().map(|r| r.community).max().unwrap_or(0);
        let k = palette_size_for(k_total, k_max);
        let colors_book = super::render::ColorBook::new(args, k);
        let community_color = if (c as usize) < k {
            colors_book.color(c as usize)
        } else {
            (90, 90, 90)
        };
        let cell_colors: Vec<plot_utils::Rgb> = sig_cell
            .iter()
            .map(|&is_sig| {
                if is_sig {
                    community_color
                } else {
                    fade_to_near_white(community_color, 0.65)
                }
            })
            .collect();

        let row_labels: Vec<Box<str>> = ligands
            .iter()
            .map(|s| (*s).to_string().into_boxed_str())
            .collect();
        let col_labels: Vec<Box<str>> = receptors
            .iter()
            .map(|s| (*s).to_string().into_boxed_str())
            .collect();
        let (row_order, col_order) = plot_utils::diagonalize_order(&mat, n_l, n_r);

        let n_sig_in_cells = sig_cell.iter().filter(|&&b| b).count();
        let title = format!(
            "Community C{c} — L × R (|z|) — {n_l}L × {n_r}R, {n_sig_in_cells} FWER-significant (faded = below cutoff)",
        );
        // Single-entry legend so the focal community's colour is
        // explicit, matching the C{c} convention used by marker Hinton
        // and propensity argmax.
        let color_legend: Vec<(Box<str>, plot_utils::Rgb)> =
            vec![(format!("C{c}").into_boxed_str(), community_color)];
        let opts = plot_utils::HintonOpts {
            row_labels: Some(&row_labels),
            col_labels: Some(&col_labels),
            row_order: Some(&row_order),
            col_order: Some(&col_order),
            col_colors: None,
            cell_colors: Some(&cell_colors),
            scale: plot_utils::HintonScale::Sqrt,
            cell_px: 18.0,
            font_px: 11.0,
            title: Some(&title),
            grid_stroke_px: 0.4,
            grid_color: (220, 220, 220),
            color_legend: Some(&color_legend),
        };
        let svg = plot_utils::render_hinton(&mat, n_l, n_r, &opts);
        let size = plot_utils::hinton_size(n_l, n_r, &opts);
        emit_svg_outputs(
            args,
            &svg,
            &out_dir.join(format!("C{c}.summary")),
            (size.width_px, size.height_px),
            emitted,
        )?;
    }

    Ok(())
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
