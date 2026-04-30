//! `senna plot-topic` — admixture-style structure-bar plots per batch
//! plus a gene × topic dictionary summary (Hinton / heatmap).
//!
//! Preferred invocation is `senna plot-topic --from {prefix}.senna.json`;
//! the manifest carries data/batch/latent/dictionary paths produced by
//! `senna topic` / `itopic` / `joint-topic`. CLI flags override
//! manifest values, mirroring the `senna plot` resolution rules at
//! `fit_plot.rs:428`.
//!
//! Outputs default to PDF only — pass `--svg` / `--png` to also emit
//! those formats. Layout under `{out}.plots/`:
//!
//! ```text
//! {out}.plots/
//! ├── struct/
//! │   ├── all.pdf                   # combined panel; widths ∝ #cells
//! │   └── by_batch/{batch}.pdf
//! └── dict/
//!     ├── hinton.pdf                # if n_genes ≤ 100
//!     └── heatmap.pdf               # if n_genes > 100
//! ```

use crate::embed_common::*;
use crate::run_manifest::{self, RunManifest};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use plot_utils::palette::{self, Palette, Rgb};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const HINTON_GENE_THRESHOLD: usize = 100;

/// What to group cells by when faceting the structure plot. `Batch` is
/// the default (one panel per data-source). `Annotation` requires an
/// `argmax.tsv` from `senna annotate` and produces one panel per cell
/// type — the canonical fastTopics structure-plot view.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum GroupBy {
    Batch,
    Annotation,
}

impl GroupBy {
    /// Output subdir suffix: `{out}.plots/struct/by_{suffix}/...`.
    fn subdir_suffix(&self) -> &'static str {
        match self {
            GroupBy::Batch => "batch",
            GroupBy::Annotation => "celltype",
        }
    }
}

/// Cell ordering inside a single batch panel.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum CellOrder {
    /// Group by argmax topic (T-id ascending), then within-group sort by
    /// descending dominant-topic probability. Default — produces the
    /// canonical "blocks" structure plot.
    Argmax,
    /// Sort by ascending `x` from `senna layout`'s `cell_coords.parquet`.
    /// Approximates fastTopic's per-group 1-D t-SNE ordering using the
    /// already-computed 2-D layout. Falls back to Argmax if no layout.
    Coord,
}

/// Hinton / heatmap magnitude → side fraction.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum HintonScaleArg {
    Sqrt,
    Log1p,
    Linear,
}

impl HintonScaleArg {
    fn into_inner(self) -> plot_utils::HintonScale {
        match self {
            HintonScaleArg::Sqrt => plot_utils::HintonScale::Sqrt,
            HintonScaleArg::Log1p => plot_utils::HintonScale::Log1p,
            HintonScaleArg::Linear => plot_utils::HintonScale::Linear,
        }
    }
}

#[derive(Args, Debug)]
pub struct PlotTopicArgs {
    #[arg(
        long,
        short = 'f',
        help = "Run manifest JSON from `senna topic`/`itopic`/`joint-topic`",
        long_help = "If set, fills in --latent / --dictionary and the batch-file \
                     list from the manifest's outputs/data sections. CLI flags \
                     still override individual values."
    )]
    pub from: Option<Box<str>>,

    #[arg(
        long,
        help = "Latent parquet (cells × K log-softmax topic proportions)"
    )]
    pub latent: Option<Box<str>>,

    #[arg(
        long,
        help = "Dictionary parquet (gene × K). Defaults to manifest's dictionary_empirical else dictionary"
    )]
    pub dictionary: Option<Box<str>>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (defaults to manifest's `prefix` when --from is used)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 12.0,
        help = "Combined-panel width (inches); per-batch width is allocated proportional to cell count"
    )]
    pub width: f32,

    #[arg(long, default_value_t = 2.0, help = "Per-panel height (inches)")]
    pub height: f32,

    #[arg(long, default_value_t = 300, help = "Output DPI for rasterized panels")]
    pub dpi: u32,

    #[arg(
        long,
        value_enum,
        default_value_t = GroupBy::Batch,
        help = "Group cells into panels by batch (default) or by `senna annotate` cell-type label"
    )]
    pub group_by: GroupBy,

    #[arg(
        long,
        help = "Argmax TSV from `senna annotate` (cell\\tcell_type\\tprobability). Defaults to manifest's annotate.argmax. Required for --group-by annotation."
    )]
    pub annotation: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = CellOrder::Argmax,
        help = "Cell ordering inside each panel"
    )]
    pub order: CellOrder,

    #[arg(
        long,
        value_enum,
        help = "Topic palette (default: manifest's defaults.palette, else `auto`)"
    )]
    pub palette: Option<Palette>,

    #[arg(
        long,
        default_value_t = 20,
        help = "Top-N genes per topic for dictionary plot (0 = use all genes)"
    )]
    pub top_genes: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = HintonScaleArg::Sqrt,
        help = "Magnitude → side mapping for Hinton (sqrt | log1p | linear)"
    )]
    pub hinton_scale: HintonScaleArg,

    #[arg(long, default_value_t = false, help = "Skip the structure-bar plot")]
    pub no_struct: bool,

    #[arg(long, default_value_t = false, help = "Skip the dictionary plot")]
    pub no_dict: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Also emit SVG (default: PDF only)"
    )]
    pub svg: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Also emit PNG (default: PDF only)"
    )]
    pub png: bool,

    #[arg(long, default_value_t = false, help = "Skip PDF output")]
    pub no_pdf: bool,
}

/// Resolved manifest + CLI inputs. Paths are absolute or already
/// resolved relative to the manifest directory.
struct ResolvedInputs {
    out: String,
    latent: String,
    dictionary: Option<String>,
    /// Per-data-file batch label list paths (parallel to `data_input`).
    /// Empty when the run had no `--batch-files`.
    batch_files: Vec<String>,
    /// Original data files (used for the cell-count-per-file fallback when
    /// `batch_files` is empty).
    data_files: Vec<String>,
    /// Optional `cell_coords.parquet` for `--order coord`.
    cell_coords: Option<String>,
    /// Optional `argmax.tsv` from `senna annotate` for `--group-by annotation`.
    annotation: Option<String>,
    palette: Palette,
}

pub fn fit_plot_topic(args: &PlotTopicArgs) -> anyhow::Result<()> {
    let resolved = resolve_inputs(args)?;
    let plot_root = format!("{}.plots", resolved.out);

    // Load topic proportions (cells × K), exp-normalize.
    let MatWithNames {
        rows: cell_names,
        cols: topic_cols,
        mat: latent,
    } = Mat::from_parquet(&resolved.latent)?;
    let n_cells = latent.nrows();
    let n_topics = latent.ncols();
    info!(
        "Loaded latent {} ({} cells × {} topics)",
        resolved.latent, n_cells, n_topics
    );

    // Topic IDs: parse from "T{c}" naming when present so the palette
    // keys by topic-id rather than column index. Falls back to 0..K.
    let topic_ids = try_parse_axis_ids(&topic_cols, "T")
        .unwrap_or_else(|| (0..n_topics as i64).collect::<Vec<_>>());

    let palette = palette::resolve(&resolved.palette, n_topics);
    // One color per *column position*, but keyed by topic-id so a topic
    // T3 shares its color with `senna plot --colour-by topic` and the
    // dictionary plot.
    let topic_colors: Vec<Rgb> = topic_ids
        .iter()
        .map(|&tid| palette::color(&palette, tid.max(0) as usize))
        .collect();

    // Row-stochastic probs via exp + per-row renorm. Stored row-major.
    let mut probs: Vec<f32> = vec![0.0; n_cells * n_topics];
    for i in 0..n_cells {
        let row_off = i * n_topics;
        let row = &mut probs[row_off..row_off + n_topics];
        let mut s = 0.0f32;
        for (j, v) in row.iter_mut().enumerate() {
            *v = latent[(i, j)].exp();
            s += *v;
        }
        if s > 0.0 {
            let inv = 1.0 / s;
            for v in row {
                *v *= inv;
            }
        }
    }

    if !args.no_struct {
        let group_labels = match args.group_by {
            GroupBy::Batch => load_batch_labels(&resolved, n_cells)?,
            GroupBy::Annotation => load_annotation_labels(&resolved, &cell_names)?,
        };
        render_structure_plots(
            &probs,
            n_topics,
            &topic_ids,
            &topic_colors,
            &group_labels,
            args,
            &resolved,
            &plot_root,
        )?;
    }

    if !args.no_dict {
        render_dict_plot(&topic_ids, &topic_colors, args, &resolved, &plot_root)?;
    }

    Ok(())
}

fn resolve_inputs(args: &PlotTopicArgs) -> anyhow::Result<ResolvedInputs> {
    let (manifest, manifest_dir): (Option<RunManifest>, PathBuf) = match &args.from {
        Some(p) => {
            let (m, dir) = RunManifest::load(Path::new(p.as_ref()))?;
            info!("Loaded run manifest {} (kind: {})", p, m.kind);
            (Some(m), dir)
        }
        None => (None, PathBuf::from(".")),
    };

    let resolve_str = |s: &str| {
        run_manifest::resolve(&manifest_dir, s)
            .to_string_lossy()
            .into_owned()
    };

    let out = args
        .out
        .as_deref()
        .map(String::from)
        .or_else(|| manifest.as_ref().map(|m| m.prefix.clone()))
        .ok_or_else(|| anyhow::anyhow!("no --out given and no manifest prefix available"))?;

    let latent = args
        .latent
        .as_deref()
        .map(String::from)
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|m| m.outputs.latent.as_deref())
                .map(resolve_str)
        })
        .ok_or_else(|| anyhow::anyhow!("no --latent given and manifest has no outputs.latent"))?;

    // Dictionary preference: empirical (full-resolution) when present,
    // else the coarse-then-expanded `dictionary`. Mirrors the same
    // preference `senna annotate` uses.
    let dictionary = args.dictionary.as_deref().map(String::from).or_else(|| {
        manifest.as_ref().and_then(|m| {
            m.outputs
                .dictionary_empirical
                .as_deref()
                .or(m.outputs.dictionary.as_deref())
                .map(&resolve_str)
        })
    });

    let batch_files = manifest
        .as_ref()
        .map(|m| m.data.batch.iter().map(|p| resolve_str(p)).collect())
        .unwrap_or_default();

    let data_files = manifest
        .as_ref()
        .map(|m| m.data.input.iter().map(|p| resolve_str(p)).collect())
        .unwrap_or_default();

    let cell_coords = manifest
        .as_ref()
        .and_then(|m| m.layout.cell_coords.as_deref())
        .map(resolve_str);

    let annotation = args.annotation.as_deref().map(String::from).or_else(|| {
        manifest
            .as_ref()
            .and_then(|m| m.annotate.argmax.as_deref())
            .map(resolve_str)
    });

    let palette = args
        .palette
        .clone()
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|m| m.defaults.palette.as_deref())
                .and_then(|s| <Palette as clap::ValueEnum>::from_str(s, true).ok())
        })
        .unwrap_or(Palette::Auto);

    Ok(ResolvedInputs {
        out,
        latent,
        dictionary,
        batch_files,
        data_files,
        cell_coords,
        annotation,
        palette,
    })
}

/// Per-cell batch label, length == `n_cells`, in `latent.parquet` row
/// order. Reads the original batch-file paths from the manifest
/// (matching pinto's "paths-in-json" pattern); falls back to a synthetic
/// label per data-file when no batch files were provided.
fn load_batch_labels(resolved: &ResolvedInputs, n_cells: usize) -> anyhow::Result<Vec<Box<str>>> {
    use matrix_util::common_io::read_lines;

    if !resolved.batch_files.is_empty() {
        let mut all = Vec::with_capacity(n_cells);
        for bf in &resolved.batch_files {
            info!("Reading batch file: {bf}");
            for s in read_lines(bf)? {
                all.push(s);
            }
        }
        if all.len() != n_cells {
            anyhow::bail!(
                "batch labels total {} != latent rows {} (manifest data.batch may be stale)",
                all.len(),
                n_cells,
            );
        }
        return Ok(all);
    }

    if resolved.data_files.is_empty() {
        // No data file list either — single synthetic batch.
        return Ok(vec!["all".into(); n_cells]);
    }

    // No batch files: derive a label per data file from its basename
    // (with `.zarr.zip`/`.zarr`/`.h5` stripped) — same convention
    // SparseIoVec uses for batch identity. Falls back to the file index
    // only if a basename can't be extracted, and disambiguates duplicate
    // basenames with a `_{i}` suffix. Cheap because num_columns() reads
    // only the on-disk index, not the full matrix.
    use data_beans::convert::try_open_or_convert;
    use data_beans::hdf5_io::strip_backend_suffix;
    let raw_labels: Vec<Box<str>> = resolved
        .data_files
        .iter()
        .enumerate()
        .map(|(idx, df)| {
            Path::new(df.as_str())
                .file_name()
                .and_then(|s| s.to_str())
                .map(strip_backend_suffix)
                .map(Box::<str>::from)
                .unwrap_or_else(|| idx.to_string().into_boxed_str())
        })
        .collect();
    let mut counts: FxHashMap<&str, usize> = FxHashMap::default();
    for n in &raw_labels {
        *counts.entry(n.as_ref()).or_insert(0) += 1;
    }
    let mut seen: FxHashMap<&str, usize> = FxHashMap::default();
    let unique_labels: Vec<Box<str>> = raw_labels
        .iter()
        .map(|n| {
            if counts[n.as_ref()] == 1 {
                n.clone()
            } else {
                let k = seen.entry(n.as_ref()).or_insert(0);
                let s = format!("{}_{}", n, k).into_boxed_str();
                *k += 1;
                s
            }
        })
        .collect();
    let mut all: Vec<Box<str>> = Vec::with_capacity(n_cells);
    for (df, label) in resolved.data_files.iter().zip(unique_labels.iter()) {
        let n = try_open_or_convert(df)?
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("data file {df} has no column count"))?;
        all.extend(std::iter::repeat_n(label.clone(), n));
    }
    if all.len() != n_cells {
        anyhow::bail!(
            "fallback batch labels {} != latent rows {} (data_files inconsistent)",
            all.len(),
            n_cells,
        );
    }
    Ok(all)
}

/// Per-cell label from `senna annotate`'s `argmax.tsv`. The TSV's
/// `cell` column is matched against `latent.parquet`'s row names —
/// annotate may have run on a subset, so missing cells are tagged
/// `"unannotated"` rather than failing.
fn load_annotation_labels(
    resolved: &ResolvedInputs,
    cell_names: &[Box<str>],
) -> anyhow::Result<Vec<Box<str>>> {
    let path = resolved.annotation.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "--group-by annotation requires --annotation PATH or `senna annotate` \
             must have populated manifest.annotate.argmax"
        )
    })?;
    info!("Reading annotation labels from {path}");
    let content = fs::read_to_string(Path::new(path))?;
    let mut by_cell: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
    for (line_no, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Header: cell\tcell_type\tprobability — skip if first column
        // is "cell" (matches what annotate writes at run.rs:71).
        let mut parts = line.split('\t');
        let cell = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("annotation TSV line {}: missing cell", line_no + 1))?;
        let label = parts.next().ok_or_else(|| {
            anyhow::anyhow!("annotation TSV line {}: missing cell_type", line_no + 1)
        })?;
        if cell == "cell" && label == "cell_type" {
            continue;
        }
        by_cell.insert(cell.into(), label.into());
    }
    let unannotated: Box<str> = "unannotated".into();
    let mut n_missing = 0usize;
    let labels: Vec<Box<str>> = cell_names
        .iter()
        .map(|c| {
            by_cell.get(c).cloned().unwrap_or_else(|| {
                n_missing += 1;
                unannotated.clone()
            })
        })
        .collect();
    if n_missing > 0 {
        info!(
            "annotation: {n_missing}/{} cells absent from {path} → tagged 'unannotated'",
            cell_names.len()
        );
    }
    Ok(labels)
}

/// Cell indices grouped by label, returned in **alphabetical** label
/// order. Matches the canonical fastTopics structure-plot facet order
/// (B cell, CD14+, CD34+, NK cell, T cell …) and is stable across
/// reruns regardless of cell input order.
fn cells_by_batch(batch_labels: &[Box<str>]) -> Vec<(Box<str>, Vec<usize>)> {
    let mut buckets: FxHashMap<Box<str>, Vec<usize>> = FxHashMap::default();
    for (i, b) in batch_labels.iter().enumerate() {
        buckets.entry(b.clone()).or_default().push(i);
    }
    let mut keys: Vec<Box<str>> = buckets.keys().cloned().collect();
    keys.sort_unstable();
    keys.into_iter()
        .map(|b| {
            let v = buckets.remove(&b).unwrap_or_default();
            (b, v)
        })
        .collect()
}

/// Global display order for the K topic columns: descending total
/// prevalence (sum of probabilities across all cells). The most
/// prevalent topic comes first, so the same color block lands at the
/// same horizontal position in every batch panel — gives the structure
/// plot visual continuity across batches without changing the topic ↔
/// color identity (which is still keyed by topic-id, see `topic_colors`
/// in `fit_plot_topic`).
fn global_topic_order(probs: &[f32], n_topics: usize) -> Vec<usize> {
    if n_topics == 0 || probs.is_empty() {
        return (0..n_topics).collect();
    }
    let n_cells = probs.len() / n_topics;
    let mut totals = vec![0.0f32; n_topics];
    for i in 0..n_cells {
        let row = &probs[i * n_topics..(i + 1) * n_topics];
        for (j, &v) in row.iter().enumerate() {
            if v.is_finite() && v > 0.0 {
                totals[j] += v;
            }
        }
    }
    let mut order: Vec<usize> = (0..n_topics).collect();
    order.sort_by(|&a, &b| {
        totals[b]
            .partial_cmp(&totals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Argmax-then-dominant-prob ordering inside one batch's cell list. The
/// primary sort key is the *display rank* of each cell's argmax topic
/// (so cells dominated by the same topic land at the same horizontal
/// position across batches); secondary key is descending dominant prob.
fn order_by_argmax(
    cells: &[usize],
    probs: &[f32],
    n_topics: usize,
    topic_rank: &[usize],
) -> Vec<usize> {
    let mut keyed: Vec<(usize, usize, f32)> = cells
        .iter()
        .map(|&i| {
            let row = &probs[i * n_topics..(i + 1) * n_topics];
            let mut best_j = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_j = j;
                }
            }
            (i, topic_rank[best_j], best_v)
        })
        .collect();
    keyed.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    keyed.into_iter().map(|(i, _, _)| i).collect()
}

/// Sort cells by ascending x from `cell_coords.parquet`. Falls back to
/// argmax order if the file is missing or has no `x` column.
fn order_by_coord(
    cells: &[usize],
    cell_coords_path: Option<&str>,
    probs: &[f32],
    n_topics: usize,
    topic_rank: &[usize],
) -> anyhow::Result<Vec<usize>> {
    let Some(path) = cell_coords_path else {
        info!("--order coord: no layout.cell_coords in manifest, falling back to argmax");
        return Ok(order_by_argmax(cells, probs, n_topics, topic_rank));
    };
    let MatWithNames { cols, mat, .. } = Mat::from_parquet(path)?;
    let Some(xj) = cols.iter().position(|c| &**c == "x") else {
        info!("--order coord: no 'x' column in {path}, falling back to argmax");
        return Ok(order_by_argmax(cells, probs, n_topics, topic_rank));
    };
    if mat.nrows() < cells.iter().copied().max().unwrap_or(0) + 1 {
        info!("--order coord: cell_coords too short, falling back to argmax");
        return Ok(order_by_argmax(cells, probs, n_topics, topic_rank));
    }
    let mut keyed: Vec<(usize, f32)> = cells.iter().map(|&i| (i, mat[(i, xj)])).collect();
    keyed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(keyed.into_iter().map(|(i, _)| i).collect())
}

/// Layout knobs for a structure-plot SVG (single-panel or combined).
/// Pixel-level dimensions are derived from `args.{width,height,dpi}` once
/// per call so the per-cell pixel density is identical between the
/// standalone per-group panels and the combined `all.*` plot.
struct StructDims {
    bar_h: u32,
    label_band: u32,
    legend_band: u32,
    /// Horizontal gap between adjacent panels in the combined view.
    panel_gap: u32,
}

#[allow(clippy::too_many_arguments)]
fn render_structure_plots(
    probs: &[f32],
    n_topics: usize,
    topic_ids: &[i64],
    topic_colors: &[Rgb],
    group_labels: &[Box<str>],
    args: &PlotTopicArgs,
    resolved: &ResolvedInputs,
    plot_root: &str,
) -> anyhow::Result<()> {
    let groups = cells_by_batch(group_labels);
    if groups.is_empty() {
        info!("No cells to plot for structure plot");
        return Ok(());
    }

    let struct_dir = format!("{plot_root}/struct");
    let by_group_dir = format!("{struct_dir}/by_{}", args.group_by.subdir_suffix());
    // Clear per-group outputs so renaming (e.g., index labels → basenames,
    // or batch reruns with a different cohort) doesn't leave stale files
    // alongside the new ones. Only this dir is owned exclusively by
    // plot-topic's per-group writer; sibling dirs (e.g. `by_celltype/`
    // when this run uses `by_batch/`) are preserved.
    if Path::new(&by_group_dir).exists() {
        fs::remove_dir_all(&by_group_dir)?;
    }
    fs::create_dir_all(&by_group_dir)?;

    // Global topic display order: descending total prevalence across all
    // cells. Cells dominated by the same topic land at the same x-band
    // in every panel, so structure-bar blocks read consistently across
    // batches. `topic_rank[j]` = position of topic-column `j` in the
    // display order; `topic_display_order[i]` = topic-column at slot `i`.
    let topic_display_order = global_topic_order(probs, n_topics);
    let mut topic_rank = vec![0usize; n_topics];
    for (pos, &j) in topic_display_order.iter().enumerate() {
        topic_rank[j] = pos;
    }

    let ordered: Vec<(Box<str>, Vec<usize>)> = match args.order {
        CellOrder::Argmax => groups
            .iter()
            .map(|(b, cs)| (b.clone(), order_by_argmax(cs, probs, n_topics, &topic_rank)))
            .collect(),
        CellOrder::Coord => groups
            .iter()
            .map(|(b, cs)| {
                let o = order_by_coord(
                    cs,
                    resolved.cell_coords.as_deref(),
                    probs,
                    n_topics,
                    &topic_rank,
                )?;
                Ok::<_, anyhow::Error>((b.clone(), o))
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
    };

    let total_cells: usize = ordered.iter().map(|(_, v)| v.len()).sum();
    if total_cells == 0 {
        info!("No cells in any group, skipping structure plot");
        return Ok(());
    }

    let dims = StructDims {
        bar_h: (args.height * args.dpi as f32).round().max(1.0) as u32,
        label_band: (args.dpi as f32 * 0.35).round().max(20.0) as u32,
        legend_band: (args.dpi as f32 * 0.6).round().max(40.0) as u32,
        // ~0.06 in @ 300 DPI ≈ 18 px; small but visible separator.
        panel_gap: (args.dpi as f32 * 0.06).round().max(8.0) as u32,
    };
    // Reserve gap room when budgeting per-panel widths so the combined
    // SVG width still respects `args.width`. With a single panel there
    // are no gaps.
    let n_panels = ordered.len();
    let total_gap_px = if n_panels > 1 {
        dims.panel_gap * (n_panels as u32 - 1)
    } else {
        0
    };
    let usable_width_px = (args.width * args.dpi as f32).round().max(1.0) as u32;
    let total_width_px = usable_width_px.saturating_sub(total_gap_px).max(1);

    // Per-group raster panels (rayon-parallel; each panel is an
    // independent tiny-skia render).
    let panels: Vec<PanelOut> = ordered
        .par_iter()
        .map(|(batch, order)| -> anyhow::Result<PanelOut> {
            let n = order.len();
            let panel_w = ((n as f64 / total_cells as f64) * total_width_px as f64)
                .round()
                .max(1.0) as u32;
            // Reorder probs into a contiguous [n × K] row-major slice so
            // structure_bar_png can iterate linearly.
            let mut buf = Vec::with_capacity(n * n_topics);
            for &cell in order {
                buf.extend_from_slice(&probs[cell * n_topics..(cell + 1) * n_topics]);
            }
            let png = plot_utils::structure_bar_png(
                &buf,
                n,
                n_topics,
                panel_w,
                dims.bar_h,
                topic_colors,
            )?;
            Ok(PanelOut {
                batch: batch.clone(),
                n_cells: n,
                width_px: panel_w,
                png,
            })
        })
        .collect::<anyhow::Result<_>>()?;

    panels.par_iter().try_for_each(|p| -> anyhow::Result<()> {
        let svg = emit_struct_svg(
            std::slice::from_ref(p),
            &dims,
            topic_ids,
            topic_colors,
            &topic_display_order,
        );
        let base = format!("{by_group_dir}/{}", sanitize(&p.batch));
        emit_outputs(
            &svg,
            p.width_px + dims.legend_band,
            dims.label_band + dims.bar_h,
            &base,
            args,
        )
    })?;

    let combined_bars_w: u32 = panels.iter().map(|p| p.width_px).sum();
    let svg = emit_struct_svg(
        &panels,
        &dims,
        topic_ids,
        topic_colors,
        &topic_display_order,
    );
    let combined_gap_w: u32 = if panels.len() > 1 {
        dims.panel_gap * (panels.len() as u32 - 1)
    } else {
        0
    };
    emit_outputs(
        &svg,
        combined_bars_w + combined_gap_w + dims.legend_band,
        dims.label_band + dims.bar_h,
        &format!("{struct_dir}/all"),
        args,
    )?;

    Ok(())
}

struct PanelOut {
    batch: Box<str>,
    n_cells: usize,
    width_px: u32,
    png: Vec<u8>,
}

const PANEL_LABEL_FONT_FRAC: f32 = 0.55;

/// Emit a structure-plot SVG laying out `panels` left-to-right with a
/// single shared topic legend on the right. A 1-element slice produces
/// the standalone per-group view; a multi-element slice produces the
/// combined `all.*` view — the layout is identical.
fn emit_struct_svg(
    panels: &[PanelOut],
    d: &StructDims,
    topic_ids: &[i64],
    topic_colors: &[Rgb],
    topic_display_order: &[usize],
) -> String {
    let bars_w: u32 = panels.iter().map(|p| p.width_px).sum();
    let n_gaps = panels.len().saturating_sub(1) as u32;
    let total_gaps_w = d.panel_gap * n_gaps;
    let total_w = bars_w + total_gaps_w + d.legend_band;
    let total_h = d.label_band + d.bar_h;
    let label_fs = (d.label_band as f32 * PANEL_LABEL_FONT_FRAC).round();

    let mut s = String::with_capacity(panels.iter().map(|p| p.png.len()).sum::<usize>() * 2 + 4096);
    let _ = write!(
        s,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <svg xmlns=\"http://www.w3.org/2000/svg\" \
             xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
             viewBox=\"0 0 {total_w} {total_h}\" width=\"{total_w}\" height=\"{total_h}\">\n",
    );
    // White background — PDF backends don't have a defined fill behind
    // the raster <image>, so missing this paints garbage on some viewers.
    let _ = writeln!(
        s,
        "  <rect x=\"0\" y=\"0\" width=\"{total_w}\" height=\"{total_h}\" fill=\"white\"/>"
    );

    let mut x_offset: u32 = 0;
    for (panel_idx, p) in panels.iter().enumerate() {
        let label = format!("{} (n={})", plot_utils::escape_xml(&p.batch), p.n_cells);
        let _ = writeln!(
            s,
            "  <text x=\"{x}\" y=\"{y}\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{label_fs}\" text-anchor=\"middle\" dominant-baseline=\"central\" \
             fill=\"black\">{label}</text>",
            x = x_offset + p.width_px / 2,
            y = d.label_band / 2,
        );
        let b64 = BASE64.encode(&p.png);
        let _ = writeln!(
            s,
            "  <image x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" \
             preserveAspectRatio=\"none\" href=\"data:image/png;base64,{b64}\"/>",
            x = x_offset,
            y = d.label_band,
            w = p.width_px,
            h = d.bar_h,
        );
        let _ = writeln!(
            s,
            "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" \
             fill=\"none\" stroke=\"black\" stroke-width=\"1\"/>",
            x = x_offset as f32 + 0.5,
            y = d.label_band as f32 + 0.5,
            w = p.width_px as f32 - 1.0,
            h = d.bar_h as f32 - 1.0,
        );
        x_offset += p.width_px;
        if panel_idx + 1 < panels.len() {
            x_offset += d.panel_gap;
        }
    }

    // Legend lists topics in display order so the top swatch matches the
    // dominant block at the leftmost x in every panel.
    emit_topic_legend(
        &mut s,
        bars_w + total_gaps_w,
        d,
        topic_ids,
        topic_colors,
        topic_display_order,
    );
    let _ = writeln!(s, "</svg>");
    s
}

fn emit_topic_legend(
    s: &mut String,
    bar_x_end: u32,
    d: &StructDims,
    topic_ids: &[i64],
    topic_colors: &[Rgb],
    topic_display_order: &[usize],
) {
    let n = topic_ids.len();
    if n == 0 || d.legend_band < 8 {
        return;
    }
    let pad_left = 8.0;
    let swatch = (d.legend_band as f32 * 0.18).clamp(8.0, 18.0);
    let line_h = swatch + 4.0;
    let total_legend_h = line_h * n as f32;
    let start_y = d.label_band as f32 + ((d.bar_h as f32 - total_legend_h) * 0.5).max(0.0);

    let _ = writeln!(s, "  <g id=\"legend\">");
    for (i, &j) in topic_display_order.iter().enumerate() {
        let tid = topic_ids[j];
        let (r, g, b) = topic_colors[j];
        let y = start_y + i as f32 * line_h;
        let x = bar_x_end as f32 + pad_left;
        let _ = writeln!(
            s,
            "    <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{sw:.1}\" height=\"{sw:.1}\" \
             fill=\"rgb({r},{g},{b})\" stroke=\"black\" stroke-width=\"0.5\"/>",
            sw = swatch,
        );
        let _ = writeln!(
            s,
            "    <text x=\"{tx:.1}\" y=\"{ty:.1}\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{fs:.1}\" dominant-baseline=\"central\" fill=\"black\">T{tid}</text>",
            tx = x + swatch + 4.0,
            ty = y + swatch * 0.5,
            fs = swatch * 0.85,
        );
    }
    let _ = writeln!(s, "  </g>");
}

// ---------------------------------------------------------------------------
// Dictionary plot (Hinton ≤ 100 rows, heatmap above).
// ---------------------------------------------------------------------------

fn render_dict_plot(
    topic_ids: &[i64],
    topic_colors: &[Rgb],
    args: &PlotTopicArgs,
    resolved: &ResolvedInputs,
    plot_root: &str,
) -> anyhow::Result<()> {
    let Some(dict_path) = resolved.dictionary.as_deref() else {
        info!("No dictionary parquet in manifest, skipping dictionary plot");
        return Ok(());
    };

    let MatWithNames {
        rows: gene_names,
        cols: dict_cols,
        mat: dict_gk,
    } = Mat::from_parquet(dict_path)?;
    let n_genes_full = dict_gk.nrows();
    let n_topics = dict_gk.ncols();
    info!(
        "Loaded dictionary {} ({} genes × {} topics)",
        dict_path, n_genes_full, n_topics
    );

    if n_topics != topic_ids.len() {
        anyhow::bail!(
            "dictionary K={} does not match latent K={}",
            n_topics,
            topic_ids.len()
        );
    }

    // Topic-loadings on the simplex / signed scale: the "weights" we
    // visualize in Hinton are non-negative magnitudes. For
    // dictionary_empirical (already simplex-normalized) values are ≥ 0;
    // for dictionary (log-prob) we exp-then-normalize per column.
    let mut weights_gk: Vec<f32> = Vec::with_capacity(n_genes_full * n_topics);
    let any_negative = (0..n_genes_full).any(|i| (0..n_topics).any(|j| dict_gk[(i, j)] < 0.0));
    if any_negative {
        // Treat as log-prob: exp + per-column renorm.
        let mut col_sums = vec![0.0f32; n_topics];
        let mut tmp = vec![0.0f32; n_genes_full * n_topics];
        for i in 0..n_genes_full {
            for j in 0..n_topics {
                let v = dict_gk[(i, j)].exp();
                tmp[i * n_topics + j] = v;
                col_sums[j] += v;
            }
        }
        for i in 0..n_genes_full {
            for j in 0..n_topics {
                let s = col_sums[j].max(1e-30);
                tmp[i * n_topics + j] /= s;
            }
        }
        weights_gk = tmp;
    } else {
        for i in 0..n_genes_full {
            for j in 0..n_topics {
                weights_gk.push(dict_gk[(i, j)].max(0.0));
            }
        }
    }

    // Gene selection.
    let gene_idx: Vec<usize> = if args.top_genes == 0 {
        (0..n_genes_full).collect()
    } else {
        top_genes_per_topic(&weights_gk, n_genes_full, n_topics, args.top_genes)
    };
    let n_genes = gene_idx.len();
    if n_genes == 0 {
        info!("No genes selected for dictionary plot, skipping");
        return Ok(());
    }
    info!(
        "Dictionary plot: {n_genes} genes × {n_topics} topics ({} mode)",
        if n_genes <= HINTON_GENE_THRESHOLD {
            "hinton"
        } else {
            "heatmap"
        }
    );

    // Verify dict columns parse to topic IDs and align with topic_ids.
    let dict_topic_ids = try_parse_axis_ids(&dict_cols, "T")
        .unwrap_or_else(|| (0..n_topics as i64).collect::<Vec<_>>());

    // Submatrix (n_genes × n_topics) row-major in selected gene order.
    let mut sub: Vec<f32> = Vec::with_capacity(n_genes * n_topics);
    let mut sub_gene_names: Vec<Box<str>> = Vec::with_capacity(n_genes);
    for &gi in &gene_idx {
        for j in 0..n_topics {
            sub.push(weights_gk[gi * n_topics + j]);
        }
        sub_gene_names.push(gene_names[gi].clone());
    }

    let dict_dir = format!("{plot_root}/dict");
    // Clear so a hinton↔heatmap switch (driven by n_genes crossing
    // HINTON_GENE_THRESHOLD) doesn't leave both old + new files
    // side-by-side.
    if Path::new(&dict_dir).exists() {
        fs::remove_dir_all(&dict_dir)?;
    }
    fs::create_dir_all(&dict_dir)?;

    // Diagonalize: rows ordered so peak topic per gene runs along the
    // main diagonal (block-band layout), columns reordered to match.
    // Same recipe Hinton uses internally; we apply it to both render
    // paths so heatmap and Hinton share the visual convention.
    let (row_order, col_order) = plot_utils::diagonalize_order(&sub, n_genes, n_topics);

    if n_genes <= HINTON_GENE_THRESHOLD {
        let topic_labels: Vec<Box<str>> = dict_topic_ids
            .iter()
            .map(|tid| format!("T{tid}").into_boxed_str())
            .collect();
        let opts = plot_utils::HintonOpts {
            row_labels: Some(&sub_gene_names),
            col_labels: Some(&topic_labels),
            row_order: Some(&row_order),
            col_order: Some(&col_order),
            col_colors: Some(topic_colors),
            cell_colors: None,
            scale: args.hinton_scale.clone().into_inner(),
            cell_px: 18.0,
            font_px: 11.0,
            title: None,
            grid_stroke_px: 0.0,
            grid_color: (220, 220, 220),
            color_legend: None,
        };
        let svg = plot_utils::render_hinton(&sub, n_genes, n_topics, &opts);
        let size = plot_utils::hinton_size(n_genes, n_topics, &opts);
        let base = format!("{dict_dir}/hinton");
        emit_outputs(&svg, size.width_px, size.height_px, &base, args)?;
    } else {
        // Heatmap rasterized via tiny-skia and embedded as <image>. Rows
        // displayed in `row_order`; columns in `col_order`. Gene labels
        // alongside each row, with font scaled to row height.
        let cell_w = 14u32;
        // Row pixel height auto-tunes to keep the heatmap legible: we
        // can't fit 1000 rows tall on one page, so cap total bar_h at
        // ~16 in @ args.dpi and shrink cell_h if needed (gene labels
        // also scale with cell_h).
        let max_bar_h = (args.dpi as f32 * 16.0).round() as u32;
        let cell_h = (max_bar_h / n_genes as u32).clamp(1, 8);
        let bar_w = n_topics as u32 * cell_w;
        let bar_h = n_genes as u32 * cell_h;

        // Estimate gene-label character width at the chosen font size.
        let label_font_px = (cell_h as f32 * 0.85).clamp(4.0, 9.0);
        let max_label_chars = sub_gene_names
            .iter()
            .map(|n| n.chars().count())
            .max()
            .unwrap_or(8) as f32;
        // Crude monospace-ish width estimate for sans-serif at this font
        // size (≈ 0.55 × font height per char).
        let label_band_left = ((max_label_chars * label_font_px * 0.55).ceil() as u32 + 8).min(220);
        let label_band_top: u32 = 28;
        let cbar_w: u32 = 12;
        let cbar_pad: u32 = 12;
        let cbar_label_w: u32 = 56; // room for "log10  -3.21" labels
        let right_band = cbar_pad + cbar_w + cbar_label_w + 8;
        let total_w = label_band_left + bar_w + right_band;
        let total_h = label_band_top + bar_h + 8;

        // Reorder sub matrix so PNG draws cells in display order.
        let mut sub_disp: Vec<f32> = Vec::with_capacity(n_genes * n_topics);
        for &rr in &row_order {
            for &cc in &col_order {
                sub_disp.push(sub[rr * n_topics + cc]);
            }
        }

        // Percentile-clipped log10 bounds for the diverging color scale.
        let (lo_log, hi_log) = log10_clip_bounds(&sub_disp, 0.01, 0.99);
        info!(
            "Heatmap log10 range (1%–99%): [{:.2}, {:.2}]",
            lo_log, hi_log
        );

        let png =
            render_dict_heatmap_png(&sub_disp, n_genes, n_topics, cell_w, cell_h, lo_log, hi_log)?;

        let mut s = String::with_capacity(png.len() * 2 + sub_gene_names.len() * 64 + 4096);
        let _ = write!(
            s,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <svg xmlns=\"http://www.w3.org/2000/svg\" \
                 xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
                 viewBox=\"0 0 {total_w} {total_h}\" width=\"{total_w}\" height=\"{total_h}\">\n",
        );
        let _ = writeln!(
            s,
            "  <rect x=\"0\" y=\"0\" width=\"{total_w}\" height=\"{total_h}\" fill=\"white\"/>"
        );
        // Topic labels at top, in display order.
        for (cc_disp, &cc) in col_order.iter().enumerate() {
            let tid = dict_topic_ids[cc];
            let cx = label_band_left as f32 + (cc_disp as f32 + 0.5) * cell_w as f32;
            let _ = writeln!(
                s,
                "  <text x=\"{cx:.1}\" y=\"{y}\" font-family=\"Helvetica, Arial, sans-serif\" \
                 font-size=\"10\" text-anchor=\"middle\" dominant-baseline=\"central\">T{tid}</text>",
                y = label_band_top / 2,
            );
        }
        // Heatmap raster.
        let b64 = BASE64.encode(&png);
        let _ = writeln!(
            s,
            "  <image x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" \
             preserveAspectRatio=\"none\" href=\"data:image/png;base64,{b64}\"/>",
            x = label_band_left,
            y = label_band_top,
            w = bar_w,
            h = bar_h,
        );
        // Colorbar (RdBu_r) on the right with min/mid/max log10 ticks.
        // Height matches the heatmap so cell rows and gradient stops are
        // directly comparable by y-position.
        let cbar_x = label_band_left + bar_w + cbar_pad;
        let cbar_y = label_band_top;
        let cbar_h = bar_h;
        let cbar_png = render_colorbar_png(cbar_w, cbar_h)?;
        let cbar_b64 = BASE64.encode(&cbar_png);
        let _ = writeln!(
            s,
            "  <image x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\" \
             preserveAspectRatio=\"none\" href=\"data:image/png;base64,{cbar_b64}\"/>",
            x = cbar_x,
            y = cbar_y,
            w = cbar_w,
            h = cbar_h,
        );
        let _ = writeln!(
            s,
            "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w}\" height=\"{h}\" \
             fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"/>",
            x = cbar_x as f32 + 0.5,
            y = cbar_y as f32 + 0.5,
            w = cbar_w.saturating_sub(1),
            h = cbar_h.saturating_sub(1),
        );
        let tick_x = (cbar_x + cbar_w + 4) as f32;
        let mid_log = 0.5 * (lo_log + hi_log);
        for (frac, val) in [(0.0_f32, hi_log), (0.5, mid_log), (1.0, lo_log)] {
            let ty = cbar_y as f32 + frac * cbar_h as f32;
            let _ = writeln!(
                s,
                "  <text x=\"{tx:.1}\" y=\"{ty:.1}\" font-family=\"Helvetica, Arial, sans-serif\" \
                 font-size=\"9\" dominant-baseline=\"central\">{v}</text>",
                tx = tick_x,
                v = format_sci(10f32.powf(val)),
            );
        }
        let _ = writeln!(
            s,
            "  <text x=\"{tx:.1}\" y=\"{ty}\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"9\" text-anchor=\"start\">β</text>",
            tx = cbar_x as f32,
            ty = cbar_y.saturating_sub(8),
        );
        // Gene labels along the left side. Skip when the row pitch is
        // too small to render legibly even at the auto-shrunk font; in
        // that case label every Nth row instead.
        let stride = if label_font_px < 5.0 {
            ((6.0 / cell_h as f32).ceil() as usize).max(1)
        } else {
            1
        };
        let _ = writeln!(
            s,
            "  <g id=\"gene-labels\" font-family=\"Helvetica, Arial, sans-serif\" \
             font-size=\"{fs:.1}\" text-anchor=\"end\" dominant-baseline=\"central\">",
            fs = label_font_px,
        );
        for (rr_disp, &rr) in row_order.iter().enumerate() {
            if rr_disp % stride != 0 {
                continue;
            }
            let cy = label_band_top as f32 + (rr_disp as f32 + 0.5) * cell_h as f32;
            let _ = writeln!(
                s,
                "    <text x=\"{x:.1}\" y=\"{cy:.1}\">{t}</text>",
                x = label_band_left as f32 - 4.0,
                t = plot_utils::escape_xml(&sub_gene_names[rr]),
            );
        }
        let _ = writeln!(s, "  </g>");
        let _ = writeln!(s, "</svg>");
        let base = format!("{dict_dir}/heatmap");
        emit_outputs(&s, total_w, total_h, &base, args)?;
    }

    Ok(())
}

/// Top-N gene indices per topic, unioned across topics, returned in
/// sorted order. Ranks by the gene's KL contribution to topic `j`,
///
///     score(g, j) = w_gj · ln((w_gj + ε) / (m_g + ε)),
///
/// where `m_g` is the gene's mean weight over the *other* `K − 1`
/// topics. The product form requires *both* high mass and high
/// specificity: housekeeping genes (high everywhere → ratio ≈ 1, log ≈ 0)
/// score near zero even when `w` is large, and ultra-rare genes (tiny
/// mass) also score near zero even when the ratio is huge. Topic-
/// specific markers — high mass *and* lifted relative to other topics —
/// dominate. ε is set to `1e−8 / K` so housekeeping doesn't get a free
/// boost from the smoothing on simplex-normalized inputs.
fn top_genes_per_topic(
    weights_gk: &[f32],
    n_genes: usize,
    n_topics: usize,
    top_n: usize,
) -> Vec<usize> {
    if top_n == 0 || n_genes == 0 || n_topics == 0 {
        return Vec::new();
    }
    let topn = top_n.min(n_genes);
    let mut keep = vec![false; n_genes];

    let mut row_sums = vec![0.0f32; n_genes];
    for i in 0..n_genes {
        for j in 0..n_topics {
            row_sums[i] += weights_gk[i * n_topics + j];
        }
    }

    let eps = 1e-8 / n_topics as f32;
    for j in 0..n_topics {
        let mut scored: Vec<(usize, f32, f32)> = (0..n_genes)
            .filter_map(|i| {
                let w = weights_gk[i * n_topics + j];
                if w <= 0.0 {
                    return None;
                }
                let other_mean = if n_topics > 1 {
                    (row_sums[i] - w).max(0.0) / (n_topics - 1) as f32
                } else {
                    0.0
                };
                let ratio = (w + eps) / (other_mean + eps);
                let score = w * ratio.ln();
                if !score.is_finite() || score <= 0.0 {
                    return None;
                }
                Some((i, score, w))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });
        for (gi, _, _) in scored.into_iter().take(topn) {
            keep[gi] = true;
        }
    }
    let mut out: Vec<usize> = (0..n_genes).filter(|&i| keep[i]).collect();
    out.sort_unstable();
    out
}

/// Compact scientific-notation formatter for colorbar tick labels:
/// `2.7e-5` style with one fractional digit. Avoids the noise of
/// printf's `{:e}` (which uses `2.7e0` for 2.7) and is short enough to
/// fit a vertical colorbar's right margin.
fn format_sci(v: f32) -> String {
    if !v.is_finite() || v == 0.0 {
        return "0".to_string();
    }
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    let mantissa = v / 10f32.powi(exp);
    // Round mantissa to 1 fractional digit; if rounding pushes it to 10,
    // bump the exponent so we still show "1.0e3" not "10.0e2".
    let rounded = (mantissa * 10.0).round() / 10.0;
    let (m, e) = if rounded.abs() >= 10.0 {
        (rounded / 10.0, exp + 1)
    } else {
        (rounded, exp)
    };
    if e == 0 {
        format!("{m:.1}")
    } else {
        format!("{m:.1}e{e}")
    }
}

/// RdBu_r diverging palette, 3-stop linear interpolation.
/// `t ∈ [0, 1]`: 0 = blue (low), 0.5 = white (mid), 1 = red (high).
/// Stops chosen to match matplotlib's `RdBu_r` endpoints.
fn rdbu_r(t: f32) -> Rgb {
    const LOW: (u8, u8, u8) = (33, 102, 172);
    const MID: (u8, u8, u8) = (247, 247, 247);
    const HIGH: (u8, u8, u8) = (178, 24, 43);
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8, u: f32| (a as f32 + (b as f32 - a as f32) * u).round() as u8;
    if t < 0.5 {
        let u = t / 0.5;
        (
            lerp(LOW.0, MID.0, u),
            lerp(LOW.1, MID.1, u),
            lerp(LOW.2, MID.2, u),
        )
    } else {
        let u = (t - 0.5) / 0.5;
        (
            lerp(MID.0, HIGH.0, u),
            lerp(MID.1, HIGH.1, u),
            lerp(MID.2, HIGH.2, u),
        )
    }
}

/// Percentile-clip bounds in log10 space across all finite, positive
/// entries of `sub`. Returns `(low_log10, high_log10)`. Uses 1st/99th
/// percentile so a handful of extreme values can't compress the rest of
/// the dynamic range. Falls back to a symmetric ±1 window if no positive
/// data is present.
fn log10_clip_bounds(sub: &[f32], lo_q: f32, hi_q: f32) -> (f32, f32) {
    let mut logs: Vec<f32> = sub
        .iter()
        .filter_map(|&v| {
            if v.is_finite() && v > 0.0 {
                Some(v.log10())
            } else {
                None
            }
        })
        .collect();
    if logs.is_empty() {
        return (-1.0, 1.0);
    }
    logs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f32| -> f32 {
        let n = logs.len();
        let idx = ((q.clamp(0.0, 1.0) * (n - 1) as f32).round() as usize).min(n - 1);
        logs[idx]
    };
    let lo = pick(lo_q);
    let hi = pick(hi_q);
    if (hi - lo).abs() < 1e-6 {
        // Constant data — give the colorbar *some* width.
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    }
}

/// RdBu_r diverging heatmap (n_genes × n_topics). Values are mapped via
/// log10 with 1st/99th percentile clipping so a few extreme entries
/// don't wash out the rest. Color = `rdbu_r((log10(w) − lo) / (hi − lo))`.
/// Cells with `w ≤ 0` or non-finite values are drawn at the low end of
/// the palette (deep blue), matching the "no signal" visual convention.
fn render_dict_heatmap_png(
    sub: &[f32],
    n_genes: usize,
    n_topics: usize,
    cell_w: u32,
    cell_h: u32,
    lo_log: f32,
    hi_log: f32,
) -> anyhow::Result<Vec<u8>> {
    use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

    let bar_w = n_topics as u32 * cell_w;
    let bar_h = n_genes as u32 * cell_h;
    let mut pixmap = Pixmap::new(bar_w, bar_h)
        .ok_or_else(|| anyhow::anyhow!("pixmap alloc failed ({bar_w}x{bar_h})"))?;
    pixmap.fill(tiny_skia::Color::from_rgba8(255, 255, 255, 255));

    let span = (hi_log - lo_log).max(1e-6);
    let identity = Transform::identity();
    for i in 0..n_genes {
        for j in 0..n_topics {
            let v = sub[i * n_topics + j];
            let t = if v.is_finite() && v > 0.0 {
                ((v.log10() - lo_log) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (r, g, b) = rdbu_r(t);
            let mut p = Paint::default();
            p.set_color_rgba8(r, g, b, 255);
            p.anti_alias = false;
            let x = j as f32 * cell_w as f32;
            let y = i as f32 * cell_h as f32;
            let mut pb = PathBuilder::new();
            pb.push_rect(
                tiny_skia::Rect::from_xywh(x, y, cell_w as f32, cell_h as f32)
                    .ok_or_else(|| anyhow::anyhow!("invalid rect"))?,
            );
            if let Some(path) = pb.finish() {
                pixmap.fill_path(&path, &p, FillRule::Winding, identity, None);
            }
        }
    }

    pixmap
        .encode_png()
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))
}

/// Render a vertical RdBu_r colorbar PNG with `bar_w × bar_h` pixels.
/// `bar_w` is typically 10–14 px; the gradient runs top (high) → bottom
/// (low) so it visually pairs with a y-axis scale.
fn render_colorbar_png(bar_w: u32, bar_h: u32) -> anyhow::Result<Vec<u8>> {
    use tiny_skia::Pixmap;
    let mut pixmap =
        Pixmap::new(bar_w, bar_h).ok_or_else(|| anyhow::anyhow!("colorbar pixmap alloc failed"))?;
    let data = pixmap.data_mut();
    for y in 0..bar_h {
        let t = 1.0 - (y as f32 / (bar_h - 1).max(1) as f32);
        let (r, g, b) = rdbu_r(t);
        for x in 0..bar_w {
            let off = ((y * bar_w + x) * 4) as usize;
            data[off] = r;
            data[off + 1] = g;
            data[off + 2] = b;
            data[off + 3] = 255;
        }
    }
    pixmap
        .encode_png()
        .map_err(|e| anyhow::anyhow!("colorbar PNG encode failed: {e}"))
}

// ---------------------------------------------------------------------------
// Output emission (PDF default, SVG/PNG opt-in).
// ---------------------------------------------------------------------------

fn emit_outputs(svg: &str, w: u32, h: u32, base: &str, args: &PlotTopicArgs) -> anyhow::Result<()> {
    if args.svg {
        let path = format!("{base}.svg");
        fs::write(&path, svg.as_bytes())?;
        info!("Wrote {path}");
    }
    let png_task = args.png.then(|| format!("{base}.png"));
    let pdf_task = (!args.no_pdf).then(|| format!("{base}.pdf"));

    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => plot_utils::render_png(svg, w, h, Path::new(p)).map(|()| Some(p.clone())),
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => plot_utils::render_pdf(svg, Path::new(p)).map(|()| Some(p.clone())),
            None => Ok(None),
        },
    );
    if let Some(p) = png_res? {
        info!("Wrote {p}");
    }
    if let Some(p) = pdf_res? {
        info!("Wrote {p}");
    }
    Ok(())
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}
