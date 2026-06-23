//! `faba gem-plot` — 2D UMAP layouts of the trained gem embeddings,
//! driven seamlessly off a `{prefix}.faba.json` manifest.
//!
//! Two plots are produced (either can be skipped):
//!   • **feature** — UMAP of the gene embedding β_g, k-means clustered.
//!     Each cluster is labelled with the genes nearest its centroid
//!     (its representative members). The dead-feature blob — rows that
//!     training never moved off the `N(0, σ²)` init (see
//!     `manifest::write_feature_prior_score`) — shows up as its own
//!     tight cluster with an uninformative label.
//!   • **cell** — UMAP of the cell embedding e_cell, k-means clustered.
//!     Each cluster is labelled with the genes it most upregulates,
//!     scored by `centroid_c · β_gᵀ` (the model's bilinear readout).
//!
//! Every plot is drawn with a black frame box (`SvgOpts.frame_stroke_px`)
//! so multiple panels share a common canvas and overlay cleanly, plus
//! colored per-cluster convex hulls. Layout coordinates + cluster ids are
//! also written to parquet (`*_coords.parquet`) so re-plotting is cheap.
//!
//! The UMAP SGD kernel and fuzzy-kNN graph are the shared
//! `matrix_util::umap` / `matrix_util::knn_graph` routines (also used by
//! `senna layout umap`); rendering reuses `plot_utils`.

use anyhow::{Context, Result};
use clap::Args;
use log::{info, warn};
use std::path::Path;

use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
use matrix_util::traits::IoOps;
use matrix_util::umap::Umap;

use plot_utils::hull::{convex_hull, median_xy, trim_outliers_by_median, Pt};
use plot_utils::palette::{self, Palette};
use plot_utils::rasterize::{rasterize_group_png, DataBounds, Extent, PointShape, RadiusSpec};
use plot_utils::svg_emit::{emit_svg, SvgOpts, TopicLayer};
use plot_utils::{render_pdf, render_png};

use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;

use super::manifest::{default_out, load_manifest, resolve, GemManifest};

const PT_PER_INCH: f32 = 72.0;
/// UMAP inits in `[-INIT_SCALE, INIT_SCALE]²` so the ±4 gradient clamp
/// doesn't dominate the layout scale (mirrors `senna layout umap`).
const INIT_SCALE: f32 = 10.0;

#[derive(Args, Debug)]
pub struct GemPlotArgs {
    #[arg(
        long,
        short = 'f',
        help = "gem run manifest (`{prefix}.faba.json`) from `faba gem`",
        long_help = "Parquet paths are resolved relative to the manifest's own \
                     directory, so a run directory can be moved freely. Reads \
                     `feature_embedding` (SIMBA co-embed: genes on the cell \
                     manifold) and `cell_embedding` (e_cell) — the two share a \
                     coordinate frame. Not `gene_base_embedding` (raw β_g), \
                     which is off-manifold."
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (default: alongside the manifest, reusing its prefix basename)"
    )]
    pub out: Option<Box<str>>,

    #[arg(long, default_value_t = 15, help = "k-means clusters per plot")]
    pub num_clusters: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Top genes named per cluster in the label"
    )]
    pub top_features: usize,

    #[arg(
        long = "mask-prior-fdr",
        default_value_t = 0.0,
        help = "Display-only: hide poor-quality genes from the FEATURE plot (drop genes whose \
                feature prior_pval > this — they never left the N(0,σ²) prior = dead/noise). \
                0 = show all. Outputs untouched; gem-annotate still sees every gene."
    )]
    pub mask_prior_fdr: f32,

    #[arg(long, default_value_t = 200, help = "k-means max iterations")]
    pub kmeans_max_iter: usize,

    #[arg(long, help = "Skip the feature (β_g) plot")]
    pub no_features: bool,

    #[arg(long, help = "Skip the cell (e_cell) plot")]
    pub no_cells: bool,

    // ---- layout (UMAP) ----
    #[arg(long, default_value_t = 500, help = "UMAP SGD epochs")]
    pub umap_epochs: usize,

    #[arg(long, default_value_t = 15, help = "kNN for the fuzzy graph")]
    pub umap_knn: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Negative samples per attractive step"
    )]
    pub umap_negative_rate: usize,

    #[arg(long, default_value_t = 1.0, help = "UMAP learning rate")]
    pub umap_lr: f32,

    #[arg(long, default_value_t = 1000, help = "kNN block size")]
    pub block_size: usize,

    #[arg(long, default_value_t = 42, help = "RNG seed (layout + init)")]
    pub seed: u64,

    // ---- canvas ----
    #[arg(long, default_value_t = 8.0, help = "Canvas width (inches)")]
    pub width: f32,

    #[arg(long, default_value_t = 8.0, help = "Canvas height (inches)")]
    pub height: f32,

    #[arg(long, default_value_t = 200.0, help = "Render resolution (px/inch)")]
    pub dpi: f32,

    #[arg(long, default_value_t = 2.0, help = "Point diameter (pt)")]
    pub point_size: f32,

    #[arg(long, default_value_t = 9.0, help = "Cluster-label font size (pt)")]
    pub label_font_size: f32,

    #[arg(long, default_value_t = 0.6, help = "Point opacity (0..=1)")]
    pub alpha: f32,

    #[arg(
        long,
        default_value_t = 2.0,
        help = "Black frame-box stroke (px); 0 disables"
    )]
    pub frame_stroke: f32,
}

/// Resolved pixel-space render settings, derived once from the canvas args.
struct RenderCfg {
    width_px: u32,
    height_px: u32,
    radius_px: f32,
    label_font_px: f32,
    alpha: f32,
    frame_stroke: f32,
}

pub fn run_gem_plot(args: &GemPlotArgs) -> Result<()> {
    let (manifest, dir) = load_manifest(&args.from)?;
    let out: String = args
        .out
        .as_deref()
        .map(str::to_owned)
        .unwrap_or_else(|| default_out(&dir, &manifest.prefix));
    mkdir_parent(&out)?;

    let cfg = RenderCfg {
        width_px: (args.width * args.dpi).round().max(1.0) as u32,
        height_px: (args.height * args.dpi).round().max(1.0) as u32,
        radius_px: (args.point_size * args.dpi / PT_PER_INCH / 2.0).max(0.25),
        label_font_px: args.label_font_size * args.dpi / PT_PER_INCH,
        alpha: args.alpha.clamp(0.0, 1.0),
        frame_stroke: args.frame_stroke.max(0.0),
    };

    // β_g is needed for both plots (feature plot lays it out; cell plot
    // scores genes against cell-cluster centroids), so load it once.
    let feat_path = resolve(&dir, &manifest.feature_embedding);
    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let beta = feat.mat; // [G × H]
    let gene_names = feat.rows;
    info!(
        "loaded feature embedding {} × {} ({} genes)",
        beta.nrows(),
        beta.ncols(),
        gene_names.len()
    );

    // Optional display-only filter: hide dead/noise genes (prior_pval > fdr) from the
    // FEATURE plot. Applied BEFORE the layout so the dead-feature blob can't distort
    // the UMAP. The cell plot keeps the full β (dead genes never score as top labels).
    let feat_filtered: Option<(DMatrix<f32>, Vec<Box<str>>)> = (args.mask_prior_fdr > 0.0)
        .then(|| mask_dead_features(&dir, &manifest, args.mask_prior_fdr, &beta, &gene_names))
        .transpose()?;
    let (feat_emb, feat_names): (&DMatrix<f32>, &[Box<str>]) = match &feat_filtered {
        Some((b, n)) => (b, n),
        None => (&beta, &gene_names),
    };

    if !args.no_features {
        plot_axis(
            AxisPlot {
                emb: feat_emb,
                row_names: feat_names,
                num_clusters: args.num_clusters,
                row_kind: "gene",
                out_base: &format!("{out}.gem_plot.feature"),
            },
            args,
            &cfg,
            FeatureLabels {
                gene_names: feat_names,
            },
        )?;
    }

    if !args.no_cells {
        let cell_path = resolve(&dir, &manifest.cell_embedding);
        let cell = DMatrix::<f32>::from_parquet(&cell_path)
            .with_context(|| format!("reading cell embedding {cell_path}"))?;
        let e_cell = cell.mat; // [N × H]
        let cell_names = cell.rows;
        if e_cell.nrows() == 0 {
            warn!("cell embedding is empty — skipping cell plot");
        } else {
            info!(
                "loaded cell embedding {} × {}",
                e_cell.nrows(),
                e_cell.ncols()
            );
            plot_axis(
                AxisPlot {
                    emb: &e_cell,
                    row_names: &cell_names,
                    num_clusters: args.num_clusters,
                    row_kind: "cell",
                    out_base: &format!("{out}.gem_plot.cell"),
                },
                args,
                &cfg,
                CellLabels {
                    beta: &beta,
                    gene_names: &gene_names,
                },
            )?;
        }
    }

    info!("done — gem plots under '{out}.gem_plot.*'");
    Ok(())
}

/// Display-only filter for the feature plot: return the feature embedding + names
/// restricted to INFORMED genes (`prior_pval <= fdr`), dropping the dead/noise
/// genes that never left the `N(0, σ²)` prior. `prior_pval` is column 2 of
/// `{prefix}.feature_prior_score.parquet` (`emb_sq_norm, chisq_stat, prior_pval`).
/// Genes absent from the prior-score are kept. The model outputs are untouched.
fn mask_dead_features(
    dir: &Path,
    manifest: &GemManifest,
    fdr: f32,
    beta: &DMatrix<f32>,
    gene_names: &[Box<str>],
) -> Result<(DMatrix<f32>, Vec<Box<str>>)> {
    // Shared dead-gene call (read-side of write_feature_prior_score).
    let keep_mask = super::manifest::feature_prior_keep(dir, manifest, fdr, gene_names)?;
    let keep: Vec<usize> = keep_mask
        .iter()
        .enumerate()
        .filter(|(_, &k)| k)
        .map(|(i, _)| i)
        .collect();
    info!(
        "feature plot: hid {} dead/noise genes (prior_pval > {fdr}); {} shown",
        gene_names.len() - keep.len(),
        keep.len()
    );
    let fbeta = beta.select_rows(keep.iter());
    let fnames = keep.iter().map(|&i| gene_names[i].clone()).collect();
    Ok((fbeta, fnames))
}

/// One embedding axis to plot: the embedding matrix, its row names, the target
/// k-means cluster count, a short kind tag (`"cell"` / `"gene"`) for logs + the
/// coords parquet, and the output path stem.
struct AxisPlot<'a> {
    emb: &'a DMatrix<f32>,
    row_names: &'a [Box<str>],
    num_clusters: usize,
    row_kind: &'a str,
    out_base: &'a str,
}

/// Lay out, cluster, label and render one embedding axis (features or cells).
fn plot_axis<L: ClusterLabeller>(
    axis: AxisPlot<'_>,
    args: &GemPlotArgs,
    cfg: &RenderCfg,
    labeller: L,
) -> Result<()> {
    let AxisPlot {
        emb,
        row_names,
        num_clusters,
        row_kind,
        out_base,
    } = axis;
    let n = emb.nrows();
    let k = num_clusters.clamp(1, n.max(1));

    let coords = layout(emb, args)?;
    let clusters = emb.kmeans_rows(KmeansArgs {
        num_clusters: k,
        max_iter: args.kmeans_max_iter,
    });
    let labels = labeller.labels(emb, &clusters, k, args.top_features);
    log_cluster_sizes(row_kind, &clusters, k);

    render_clusters(&coords, &clusters, k, &labels, out_base, cfg)?;
    write_coords_parquet(
        &coords,
        &clusters,
        row_names,
        row_kind,
        &format!("{out_base}_coords.parquet"),
    )?;
    Ok(())
}

//////////////////////////////
// Layout
//////////////////////////////

/// Build the fuzzy kNN graph over the embedding rows and run UMAP SGD.
/// Returns an `n × 2` coordinate matrix. The embedding is used as-is (no
/// L2 normalization) so the small-norm dead-feature blob stays compact.
fn layout(emb: &DMatrix<f32>, args: &GemPlotArgs) -> Result<DMatrix<f32>> {
    let n = emb.nrows();
    let knn = args.umap_knn.min(n.saturating_sub(1)).max(1);
    let graph = KnnGraph::from_rows(
        emb,
        KnnGraphArgs {
            knn,
            block_size: args.block_size,
            reciprocal: false,
        },
    )?;
    let fuzzy = graph.fuzzy_kernel_weights();
    let edges: Vec<(usize, usize, f32)> = graph
        .edges
        .par_iter()
        .zip(fuzzy.par_iter())
        .filter_map(|(&(i, j), &w)| (w > 0.0).then_some((i, j, w)))
        .collect();
    info!(
        "UMAP graph: {} nodes, {} edges (mean deg = {:.1}); SGD epochs={}",
        n,
        edges.len(),
        2.0 * edges.len() as f32 / n.max(1) as f32,
        args.umap_epochs,
    );

    let init = random_init_flat(n, args.seed);
    let umap = Umap {
        n_epochs: args.umap_epochs,
        negative_sample_rate: args.umap_negative_rate,
        learning_rate: args.umap_lr,
        seed: args.seed,
    };
    let flat = umap.fit(&edges, n, &init);

    let mut coords = DMatrix::<f32>::zeros(n, 2);
    for i in 0..n {
        coords[(i, 0)] = flat[i * 2];
        coords[(i, 1)] = flat[i * 2 + 1];
    }
    Ok(coords)
}

fn random_init_flat(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n * 2)
        .map(|_| (rng.random_range(0.0_f32..1.0) * 2.0 - 1.0) * INIT_SCALE)
        .collect()
}

//////////////////////////////
// Cluster labelling
//////////////////////////////

/// Strategy for turning a clustering into one display label per cluster.
trait ClusterLabeller {
    fn labels(&self, emb: &DMatrix<f32>, clusters: &[usize], k: usize, top_n: usize)
        -> Vec<String>;
}

/// Feature plot: label each cluster with the genes nearest its centroid
/// (the most representative members of the cluster).
struct FeatureLabels<'a> {
    gene_names: &'a [Box<str>],
}

impl ClusterLabeller for FeatureLabels<'_> {
    fn labels(
        &self,
        emb: &DMatrix<f32>,
        clusters: &[usize],
        k: usize,
        top_n: usize,
    ) -> Vec<String> {
        let centroids = cluster_centroids(emb, clusters, k);
        let h = emb.ncols();
        (0..k)
            .map(|c| {
                // Score each member once by distance to its centroid (nearest
                // = most representative), then partial-select the top-n.
                let scored: Vec<(usize, f32)> = (0..emb.nrows())
                    .filter(|&i| clusters[i] == c)
                    .map(|i| (i, row_dist2(emb, i, &centroids[c], h)))
                    .collect();
                let top = top_n_indices(scored, top_n, Order::Ascending);
                let names: Vec<&str> = top.iter().map(|&i| self.gene_names[i].as_ref()).collect();
                format_label(c, &names)
            })
            .collect()
    }
}

/// Cell plot: label each cluster with the genes it most upregulates,
/// scored by the model's bilinear readout `centroid_c · β_gᵀ`.
struct CellLabels<'a> {
    beta: &'a DMatrix<f32>,
    gene_names: &'a [Box<str>],
}

impl ClusterLabeller for CellLabels<'_> {
    fn labels(
        &self,
        emb: &DMatrix<f32>,
        clusters: &[usize],
        k: usize,
        top_n: usize,
    ) -> Vec<String> {
        let centroids = cluster_centroids(emb, clusters, k);
        let g = self.beta.nrows();
        let h = self.beta.ncols().min(emb.ncols());
        (0..k)
            .map(|c| {
                let cen = &centroids[c];
                // score_g = centroid_c · β_g (over shared dims), top-n highest.
                let scored: Vec<(usize, f32)> = (0..g)
                    .map(|gi| (gi, (0..h).map(|j| cen[j] * self.beta[(gi, j)]).sum()))
                    .collect();
                let top = top_n_indices(scored, top_n, Order::Descending);
                let names: Vec<&str> = top.iter().map(|&gi| self.gene_names[gi].as_ref()).collect();
                format_label(c, &names)
            })
            .collect()
    }
}

/// Per-cluster mean embedding (`k` rows of length `H`). Empty clusters
/// get a zero centroid.
fn cluster_centroids(emb: &DMatrix<f32>, clusters: &[usize], k: usize) -> Vec<Vec<f32>> {
    let h = emb.ncols();
    let mut cent = vec![vec![0.0_f64; h]; k];
    let mut cnt = vec![0_usize; k];
    for i in 0..emb.nrows() {
        let c = clusters[i];
        if c >= k {
            continue;
        }
        cnt[c] += 1;
        for j in 0..h {
            cent[c][j] += emb[(i, j)] as f64;
        }
    }
    cent.iter_mut()
        .zip(&cnt)
        .map(|(row, &n)| {
            if n > 0 {
                row.iter().map(|&v| (v / n as f64) as f32).collect()
            } else {
                vec![0.0_f32; h]
            }
        })
        .collect()
}

fn row_dist2(emb: &DMatrix<f32>, i: usize, centroid: &[f32], h: usize) -> f32 {
    (0..h)
        .map(|j| {
            let d = emb[(i, j)] - centroid[j];
            d * d
        })
        .sum()
}

fn format_label(cluster: usize, names: &[&str]) -> String {
    if names.is_empty() {
        format!("C{cluster}")
    } else {
        format!("C{cluster}: {}", names.join(", "))
    }
}

/// Sort direction for [`top_n_indices`].
#[derive(Clone, Copy)]
enum Order {
    /// Smallest keys first (e.g. nearest-to-centroid).
    Ascending,
    /// Largest keys first (e.g. highest score).
    Descending,
}

/// Indices of the `top_n` items by key, in `order`. Partial-selects in O(n)
/// rather than fully sorting, computing each key only once (the caller passes
/// `(index, key)` pairs).
fn top_n_indices(mut scored: Vec<(usize, f32)>, top_n: usize, order: Order) -> Vec<usize> {
    let cmp = move |a: &(usize, f32), b: &(usize, f32)| match order {
        Order::Ascending => a.1.total_cmp(&b.1),
        Order::Descending => b.1.total_cmp(&a.1),
    };
    let n = top_n.min(scored.len());
    if n < scored.len() {
        scored.select_nth_unstable_by(n, &cmp);
        scored.truncate(n);
    }
    scored.sort_by(cmp);
    scored.into_iter().map(|(i, _)| i).collect()
}

fn log_cluster_sizes(kind: &str, clusters: &[usize], k: usize) {
    let mut sizes = vec![0_usize; k];
    for &c in clusters {
        if c < k {
            sizes[c] += 1;
        }
    }
    info!("{kind} clusters (k={k}) sizes: {sizes:?}");
}

//////////////////////////////
// Rendering
//////////////////////////////

fn render_clusters(
    coords: &DMatrix<f32>,
    clusters: &[usize],
    k: usize,
    labels: &[String],
    out_base: &str,
    cfg: &RenderCfg,
) -> Result<()> {
    let n = coords.nrows();
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for i in 0..n {
        let (x, y) = (coords[(i, 0)], coords[(i, 1)]);
        if x.is_finite() && y.is_finite() {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
    }
    let bounds = DataBounds::from_minmax(xmin, xmax, ymin, ymax);
    let ext = Extent {
        w: cfg.width_px,
        h: cfg.height_px,
    };
    let pal = palette::resolve(&Palette::Auto, k);

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in clusters.iter().enumerate() {
        if c < k {
            groups[c].push(i);
        }
    }

    let layers: Vec<TopicLayer> = groups
        .par_iter()
        .enumerate()
        .map(|(c, idxs)| -> Result<TopicLayer> {
            let pts: Vec<Pt> = idxs
                .iter()
                .map(|&i| (coords[(i, 0)], coords[(i, 1)]))
                .collect();
            let pts_px: Vec<(f32, f32)> = pts.iter().map(|&p| bounds.to_pixel(p, ext)).collect();
            let color = palette::color(&pal, c);
            let png = rasterize_group_png(
                &pts_px,
                ext,
                RadiusSpec::Scalar(cfg.radius_px),
                color,
                cfg.alpha,
                PointShape::Circle,
            )?;
            let hull_px: Vec<Pt> = if pts.len() >= 3 {
                let trimmed = trim_outliers_by_median(&pts, 0.9);
                convex_hull(&trimmed)
                    .iter()
                    .map(|&p| bounds.to_pixel(p, ext))
                    .collect()
            } else {
                Vec::new()
            };
            let label_xy_px = bounds.to_pixel(median_xy(&pts), ext);
            Ok(TopicLayer {
                label: labels.get(c).cloned().unwrap_or_else(|| format!("C{c}")),
                png,
                hull_px,
                label_xy_px,
                color,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let svg = emit_svg(
        &layers,
        &SvgOpts {
            width_px: cfg.width_px,
            height_px: cfg.height_px,
            draw_hulls: true,
            draw_labels: true,
            label_font_size_px: cfg.label_font_px,
            hull_stroke_px: (cfg.radius_px * 0.8).max(1.0),
            hull_fill_alpha: 0.0,
            frame_stroke_px: cfg.frame_stroke,
        },
    );

    let svg_path = format!("{out_base}.svg");
    std::fs::write(&svg_path, svg.as_bytes()).with_context(|| format!("writing {svg_path}"))?;
    let png_path = format!("{out_base}.png");
    let pdf_path = format!("{out_base}.pdf");
    let (png_res, pdf_res) = rayon::join(
        || render_png(&svg, cfg.width_px, cfg.height_px, Path::new(&png_path)),
        || render_pdf(&svg, Path::new(&pdf_path)),
    );
    png_res.with_context(|| format!("rendering {png_path}"))?;
    pdf_res.with_context(|| format!("rendering {pdf_path}"))?;
    info!("wrote {svg_path}, {png_path}, {pdf_path}");
    Ok(())
}

/// `[n × 3]` parquet: `umap_1`, `umap_2`, `cluster` keyed by row name.
fn write_coords_parquet(
    coords: &DMatrix<f32>,
    clusters: &[usize],
    row_names: &[Box<str>],
    row_kind: &str,
    path: &str,
) -> Result<()> {
    let n = coords.nrows();
    let mut out = DMatrix::<f32>::zeros(n, 3);
    for i in 0..n {
        out[(i, 0)] = coords[(i, 0)];
        out[(i, 1)] = coords[(i, 1)];
        out[(i, 2)] = clusters[i] as f32;
    }
    let col_names: Vec<Box<str>> = ["umap_1", "umap_2", "cluster"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    out.to_parquet_with_names(path, (Some(row_names), Some(row_kind)), Some(&col_names))
        .with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");
    Ok(())
}
