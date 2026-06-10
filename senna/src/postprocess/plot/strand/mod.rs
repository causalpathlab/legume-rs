//! `senna plot-strand` — Watson/Crick mirrored genomic-activity
//! ideograms.
//!
//! Motivated by the CD34+ bone-marrow "silent sister" hypothesis: for
//! each cell type, draw per-chromosome gene **activity split by strand**
//! — forward/Watson genes as a filled pileup rising *upward*,
//! reverse/Crick genes *mirrored downward* around a shared horizontal
//! chromosome axis (cf. Strand-seq ideograms).
//!
//! Inputs come from a trained-then-annotated run:
//!   - gene × cell-type activity, derived from `senna annotate-by-enrichment`'s
//!     `cluster_expression.parquet` (gene × cluster) collapsed to cell
//!     types via `cluster_celltype_q.parquet` (cluster × cell-type), or
//!     any gene × group matrix passed with `--activity`.
//!   - gene coordinates + strand from a GTF/GFF (`--gtf`), matched to the
//!     activity row names with the same alias-tolerant rule the topic
//!     models use (`auxiliary_data::feature_names`).
//!
//! Output: one figure per cell type (chromosomes stacked vertically),
//! plus an optional consensus figure. We hand-build the SVG and render
//! it through the shared `plot_utils` resvg/svg2pdf path — no new
//! plotting primitives. Placement/binning live in [`place`]; SVG
//! assembly in [`render`].

mod place;
mod render;

use crate::embed_common::*;
use crate::run_manifest::{self, RunManifest};
use place::{bin_group, place_genes, robust_max, sum_grids, BinGrid};
use rayon::prelude::*;
use render::render_one;
use rustc_hash::FxHashMap;
use std::path::Path;

////////////////////////////////////////////////////////////////////////
// CLI
////////////////////////////////////////////////////////////////////////

#[derive(Args, Debug)]
pub struct PlotStrandArgs {
    #[arg(
        long,
        short = 'f',
        help = "Run manifest JSON from `senna {topic,...}` enriched by `senna annotate-by-enrichment`",
        long_help = "Fills --activity (from annotate.cluster_expression + \
                     annotate.cluster_celltype_q) and --out (from the manifest \
                     prefix). Explicit CLI flags still win. Paths inside the \
                     manifest resolve relative to its own directory."
    )]
    pub from: Option<Box<str>>,

    #[arg(
        long,
        help = "GTF/GFF with gene coordinates + strand (GENCODE-style). Required."
    )]
    pub gtf: Box<str>,

    #[arg(
        long,
        help = "Override the gene × group activity matrix (parquet, gene-name rows). \
                Default: derived gene × cell-type matrix from the manifest's annotate outputs."
    )]
    pub activity: Option<Box<str>>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (defaults to the manifest `prefix`). Writes {out}.strand/<celltype>.pdf"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Restrict to these chromosomes (comma-separated, 'chr' prefix optional). \
                Default: all autosomes + X/Y/M present in the GTF, in karyotype order."
    )]
    pub chromosomes: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 400,
        help = "Number of genomic bins across the longest chromosome (others scale by length)"
    )]
    pub bins: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Label the top-N genes (by activity) per chromosome/cell-type (0 = off)"
    )]
    pub top_genes: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = HeightScale::Sqrt,
        help = "Pileup-height scale for the binned activity (default: sqrt). Compresses spikes so low-level signal stays visible."
    )]
    pub scale: HeightScale,

    #[arg(
        long,
        default_value_t = true,
        help = "Also emit a consensus figure summing activity across all cell types",
        action = clap::ArgAction::Set
    )]
    pub consensus: bool,

    #[arg(
        long,
        default_value = "#E69F00",
        help = "Watson (forward) fill colour, drawn upward"
    )]
    pub watson_color: Box<str>,

    #[arg(
        long,
        default_value = "#0F8B8D",
        help = "Crick (reverse) fill colour, drawn downward"
    )]
    pub crick_color: Box<str>,

    #[arg(
        long,
        default_value = "#7A7A7A",
        help = "Consensus fill colour (both strands), matching the Strand-seq consensus track"
    )]
    pub consensus_color: Box<str>,

    #[arg(long, default_value_t = 7.0, help = "Figure width (inches)")]
    pub width: f32,

    #[arg(
        long,
        default_value_t = 0.30,
        help = "Per-chromosome track height (inches; split half Watson / half Crick)"
    )]
    pub track_height: f32,

    #[arg(long, default_value_t = 300, help = "Output DPI")]
    pub dpi: u32,

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

/// Up (Watson) / down (Crick) fill colours for a single figure.
#[derive(Clone, Copy)]
pub(super) struct Strands<'a> {
    pub(super) up: &'a str,
    pub(super) down: &'a str,
}

/// Non-linear remap applied to bin heights before normalization, so a
/// few high-expression spikes don't flatten everything else.
#[derive(ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub enum HeightScale {
    /// Raw activity.
    Linear,
    /// `sqrt(v)` — the default; gentle spike compression.
    #[default]
    Sqrt,
    /// `ln(1 + v)` — stronger compression.
    Log,
}

impl HeightScale {
    /// Monotonic, maps 0 → 0; applied to both value and scale so the
    /// normalized fraction stays in `[0, 1]`.
    pub(super) fn apply(self, v: f32) -> f32 {
        let v = v.max(0.0);
        match self {
            HeightScale::Linear => v,
            HeightScale::Sqrt => v.sqrt(),
            HeightScale::Log => v.ln_1p(),
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Entry point
////////////////////////////////////////////////////////////////////////

pub fn fit_plot_strand(args: &PlotStrandArgs) -> anyhow::Result<()> {
    // ----- Resolve inputs (manifest defaults, CLI wins) -----
    let (activity_path, out_prefix) = resolve_inputs(args)?;

    // ----- Load gene × group activity -----
    let Activity {
        mat,
        gene_names,
        group_names,
    } = load_activity(&activity_path)?;
    info!(
        "Activity matrix: {} genes × {} groups",
        gene_names.len(),
        group_names.len()
    );

    // ----- Attach genomic coordinates + strand (alias-tolerant) -----
    let placements = place_genes(&gene_names, &args.gtf, args)?;
    anyhow::ensure!(
        !placements.chromosomes.is_empty(),
        "no genes matched the GTF on the requested chromosomes"
    );
    info!(
        "Placed {} genes across {} chromosomes",
        placements.placed.len(),
        placements.chromosomes.len()
    );

    // ----- Bin every group, then a global robust scale -----
    let grids: Vec<BinGrid> = (0..group_names.len())
        .into_par_iter()
        .map(|c| bin_group(&mat, c, &placements))
        .collect();

    let consensus_grid = args.consensus.then(|| sum_grids(&grids, &placements));

    let robust_max = robust_max(
        grids
            .iter()
            .chain(consensus_grid.iter())
            .flat_map(|g| g.iter_values()),
    );
    anyhow::ensure!(
        robust_max > 0.0,
        "all activity values are zero/negative — nothing to draw"
    );

    // ----- Render one figure per group (+ consensus) -----
    let out_dir = format!("{out_prefix}.strand");
    std::fs::create_dir_all(&out_dir)?;

    let mut jobs: Vec<(String, &BinGrid, Strands)> = group_names
        .iter()
        .zip(grids.iter())
        .map(|(name, grid)| {
            (
                name.to_string(),
                grid,
                Strands {
                    up: args.watson_color.as_ref(),
                    down: args.crick_color.as_ref(),
                },
            )
        })
        .collect();
    if let Some(grid) = consensus_grid.as_ref() {
        jobs.push((
            "_consensus".to_string(),
            grid,
            Strands {
                up: args.consensus_color.as_ref(),
                down: args.consensus_color.as_ref(),
            },
        ));
    }

    let written: usize = jobs
        .par_iter()
        .map(|(name, grid, strands)| {
            render_one(
                name,
                grid,
                *strands,
                robust_max,
                &placements,
                &out_dir,
                args,
            )
        })
        .collect::<anyhow::Result<Vec<usize>>>()?
        .iter()
        .sum();
    info!("Wrote {written} files under {out_dir}/");

    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Input resolution
////////////////////////////////////////////////////////////////////////

/// Returns `(activity_parquet_path, out_prefix)`.
fn resolve_inputs(args: &PlotStrandArgs) -> anyhow::Result<(String, String)> {
    // No --from: --activity and --out are both required.
    let Some(from) = args.from.as_deref() else {
        let activity = args.activity.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--activity PATH is required when --from is not given")
        })?;
        let out = args
            .out
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--out PREFIX is required when --from is not given"))?;
        return Ok((activity.to_string(), out.to_string()));
    };

    let (m, dir) = RunManifest::load(Path::new(from))?;
    info!("Loaded run manifest {from} (kind: {})", m.kind);
    let resolve = |s: &str| {
        run_manifest::resolve(&dir, s)
            .to_string_lossy()
            .into_owned()
    };

    // Out prefix first (CLI wins) — the derived activity is written next
    // to it, not next to `m.prefix`, which may be an absolute path from
    // the machine the run was trained on.
    let out = args
        .out
        .as_deref()
        .map(String::from)
        .unwrap_or_else(|| m.prefix.clone());

    // --activity wins; else derive a gene × cell-type matrix from annotate.
    let activity = if let Some(p) = args.activity.as_deref() {
        p.to_string()
    } else {
        let cluster_expr = m.annotate.cluster_expression.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "manifest {from} has no annotate.cluster_expression; run `senna annotate-by-enrichment` \
                 first, or pass --activity with a gene × group parquet"
            )
        })?;
        let q = m.annotate.cluster_celltype_q.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "manifest {from} has no annotate.cluster_celltype_q; run `senna annotate-by-enrichment` first"
            )
        })?;
        let derived = format!("{out}.celltype_expression.parquet");
        mkdir_parent(&derived)?;
        derive_celltype_activity(&resolve(cluster_expr), &resolve(q), &derived)?;
        derived
    };

    Ok((activity, out))
}

/// Collapse `cluster_expression` (gene × cluster) to gene × cell-type by
/// assigning each cluster to its argmax cell type in `cluster_celltype_q`
/// (cluster × cell-type) and averaging the clusters of each cell type.
fn derive_celltype_activity(
    cluster_expr_path: &str,
    q_path: &str,
    out_path: &str,
) -> anyhow::Result<()> {
    let MatWithNames {
        rows: genes,
        cols: clusters_e,
        mat: profile_gk,
    } = Mat::from_parquet(cluster_expr_path)?;
    let MatWithNames {
        rows: clusters_q,
        cols: celltypes,
        mat: q_kc,
    } = Mat::from_parquet(q_path)?;

    // cluster label -> argmax cell-type index
    let mut cluster_to_ct: FxHashMap<&str, usize> = FxHashMap::default();
    for (r, cl) in clusters_q.iter().enumerate() {
        let mut best_j = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for j in 0..q_kc.ncols() {
            let v = q_kc[(r, j)];
            if v > best_v {
                best_v = v;
                best_j = j;
            }
        }
        cluster_to_ct.insert(cl.as_ref(), best_j);
    }

    let n_ct = celltypes.len();
    let mut out = Mat::zeros(genes.len(), n_ct);
    let mut counts = vec![0u32; n_ct];
    for (j, cl) in clusters_e.iter().enumerate() {
        let Some(&ct) = cluster_to_ct.get(cl.as_ref()) else {
            log::warn!(
                "cluster {cl} in cluster_expression has no row in cluster_celltype_q — skipped"
            );
            continue;
        };
        counts[ct] += 1;
        for g in 0..genes.len() {
            out[(g, ct)] += profile_gk[(g, j)];
        }
    }
    // Mean over the clusters assigned to each cell type.
    for ct in 0..n_ct {
        if counts[ct] > 1 {
            let inv = 1.0 / counts[ct] as f32;
            for g in 0..genes.len() {
                out[(g, ct)] *= inv;
            }
        }
    }
    // Keep only cell types that received at least one cluster.
    let keep: Vec<usize> = (0..n_ct).filter(|&ct| counts[ct] > 0).collect();
    let kept_names: Vec<Box<str>> = keep.iter().map(|&ct| celltypes[ct].clone()).collect();
    let mut kept = Mat::zeros(genes.len(), keep.len());
    for (newc, &ct) in keep.iter().enumerate() {
        for g in 0..genes.len() {
            kept[(g, newc)] = out[(g, ct)];
        }
    }
    kept.to_parquet_with_names(out_path, (Some(&genes), Some("gene")), Some(&kept_names))?;
    info!(
        "Derived gene × cell-type activity ({} cell types) → {out_path}",
        kept_names.len()
    );
    Ok(())
}

struct Activity {
    mat: Mat,
    gene_names: Vec<Box<str>>,
    group_names: Vec<Box<str>>,
}

fn load_activity(path: &str) -> anyhow::Result<Activity> {
    let MatWithNames { rows, cols, mat } = Mat::from_parquet(path)?;
    anyhow::ensure!(mat.ncols() >= 1, "{path}: activity matrix has no columns");
    anyhow::ensure!(
        rows.len() == mat.nrows(),
        "{path}: row-name count {} != matrix rows {}",
        rows.len(),
        mat.nrows()
    );
    Ok(Activity {
        mat,
        gene_names: rows,
        group_names: cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_util::traits::IoOps;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("strand_{}_{}", std::process::id(), name))
    }

    #[test]
    fn derive_celltype_collapses_clusters_by_argmax_mean() {
        // 2 genes × 3 clusters; clusters K0,K1 → celltype A, K2 → B.
        let genes: Vec<Box<str>> = vec!["G0".into(), "G1".into()];
        let clusters: Vec<Box<str>> = vec!["K0".into(), "K1".into(), "K2".into()];
        let mut expr = Mat::zeros(2, 3);
        expr[(0, 0)] = 2.0;
        expr[(0, 1)] = 4.0; // gene0 in A clusters → mean 3
        expr[(0, 2)] = 10.0; // gene0 in B → 10
        expr[(1, 0)] = 1.0;
        expr[(1, 1)] = 1.0; // gene1 in A → mean 1
        expr[(1, 2)] = 5.0; // gene1 in B → 5
        let expr_path = tmp("expr.parquet");
        expr.to_parquet_with_names(
            expr_path.to_str().unwrap(),
            (Some(&genes), Some("gene")),
            Some(&clusters),
        )
        .unwrap();

        let celltypes: Vec<Box<str>> = vec!["A".into(), "B".into()];
        let mut q = Mat::zeros(3, 2);
        q[(0, 0)] = 0.9; // K0 → A
        q[(1, 0)] = 0.8; // K1 → A
        q[(2, 1)] = 0.7; // K2 → B
        let q_path = tmp("q.parquet");
        q.to_parquet_with_names(
            q_path.to_str().unwrap(),
            (Some(&clusters), Some("cluster")),
            Some(&celltypes),
        )
        .unwrap();

        let out_path = tmp("celltype.parquet");
        derive_celltype_activity(
            expr_path.to_str().unwrap(),
            q_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        )
        .unwrap();

        let MatWithNames { cols, mat, .. } = Mat::from_parquet(out_path.to_str().unwrap()).unwrap();
        for f in [&expr_path, &q_path, &out_path] {
            std::fs::remove_file(f).ok();
        }
        assert_eq!(cols.len(), 2, "two cell types");
        let a = cols.iter().position(|c| &**c == "A").unwrap();
        let b = cols.iter().position(|c| &**c == "B").unwrap();
        assert!((mat[(0, a)] - 3.0).abs() < 1e-5, "gene0 A = mean(2,4) = 3");
        assert!((mat[(0, b)] - 10.0).abs() < 1e-5, "gene0 B = 10");
        assert!((mat[(1, a)] - 1.0).abs() < 1e-5, "gene1 A = 1");
    }
}
