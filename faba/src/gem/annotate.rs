//! `faba gem-annotate` — light marker-set cell-type annotation by projecting
//! cell types onto the **frozen gem feature embedding**.
//!
//! A thin adapter over the shared, model-agnostic
//! [`graph_embedding_util::type_annotation::annotate_embeddings`]: it loads
//! β_g (`feature_embedding`) and e_cell (`cell_embedding`) from a
//! `{prefix}.faba.json` manifest and hands them to the shared routine, which
//! embeds each cell type as the L2-normalized centroid of its marker feature
//! embeddings (the space the cells live in), clusters the cells, and emits a
//! two-layer (fine + coarse) annotation. Outputs:
//!
//! - `{out}.gem_annot.annot.parquet` — per cell: community, coarse + fine
//!   label, score (z), and p-value (`pnorm(-z)`) for each layer
//! - `{out}.gem_annot.membership.tsv` — `cell<TAB>coarse_label` (no header);
//!   feeds `faba gem-summary` and `data-beans stat -s row -g` to group any
//!   count matrix by cell type
//! - `{out}.gem_annot.community_profile.parquet` — one row per community
//! - `{out}.gem_annot.type_map.parquet` — fine → coarse merge record
//! - `{out}.gem_annot.{type,coarse}_embedding.parquet` — `[· × H]` anchors
//!   (drop-in overlay for `faba gem-plot`)

use anyhow::{Context, Result};
use clap::Args;

use graph_embedding_util::type_annotation::{annotate_embeddings, AnnotateProjConfig};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use super::manifest::{default_out, load_manifest, resolve};

#[derive(Args, Debug)]
pub struct GemAnnotateArgs {
    #[arg(
        long,
        short = 'f',
        help = "gem run manifest (`{prefix}.faba.json`) from `faba gem`"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 'm',
        help = "Marker TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (default: alongside the manifest)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 200,
        help = "Permutation draws per type for the null (0 = skip z-scores)"
    )]
    pub num_perm: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed (permutation null + clustering)"
    )]
    pub seed: u64,

    #[arg(
        long,
        help = "Disable IDF down-weighting of markers shared across many types"
    )]
    pub no_idf: bool,

    #[arg(
        long = "no-coarsen",
        help = "Disable cell-grounded coarsening (emit only the fine layer mirrored as coarse)"
    )]
    pub no_coarsen: bool,

    #[arg(
        long,
        default_value_t = 30,
        help = "k for the cell kNN graph used by the coarsening clusterer"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden resolution for cell clustering (higher → more, finer communities)"
    )]
    pub resolution: f64,
}

pub fn run_gem_annotate(args: &GemAnnotateArgs) -> Result<()> {
    let (manifest, dir) = load_manifest(&args.from)?;
    let out: String = args
        .out
        .as_deref()
        .map(str::to_owned)
        .unwrap_or_else(|| default_out(&dir, &manifest.prefix));
    mkdir_parent(&out)?;

    let feat_path = resolve(&dir, &manifest.feature_embedding);
    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell_path = resolve(&dir, &manifest.cell_embedding);
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    let cfg = AnnotateProjConfig {
        n_perm: args.num_perm,
        seed: args.seed,
        knn: args.knn,
        resolution: args.resolution,
        coarsen: !args.no_coarsen,
    };
    annotate_embeddings(
        &feat.mat,
        &feat.rows,
        &cell.mat,
        &cell.rows,
        &args.markers,
        &format!("{out}.gem_annot"),
        !args.no_idf,
        &cfg,
    )?;
    Ok(())
}
