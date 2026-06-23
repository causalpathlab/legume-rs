//! `pinto cage-annotate` — marker-set cell-type annotation by projecting cell
//! types onto the **frozen cage feature embedding**.
//!
//! A thin adapter over the shared, model-agnostic
//! [`graph_embedding_util::type_annotation::annotate_embeddings`]: it loads
//! `{prefix}.feature_embedding.parquet` (gene × D) and
//! `{prefix}.cell_embedding.parquet` (cell × D) written by `pinto cage` and
//! hands them to the shared routine. Outputs the two-layer
//! `{out}.cage_annot.{annot,community_profile,type_map,type_embedding,coarse_embedding}.parquet`
//! (per-cell fine+coarse labels with z/p, community summaries, and type anchors).

use anyhow::{Context, Result};
use clap::Args;

use graph_embedding_util::type_annotation::{
    annotate_embeddings, AnnotateProjConfig, InputEmbeddings,
};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

#[derive(Args, Debug)]
pub struct CageAnnotateArgs {
    #[arg(
        long,
        short = 'f',
        help = "cage output prefix (the `-o` value), or its `{prefix}.pinto.json`"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 'm',
        help = "Marker TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(long, short = 'o', help = "Output prefix (default: the cage prefix)")]
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
        help = "k for the shared cell kNN graph (fine-score smoothing + Leiden coarsening + UMAP layout)"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden resolution for cell clustering (higher → more, finer communities)"
    )]
    pub resolution: f64,
}

pub fn run_cage_annotate(args: &CageAnnotateArgs) -> Result<()> {
    // Accept either the bare cage prefix or its `.pinto.json` manifest path.
    let prefix = args
        .from
        .strip_suffix(".pinto.json")
        .unwrap_or(&args.from)
        .to_string();

    let feat_path = format!("{prefix}.feature_embedding.parquet");
    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell_path = format!("{prefix}.cell_embedding.parquet");
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    let out = args
        .out
        .as_deref()
        .map(str::to_owned)
        .unwrap_or_else(|| prefix.clone());
    mkdir_parent(&out)?;

    let cfg = AnnotateProjConfig {
        n_perm: args.num_perm,
        seed: args.seed,
        knn: args.knn,
        resolution: args.resolution,
        coarsen: !args.no_coarsen,
        ..AnnotateProjConfig::default()
    };
    annotate_embeddings(
        &InputEmbeddings {
            feature_emb: &feat.mat,
            gene_names: &feat.rows,
            cell_emb: &cell.mat,
            cell_names: &cell.rows,
        },
        &args.markers,
        &format!("{out}.cage_annot"),
        !args.no_idf,
        &cfg,
    )?;
    Ok(())
}
