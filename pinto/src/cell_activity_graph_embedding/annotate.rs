//! `pinto cage-annotate` — marker-set cell-type annotation of a `pinto cage`
//! run.
//!
//! A thin pinto front-end over the shared, model-agnostic **term-ORA** core
//! [`graph_embedding_util::type_annotation::annotate_embeddings_ora`] (Euclidean
//! nearest-centroid assignment → distance-outlier QC → Leiden clustering →
//! cluster×term hypergeometric over-representation, permutation-calibrated →
//! optional TreeBH Cell-Ontology calling). It is the embedding-grounded twin of
//! `senna annotate-by-projection` and `faba annotate`, reading cage's parquet
//! outputs by prefix (pinto has no run manifest).
//!
//! Loads `{prefix}.feature_embedding.parquet` (gene × D) and
//! `{prefix}.cell_embedding.parquet` (cell × D) written by `pinto cage`, then
//! writes the shared per-cell contract at `{out}.cage_annot.*`
//! (`annot.parquet`, `membership.tsv`, `argmax.tsv`, the cluster × term
//! `p`/`q`/`Q` matrices, and — with `--obo`/`--label-cl` — the TreeBH ontology
//! assignment), directly comparable to the senna/faba passes.

use anyhow::{Context, Result};
use clap::Args;

use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, InputEmbeddings, TermOraConfig,
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
        default_value_t = 30,
        help = "k for the cosine cell kNN graph fed to Leiden clustering"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution (higher → more, finer clusters)"
    )]
    pub resolution: f64,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed (clustering + permutation null)"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 500,
        help = "Permutation draws calibrating the over-representation statistic"
    )]
    pub num_perm: usize,

    #[arg(
        long = "no-assign-qc",
        help = "Disable pruning of high-distance cell→term assignments"
    )]
    pub no_assign_qc: bool,

    #[arg(
        long,
        default_value_t = 2.5,
        help = "MAD multiplier for the assignment-distance outlier gate"
    )]
    pub assign_mad: f64,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "FDR α for the cluster call + Q sparsity (BH on the permutation p)"
    )]
    pub fdr_alpha: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature for the row-normalized Q over significant terms"
    )]
    pub q_temperature: f32,

    #[arg(
        long,
        help = "Disable IDF down-weighting of markers shared across many types"
    )]
    pub no_idf: bool,

    #[arg(
        long,
        help = "Cell Ontology OBO path — runs the TreeBH ontology layer (needs --label-cl)"
    )]
    pub obo: Option<Box<str>>,

    #[arg(long, help = "Curated `label<TAB>CL:id` map (paired with --obo)")]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "TreeBH per-level selective-FDR target"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long,
        help = "Benjamini–Yekutieli within ontology families (any dependence)"
    )]
    pub ontology_by: bool,
}

pub fn run_cage_annotate(args: &CageAnnotateArgs) -> Result<()> {
    // Accept either the bare cage prefix or its `.pinto.json` manifest path.
    let prefix = args.from.strip_suffix(".pinto.json").unwrap_or(&args.from);

    let feat_path = format!("{prefix}.feature_embedding.parquet");
    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell_path = format!("{prefix}.cell_embedding.parquet");
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;

    let cfg = TermOraConfig {
        knn: args.knn,
        resolution: args.resolution,
        seed: args.seed,
        n_perm: args.num_perm,
        assign_qc: !args.no_assign_qc,
        assign_mad: args.assign_mad,
        fdr_alpha: args.fdr_alpha,
        q_temperature: args.q_temperature,
        obo: args.obo.as_deref().map(str::to_owned),
        label_cl: args.label_cl.as_deref().map(str::to_owned),
        ontology_fdr_q: args.ontology_fdr_q,
        ontology_by: args.ontology_by,
    };

    annotate_embeddings_ora(
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
