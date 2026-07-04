//! `pinto annotate` — marker-set cell-type annotation by projection, for **any**
//! pinto embedding run (not cage-specific).
//!
//! A thin pinto front-end over the shared, model-agnostic **term-ORA** core
//! [`graph_embedding_util::type_annotation::annotate_embeddings_ora`] (Euclidean
//! nearest-centroid assignment → distance-outlier QC → Leiden clustering →
//! cluster×term hypergeometric over-representation, permutation-calibrated →
//! optional TreeBH Cell-Ontology calling). The embedding-grounded twin of
//! `senna annotate-by-projection` and `faba annotate`.
//!
//! Input is a co-embedded (gene, cell) pair in one inner-product space:
//! `{prefix}.feature_embedding.parquet` (gene × D) + `{prefix}.cell_embedding.parquet`
//! (cell × D). `pinto cage` writes these directly; `pinto lc-etm` writes them via
//! its SIMBA co-embedding of the topic result (`Z = propensity·α`, genes on the
//! cell manifold) — so both, and any future embedding output, annotate the same
//! way. Pass a shared `--from` prefix, or point `--feature-embedding` /
//! `--cell-embedding` at explicit parquet paths.
//!
//! Writes the shared per-cell contract at `{out}.annot.*` (`annot.parquet`,
//! `membership.tsv`, `argmax.tsv`, the cluster × term `p`/`q`/`Q` matrices, and —
//! with `--obo`/`--label-cl` — the TreeBH ontology assignment), directly
//! comparable to the senna/faba passes.

use anyhow::{Context, Result};
use clap::Args;

use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, InputEmbeddings, TermOraConfig,
};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

#[derive(Args, Debug)]
pub struct AnnotateArgs {
    #[arg(
        long,
        short = 'f',
        help = "Embedding-run prefix (cage / lc-etm `-o` value), or its `{prefix}.pinto.json`",
        long_help = "Shared output prefix of a pinto embedding run. Reads\n\
                     `{prefix}.feature_embedding.parquet` + `{prefix}.cell_embedding.parquet`.\n\
                     A `{prefix}.pinto.json` path is also accepted (suffix stripped).\n\
                     Override either side explicitly with --feature-embedding /\n\
                     --cell-embedding (both required together when --from is omitted)."
    )]
    pub from: Option<Box<str>>,

    #[arg(
        long,
        help = "Explicit gene × D feature-embedding parquet (overrides `{from}.feature_embedding.parquet`)"
    )]
    pub feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        help = "Explicit cell × D cell-embedding parquet (overrides `{from}.cell_embedding.parquet`)"
    )]
    pub cell_embedding: Option<Box<str>>,

    #[arg(
        long,
        short = 'm',
        help = "Marker TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (default: the `--from` prefix)"
    )]
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

pub fn run_annotate(args: &AnnotateArgs) -> Result<()> {
    // Accept either a bare prefix (`{prefix}.pinto.json` suffix tolerated) or
    // explicit per-side parquet paths.
    let prefix = args
        .from
        .as_deref()
        .map(|f| f.strip_suffix(".pinto.json").unwrap_or(f));

    let feat_path = args
        .feature_embedding
        .as_deref()
        .map(str::to_string)
        .or_else(|| prefix.map(|pre| format!("{pre}.feature_embedding.parquet")))
        .context("provide --from or --feature-embedding")?;
    let cell_path = args
        .cell_embedding
        .as_deref()
        .map(str::to_string)
        .or_else(|| prefix.map(|pre| format!("{pre}.cell_embedding.parquet")))
        .context("provide --from or --cell-embedding")?;

    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    // Output prefix defaults to `--from`; required explicitly when only bare
    // `--feature-embedding`/`--cell-embedding` paths were given.
    let out = args
        .out
        .as_deref()
        .or(prefix)
        .map(str::to_string)
        .context("provide --out (or --from)")?;
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
        &format!("{out}.annot"),
        !args.no_idf,
        &cfg,
    )?;
    Ok(())
}
