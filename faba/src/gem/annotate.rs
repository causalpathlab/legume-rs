//! `faba gem-annotate` — **firm** marker-set cell-type annotation by projection
//! onto the frozen gem feature embedding.
//!
//! A thin adapter over the shared
//! [`graph_embedding_util::type_annotation::annotate_embeddings_ora`]: it loads
//! the SIMBA co-embed (`feature_embedding`, genes on the cell manifold) and
//! `e_cell` (`cell_embedding`) from a `{prefix}.faba.json` manifest and hands
//! them to the firm term-ORA routine, which
//!
//! 1. builds each cell type's **un-normalized IDF-weighted centroid** of its
//!    marker feature embeddings (a prototype in the embedding space);
//! 2. hard-assigns every cell to its **nearest centroid** (Euclidean) and QC-
//!    prunes distance-outlier assignments to `unassigned`;
//! 3. clusters the cells (Leiden on the cosine cell kNN graph); and
//! 4. tests, per (cluster, term), the **over-representation** of that term among
//!    the cluster's cells with the hypergeometric null, **calibrated by
//!    permuting the per-cell labels**; each cluster is called its top
//!    FDR-significant term and its cells inherit that label.
//!
//! Optionally (`--obo`+`--label-cl`) it folds the cluster × term matrix through
//! the shared TreeBH ontology core for multi-resolution Cell-Ontology calling —
//! the same engine `senna annotate` uses.
//!
//! Outputs (under `{out}.gem_annot`):
//! - `.annot.parquet` — per cell: `community`, firm `coarse_label` (+ `coarse_p`,
//!   `coarse_q`), soft per-cell `fine_label` + `fine_distance`, `is_outlier`
//! - `.membership.tsv` — `cell<TAB>coarse_label` (feeds `faba gem-summary` /
//!   `data-beans stat -s row -g`)
//! - `.argmax.tsv` — `cell<TAB>cell_type<TAB>probability`
//! - `.cluster_term_{p,q,Q}.parquet` — cluster × term permutation p / BH q /
//!   FDR-sparse softmax Q
//! - `.null_calibration.tsv` — permutation-null calibration diagnostics
//! - `.marker_embedding.parquet` — matched marker genes on the cell manifold
//! - `.ontology_assignment.tsv` + `.ontology_node_mass.parquet` (with `--obo`)

use anyhow::{Context, Result};
use clap::Args;

use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, InputEmbeddings, TermOraConfig,
};
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
        default_value_t = 30,
        help = "k for the cosine cell kNN graph fed to Leiden clustering"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden resolution for cell clustering (higher → more, finer clusters)"
    )]
    pub resolution: f64,

    #[arg(
        long,
        default_value_t = 500,
        help = "Permutation draws calibrating the over-representation null (0 = analytic p only)"
    )]
    pub num_perm: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed (clustering + permutation null)"
    )]
    pub seed: u64,

    #[arg(
        long,
        help = "Disable IDF down-weighting of markers shared across many types"
    )]
    pub no_idf: bool,

    #[arg(
        long = "no-assign-qc",
        help = "Keep every cell→term assignment (skip the distance-outlier prune)"
    )]
    pub no_assign_qc: bool,

    #[arg(
        long = "assign-mad",
        default_value_t = 2.5,
        help = "Outlier gate: prune a cell whose distance to its centroid exceeds median + k·MAD"
    )]
    pub assign_mad: f64,

    #[arg(
        long = "fdr-alpha",
        default_value_t = 0.1,
        help = "FDR α for the per-cluster term call + Q sparsity (BH on the permutation p)"
    )]
    pub fdr_alpha: f32,

    #[arg(
        long = "q-temperature",
        default_value_t = 1.0,
        help = "Softmax temperature when row-normalizing Q over significant terms"
    )]
    pub q_temperature: f32,

    // ---- optional ontology (TreeBH) layer ----
    #[arg(
        long = "obo",
        help = "Cell Ontology .obo (e.g. cl-basic.obo). With --label-cl, runs TreeBH ontology \
                calling on the cluster × term matrix → {out}.ontology_assignment.tsv"
    )]
    pub obo: Option<Box<str>>,

    #[arg(
        long = "label-cl",
        help = "Curated `label<TAB>CL:id` map, one row per marker celltype. Required with --obo"
    )]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long = "ontology-fdr-q",
        default_value_t = 0.1,
        help = "Ontology TreeBH per-level FDR target (lower → descends less, abstains more)"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long = "ontology-by",
        help = "Ontology: Benjamini–Yekutieli within families (any dependence; more conservative)"
    )]
    pub ontology_by: bool,
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
        &format!("{out}.gem_annot"),
        !args.no_idf,
        &cfg,
    )?;
    Ok(())
}
