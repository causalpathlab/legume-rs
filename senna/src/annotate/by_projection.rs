//! `senna annotate-by-projection` — light marker-set cell-type annotation by
//! projecting cell types onto the **frozen feature embedding** of an
//! embedding run (bge / fne).
//!
//! A thin adapter over the shared, model-agnostic
//! [`graph_embedding_util::type_annotation::annotate_embeddings`]: it reads a
//! `run.senna.json` manifest, loads `outputs.feature_embedding` (gene × H)
//! and `outputs.latent` (cell × H — for bge/fne the latent *is* the cell
//! embedding), and hands them to the shared routine. Outputs the two-layer
//! `{out}.{kind}_annot.{annot,community_profile,type_map,type_embedding,coarse_embedding}.parquet`
//! plus `{out}.{kind}_annot.{membership,argmax}.tsv`, and points
//! `manifest.annotate.argmax` at the argmax TSV so `senna plot` auto-colours by it.
//!
//! This is the *light, per-cell, clustering-free* complement to
//! [`super::by_enrichment::run`] (cluster-level marker enrichment with FDR). It is
//! restricted to embedding kinds: a topic run's `latent` is in topic space,
//! not the H-space its `feature_embedding` lives in.

use anyhow::{Context, Result};
use clap::Args;
use std::path::Path;

use graph_embedding_util::type_annotation::{
    annotate_embeddings, AnnotateProjConfig, InputEmbeddings, ANNOT_OUTPUT_SUFFIXES,
};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use super::finalize::{clean_outputs, finalize_annotation, AnnotationArtifacts};
use super::ontology::{annotate_ontology_core, OntologyParams, OntologyScore};
use crate::embed_common::{axis_id_names, Mat};
use crate::run_manifest::{derive_out_prefix, resolve, RunKind, RunManifest};

#[derive(Args, Debug)]
pub struct AnnotateProjectArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest from `senna bge` / `senna fne` / `senna resolve-embedding-space`"
    )]
    pub from: Box<str>,

    #[arg(
        short = 'm',
        long = "markers",
        required = true,
        help = "Marker TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix. Defaults to `--from` with `.senna.json`/`.json` stripped"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long = "num-perm",
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
        long = "no-idf",
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

    #[arg(
        long = "no-smooth-fine",
        help = "Disable kNN-graph smoothing of the per-cell fine scores before argmax"
    )]
    pub no_smooth_fine: bool,

    #[arg(
        long = "fine-fdr",
        default_value_t = 1.0,
        help = "Opt-in: mark a fine call 'unassigned' when its best type has BH q ≥ this (1.0 = off; needs --num-perm > 0)"
    )]
    pub fine_fdr: f32,

    #[arg(
        long = "min-margin",
        default_value_t = 0.0,
        help = "Opt-in: mark a fine call 'unassigned' when its top1−top2 z-margin < this (0 = off; needs --num-perm > 0)"
    )]
    pub min_margin: f32,

    #[arg(
        long = "no-clean",
        help = "Keep existing {out}.{kind}_annot.* outputs (default: erase them first for a fresh re-run)"
    )]
    pub no_clean: bool,

    // ----- optional ontology annotation (TreeBH) over the projection z -----
    #[arg(
        long = "obo",
        help = "Cell Ontology .obo. Given WITH --label-cl, runs TreeBH on the community-pooled \
                projection z → {out}.{kind}_annot.ontology_assignment.tsv (needs --num-perm > 0)"
    )]
    pub obo: Option<Box<str>>,

    #[arg(
        long = "label-cl",
        help = "Curated `label<TAB>CL:id` map, one row per marker celltype. Required together with --obo"
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
        default_value_t = false,
        help = "Ontology: Benjamini–Yekutieli within families (any dependence; more conservative)"
    )]
    pub ontology_by: bool,
}

pub fn run(args: &AnnotateProjectArgs) -> Result<()> {
    let (mut manifest, dir) = RunManifest::load(Path::new(args.from.as_ref()))?;
    anyhow::ensure!(
        matches!(
            manifest.kind,
            RunKind::Bge | RunKind::Fne | RunKind::ResolveEmbeddingSpace
        ),
        "annotate-by-projection needs an embedding run (bge / fne / resolve-embedding-space) whose \
         `latent` is an H-space cell embedding; got kind={}. For topic runs use \
         `senna annotate-by-enrichment` (cluster-level enrichment), or first run \
         `senna resolve-embedding-space` to lift the topic θ into an embedding.",
        manifest.kind
    );

    // Feature embedding ρ. `bge --resolve-etm` records it explicitly; a plain
    // `bge`/`fne` run (no ETM) writes it as `dictionary` (there `dictionary` IS
    // E_feat, not a topic simplex), so fall back to that. ETM runs always set
    // `feature_embedding`, so the fallback never mis-picks a β simplex.
    let feat_rel = manifest
        .outputs
        .feature_embedding
        .as_deref()
        .or(manifest.outputs.dictionary.as_deref())
        .context("manifest has neither outputs.feature_embedding nor outputs.dictionary")?;
    // Prefer the explicit H-space cell embedding when the run records one
    // (`bge --resolve-etm`, `resolve-embedding-space`): there `latent` is the
    // topic θ, NOT the embedding ρ lives in. Plain bge/fne set no
    // `cell_embedding`, and their `latent` IS the cell embedding — fall back.
    let cell_rel = manifest
        .outputs
        .cell_embedding
        .as_deref()
        .or(manifest.outputs.latent.as_deref())
        .context(
            "manifest has neither outputs.cell_embedding nor outputs.latent (cell embedding). \
             For topic-space annotation use `senna annotate-by-topic`.",
        )?;

    let feat_path = resolve(&dir, feat_rel).to_string_lossy().into_owned();
    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell_path = resolve(&dir, cell_rel).to_string_lossy().into_owned();
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => derive_out_prefix(&args.from),
    };
    mkdir_parent(&out)?;
    anyhow::ensure!(
        args.obo.is_some() == args.label_cl.is_some(),
        "--obo and --label-cl must be given together for ontology annotation"
    );
    if args.obo.is_some() {
        anyhow::ensure!(
            args.num_perm > 0,
            "ontology annotation needs --num-perm > 0 (projection scores must be z-scores, not cosine)"
        );
    }

    let cfg = AnnotateProjConfig {
        n_perm: args.num_perm,
        seed: args.seed,
        knn: args.knn,
        resolution: args.resolution,
        coarsen: !args.no_coarsen,
        smooth_fine: !args.no_smooth_fine,
        fine_fdr: args.fine_fdr,
        min_margin: args.min_margin,
        ..AnnotateProjConfig::default()
    };
    let annot_prefix = format!("{out}.{}_annot", manifest.kind);
    if !args.no_clean {
        clean_outputs(&annot_prefix, ANNOT_OUTPUT_SUFFIXES);
    }
    let res = annotate_embeddings(
        &InputEmbeddings {
            feature_emb: &feat.mat,
            gene_names: &feat.rows,
            cell_emb: &cell.mat,
            cell_names: &cell.rows,
        },
        &args.markers,
        &annot_prefix,
        !args.no_idf,
        &cfg,
    )?;

    // Optional ontology annotation: Stouffer-pool the per-cell projection z into
    // a community × celltype z-matrix and feed the shared TreeBH core (same
    // engine as annotate-by-enrichment). NON-FATAL: projection outputs are
    // already written, so a bad map/OBO is logged, not propagated.
    let mut ontology_assign: Option<String> = None;
    let mut ontology_mass: Option<String> = None;
    if let (Some(obo), Some(label_cl)) = (args.obo.as_deref(), args.label_cl.as_deref()) {
        // Community × celltype signal = `res.enrich`: the column-centered mean
        // per-cell projection z within each community — the same discriminative
        // statistic projection uses to NAME communities. Centering removes the
        // common-mode "every community scores moderately for T" baseline, so what
        // remains is each community's enrichment RELATIVE to the others — a
        // graded z fed straight to the ontology walk (Φ(−z)). (Naive pooling of
        // per-cell z fails both ways: Stouffer Σz/√n over-inflates by √n on
        // correlated cells; the plain mean washes the signal out.)
        let (nc, ct) = (res.n_coarse, res.n_types);
        let mut score = Mat::zeros(nc, ct);
        for comm in 0..nc {
            for c in 0..ct {
                score[(comm, c)] = res.enrich[comm * ct + c];
            }
        }
        // No posterior Q here — the core derives the node-mass from `score`.
        let comm_names = axis_id_names("comm", nc);
        match annotate_ontology_core(
            &OntologyParams {
                out: &annot_prefix,
                label_cl,
                obo,
                fdr_q: args.ontology_fdr_q,
                by: args.ontology_by,
            },
            OntologyScore::Z(&score),
            None,
            &comm_names,
            &res.type_names,
        ) {
            Ok((a, m)) => {
                ontology_assign = Some(a);
                ontology_mass = Some(m);
            }
            Err(e) => {
                log::error!("ontology annotation failed ({e}); projection outputs are intact")
            }
        }
    }

    // Wire the per-cell labels into the manifest so `senna plot` colours by
    // them automatically. `annotate_embeddings` wrote `{prefix}.argmax.tsv`
    // (+ `membership.tsv`) via the shared writer, so the I/O contract matches
    // `annotate-by-enrichment`; the manifest wiring is shared too.
    let argmax_abs = format!("{annot_prefix}.argmax.tsv");
    finalize_annotation(
        &mut manifest,
        Path::new(args.from.as_ref()),
        &dir,
        &AnnotationArtifacts {
            argmax_abs: &argmax_abs,
            markers: &args.markers,
            annotation_abs: None,
            cluster_celltype_q_abs: None,
            cluster_celltype_es_abs: None,
            cluster_expression_abs: None,
            ontology_assignment_abs: ontology_assign.as_deref(),
            ontology_node_mass_abs: ontology_mass.as_deref(),
        },
    )?;
    Ok(())
}
