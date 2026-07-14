//! `senna annotate-by-projection` — firm marker-set annotation by projection
//! onto a co-embedded feature space.
//!
//! A thin senna front-end over the shared firm term-ORA core
//! ([`graph_embedding_util::type_annotation::annotate_embeddings_ora`]): it
//! reads `outputs.feature_embedding` (genes on the cell manifold) + the cell
//! embedding (`outputs.cell_embedding`, else `outputs.latent`) from the run
//! manifest and hands them to the firm routine — Euclidean nearest-centroid
//! assignment → distance-outlier QC → Leiden clustering → cluster × term
//! hypergeometric over-representation (permutation-calibrated) → optional TreeBH
//! ontology calling.
//!
//! **Embedding-grounded**, so it never re-reads raw counts — complementary to
//! `annotate-by-enrichment` (raw-count-grounded). Applies only to runs with a
//! genuine co-embedded gene space (bge / fne / resolve-embedding-space);
//! topic/svd runs have no such embedding and use `annotate-by-enrichment`.
//!
//! Shares the per-cell contract (`{out}.{annot.parquet,membership.tsv,
//! argmax.tsv}`) and the TreeBH ontology core with the other passes, so their
//! outputs are directly comparable.

use super::args::AnnotateProjectionArgs;
use super::finalize::{clean_outputs, finalize_annotation, AnnotationArtifacts};
use crate::run_manifest::{self, RunManifest};
use anyhow::{Context, Result};
use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, Abstain, InputEmbeddings, MarkerBootstrapConfig, TermOraConfig,
    TERM_ORA_OUTPUT_SUFFIXES,
};
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use std::path::Path;

pub fn run(args: &AnnotateProjectionArgs) -> Result<()> {
    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => run_manifest::derive_out_prefix(&args.from),
    };
    mkdir_parent(&out)?;
    if !args.no_clean {
        clean_outputs(&out, TERM_ORA_OUTPUT_SUFFIXES);
    }

    let (mut manifest, dir) = RunManifest::load(Path::new(args.from.as_ref()))?;
    info!("Loaded manifest ({}): kind={}", args.from, manifest.kind);
    let resolve = |rel: &str| -> String {
        run_manifest::resolve(&dir, rel)
            .to_string_lossy()
            .into_owned()
    };

    // Feature side: genes on the cell manifold (required for projection).
    let feat_rel = manifest
        .outputs
        .feature_embedding
        .as_deref()
        .ok_or_else(|| {
            anyhow::anyhow!(
            "manifest has no `outputs.feature_embedding` — projection needs a co-embedded gene \
             space (a `senna bge` / `fne` / `resolve-embedding-space` run). For topic/svd runs \
             use `senna annotate-by-enrichment`."
        )
        })?;
    // Cell side: prefer the explicit cell_embedding; fall back to latent (plain bge/fne).
    let cell_rel = manifest
        .outputs
        .cell_embedding
        .as_deref()
        .or(manifest.outputs.latent.as_deref())
        .ok_or_else(|| {
            anyhow::anyhow!("manifest has neither `outputs.cell_embedding` nor `outputs.latent`")
        })?;

    let feat_path = resolve(feat_rel);
    let cell_path = resolve(cell_rel);
    let feat = DMatrix::<f32>::from_parquet(&feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;
    info!(
        "projection inputs: features [{} × {}], cells [{} × {}]",
        feat.mat.nrows(),
        feat.mat.ncols(),
        cell.mat.nrows(),
        cell.mat.ncols()
    );

    let cfg = TermOraConfig {
        min_panel_coverage: 0.0, // the default: report + warn on a thin panel, never refuse
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
        panel_perm: args.panel_perm,
        support_perm: args.support_perm,
        // ON by default, as in `faba annotate`: a bare `argmin` over marker centroids always
        // returns something, and returns it with no error bar.
        bootstrap: (!args.no_bootstrap_markers).then_some(MarkerBootstrapConfig {
            n_boot: args.n_boot,
            abstain: if args.abstain_separable {
                Abstain::Separable(args.abstain_alpha)
            } else {
                Abstain::Support(args.min_support)
            },
            set_coverage: args.set_coverage,
            max_set_size: args.max_set_size,
            recluster: !args.no_recluster,
        }),
    };

    annotate_embeddings_ora(
        &InputEmbeddings {
            feature_emb: &feat.mat,
            gene_names: &feat.rows,
            cell_emb: &cell.mat,
            cell_names: &cell.rows,
        },
        &args.markers,
        &out,
        !args.no_idf,
        &cfg,
    )?;

    ///////////////////////////////////////////////
    // wire into the manifest (shared finalizer) //
    ///////////////////////////////////////////////
    let annot = format!("{out}.annot.parquet");
    let argmax = format!("{out}.argmax.tsv");
    let onto_assign = format!("{out}.ontology_assignment.tsv");
    let onto_mass = format!("{out}.ontology_node_mass.parquet");
    let has_onto = Path::new(&onto_assign).exists();
    finalize_annotation(
        &mut manifest,
        Path::new(args.from.as_ref()),
        &dir,
        &AnnotationArtifacts {
            argmax_abs: &argmax,
            markers: &args.markers,
            annotation_abs: Some(&annot),
            // Projection emits cluster × term (not cluster × celltype enrichment);
            // those manifest fields stay None for this pass.
            cluster_celltype_q_abs: None,
            cluster_celltype_es_abs: None,
            cluster_expression_abs: None,
            ontology_assignment_abs: has_onto.then_some(onto_assign.as_str()),
            ontology_node_mass_abs: has_onto.then_some(onto_mass.as_str()),
        },
    )?;

    info!("senna annotate-by-projection complete");
    Ok(())
}
