//! `annotate-by-projection` — firm marker-set annotation by projection onto a
//! co-embedded feature space.
//!
//! A thin front-end over the shared firm term-ORA core
//! ([`graph_embedding_util::type_annotation::annotate_embeddings_ora`]): it
//! takes the co-embedded gene vectors (genes on the cell manifold) + the cell
//! embedding — both loaded by the caller, since finding them is manifest work —
//! and hands them to the firm routine: Euclidean nearest-centroid
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

use crate::args::AnnotateProjectionArgs;
use crate::mat_io::{Mat, MatWithNames};
use crate::outputs::{clean_outputs, AnnotationOutputs};
use anyhow::Result;
use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, Abstain, InputEmbeddings, MarkerBootstrapConfig, TermOraConfig,
    TERM_ORA_OUTPUT_SUFFIXES,
};
use legume_numeric::matrix::common_io::mkdir_parent;
use log::info;
use std::path::Path;

/// The two embeddings the projection pass scores against each other. Finding
/// them means reading `outputs.feature_coembedding` / `outputs.cell_embedding`,
/// so the caller loads them — see `senna::annotate_manifest`.
pub struct ProjectionInputs<'a> {
    /// Genes on the cell manifold (the co-embedded feature space).
    pub feature_embedding: &'a MatWithNames<Mat>,
    /// Cells in the same space.
    pub cell_embedding: &'a MatWithNames<Mat>,
}

pub fn run(
    args: &AnnotateProjectionArgs,
    default_out: &str,
    inputs: &ProjectionInputs<'_>,
) -> Result<AnnotationOutputs> {
    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => default_out.to_string(),
    };
    mkdir_parent(&out)?;
    if !args.no_clean {
        clean_outputs(&out, TERM_ORA_OUTPUT_SUFFIXES);
    }

    let feat = inputs.feature_embedding;
    let cell = inputs.cell_embedding;
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
        min_markers: args.min_markers,
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
        // ON by default, as in `senna annotate-by-projection`: a bare `argmin` over marker centroids always
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

    ///////////////////////////////////////
    // what the caller records in a manifest //
    ///////////////////////////////////////
    let onto_assign = format!("{out}.ontology_assignment.tsv");
    let onto_mass = format!("{out}.ontology_node_mass.parquet");
    let has_onto = Path::new(&onto_assign).exists();

    info!("annotate-by-projection complete → {out}.*");
    Ok(AnnotationOutputs {
        argmax: Some(format!("{out}.argmax.tsv")),
        annotation: Some(format!("{out}.annot.parquet")),
        // Projection emits cluster × term (not cluster × celltype enrichment);
        // those manifest fields stay None for this pass.
        ontology_assignment: has_onto.then_some(onto_assign),
        ontology_node_mass: has_onto.then_some(onto_mass),
        ..AnnotationOutputs::default()
    })
}
