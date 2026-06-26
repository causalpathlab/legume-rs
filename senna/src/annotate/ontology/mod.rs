//! `senna annotate-ontology` — hierarchical, multi-resolution cell-type calling.
//!
//! A thin front-end over the shared, generic TreeBH ontology core
//! ([`enrichment::annotate_ontology_core`]). It reads the cluster × celltype
//! matrix written by `annotate-by-enrichment`, loads the Cell Ontology, and
//! places each cluster on the CL `is_a` tree at the *deepest resolution the
//! data supports* (abstaining on sibling ties). The ontology is injected into
//! the generic core via closures, so the calling/tree/TreeBH math lives once in
//! `enrichment` and is reused by `faba gem-annotate`'s term-ORA path too.
//!
//! Outputs `{out}.ontology_assignment.tsv` and (for soft viz colouring)
//! `{out}.ontology_node_mass.parquet` (Σ of descendant-leaf Q per node).

use super::args::AnnotateOntologyArgs;
use crate::embed_common::Mat;
use crate::run_manifest::{self, RunManifest};
use anyhow::{anyhow, Context, Result};
use graph_embedding_util::type_annotation::annotate_ontology_from_obo;
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;
use std::path::Path;

// Re-export the score enum so sibling modules (e.g. `by_enrichment`) name it
// through `super::ontology::OntologyScore`, keeping their call sites stable.
pub(crate) use enrichment::OntologyScore;

/// Replace the `cluster_celltype_q.parquet` suffix of the manifest's Q path with
/// another enrichment artifact suffix.
fn sibling_artifact(q_path: &str, suffix: &str) -> Result<String> {
    q_path
        .strip_suffix("cluster_celltype_q.parquet")
        .map(|stem| format!("{stem}{suffix}"))
        .ok_or_else(|| {
            anyhow!(
                "manifest cluster_celltype_q path {q_path:?} does not end in \
                 'cluster_celltype_q.parquet'; cannot locate sibling {suffix}"
            )
        })
}

/// Load the Cell Ontology + the curated `label→CL` map and run the shared TreeBH
/// core. The OBO glue (load + inject the generic-core access closures) lives once
/// in [`graph_embedding_util::type_annotation::annotate_ontology_from_obo`]; this
/// is the senna-side entry shared by the standalone `annotate-ontology`
/// subcommand ([`run`]) and the inline ontology pass in `annotate-by-enrichment`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn annotate_ontology_with_obo(
    out: &str,
    label_cl: &str,
    obo: &str,
    fdr_q: f64,
    by: bool,
    score: OntologyScore<'_>,
    q: Option<&Mat>,
    cluster_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> Result<(String, String)> {
    annotate_ontology_from_obo(
        out,
        label_cl,
        obo,
        fdr_q,
        by,
        score,
        q,
        cluster_names,
        celltype_names,
    )
}

pub fn run(args: &AnnotateOntologyArgs) -> Result<()> {
    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => run_manifest::derive_out_prefix(&args.from),
    };
    mkdir_parent(&out)?;

    // ----- manifest → enrichment matrices -----
    let (mut manifest, manifest_dir) = RunManifest::load(Path::new(args.from.as_ref()))?;
    let q_rel = manifest
        .annotate
        .cluster_celltype_q
        .clone()
        .ok_or_else(|| {
            anyhow!(
                "manifest has no `annotate.cluster_celltype_q` — run \
             `senna annotate-by-enrichment --from {} -m <markers>` first",
                args.from
            )
        })?;
    let q_abs = run_manifest::resolve(&manifest_dir, &q_rel)
        .to_string_lossy()
        .into_owned();

    // Score preference: explicit --use-perm-p → pooled count p; otherwise the
    // correlation-preserving permutation z when present, else the row-
    // randomization restandardized ES. Both z forms use leaf p = Φ(−z).
    let (score_abs, from_z) = if args.use_perm_p {
        (
            sibling_artifact(&q_abs, "cluster_celltype_p.parquet")?,
            false,
        )
    } else {
        let perm_z = sibling_artifact(&q_abs, "cluster_celltype_perm_z.parquet")?;
        if Path::new(&perm_z).exists() {
            (perm_z, true)
        } else {
            (
                sibling_artifact(&q_abs, "cluster_celltype_es_std.parquet")?,
                true,
            )
        }
    };

    let score = Mat::from_parquet_with_row_names(&score_abs, Some(0))
        .with_context(|| format!("reading score matrix {score_abs}"))?;
    let q = Mat::from_parquet_with_row_names(&q_abs, Some(0))
        .with_context(|| format!("reading Q matrix {q_abs}"))?;
    anyhow::ensure!(
        score.cols == q.cols,
        "score and Q matrices have mismatched celltype columns"
    );
    anyhow::ensure!(
        score.rows == q.rows,
        "score and Q matrices have mismatched cluster rows (different order/clustering)"
    );
    let cluster_names = score.rows.clone();
    let celltype_names = score.cols.clone();
    let (n_clusters, n_types) = (score.mat.nrows(), score.mat.ncols());
    info!(
        "loaded {n_clusters} clusters × {n_types} celltypes from {} ({})",
        score_abs,
        if from_z { "z→p" } else { "permutation p" }
    );

    let score_in = if from_z {
        OntologyScore::Z(&score.mat)
    } else {
        OntologyScore::Pvalue(&score.mat)
    };
    let (assign_path, mass_path) = annotate_ontology_with_obo(
        &out,
        &args.label_cl,
        &args.obo,
        args.fdr_q,
        args.by,
        score_in,
        Some(&q.mat),
        &cluster_names,
        &celltype_names,
    )?;

    manifest.annotate.ontology_assignment =
        Some(run_manifest::rel_to_manifest(&manifest_dir, &assign_path));
    manifest.annotate.ontology_node_mass =
        Some(run_manifest::rel_to_manifest(&manifest_dir, &mass_path));
    manifest.save(Path::new(args.from.as_ref()))?;

    info!("senna annotate-ontology complete");
    Ok(())
}
