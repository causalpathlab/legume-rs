//! Shared manifest finalization for both annotation subcommands
//! (`annotate-by-projection`, `annotate-by-enrichment`), so they wire the same
//! `manifest.annotate.*` fields + `defaults.colour_by` and save consistently —
//! one place to keep the two passes' I/O contract in lock-step.

use crate::run_manifest::{self, RunManifest};
use std::path::Path;

/// `{prefix}{suffix}` files written by `annotate-by-enrichment` (relative to its
/// bare `{out}` prefix). NOTE: this prefix is shared with the training run's
/// artifacts (`{out}.cell_embedding.parquet`, `{out}.senna.json`, …), so the
/// list is EXPLICIT — never a glob — to avoid deleting the embedding/manifest.
pub(crate) const ENRICHMENT_OUTPUT_SUFFIXES: &[&str] = &[
    ".annotation.parquet",
    ".argmax.tsv",
    ".membership.tsv",
    ".cluster_celltype_q.parquet",
    ".cluster_celltype_es.parquet",
    ".cluster_celltype_es_std.parquet",
    ".cluster_celltype_p.parquet",
    ".cluster_celltype_q_values.parquet",
    ".cluster_celltype_perm_z.parquet",
    ".cluster_expression.parquet",
    ".ontology_assignment.tsv",
    ".ontology_node_mass.parquet",
];

/// Erase the exact `{prefix}{suffix}` output files (if present) for a fresh
/// re-run. Only the listed files are removed — sibling artifacts (the
/// embedding, the manifest) are never touched. Best-effort: a removal error is
/// logged, not fatal.
pub(crate) fn clean_outputs(prefix: &str, suffixes: &[&str]) {
    let mut removed = 0usize;
    for s in suffixes {
        let path = format!("{prefix}{s}");
        if Path::new(&path).exists() {
            match matrix_util::common_io::remove_file(&path) {
                Ok(()) => removed += 1,
                Err(e) => log::warn!("--clean: could not remove {path}: {e}"),
            }
        }
    }
    if removed > 0 {
        log::info!("--clean: removed {removed} existing output file(s) under {prefix}");
    }
}

/// Absolute paths of the artifacts an annotation pass produced. `argmax_abs` +
/// `markers` are common to both methods; the posterior and cluster-level fields
/// are method-specific (`None` for `annotate-by-projection`).
pub(crate) struct AnnotationArtifacts<'a> {
    pub argmax_abs: &'a str,
    pub markers: &'a str,
    pub annotation_abs: Option<&'a str>,
    pub cluster_celltype_q_abs: Option<&'a str>,
    pub cluster_celltype_es_abs: Option<&'a str>,
    pub cluster_expression_abs: Option<&'a str>,
    /// Ontology outputs (`annotate-by-enrichment --obo --label-cl`). Set to
    /// `None` to CLEAR any stale manifest pointers on a re-run without ontology.
    pub ontology_assignment_abs: Option<&'a str>,
    pub ontology_node_mass_abs: Option<&'a str>,
}

/// Wire the artifacts into the manifest as paths relative to `manifest_dir`,
/// flip the default plot colour to `annotation`, and save back to `from`.
pub(crate) fn finalize_annotation(
    manifest: &mut RunManifest,
    from: &Path,
    manifest_dir: &Path,
    art: &AnnotationArtifacts<'_>,
) -> anyhow::Result<()> {
    let rel = |abs: &str| run_manifest::rel_to_manifest(manifest_dir, abs);
    manifest.annotate.argmax = Some(rel(art.argmax_abs));
    manifest.annotate.markers = Some(art.markers.to_string());
    if let Some(a) = art.annotation_abs {
        manifest.annotate.annotation = Some(rel(a));
    }
    if let Some(q) = art.cluster_celltype_q_abs {
        manifest.annotate.cluster_celltype_q = Some(rel(q));
    }
    if let Some(e) = art.cluster_celltype_es_abs {
        manifest.annotate.cluster_celltype_es = Some(rel(e));
    }
    if let Some(x) = art.cluster_expression_abs {
        manifest.annotate.cluster_expression = Some(rel(x));
    }
    // Overwrite (not conditionally set) so a re-run without ontology clears any
    // stale pointers from a previous standalone `annotate-ontology`.
    manifest.annotate.ontology_assignment = art.ontology_assignment_abs.map(&rel);
    manifest.annotate.ontology_node_mass = art.ontology_node_mass_abs.map(&rel);
    manifest.defaults.colour_by = Some("annotation".into());
    manifest.save(from)
}
