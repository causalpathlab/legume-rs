//! Loading a run's co-embedded **gene** table for marker matching — the
//! per-run `{out}.feature_embedding.parquet` slot on the run manifest.
//!
//! A `gem` run's feature axis is keyed by feature ROW, not by gene: a spliced
//! and an unspliced row per gene. A marker panel names genes, so matching it
//! against the raw table would silently pull both rows into the same
//! centroid, averaging the mature identity together with the nascent one.
//! [`select_spliced_rows`] keeps the mature (spliced) rows and strips the
//! track suffix back to the gene key, which is what the marker matcher
//! expects. Every other kind's feature embedding is already gene-keyed and
//! needs no such split.

use anyhow::{Context, Result};
use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::traits::{IoOps, MatWithNames};
use log::info;

use crate::run_manifest::{self, RunKind};

/// The feature-row suffix annotation reads.
///
/// Spliced only, and not a parameter. A marker call is a statement about MATURE
/// identity; the nascent program is a different quantity, and averaging the two
/// under one gene name is what selecting a single suffix exists to prevent.
const SPLICED_SUFFIX: &str = "/count/spliced";

/// Keep only the spliced rows out of a gem feature embedding, re-keyed by gene
/// so a marker panel can match them.
///
/// Takes the table rather than a prefix: the caller resolves it through the run
/// manifest, which is the one place that knows where a run's outputs are.
///
/// Errors when the modality selects nothing — that is a real misconfiguration (a
/// spliced-only gem run has no unspliced rows) and silently annotating against
/// an empty gene set would be worse.
pub fn select_spliced_rows(
    feat: MatWithNames<DMatrix<f32>>,
    path: &str,
) -> Result<MatWithNames<DMatrix<f32>>> {
    let suffix = SPLICED_SUFFIX;
    let keep: Vec<usize> = feat
        .rows
        .iter()
        .enumerate()
        .filter(|(_, name)| name.ends_with(suffix))
        .map(|(i, _)| i)
        .collect();
    anyhow::ensure!(
        !keep.is_empty(),
        "{path} has no `{suffix}` feature rows (found {} rows, e.g. `{}`). A spliced-only \
         `senna gem` run has no unspliced program to annotate.",
        feat.rows.len(),
        feat.rows.first().map_or("", |s| s.as_ref())
    );

    let rows: Vec<Box<str>> = keep
        .iter()
        .map(|&i| {
            let name = feat.rows[i].as_ref();
            Box::from(name.strip_suffix(suffix).unwrap_or(name))
        })
        .collect();
    let mat = feat.mat.select_rows(&keep);
    info!(
        "gene embedding: {} of {} feature rows are `{suffix}` → {} genes [{} × {}]",
        keep.len(),
        feat.rows.len(),
        rows.len(),
        mat.nrows(),
        mat.ncols()
    );
    Ok(MatWithNames {
        mat,
        rows,
        cols: feat.cols,
    })
}

/// Load the marker-matching gene table for a run: `outputs.feature_coembedding`
/// off the run's manifest — genes on the cell manifold, which is what a
/// Euclidean nearest-centroid call against the cells needs. A run that never
/// co-embeds (`fne`, the masked family) has only `outputs.feature_embedding`,
/// its ρ, which shares the cells' space by construction and is used as is. A
/// run that DOES co-embed but recorded none (an interrupted `bge` / `gem`)
/// is refused: its ρ is the off-manifold cloud, and matching markers on it
/// would be ill-posed. For a [`RunKind::Gem`] run only, [`select_spliced_rows`]
/// is applied; every other kind's table is already gene-keyed and is
/// returned as read.
///
/// `prefix` is only used in error messages. Callers that already hold a loaded
/// manifest should pass it in (avoids a second `load_for`).
pub(crate) fn load_marker_feature_embedding_from(
    manifest: &crate::run_manifest::RunManifest,
    dir: &std::path::Path,
    prefix: &str,
) -> Result<MatWithNames<DMatrix<f32>>> {
    let coembeds = matches!(
        manifest.kind,
        RunKind::Bge | RunKind::Gem | RunKind::Simba | RunKind::ResolveEmbeddingSpace
    );
    let (slot, rel) = match (
        manifest.outputs.feature_coembedding.as_deref(),
        manifest.outputs.feature_embedding.as_deref(),
    ) {
        (Some(rel), _) => ("feature_coembedding", rel),
        (None, Some(_)) if coembeds => anyhow::bail!(
            "{prefix}: a {} run with no `outputs.feature_coembedding` — the co-embed was not \
             written (an interrupted run?), and its raw gene embedding ρ is not on the cell \
             manifold. Re-run the fit to completion.",
            manifest.kind
        ),
        (None, Some(rel)) => ("feature_embedding", rel),
        (None, None) => anyhow::bail!(
            "{prefix}: manifest has neither `outputs.feature_coembedding` nor \
             `outputs.feature_embedding` — this needs a gene embedding (a `senna gem` / `bge` / \
             `fne` / `resolve-embedding-space` run)"
        ),
    };
    let path = run_manifest::resolve(dir, rel)
        .to_string_lossy()
        .into_owned();
    let feat = DMatrix::<f32>::from_parquet(&path)
        .with_context(|| format!("reading gene embedding {path} (`outputs.{slot}`)"))?;
    if manifest.kind == RunKind::Gem {
        select_spliced_rows(feat, &path)
    } else {
        Ok(feat)
    }
}

#[cfg(test)]
#[path = "marker_embedding/tests.rs"]
mod tests;
