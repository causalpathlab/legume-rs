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
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::{IoOps, MatWithNames};

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

/// Load the marker-matching gene table for a run: resolve `outputs.feature_embedding`
/// off `{prefix}`'s run manifest and, for a [`RunKind::Gem`] run only, apply
/// [`select_spliced_rows`]. Every other kind's feature embedding is already
/// gene-keyed and is returned as read.
pub(crate) fn load_marker_feature_embedding(prefix: &str) -> Result<MatWithNames<DMatrix<f32>>> {
    let (manifest, dir) = run_manifest::load_for(prefix)?;
    let rel = manifest
        .outputs
        .feature_embedding
        .as_deref()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{prefix}: manifest has no `outputs.feature_embedding` — this needs a co-embedded \
             gene space (a `senna gem` / `bge` / `fne` / `resolve-embedding-space` run)"
            )
        })?;
    let path = run_manifest::resolve(&dir, rel)
        .to_string_lossy()
        .into_owned();
    let feat = DMatrix::<f32>::from_parquet(&path)
        .with_context(|| format!("reading gene embedding {path}"))?;
    if manifest.kind == RunKind::Gem {
        select_spliced_rows(feat, &path)
    } else {
        Ok(feat)
    }
}

#[cfg(test)]
#[path = "marker_embedding/tests.rs"]
mod tests;
