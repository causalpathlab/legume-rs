//! `{prefix}.pb_reference.zarr` — a run's pseudobulks, carried forward so the
//! next `senna update` does not have to re-read the cells they came from.
//!
//! **Why a backend and not a serialized statistic.** Every family trains only
//! on pseudobulks, so the pseudobulks *are* the training set. Written as an
//! ordinary sparse backend they can simply be handed back to the loader
//! alongside the new data, and the whole existing pipeline — projection, PB
//! partitioning, cross-batch matching, `optimize` — runs over the union with no
//! new estimator and no second code path. Carrying them as a `CollapsedStat`
//! blob would instead need bespoke append, merge and re-fit logic.
//!
//! **What is stored.** One column per finest-level pseudobulk, holding the
//! **batch-adjusted per-cell rate** (`mu_adjusted`, falling back to
//! `mu_observed` when a run had no batch structure). Two consequences:
//!
//! - It is a *rate*, so a carried column is directly comparable with a real
//!   cell — which is what makes the cross-batch matching, itself already in
//!   rate space, treat them alike.
//! - It is *adjusted*, so the batch correction this run computed at cell
//!   resolution is baked into the values. The next round therefore estimates δ
//!   only for its new batches, against an already-clean reference. That is the
//!   sense in which the expensive cell-level work is not thrown away.
//!
//! The sidecar carries each column's cell count, which becomes its
//! [multiplicity](data_beans::sparse_io_vector::SparseIoVec::register_column_multiplicity)
//! on the way back in — without it a pseudobulk of 200 cells would weigh the
//! same as one cell.

use crate::embed_common::*;
use serde::{Deserialize, Serialize};

/// Suffixes under the run's `--out` prefix.
pub const BACKEND_SUFFIX: &str = "pb_reference.zarr";
pub const SIDECAR_SUFFIX: &str = "pb_reference.json";

#[must_use]
pub fn backend_path(prefix: &str) -> String {
    format!("{prefix}.{BACKEND_SUFFIX}")
}

#[must_use]
pub fn sidecar_path(prefix: &str) -> String {
    format!("{prefix}.{SIDECAR_SUFFIX}")
}

/// What the backend's columns mean, alongside it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PbReferenceMeta {
    /// senna version that wrote it.
    pub senna_version: Box<str>,
    /// Cells behind each column, in column order — the multiplicity to
    /// register when loading it back.
    pub cell_counts: Vec<f32>,
    /// Batch label to give every carried column. Distinct from any real batch
    /// so the cross-batch matching treats carried and new data as different
    /// batches, which is what identifies δ between rounds.
    pub batch_label: Box<str>,
    /// True when the stored values are `mu_adjusted` rather than
    /// `mu_observed`; false means the source run had no batch structure to
    /// adjust for.
    pub batch_adjusted: bool,
    /// Rounds of accumulation behind this file, for provenance in the log.
    pub generation: u32,
}

/// The reserved batch label. A run's own batches come from user-supplied batch
/// files, so this cannot collide unless someone names a batch after it.
pub const REFERENCE_BATCH: &str = "__pb_reference__";

/// Column-name prefix for carried pseudobulks. Written by [`write`] and relied
/// on by [`ReferenceInput::weights_for`] to tell them from real cells, so the
/// two must agree — hence one constant.
pub const COLUMN_PREFIX: &str = "PBREF_";

/// Cells per finest-level pseudobulk, from the cell → pb membership.
fn cell_counts_from(cell_to_pb_finest: &[usize], n_pb: usize) -> Vec<f32> {
    let mut counts = vec![0.0f32; n_pb];
    for &pb in cell_to_pb_finest {
        if pb < n_pb {
            counts[pb] += 1.0;
        }
    }
    counts
}

/// Write this run's pseudobulks as `{prefix}.pb_reference.{zarr,json}`.
///
/// `cell_to_pb_finest` is the finest level of the run's cell → pb membership,
/// used only to count how many cells each column stands for.
pub fn write(
    prefix: &str,
    finest: &CollapsedOut,
    cell_to_pb_finest: &[usize],
    gene_names: &[Box<str>],
    generation: u32,
) -> anyhow::Result<()> {
    let batch_adjusted = finest.mu_adjusted.is_some();
    let rate_dp: &Mat = preferred_posterior_mean(finest);
    let (n_genes, n_pb) = (rate_dp.nrows(), rate_dp.ncols());
    anyhow::ensure!(
        n_genes == gene_names.len(),
        "pb_reference: {n_genes} rows but {} gene names",
        gene_names.len(),
    );

    let cell_counts = cell_counts_from(cell_to_pb_finest, n_pb);
    let n_empty = cell_counts.iter().filter(|&&c| c <= 0.0).count();
    anyhow::ensure!(
        n_empty < n_pb,
        "pb_reference: every pseudobulk is empty — the cell → pb membership does not match the \
         collapsed output"
    );

    // Zeros are dropped, as in any sparse write; a pseudobulk rate is dense
    // enough that this is a modest saving, and exact — the loader reads a
    // missing entry as zero, which is what it was.
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for pb in 0..n_pb {
        if cell_counts[pb] <= 0.0 {
            continue;
        }
        for g in 0..n_genes {
            let v = rate_dp[(g, pb)];
            if v > 0.0 && v.is_finite() {
                triplets.push((g as u64, pb as u64, v));
            }
        }
    }
    anyhow::ensure!(
        !triplets.is_empty(),
        "pb_reference: refusing to write an all-zero reference"
    );

    let path = backend_path(prefix);
    remove_file(&path)?;
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (n_genes, n_pb, triplets.len()),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )?;
    backend.register_row_names_vec(gene_names);
    backend.register_column_names_vec(&axis_id_names(COLUMN_PREFIX, n_pb));

    let meta = PbReferenceMeta {
        senna_version: env!("CARGO_PKG_VERSION").into(),
        cell_counts,
        batch_label: REFERENCE_BATCH.into(),
        batch_adjusted,
        generation,
    };
    std::fs::write(sidecar_path(prefix), serde_json::to_string_pretty(&meta)?)?;

    info!(
        "Wrote {path}: {n_pb} pseudobulks over {n_genes} genes ({} cells, {}adjusted, gen {generation})",
        meta.cell_counts.iter().sum::<f32>() as usize,
        if batch_adjusted { "" } else { "un" },
    );
    Ok(())
}

/// Emit the carried pseudobulks when the run asked for them.
///
/// Returns whether anything was written, for the manifest slot. The generation
/// counter comes from the parent when this run was itself an update, so the log
/// says how many rounds of accumulation are behind the file.
pub fn emit_if_requested(
    enabled: bool,
    prefix: &str,
    finest: &CollapsedOut,
    cell_to_pb_per_level: Option<&[Vec<usize>]>,
    gene_names: &[Box<str>],
    parent: Option<&str>,
) -> anyhow::Result<bool> {
    if !enabled {
        return Ok(false);
    }
    // Finest-last, matching `collapsed_levels`.
    let Some(finest_membership) = cell_to_pb_per_level.and_then(<[Vec<usize>]>::last) else {
        log::warn!(
            "--emit-pb-reference: this run has no cell → pb membership, so the carried \
             pseudobulks would have no cell counts and would weigh one cell apiece. Skipping."
        );
        return Ok(false);
    };
    let generation = parent
        .map(|p| read_meta(p).map(|m| m.map_or(0, |m| m.generation)))
        .transpose()?
        .unwrap_or(0)
        + 1;
    write(prefix, finest, finest_membership, gene_names, generation)?;
    Ok(true)
}

/// A parent's carried pseudobulks, prepared as an ordinary input for the next
/// fit: a backend path, a synthesized batch file, and the per-column weights.
///
/// `#[serde(skip)]` on the field that holds this — it is derived per invocation
/// from the parent, never part of a recorded fit configuration.
#[derive(Debug, Clone, Default)]
pub struct ReferenceInput {
    pub backend: Box<str>,
    pub batch_file: Box<str>,
    /// One weight per reference column, in column order.
    pub cell_counts: Vec<f32>,
}

impl ReferenceInput {
    /// Cells the reference stands for.
    #[must_use]
    pub fn cells_represented(&self) -> f32 {
        self.cell_counts.iter().sum()
    }

    /// Weights for the whole loaded cohort, given its column names.
    ///
    /// The reference is appended last, so its columns are the trailing
    /// `cell_counts.len()`. That is an assumption about how the loader lays
    /// columns out, and applying weights to the wrong ones would silently
    /// corrupt every pseudobulk size — a wrong denominator with no shape
    /// mismatch to catch it. So the tail is *identified*, not just counted:
    /// carried columns are named `PBREF_*` at write time, and the loader
    /// suffixes `@<backend basename>` when several backends are pushed.
    ///
    /// Both halves are checked. If the tail does not look like the reference,
    /// or a `PBREF_` column shows up outside it, this fails rather than
    /// guessing.
    pub fn weights_for(&self, column_names: &[Box<str>]) -> anyhow::Result<Vec<f32>> {
        let n_ref = self.cell_counts.len();
        let n_total = column_names.len();
        anyhow::ensure!(
            n_total > n_ref,
            "pb_reference has {n_ref} columns and {n_total} were loaded — that leaves no new \
             cells, so the reference is the whole cohort"
        );
        let split = n_total - n_ref;
        let is_ref = |n: &str| n.starts_with(COLUMN_PREFIX);

        if let Some(i) = column_names[split..].iter().position(|n| !is_ref(n)) {
            anyhow::bail!(
                "expected the last {n_ref} loaded columns to be the carried pseudobulks, but \
                 column {} is `{}`. The loader did not append the reference last, so weighting \
                 by position would put every multiplicity on the wrong column.",
                split + i,
                column_names[split + i],
            );
        }
        if let Some(i) = column_names[..split].iter().position(|n| is_ref(n)) {
            anyhow::bail!(
                "column {i} (`{}`) looks like a carried pseudobulk but sits among the new cells; \
                 the reference must be contiguous at the end.",
                column_names[i],
            );
        }

        let mut w = vec![1.0f32; n_total];
        w[split..].copy_from_slice(&self.cell_counts);
        Ok(w)
    }
}

/// Prepare a parent's carried pseudobulks for reuse, writing the batch file
/// the loader needs beside `out`.
///
/// `None` when the parent has none, which is the signal to fall back to
/// re-reading its cells.
pub fn prepare(parent: &str, out: &str) -> anyhow::Result<Option<ReferenceInput>> {
    let Some(meta) = read_meta(parent)? else {
        return Ok(None);
    };
    let backend = backend_path(parent);
    anyhow::ensure!(
        std::path::Path::new(&backend).exists(),
        "{parent} records carried pseudobulks but {backend} is missing"
    );

    // The loader takes batch labels as a file, one line per column.
    let batch_file = format!("{out}.pb_reference_batch.txt");
    let body: String = std::iter::repeat_n(meta.batch_label.as_ref(), meta.cell_counts.len())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&batch_file, body + "\n")?;

    Ok(Some(ReferenceInput {
        backend: backend.into(),
        batch_file: batch_file.into(),
        cell_counts: meta.cell_counts,
    }))
}

/// Read the sidecar for a run's carried pseudobulks, or `None` when it has none.
pub fn read_meta(prefix: &str) -> anyhow::Result<Option<PbReferenceMeta>> {
    let path = sidecar_path(prefix);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(Some(serde_json::from_str(&s).map_err(|e| {
            anyhow::anyhow!("{path}: not a pb_reference sidecar ({e})")
        })?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
