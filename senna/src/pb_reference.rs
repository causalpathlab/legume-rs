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
//! **batch-adjusted per-cell evidence rate** — `sum/denom` with no prior in
//! either, from `mu_adjusted` (falling back to `mu_observed` when a run had no
//! batch structure), zero wherever the posterior has no data support. Columns
//! a run itself carried in are the exception: the collapse keeps each in its
//! own singleton finest group and [`write`] re-emits it from `mu_observed` —
//! its stored, already-adjusted values back out — so the reference is
//! **append-only**: it extends by the new data's groups each round and never
//! re-averages what earlier rounds summarized. Three consequences:
//!
//! - It is a *rate*, so a carried column is directly comparable with a real
//!   cell — which is what makes the cross-batch matching, itself already in
//!   rate space, treat them alike.
//! - It is *adjusted*, so the batch correction this run computed is baked into
//!   the values. The next round anchors on it (see
//!   `MultilevelParams::anchor_batches`): new batches are corrected toward
//!   this frame, and the frame itself is never re-adjusted.
//! - Paired with the sidecar's cell counts it is a **bijection of the
//!   sufficient statistics** `(observed_sum, size)`: the next round's collapse
//!   reconstructs them exactly through the weighted-column path, so carrying
//!   pseudobulks forward is lossless at the finest level.
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
    /// The round each column was first summarized in, in column order —
    /// carried columns keep their origin round across re-emissions, so a
    /// grown reference can be audited (or a bad round dropped) later.
    /// Empty in sidecars written before this field existed; treat as
    /// "all from `generation`".
    #[serde(default)]
    pub column_generation: Vec<u32>,
}

/// The reserved batch label. A run's own batches come from user-supplied batch
/// files, so this cannot collide unless someone names a batch after it.
pub const REFERENCE_BATCH: &str = "__pb_reference__";

/// Column-name prefix for carried pseudobulks. Written by [`write`] and relied
/// on by [`weights_for`] to tell them from real cells, so the
/// two must agree — hence one constant.
pub const COLUMN_PREFIX: &str = "PBREF_";

/// Cells per finest-level pseudobulk, from the cell → pb membership.
///
/// `column_weight` is the cohort's [`SparseIoVec::column_multiplicities`]:
/// what each loaded column stands for — 1 for a real cell, its own cell count
/// for a pseudobulk this run itself carried in, `None` when no multiplicities
/// were registered at all. Without it the mass of every earlier cohort would
/// collapse to "one cell per carried column" each round, so a model that
/// absorbed 900 cells and then 400 more would record 735 rather than 1300 —
/// old data quietly decaying toward irrelevance the longer a model is grown,
/// which is the one thing carrying pseudobulks forward exists to prevent.
/// A `Some` of the wrong length is refused for the same reason: it would
/// silently mis-weight the tail.
///
/// Total mass is therefore conserved: `Σ cell_counts == Σ column_weight` over
/// the columns that land in range.
pub fn cell_counts_from(
    cell_to_pb_finest: &[usize],
    n_pb: usize,
    column_weight: Option<&[f32]>,
) -> anyhow::Result<Vec<f32>> {
    if let Some(w) = column_weight {
        anyhow::ensure!(
            w.len() == cell_to_pb_finest.len(),
            "cell_counts_from: {} column weights for {} columns — the weights were built in a \
             different column space",
            w.len(),
            cell_to_pb_finest.len(),
        );
    }
    let mut counts = vec![0.0f32; n_pb];
    for (col, &pb) in cell_to_pb_finest.iter().enumerate() {
        if pb < n_pb {
            counts[pb] += column_weight.map_or(1.0, |w| w[col]);
        }
    }
    Ok(counts)
}

/// Write this run's pseudobulks as `{prefix}.pb_reference.{zarr,json}`.
///
/// `cell_to_pb_finest` is the finest level of the run's cell → pb membership,
/// and `column_weight` is each column's multiplicity — see
/// [`cell_counts_from`] for why counting columns is not enough.
///
/// **Append-only across rounds.** `n_carried_tail` is how many trailing
/// loaded columns are a parent's carried reference (0 on a fresh fit). The
/// collapse keeps each of those in its own singleton finest group
/// (`split_anchored_finest_groups`), so the emitted reference is the parent's
/// columns reproduced from their own evidence plus the new data's groups —
/// the memory *extends* every round rather than being re-averaged into
/// (batch × group) blends, which would compound resolution loss. Two
/// consequences enforced here:
///
/// - A pure-carried group re-emits from **`mu_observed`**, not the adjusted
///   posterior: its stored values are already in the reference frame (δ for
///   the anchor batch sits at the prior), and observed evidence is exactly
///   `y·w / w` — the stored rate back out, up to one f32 rounding. Running
///   it through `mu_adjusted` instead would re-apply the near-1 anchor δ
///   each round and drift the frame.
/// - A group holding new mass emits from `mu_adjusted` as before — corrected
///   *toward* the frozen frame, then frozen in turn.
pub fn write(
    prefix: &str,
    finest: &CollapsedOut,
    cell_to_pb_finest: &[usize],
    column_weight: Option<&[f32]>,
    gene_names: &[Box<str>],
    n_carried_tail: usize,
    parent_meta: Option<&PbReferenceMeta>,
) -> anyhow::Result<()> {
    // Evidence, not posterior. Two distinct mistakes hid in "store the
    // posterior mean, keep v > 0":
    //
    // - A Gamma posterior mean is nonzero at EVERY entry — unobserved genes
    //   sit at the prior floor `a0/(b0 + n)` — so the reference serialized
    //   100.0% dense (measured: 34M of 34M entries), and that haze rode every
    //   counterfactual and polluted the observed sums at multiplicity weight.
    // - Even on supported entries, the mean `(a0 + sum)/(b0 + n)` carries
    //   prior shrinkage that the next round re-ingests as data.
    //
    // `evidence_mean` fixes both at once: `sum/denom` where there is data,
    // zero where there is not. Paired with the sidecar's cell counts it is a
    // bijection of the sufficient statistics — the next round's collapse
    // reconstructs `(observed_sum, size)` exactly, so carrying pseudobulks
    // forward is lossless at the finest level rather than approximately right.
    let param = finest.mu_adjusted.as_ref().unwrap_or(&finest.mu_observed);
    let batch_adjusted = finest.mu_adjusted.is_some();
    let (n_genes, n_pb) = (
        param.posterior_mean().nrows(),
        param.posterior_mean().ncols(),
    );
    anyhow::ensure!(
        n_genes == gene_names.len(),
        "pb_reference: {n_genes} rows but {} gene names",
        gene_names.len(),
    );

    let cell_counts = cell_counts_from(cell_to_pb_finest, n_pb, column_weight)?;
    let n_empty = cell_counts.iter().filter(|&&c| c <= 0.0).count();
    anyhow::ensure!(
        n_empty < n_pb,
        "pb_reference: every pseudobulk is empty — the cell → pb membership does not match the \
         collapsed output"
    );

    // Which groups hold new mass, and which are a carried column passing
    // through. The carried columns are the trailing `n_carried_tail` of the
    // loaded cohort (the identified tail — see `weights_for`); a group is
    // "pure carried" when every member sits in that tail. Alongside, remember
    // each pure-carried singleton's source column so its origin round can be
    // looked up in the parent sidecar.
    let split = cell_to_pb_finest.len().saturating_sub(n_carried_tail);
    // The append-only collapse keeps each carried column in a singleton
    // group; a multi-member or mixed group (a deliberate re-collapse, or an
    // older binary) is a fresh summary and takes the current round. New
    // columns all precede the tail, so a mixed group is marked re-summarized
    // before its carried member is seen.
    let mut carried_col: Vec<Option<usize>> = vec![None; n_pb];
    let mut resummarized = vec![false; n_pb];
    for (col, &pb) in cell_to_pb_finest.iter().enumerate() {
        if pb >= n_pb {
            continue;
        }
        if col < split || carried_col[pb].is_some() {
            resummarized[pb] = true;
        } else {
            carried_col[pb] = Some(col - split);
        }
    }
    let pure_carried = |pb: usize| !resummarized[pb] && carried_col[pb].is_some();

    let generation = parent_meta.map_or(0, |m| m.generation) + 1;
    let column_generation: Vec<u32> = (0..n_pb)
        .map(|pb| match carried_col[pb] {
            Some(j) if !resummarized[pb] => parent_meta.map_or(generation, |m| {
                m.column_generation.get(j).copied().unwrap_or(m.generation)
            }),
            _ => generation,
        })
        .collect();

    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (pb, &count) in cell_counts.iter().enumerate() {
        if count <= 0.0 {
            continue;
        }
        let src = if pure_carried(pb) {
            &finest.mu_observed
        } else {
            param
        };
        for g in 0..n_genes {
            let v = src.evidence_mean(g, pb);
            if v > 0.0 && v.is_finite() {
                triplets.push((g as u64, pb as u64, v));
            }
        }
    }
    anyhow::ensure!(
        !triplets.is_empty(),
        "pb_reference: refusing to write an all-zero reference"
    );
    let density = triplets.len() as f64 / (n_genes as f64 * n_pb as f64);
    info!(
        "pb_reference support: {} entries ({:.1}% dense)",
        triplets.len(),
        100.0 * density
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

    let n_passthrough = (0..n_pb).filter(|&pb| pure_carried(pb)).count();
    let meta = PbReferenceMeta {
        senna_version: env!("CARGO_PKG_VERSION").into(),
        cell_counts,
        batch_label: REFERENCE_BATCH.into(),
        batch_adjusted,
        generation,
        column_generation,
    };
    std::fs::write(sidecar_path(prefix), serde_json::to_string_pretty(&meta)?)?;

    info!(
        "Wrote {path}: {n_pb} pseudobulks over {n_genes} genes ({} cells, {}adjusted, gen \
         {generation}; {} carried through, {} new)",
        meta.cell_counts.iter().sum::<f32>() as usize,
        if batch_adjusted { "" } else { "un" },
        n_passthrough,
        n_pb - n_passthrough,
    );
    Ok(())
}

/// Emit the carried pseudobulks when the run asked for them.
///
/// Returns the manifest's `pb_reference_suffix` — `Some(BACKEND_SUFFIX)` when
/// the file was written — so call sites assign it straight into the
/// `RunDescription` instead of each re-mapping a bool. The generation counter
/// comes from the parent when this run was itself an update, so the log says
/// how many rounds of accumulation are behind the file.
///
/// `column_weight` is the loaded cohort's
/// [`SparseIoVec::column_multiplicities`], captured while the backend is
/// still alive — taking the slice rather than the `SparseIoVec` lets callers
/// run this *after* the steps that consume the backend (CNV detection), so
/// the triplet build never overlaps it in memory.
///
/// `parent` is the run this one continues (`--init-from`); absent, the count
/// restarts at 1, which is honest: nothing on disk connects this run to an
/// earlier one.
///
/// `reference` is the carried input this run actually *loaded* (`None` on a
/// fresh fit or an exact re-collapse) — its column count identifies the
/// carried tail so [`write`] can pass those columns through instead of
/// re-summarizing them. It is deliberately separate from `parent`: an update
/// without `--use-pb-reference` has a parent (for the generation counter) but
/// no carried tail.
#[allow(clippy::too_many_arguments)] // one slot per provenance source, each documented above
pub fn emit_if_requested(
    enabled: bool,
    prefix: &str,
    finest: &CollapsedOut,
    cell_to_pb_per_level: Option<&[Vec<usize>]>,
    column_weight: Option<&[f32]>,
    gene_names: &[Box<str>],
    parent: Option<&str>,
    reference: Option<&ReferenceInput>,
) -> anyhow::Result<Option<&'static str>> {
    if !enabled {
        return Ok(None);
    }
    // Finest-last, matching `collapsed_levels`.
    let Some(finest_membership) = cell_to_pb_per_level.and_then(<[Vec<usize>]>::last) else {
        log::warn!(
            "--emit-pb-reference: this run has no cell → pb membership, so the carried \
             pseudobulks would have no cell counts and would weigh one cell apiece. Skipping."
        );
        return Ok(None);
    };
    let parent_meta = parent.map(read_meta).transpose()?.flatten();
    write(
        prefix,
        finest,
        finest_membership,
        column_weight,
        gene_names,
        reference.map_or(0, |r| r.cell_counts.len()),
        parent_meta.as_ref(),
    )?;
    Ok(Some(BACKEND_SUFFIX))
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
}

/// NB-Fisher gene weights for a cohort whose columns carry multiplicities —
/// the pseudobulk-fitted branch of `refine_weighting::fit_fisher_weights`,
/// which owns the cell-vs-pseudobulk policy and the measurements behind it.
///
/// **The load-bearing choice here is `mu_observed`, NOT the batch-adjusted
/// posterior** — even though the carried columns were *stored* adjusted, and
/// even though `preferred_posterior_mean` is the idiom everywhere else. The
/// NB trend describes the observation process, and between-batch spread is
/// part of the variance it is meant to see; the cell-level pass this has to
/// agree with reads raw counts. Removing δ first shrinks apparent dispersion
/// and reorders which genes look over-dispersed — measured at Spearman
/// ρ 0.47 against the exact re-collapse, versus ρ 0.98 for the observed
/// posterior. A "consistency" cleanup that swaps this back to
/// `preferred_posterior_mean` compiles fine and quietly re-breaks it, which
/// is why the choice is pinned by a test rather than only by this comment.
pub fn fisher_weights_for_weighted_cohort(
    collapsed: &CollapsedOut,
    cell_to_pb_finest: &[usize],
    column_weight: Option<&[f32]>,
    coarsening: Option<&FeatureCoarsening>,
) -> anyhow::Result<Vec<f32>> {
    let mu_ds = collapsed.mu_observed.posterior_mean();
    let size_s = cell_counts_from(cell_to_pb_finest, mu_ds.ncols(), column_weight)?;
    data_beans_alg::gene_weighting::fisher_weights_from_pseudobulk(mu_ds, &size_s, coarsening)
}

/// Keep-mask over the loaded columns that excludes the carried pseudobulks,
/// intersected with whatever QC already dropped. `None` reference passes the
/// QC mask through untouched.
///
/// Carried pseudobulks are training inputs, not cells. Left in, every
/// downstream per-cell artifact — `clustering`, `plot`, `annotate` — would
/// treat each one as an extra cell whose "barcode" is `PBREF_37`.
#[must_use]
pub fn exclude_carried(
    reference: Option<&ReferenceInput>,
    n_columns: usize,
    keep: Option<Vec<usize>>,
) -> Option<Vec<usize>> {
    let Some(r) = reference else { return keep };
    let n_real = n_columns.saturating_sub(r.cell_counts.len());
    info!(
        "Excluding {} carried pseudobulks from the per-cell outputs ({n_real} cells remain)",
        r.cell_counts.len(),
    );
    Some(match keep {
        Some(keep) => keep.into_iter().filter(|&i| i < n_real).collect(),
        None => (0..n_real).collect(),
    })
}

/// One weight per loaded column: 1 for a real cell, its cell count for a
/// carried pseudobulk.
///
/// The reference is appended last, so its columns are the trailing
/// `cell_counts.len()`. That is an assumption about how the loader lays columns
/// out, and applying weights to the wrong ones would silently corrupt every
/// pseudobulk size — a wrong denominator with no shape mismatch to catch it. So
/// the tail is *identified*, not just counted: carried columns are named
/// `PBREF_*` at write time, and the loader suffixes `@<backend basename>` when
/// several backends are pushed.
///
/// Both halves are checked. If the tail does not look like the reference, or a
/// `PBREF_` column shows up outside it, this fails rather than guessing.
pub fn weights_for(cell_counts: &[f32], column_names: &[Box<str>]) -> anyhow::Result<Vec<f32>> {
    let n_ref = cell_counts.len();
    let n_total = column_names.len();
    anyhow::ensure!(
        n_total > n_ref,
        "pb_reference has {n_ref} columns and {n_total} were loaded — that leaves no new cells, \
         so the reference is the whole cohort"
    );
    let split = n_total - n_ref;
    let is_ref = |n: &str| n.starts_with(COLUMN_PREFIX);

    if let Some(i) = column_names[split..].iter().position(|n| !is_ref(n)) {
        anyhow::bail!(
            "expected the last {n_ref} loaded columns to be the carried pseudobulks, but column \
             {} is `{}`. The loader did not append the reference last, so weighting by position \
             would put every multiplicity on the wrong column.",
            split + i,
            column_names[split + i],
        );
    }
    if let Some(i) = column_names[..split].iter().position(|n| is_ref(n)) {
        anyhow::bail!(
            "column {i} (`{}`) looks like a carried pseudobulk but sits among the new cells; the \
             reference must be contiguous at the end.",
            column_names[i],
        );
    }

    let mut w = vec![1.0f32; n_total];
    w[split..].copy_from_slice(cell_counts);
    Ok(w)
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
    matrix_util::common_io::write_types(
        &vec![meta.batch_label.clone(); meta.cell_counts.len()],
        &batch_file,
    )?;

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
