//! Resolving which node the forest is rooted at.
//!
//! Four sources, in precedence order: an explicit `--root-node`, a `--root-cell`
//! mapped to its node, a `--root-type` matched against the trajectory annotation,
//! and finally gem's own DAG root.

use anyhow::{Context, Result};
use log::{info, warn};
use std::path::Path;

use graph_embedding_util::type_annotation::CommunityCalls;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

/// Resolve the root MST node, in priority order: `--root-node` (validated), `--root-cell`
/// (the node of the named cell's cluster), `type_root` (`--root-type`, a marker-named
/// node), `gem_root` (gem's velocity-DAG source, from `--root-from-gem`), the
/// velocity-flux-picked root, else node 0.
pub(super) fn resolve_root_hint(
    root_node: Option<usize>,
    root_cell: Option<&str>,
    cell_names: &[Box<str>],
    labels: &[usize],
    k: usize,
    type_root: Option<usize>,
    gem_root: Option<usize>,
) -> Result<Option<usize>> {
    if let Some(r) = root_node {
        anyhow::ensure!(r < k, "--root-node {r} out of range (K = {k})");
        Ok(Some(r))
    } else if let Some(name) = root_cell {
        let idx = cell_names
            .iter()
            .position(|c| c.as_ref() == name)
            .with_context(|| format!("--root-cell '{name}' not found in latent"))?;
        Ok(Some(labels[idx]))
    } else {
        Ok(type_root.or(gem_root))
    }
}

/// Map gem's inferred root to an MST centroid (for `--root-from-gem`): read
/// `{prefix}.dag_pseudotime.parquet` and return the MST node that dominates gem's low-τ
/// (velocity-DAG source) region — the **modal** cluster among the lowest-τ cells, which
/// is robust to a single τ≈0 outlier (a rare cell the old min-τ-cell pick could land on).
/// Returns `None` (with a warning) — so the caller falls back to the velocity-flux root —
/// when the file is absent/unreadable, no low-τ barcode matches the latent, or gem's
/// `lineage_qc.json` reports zero terminal fates (a structureless DAG with meaningless τ).
pub(super) fn gem_root_node(
    prefix: &str,
    cell_names: &[Box<str>],
    labels: &[usize],
    k: usize,
) -> Option<usize> {
    if gem_dag_n_terminals(prefix) == Some(0) {
        warn!(
            "--root-from-gem: gem's velocity-DAG has no terminal structure (lineage_qc.json); \
             using the velocity-flux root instead"
        );
        return None;
    }
    let path = format!("{prefix}.dag_pseudotime.parquet");
    if !Path::new(&path).exists() {
        warn!("--root-from-gem: {path} absent; falling back to the velocity-flux root");
        return None;
    }
    let pt = match DMatrix::<f32>::from_parquet(&path) {
        Ok(pt) => pt,
        Err(e) => {
            warn!("--root-from-gem: cannot read {path} ({e}); falling back to velocity root");
            return None;
        }
    };
    // Barcode → MST node lookup, then vote the modal node over the lowest-τ cells.
    let bc_label: std::collections::HashMap<&str, usize> = cell_names
        .iter()
        .zip(labels)
        .map(|(c, &l)| (c.as_ref(), l))
        .collect();
    let nrow = pt.mat.nrows();
    let mut order: Vec<usize> = (0..nrow).collect();
    order.sort_by(|&a, &b| {
        pt.mat[(a, 0)]
            .partial_cmp(&pt.mat[(b, 0)])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_low = (nrow / 20).clamp(5.min(nrow), nrow); // lowest ~5% of τ, ≥ 5 cells (or all)
    let mut votes = vec![0usize; k];
    for &r in order.iter().take(n_low) {
        if let Some(&lab) = pt.rows.get(r).and_then(|bc| bc_label.get(bc.as_ref())) {
            if lab < k {
                votes[lab] += 1;
            }
        }
    }
    let root = (0..k).max_by_key(|&c| votes[c])?;
    if votes[root] == 0 {
        warn!("--root-from-gem: no low-τ barcode matched the latent; using flux root");
        return None;
    }
    info!(
        "--root-from-gem: low-τ region ({n_low} cells) → MST node {root} ({} votes)",
        votes[root]
    );
    Some(root)
}

/// Terminal-fate count from gem's `{prefix}.lineage_qc.json`. `None` when the file is
/// absent/unreadable or the field is missing — the caller then does NOT veto the gem
/// root (no signal). `Some(0)` means a structureless DAG whose τ is meaningless.
pub(super) fn gem_dag_n_terminals(prefix: &str) -> Option<usize> {
    let s = std::fs::read_to_string(format!("{prefix}.lineage_qc.json")).ok()?;
    let qc: serde_json::Value = serde_json::from_str(&s).ok()?;
    qc.get("n_terminals")?.as_u64().map(|n| n as usize)
}

/// `--root-type`: the MST node whose per-node call matches `root_type` (case-insensitive)
/// with the highest confidence, or `None` (with a warning) when no node carries that type.
pub(super) fn root_type_node(calls: &CommunityCalls, root_type: &str) -> Option<usize> {
    let node = (0..calls.labels.len())
        .filter(|&i| calls.labels[i].eq_ignore_ascii_case(root_type))
        .max_by(|&a, &b| {
            calls.confidence[a]
                .partial_cmp(&calls.confidence[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    match node {
        Some(i) => {
            info!(
                "--root-type '{root_type}' → MST node {i} (confidence {:.3})",
                calls.confidence[i]
            );
            Some(i)
        }
        None => {
            warn!("--root-type '{root_type}' matched no trajectory node; using the next root rule");
            None
        }
    }
}

#[cfg(test)]
#[path = "root_tests.rs"]
mod root_tests;
