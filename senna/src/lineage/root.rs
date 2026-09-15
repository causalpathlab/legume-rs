//! Resolving which node the forest is rooted at.
//!
//! Three sources, in precedence order: an explicit `--root-node`, a `--root-cell`
//! mapped to its node, and a `--root-type` matched against the trajectory
//! annotation. Anything left unresolved here falls back to the velocity-flux
//! root at the call site.

use anyhow::{Context, Result};
use log::{info, warn};

use graph_embedding_util::type_annotation::CommunityCalls;

/// Resolve the root MST node, in priority order: `--root-node` (validated), `--root-cell`
/// (the node of the named cell's cluster), `type_root` (`--root-type`, a marker-named
/// node), else the velocity-flux-picked root (`None` here).
pub(super) fn resolve_root_hint(
    root_node: Option<usize>,
    root_cell: Option<&str>,
    cell_names: &[Box<str>],
    labels: &[usize],
    k: usize,
    type_root: Option<usize>,
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
        Ok(type_root)
    }
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
