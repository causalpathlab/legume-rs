//! Split cells into cores for per-panel plotting.
//!
//! Pinto persists batch / data-file identity in `coord_pairs.parquet`
//! as the `left_batch` column (appended in
//! `pinto/src/util/input.rs` multi-batch path). One core per unique
//! batch label; a single-batch run becomes a single core named `all`.
//!
//! No spatial-connectivity detection — the user declined that in
//! favor of the simpler batch-based split.

use super::load::CellTable;
use super::viridis;
use plot_utils::rasterize::DataBounds;

pub struct CoreSpec {
    /// Sanitized label used in output filenames.
    pub name: String,
    /// Indices into CellTable.names / coords for this core.
    pub cell_ixs: Vec<usize>,
    /// Robust-clipped data bounds.
    pub bounds: DataBounds,
}

impl CoreSpec {
    pub fn n(&self) -> usize {
        self.cell_ixs.len()
    }
}

/// Group cells by batch label (or "all" if none), drop cores below
/// `min_core_cells`, compute per-core bounds with robust percentile
/// clipping (driven by `coord_clip`).
pub fn partition_cells(cells: &CellTable, min_core_cells: usize, coord_clip: f32) -> Vec<CoreSpec> {
    let mut buckets: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for i in 0..cells.n() {
        let label = cells
            .batches
            .as_ref()
            .map(|b| sanitize(&b[i]))
            .unwrap_or_else(|| "all".to_string());
        buckets.entry(label).or_default().push(i);
    }

    buckets
        .into_iter()
        .filter(|(_, v)| v.len() >= min_core_cells.max(1))
        .map(|(name, cell_ixs)| {
            let xs: Vec<f32> = cell_ixs.iter().map(|&i| cells.coords[i].0).collect();
            let ys: Vec<f32> = cell_ixs.iter().map(|&i| cells.coords[i].1).collect();
            let (xmin, xmax) = viridis::robust_range(&xs, coord_clip);
            let (ymin, ymax) = viridis::robust_range(&ys, coord_clip);
            let bounds = DataBounds::from_minmax(xmin, xmax, ymin, ymax);
            CoreSpec {
                name,
                cell_ixs,
                bounds,
            }
        })
        .collect()
}

/// Replace filename-unsafe characters so a batch label like
/// `sample 1/slide A` doesn't blow up the output path.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            c => c,
        })
        .collect()
}
