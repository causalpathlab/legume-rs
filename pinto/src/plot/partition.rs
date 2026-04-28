//! Build per-image plotting `CoreSpec`s. When `coord_pairs.parquet`
//! carries a `left_batch`/`right_batch` column, each batch gets its own
//! core (own cell list, own data bounds) so multi-image runs don't
//! collapse different tissue sections onto a single oversized frame.
//! Single-batch runs (no batch column) collapse to a single `name =
//! "all"` core.

use super::load::CellTable;
use super::viridis;
use plot_utils::rasterize::DataBounds;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct CoreSpec {
    /// Sanitized label used in output filenames / per-batch subdirs.
    pub name: String,
    /// Original (unsanitized) batch label when this core represents a
    /// single batch; `None` for the synthetic single-batch `"all"` core.
    /// Used to (a) decide whether to nest output under a per-batch
    /// subdir and (b) match against `LrResult.batch` in the overlay.
    pub batch_label: Option<Box<str>>,
    /// Indices into CellTable.names / coords for this core.
    pub cell_ixs: Vec<usize>,
    /// Robust-clipped data bounds.
    pub bounds: DataBounds,
}

impl CoreSpec {
    pub fn n(&self) -> usize {
        self.cell_ixs.len()
    }

    /// Nest `base` under this core's name when it owns a batch label;
    /// pass `base` through unchanged for the synthetic single-batch
    /// `"all"` core. Single source of truth for the "per-batch runs add
    /// one path level" rule used by every plot kind.
    pub fn subdir_in(&self, base: &Path) -> PathBuf {
        if self.batch_label.is_some() {
            base.join(&self.name)
        } else {
            base.to_path_buf()
        }
    }
}

/// One core per batch when `cells.batches` is populated; otherwise a
/// single `"all"` core covering every cell. Cores below
/// `min_core_cells` are dropped with an info log so a tiny outlier
/// section doesn't crash the run.
pub fn partition_cells(cells: &CellTable, min_core_cells: usize, coord_clip: f32) -> Vec<CoreSpec> {
    let make_core =
        |name: String, batch_label: Option<Box<str>>, cell_ixs: Vec<usize>| -> CoreSpec {
            let xs: Vec<f32> = cell_ixs.iter().map(|&i| cells.coords[i].0).collect();
            let ys: Vec<f32> = cell_ixs.iter().map(|&i| cells.coords[i].1).collect();
            let (xmin, xmax) = viridis::robust_range(&xs, coord_clip);
            let (ymin, ymax) = viridis::robust_range(&ys, coord_clip);
            let bounds = DataBounds::from_minmax(xmin, xmax, ymin, ymax);
            CoreSpec {
                name,
                batch_label,
                cell_ixs,
                bounds,
            }
        };

    match &cells.batches {
        None => {
            let cell_ixs: Vec<usize> = (0..cells.n()).collect();
            if cell_ixs.len() < min_core_cells.max(1) {
                return Vec::new();
            }
            vec![make_core("all".to_string(), None, cell_ixs)]
        }
        Some(batches) => {
            // Preserve first-seen batch order so the output dir listing
            // tracks the input parquet's row order rather than hash
            // iteration order. `get_mut` on hit avoids cloning the
            // batch label per cell — only on first sight do we pay the
            // two `Box<str>` clones (one for `order`, one for the map
            // key).
            let mut order: Vec<Box<str>> = Vec::new();
            let mut by_batch: HashMap<Box<str>, Vec<usize>> = HashMap::new();
            for (i, b) in batches.iter().enumerate() {
                if let Some(v) = by_batch.get_mut(b) {
                    v.push(i);
                } else {
                    let key = b.clone();
                    order.push(key.clone());
                    by_batch.insert(key, vec![i]);
                }
            }
            let mut cores: Vec<CoreSpec> = Vec::with_capacity(order.len());
            for b in order {
                let cell_ixs = by_batch.remove(&b).expect("batch key present");
                if cell_ixs.len() < min_core_cells.max(1) {
                    log::info!(
                        "partition: batch {b:?} has {} cells (< --min-core-cells {min_core_cells}); skipping",
                        cell_ixs.len(),
                    );
                    continue;
                }
                let name = sanitize(&b);
                cores.push(make_core(name, Some(b), cell_ixs));
            }
            cores
        }
    }
}

/// Replace filename-unsafe characters so a batch label like
/// `sample 1/slide A` doesn't blow up the output path.
pub fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            c => c,
        })
        .collect()
}
