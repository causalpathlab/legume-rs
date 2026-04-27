//! Build the single plotting `CoreSpec` (`name = "all"`) covering
//! every cell. The render pipeline is single-panel; per-batch
//! disambiguation lives in filenames where needed (e.g. LR overlays).

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

/// Build a single core covering all cells. Returns an empty vec if
/// the cell count is below `min_core_cells`.
pub fn partition_cells(cells: &CellTable, min_core_cells: usize, coord_clip: f32) -> Vec<CoreSpec> {
    let cell_ixs: Vec<usize> = (0..cells.n()).collect();
    if cell_ixs.len() < min_core_cells.max(1) {
        return Vec::new();
    }
    let xs: Vec<f32> = cell_ixs.iter().map(|&i| cells.coords[i].0).collect();
    let ys: Vec<f32> = cell_ixs.iter().map(|&i| cells.coords[i].1).collect();
    let (xmin, xmax) = viridis::robust_range(&xs, coord_clip);
    let (ymin, ymax) = viridis::robust_range(&ys, coord_clip);
    let bounds = DataBounds::from_minmax(xmin, xmax, ymin, ymax);
    vec![CoreSpec {
        name: "all".to_string(),
        cell_ixs,
        bounds,
    }]
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
