//! Parquet readers for the pinto-plot pipeline.
//!
//! All readers here are thin wrappers over `matrix-util` primitives so
//! we share one code path with senna / the rest of pinto. The only
//! pinto-specific piece is [`read_cells_from_coord_pairs`], which
//! dedupes `coord_pairs.parquet` into a per-cell table `(name, x, y,
//! optional batch)` — the primary source of truth since pinto has no
//! run manifest.

use crate::util::common::*;
use matrix_util::parquet::peek_parquet_field_names;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use std::path::Path;

/// One row per cell. `batch` is `None` if the fit was single-batch
/// (i.e. `coord_pairs.parquet` lacks a `left_batch` column).
pub struct CellTable {
    /// Cell barcode, unique and stable-ordered.
    pub names: Vec<Box<str>>,
    /// (x, y) coordinate per cell.
    pub coords: Vec<(f32, f32)>,
    /// Optional batch label per cell.
    pub batches: Option<Vec<Box<str>>>,
    /// `name → index into names/coords/batches`
    pub index: HashMap<Box<str>, usize>,
    /// Bare coordinate column names (without `left_`/`right_` prefix).
    /// Pass-through from `coord_pairs.parquet` so downstream readers
    /// (e.g. `read_propensity`) can exclude coord trailers that
    /// `pinto prop` may append to `.propensity.parquet`.
    pub coord_col_names: Vec<Box<str>>,
}

impl CellTable {
    pub fn n(&self) -> usize {
        self.names.len()
    }
}

/// Read `{prefix}.coord_pairs.parquet`, union left+right, dedupe.
///
/// We insist on finding at least `left_*` / `right_*` numeric coord
/// columns paired by suffix — anything else is a fit without `--coord`,
/// which we reject at plot time.
pub fn read_cells_from_coord_pairs(path: &Path) -> anyhow::Result<CellTable> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?;
    let fields = peek_parquet_field_names(path_str)?;

    // Locate the first paired left_*/right_* coord column (besides
    // left_batch / left_cell). We only need one axis pair for x; we
    // pick the first two paired columns as (x, y).
    let left_coords: Vec<Box<str>> = fields
        .iter()
        .filter(|f| {
            f.starts_with("left_") && f.as_ref() != "left_cell" && f.as_ref() != "left_batch"
        })
        .cloned()
        .collect();

    if left_coords.len() < 2 {
        anyhow::bail!(
            "coord_pairs.parquet {path:?} has fewer than 2 coordinate columns \
             (needs left_x + left_y pair). Was this run fit without --coord?"
        );
    }

    let x_col_left = left_coords[0].clone();
    let y_col_left = left_coords[1].clone();
    let x_col_right: Box<str> = format!("right_{}", strip_left(&x_col_left)).into_boxed_str();
    let y_col_right: Box<str> = format!("right_{}", strip_left(&y_col_left)).into_boxed_str();

    let has_batch = fields.iter().any(|f| f.as_ref() == "left_batch")
        && fields.iter().any(|f| f.as_ref() == "right_batch");

    // Read row-by-row: small enough (at most ~few million edges) and
    // avoids teaching `Mat::from_parquet` about mixed string+float.
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let name_to_idx: HashMap<Box<str>, usize> = schema
        .get_fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().to_string().into_boxed_str(), i))
        .collect();

    let fetch = |n: &str| -> anyhow::Result<usize> {
        name_to_idx
            .get(n)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("column {n} missing in {path:?}"))
    };

    let li_cell = fetch("left_cell")?;
    let ri_cell = fetch("right_cell")?;
    let li_x = fetch(&x_col_left)?;
    let li_y = fetch(&y_col_left)?;
    let ri_x = fetch(&x_col_right)?;
    let ri_y = fetch(&y_col_right)?;
    let (li_b, ri_b) = if has_batch {
        (Some(fetch("left_batch")?), Some(fetch("right_batch")?))
    } else {
        (None, None)
    };

    let mut index: HashMap<Box<str>, usize> = HashMap::default();
    let mut names: Vec<Box<str>> = Vec::new();
    let mut coords: Vec<(f32, f32)> = Vec::new();
    let mut batches: Vec<Box<str>> = Vec::new();

    let row_iter = reader.get_row_iter(None)?;
    for record in row_iter {
        let row = record?;
        for (ic, ix, iy, ib) in [(li_cell, li_x, li_y, li_b), (ri_cell, ri_x, ri_y, ri_b)] {
            let name = row.get_string(ic)?.clone().into_boxed_str();
            if index.contains_key(&name) {
                continue;
            }
            let x = row.get_float(ix)?;
            let y = row.get_float(iy)?;
            let batch = match ib {
                Some(b) => Some(row.get_string(b)?.clone().into_boxed_str()),
                None => None,
            };
            index.insert(name.clone(), names.len());
            names.push(name);
            coords.push((x, y));
            if let Some(b) = batch {
                batches.push(b);
            }
        }
    }

    let batches = if has_batch { Some(batches) } else { None };
    let coord_col_names = vec![
        strip_left(&x_col_left).to_string().into_boxed_str(),
        strip_left(&y_col_left).to_string().into_boxed_str(),
    ];
    Ok(CellTable {
        names,
        coords,
        batches,
        index,
        coord_col_names,
    })
}

fn strip_left(col: &str) -> &str {
    col.strip_prefix("left_").unwrap_or(col)
}

/// Read a propensity parquet. Schemas differ across pinto variants:
///
/// - `pinto lc`: columns named `"0"`, `"1"`, …, `"{K-1}"` (plain numeric
///   topic ids, no `cluster`, no coord trailer).
/// - `pinto prop`: columns `propensity_0 … propensity_{K-1}, cluster`,
///   plus an optional coord trailer (e.g. `pxl_row_in_fullres`,
///   `pxl_col_in_fullres`).
///
/// This reader keeps every float column as a propensity slot *except*
/// names that appear in `exclude_cols` (used to drop coord / metadata
/// columns) and the explicit `cluster` column (consumed separately).
/// Returns `(propensity[N×K], cluster[N], cell_names[N])`.
pub fn read_propensity(
    path: &Path,
    exclude_cols: &HashSet<Box<str>>,
) -> anyhow::Result<(Mat, Vec<i64>, Vec<Box<str>>)> {
    let MatWithNames { rows, cols, mat } = Mat::from_parquet(
        path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?,
    )?;

    let mut prop_idx: Vec<usize> = Vec::new();
    let mut cluster_idx: Option<usize> = None;
    for (j, name) in cols.iter().enumerate() {
        if name.as_ref() == "cluster" {
            cluster_idx = Some(j);
        } else if !exclude_cols.contains(name) {
            prop_idx.push(j);
        }
    }

    if prop_idx.is_empty() {
        anyhow::bail!("{path:?}: no propensity columns found (all columns excluded)");
    }

    let n = mat.nrows();
    let k = prop_idx.len();
    let mut prop = Mat::zeros(n, k);
    for (out_j, &src_j) in prop_idx.iter().enumerate() {
        for i in 0..n {
            prop[(i, out_j)] = mat[(i, src_j)];
        }
    }

    let cluster = match cluster_idx {
        Some(j) => (0..n).map(|i| mat[(i, j)] as i64).collect::<Vec<_>>(),
        None => (0..n).map(|i| argmax_row(&mat, i, &prop_idx)).collect(),
    };

    Ok((prop, cluster, rows))
}

fn argmax_row(mat: &Mat, row: usize, cols: &[usize]) -> i64 {
    let mut best_j = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (rank, &j) in cols.iter().enumerate() {
        let v = mat[(row, j)];
        if v > best_v {
            best_v = v;
            best_j = rank;
        }
    }
    best_j as i64
}

/// A cell-pair edge: (left_cell_name, right_cell_name).
pub type EdgePair = (Box<str>, Box<str>);

/// Read a link-community parquet: E rows, each `(left_cell, right_cell,
/// community)`. Returns `(pairs, community)` with parallel lengths.
pub fn read_link_community(path: &Path) -> anyhow::Result<(Vec<EdgePair>, Vec<i64>)> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?;
    let file = File::open(path_str)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let name_to_idx: HashMap<Box<str>, usize> = schema
        .get_fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().to_string().into_boxed_str(), i))
        .collect();

    let li = *name_to_idx
        .get("left_cell")
        .ok_or_else(|| anyhow::anyhow!("{path:?}: missing left_cell"))?;
    let ri = *name_to_idx
        .get("right_cell")
        .ok_or_else(|| anyhow::anyhow!("{path:?}: missing right_cell"))?;
    let ci = *name_to_idx
        .get("community")
        .ok_or_else(|| anyhow::anyhow!("{path:?}: missing community column"))?;

    let mut pairs: Vec<(Box<str>, Box<str>)> = Vec::new();
    let mut community: Vec<i64> = Vec::new();
    for record in reader.get_row_iter(None)? {
        let row = record?;
        let l = row.get_string(li)?.clone().into_boxed_str();
        let r = row.get_string(ri)?.clone().into_boxed_str();
        // Schema may write community as float or int depending on
        // which pinto variant produced it; try float first.
        let c: i64 = row
            .get_float(ci)
            .map(|v| v as i64)
            .or_else(|_| row.get_double(ci).map(|v| v as i64))
            .or_else(|_| row.get_long(ci))
            .or_else(|_| row.get_int(ci).map(|v| v as i64))?;
        pairs.push((l, r));
        community.push(c);
    }
    Ok((pairs, community))
}

/// Read a gene_topic parquet: G × K. Returns (mat, gene_names).
pub fn read_gene_topic(path: &Path) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
    let MatWithNames { rows, mat, .. } = Mat::from_parquet(
        path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?,
    )?;
    Ok((mat, rows))
}
