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
use parquet::record::{Row, RowAccessor};
use std::fs::File;
use std::path::Path;

/// Read a column as a label string, regardless of physical type.
///
/// `pinto svd --coord` writes whatever coordinate columns the user
/// supplied as FLOAT, including a "batch" column that's actually a
/// numeric batch index. The plot pipeline still wants a `Box<str>`
/// label, so accept any numeric/string type and stringify.
pub(crate) fn row_label(row: &Row, idx: usize) -> anyhow::Result<Box<str>> {
    if let Ok(s) = row.get_string(idx) {
        return Ok(s.clone().into_boxed_str());
    }
    if let Ok(v) = row.get_float(idx) {
        return Ok(stringify_numeric(v as f64));
    }
    if let Ok(v) = row.get_double(idx) {
        return Ok(stringify_numeric(v));
    }
    if let Ok(v) = row.get_long(idx) {
        return Ok(v.to_string().into_boxed_str());
    }
    if let Ok(v) = row.get_int(idx) {
        return Ok(v.to_string().into_boxed_str());
    }
    anyhow::bail!("column {idx} has no string/numeric value");
}

fn stringify_numeric(v: f64) -> Box<str> {
    if v.is_finite() && v.fract() == 0.0 {
        format!("{}", v as i64).into_boxed_str()
    } else {
        format!("{v}").into_boxed_str()
    }
}

/// Read a numeric column as `f32`, accepting FLOAT or DOUBLE.
///
/// `pinto svd --coord` writes coordinates as FLOAT; coord_pairs files
/// produced by other paths (R/data.table) come in as DOUBLE. The plot
/// pipeline is single-precision throughout, so we narrow on read.
fn row_f32(row: &Row, idx: usize) -> anyhow::Result<f32> {
    if let Ok(v) = row.get_float(idx) {
        return Ok(v);
    }
    if let Ok(v) = row.get_double(idx) {
        return Ok(v as f32);
    }
    anyhow::bail!("column {idx} is not a numeric type")
}

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
/// `coord_columns`, when supplied (typically from `metadata.json`'s
/// `outputs.coord_columns`), names the bare coord basenames in `(x, y)`
/// order — e.g. `["pxl_row_in_fullres", "pxl_col_in_fullres"]`. Pass
/// `None` to fall back to the legacy auto-discovery (first two paired
/// `left_*` / `right_*` columns by schema order), which is correct for
/// pinto-written files but brittle when users splice extra columns in.
pub fn read_cells_from_coord_pairs(
    path: &Path,
    coord_columns: Option<&[String]>,
) -> anyhow::Result<CellTable> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?;
    let fields = peek_parquet_field_names(path_str)?;

    let (x_bare, y_bare) = match coord_columns {
        Some(cols) if cols.len() >= 2 => (cols[0].clone(), cols[1].clone()),
        Some(_) | None => {
            // Legacy fallback: scan for paired left_* columns and pick
            // the first two as (x, y). Excludes the `left_cell` /
            // `left_batch` non-coord columns.
            let left_coords: Vec<Box<str>> = fields
                .iter()
                .filter(|f| {
                    f.starts_with("left_")
                        && f.as_ref() != "left_cell"
                        && f.as_ref() != "left_batch"
                })
                .cloned()
                .collect();
            if left_coords.len() < 2 {
                anyhow::bail!(
                    "coord_pairs.parquet {path:?} has fewer than 2 coordinate columns \
                     (needs left_x + left_y pair) and metadata.json carried no \
                     coord_columns hint. Was this run fit without --coord?"
                );
            }
            (
                strip_left(&left_coords[0]).to_string(),
                strip_left(&left_coords[1]).to_string(),
            )
        }
    };

    let x_col_left: Box<str> = format!("left_{x_bare}").into_boxed_str();
    let y_col_left: Box<str> = format!("left_{y_bare}").into_boxed_str();
    let x_col_right: Box<str> = format!("right_{x_bare}").into_boxed_str();
    let y_col_right: Box<str> = format!("right_{y_bare}").into_boxed_str();

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
            let name = row_label(&row, ic)?;
            if index.contains_key(&name) {
                continue;
            }
            let x = row_f32(&row, ix)?;
            let y = row_f32(&row, iy)?;
            let batch = match ib {
                Some(b) => Some(row_label(&row, b)?),
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
    let coord_col_names = vec![x_bare.into_boxed_str(), y_bare.into_boxed_str()];
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
/// - `pinto lc`: columns named `"0"`, `"1"`, …, `"{K-1}"` plus optional
///   `entropy`. No `cluster`, no coord trailer.
/// - `pinto prop`: columns `propensity_0 … propensity_{K-1}, cluster,
///   entropy`, plus an optional coord trailer (e.g. `pxl_row_in_fullres`,
///   `pxl_col_in_fullres`).
/// - `pinto dsvd`: columns `0 … K-1, cluster, entropy`.
///
/// This reader keeps every float column as a propensity slot *except*
/// names that appear in `exclude_cols`, the explicit `cluster` column,
/// and the optional `entropy` column (each consumed separately).
/// Returns `(propensity[N×K], cluster[N], entropy[N] when present, cell_names[N])`.
pub type PropensityRead = (Mat, Vec<i64>, Option<Vec<f32>>, Vec<Box<str>>);

pub fn read_propensity(
    path: &Path,
    exclude_cols: &HashSet<Box<str>>,
) -> anyhow::Result<PropensityRead> {
    let MatWithNames { rows, cols, mat } = Mat::from_parquet(
        path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?,
    )?;

    let mut prop_idx: Vec<usize> = Vec::new();
    let mut cluster_idx: Option<usize> = None;
    let mut entropy_idx: Option<usize> = None;
    for (j, name) in cols.iter().enumerate() {
        match name.as_ref() {
            "cluster" => cluster_idx = Some(j),
            "entropy" => entropy_idx = Some(j),
            _ if !exclude_cols.contains(name) => prop_idx.push(j),
            _ => {}
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

    let entropy = entropy_idx.map(|j| (0..n).map(|i| mat[(i, j)]).collect::<Vec<_>>());

    Ok((prop, cluster, entropy, rows))
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
        let l = row_label(&row, li)?;
        let r = row_label(&row, ri)?;
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

/// Read a gene_community parquet: G × K. Returns (mat, gene_names).
///
/// `pinto lc` writes this file in *melted* form (one row per
/// gene-community pair, with columns `gene`, `community`, `mean`, `sd`,
/// `log_mean`, `log_sd`). We pivot the `mean` column back to a wide
/// G × K matrix here so downstream code can index `gt[(g, k)]` as
/// "posterior mean for gene g in community k". Communities are sorted
/// numerically by their string label (the writer emits `"0".."K-1"`).
pub fn read_gene_community(path: &Path) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
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
    let gene_idx = *name_to_idx
        .get("gene")
        .ok_or_else(|| anyhow::anyhow!("{path_str}: missing `gene` column"))?;
    let community_idx = *name_to_idx
        .get("community")
        .ok_or_else(|| anyhow::anyhow!("{path_str}: missing `community` column"))?;
    let mean_idx = *name_to_idx
        .get("mean")
        .ok_or_else(|| anyhow::anyhow!("{path_str}: missing `mean` column"))?;

    let mut gene_pos: HashMap<Box<str>, usize> = HashMap::default();
    let mut gene_names: Vec<Box<str>> = Vec::new();
    let mut community_pos: HashMap<Box<str>, usize> = HashMap::default();
    let mut community_labels: Vec<Box<str>> = Vec::new();
    let mut triples: Vec<(usize, usize, f32)> = Vec::new();

    for record in reader.get_row_iter(None)? {
        let row = record?;
        let gene = row.get_string(gene_idx)?.clone().into_boxed_str();
        let community = row.get_string(community_idx)?.clone().into_boxed_str();
        let mean = row
            .get_float(mean_idx)
            .or_else(|_| row.get_double(mean_idx).map(|v| v as f32))?;
        let g_pos = *gene_pos.entry(gene.clone()).or_insert_with(|| {
            gene_names.push(gene.clone());
            gene_names.len() - 1
        });
        let c_pos = *community_pos.entry(community.clone()).or_insert_with(|| {
            community_labels.push(community.clone());
            community_labels.len() - 1
        });
        triples.push((g_pos, c_pos, mean));
    }

    // Sort communities numerically by their string label so column k
    // in the returned matrix corresponds to community `k` in the
    // propensity parquet (writer emits "0".."K-1").
    let mut col_order: Vec<usize> = (0..community_labels.len()).collect();
    col_order.sort_by_key(|&i| community_labels[i].parse::<i64>().unwrap_or(i64::MAX));
    let new_index: HashMap<usize, usize> = col_order
        .iter()
        .enumerate()
        .map(|(new_i, &old_i)| (old_i, new_i))
        .collect();

    let n_genes = gene_names.len();
    let n_communities = community_labels.len();
    let mut mat = Mat::zeros(n_genes, n_communities);
    for (g, c_old, v) in triples {
        let c_new = new_index[&c_old];
        mat[(g, c_new)] = v;
    }
    Ok((mat, gene_names))
}
