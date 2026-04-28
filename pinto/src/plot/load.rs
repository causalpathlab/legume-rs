//! Parquet readers for the pinto-plot pipeline.
//!
//! All readers here are thin wrappers over `matrix-util` primitives so
//! we share one code path with senna / the rest of pinto. The only
//! pinto-specific piece is [`read_cells_from_coord_pairs`], which
//! dedupes `coord_pairs.parquet` into a per-cell table `(name, x, y,
//! optional batch)` — the primary source of truth since pinto has no
//! run manifest.

use crate::util::common::*;
use data_beans::hdf5_io::strip_backend_suffix;
use matrix_util::common_io::basename;
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

    // Pull propensity columns by NAME. The current writer emits
    // `C{c}` (e.g. "C0","C1",…,"C{K-1}") so the column name itself
    // declares "this is community c". Older parquets that wrote bare
    // integer names ("0","1",…) are still accepted for backwards-compat.
    // This makes "column j ↔ community j" an explicit, checked invariant
    // rather than a positional convention.
    let mut cluster_idx: Option<usize> = None;
    let mut entropy_idx: Option<usize> = None;
    let mut prop_named: Vec<(i64, usize)> = Vec::new();
    for (j, name) in cols.iter().enumerate() {
        match name.as_ref() {
            "cluster" => cluster_idx = Some(j),
            "entropy" => entropy_idx = Some(j),
            _ if exclude_cols.contains(name) => {}
            n => match parse_community_col_name(n) {
                Some(c) => prop_named.push((c, j)),
                None => anyhow::bail!(
                    "{path:?}: propensity column {n:?} is not a community ID. \
                     Expected names \"C0\",\"C1\",…,\"C{{K-1}}\" plus optional \"entropy\"/coord cols."
                ),
            },
        }
    }

    if prop_named.is_empty() {
        anyhow::bail!("{path:?}: no propensity columns found (all columns excluded)");
    }

    // Pin "matrix column j ↔ community j" by mapping each named column
    // to its parsed ID. Missing intermediate IDs (e.g. labels {0,1,3,5}
    // → 2 and 4 absent) are tolerated by zero-filling those columns and
    // warning, so K stays = max_id+1 and every other plot's
    // `colors.color(c)` keeps working for the present communities. Negative
    // IDs are rejected — the writer never emits them, so seeing one means
    // genuine schema corruption.
    prop_named.sort_by_key(|&(c, _)| c);
    if let Some(&(c0, _)) = prop_named.first() {
        if c0 < 0 {
            anyhow::bail!(
                "{path:?}: propensity has negative community ID {c0}; expected non-negative integers."
            );
        }
    }
    let max_id = prop_named.last().map(|&(c, _)| c).unwrap_or(-1);
    let k = (max_id + 1).max(0) as usize;
    let present: HashSet<i64> = prop_named.iter().map(|&(c, _)| c).collect();
    let missing: Vec<i64> = (0..k as i64).filter(|c| !present.contains(c)).collect();
    if !missing.is_empty() {
        log::warn!(
            "{path:?}: propensity is missing community columns {missing:?} \
             (have 0..{} with gaps); zero-filling so plot indices stay aligned.",
            k - 1,
        );
    }

    let n = mat.nrows();
    let mut prop = Mat::zeros(n, k);
    // Source-column index per *present* community ID. Index by community
    // ID so the loop below can populate non-contiguous IDs directly.
    let mut src_for: Vec<Option<usize>> = vec![None; k];
    for &(c, j) in &prop_named {
        src_for[c as usize] = Some(j);
    }
    for (out_j, slot) in src_for.iter().enumerate() {
        if let Some(src_j) = *slot {
            for i in 0..n {
                prop[(i, out_j)] = mat[(i, src_j)];
            }
        }
    }
    // For argmax fallback below, only consider present columns so that
    // an all-zero (missing) community can't accidentally become the
    // argmax when a row has no signal.
    let prop_idx: Vec<usize> = prop_named.iter().map(|&(_, j)| j).collect();

    let cluster = match cluster_idx {
        Some(j) => (0..n).map(|i| mat[(i, j)] as i64).collect::<Vec<_>>(),
        None => (0..n)
            .map(|i| {
                if prop_idx.is_empty() {
                    -1
                } else {
                    let rank = argmax_row(&mat, i, &prop_idx) as usize;
                    prop_named[rank].0
                }
            })
            .collect(),
    };

    let entropy = entropy_idx.map(|j| (0..n).map(|i| mat[(i, j)]).collect::<Vec<_>>());

    Ok((prop, cluster, entropy, rows))
}

/// Parse a community-column / community-label string into its integer ID.
///
/// Accepts the current `C{c}` schema and two legacy forms — bare integer
/// (`"5"`, older `pinto lc`) and `propensity_{c}` (older `pinto propensity`)
/// — so existing parquets on disk still load. Returns `None` for anything
/// else (e.g. `"entropy"`, coord names, malformed labels).
fn parse_community_col_name(name: &str) -> Option<i64> {
    if let Some(rest) = name.strip_prefix('C') {
        return rest.parse::<i64>().ok();
    }
    if let Some(rest) = name.strip_prefix("propensity_") {
        return rest.parse::<i64>().ok();
    }
    name.parse::<i64>().ok()
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

    // Map each label to its parsed community ID, then place column j of
    // the returned matrix at community j (matching `read_propensity`'s
    // invariant). Missing intermediate IDs are zero-filled with a warn,
    // not a hard error, so a clustering that drops a community between
    // levels still produces a usable plot.
    let mut parsed: Vec<(i64, usize)> = community_labels
        .iter()
        .enumerate()
        .map(|(i, lab)| {
            let c = parse_community_col_name(lab).ok_or_else(|| {
                anyhow::anyhow!(
                    "{path_str}: gene_community `community` label {lab:?} is not a community ID \
                     (expected \"C{{c}}\" or bare integer)."
                )
            })?;
            Ok::<_, anyhow::Error>((c, i))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    parsed.sort_by_key(|&(c, _)| c);
    if let Some(&(c0, _)) = parsed.first() {
        if c0 < 0 {
            anyhow::bail!(
                "{path_str}: gene_community has negative community ID {c0}; expected non-negative integers."
            );
        }
    }
    let max_id = parsed.last().map(|&(c, _)| c).unwrap_or(-1);
    let n_communities = (max_id + 1).max(0) as usize;
    let present: HashSet<i64> = parsed.iter().map(|&(c, _)| c).collect();
    let missing: Vec<i64> = (0..n_communities as i64)
        .filter(|c| !present.contains(c))
        .collect();
    if !missing.is_empty() {
        log::warn!(
            "{path_str}: gene_community is missing communities {missing:?} \
             (have 0..{} with gaps); zero-filling so plot indices stay aligned.",
            n_communities.saturating_sub(1),
        );
    }
    // old c_pos (insertion order) → new column = community ID
    let new_index: HashMap<usize, usize> = parsed
        .iter()
        .map(|&(c, old_i)| (old_i, c as usize))
        .collect();

    let n_genes = gene_names.len();
    let mut mat = Mat::zeros(n_genes, n_communities);
    for (g, c_old, v) in triples {
        let c_new = new_index[&c_old];
        mat[(g, c_new)] = v;
    }
    Ok((mat, gene_names))
}

/// Map "0","1",… numeric batch labels to the friendly basenames of the
/// upstream input files (`metadata.json::data_files`).
///
/// `pinto svd` assigns string-of-integer batch labels (`"0"`, `"1"`, …)
/// when the user passes multiple data files without an explicit
/// `--batch-files`, and these labels propagate through
/// `coord_pairs.parquet` into the plot. The user-facing batch name is
/// the input file's basename (without the `.zarr` / `.h5` / `.zarr.zip`
/// extension), which is what the data-beans merger uses internally —
/// reusing that here keeps the plot dirs consistent with how the
/// upstream code names batches.
///
/// Returns the resolved name table when all current labels parse as
/// non-negative integers `< data_files.len()`. `None` means the labels
/// are already strings (user supplied real batch names) or the
/// metadata is missing — no remap is applied in either case.
pub fn resolve_batch_name_map(
    cells_batches: &[Box<str>],
    data_files: &[String],
) -> Option<Vec<Box<str>>> {
    if data_files.is_empty() {
        return None;
    }
    // Every existing label must be a non-negative integer that indexes
    // into `data_files`; one non-numeric label is enough to bail (the
    // user already supplied human-readable batch names).
    for b in cells_batches {
        let parsed: Result<usize, _> = b.parse();
        match parsed {
            Ok(i) if i < data_files.len() => {}
            _ => return None,
        }
    }
    Some(unique_batch_names_from_data_files(data_files))
}

/// Apply the index→name table from `resolve_batch_name_map` to a slice
/// of batch labels. Caller owns the slice; labels that don't parse are
/// left as-is (defensive — `resolve_batch_name_map` already guards).
pub fn remap_batch_labels(labels: &mut [Box<str>], name_map: &[Box<str>]) {
    for b in labels.iter_mut() {
        if let Ok(i) = b.parse::<usize>() {
            if let Some(friendly) = name_map.get(i) {
                *b = friendly.clone();
            }
        }
    }
}

/// Mirror of `data_beans::handlers::merging::generate_unique_batch_names`,
/// reimplemented here because that helper sits inside a non-public
/// `handlers` module. Strips backend suffixes from each file's
/// basename and disambiguates duplicates with a `_{n}` counter so the
/// returned vector is index-aligned with `data_files`.
fn unique_batch_names_from_data_files(data_files: &[String]) -> Vec<Box<str>> {
    let bare: Vec<Box<str>> = data_files
        .iter()
        .map(|f| {
            basename(f)
                .map(|b| {
                    let stripped = strip_backend_suffix(&b);
                    if stripped.len() == b.len() {
                        b
                    } else {
                        stripped.into()
                    }
                })
                .unwrap_or_else(|_| f.clone().into_boxed_str())
        })
        .collect();
    let mut counts: HashMap<Box<str>, usize> = HashMap::default();
    for n in &bare {
        *counts.entry(n.clone()).or_insert(0) += 1;
    }
    let mut counters: HashMap<Box<str>, usize> = HashMap::default();
    bare.iter()
        .map(|n| match counts.get(n).copied().unwrap_or(0) {
            0 | 1 => n.clone(),
            _ => {
                let c = counters.entry(n.clone()).or_insert(0);
                let out = format!("{n}_{c}").into_boxed_str();
                *c += 1;
                out
            }
        })
        .collect()
}
