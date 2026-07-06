//! Loading for `faba assoc`: the lineage's per-cell pseudotime + branch, and the
//! modality site matrix with its two channels paired into per-cell (edited, total).

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;

use data_beans::hdf5_io::resolve_backend_file;
use data_beans::sparse_io::open_sparse_matrix;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use super::Modality;
use faba::feature_name::parse_feature_row;

/// Per-cell lineage: a common pseudotime axis + primary branch, in cell order.
pub struct Lineage {
    pub cell_names: Vec<Box<str>>,
    pub pseudotime: Vec<f32>,
    pub branch: Vec<usize>,
    pub n_branches: usize,
}

/// Read `{prefix}.pseudotime.parquet` (columns `pseudotime`, `branch`; rows = cells).
pub fn load_lineage(prefix: &str) -> Result<Lineage> {
    let path = format!("{prefix}.pseudotime.parquet");
    let m = DMatrix::<f32>::from_parquet(&path).with_context(|| format!("reading {path}"))?;
    let col = |name: &str| {
        m.cols
            .iter()
            .position(|c| c.as_ref() == name)
            .with_context(|| format!("{path} missing column '{name}'"))
    };
    let (cpt, cbr) = (col("pseudotime")?, col("branch")?);
    let n = m.mat.nrows();
    let pseudotime: Vec<f32> = (0..n).map(|i| m.mat[(i, cpt)]).collect();
    let branch: Vec<usize> = (0..n).map(|i| m.mat[(i, cbr)].max(0.0) as usize).collect();
    let n_branches = branch.iter().copied().max().map_or(0, |x| x + 1);
    Ok(Lineage {
        cell_names: m.rows,
        pseudotime,
        branch,
        n_branches,
    })
}

/// A modality site with per-lineage-cell edited `k` and total `n` (= edited+unedited),
/// aligned to the lineage cell order (0 where the cell is absent from the matrix).
pub struct Site {
    pub gene: Box<str>,
    pub subunit: Box<str>,
    pub k: Vec<u32>,
    pub n: Vec<u32>,
}

/// Open the modality site matrices, pair the two channels per (gene, subunit), and
/// return per-site (k, n) vectors aligned to `cell_names`. Only sites with both
/// channels present are returned. Multiple files are concatenated (per-file sites).
pub fn load_sites(
    paths: &[String],
    modality: Modality,
    cell_names: &[Box<str>],
) -> Result<Vec<Site>> {
    let cell_idx: FxHashMap<&str, usize> = cell_names
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_ref(), i))
        .collect();
    let ncell = cell_names.len();
    let (pos_ch, neg_ch) = modality.channels();
    let tok = modality.token();

    let mut sites: Vec<Site> = Vec::new();
    for path in paths {
        let (backend, resolved) = resolve_backend_file(path, None)
            .with_context(|| format!("resolving backend for {path}"))?;
        let data =
            open_sparse_matrix(&resolved, &backend).with_context(|| format!("opening {path}"))?;
        let row_names = data.row_names()?;
        let col_names = data.column_names()?;
        let col_to_cell: Vec<Option<usize>> = col_names
            .iter()
            .map(|bc| cell_idx.get(bc.as_ref()).copied())
            .collect();

        // Group rows by (gene, subunit); record the pos/neg channel row indices.
        // Value: (positive-channel row idx, negative-channel row idx).
        type ChannelRows = FxHashMap<(Box<str>, Box<str>), (Option<usize>, Option<usize>)>;
        let mut groups: ChannelRows = FxHashMap::default();
        for (ri, name) in row_names.iter().enumerate() {
            let Some(fr) = parse_feature_row(name) else {
                continue;
            };
            if fr.modality != tok {
                continue;
            }
            let sub = fr.subunit.unwrap_or("");
            let e = groups
                .entry((fr.gene.into(), sub.into()))
                .or_insert((None, None));
            if fr.channel == pos_ch {
                e.0 = Some(ri);
            } else if fr.channel == neg_ch {
                e.1 = Some(ri);
            }
        }

        // Sites with both channels → request their rows in one read.
        let mut meta: Vec<(Box<str>, Box<str>)> = Vec::new();
        let mut req_rows: Vec<usize> = Vec::new();
        let mut local_map: Vec<(usize, bool)> = Vec::new(); // (local site idx, is_positive)
        for ((gene, sub), (pos, neg)) in groups {
            if let (Some(p), Some(m)) = (pos, neg) {
                let li = meta.len();
                meta.push((gene, sub));
                req_rows.push(p);
                local_map.push((li, true));
                req_rows.push(m);
                local_map.push((li, false));
            }
        }
        if meta.is_empty() {
            continue;
        }

        let (_nr, _nc, triplets) = data.read_triplets_by_rows(req_rows)?;
        let mut kk = vec![vec![0u32; ncell]; meta.len()];
        let mut nn = vec![vec![0u32; ncell]; meta.len()];
        for (row, col, val) in triplets {
            let (li, is_pos) = local_map[row as usize];
            if let Some(ci) = col_to_cell[col as usize] {
                let v = val.max(0.0).round() as u32;
                nn[li][ci] += v;
                if is_pos {
                    kk[li][ci] += v;
                }
            }
        }
        for (li, (gene, sub)) in meta.into_iter().enumerate() {
            sites.push(Site {
                gene,
                subunit: sub,
                k: std::mem::take(&mut kk[li]),
                n: std::mem::take(&mut nn[li]),
            });
        }
    }
    Ok(sites)
}
