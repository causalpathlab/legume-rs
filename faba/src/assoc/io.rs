//! Loading for `faba assoc`: the lineage's per-cell pseudotime + branch, and the
//! modality site matrix with its two channels paired into per-cell (edited, total).

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;

use data_beans::hdf5_io::resolve_backend_file;
use data_beans::sparse_io::{open_sparse_matrix, COLUMN_SEP};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use super::Modality;
use faba::feature_name::parse_feature_row;
use matrix_util::common_io::basename;

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

/// One branch's covered cells for a site: edited `k`, coverage `n`, pseudotime `x`,
/// and total coverage — the per-(site, branch) input both trend estimators fit.
pub struct BranchData {
    pub k: Vec<u32>,
    pub n: Vec<u32>,
    pub x: Vec<f32>,
    pub cov: u64,
}

/// Split a site's covered cells (`n > 0`) into per-branch buckets in a single pass
/// over the cells — `O(ncell)`, versus rescanning once per branch. Bucket `l` holds
/// the cells whose primary branch is `l`, in cell order.
pub fn branch_buckets(site: &Site, lin: &Lineage) -> Vec<BranchData> {
    let mut buckets: Vec<BranchData> = (0..lin.n_branches)
        .map(|_| BranchData {
            k: Vec::new(),
            n: Vec::new(),
            x: Vec::new(),
            cov: 0,
        })
        .collect();
    for c in 0..site.n.len() {
        if site.n[c] > 0 {
            let b = &mut buckets[lin.branch[c]];
            b.k.push(site.k[c]);
            b.n.push(site.n[c]);
            b.x.push(lin.pseudotime[c]);
            b.cov += site.n[c] as u64;
        }
    }
    buckets
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

    // One site per (gene, subunit), pooled across the per-sample files: a gene recurs
    // once per replicate matrix, but each file's cells map to disjoint lineage indices
    // (distinct `@sample` tags), so summing k/n into a shared site fills different cells
    // rather than double-counting. `site_index` maps the key to its slot in `sites`.
    let mut sites: Vec<Site> = Vec::new();
    let mut site_index: FxHashMap<(Box<str>, Box<str>), usize> = FxHashMap::default();
    for path in paths {
        let (backend, resolved) = resolve_backend_file(path, None)
            .with_context(|| format!("resolving backend for {path}"))?;
        let data =
            open_sparse_matrix(&resolved, &backend).with_context(|| format!("opening {path}"))?;
        let row_names = data.row_names()?;
        let col_names = data.column_names()?;
        // The lineage tags each cell `{barcode}@{sample_id}` (gem's Union naming); a
        // per-sample modality matrix stores bare barcodes. Match `{barcode}@{sample_id}`
        // first, then fall back to the bare barcode (single-sample lineages that were
        // never `@`-tagged) — without this every cell drops and no (site, branch)
        // clears QC.
        let sid = site_sample_id(path, tok);
        let mut tagged = String::new(); // reused across columns to avoid a per-cell alloc
        let col_to_cell: Vec<Option<usize>> = col_names
            .iter()
            .map(|bc| {
                tagged.clear();
                tagged.push_str(bc.as_ref());
                tagged.push_str(COLUMN_SEP);
                tagged.push_str(&sid);
                cell_idx
                    .get(tagged.as_str())
                    .or_else(|| cell_idx.get(bc.as_ref()))
                    .copied()
            })
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
            let kv = std::mem::take(&mut kk[li]);
            let nv = std::mem::take(&mut nn[li]);
            let key = (gene, sub);
            if let Some(&idx) = site_index.get(&key) {
                // Same gene seen in an earlier replicate file — accumulate coverage.
                let s = &mut sites[idx];
                for c in 0..ncell {
                    s.k[c] += kv[c];
                    s.n[c] += nv[c];
                }
            } else {
                site_index.insert(key.clone(), sites.len());
                sites.push(Site {
                    gene: key.0,
                    subunit: key.1,
                    k: kv,
                    n: nv,
                });
            }
        }
    }
    Ok(sites)
}

/// Per-file sample id matching gem's `@sample` cell tag. The faba editing pipeline
/// names a modality matrix `{sample_id}_{modality}[_site].zarr.zip`, so stripping the
/// modality suffix (`_m6a_site`, then `_m6a`) recovers gem's sample id
/// (`rep1_wt_m6a.zarr.zip` → `rep1_wt`), letting the bare barcodes rejoin the
/// lineage's `{barcode}@rep1_wt` cells. A shared-suffix heuristic is deliberately
/// avoided — a WT-only file set shares `_wt_m6a` and would over-strip to `rep1`.
fn site_sample_id(path: &str, tok: &str) -> Box<str> {
    let base = basename(path).unwrap_or_else(|_| path.into());
    for suf in [format!("_{tok}_site"), format!("_{tok}")] {
        if let Some(s) = base.strip_suffix(suf.as_str()) {
            return s.into();
        }
    }
    base
}

#[cfg(test)]
mod tests {
    use super::site_sample_id;

    #[test]
    fn sample_id_strips_modality_suffix_not_shared_suffix() {
        // gem tags cells `@rep1_wt` (it stripped `_genes` over the mixed wt/mut set);
        // the modality files must recover the SAME id by stripping the modality token,
        // not the suffix shared by a wt-only file set (`_wt_m6a` → would give `rep1`).
        assert_eq!(&*site_sample_id("rep1_wt_m6a.zarr.zip", "m6a"), "rep1_wt");
        assert_eq!(&*site_sample_id("rep2_wt_m6a.zarr.zip", "m6a"), "rep2_wt");
        // site-level matrices carry the `_site` suffix first.
        assert_eq!(
            &*site_sample_id("out/rep3_mut_m6a_site.zarr.zip", "m6a"),
            "rep3_mut"
        );
        // other modalities.
        assert_eq!(&*site_sample_id("rep1_wt_atoi.zarr.zip", "atoi"), "rep1_wt");
        // no modality suffix → bare basename, so the bare-barcode fallback applies.
        assert_eq!(&*site_sample_id("sampleA.zarr.zip", "m6a"), "sampleA");
    }
}
