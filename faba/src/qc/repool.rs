//! Re-derive a gene-level editing matrix from a filtered `_site` matrix.
//!
//! The producer pools EVERY putative site into `{batch}_{modality}`, so after
//! `qc` cuts the site axis that matrix would still carry the coverage of the
//! sites it removed. Summing the kept site rows per `{gene}/{modality}/{channel}`
//! keeps gene level coherent with the site cut by construction.

use crate::common::*;
use data_beans::aux::feature_rows::{feature_row, parse_feature_row};
use rustc_hash::FxHashMap;

use super::matrix::{OutSpec, Written};

/// Sum `rows` (kept site rows, ascending) over `cols` (kept cells, ascending)
/// into gene-level channel rows, drop pooled rows with fewer than
/// `row_nnz_cutoff` non-zero cells, and write the result. `None` when nothing
/// survives.
///
/// Columns are read in blocks and each column is scattered into a dense
/// scratch over the pooled rows, so the output triplets come out already
/// grouped by column and no per-entry hashing is needed.
pub fn repool_gene_level(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    cols: &[usize],
    rows: &[usize],
    row_names: &[Box<str>],
    col_names: &[Box<str>],
    row_nnz_cutoff: usize,
    spec: &OutSpec<'_>,
) -> anyhow::Result<Option<Written>> {
    if cols.is_empty() || rows.is_empty() {
        return Ok(None);
    }
    // Old site row -> pooled row id, only for kept rows.
    let mut pooled_names: Vec<Box<str>> = Vec::new();
    let mut pooled_index: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut row_to_pooled: Vec<Option<u32>> = vec![None; row_names.len()];
    for &r in rows {
        let Some(f) = parse_feature_row(&row_names[r]) else {
            continue;
        };
        let name = feature_row(f.gene, f.modality, f.channel, None);
        let id = *pooled_index.entry(name.clone()).or_insert_with(|| {
            pooled_names.push(name.clone());
            (pooled_names.len() - 1) as u32
        });
        row_to_pooled[r] = Some(id);
    }
    let n_pooled = pooled_names.len();

    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    let mut pooled_nnz = vec![0usize; n_pooled];
    let mut scratch = vec![0f32; n_pooled];
    let mut touched: Vec<u32> = Vec::new();
    for (lb, ub) in
        legume_numeric::matrix::utils::generate_minibatch_intervals(cols.len(), 0, Some(8192))
    {
        let (_, _, block) = data.read_triplets_by_columns(cols[lb..ub].to_vec())?;
        let mut per_col: Vec<Vec<(u32, f32)>> = vec![Vec::new(); ub - lb];
        for (r, c_local, x) in block {
            if let Some(p) = row_to_pooled[r as usize] {
                per_col[c_local as usize].push((p, x));
            }
        }
        for (c_local, entries) in per_col.into_iter().enumerate() {
            for (p, x) in entries {
                if scratch[p as usize] == 0.0 {
                    touched.push(p);
                }
                scratch[p as usize] += x;
            }
            touched.sort_unstable();
            for &p in &touched {
                let x = std::mem::take(&mut scratch[p as usize]);
                if x > 0.0 {
                    pooled_nnz[p as usize] += 1;
                    triplets.push((p as u64, (lb + c_local) as u64, x));
                }
            }
            touched.clear();
        }
    }

    // Row cutoff on the pooled rows, then the sorted vocabulary every
    // producer writes.
    let keep_row: Vec<bool> = pooled_nnz
        .iter()
        .map(|&n| n >= row_nnz_cutoff.max(1))
        .collect();
    let mut order: Vec<u32> = (0..n_pooled as u32)
        .filter(|&p| keep_row[p as usize])
        .collect();
    order.sort_by(|a, b| pooled_names[*a as usize].cmp(&pooled_names[*b as usize]));
    if order.is_empty() {
        return Ok(None);
    }
    let mut new_row = vec![u64::MAX; n_pooled];
    for (i, &p) in order.iter().enumerate() {
        new_row[p as usize] = i as u64;
    }
    triplets.retain(|(p, _, _)| keep_row[*p as usize]);
    for t in triplets.iter_mut() {
        t.0 = new_row[t.0 as usize];
    }

    let out_rows: Vec<Box<str>> = order
        .iter()
        .map(|&p| pooled_names[p as usize].clone())
        .collect();
    let out_cols: Vec<Box<str>> = cols.iter().map(|&j| col_names[j].clone()).collect();
    let shape = (out_rows.len(), out_cols.len(), triplets.len());
    let path = spec.path();
    remove_file(&path.write_path)?;
    remove_file(&path.target_path)?;
    let mut backend = create_sparse_from_triplets_owned(
        triplets,
        shape,
        Some(&path.write_path),
        Some(spec.backend),
    )?;
    backend.register_row_names_vec(&out_rows);
    backend.register_column_names_vec(&out_cols);
    drop(backend);
    path.finalize()?;
    Ok(Some(Written {
        target: path.target_path.clone(),
        nrow: shape.0,
        ncol: shape.1,
        nnz: shape.2,
    }))
}
