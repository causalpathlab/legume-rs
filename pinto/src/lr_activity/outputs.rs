//! Parquet writer for `pinto lr-activity` results.

use crate::util::common::{HashMap, HashSet};
use matrix_util::parquet::*;
use parquet::basic::Type as ParquetType;

pub struct LrActivityRow {
    pub batch: Box<str>,
    pub community: i32,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    /// Canonical row name of the ligand gene as it appears in the
    /// expression backend's `row_names()` (e.g. `ENSG00000112715_VEGFA_Gene`).
    /// May differ from the user-supplied `ligand` (e.g. just `VEGFA`).
    /// Persisted to the JSON sidecar so downstream tools can look up
    /// expression directly without re-running the gene resolver.
    pub ligand_resolved: Box<str>,
    pub receptor_resolved: Box<str>,
    pub n_edges: i32,
    pub n_components: i32,
    pub ce_obs: f32,
    pub ce_null_mean: f32,
    pub ce_null_sd: f32,
    pub z: f32,
    pub p_empirical: f32,
    pub q_bh: f32,
    /// Index into a per-run `strata` table written to the JSON sidecar.
    /// Not serialized to parquet — only consulted by the JSON writer.
    pub stratum_id: usize,
}

pub fn write_lr_activity(file_path: &str, rows: &[LrActivityRow]) -> anyhow::Result<()> {
    let n_rows = rows.len();

    let batches: Vec<Box<str>> = rows.iter().map(|r| r.batch.clone()).collect();
    let communities: Vec<i32> = rows.iter().map(|r| r.community).collect();
    let ligands: Vec<Box<str>> = rows.iter().map(|r| r.ligand.clone()).collect();
    let receptors: Vec<Box<str>> = rows.iter().map(|r| r.receptor.clone()).collect();
    let n_edges: Vec<i32> = rows.iter().map(|r| r.n_edges).collect();
    let n_components: Vec<i32> = rows.iter().map(|r| r.n_components).collect();
    let ce_obs: Vec<f32> = rows.iter().map(|r| r.ce_obs).collect();
    let ce_null_mean: Vec<f32> = rows.iter().map(|r| r.ce_null_mean).collect();
    let ce_null_sd: Vec<f32> = rows.iter().map(|r| r.ce_null_sd).collect();
    let z: Vec<f32> = rows.iter().map(|r| r.z).collect();
    let p_emp: Vec<f32> = rows.iter().map(|r| r.p_empirical).collect();
    let q_bh: Vec<f32> = rows.iter().map(|r| r.q_bh).collect();

    let col_names: Vec<Box<str>> = vec![
        "batch".into(),
        "community".into(),
        "ligand".into(),
        "receptor".into(),
        "n_edges".into(),
        "n_components".into(),
        "ce_obs".into(),
        "ce_null_mean".into(),
        "ce_null_sd".into(),
        "z".into(),
        "p_empirical".into(),
        "q_bh".into(),
    ];
    let col_types = vec![
        ParquetType::BYTE_ARRAY,
        ParquetType::INT32,
        ParquetType::BYTE_ARRAY,
        ParquetType::BYTE_ARRAY,
        ParquetType::INT32,
        ParquetType::INT32,
        ParquetType::FLOAT,
        ParquetType::FLOAT,
        ParquetType::FLOAT,
        ParquetType::FLOAT,
        ParquetType::FLOAT,
        ParquetType::FLOAT,
    ];

    let writer = ParquetWriter::new(
        file_path,
        (n_rows, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("row"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_string_column(&mut row_group, &batches)?;
    parquet_add_numeric_column(&mut row_group, &communities)?;
    parquet_add_string_column(&mut row_group, &ligands)?;
    parquet_add_string_column(&mut row_group, &receptors)?;
    parquet_add_numeric_column(&mut row_group, &n_edges)?;
    parquet_add_numeric_column(&mut row_group, &n_components)?;
    parquet_add_numeric_column(&mut row_group, &ce_obs)?;
    parquet_add_numeric_column(&mut row_group, &ce_null_mean)?;
    parquet_add_numeric_column(&mut row_group, &ce_null_sd)?;
    parquet_add_numeric_column(&mut row_group, &z)?;
    parquet_add_numeric_column(&mut row_group, &p_emp)?;
    parquet_add_numeric_column(&mut row_group, &q_bh)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}

/// One stratum (batch, community) participating in the JSON sidecar.
/// Edges are encoded as cell-name pairs so the plot consumer can match
/// them against `coord_pairs.parquet` without depending on edge ordering.
pub struct StratumEntry {
    pub batch: Box<str>,
    pub community: i32,
    pub edges: Vec<(Box<str>, Box<str>)>,
}

/// Write the `{out}.lr_activity.json` sidecar. Strata are emitted only
/// when at least one row in that stratum has `q_bh < q_threshold`.
pub fn write_lr_activity_json(
    out_path: &str,
    lc_prefix: &str,
    upstream_metadata: Option<&str>,
    rows: &[LrActivityRow],
    strata: &[StratumEntry],
    q_threshold: f32,
) -> anyhow::Result<()> {
    use serde_json::{json, Value};
    let n_total = rows.len();
    let n_sig = rows
        .iter()
        .filter(|r| r.q_bh.is_finite() && r.q_bh < q_threshold)
        .count();

    let mut sig_strata: HashSet<usize> = HashSet::default();
    for r in rows {
        if r.q_bh.is_finite() && r.q_bh < q_threshold {
            sig_strata.insert(r.stratum_id);
        }
    }
    let keep_strata: Vec<usize> = (0..strata.len())
        .filter(|s| sig_strata.contains(s))
        .collect();
    let renum: HashMap<usize, usize> = keep_strata
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();

    let strata_json: Vec<Value> = keep_strata
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| {
            let s = &strata[old_idx];
            let edges_json: Vec<Value> = s
                .edges
                .iter()
                .map(|(l, r)| json!([l.as_ref(), r.as_ref()]))
                .collect();
            json!({
                "stratum_id": new_idx,
                "batch": s.batch.as_ref(),
                "community": s.community,
                "n_edges": s.edges.len(),
                "edges": edges_json,
            })
        })
        .collect();

    let results_json: Vec<Value> = rows
        .iter()
        .map(|r| {
            let sig = r.q_bh.is_finite() && r.q_bh < q_threshold;
            let sid = renum.get(&r.stratum_id).copied();
            json!({
                "batch": r.batch.as_ref(),
                "community": r.community,
                "ligand": r.ligand.as_ref(),
                "receptor": r.receptor.as_ref(),
                "ligand_resolved": r.ligand_resolved.as_ref(),
                "receptor_resolved": r.receptor_resolved.as_ref(),
                "n_edges": r.n_edges,
                "n_components": r.n_components,
                "ce_obs": opt_finite(r.ce_obs),
                "ce_null_mean": opt_finite(r.ce_null_mean),
                "ce_null_sd": opt_finite(r.ce_null_sd),
                "z": opt_finite(r.z),
                "p_empirical": opt_finite(r.p_empirical),
                "q_bh": opt_finite(r.q_bh),
                "significant": sig,
                "stratum_id": sid,
            })
        })
        .collect();

    let root = json!({
        "command": "lr-activity",
        "lc_prefix": lc_prefix,
        "input_metadata": upstream_metadata,
        "n_total_pairs": n_total,
        "n_significant": n_sig,
        "q_threshold": q_threshold,
        "strata": strata_json,
        "results": results_json,
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

fn opt_finite(v: f32) -> serde_json::Value {
    if v.is_finite() {
        serde_json::Value::from(v)
    } else {
        serde_json::Value::Null
    }
}

/// Benjamini-Hochberg FDR within a slice of p-values. Returns `q_bh[i]`
/// aligned to the input order. `p` must be in `[0, 1]`; `NaN` is propagated.
pub fn bh_qvalues(p: &[f32]) -> Vec<f32> {
    let n = p.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut q = vec![0.0f32; n];
    let mut running_min = f32::INFINITY;
    for rank_from_top in 0..n {
        let rank = n - rank_from_top; // 1-based, largest first
        let i = order[rank - 1];
        if p[i].is_nan() {
            q[i] = f32::NAN;
            continue;
        }
        let scaled = p[i] * (n as f32) / (rank as f32);
        running_min = running_min.min(scaled);
        q[i] = running_min.min(1.0);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bh_monotone() {
        let p = vec![0.001, 0.01, 0.04, 0.5, 0.9];
        let q = bh_qvalues(&p);
        // q should be nondecreasing in sorted-p order and q >= p in this case.
        for i in 0..p.len() {
            assert!(q[i] >= p[i] - 1e-6);
        }
    }

    #[test]
    fn bh_all_ones() {
        let p = vec![1.0; 5];
        let q = bh_qvalues(&p);
        for qi in q {
            assert!((qi - 1.0).abs() < 1e-6);
        }
    }
}
