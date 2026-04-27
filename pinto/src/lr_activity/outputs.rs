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
    /// Number of pseudobulk samples participating in this (batch, community).
    pub n_samples: i32,
    /// Observed weighted covariance between sender-pseudobulk `log1p(w·L)`
    /// and receiver-pseudobulk `log1p(w·R)` across samples.
    pub stat_obs: f32,
    pub null_mean: f32,
    pub null_sd: f32,
    pub z: f32,
    /// Empirical permutation p (mid-p, tie-aware). Diagnostic only.
    pub p_empirical: f32,
    /// Parametric one-sided p from the Gaussian tail of per-pair `z`.
    /// Diagnostic only — `null_sd` is noisy under bucketed permutation.
    pub p_z: f32,
    /// Restandardized z (Efron-Tibshirani GSA-style): `(stat_obs - μ_emp) / σ_emp`
    /// where `μ_emp, σ_emp` are robust (median, MAD) moments of `stat_obs`
    /// across all pairs in the stratum. Calibrates against the empirical
    /// null bulk instead of the noisy per-pair permutation null.
    pub z_re: f32,
    /// Two-sided Gaussian tail p of `z_re`. Diagnostic.
    pub p_re: f32,
    /// Westfall-Young single-step minP adjusted p-value (FWER-controlled).
    /// Joint sample permutation across all pairs in the stratum preserves
    /// the dependence structure (shared genes, propensity buckets, batch
    /// confounders); much cleaner than FDR for individual-pair claims.
    pub fwer_wy: f32,
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
    let n_samples: Vec<i32> = rows.iter().map(|r| r.n_samples).collect();
    let stat_obs: Vec<f32> = rows.iter().map(|r| r.stat_obs).collect();
    let null_mean: Vec<f32> = rows.iter().map(|r| r.null_mean).collect();
    let null_sd: Vec<f32> = rows.iter().map(|r| r.null_sd).collect();
    let z: Vec<f32> = rows.iter().map(|r| r.z).collect();
    let p_emp: Vec<f32> = rows.iter().map(|r| r.p_empirical).collect();
    let p_z: Vec<f32> = rows.iter().map(|r| r.p_z).collect();
    let z_re: Vec<f32> = rows.iter().map(|r| r.z_re).collect();
    let p_re: Vec<f32> = rows.iter().map(|r| r.p_re).collect();
    let fwer_wy: Vec<f32> = rows.iter().map(|r| r.fwer_wy).collect();

    let col_names: Vec<Box<str>> = vec![
        "batch".into(),
        "community".into(),
        "ligand".into(),
        "receptor".into(),
        "n_samples".into(),
        "stat_obs".into(),
        "null_mean".into(),
        "null_sd".into(),
        "z".into(),
        "p_empirical".into(),
        "p_z".into(),
        "z_re".into(),
        "p_re".into(),
        "fwer_wy".into(),
    ];
    let col_types = vec![
        ParquetType::BYTE_ARRAY,
        ParquetType::INT32,
        ParquetType::BYTE_ARRAY,
        ParquetType::BYTE_ARRAY,
        ParquetType::INT32,
        ParquetType::FLOAT,
        ParquetType::FLOAT,
        ParquetType::FLOAT,
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
    parquet_add_numeric_column(&mut row_group, &n_samples)?;
    parquet_add_numeric_column(&mut row_group, &stat_obs)?;
    parquet_add_numeric_column(&mut row_group, &null_mean)?;
    parquet_add_numeric_column(&mut row_group, &null_sd)?;
    parquet_add_numeric_column(&mut row_group, &z)?;
    parquet_add_numeric_column(&mut row_group, &p_emp)?;
    parquet_add_numeric_column(&mut row_group, &p_z)?;
    parquet_add_numeric_column(&mut row_group, &z_re)?;
    parquet_add_numeric_column(&mut row_group, &p_re)?;
    parquet_add_numeric_column(&mut row_group, &fwer_wy)?;

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
/// when at least one row in that stratum has `fwer_wy < fwer_threshold`
/// (Westfall-Young single-step minP, family-wise error rate).
pub fn write_lr_activity_json(
    out_path: &str,
    lc_prefix: &str,
    upstream_metadata: Option<&str>,
    rows: &[LrActivityRow],
    strata: &[StratumEntry],
    fwer_threshold: f32,
) -> anyhow::Result<()> {
    use serde_json::{json, Value};
    let is_sig = |r: &LrActivityRow| {
        r.fwer_wy.is_finite() && r.fwer_wy < fwer_threshold && r.z_re.is_finite() && r.z_re > 0.0
    };
    let n_total = rows.len();
    let n_sig = rows.iter().filter(|r| is_sig(r)).count();

    let mut sig_strata: HashSet<usize> = HashSet::default();
    for r in rows {
        if is_sig(r) {
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

    // Only significant rows go in the JSON — full table is already in
    // {out}.lr_activity.parquet. Plot consumer filters on `significant`
    // anyway, so non-sig rows would just bloat the file.
    let results_json: Vec<Value> = rows
        .iter()
        .filter(|r| is_sig(r))
        .map(|r| {
            let sid = renum.get(&r.stratum_id).copied();
            json!({
                "batch": r.batch.as_ref(),
                "community": r.community,
                "ligand": r.ligand.as_ref(),
                "receptor": r.receptor.as_ref(),
                "ligand_resolved": r.ligand_resolved.as_ref(),
                "receptor_resolved": r.receptor_resolved.as_ref(),
                "n_samples": r.n_samples,
                "stat_obs": opt_finite(r.stat_obs),
                "z": opt_finite(r.z),
                "z_re": opt_finite(r.z_re),
                "fwer_wy": opt_finite(r.fwer_wy),
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
        "fwer_threshold": fwer_threshold,
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

/// ASCII histogram of restandardized p-values (`p_re`) across all finite
/// rows — this is what drives Storey's q. Empirical permutation p
/// (`p_empirical`) is on the parquet for diagnostics.
pub fn pvalue_histogram(rows: &[LrActivityRow], max_width: usize) -> String {
    const N_BINS: usize = 20;
    let mut bins = [0usize; N_BINS];
    let mut n = 0usize;
    let mut n_nan = 0usize;
    for r in rows {
        if !r.p_re.is_finite() {
            n_nan += 1;
            continue;
        }
        let p = r.p_re.clamp(0.0, 1.0);
        let mut b = (p * N_BINS as f32) as usize;
        if b >= N_BINS {
            b = N_BINS - 1;
        }
        bins[b] += 1;
        n += 1;
    }
    let max_bin = *bins.iter().max().unwrap_or(&1).max(&1);

    let mut lines = Vec::new();
    lines.push(format!(
        "Two-sided restandardized p-value (p_re) distribution: {} rows  ({} non-finite skipped)",
        n, n_nan
    ));
    lines.push(String::new());
    for (i, &count) in bins.iter().enumerate() {
        let lo = i as f32 / N_BINS as f32;
        let hi = (i + 1) as f32 / N_BINS as f32;
        let bar_len = ((count as f64 / max_bin as f64) * max_width as f64) as usize;
        let bar = "\u{2588}".repeat(bar_len);
        let pct = if n > 0 {
            100.0 * count as f64 / n as f64
        } else {
            0.0
        };
        lines.push(format!(
            "  [{:.2}, {:.2})  {:>7} ({:>5.1}%)  {}",
            lo, hi, count, pct, bar
        ));
    }
    lines.join("\n")
}
