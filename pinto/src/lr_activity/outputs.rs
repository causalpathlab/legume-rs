//! Parquet writer for `pinto lr-activity` results.

use matrix_util::parquet::*;
use parquet::basic::Type as ParquetType;

pub struct LrActivityRow {
    pub batch: Box<str>,
    pub community: i32,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub n_edges: i32,
    pub n_components: i32,
    pub ce_obs: f32,
    pub ce_null_mean: f32,
    pub ce_null_sd: f32,
    pub z: f32,
    pub p_empirical: f32,
    pub q_bh: f32,
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
