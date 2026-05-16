//! Per-step score-trace records and parquet writer, shared by every
//! pinto subcommand that wants to log loss / score history.
//!
//! `level = -1` marks post-EM final rows in `lc`. `sweep` is the
//! 0-based step index within a logical training/sampling unit.
//! `score / total_mass = -mean_H(p_k)` and
//! `mutual_information = H(p_global) − mean_H(p_k|community)` for `lc`;
//! `cage` reuses the same record shape with per-epoch loss in `score`.

use matrix_util::parquet::*;
use parquet::basic::Type as ParquetType;

#[derive(Clone, Copy, Debug)]
pub struct ScoreEntry {
    pub level: i32,
    pub sweep: i32,
    pub score: f64,
    pub n_edges: usize,
    pub total_mass: f64,
    pub mutual_information: f64,
}

/// Write score trace to parquet with six columns:
/// `(level, sweep, score, n_edges, total_mass, mutual_information)`.
pub fn write_score_trace(file_path: &str, entries: &[ScoreEntry]) -> anyhow::Result<()> {
    let n_rows = entries.len();

    let levels: Vec<i32> = entries.iter().map(|e| e.level).collect();
    let sweeps: Vec<i32> = entries.iter().map(|e| e.sweep).collect();
    let scores: Vec<f64> = entries.iter().map(|e| e.score).collect();
    let n_edges: Vec<i32> = entries.iter().map(|e| e.n_edges as i32).collect();
    let total_mass: Vec<f64> = entries.iter().map(|e| e.total_mass).collect();
    let mi: Vec<f64> = entries.iter().map(|e| e.mutual_information).collect();

    let col_names: Vec<Box<str>> = vec![
        "level".into(),
        "sweep".into(),
        "score".into(),
        "n_edges".into(),
        "total_mass".into(),
        "mutual_information".into(),
    ];
    let col_types = vec![
        ParquetType::INT32,
        ParquetType::INT32,
        ParquetType::DOUBLE,
        ParquetType::INT32,
        ParquetType::DOUBLE,
        ParquetType::DOUBLE,
    ];

    let writer = ParquetWriter::new(
        file_path,
        (n_rows, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("step"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_numeric_column(&mut row_group, &levels)?;
    parquet_add_numeric_column(&mut row_group, &sweeps)?;
    parquet_add_numeric_column(&mut row_group, &scores)?;
    parquet_add_numeric_column(&mut row_group, &n_edges)?;
    parquet_add_numeric_column(&mut row_group, &total_mass)?;
    parquet_add_numeric_column(&mut row_group, &mi)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}
