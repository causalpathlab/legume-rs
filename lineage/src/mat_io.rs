//! The handful of dense-matrix helpers the lineage family shares.
//!
//! These are local copies of the same helpers in `senna::embed_common`.
//! Depending on that module would mean `lineage → senna`, which cycles once
//! `senna` re-exports this crate, so the two small bodies live here instead.

use matrix_util::common_io::file_ext;
use matrix_util::traits::IoOps;

pub type Mat = nalgebra::DMatrix<f32>;

pub use matrix_util::traits::MatWithNames;

/// Read a matrix from a parquet or delimited text file.
pub fn read_mat(file_path: &str) -> anyhow::Result<MatWithNames<Mat>> {
    Ok(match file_ext(file_path)?.as_ref() {
        "parquet" => Mat::from_parquet(file_path)?,
        _ => Mat::read_data(file_path, &['\t', ','], None, Some(0), None, None)?,
    })
}

/// Build `{prefix}0..{prefix}{k-1}` axis-id column names — the explicit
/// "this column is topic/cluster N" convention every K-dim writer uses, so a
/// reader can recover the integer ID from the column name alone.
#[must_use]
pub fn axis_id_names(prefix: &str, k: usize) -> Vec<Box<str>> {
    (0..k)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

/// Row-wise L2 normalization in place: Euclidean distance on the result
/// equals cosine distance on the input. A ~zero row is left unchanged —
/// normalizing it would blow it up to an arbitrary unit direction.
pub fn l2_normalize_rows_inplace(m: &mut Mat) {
    for mut row in m.row_iter_mut() {
        let norm = row.norm();
        if norm > 1e-9 {
            row /= norm;
        }
    }
}
