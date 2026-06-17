//! Centralized parquet output writers for senna training routines.
//!
//! Every senna `fit-*` routine writes the same handful of parquet files:
//! latent (cell × K), dictionary (gene × K), `pb_gene` (gene × P), and
//! `fisher_weights` (gene × 1). Centralizing the `to_parquet_with_names`
//! call sites keeps file naming, axis labels, and column-id conventions
//! identical across topic / masked-topic / joint-topic / svd / joint-svd.

use crate::embed_common::{axis_id_names, Mat};
use matrix_util::traits::IoOps;

/// Apply the near-empty output keep-mask (cell QC) to a per-cell matrix
/// (rows = cells) and its name slice. Returns `None` when `keep_idx` is
/// `None` or already a full-length identity (no near-empty cells), so
/// callers can write the original without copying. `Some((mat, names))`
/// is the subset to write instead.
///
/// Shared by every per-cell writer so latent / proj / `cell_to_pb` stay
/// row-aligned after dropping near-empty cells.
pub fn cell_subset(
    mat: &Mat,
    names: &[Box<str>],
    keep_idx: Option<&[usize]>,
) -> Option<(Mat, Vec<Box<str>>)> {
    let idx = keep_idx?;
    if idx.len() == mat.nrows() {
        return None; // full identity → nothing dropped
    }
    Some((
        mat.select_rows(idx.iter()),
        idx.iter().map(|&i| names[i].clone()).collect(),
    ))
}

/// Save cell × K latent matrix as `{out}.latent.parquet`.
/// Columns: `T0..T(K-1)` (topic / component convention shared across
/// topic and svd routines so `senna plot --colour-by topic` reads the
/// file identically regardless of upstream).
///
/// `keep_idx` is the optional near-empty output keep-mask from cell QC
/// (see [`cell_subset`]); pass `None` to emit every cell.
pub fn save_latent(
    out: &str,
    latent_nk: &Mat,
    cell_names: &[Box<str>],
    keep_idx: Option<&[usize]>,
) -> anyhow::Result<()> {
    let cols = axis_id_names("T", latent_nk.ncols());
    let path = format!("{out}.latent.parquet");
    match cell_subset(latent_nk, cell_names, keep_idx) {
        Some((mat, names)) => {
            mat.to_parquet_with_names(&path, (Some(&names), Some("cell")), Some(&cols))?;
        }
        None => {
            latent_nk.to_parquet_with_names(
                &path,
                (Some(cell_names), Some("cell")),
                Some(&cols),
            )?;
        }
    }
    Ok(())
}

/// Save gene × K dictionary matrix as `{out}.dictionary.parquet`.
/// Columns: `T0..T(K-1)`.
pub fn save_dictionary(out: &str, dict_dk: &Mat, gene_names: &[Box<str>]) -> anyhow::Result<()> {
    let cols = axis_id_names("T", dict_dk.ncols());
    dict_dk.to_parquet_with_names(
        &format!("{out}.dictionary.parquet"),
        (Some(gene_names), Some("gene")),
        Some(&cols),
    )?;
    Ok(())
}

/// Save gene × P pseudobulk gene aggregates as `{out}.pb_gene.parquet`.
/// Columns: `PB_0..PB_(P-1)`.
pub fn save_pb_gene(out: &str, pb_gene_gp: &Mat, gene_names: &[Box<str>]) -> anyhow::Result<()> {
    let cols = axis_id_names("PB_", pb_gene_gp.ncols());
    pb_gene_gp.to_parquet_with_names(
        &format!("{out}.pb_gene.parquet"),
        (Some(gene_names), Some("gene")),
        Some(&cols),
    )?;
    Ok(())
}

pub use data_beans_alg::gene_weighting::save_fisher_weights;
