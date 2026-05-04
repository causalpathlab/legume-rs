//! Centralized parquet output writers for senna training routines.
//!
//! Every senna `fit-*` routine writes the same handful of parquet files:
//! latent (cell × K), dictionary (gene × K), pb_gene (gene × P), and
//! fisher_weights (gene × 1). Centralizing the `to_parquet_with_names`
//! call sites keeps file naming, axis labels, and column-id conventions
//! identical across topic / itopic / joint-topic / svd / joint-svd.

use crate::embed_common::{axis_id_names, Mat};
use matrix_util::traits::IoOps;

/// Save cell × K latent matrix as `{out}.latent.parquet`.
/// Columns: `T0..T(K-1)` (topic / component convention shared across
/// topic and svd routines so `senna plot --colour-by topic` reads the
/// file identically regardless of upstream).
pub fn save_latent(out: &str, latent_nk: &Mat, cell_names: &[Box<str>]) -> anyhow::Result<()> {
    let cols = axis_id_names("T", latent_nk.ncols());
    latent_nk.to_parquet_with_names(
        &format!("{out}.latent.parquet"),
        (Some(cell_names), Some("cell")),
        Some(&cols),
    )?;
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
