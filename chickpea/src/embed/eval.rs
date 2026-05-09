//! Output writers for trained joint multi-modal embeddings.
//!
//! Parquet outputs consistent with senna / chickpea topic conventions
//! (`to_parquet_with_names` keyed on row names + `H{i}` column ids).

use crate::embed::model::JointEmbedModel;
use candle_util::candle_core::Tensor;
use matrix_util::traits::IoOps;

pub struct OutputContext<'a> {
    pub feature_names: &'a [Box<str>],
    pub barcodes: &'a [Box<str>],
}

pub fn save_outputs(
    model: &JointEmbedModel,
    ctx: &OutputContext,
    out_prefix: &str,
) -> anyhow::Result<()> {
    save_embedding(
        &format!("{out_prefix}.e_feat.parquet"),
        &model.e_feat,
        ctx.feature_names,
        "feature",
    )?;
    save_embedding(
        &format!("{out_prefix}.e_cell.parquet"),
        &model.e_cell,
        ctx.barcodes,
        "cell",
    )?;
    save_bias(
        &format!("{out_prefix}.b_feat.parquet"),
        &model.b_feat,
        ctx.feature_names,
        "feature",
    )?;
    save_bias(
        &format!("{out_prefix}.b_cell.parquet"),
        &model.b_cell,
        ctx.barcodes,
        "cell",
    )?;
    Ok(())
}

fn h_cols(h: usize) -> Vec<Box<str>> {
    (0..h).map(|i| format!("h{i}").into_boxed_str()).collect()
}

fn save_embedding(
    path: &str,
    table: &Tensor,
    row_names: &[Box<str>],
    row_axis: &str,
) -> anyhow::Result<()> {
    let h = table.dim(1)?;
    let cols = h_cols(h);
    table.to_parquet_with_names(path, (Some(row_names), Some(row_axis)), Some(&cols))?;
    Ok(())
}

fn save_bias(
    path: &str,
    bias: &Tensor,
    row_names: &[Box<str>],
    row_axis: &str,
) -> anyhow::Result<()> {
    let bias_2d = bias.unsqueeze(1)?;
    let col = vec![Box::<str>::from("bias")];
    bias_2d.to_parquet_with_names(path, (Some(row_names), Some(row_axis)), Some(&col))?;
    Ok(())
}
