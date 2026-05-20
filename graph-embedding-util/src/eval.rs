//! Output writers shared by all callers.
//!
//! Conforms to senna's parquet conventions (so `senna {clustering,
//! annotate, layout, plot} --from` work directly on outputs from any
//! caller):
//! - `{out}.latent.parquet` (cell × H), col prefix `h`
//! - `{out}.dictionary.parquet` (feature × H), col prefix `h`
//! - `{out}.cell_bias.parquet` / `{out}.feature_bias.parquet`

use crate::model::JointEmbedModel;
use candle_util::candle_core::Tensor;
use matrix_util::traits::IoOps;

pub struct OutputContext<'a> {
    /// Names for the rows of `model.e_feat` / `model.b_feat`. The
    /// dictionary parquet is keyed directly on these — gbe trains
    /// `E_feat` at fine gene resolution, so no replication is needed.
    pub feature_names: &'a [Box<str>],
    pub barcodes: &'a [Box<str>],
}

pub fn save_outputs(
    model: &JointEmbedModel,
    ctx: &OutputContext,
    out_prefix: &str,
) -> anyhow::Result<()> {
    save_embedding(
        &format!("{out_prefix}.latent.parquet"),
        &model.e_cell,
        ctx.barcodes,
        "cell",
    )?;
    save_bias(
        &format!("{out_prefix}.cell_bias.parquet"),
        &model.b_cell,
        ctx.barcodes,
        "cell",
    )?;
    save_embedding(
        &format!("{out_prefix}.dictionary.parquet"),
        &model.e_feat,
        ctx.feature_names,
        "feature",
    )?;
    save_bias(
        &format!("{out_prefix}.feature_bias.parquet"),
        &model.b_feat,
        ctx.feature_names,
        "feature",
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

/// Write a 1-D bias tensor `[N]` as an `[N, 1]` `bias` column parquet.
/// Public so callers that emit a custom output layout (e.g. `senna bge
/// --resolve-etm`) reuse the same bias format instead of re-implementing it.
pub fn save_bias(
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
