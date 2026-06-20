//! Output writers shared by all callers.
//!
//! Conforms to senna's parquet conventions (so `senna {clustering,
//! annotate, layout, plot} --from` work directly on outputs from any
//! caller):
//! - `{out}.latent.parquet` (cell × H), col prefix `h`
//! - `{out}.dictionary.parquet` (feature × H), col prefix `h`
//! - `{out}.feature_bias.parquet` (per-feature bias).
//! - `{out}.cell_bias.parquet` (per-cell bias `b_cell`, the depth sink).

use crate::model::JointEmbedModel;
use candle_util::candle_core::Tensor;
use log::info;
use matrix_util::traits::IoOps;

pub struct OutputContext<'a> {
    /// Names for the rows of `model.e_feat` / `model.b_feat`. The
    /// dictionary parquet is keyed directly on these — gbe trains
    /// `E_feat` at fine gene resolution, so no replication is needed.
    pub feature_names: &'a [Box<str>],
    pub barcodes: &'a [Box<str>],
    /// Optional cell keep-mask (QC): emit only these cell rows of the
    /// per-cell latent, with matching barcodes. `None` = emit every cell.
    pub cell_keep_idx: Option<&'a [usize]>,
}

pub fn save_outputs(
    model: &JointEmbedModel,
    ctx: &OutputContext,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let latent_path = format!("{out_prefix}.latent.parquet");
    let cell_bias_path = format!("{out_prefix}.cell_bias.parquet");
    match ctx.cell_keep_idx {
        // Drop QC-failed cells from the per-cell outputs (the row indices and
        // barcodes are subset by the same mask, so they cannot desync).
        Some(keep) => {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx_t = Tensor::from_vec(idx, keep.len(), model.e_cell.device())?;
            let kept = model.e_cell.index_select(&idx_t, 0)?;
            let kept_b = model.b_cell.index_select(&idx_t, 0)?;
            let names: Vec<Box<str>> = keep.iter().map(|&i| ctx.barcodes[i].clone()).collect();
            save_embedding(&latent_path, &kept, &names, "cell")?;
            save_bias(&cell_bias_path, &kept_b, &names, "cell")?;
        }
        None => {
            save_embedding(&latent_path, &model.e_cell, ctx.barcodes, "cell")?;
            save_bias(&cell_bias_path, &model.b_cell, ctx.barcodes, "cell")?;
        }
    }
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

/// Canonical embedding-coordinate column names `h0..h{H-1}` — the single
/// convention shared by every embedding writer (senna `save_embedding`,
/// `faba gem`, `pinto cage`, `annotate-by-projection`). One source of truth so
/// the `{out}.*_embedding.parquet` column schema never drifts (`h` vs `dim_` vs
/// `e`). Embedding columns are always read positionally, so the name is purely
/// for human/schema legibility — but it must be consistent across tools.
pub fn embedding_col_names(h: usize) -> Vec<Box<str>> {
    (0..h).map(|i| format!("h{i}").into_boxed_str()).collect()
}

/// Write an embedding table `[N, H]` to parquet with `h0..h{H-1}` columns and
/// `row_names` on the `row_axis`. Public so callers can emit auxiliary
/// embeddings (e.g. bge's pre-cell-QC "before" cell embedding) in the same
/// layout as the standard latent output.
pub fn save_embedding(
    path: &str,
    table: &Tensor,
    row_names: &[Box<str>],
    row_axis: &str,
) -> anyhow::Result<()> {
    let h = table.dim(1)?;
    let cols = embedding_col_names(h);
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

/// SIMBA-style feature co-embedding, shared by `senna bge` and `senna rest`:
/// re-embed every feature onto the cell manifold (feature = softmax-over-cells
/// weighted average of the cell embeddings) via [`crate::feature_coembedding`]
/// and write it as `{out}.feature_embedding.parquet`, *overriding* the raw
/// learned feature embedding (the raw embedding is the disjoint off-manifold
/// cloud — nothing downstream consumes it, so it is not written). `e_cell` is
/// the reference cell embedding (left unchanged, SIMBA's anchor) and `e_feat`
/// the raw feature embedding (both `[*, H]` on the same device); `target_eff`
/// is the eff-cells temperature target from [`crate::cell_clusters`].
pub fn write_feature_coembedding(
    out_prefix: &str,
    e_cell: &Tensor,
    e_feat: &Tensor,
    feature_names: &[Box<str>],
    target_eff: f64,
) -> anyhow::Result<()> {
    let (coembed, t) = crate::feature_coembedding(e_cell, e_feat, target_eff)?;
    save_embedding(
        &format!("{out_prefix}.feature_embedding.parquet"),
        &coembed,
        feature_names,
        "feature",
    )?;
    info!("Feature co-embedding (SIMBA-style, T={t:.4}) → {out_prefix}.feature_embedding.parquet");
    Ok(())
}
