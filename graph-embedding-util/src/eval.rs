//! Output writers shared by all callers.
//!
//! Conforms to senna's parquet conventions (so `senna {clustering,
//! annotate, layout, plot} --from` work directly on outputs from any
//! caller):
//! - `{out}.latent.parquet` (cell × H), col prefix `h`
//! - `{out}.dictionary.parquet` (feature × H), col prefix `h`
//! - `{out}.feature_bias.parquet` (per-feature bias). There is no
//!   `cell_bias.parquet`: bge drops the per-cell bias `b_cell`
//!   (score = E_feat·E_cell + b_feat).

use crate::model::JointEmbedModel;
use candle_util::candle_core::Tensor;
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
    match ctx.cell_keep_idx {
        // Drop QC-failed cells from the per-cell output (the row indices and
        // barcodes are subset by the same mask, so they cannot desync).
        Some(keep) => {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx_t = Tensor::from_vec(idx, keep.len(), model.e_cell.device())?;
            let kept = model.e_cell.index_select(&idx_t, 0)?;
            let names: Vec<Box<str>> = keep.iter().map(|&i| ctx.barcodes[i].clone()).collect();
            save_embedding(&latent_path, &kept, &names, "cell")?;
        }
        None => save_embedding(&latent_path, &model.e_cell, ctx.barcodes, "cell")?,
    }
    // No `cell_bias.parquet` — bge drops the per-cell bias `b_cell`.
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

/// Write the per-condition feature gate parameters for inspection:
/// - `{out}.gene_program_loadings.parquet` — `z [D, K]`, rows = features,
///   cols `k0..k{K-1}`.
/// - `{out}.program_condition_deviation.parquet` — `δ [K, S, H]` reshaped
///   to `[K·S, H]`, rows `program_{k}/{condition}`, cols `h0..h{H-1}`.
///
/// These are the only condition-specific artifacts; the baseline `E_feat`
/// (dictionary) and `E_cell` (latent) stay condition-free. With a single
/// condition `δ ≡ 0`, so callers typically skip this.
pub fn save_gate(
    model: &JointEmbedModel,
    feature_names: &[Box<str>],
    condition_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    // z: [D, K] gene × program loadings.
    let k = model.num_programs;
    let prog_cols: Vec<Box<str>> = (0..k).map(|i| format!("k{i}").into_boxed_str()).collect();
    model.z.to_parquet_with_names(
        &format!("{out_prefix}.gene_program_loadings.parquet"),
        (Some(feature_names), Some("feature")),
        Some(&prog_cols),
    )?;

    // δ: [K, S, H] → [K*S, H], row = program_{k}/{condition}.
    let (n_prog, n_cond, h) = model.delta.dims3()?;
    let delta_2d = model.delta.reshape((n_prog * n_cond, h))?;
    let mut row_names: Vec<Box<str>> = Vec::with_capacity(n_prog * n_cond);
    for p in 0..n_prog {
        for c in 0..n_cond {
            let cond = condition_names.get(c).map(|x| x.as_ref()).unwrap_or("?");
            row_names.push(format!("program_{p}/{cond}").into_boxed_str());
        }
    }
    let cols = h_cols(h);
    delta_2d.to_parquet_with_names(
        &format!("{out_prefix}.program_condition_deviation.parquet"),
        (Some(&row_names), Some("program_condition")),
        Some(&cols),
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
