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

/// Replicate supergene-trained `E_feat` / `b_feat` back to per-gene
/// rows at output time, so downstream tools (senna annotate, pinto
/// cell-community) stay gene-keyed. Both fields are required when this
/// is `Some`; modeled together so the "missing means absent" coupling
/// can't be split.
pub struct GeneAxisMapping<'a> {
    pub names: &'a [Box<str>],
    pub to_supergene: &'a [usize],
}

pub struct OutputContext<'a> {
    /// Names for the rows of `model.e_feat` / `model.b_feat` AS-TRAINED.
    /// When `gene_axis` is `Some`, the dictionary parquet is keyed on
    /// `gene_axis.names` instead and this field applies only to
    /// `feature_bias.parquet` row labels for the supergene axis.
    pub feature_names: &'a [Box<str>],
    pub barcodes: &'a [Box<str>],
    pub gene_axis: Option<GeneAxisMapping<'a>>,
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

    if let Some(g) = ctx.gene_axis.as_ref() {
        anyhow::ensure!(
            g.names.len() == g.to_supergene.len(),
            "gene names {} != gene_to_supergene {}",
            g.names.len(),
            g.to_supergene.len()
        );
        let dev = model.e_feat.device();
        let idx: Vec<u32> = g.to_supergene.iter().map(|&sg| sg as u32).collect();
        let idx_t = Tensor::from_vec(idx, g.names.len(), dev)?;
        let e_feat_genes = model.e_feat.index_select(&idx_t, 0)?;
        let b_feat_genes = model.b_feat.index_select(&idx_t, 0)?;
        save_embedding(
            &format!("{out_prefix}.dictionary.parquet"),
            &e_feat_genes,
            g.names,
            "feature",
        )?;
        save_bias(
            &format!("{out_prefix}.feature_bias.parquet"),
            &b_feat_genes,
            g.names,
            "feature",
        )?;
        save_gene_to_supergene_csv(
            &format!("{out_prefix}.gene_groups.csv"),
            g.names,
            g.to_supergene,
        )?;
    } else {
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
    }
    Ok(())
}

fn save_gene_to_supergene_csv(
    path: &str,
    gene_names: &[Box<str>],
    g2sg: &[usize],
) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "gene,supergene")?;
    for (g, sg) in gene_names.iter().zip(g2sg.iter()) {
        writeln!(w, "{},{}", g, sg)?;
    }
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
