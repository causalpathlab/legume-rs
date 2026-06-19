//! `senna bge` output writers: feature/cell QC reports and the SIMBA-style
//! feature co-embedding. Split out of the bge driver to keep `mod.rs` focused
//! on the clap → `FitConfig` translation.

use crate::embed_common::*;
use graph_embedding_util as ge;

/// Empirical-Bayes null-feature QC on the trained feature embedding `E_feat`
/// (flat `[n × h]`): logs the fitted null (σ̂², π̂₀, #null) and writes
/// `{out}.feature_qc.parquet` with each feature's `norm²` and `live` flag. A
/// diagnostic — it does not (yet) drop features. Shares the call with faba gem.
pub(super) fn write_feature_qc(
    e_feat: &[f32],
    n: usize,
    h: usize,
    feature_names: &[Box<str>],
    fdr: f32,
    out_prefix: &str,
) -> anyhow::Result<ge::null_call::NullCall> {
    use matrix_util::dmatrix_io::DMatrix;
    use matrix_util::traits::IoOps;

    let null = ge::null_call::embedding_null_call(e_feat, n, h, fdr);
    info!(
        "bge feature null call — σ̂²={:.4}, ν̂={:.1}/{}, π̂₀={:.2}; {} / {} features null at FDR {} → {}.feature_qc.parquet",
        null.sigma2, null.eff_dof, h, null.pi0, n - null.n_live, n, fdr, out_prefix
    );

    let mut m = DMatrix::<f32>::zeros(n, 2);
    for f in 0..n {
        let s: f32 = e_feat[f * h..(f + 1) * h].iter().map(|&x| x * x).sum();
        m[(f, 0)] = s;
        m[(f, 1)] = f32::from(u8::from(null.live[f]));
    }
    let cols: Vec<Box<str>> = ["norm2", "live"].iter().map(|s| Box::from(*s)).collect();
    let path = format!("{out_prefix}.feature_qc.parquet");
    m.to_parquet_with_names(&path, (Some(feature_names), Some("feature")), Some(&cols))?;
    Ok(null)
}

/// Write the per-cell empty-droplet QC report `{out}.cell_qc.parquet`:
/// pre-L2 projection norm + a `kept` flag (1 = real cell, 0 = empty), one row
/// per cell, barcode-indexed.
pub(super) fn write_cell_qc(
    cell_nrms: &[f32],
    drop: &[bool],
    barcodes: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::dmatrix_io::DMatrix;
    use matrix_util::traits::IoOps;
    let n = cell_nrms.len();
    let mut m = DMatrix::<f32>::zeros(n, 2);
    for c in 0..n {
        m[(c, 0)] = cell_nrms[c];
        m[(c, 1)] = if drop.get(c).copied().unwrap_or(false) {
            0.0
        } else {
            1.0
        };
    }
    let cols: Vec<Box<str>> = ["pre_l2_norm", "kept"]
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    let path = format!("{out_prefix}.cell_qc.parquet");
    m.to_parquet_with_names(&path, (Some(barcodes), Some("cell")), Some(&cols))?;
    Ok(())
}

/// SIMBA-style feature co-embedding: re-embed every feature onto the cell
/// manifold (gene = softmax-over-cells weighted average of cell embeddings) and
/// write it as `{out}.feature_embedding.parquet` (overriding ρ), preserving the
/// raw ρ as `{out}.feature_embedding_raw.parquet`. `e_cell` is the QC-kept cell
/// embedding and `e_feat` the raw ρ (both `[*, H]`, on the same device);
/// `target_eff` is the eff-cells temperature target from `ge::cell_clusters`.
pub(super) fn write_feature_coembedding(
    out_prefix: &str,
    e_cell: &candle_core::Tensor,
    e_feat: &candle_core::Tensor,
    feature_names: &[Box<str>],
    target_eff: f64,
) -> anyhow::Result<()> {
    let (coembed, t) = ge::feature_coembedding(e_cell, e_feat, target_eff)?;
    ge::eval::save_embedding(
        &format!("{out_prefix}.feature_embedding.parquet"),
        &coembed,
        feature_names,
        "feature",
    )?;
    ge::eval::save_embedding(
        &format!("{out_prefix}.feature_embedding_raw.parquet"),
        e_feat,
        feature_names,
        "feature",
    )?;
    info!(
        "Feature co-embedding (SIMBA-style, T={t:.4}) → {out_prefix}.feature_embedding.parquet; \
         raw ρ → feature_embedding_raw.parquet"
    );
    Ok(())
}
