//! `senna bge` output writers: feature/cell QC reports and the joint-biplot
//! cell embedding. Split out of the bge driver to keep `mod.rs` focused on
//! the clap → `FitConfig` translation.

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

/// Write the joint-biplot cell embedding `{out}.cell_embedding_scaled.parquet`
/// (`[n_cells × H]`, depth-free + gene-co-scaled). For overlaying cells on the
/// gene dictionary `ρ`; not used by clustering. See `project_cells_phase2`.
pub(super) fn write_cell_embedding_scaled(
    scaled: &[f32],
    barcodes: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::dmatrix_io::DMatrix;
    use matrix_util::traits::IoOps;
    let n = barcodes.len();
    if n == 0 || !scaled.len().is_multiple_of(n) {
        return Ok(());
    }
    let h = scaled.len() / n;
    let mut m = DMatrix::<f32>::zeros(n, h);
    for r in 0..n {
        for c in 0..h {
            m[(r, c)] = scaled[r * h + c];
        }
    }
    let cols: Vec<Box<str>> = (0..h).map(|i| format!("h{i}").into_boxed_str()).collect();
    m.to_parquet_with_names(
        &format!("{out_prefix}.cell_embedding_scaled.parquet"),
        (Some(barcodes), Some("cell")),
        Some(&cols),
    )?;
    Ok(())
}
