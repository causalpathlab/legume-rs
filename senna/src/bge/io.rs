//! `senna bge` output writers: feature/cell QC reports. The SIMBA-style feature
//! co-embedding writer is shared with `senna rest` and lives in
//! [`crate::embed_common::write_feature_coembedding`]. Split out of the bge
//! driver to keep `mod.rs` focused on the clap → `FitConfig` translation.

use crate::embed_common::*;
use graph_embedding_util as ge;

/// Empirical-Bayes null-feature QC on the trained feature embedding `E_feat`
/// (flat `[n × h]`): logs the fitted null (σ̂², π̂₀, #null) and writes
/// `{out}.feature_qc.parquet` with each feature's `norm²` and `live` flag, and
/// returns the `NullCall` (live mask) the bge driver uses for its two-pass
/// null-feature refine (drop + re-fit when `--feature-null-fdr > 0`).
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
