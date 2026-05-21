//! Global ATAC embedding and embedding-space marginal / LD computations.
//!
//! We build one pseudobulk-level ATAC embedding (rSVD of the standardized log1p
//! ATAC pb matrix), project each gene into it, and read both the marginal
//! peak→gene association and the peak–peak LD off inner products in that space.
//! See `docs/peak_to_gene_math.md` for the derivation.

use crate::common::*;
use nalgebra::DVector;

/// Global ATAC embedding from the pb-level accessibility matrix.
///
/// From `Ã = U Σ Vᵀ` (rSVD of the row-centered log1p ATAC pb `[P×S]`):
/// - `w = U Σ` `[P×d]` — peak vectors,
/// - `v = V` `[S×d]` — orthonormal pseudobulk-sample factors.
pub struct AtacEmbedding {
    pub w: Mat,
    pub v: Mat,
    pub d: usize,
}

/// Build the global ATAC embedding: log1p, center each peak over samples, rSVD.
pub fn build_atac_embedding(atac_pb: &Mat, embedding_dim: usize) -> anyhow::Result<AtacEmbedding> {
    let (p, s) = (atac_pb.nrows(), atac_pb.ncols());
    let mut a = atac_pb.map(|v| (v + 1.0).ln());
    // Center each peak (row) over samples.
    for i in 0..p {
        let mut m = 0.0f32;
        for j in 0..s {
            m += a[(i, j)];
        }
        m /= s as f32;
        for j in 0..s {
            a[(i, j)] -= m;
        }
    }

    let k = embedding_dim.min(p).min(s).max(1);
    let (u, sigma, v) = a.rsvd(k)?; // u[P,k], sigma[k], v[S,k]
    let kk = sigma.len();
    if kk < embedding_dim {
        info!(
            "ATAC embedding rank {} (min(P={}, S={}, d={}))",
            kk, p, s, embedding_dim
        );
    }

    // W = U Σ
    let mut w = u;
    for c in 0..kk {
        let sc = sigma[c];
        for r in 0..w.nrows() {
            w[(r, c)] *= sc;
        }
    }
    Ok(AtacEmbedding { w, v, d: kk })
}

/// A gene projected into the ATAC sample-factor space.
pub struct GeneProjection {
    /// `g̃ = Vᵀ x̃` where `x̃ = log1p(gene)` centered over samples.
    pub g_tilde: DVector<f32>,
    /// `‖x̃‖²` (total centered log variance of the gene over samples).
    pub x_norm2: f64,
}

/// Project a gene's pb profile into the ATAC embedding: `g̃ = Vᵀ·center(log1p(gene))`.
pub fn project_gene(emb: &AtacEmbedding, gene_rate: &[f32]) -> GeneProjection {
    let s = gene_rate.len();
    let mut x = DVector::<f32>::zeros(s);
    let mut m = 0.0f32;
    for (j, &g) in gene_rate.iter().enumerate() {
        x[j] = (g + 1.0).ln();
        m += x[j];
    }
    m /= s as f32;
    x.iter_mut().for_each(|v| *v -= m);
    let g_tilde = emb.v.tr_mul(&x); // [d]
    let x_norm2 = x.dot(&x) as f64;
    GeneProjection { g_tilde, x_norm2 }
}

/// For one projected gene and its cis peaks, compute both the marginal z-scores
/// and the peak–peak LD `R`, sharing the cis-peak norms `‖W_p‖`.
///
/// - marginal `z_p = β̂/SE` of OLS(`x̃` on `ŷ_p = V W_pᵀ`), where
///   `β̂ = (g̃·W_p)/‖W_p‖²` and `RSS = ‖x̃‖² − (g̃·W_p)²/‖W_p‖²`;
/// - LD `R_{ij} = (W_i·W_j)/(‖W_i‖‖W_j‖)` (cosine; unit diagonal, PSD, rank ≤ d).
pub fn cis_link_stats(
    proj: &GeneProjection,
    emb: &AtacEmbedding,
    cis: &[usize],
    n_eff: f64,
) -> (Vec<f32>, Mat) {
    let d = emb.d;
    let c = cis.len();

    // Per cis peak: g̃·W_p and ‖W_p‖², in one pass.
    let mut gw = vec![0.0f64; c];
    let mut wn2 = vec![0.0f64; c];
    for (i, &p) in cis.iter().enumerate() {
        for k in 0..d {
            let wc = emb.w[(p, k)] as f64;
            gw[i] += proj.g_tilde[k] as f64 * wc;
            wn2[i] += wc * wc;
        }
    }

    // Marginal z (σ²=1 OLS in embedding space).
    let z: Vec<f32> = (0..c)
        .map(|i| {
            if wn2[i] < 1e-12 || n_eff <= 2.0 {
                return 0.0;
            }
            let rss = (proj.x_norm2 - gw[i] * gw[i] / wn2[i]).max(0.0);
            let se = (rss / (n_eff - 2.0) / wn2[i]).sqrt();
            if se < 1e-12 {
                0.0
            } else {
                ((gw[i] / wn2[i]) / se) as f32
            }
        })
        .collect();

    // LD = cosine of peak embeddings, reusing the norms.
    let norms: Vec<f64> = wn2.iter().map(|&v| v.sqrt()).collect();
    let mut r = Mat::zeros(c, c);
    for i in 0..c {
        r[(i, i)] = if norms[i] < 1e-12 { 0.0 } else { 1.0 };
        for j in (i + 1)..c {
            let mut dot = 0.0f64;
            for k in 0..d {
                dot += emb.w[(cis[i], k)] as f64 * emb.w[(cis[j], k)] as f64;
            }
            let den = norms[i] * norms[j];
            let v = if den < 1e-12 { 0.0 } else { (dot / den) as f32 };
            r[(i, j)] = v;
            r[(j, i)] = v;
        }
    }
    (z, r)
}

/// PVE (winner's-curse) shrinkage of a z-score: `z·√((n-1)/(z²+n-2))`.
pub fn pve_adjust(z: f32, n: usize) -> f32 {
    let nf = n as f64;
    if nf <= 2.0 {
        return z;
    }
    let z = z as f64;
    (z * ((nf - 1.0) / (z * z + nf - 2.0)).sqrt()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2g::finemap::{finemap_gene, FinemapParams};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    /// Full embedding pipeline: a causal peak co-occurring with the gene amid
    /// correlated decoys should win the PIP.
    #[test]
    fn embedding_recovers_causal_amid_ld() {
        let s = 400usize;
        let c = 12usize;
        let causal = 4usize;
        let mut rng = SmallRng::seed_from_u64(123);

        let f: Vec<f32> = (0..s).map(|_| rng.random_range(0.1f32..3.0)).collect();
        let mut atac = Mat::zeros(c, s);
        for j in 0..c {
            for k in 0..s {
                let noise = rng.random_range(0.0f32..1.0);
                atac[(j, k)] = if j == causal {
                    (2.5 * f[k] + 0.2 * noise).max(0.0)
                } else if (2..7).contains(&j) {
                    (0.8 * f[k] + 1.5 * noise).max(0.0)
                } else {
                    rng.random_range(0.0f32..3.0)
                };
            }
        }
        let gene: Vec<f32> = (0..s)
            .map(|k| (3.0 * f[k] + rng.random_range(0.0f32..0.8)).max(0.0))
            .collect();

        let emb = build_atac_embedding(&atac, 10).unwrap();
        let proj = project_gene(&emb, &gene);
        let cis: Vec<usize> = (0..c).collect();
        let (z_raw, r) = cis_link_stats(&proj, &emb, &cis, s as f64);
        let z: Vec<f32> = z_raw.iter().map(|&zc| pve_adjust(zc, s)).collect();

        let params = FinemapParams {
            num_components: 5,
            prior_var: 5.0,
        };
        let (pip, _, _) = finemap_gene(&r, &z, &params);
        let argmax = pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, causal, "PIPs: {pip:?}");
        assert!(pip[causal] > 0.5, "causal PIP too low: {}", pip[causal]);
    }
}
