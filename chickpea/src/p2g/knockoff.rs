//! GhostKnockoff FDR layer for peak→gene links.
//!
//! Per gene we already have the embedding-space marginal z-scores and the
//! peak–peak LD `R` from [`crate::p2g::embed::cis_link_stats`]. Under the RSS
//! null `z ~ N(0, R)`, we draw model-X (Gaussian) knockoff z-scores that share
//! the LD but are conditionally null, contrast each peak with its knockoff by
//! `W_j = |z_j| − |z̃_j|`, and apply the knockoff+ filter — pooled across genes
//! for genome-wide FDR control over links. (A SuSiE-PIP importance — running
//! the augmented `(z, z̃)` system through `finemap_gene` — is deferred to v2.)
//!
//! Refs: GhostKnockoff (He et al., Nat Commun 2022); model-X knockoffs
//! (Candès et al. 2018, JRSS-B); knockoff+ FDR (Barber & Candès 2015, AoS).

use crate::common::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Knockoff sampling + scoring parameters.
pub struct KnockoffParams {
    /// LD ridge: `R_λ = (1-λ)R + λI` (makes the rank-≤d cosine LD PD).
    pub ridge: f64,
    /// Base RNG seed; per-gene streams derive from this + gene id.
    pub seed: u64,
}

/// Importance statistic `W_j = |z_j| − |z̃_j|` for one gene's `C` cis peaks.
///
/// `z_raw` are the raw (pre-PVE) marginal z-scores; `r` the `[C×C]` LD. Larger
/// positive `W` ⇒ more evidence the peak is a true link. Returns a `[C]` vector
/// (empty when `C == 0`; neutral zeros if the linear algebra degenerates).
pub fn knockoff_w(z_raw: &[f32], r: &Mat, params: &KnockoffParams, gene_id: usize) -> Vec<f32> {
    let c = r.nrows();
    if c == 0 {
        return Vec::new();
    }

    // R_λ = (1-λ)R + λI, in f64 for stable linear algebra.
    let lambda = params.ridge;
    let mut r_lam = DMatrix::<f64>::zeros(c, c);
    for i in 0..c {
        for j in 0..c {
            r_lam[(i, j)] = (1.0 - lambda) * r[(i, j)] as f64;
        }
        r_lam[(i, i)] += lambda;
    }

    let s = equicorrelated_s(&r_lam);
    let r_inv = match r_lam.clone().try_inverse() {
        Some(inv) => inv,
        None => return vec![0.0; c],
    };

    let mut rng =
        SmallRng::seed_from_u64(params.seed ^ (gene_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let z = DVector::<f64>::from_iterator(c, z_raw.iter().map(|&v| v as f64));
    let z_tilde = match knockoff_z(&z, &r_lam, s, &r_inv, &mut rng) {
        Some(zt) => zt,
        None => return vec![0.0; c],
    };

    // v1 importance: marginal z-score contrast W_j = |z_j| − |z̃_j|. Flip-sign
    // holds by construction (swapping z_j ↔ z̃_j negates W_j). SuSiE-PIP-based
    // importance over the augmented (z, z̃) system is deferred to v2.
    (0..c)
        .map(|j| z_raw[j].abs() - z_tilde[j].abs() as f32)
        .collect()
}

/// Pooled knockoff+ threshold for target FDR `q`. Returns the smallest `t > 0`
/// with `(1 + #{W ≤ −t}) / (#{W ≥ t} ∨ 1) ≤ q`, or `+∞` if none qualifies
/// (i.e. no selections). NaN entries are ignored.
pub fn knockoff_threshold(w: &[f32], q: f64) -> f32 {
    let mut cands: Vec<f32> = w
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .filter(|&x| x > 0.0)
        .collect();
    cands.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cands.dedup();
    for &t in &cands {
        let pos = w.iter().filter(|&&x| x.is_finite() && x >= t).count();
        let neg = w.iter().filter(|&&x| x.is_finite() && x <= -t).count();
        let fdp = (1.0 + neg as f64) / (pos.max(1) as f64);
        if fdp <= q {
            return t;
        }
    }
    f32::INFINITY
}

/// Equicorrelated knockoff diagonal `s = min(1, 2·λ_min(R_λ))` (so that
/// `2R_λ − sI ⪰ 0`). `R_λ` must be symmetric PD.
fn equicorrelated_s(r_lam: &DMatrix<f64>) -> f64 {
    // Eigenvalues only — no eigenvectors, no clone; we just need λ_min.
    let lambda_min = r_lam
        .symmetric_eigenvalues()
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    (2.0 * lambda_min).clamp(0.0, 1.0)
}

/// Sample knockoff z-scores `z̃ | z ~ N((R_λ − sI)R_λ⁻¹ z, 2sI − s²R_λ⁻¹)`.
fn knockoff_z(
    z: &DVector<f64>,
    r_lam: &DMatrix<f64>,
    s: f64,
    r_inv: &DMatrix<f64>,
    rng: &mut SmallRng,
) -> Option<DVector<f64>> {
    let c = z.len();

    // mean = (R_λ − sI) R_λ⁻¹ z
    let mut p = r_lam.clone();
    for i in 0..c {
        p[(i, i)] -= s;
    }
    let pr = &p * r_inv;
    let mean = &pr * z;

    // noise cov V = 2sI − s² R_λ⁻¹ (symmetric PSD); sample e = chol(V)·η.
    // Cholesky reads only V's lower triangle, so no symmetrization is needed.
    let mut v = r_inv.scale(-s * s);
    for i in 0..c {
        v[(i, i)] += 2.0 * s;
    }

    let chol = match v.clone().cholesky() {
        Some(ch) => ch,
        None => {
            let mut vj = v;
            for i in 0..c {
                vj[(i, i)] += 1e-8;
            }
            vj.cholesky()?
        }
    };
    let l = chol.l();
    let eta = DVector::<f64>::from_fn(c, |_, _| StandardNormal.sample(rng));
    Some(mean + &l * &eta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mvn(l: &DMatrix<f64>, rng: &mut SmallRng) -> DVector<f64> {
        let c = l.nrows();
        let eta = DVector::<f64>::from_fn(c, |_, _| StandardNormal.sample(rng));
        l * &eta
    }

    /// The construction must reproduce the exchangeable joint covariance:
    /// `cov(z, z̃) = R_λ − sI` and `var(z̃) = R_λ`.
    #[test]
    fn knockoff_matches_target_covariance() {
        let c = 3;
        let mut r_lam = DMatrix::<f64>::identity(c, c);
        r_lam[(0, 1)] = 0.5;
        r_lam[(1, 0)] = 0.5;
        r_lam[(1, 2)] = 0.3;
        r_lam[(2, 1)] = 0.3;
        for i in 0..c {
            r_lam[(i, i)] += 0.1;
        }
        let s = equicorrelated_s(&r_lam);
        let r_inv = r_lam.clone().try_inverse().unwrap();
        let r_chol = r_lam.clone().cholesky().unwrap().l();

        let n = 40_000usize;
        let mut rng = SmallRng::seed_from_u64(7);
        let mut cross = DMatrix::<f64>::zeros(c, c);
        let mut var = DMatrix::<f64>::zeros(c, c);
        for _ in 0..n {
            let z = sample_mvn(&r_chol, &mut rng);
            let zt = knockoff_z(&z, &r_lam, s, &r_inv, &mut rng).unwrap();
            let zt_t = zt.transpose();
            cross += &z * &zt_t;
            var += &zt * &zt_t;
        }
        cross /= n as f64;
        var /= n as f64;
        for i in 0..c {
            for j in 0..c {
                let tgt_cross = r_lam[(i, j)] - if i == j { s } else { 0.0 };
                assert!(
                    (cross[(i, j)] - tgt_cross).abs() < 0.05,
                    "cross[{i},{j}]={} vs {tgt_cross}",
                    cross[(i, j)]
                );
                assert!(
                    (var[(i, j)] - r_lam[(i, j)]).abs() < 0.05,
                    "var[{i},{j}]={} vs {}",
                    var[(i, j)],
                    r_lam[(i, j)]
                );
            }
        }
    }

    fn make_block_ld(c: usize) -> Mat {
        // Block-constant correlation (within-block 0.6), PSD for block size 5.
        let mut r = Mat::identity(c, c);
        let block = 5;
        for i in 0..c {
            for j in 0..c {
                if i != j && i / block == j / block {
                    r[(i, j)] = 0.6;
                }
            }
        }
        r
    }

    fn ld_f64(r: &Mat) -> DMatrix<f64> {
        let c = r.nrows();
        DMatrix::<f64>::from_fn(c, c, |i, j| r[(i, j)] as f64)
    }

    /// Pooled empirical FDR should track the target `q`, with non-trivial
    /// power, when many genes are filtered jointly. Deterministic (seeded);
    /// the slack absorbs Monte-Carlo variation around `E[FDP] ≤ q`.
    #[test]
    fn pooled_fdr_is_controlled() {
        use rand::RngExt;
        let c = 20usize;
        let n_genes = 300usize;
        let causal_per_gene = 2usize;
        let q = 0.1;
        let params = KnockoffParams {
            ridge: 0.05,
            seed: 1,
        };

        let r = make_block_ld(c);
        let r_f64 = ld_f64(&r);
        let mut r_pd = r_f64.clone();
        for i in 0..c {
            r_pd[(i, i)] += 1e-3;
        }
        let r_chol = r_pd.cholesky().unwrap().l();

        let mut rng = SmallRng::seed_from_u64(123);
        let mut all_w: Vec<f32> = Vec::new();
        let mut is_causal: Vec<bool> = Vec::new();

        for g in 0..n_genes {
            let mut causal = std::collections::HashSet::new();
            while causal.len() < causal_per_gene {
                causal.insert(rng.random_range(0..c));
            }
            let mut beta = DVector::<f64>::zeros(c);
            for &idx in &causal {
                beta[idx] = 6.0;
            }
            let mean = &r_f64 * &beta;
            let eta = DVector::<f64>::from_fn(c, |_, _| StandardNormal.sample(&mut rng));
            let noise = &r_chol * &eta;
            let z: Vec<f32> = (0..c).map(|i| (mean[i] + noise[i]) as f32).collect();

            let w = knockoff_w(&z, &r, &params, g);
            for (j, &wj) in w.iter().enumerate() {
                all_w.push(wj);
                is_causal.push(causal.contains(&j));
            }
        }

        let tau = knockoff_threshold(&all_w, q);
        let selected: Vec<usize> = (0..all_w.len())
            .filter(|&i| all_w[i].is_finite() && all_w[i] >= tau)
            .collect();
        let n_sel = selected.len();
        let n_false = selected.iter().filter(|&&i| !is_causal[i]).count();
        let fdp = if n_sel > 0 {
            n_false as f64 / n_sel as f64
        } else {
            0.0
        };

        assert!(n_sel > 0, "no selections — knockoff filter has no power");
        assert!(
            fdp <= q + 0.08,
            "empirical FDP {fdp:.3} exceeds q={q} (+slack); {n_sel} selected, {n_false} false"
        );
    }

    /// End-to-end calibration on the **real embedding pipeline**: build a global
    /// ATAC embedding from a factor-structured pseudobulk matrix, then for each
    /// gene score a cis set whose null peaks load on factors orthogonal to the
    /// gene's program. Unlike `pooled_fdr_is_controlled` (hand-built z ~ N(Rβ,R)),
    /// this passes z/R through `build_atac_embedding` + `cis_link_stats`, so it
    /// tests whether the embedding-space statistics keep the knockoff FDR honest.
    #[test]
    fn embedding_knockoff_controls_fdr_on_nulls() {
        use crate::p2g::embed::{build_atac_embedding, cis_link_stats, project_gene};
        use rand::RngExt;

        let s = 200usize; // pseudobulk samples
        let k = 15usize; // latent factors (programs)
        let per_factor = 40usize;
        let p = k * per_factor; // peaks
        let n_genes = 150usize;
        let c = 20usize; // cis peaks per gene
        let n_causal = 3usize; // causal cis peaks per gene (load on gene's factor)
        let d = 30usize; // embedding rank
        let q = 0.1;

        let mut rng = SmallRng::seed_from_u64(2024);
        let randn = |rng: &mut SmallRng| -> f32 {
            let v: f64 = StandardNormal.sample(rng);
            v as f32
        };

        // Latent factor profiles over samples.
        let factors: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..s).map(|_| randn(&mut rng)).collect())
            .collect();

        // Global ATAC pb: each peak loads on one factor (block LD); +baseline +noise.
        let peak_factor: Vec<usize> = (0..p).map(|pp| pp % k).collect();
        let mut atac = Mat::zeros(p, s);
        for pp in 0..p {
            let load = rng.random_range(0.5f32..1.5);
            let f = &factors[peak_factor[pp]];
            for ss in 0..s {
                atac[(pp, ss)] = (2.0 + load * f[ss] + 0.3 * randn(&mut rng)).max(0.01);
            }
        }
        let emb = build_atac_embedding(&atac, d).unwrap();

        let mut by_factor: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (pp, &f) in peak_factor.iter().enumerate() {
            by_factor[f].push(pp);
        }

        let params = KnockoffParams {
            ridge: 0.05,
            seed: 7,
        };

        let mut all_w: Vec<f32> = Vec::new();
        let mut is_null: Vec<bool> = Vec::new();

        for gid in 0..n_genes {
            let sf = gid % k; // gene's signal factor
            let gf = &factors[sf];
            let gene: Vec<f32> = (0..s)
                .map(|ss| (2.0 + 2.0 * gf[ss] + 0.5 * randn(&mut rng)).max(0.01))
                .collect();

            // cis set: n_causal peaks on the gene's factor (truly associated) +
            // null peaks on *other* factors (orthogonal → no association).
            let mut cis: Vec<usize> = Vec::with_capacity(c);
            let mut null_flag: Vec<bool> = Vec::with_capacity(c);
            for i in 0..n_causal {
                let pool = &by_factor[sf];
                cis.push(pool[(gid * 7 + i) % pool.len()]);
                null_flag.push(false);
            }
            while cis.len() < c {
                let f = rng.random_range(0..k);
                if f == sf {
                    continue;
                }
                let pool = &by_factor[f];
                let pk = pool[rng.random_range(0..pool.len())];
                if !cis.contains(&pk) {
                    cis.push(pk);
                    null_flag.push(true);
                }
            }

            let proj = project_gene(&emb, &gene);
            let (z_raw, r) = cis_link_stats(&proj, &emb, &cis, s as f64);
            let w = knockoff_w(&z_raw, &r, &params, gid);
            for (j, &wj) in w.iter().enumerate() {
                all_w.push(wj);
                is_null.push(null_flag[j]);
            }
        }

        let tau = knockoff_threshold(&all_w, q);
        let sel: Vec<usize> = (0..all_w.len())
            .filter(|&i| all_w[i].is_finite() && all_w[i] >= tau)
            .collect();
        let n_sel = sel.len();
        let n_false = sel.iter().filter(|&&i| is_null[i]).count();
        let n_causal_total = is_null.iter().filter(|&&x| !x).count();
        let fdp = if n_sel > 0 {
            n_false as f64 / n_sel as f64
        } else {
            0.0
        };
        let power = (n_sel - n_false) as f64 / n_causal_total as f64;
        eprintln!(
            "embedding-knockoff FDR: selected={n_sel} false={n_false} FDP={fdp:.3} power={power:.3}"
        );
        assert!(n_sel > 0, "no power on the embedding sim");
        assert!(
            fdp <= q + 0.10,
            "embedding-space FDP {fdp:.3} exceeds q={q} (+slack); {n_sel} sel, {n_false} false"
        );
    }
}
