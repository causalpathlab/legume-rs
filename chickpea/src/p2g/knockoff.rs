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

    /// Diagnostic: can a topic-adjusted (partial) association separate DIRECT
    /// causal peaks from topic-CONFOUNDED ones? Model-2 confounding: the gene is
    /// driven by its causal peaks' accessibility (incl. their peak-specific
    /// variation); confounded peaks only share the gene's topic factor. Marginal
    /// association can't tell them apart; partialling out the top-m shared
    /// embedding factors should keep causal and collapse confounded.
    #[test]
    #[ignore = "diagnostic: partial-z decomposition of causal vs confounded hits"]
    fn decompose_causal_vs_confounded_via_partial_z() {
        use crate::p2g::embed::build_atac_embedding;

        let s = 300usize;
        let k = 6usize;
        let per_factor = 50usize;
        let p = k * per_factor;
        let n_genes = 120usize;
        let d = 60usize;
        let m = k; // partial out the topic factors

        let mut rng = SmallRng::seed_from_u64(11);
        let randn = |rng: &mut SmallRng| -> f32 {
            let v: f64 = StandardNormal.sample(rng);
            v as f32
        };

        let factors: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..s).map(|_| randn(&mut rng)).collect())
            .collect();
        let peak_factor: Vec<usize> = (0..p).map(|pp| pp % k).collect();
        // peak-specific sample variation — the carrier of direct causal signal.
        let spec: Vec<Vec<f32>> = (0..p)
            .map(|_| (0..s).map(|_| randn(&mut rng)).collect())
            .collect();

        let mut atac = Mat::zeros(p, s);
        for pp in 0..p {
            let f = &factors[peak_factor[pp]];
            for ss in 0..s {
                atac[(pp, ss)] = (3.0 + f[ss] + 0.8 * spec[pp][ss]).max(0.01);
            }
        }
        let emb = build_atac_embedding(&atac, d).unwrap();

        let clog = |v: &[f32]| -> DVector<f32> {
            let mut x = DVector::from_iterator(s, v.iter().map(|&a| (a + 1.0).ln()));
            let mean = x.mean();
            x.add_scalar_mut(-mean);
            x
        };
        let peak_clog: Vec<DVector<f32>> = (0..p)
            .map(|pp| clog(&atac.row(pp).iter().copied().collect::<Vec<_>>()))
            .collect();

        let vm = emb.v.columns(0, m).into_owned(); // [S × m] orthonormal topics
        let partial = |x: &DVector<f32>| -> DVector<f32> {
            let proj = &vm * (vm.transpose() * x);
            x.clone() - proj
        };
        let corr = |a: &DVector<f32>, b: &DVector<f32>| -> f64 {
            let (na, nb) = (a.norm(), b.norm());
            if na < 1e-8 || nb < 1e-8 {
                0.0
            } else {
                (a.dot(b) / (na * nb)) as f64
            }
        };

        let mut by_factor: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (pp, &f) in peak_factor.iter().enumerate() {
            by_factor[f].push(pp);
        }

        let (mut mca, mut mco, mut mnu) = (0.0f64, 0.0, 0.0);
        let (mut pca, mut pco, mut pnu) = (0.0f64, 0.0, 0.0);
        let (mut nca, mut nco, mut nnu) = (0usize, 0, 0);

        for gid in 0..n_genes {
            let sf = gid % k;
            let causal: Vec<usize> = (0..3)
                .map(|i| by_factor[sf][(gid * 5 + i) % by_factor[sf].len()])
                .collect();
            let mut gene = vec![0f32; s];
            for ss in 0..s {
                let mut v = 3.0;
                for &cp in &causal {
                    v += factors[sf][ss] + 0.8 * spec[cp][ss];
                }
                v += 0.5 * randn(&mut rng);
                gene[ss] = v.max(0.01);
            }
            let gx = clog(&gene);
            let gx_p = partial(&gx);

            let confounded: Vec<usize> = by_factor[sf]
                .iter()
                .copied()
                .filter(|pp| !causal.contains(pp))
                .take(8)
                .collect();
            let null: Vec<usize> = (0..p).filter(|pp| peak_factor[*pp] != sf).take(8).collect();

            for &cp in &causal {
                mca += corr(&gx, &peak_clog[cp]).abs();
                pca += corr(&gx_p, &partial(&peak_clog[cp])).abs();
                nca += 1;
            }
            for &cp in &confounded {
                mco += corr(&gx, &peak_clog[cp]).abs();
                pco += corr(&gx_p, &partial(&peak_clog[cp])).abs();
                nco += 1;
            }
            for &cp in &null {
                mnu += corr(&gx, &peak_clog[cp]).abs();
                pnu += corr(&gx_p, &partial(&peak_clog[cp])).abs();
                nnu += 1;
            }
        }
        let avg = |x: f64, n: usize| x / n as f64;
        eprintln!("mean |corr|       causal  confound   null");
        eprintln!(
            "  marginal:       {:.3}    {:.3}    {:.3}",
            avg(mca, nca),
            avg(mco, nco),
            avg(mnu, nnu)
        );
        eprintln!(
            "  partial(m={m}):    {:.3}    {:.3}    {:.3}",
            avg(pca, nca),
            avg(pco, nco),
            avg(pnu, nnu)
        );
        let marg_gap = avg(mca, nca) - avg(mco, nco);
        let part_gap = avg(pca, nca) - avg(pco, nco);
        eprintln!("  causal−confound gap: marginal={marg_gap:.3}  partial={part_gap:.3}");
        assert!(
            part_gap > marg_gap,
            "partial did not improve causal/confound separation"
        );
    }

    /// Prototype: on *identifiable* confounding (Model-2: the gene is driven by
    /// its cis causal peaks' own accessibility, so causal ≠ confounded given the
    /// topic), the knockoff controls the confounded FDP through `R` — both the
    /// marginal (full-R) and the LOCO+DML (off-chromosome confounder, residual-
    /// on-residual z) variants. This is the positive control: when the effect is
    /// statistically identifiable the machinery works. (The data-beans-sim's
    /// ~0.62 confounded FDP comes instead from its near-UNidentifiable model —
    /// RNA rate = (M·β_ext)·θ ⇒ gene ⊥ peak | θ — not from the knockoff.)
    /// Topics are genome-wide; W₋c is estimated off the gene's chromosome so it
    /// captures the trans topic without absorbing the cis signal.
    #[test]
    #[ignore = "prototype: knockoff controls confounded FDP on identifiable data (marginal + LOCO/DML)"]
    fn loco_dml_controls_confounded_fdp() {
        use crate::p2g::embed::build_atac_embedding;

        let s = 400usize;
        let k = 6usize; // genome-wide topics
        let n_chr = 10usize;
        let per_chr = 80usize;
        let p = n_chr * per_chr;
        let n_genes = 120usize;
        let m = 8usize; // confounder dims (≥ k)
        let q = 0.1;

        let mut rng = SmallRng::seed_from_u64(2025);
        let randn = |rng: &mut SmallRng| -> f32 {
            let v: f64 = StandardNormal.sample(rng);
            v as f32
        };

        let topics: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..s).map(|_| randn(&mut rng)).collect())
            .collect();
        // peak attributes: chromosome (block) + topic (genome-wide) + specific noise.
        let chr: Vec<usize> = (0..p).map(|pp| pp / per_chr).collect();
        let ptopic: Vec<usize> = (0..p)
            .map(|_| (randn(&mut rng).abs() as usize * 7 + 3) % k)
            .collect();
        let spec: Vec<Vec<f32>> = (0..p)
            .map(|_| (0..s).map(|_| randn(&mut rng)).collect())
            .collect();

        let mut atac = Mat::zeros(p, s);
        for pp in 0..p {
            let t = &topics[ptopic[pp]];
            for ss in 0..s {
                atac[(pp, ss)] = (3.0 + 1.2 * t[ss] + 0.8 * spec[pp][ss]).max(0.01);
            }
        }

        // LOCO confounder W₋c: off-chromosome pb embedding (one per chromosome).
        let w_by_chr: Vec<Mat> = (0..n_chr)
            .map(|c| {
                let rows: Vec<usize> = (0..p).filter(|&pp| chr[pp] != c).collect();
                let mut sub = Mat::zeros(rows.len(), s);
                for (i, &pp) in rows.iter().enumerate() {
                    for ss in 0..s {
                        sub[(i, ss)] = atac[(pp, ss)];
                    }
                }
                let emb = build_atac_embedding(&sub, m).unwrap();
                let mm = m.min(emb.d);
                emb.v.columns(0, mm).into_owned()
            })
            .collect();

        let clog = |v: &[f32]| -> DVector<f32> {
            DVector::from_iterator(s, v.iter().map(|&a| (a + 1.0).ln()))
        };
        let peak_clog: Vec<DVector<f32>> = (0..p)
            .map(|pp| clog(&atac.row(pp).iter().copied().collect::<Vec<_>>()))
            .collect();
        let resid = |w: &Mat, x: &DVector<f32>| -> DVector<f32> { x - &(w * (w.transpose() * x)) };
        let corr = |a: &DVector<f32>, b: &DVector<f32>| -> f64 {
            let (ma, mb) = (a.mean(), b.mean());
            let (mut sab, mut sa, mut sb) = (0.0f64, 0.0, 0.0);
            for i in 0..a.len() {
                let (da, db) = ((a[i] - ma) as f64, (b[i] - mb) as f64);
                sab += da * db;
                sa += da * da;
                sb += db * db;
            }
            if sa < 1e-12 || sb < 1e-12 {
                0.0
            } else {
                sab / (sa.sqrt() * sb.sqrt())
            }
        };
        let tstat = |r: f64, df: f64| -> f32 {
            let r = r.clamp(-0.999, 0.999);
            (r * (df / (1.0 - r * r)).sqrt()) as f32
        };

        let mut by_chr_topic: std::collections::HashMap<(usize, usize), Vec<usize>> =
            Default::default();
        for pp in 0..p {
            by_chr_topic
                .entry((chr[pp], ptopic[pp]))
                .or_default()
                .push(pp);
        }
        let chr_peaks: Vec<Vec<usize>> = (0..n_chr)
            .map(|c| (0..p).filter(|&pp| chr[pp] == c).collect())
            .collect();

        let params = KnockoffParams {
            ridge: 0.05,
            seed: 3,
        };
        let (mut w_base, mut w_loco, mut labels) =
            (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<u8>::new());

        for gid in 0..n_genes {
            let c = gid % n_chr;
            let t = gid % k;
            let topic_peaks = by_chr_topic.get(&(c, t)).cloned().unwrap_or_default();
            if topic_peaks.len() < 10 {
                continue;
            }
            let causal: Vec<usize> = topic_peaks[..3].to_vec();
            let confound: Vec<usize> = topic_peaks[3..10].to_vec();
            let cisnull: Vec<usize> = chr_peaks[c]
                .iter()
                .copied()
                .filter(|pp| ptopic[*pp] != t)
                .take(10)
                .collect();
            let cis: Vec<usize> = causal
                .iter()
                .chain(&confound)
                .chain(&cisnull)
                .copied()
                .collect();
            let lab: Vec<u8> = std::iter::repeat_n(0u8, causal.len())
                .chain(std::iter::repeat_n(1, confound.len()))
                .chain(std::iter::repeat_n(2, cisnull.len()))
                .collect();

            // gene = Σ causal accessibility (Model-2) + noise.
            let mut gene = vec![0f32; s];
            for ss in 0..s {
                let mut v = 3.0;
                for &cp in &causal {
                    v += 1.2 * topics[t][ss] + 0.8 * spec[cp][ss];
                }
                v += 0.4 * randn(&mut rng);
                gene[ss] = v.max(0.01);
            }
            let gx = clog(&gene);
            let w = &w_by_chr[c];
            let mm = w.ncols();
            let gx_r = resid(w, &gx);

            // marginal (baseline) and LOCO-DML (residualized) z + R, then knockoff.
            let cl: Vec<&DVector<f32>> = cis.iter().map(|&pp| &peak_clog[pp]).collect();
            let cl_r: Vec<DVector<f32>> = cis.iter().map(|&pp| resid(w, &peak_clog[pp])).collect();

            let z_base: Vec<f32> = cl
                .iter()
                .map(|pr| tstat(corr(&gx, pr), (s - 2) as f64))
                .collect();
            let z_loco: Vec<f32> = cl_r
                .iter()
                .map(|pr| tstat(corr(&gx_r, pr), (s - mm - 2) as f64))
                .collect();
            let cc = cis.len();
            let mut r_base = Mat::identity(cc, cc);
            let mut r_loco = Mat::identity(cc, cc);
            for i in 0..cc {
                for j in (i + 1)..cc {
                    let rb = corr(cl[i], cl[j]) as f32;
                    let rl = corr(&cl_r[i], &cl_r[j]) as f32;
                    r_base[(i, j)] = rb;
                    r_base[(j, i)] = rb;
                    r_loco[(i, j)] = rl;
                    r_loco[(j, i)] = rl;
                }
            }
            let wb = knockoff_w(&z_base, &r_base, &params, gid);
            let wl = knockoff_w(&z_loco, &r_loco, &params, gid);
            w_base.extend(wb);
            w_loco.extend(wl);
            labels.extend(lab);
        }

        let decomp = |allw: &[f32], name: &str| -> (f64, f64) {
            let t = knockoff_threshold(allw, q);
            let sel: Vec<usize> = (0..allw.len())
                .filter(|&i| allw[i].is_finite() && allw[i] >= t)
                .collect();
            let n = sel.len();
            let ca = sel.iter().filter(|&&i| labels[i] == 0).count();
            let co = sel.iter().filter(|&&i| labels[i] == 1).count();
            let nu = sel.iter().filter(|&&i| labels[i] == 2).count();
            let tot_causal = labels.iter().filter(|&&l| l == 0).count();
            let fdp = if n > 0 {
                (co + nu) as f64 / n as f64
            } else {
                0.0
            };
            let power = ca as f64 / tot_causal.max(1) as f64;
            eprintln!(
                "{name:>12}: sel={n:>3} causal={ca:>3} confound={co:>3} null={nu:>3} FDP={fdp:.2} power={power:.2}"
            );
            (fdp, power)
        };
        let (fdp_base, pow_base) = decomp(&w_base, "marginal");
        let (fdp_loco, pow_loco) = decomp(&w_loco, "LOCO+DML");
        // On identifiable confounding both variants control FDP via R, with power.
        assert!(
            fdp_base <= q + 0.10,
            "marginal (full-R) FDP {fdp_base:.2} not controlled"
        );
        assert!(
            fdp_loco <= q + 0.12,
            "LOCO+DML FDP {fdp_loco:.2} not controlled"
        );
        assert!(
            pow_base > 0.5 && pow_loco > 0.5,
            "power too low (marginal {pow_base:.2}, LOCO {pow_loco:.2})"
        );
    }
}
