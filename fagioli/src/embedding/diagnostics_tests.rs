//! Tests for the geometry falsification test.
//!
//! The point of these is adversarial: the diagnostic has to *fail* a fitted
//! geometry that reproduces sample overlap, including when overlap and genetic
//! correlation are deliberately confounded. A diagnostic that only passes good
//! inputs is worthless.

use super::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

fn spectrum(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let frac = i as f32 / k as f32;
            (5.0 * (1.0 - frac)).exp() * 0.02
        })
        .collect()
}

/// A PSD `T x T` matrix built as `L L' + ridge I` from an r-column loading.
fn psd_from_loadings(loadings: &DMatrix<f32>, ridge: f32) -> DMatrix<f32> {
    let t = loadings.nrows();
    let mut m = loadings * loadings.transpose();
    for i in 0..t {
        m[(i, i)] += ridge;
    }
    m
}

fn random_loadings(t: usize, r: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    DMatrix::from_fn(t, r, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    })
}

fn chol_l(m: &DMatrix<f32>) -> DMatrix<f32> {
    m.clone()
        .cholesky()
        .expect("matrix should be positive definite")
        .l()
}

/// Draw `(V'z)` whose cross-moment is exactly `d⁴ G + c d² I + Ω`.
fn simulate_vt_z(
    d_sq: &[f32],
    g_true: &DMatrix<f32>,
    omega_true: &DMatrix<f32>,
    c: f32,
    rng: &mut SmallRng,
) -> DMatrix<f32> {
    let k = d_sq.len();
    let t = g_true.nrows();
    let lg = chol_l(g_true);
    let lo = chol_l(omega_true);

    let mut out = DMatrix::<f32>::zeros(k, t);
    for ki in 0..k {
        let d2 = d_sq[ki];
        // genetic part: d² · L_g η,  η ~ N(0, I)  →  contributes d⁴ G
        let eta = DVector::from_fn(t, |_, _| {
            let v: f64 = StandardNormal.sample(rng);
            v as f32
        });
        let gen = &lg * &eta * d2;
        // overlap part: L_o ξ  →  contributes Ω
        let xi = DVector::from_fn(t, |_, _| {
            let v: f64 = StandardNormal.sample(rng);
            v as f32
        });
        let ovl = &lo * &xi;
        // LD noise: independent across traits, variance c·d²
        for tt in 0..t {
            let e: f64 = StandardNormal.sample(rng);
            let ld = (c * d2).sqrt() * e as f32;
            out[(ki, tt)] = gen[tt] + ovl[tt] + ld;
        }
    }
    out
}

/// Build a `TraitGeometry` through the real code path, minus genotype I/O.
fn synthetic_geometry(
    g_true: &DMatrix<f32>,
    omega_true: &DMatrix<f32>,
    num_blocks: usize,
    k: usize,
    seed: u64,
) -> TraitGeometry {
    let t = g_true.nrows();
    let d_sq = spectrum(k);
    let mut rng = SmallRng::seed_from_u64(seed);

    let per_block: Vec<CrossTraitMoments> = (0..num_blocks)
        .map(|_| {
            let vt_z = simulate_vt_z(&d_sq, g_true, omega_true, 1.0, &mut rng);
            cross_trait_moments(&d_sq, &vt_z, k).expect("moment fit")
        })
        .collect();

    let mut sum_gen = DMatrix::<f32>::zeros(t, t);
    let mut sum_ovl = DMatrix::<f32>::zeros(t, t);
    for m in &per_block {
        sum_gen += &m.genetic_cov;
        sum_ovl += &m.overlap;
    }

    TraitGeometry {
        genetic_correlation: to_correlation(&sum_gen),
        overlap_correlation: to_correlation(&sum_ovl),
        per_block,
    }
}

/// Correlation between the strict upper triangles of two matrices.
fn offdiag_corr(a: &DMatrix<f32>, b: &DMatrix<f32>) -> f32 {
    let t = a.nrows();
    let (mut va, mut vb) = (Vec::new(), Vec::new());
    for i in 0..t {
        for j in (i + 1)..t {
            va.push(a[(i, j)]);
            vb.push(b[(i, j)]);
        }
    }
    let n = va.len() as f32;
    let (ma, mb) = (va.iter().sum::<f32>() / n, vb.iter().sum::<f32>() / n);
    let mut sab = 0.0f64;
    let (mut saa, mut sbb) = (0.0f64, 0.0f64);
    for i in 0..va.len() {
        let (da, db) = ((va[i] - ma) as f64, (vb[i] - mb) as f64);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    (sab / (saa * sbb).sqrt()) as f32
}

use nalgebra::DVector;

#[test]
fn test_cross_trait_moments_recover_covariance_and_overlap() {
    let t = 8;
    let k = 3000;
    let mut rng = SmallRng::seed_from_u64(3);

    let g_true = psd_from_loadings(&random_loadings(t, 3, &mut rng), 0.2);
    let omega_true = psd_from_loadings(&random_loadings(t, 2, &mut rng), 0.5);

    // Average many blocks to beat down the chi-square noise in the moments.
    let geom = synthetic_geometry(&g_true, &omega_true, 40, k, 17);

    let rg_true = to_correlation(&g_true);
    let ovl_true = to_correlation(&omega_true);

    let rg_corr = offdiag_corr(&geom.genetic_correlation, &rg_true);
    let ovl_corr = offdiag_corr(&geom.overlap_correlation, &ovl_true);
    println!("recovery: r_g {rg_corr:.3}, Ω {ovl_corr:.3}");

    assert!(rg_corr > 0.85, "genetic correlation not recovered: {rg_corr}");
    assert!(ovl_corr > 0.85, "overlap not recovered: {ovl_corr}");
}

/// A geometry that reproduces `r_g` must pass.
#[test]
fn test_verdict_passes_when_fitted_tracks_rg() {
    let t = 10;
    let mut rng = SmallRng::seed_from_u64(5);
    let g_true = psd_from_loadings(&random_loadings(t, 3, &mut rng), 0.2);
    let omega_true = psd_from_loadings(&random_loadings(t, 2, &mut rng), 0.5);

    let geom = synthetic_geometry(&g_true, &omega_true, 30, 2500, 21);

    // Stand-in for a fit that recovered the true genetic geometry.
    let verdict = compare_geometry(&to_correlation(&g_true), &geom).expect("verdict");

    println!(
        "passes={} partial_rg={:.3} partial_Ω={:.3} SE={:.3} corr(r_g,Ω)={:.3}",
        verdict.passes,
        verdict.partial_rg,
        verdict.partial_overlap,
        verdict.partial_rg_se,
        verdict.corr_rg_overlap
    );

    assert!(verdict.passes, "a geometry equal to r_g should pass");
    assert!(verdict.partial_rg > 0.5, "partial r_g too weak: {}", verdict.partial_rg);
    assert!(
        verdict.partial_rg.abs() > verdict.partial_overlap.abs(),
        "r_g should dominate Ω"
    );
    assert!(verdict.partial_rg_se > 0.0, "jackknife SE should be positive");
}

/// The test that matters: a geometry that reproduces sample overlap must FAIL,
/// because that is the failure mode the whole design exists to catch.
#[test]
fn test_verdict_fails_when_fitted_tracks_overlap() {
    let t = 10;
    let mut rng = SmallRng::seed_from_u64(5);
    let g_true = psd_from_loadings(&random_loadings(t, 3, &mut rng), 0.2);
    let omega_true = psd_from_loadings(&random_loadings(t, 2, &mut rng), 0.5);

    let geom = synthetic_geometry(&g_true, &omega_true, 30, 2500, 21);

    // A fit that learned cohort structure instead of biology.
    let verdict = compare_geometry(&to_correlation(&omega_true), &geom).expect("verdict");

    println!(
        "passes={} partial_rg={:.3} partial_Ω={:.3}",
        verdict.passes, verdict.partial_rg, verdict.partial_overlap
    );

    assert!(
        !verdict.passes,
        "a geometry equal to Ω must not pass: partial_rg={} partial_Ω={}",
        verdict.partial_rg, verdict.partial_overlap
    );
    assert!(
        verdict.partial_overlap.abs() > verdict.partial_rg.abs(),
        "Ω should dominate r_g for an overlap-driven fit"
    );
}

/// When `r_g` and `Ω` are deliberately confounded, the marginal correlation is
/// misleading and only the partial correlation separates them.
#[test]
fn test_partial_correlation_separates_confounded_rg_and_overlap() {
    let t = 12;
    let mut rng = SmallRng::seed_from_u64(9);

    // Ω shares most of its loading structure with G.
    let l_g = random_loadings(t, 3, &mut rng);
    let l_indep = random_loadings(t, 3, &mut rng);
    let alpha = 0.85f32;
    let l_o = &l_g * alpha + &l_indep * (1.0 - alpha * alpha).sqrt();

    let g_true = psd_from_loadings(&l_g, 0.2);
    let omega_true = psd_from_loadings(&l_o, 0.4);

    let geom = synthetic_geometry(&g_true, &omega_true, 30, 2500, 33);

    let verdict = compare_geometry(&to_correlation(&omega_true), &geom).expect("verdict");
    println!(
        "confounded: corr(r_g,Ω)={:.3}; marginal rg={:.3} Ω={:.3}; partial rg={:.3} Ω={:.3}",
        verdict.corr_rg_overlap,
        verdict.corr_rg,
        verdict.corr_overlap,
        verdict.partial_rg,
        verdict.partial_overlap
    );

    // The confound is real: marginally, an overlap-driven fit still looks like
    // it correlates with r_g.
    assert!(
        verdict.corr_rg_overlap.abs() > 0.5,
        "this test needs r_g and Ω confounded, got {}",
        verdict.corr_rg_overlap
    );
    assert!(
        verdict.corr_rg > 0.3,
        "marginal correlation with r_g should be misleadingly high, got {}",
        verdict.corr_rg
    );
    // But the partial correlation is not fooled.
    assert!(
        !verdict.passes,
        "partial correlation should still reject an overlap-driven fit"
    );
}

/// The jackknife SE reads ~0 on 30 large clean blocks, which is correct but
/// indistinguishable from an inert estimator. Give it few, short, noisy blocks
/// and it must respond.
#[test]
fn test_jackknife_se_grows_when_blocks_are_few_and_noisy() {
    let t = 10;
    let mut rng = SmallRng::seed_from_u64(5);
    let g_true = psd_from_loadings(&random_loadings(t, 3, &mut rng), 0.2);
    let omega_true = psd_from_loadings(&random_loadings(t, 2, &mut rng), 0.5);
    let fitted = to_correlation(&g_true);

    // Many long blocks: r_g is pinned down, so leave-one-out barely moves it.
    let clean = synthetic_geometry(&g_true, &omega_true, 30, 2500, 21);
    let se_clean = compare_geometry(&fitted, &clean).expect("verdict").partial_rg_se;

    // Few short blocks: each one carries real weight.
    let noisy = synthetic_geometry(&g_true, &omega_true, 4, 120, 77);
    let se_noisy = compare_geometry(&fitted, &noisy).expect("verdict").partial_rg_se;

    println!("jackknife SE: clean {se_clean:.5}, noisy {se_noisy:.5}");
    assert!(
        se_noisy > 10.0 * se_clean.max(1e-9),
        "SE should grow sharply with block noise: clean {se_clean}, noisy {se_noisy}"
    );
    assert!(se_noisy.is_finite(), "SE must stay finite");
}

#[test]
fn test_to_correlation_has_unit_diagonal_and_is_bounded() {
    let mut rng = SmallRng::seed_from_u64(41);
    let m = psd_from_loadings(&random_loadings(6, 2, &mut rng), 0.3);
    let c = to_correlation(&m);
    for i in 0..6 {
        assert!((c[(i, i)] - 1.0).abs() < 1e-6);
        for j in 0..6 {
            assert!(c[(i, j)] >= -1.0 && c[(i, j)] <= 1.0);
        }
    }
}

#[test]
fn test_to_correlation_survives_a_zero_variance_trait() {
    let mut m = DMatrix::<f32>::identity(4, 4);
    m[(2, 2)] = 0.0;
    let c = to_correlation(&m);
    assert!(c.iter().all(|v| v.is_finite()), "must not produce NaN");
    assert_eq!(c[(2, 0)], 0.0);
}

#[test]
fn test_verdict_declines_with_too_few_trait_pairs() {
    let t = 3; // only 3 pairs, below the floor
    let mut rng = SmallRng::seed_from_u64(2);
    let g_true = psd_from_loadings(&random_loadings(t, 2, &mut rng), 0.3);
    let omega_true = psd_from_loadings(&random_loadings(t, 1, &mut rng), 0.5);
    let geom = synthetic_geometry(&g_true, &omega_true, 5, 1000, 4);

    assert!(compare_geometry(&to_correlation(&g_true), &geom).is_none());
}
