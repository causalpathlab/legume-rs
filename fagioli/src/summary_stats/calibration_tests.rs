//! Tests for eigenspace null calibration.
//!
//! Every test simulates directly from the moment law the module claims,
//! `E[(V'z)²_k] = N σ²_β d⁴_k + c d²_k + τ`, so a failure means the estimator
//! is wrong rather than the data being unrealistic.

use super::*;
use crate::summary_stats::rss_svd::RssSvdNal;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// A decaying eigenvalue spectrum spanning ~two orders of magnitude, which is
/// what a real LD block's `d²` looks like and what makes `d⁴` separable from `d²`.
fn spectrum(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let frac = i as f32 / k as f32;
            (5.0 * (1.0 - frac)).exp() * 0.02
        })
        .collect()
}

/// Draw `(V'z)_k` whose second moment is exactly `poly·d⁴ + c·d² + tau`.
fn simulate_vt_z(
    d_sq: &[f32],
    num_traits: usize,
    poly: f32,
    c: f32,
    tau: f32,
    seed: u64,
) -> DMatrix<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let k = d_sq.len();
    DMatrix::from_fn(k, num_traits, |ki, _| {
        let d2 = d_sq[ki];
        // signal: √(poly) · d² · b,  b ~ N(0,1)  →  E[·²] = poly · d⁴
        let b: f64 = StandardNormal.sample(&mut rng);
        let signal = poly.sqrt() * d2 * b as f32;
        // noise: N(0, c·d² + τ)
        let e: f64 = StandardNormal.sample(&mut rng);
        let noise = (c * d2 + tau).sqrt() * e as f32;
        signal + noise
    })
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

#[test]
fn test_three_term_fit_recovers_all_coefficients() {
    let k = 4000;
    let t = 40;
    let (poly, c, tau) = (0.8f32, 1.0f32, 2.0f32);

    let d_sq = spectrum(k);
    let vt_z = simulate_vt_z(&d_sq, t, poly, c, tau, 7);

    let cal = calibrate_block(&d_sq, &vt_z).expect("fit should identify");

    let (poly_hat, c_hat, tau_hat) = (
        mean(&cal.polygenic),
        mean(&cal.c),
        mean(&cal.tau),
    );
    println!(
        "three-term: poly {:.3} (true {:.3}), c {:.3} (true {:.3}), tau {:.3} (true {:.3})",
        poly_hat, poly, c_hat, c, tau_hat, tau
    );

    assert!(
        (poly_hat - poly).abs() < 0.25 * poly,
        "polygenic coefficient not recovered: {poly_hat} vs {poly}"
    );
    assert!(
        (c_hat - c).abs() < 0.35 * c,
        "LD coefficient not recovered: {c_hat} vs {c}"
    );
    assert!(
        (tau_hat - tau).abs() < 0.35 * tau,
        "nugget not recovered: {tau_hat} vs {tau}"
    );
}

/// The reason the third column exists. With polygenic signal present, a fit on
/// `d²` plus an intercept has nowhere to put the `d⁴` component, so it lands in
/// the `d²` slope — which is supposed to estimate `c` — and drags the intercept
/// with it. Both errors grow with the polygenic strength while the three-term
/// fit stays put, which is the signature of misspecification rather than noise.
#[test]
fn test_two_term_fit_absorbs_the_polygenic_term_into_c() {
    let k = 4000;
    let t = 40;
    let (c, tau) = (1.0f32, 2.0f32);

    let mut two_term_slopes = Vec::new();
    let mut two_term_tau_err = Vec::new();
    let mut three_term_tau_err = Vec::new();

    println!("  poly | 3-term tau | 2-term tau | 2-term slope (true c = 1)");
    for (i, &poly) in [0.0f32, 0.4, 0.8, 1.6].iter().enumerate() {
        let d_sq = spectrum(k);
        let vt_z = simulate_vt_z(&d_sq, t, poly, c, tau, 11 + i as u64);

        let cal = calibrate_block(&d_sq, &vt_z).expect("fit should identify");
        let tau_three = mean(&cal.tau);

        let y_raw: Vec<Vec<f32>> = (0..k)
            .map(|ki| (0..t).map(|tt| vt_z[(ki, tt)]).collect())
            .collect();
        let (intercepts, slopes) = RssSvdNal::estimate_ldsc_intercept(&d_sq, &y_raw, t);
        let tau_two = mean(&intercepts);
        let slope_two = mean(&slopes);

        println!(
            "  {:.1}  |   {:.3}    |   {:.3}    |   {:.3}",
            poly, tau_three, tau_two, slope_two
        );

        two_term_slopes.push(slope_two);
        two_term_tau_err.push((tau_two - tau).abs());
        three_term_tau_err.push((tau_three - tau).abs());

        // The three-term fit is unmoved by how much polygenic signal there is.
        assert!(
            (tau_three - tau).abs() < 0.35 * tau,
            "three-term should recover tau at poly={poly}, got {tau_three}"
        );
    }

    // The two-term d² slope should track the polygenic term rather than c.
    assert!(
        two_term_slopes[0] < 1.5 * c,
        "with no polygenic signal the two-term slope should be near c, got {}",
        two_term_slopes[0]
    );
    assert!(
        two_term_slopes[3] > 3.0 * c,
        "the two-term slope should be badly inflated at high polygenicity, got {}",
        two_term_slopes[3]
    );
    assert!(
        two_term_slopes.windows(2).all(|w| w[1] > w[0]),
        "two-term slope should grow monotonically with polygenicity: {two_term_slopes:?}"
    );

    // And its intercept drifts away from tau while the three-term one does not.
    assert!(
        two_term_tau_err[3] > 3.0 * two_term_tau_err[0],
        "two-term nugget error should grow with polygenicity: {two_term_tau_err:?}"
    );
    assert!(
        two_term_tau_err[3] > 2.0 * three_term_tau_err[3],
        "at high polygenicity the three-term nugget should be the more accurate one \
         ({:?} vs {:?})",
        three_term_tau_err[3],
        two_term_tau_err[3]
    );
}

/// λ = τ is what makes the null flat, and it beats both a too-small and a
/// too-large ridge.
#[test]
fn test_lambda_equal_tau_minimizes_whiteness_deviation() {
    let k = 3000;
    let t = 30;
    let tau = 2.0f32;

    // Pure null: no polygenic term, so E[z̃²] should be exactly flat at λ = τ.
    let d_sq = spectrum(k);
    let vt_z = simulate_vt_z(&d_sq, t, 0.0, 1.0, tau, 23);

    let dev_at = |lam: f64| whiteness_deviation(&whiteness_curve(&d_sq, &vt_z, lam, 12));

    let dev_tau = dev_at(tau as f64);
    let dev_small = dev_at(0.001);
    let dev_large = dev_at(20.0 * tau as f64);

    println!(
        "whiteness deviation: λ=0.001 {:.3}, λ=τ={:.1} {:.3}, λ=20τ {:.3}",
        dev_small, tau, dev_tau, dev_large
    );

    assert!(
        dev_tau < dev_small,
        "λ=τ ({dev_tau}) should beat an under-regularized λ ({dev_small})"
    );
    assert!(
        dev_tau < dev_large,
        "λ=τ ({dev_tau}) should beat an over-regularized λ ({dev_large})"
    );
    assert!(
        dev_tau < 0.2,
        "null should be close to white at λ=τ, deviation {dev_tau}"
    );
}

/// The fitted τ, fed straight back as λ, whitens the block it came from.
#[test]
fn test_fitted_lambda_whitens_its_own_block() {
    let k = 3000;
    let t = 30;
    let tau = 3.0f32;

    let d_sq = spectrum(k);
    let vt_z = simulate_vt_z(&d_sq, t, 0.0, 1.0, tau, 31);

    let cal = calibrate_block(&d_sq, &vt_z).expect("fit should identify");
    let lambda = cal.lambda_white();
    println!("fitted λ_white {:.3} vs true τ {:.3}", lambda, tau);

    let dev = whiteness_deviation(&whiteness_curve(&d_sq, &vt_z, lambda, 12));
    assert!(dev < 0.25, "fitted λ should whiten its own block, deviation {dev}");
}

#[test]
fn test_pooling_uses_medians_and_survives_one_bad_block() {
    let k = 2000;
    let t = 6;
    let d_sq = spectrum(k);

    let mut blocks: Vec<BlockCalibration> = (0..5)
        .map(|i| {
            let vt_z = simulate_vt_z(&d_sq, t, 0.3, 1.0, 2.0, 100 + i);
            calibrate_block(&d_sq, &vt_z).expect("fit")
        })
        .collect();

    // One pathological block with a wildly inflated nugget.
    let bad = simulate_vt_z(&d_sq, t, 0.3, 1.0, 500.0, 999);
    blocks.push(calibrate_block(&d_sq, &bad).expect("fit"));

    let pooled = pool_calibrations(&blocks, t).expect("pooling");
    println!(
        "pooled c {:.3}, tau {:.3} over {} blocks",
        pooled.c, pooled.tau, pooled.num_blocks
    );

    assert_eq!(pooled.num_blocks, 6);
    assert!(
        pooled.tau < 10.0,
        "median pooling should reject the outlier block, got {}",
        pooled.tau
    );
    assert_eq!(pooled.trait_inflation.len(), t);
    assert_eq!(pooled.polygenic.len(), t);
}

#[test]
fn test_short_block_declines_to_fit() {
    let d_sq = spectrum(4);
    let vt_z = DMatrix::<f32>::zeros(4, 3);
    assert!(calibrate_block(&d_sq, &vt_z).is_none());
}

/// At equal group sizes the Busing weighting must collapse to the textbook
/// delete-one jackknife, which is the cheapest check that `h_j` is applied the
/// right way round.
#[test]
fn test_delete_a_group_reduces_to_delete_one_at_equal_sizes() {
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let sizes = [10usize; 5];

    let got = delete_a_group_se(&values, &sizes);

    // Textbook delete-one jackknife SE of a mean: sqrt((g-1)/g * Σ(x_j - x̄)²) / ...
    // For the mean statistic the pseudo-values are just the observations, so the
    // jackknife variance is the usual sample variance divided by g.
    let g = values.len() as f32;
    let mean = values.iter().sum::<f32>() / g;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (g - 1.0);
    let expected = (var / g).sqrt();

    println!("delete-a-group {got:.6} vs delete-one {expected:.6}");
    assert!(
        (got - expected).abs() < 1e-4,
        "equal-size case should match delete-one: {got} vs {expected}"
    );
}

/// With unequal sizes the weighting must actually change the answer, otherwise
/// the `h_j` term is dead code.
#[test]
fn test_delete_a_group_differs_from_unweighted_when_sizes_are_unequal() {
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let lopsided = [200usize, 10, 10, 10, 10];
    let equal = [50usize; 5];

    let se_lopsided = delete_a_group_se(&values, &lopsided);
    let se_equal = delete_a_group_se(&values, &equal);

    println!("SE: lopsided {se_lopsided:.6}, equal {se_equal:.6}");
    assert!(se_lopsided > 0.0 && se_equal > 0.0);
    assert!(
        (se_lopsided - se_equal).abs() > 1e-3,
        "unequal block sizes should change the SE ({se_lopsided} vs {se_equal})"
    );
}

#[test]
fn test_delete_a_group_is_zero_for_a_constant() {
    let values = [3.0f32; 6];
    let sizes = [11usize, 250, 40, 7, 900, 33];
    assert!(delete_a_group_se(&values, &sizes).abs() < 1e-5);
}

#[test]
fn test_whiteness_curve_bins_are_ordered_and_complete() {
    let k = 500;
    let d_sq = spectrum(k);
    let vt_z = simulate_vt_z(&d_sq, 4, 0.0, 1.0, 1.0, 5);

    let curve = whiteness_curve(&d_sq, &vt_z, 1.0, 10);
    assert_eq!(curve.iter().map(|b| b.count).sum::<usize>(), k);
    for w in curve.windows(2) {
        assert!(w[0].d_sq <= w[1].d_sq, "bins should ascend in d²");
    }
}

/// Real LD spectra are concentrated, which makes the `d⁴` and `d²` columns
/// nearly collinear and the three-term fit ill-conditioned. This measures how
/// bad it gets, comparing the synthetic spectrum used above against one taken
/// from genotype-like data.
#[test]
fn test_design_conditioning_on_realistic_spectra() {
    use matrix_util::traits::{MatOps, RandomizedAlgs};
    use rand_distr::StandardNormal as SN;

    fn corr_d4_d2(d_sq: &[f32]) -> f32 {
        let n = d_sq.len() as f64;
        let x: Vec<f64> = d_sq.iter().map(|&v| (v as f64) * (v as f64)).collect();
        let y: Vec<f64> = d_sq.iter().map(|&v| v as f64).collect();
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..d_sq.len() {
            let (dx, dy) = (x[i] - mx, y[i] - my);
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        (sxy / (sxx * syy).sqrt()) as f32
    }

    // The synthetic spectrum these unit tests use.
    let synthetic = spectrum(2000);
    let c_syn = corr_d4_d2(&synthetic);

    // A spectrum from genotype-like data with block LD.
    let (n, p, n_hap) = (800usize, 120usize, 6usize);
    let mut rng = SmallRng::seed_from_u64(3);
    let hap = DMatrix::from_fn(n, n_hap, |_, _| {
        let v: f64 = SN.sample(&mut rng);
        v as f32
    });
    let mut x = DMatrix::<f32>::zeros(n, p);
    for j in 0..p {
        let rho = 0.85 - 0.4 * (j as f32 / p as f32);
        for i in 0..n {
            let e: f64 = SN.sample(&mut rng);
            let latent = rho * hap[(i, j % n_hap)] + (1.0 - rho * rho).sqrt() * e as f32;
            x[(i, j)] = if latent < -0.6 { 0.0 } else if latent < 0.6 { 1.0 } else { 2.0 };
        }
    }
    x.scale_columns_inplace();
    let (_u, d, _v) = (&x * (1.0 / (n as f32).sqrt())).rsvd(80).unwrap();
    let real: Vec<f32> = d.iter().map(|&v| v * v).collect();
    let c_real = corr_d4_d2(&real);

    let top = real.first().copied().unwrap_or(0.0);
    let bot = real.last().copied().unwrap_or(0.0);
    println!(
        "corr(d⁴, d²): synthetic {c_syn:.4}, genotype-like {c_real:.4}; \
         spectrum range {bot:.4} .. {top:.4} (ratio {:.1})",
        top / bot.max(1e-9)
    );

    // Not an assertion about which is "right" -- this documents that the design
    // the calibration relies on is far better conditioned on the synthetic
    // spectrum than on a realistic one.
    assert!(c_real > 0.0 && c_syn > 0.0);
}
