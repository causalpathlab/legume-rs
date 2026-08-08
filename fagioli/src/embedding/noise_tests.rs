//! Tests for the noise model.
//!
//! The load-bearing one is [`test_snp_space_negatives_match_eigenspace_draw`]:
//! it checks empirically that the expensive `O(pK)` SNP-space route the plan
//! called for produces the same distribution as the `O(K)` scalar draw. That
//! equivalence is what removes the justification for preferring NCE to
//! maximum likelihood, so it is worth testing rather than merely deriving.

use super::*;
use matrix_util::traits::{RandomizedAlgs, SampleOps};
use nalgebra::DVector;

fn spectrum(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let frac = i as f32 / k as f32;
            (4.0 * (1.0 - frac)).exp() * 0.05
        })
        .collect()
}

#[test]
fn test_negative_scale_is_one_when_null_whitens() {
    // c = 1 and λ = τ is exactly the calibrated whitening condition.
    let d_sq = spectrum(200);
    let model = NoiseModel::new(&[d_sq], 1.0, 2.0, 2.0);
    for &s in &model.scale[0] {
        assert!((s - 1.0).abs() < 1e-5, "expected unit sd, got {s}");
    }
}

#[test]
fn test_negative_scale_departs_from_one_when_lambda_is_wrong() {
    let d_sq = spectrum(200);
    let right = NoiseModel::new(std::slice::from_ref(&d_sq), 1.0, 2.0, 2.0);
    let wrong = NoiseModel::new(&[d_sq], 1.0, 2.0, 0.002);

    let dev = |m: &NoiseModel| {
        m.scale[0]
            .iter()
            .map(|s| (s - 1.0).abs())
            .fold(0.0f32, f32::max)
    };
    let (d_right, d_wrong) = (dev(&right), dev(&wrong));
    println!("max |sd-1|: λ=τ {d_right:.4}, λ=0.002 {d_wrong:.4}");
    assert!(d_right < 1e-4);
    assert!(
        d_wrong > 1.0,
        "an under-regularized λ should leave the small-d tail badly hot, got {d_wrong}"
    );
}

/// The claim that makes sampling cheap: drawing `ε ~ N(0, cR + τI)` in SNP
/// space and projecting through `D̃⁻¹V_R'` gives the same per-coordinate
/// variance as drawing the scalar directly. If this ever fails, the `O(K)`
/// shortcut is unsound and the `O(pK)` route has to come back.
#[test]
fn test_snp_space_negatives_match_eigenspace_draw() {
    let n = 400;
    let p = 120;
    let max_rank = 60;
    let (c, tau) = (1.0f32, 1.5f32);
    let lambda = 1.5f64;

    // A genotype-like matrix with real LD, so R is not the identity.
    let mut rng = SmallRng::seed_from_u64(4);
    let base = DMatrix::<f32>::rnorm(n, p / 4);
    let mut x = DMatrix::<f32>::zeros(n, p);
    for j in 0..p {
        let anchor = j / 4;
        let noise: DVector<f32> = DVector::from_fn(n, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        });
        x.set_column(j, &(base.column(anchor) * 0.8 + noise * 0.6));
    }
    let mut x_std = x.clone();
    {
        use matrix_util::traits::MatOps;
        x_std.scale_columns_inplace();
    }

    let x_scaled = &x_std * (1.0 / (n as f32).sqrt());
    let (_u, d, v_r) = x_scaled.rsvd(max_rank).unwrap();
    let k = d.len();
    let d_sq: Vec<f32> = d.iter().map(|&di| di * di).collect();

    // A with A A' = c R + τ I, applied as
    //   A g = √τ g + V_R[√(c d² + τ) − √τ] V_R' g.
    let draws = 4000usize;
    let mut var_snp = vec![0.0f64; k];
    for _ in 0..draws {
        let g: DVector<f32> = DVector::from_fn(p, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        });
        let vt_g = v_r.tr_mul(&g);
        let mut eps = &g * tau.sqrt();
        let mut adjust = DVector::<f32>::zeros(k);
        for ki in 0..k {
            adjust[ki] = ((c * d_sq[ki] + tau).sqrt() - tau.sqrt()) * vt_g[ki];
        }
        eps += &v_r * adjust;

        // Push through the same whitening the real data gets.
        let vt_eps = v_r.tr_mul(&eps);
        for ki in 0..k {
            let z = vt_eps[ki] / (d_sq[ki] + lambda as f32).sqrt();
            var_snp[ki] += (z * z) as f64;
        }
    }

    let model = NoiseModel::new(std::slice::from_ref(&d_sq), c, tau, lambda);
    let predicted: Vec<f32> = model.scale[0].iter().map(|s| s * s).collect();

    // Compare empirical variance from the SNP-space route against the scalar.
    let mut worst = 0.0f32;
    for ki in 0..k {
        let emp = (var_snp[ki] / draws as f64) as f32;
        let rel = ((emp - predicted[ki]) / predicted[ki].max(1e-6)).abs();
        worst = worst.max(rel);
    }
    println!("worst relative variance mismatch over {k} coordinates: {worst:.4}");
    assert!(
        worst < 0.15,
        "SNP-space and eigenspace negatives should agree; worst relative error {worst}"
    );
}

#[test]
fn test_sampled_negatives_have_the_stated_variance() {
    let d_sq = spectrum(64);
    let model = NoiseModel::new(&[d_sq], 1.0, 2.0, 0.5);
    let t = 3;
    let draws = 3000;

    let k = model.scale[0].len();
    let mut acc = vec![0.0f64; k];
    for r in 0..draws {
        let mut rng = NoiseModel::block_rng(11, 0, r);
        let s = model.sample_block(0, t, &mut rng);
        for ki in 0..k {
            for tt in 0..t {
                acc[ki] += (s[(ki, tt)] * s[(ki, tt)]) as f64;
            }
        }
    }

    let mut worst = 0.0f32;
    for (ki, &sum) in acc.iter().enumerate() {
        let emp = (sum / (draws * t) as f64) as f32;
        let want = model.scale[0][ki].powi(2);
        worst = worst.max(((emp - want) / want).abs());
    }
    println!("worst relative variance error: {worst:.4}");
    assert!(worst < 0.12, "sampled variance off by {worst}");
}

#[test]
fn test_block_rng_is_deterministic_and_varies_by_stream() {
    let d_sq = spectrum(16);
    let model = NoiseModel::new(&[d_sq], 1.0, 1.0, 1.0);

    let a = model.sample_block(0, 2, &mut NoiseModel::block_rng(7, 0, 0));
    let b = model.sample_block(0, 2, &mut NoiseModel::block_rng(7, 0, 0));
    let c = model.sample_block(0, 2, &mut NoiseModel::block_rng(7, 0, 1));
    let d = model.sample_block(0, 2, &mut NoiseModel::block_rng(8, 0, 0));

    assert_eq!(a, b, "same stream must reproduce");
    assert_ne!(a, c, "different replicate must differ");
    assert_ne!(a, d, "different seed must differ");
}

#[test]
fn test_omega_sqrt_pair_inverts_and_reproduces_omega() {
    let t = 5;
    let mut rng = SmallRng::seed_from_u64(23);
    let l = DMatrix::from_fn(t, 3, |_, _| {
        let v: f64 = StandardNormal.sample(&mut rng);
        v as f32
    });
    let mut omega = &l * l.transpose();
    for i in 0..t {
        omega[(i, i)] += 1.0;
    }

    let (sqrt, inv_sqrt) = omega_sqrt_pair(&omega).unwrap();

    // Ω^{1/2} Ω^{1/2} ≈ Ω
    let recon = &sqrt * &sqrt;
    let err = (&recon - &omega).abs().max();
    println!("‖Ω^{{1/2}}Ω^{{1/2}} − Ω‖_max = {err:.5}");
    assert!(err < 1e-2, "square root does not reproduce Ω: {err}");

    // Ω^{1/2} Ω^{-1/2} ≈ I
    let ident = &sqrt * &inv_sqrt;
    let ident_err = (0..t)
        .flat_map(|i| (0..t).map(move |j| (i, j)))
        .map(|(i, j)| (ident[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs())
        .fold(0.0f32, f32::max);
    assert!(ident_err < 1e-3, "inverse square root is off by {ident_err}");
}

/// A near-singular Ω is the normal case when two traits are near-duplicates;
/// the ridge has to keep Ω^{-1/2} finite.
#[test]
fn test_omega_sqrt_pair_survives_a_duplicate_trait() {
    let t = 4;
    let mut omega = DMatrix::<f32>::identity(t, t);
    omega[(0, 1)] = 0.999;
    omega[(1, 0)] = 0.999;

    let (sqrt, inv_sqrt) = omega_sqrt_pair(&omega).unwrap();
    assert!(sqrt.iter().all(|v| v.is_finite()));
    assert!(inv_sqrt.iter().all(|v| v.is_finite()));
    assert!(
        inv_sqrt.abs().max() < 1e4,
        "Ω^{{-1/2}} blew up: {}",
        inv_sqrt.abs().max()
    );
}
