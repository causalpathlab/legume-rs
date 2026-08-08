//! Tests for the two-axis whitening.

use super::*;
use crate::embedding::noise::omega_sqrt_pair;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

fn random_matrix(r: usize, c: usize, seed: u64) -> DMatrix<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    DMatrix::from_fn(r, c, |_, _| {
        let v: f64 = StandardNormal.sample(&mut rng);
        v as f32
    })
}

/// `unwhiten_traits` must invert exactly what the model absorbed:
/// `V̌ = Ω^{-1/2} Λ_N V`, so `V = Λ_N⁻¹ Ω^{1/2} V̌`.
#[test]
fn test_unwhiten_inverts_the_trait_side_whitening() {
    let t = 6;
    let h = 3;

    let l = random_matrix(t, 2, 5);
    let mut omega = &l * l.transpose();
    for i in 0..t {
        omega[(i, i)] += 1.0;
    }
    let (omega_sqrt, omega_inv_sqrt) = omega_sqrt_pair(&omega).unwrap();

    let sqrt_n: Vec<f32> = (0..t).map(|i| ((i + 3) as f32 * 100.0).sqrt()).collect();
    let v_true = random_matrix(t, h, 9);

    // Forward: what the model actually learns.
    let mut v_check = v_true.clone();
    for (i, &s) in sqrt_n.iter().enumerate() {
        v_check.row_mut(i).scale_mut(s);
    }
    let v_check = &omega_inv_sqrt * v_check;

    let recovered = unwhiten_traits(&v_check, Some(&omega_sqrt), &sqrt_n);

    let err = (&recovered - &v_true).abs().max();
    println!("‖V̂ − V‖_max = {err:.6}");
    assert!(err < 1e-3, "round trip lost the trait embedding: {err}");
}

#[test]
fn test_unwhiten_ignores_zero_sample_sizes() {
    let t = 3;
    let h = 2;
    let v_check = random_matrix(t, h, 11);
    // A zero N would divide by zero; the row should pass through untouched.
    // Ω = I here, exercising the default no-overlap path.
    let recovered = unwhiten_traits(&v_check, None, &[0.0, 2.0, 0.0]);
    assert!(recovered.iter().all(|v| v.is_finite()));
    assert!((recovered[(0, 0)] - v_check[(0, 0)]).abs() < 1e-6);
    assert!((recovered[(1, 0)] - v_check[(1, 0)] / 2.0).abs() < 1e-6);
}

/// Under the null, whitening both axes should leave `ž` standard normal — that
/// is the property the whole objective rests on.
#[test]
fn test_null_zscores_whiten_to_unit_variance() {
    use matrix_util::traits::{MatOps, RandomizedAlgs, SampleOps};

    let n = 500;
    let p = 100;
    let t = 3;
    let max_rank = 50;

    // Genotypes with LD, so R is far from the identity.
    let mut rng = SmallRng::seed_from_u64(31);
    let base = DMatrix::<f32>::rnorm(n, p / 5);
    let mut x = DMatrix::<f32>::zeros(n, p);
    for j in 0..p {
        let anchor = j / 5;
        let noise = DMatrix::from_fn(n, 1, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        });
        x.set_column(j, &(base.column(anchor) * 0.9 + noise.column(0) * 0.5));
    }
    x.scale_columns_inplace();

    let x_scaled = &x * (1.0 / (n as f32).sqrt());
    let (_u, d, v_r) = x_scaled.rsvd(max_rank).unwrap();
    let k = d.len();
    let d_sq: Vec<f32> = d.iter().map(|&di| di * di).collect();

    // Null z-scores: z ~ N(0, R) per trait, i.e. z = V_R D η.
    let reps = 400;
    let lambda = 0.0f32; // exact null, no nugget, so λ = τ = 0
    let mut acc = vec![0.0f64; k];
    for _ in 0..reps {
        let eta = DMatrix::from_fn(k, t, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        });
        let scaled = DMatrix::from_fn(k, t, |ki, tt| d[ki] * eta[(ki, tt)]);
        let z = &v_r * scaled;

        // D̃⁻¹ V_R' z
        let vt_z = v_r.tr_mul(&z);
        for ki in 0..k {
            let s = (d_sq[ki] + lambda).sqrt().max(f32::MIN_POSITIVE);
            for tt in 0..t {
                let w = vt_z[(ki, tt)] / s;
                acc[ki] += (w * w) as f64;
            }
        }
    }

    let mean_var: f64 = acc.iter().sum::<f64>() / (k * t * reps) as f64;
    println!("mean whitened variance under the null: {mean_var:.4}");
    assert!(
        (mean_var - 1.0).abs() < 0.1,
        "whitened null should have unit variance, got {mean_var}"
    );
}
