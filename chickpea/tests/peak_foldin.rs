//! Peak rows folded in against frozen pseudobulk embeddings: one IRLS step of
//! the per-peak Poisson GLM `n_pu ~ Poisson(s_u exp(<e_u, φ_p> + b_p))`, for all
//! peaks at once.

use chickpea::p2g::peak_foldin::fold_in_peaks;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const U: usize = 60;
const H: usize = 3;

fn units(rng: &mut StdRng) -> (DMatrix<f32>, Vec<f32>) {
    let e = DMatrix::from_fn(U, H, |_, _| rng.random::<f32>() - 0.5);
    let s: Vec<f32> = (0..U)
        .map(|_| 1_000.0 + 1_000.0 * rng.random::<f32>())
        .collect();
    (e, s)
}

/// Expected counts of each peak (rows) in each unit (columns).
fn expected(e: &DMatrix<f32>, s: &[f32], phi: &DMatrix<f32>, b: &[f32]) -> CsrMatrix<f32> {
    let mut coo = CooMatrix::new(phi.nrows(), U);
    for p in 0..phi.nrows() {
        for u in 0..U {
            let eta: f32 = (0..H).map(|k| e[(u, k)] * phi[(p, k)]).sum::<f32>() + b[p];
            coo.push(p, u, s[u] * eta.exp());
        }
    }
    CsrMatrix::from(&coo)
}

/// Exact MAP of one peak's GLM (φ and b jointly) by Newton's method.
fn exact_glm(e: &DMatrix<f32>, s: &[f32], n: &[f32]) -> DVector<f64> {
    let x = DMatrix::<f64>::from_fn(
        U,
        H + 1,
        |u, k| if k < H { f64::from(e[(u, k)]) } else { 1.0 },
    );
    let mut beta = DVector::<f64>::zeros(H + 1);
    let tot_n: f64 = n.iter().map(|&v| f64::from(v)).sum();
    let tot_s: f64 = s.iter().map(|&v| f64::from(v)).sum();
    beta[H] = (tot_n / tot_s).ln();
    for _ in 0..50 {
        let mu: Vec<f64> = (0..U)
            .map(|u| f64::from(s[u]) * (x.row(u) * &beta)[0].exp())
            .collect();
        let grad = x.transpose() * DVector::from_fn(U, |u, _| f64::from(n[u]) - mu[u]);
        let hess = x.transpose() * DMatrix::from_diagonal(&DVector::from_vec(mu.clone())) * &x;
        beta += hess.lu().solve(&grad).unwrap();
    }
    beta
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (n(a) * n(b)).max(1e-12)
}

#[test]
fn a_weak_planted_signal_is_recovered() {
    let mut rng = StdRng::seed_from_u64(1);
    let (e, s) = units(&mut rng);
    let phi = DMatrix::from_fn(20, H, |_, _| 0.1 * (rng.random::<f32>() - 0.5));
    let b: Vec<f32> = (0..20).map(|p| -4.0 - 0.1 * p as f32).collect();
    let got = fold_in_peaks(&expected(&e, &s, &phi, &b), &s, &e, 1e-6).unwrap();
    for p in 0..20 {
        for k in 0..H {
            let err = (got.phi[(p, k)] - phi[(p, k)]).abs();
            assert!(
                err < 0.01,
                "peak {p} coord {k}: {} vs {}",
                got.phi[(p, k)],
                phi[(p, k)]
            );
        }
        assert!(
            (got.bias[p] - b[p]).abs() < 0.05,
            "peak {p} bias {} vs {}",
            got.bias[p],
            b[p]
        );
    }
}

#[test]
fn a_moderate_signal_points_where_the_exact_glm_does() {
    let mut rng = StdRng::seed_from_u64(2);
    let (e, s) = units(&mut rng);
    let phi = DMatrix::from_fn(20, H, |_, _| 2.0 * (rng.random::<f32>() - 0.5));
    let b = vec![-4.0f32; 20];
    let counts = expected(&e, &s, &phi, &b);
    let got = fold_in_peaks(&counts, &s, &e, 1e-6).unwrap();
    for p in 0..20 {
        let row: Vec<f32> = counts.row(p).values().to_vec();
        let exact = exact_glm(&e, &s, &row);
        let exact_phi: Vec<f32> = (0..H).map(|k| exact[k] as f32).collect();
        let one_step: Vec<f32> = got.phi.row(p).iter().copied().collect();
        let c = cosine(&one_step, &exact_phi);
        assert!(c > 0.95, "peak {p}: cosine {c:.3} to the exact GLM");
    }
}

#[test]
fn an_empty_peak_gets_a_zero_row_and_a_finite_bias() {
    let mut rng = StdRng::seed_from_u64(3);
    let (e, s) = units(&mut rng);
    let mut coo = CooMatrix::new(2, U);
    coo.push(0, 0, 5.0);
    let got = fold_in_peaks(&CsrMatrix::from(&coo), &s, &e, 1e-6).unwrap();
    assert!(got.phi.row(1).iter().all(|&v| v == 0.0));
    assert!(got.bias[1].is_finite());
}
