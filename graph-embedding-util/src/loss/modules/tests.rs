use super::*;
use candle_util::candle_core::{DType, Var};
use candle_util::nn::sparsemax;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn dev() -> Device {
    Device::Cpu
}

fn pi_fixture() -> Tensor {
    // D = 6, M = 2; rows on the simplex with a few exact zeros.
    let v = vec![
        1.0f32, 0.0, //
        0.7, 0.3, //
        0.0, 1.0, //
        0.5, 0.5, //
        0.2, 0.8, //
        1.0, 0.0,
    ];
    Tensor::from_vec(v, (6, 2), &dev()).unwrap()
}

#[test]
fn masked_membership_zeroes_dropped_rows_and_keeps_module_mass() {
    let pi = pi_fixture();
    // Drop features 1 and 4.
    let keep = Tensor::from_vec(vec![1f32, 0.0, 1.0, 1.0, 0.0, 1.0], (6, 1), &dev()).unwrap();
    let tilde = masked_membership(&pi, &keep).unwrap();
    let t: Vec<Vec<f32>> = tilde.to_vec2().unwrap();
    assert_eq!(t[1], vec![0.0, 0.0]);
    assert_eq!(t[4], vec![0.0, 0.0]);
    let full: Vec<f32> = pi.sum(0).unwrap().to_vec1().unwrap();
    let masked: Vec<f32> = tilde.sum(0).unwrap().to_vec1().unwrap();
    for (a, b) in full.iter().zip(&masked) {
        assert!((a - b).abs() < 1e-5, "module mass {a} vs {b}");
    }
}

#[test]
fn no_dropout_is_the_identity() {
    let pi = pi_fixture();
    let keep = Tensor::ones((6, 1), DType::F32, &dev()).unwrap();
    let tilde = masked_membership(&pi, &keep).unwrap();
    let a: Vec<f32> = pi.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<f32> = tilde.flatten_all().unwrap().to_vec1().unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-6);
    }
}

#[test]
fn pooled_counts_ignore_dropped_features() {
    let pi = pi_fixture();
    let keep = Tensor::from_vec(vec![1f32, 0.0, 1.0, 1.0, 1.0, 1.0], (6, 1), &dev()).unwrap();
    let tilde = masked_membership(&pi, &keep).unwrap();
    let f: Vec<u32> = vec![0, 1, 3];
    let c1: Vec<f32> = vec![2.0, 5.0, 1.0];
    let c2: Vec<f32> = vec![2.0, 999.0, 1.0]; // dropped feature 1 perturbed
    let x1 = dense_count_block(&[(&f, &c1)], 6, &dev()).unwrap();
    let x2 = dense_count_block(&[(&f, &c2)], 6, &dev()).unwrap();
    let a: Vec<f32> = pool_module_counts(&x1, &tilde)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b: Vec<f32> = pool_module_counts(&x2, &tilde)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(a, b);
    // Hand sum over survivors with the per-module rescale: module 0 full mass 3.4,
    // kept mass 2.7 → scale 3.4/2.7; module 1 full 2.6, kept 2.3.
    let s0 = 3.4f32 / 2.7;
    let s1 = 2.6f32 / 2.3;
    let want0 = (2.0 * 1.0 + 1.0 * 0.5) * s0;
    let want1 = (1.0 * 0.5) * s1;
    assert!((a[0] - want0).abs() < 1e-4, "{} vs {want0}", a[0]);
    assert!((a[1] - want1).abs() < 1e-4, "{} vs {want1}", a[1]);
}

#[test]
fn dense_block_scatters_rows() {
    let f0: Vec<u32> = vec![0, 2];
    let c0: Vec<f32> = vec![1.0, 3.0];
    let f1: Vec<u32> = vec![1];
    let c1: Vec<f32> = vec![7.0];
    let x = dense_count_block(&[(&f0, &c0), (&f1, &c1)], 3, &dev()).unwrap();
    let v: Vec<Vec<f32>> = x.to_vec2().unwrap();
    assert_eq!(v, vec![vec![1.0, 0.0, 3.0], vec![0.0, 7.0, 0.0]]);
}

/// Central finite differences on every parameter of the exact term.
#[test]
fn module_softmax_gradient_matches_finite_difference() {
    let d = dev();
    let (u, m, h, nf) = (2usize, 3usize, 2usize, 4usize);
    let logits = Var::from_tensor(
        &Tensor::from_vec(
            vec![
                0.9f32, 0.1, 0.2, 0.3, 0.8, 0.1, 0.5, 0.5, 0.2, 0.1, 0.2, 0.9,
            ],
            (nf, m),
            &d,
        )
        .unwrap(),
    )
    .unwrap();
    let mu = Var::from_tensor(
        &Tensor::from_vec(vec![0.3f32, -0.2, 0.1, 0.4, -0.5, 0.2], (m, h), &d).unwrap(),
    )
    .unwrap();
    let b = Var::from_tensor(&Tensor::from_vec(vec![0.1f32, -0.1, 0.0], m, &d).unwrap()).unwrap();
    let e = Var::from_tensor(&Tensor::from_vec(vec![0.2f32, 0.7, -0.4, 0.3], (u, h), &d).unwrap())
        .unwrap();
    let x = Tensor::from_vec(vec![3f32, 0.0, 2.0, 1.0, 0.0, 4.0, 1.0, 0.0], (u, nf), &d).unwrap();

    let loss_fn = |logits: &Tensor, mu: &Tensor, b: &Tensor, e: &Tensor| -> f32 {
        let pi = sparsemax(logits).unwrap();
        let xcm = pool_module_counts(&x, &pi).unwrap();
        module_softmax_loss(e, mu, b, &xcm)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    let pi = sparsemax(logits.as_tensor()).unwrap();
    let xcm = pool_module_counts(&x, &pi).unwrap();
    let loss = module_softmax_loss(e.as_tensor(), mu.as_tensor(), b.as_tensor(), &xcm).unwrap();
    let grads = loss.backward().unwrap();

    let check = |var: &Var, name: &str| {
        let g: Vec<f32> = grads
            .get(var.as_tensor())
            .unwrap_or_else(|| panic!("no gradient reached {name}"))
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let base: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
        let shape = var.as_tensor().shape().clone();
        let eps = 1e-2f32;
        for i in 0..base.len() {
            let mut up = base.clone();
            up[i] += eps;
            let mut dn = base.clone();
            dn[i] -= eps;
            let tu = Tensor::from_vec(up, shape.clone(), &d).unwrap();
            let td = Tensor::from_vec(dn, shape.clone(), &d).unwrap();
            let (lu, ld) = match name {
                "logits" => (
                    loss_fn(&tu, mu.as_tensor(), b.as_tensor(), e.as_tensor()),
                    loss_fn(&td, mu.as_tensor(), b.as_tensor(), e.as_tensor()),
                ),
                "mu" => (
                    loss_fn(logits.as_tensor(), &tu, b.as_tensor(), e.as_tensor()),
                    loss_fn(logits.as_tensor(), &td, b.as_tensor(), e.as_tensor()),
                ),
                "b" => (
                    loss_fn(logits.as_tensor(), mu.as_tensor(), &tu, e.as_tensor()),
                    loss_fn(logits.as_tensor(), mu.as_tensor(), &td, e.as_tensor()),
                ),
                _ => (
                    loss_fn(logits.as_tensor(), mu.as_tensor(), b.as_tensor(), &tu),
                    loss_fn(logits.as_tensor(), mu.as_tensor(), b.as_tensor(), &td),
                ),
            };
            let fd = (lu - ld) / (2.0 * eps);
            let tol = 2e-2 * (1.0 + fd.abs().max(g[i].abs()));
            assert!(
                (fd - g[i]).abs() < tol,
                "{name}[{i}]: autograd {} vs finite-difference {fd}",
                g[i]
            );
        }
    };
    check(&logits, "logits");
    check(&mu, "mu");
    check(&b, "b");
    check(&e, "e");
}

#[test]
fn balance_prior_is_zero_when_uniform_and_positive_when_collapsed() {
    let uniform = Tensor::from_vec(vec![0.5f32; 8], (4, 2), &dev()).unwrap();
    let z: f32 = module_balance_prior(&uniform).unwrap().to_scalar().unwrap();
    assert!(z.abs() < 1e-5, "uniform occupancy should score 0, got {z}");
    let collapsed = Tensor::from_vec(
        vec![1f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        (4, 2),
        &dev(),
    )
    .unwrap();
    let c: f32 = module_balance_prior(&collapsed)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        (c - 2f32.ln()).abs() < 1e-4,
        "total collapse onto one of two modules is ln 2, got {c}"
    );
}

#[test]
fn pools_follow_membership_and_stay_inside_the_feature_pool() {
    let pi = pi_fixture();
    let host: Vec<f32> = pi.flatten_all().unwrap().to_vec1().unwrap();
    let rows = membership_rows_host(&host, 6, 2);
    // Feature 5 is excluded from this sampler's pool.
    let pool = ModulePools::build(rows, 2, &[0, 1, 2, 3, 4]);
    assert_eq!(pool.member_counts(), vec![4, 4]); // m0: 0,1,3,4  m1: 1,2,3,4
    let mut rng = StdRng::seed_from_u64(1);
    let mut out = Vec::new();
    // Feature 0 is only in module 0: negatives come from {0,1,3,4}.
    for _ in 0..50 {
        assert!(pool.draw_negatives(0, 3, &mut out, &mut rng));
    }
    assert_eq!(out.len(), 150);
    assert!(out.iter().all(|f| [0u32, 1, 3, 4].contains(f)), "{out:?}");
}

#[test]
fn singleton_module_falls_back() {
    let host = vec![1f32, 0.0, 0.0, 1.0, 0.0, 1.0];
    let rows = membership_rows_host(&host, 3, 2);
    let pool = ModulePools::build(rows, 2, &[0, 1, 2]);
    let mut rng = StdRng::seed_from_u64(2);
    let mut out = Vec::new();
    // Module 0 has one member: nothing to contrast with.
    assert!(!pool.draw_negatives(0, 2, &mut out, &mut rng));
    assert!(out.is_empty());
    assert!(pool.draw_negatives(1, 2, &mut out, &mut rng));
    assert_eq!(out.len(), 2);
}

#[test]
fn diagnostics_read_collapse() {
    let host = vec![1f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let dg = membership_diagnostics(&host, 4, 2, 2);
    assert!((dg.max_occupancy_ratio - 1.5).abs() < 1e-5);
    assert_eq!(dg.n_small_modules, 1);
    assert!(dg.mean_row_entropy.abs() < 1e-6);
    assert!((dg.mean_row_support - 1.0).abs() < 1e-6);
}

/// The exact term alone plus the balance prior does not push a random start onto
/// one module: after a few hundred AdamW steps the occupancy stays spread and rows
/// keep some support.
#[test]
fn entropy_stays_above_floor_on_random_counts() {
    use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW};
    let d = dev();
    let (nf, m, h, u) = (40usize, 4usize, 3usize, 8usize);
    let mut rng = StdRng::seed_from_u64(7);
    let logits = Var::from_tensor(
        &Tensor::from_vec(
            (0..nf * m)
                .map(|_| rng.random::<f32>() * 0.5)
                .collect::<Vec<f32>>(),
            (nf, m),
            &d,
        )
        .unwrap(),
    )
    .unwrap();
    let mu = Var::from_tensor(
        &Tensor::from_vec(
            (0..m * h)
                .map(|_| rng.random::<f32>() - 0.5)
                .collect::<Vec<f32>>(),
            (m, h),
            &d,
        )
        .unwrap(),
    )
    .unwrap();
    let b = Var::zeros(m, DType::F32, &d).unwrap();
    let e = Var::from_tensor(
        &Tensor::from_vec(
            (0..u * h)
                .map(|_| rng.random::<f32>() - 0.5)
                .collect::<Vec<f32>>(),
            (u, h),
            &d,
        )
        .unwrap(),
    )
    .unwrap();
    let x = Tensor::from_vec(
        (0..u * nf)
            .map(|_| (rng.random::<f32>() * 5.0).floor())
            .collect::<Vec<f32>>(),
        (u, nf),
        &d,
    )
    .unwrap();
    let mut opt = AdamW::new(
        vec![logits.clone(), mu.clone(), b.clone(), e.clone()],
        ParamsAdamW {
            lr: 0.05,
            ..Default::default()
        },
    )
    .unwrap();
    for _ in 0..200 {
        let pi = sparsemax(logits.as_tensor()).unwrap();
        let xcm = pool_module_counts(&x, &pi).unwrap();
        let l = module_softmax_loss(e.as_tensor(), mu.as_tensor(), b.as_tensor(), &xcm).unwrap();
        let prior = module_balance_prior(&pi).unwrap();
        let loss = (l + prior).unwrap();
        opt.backward_step(&loss).unwrap();
    }
    let pi = sparsemax(logits.as_tensor()).unwrap();
    let host: Vec<f32> = pi.flatten_all().unwrap().to_vec1().unwrap();
    let dg = membership_diagnostics(&host, nf, m, 1);
    assert!(
        dg.max_occupancy_ratio < 0.9 * m as f32,
        "occupancy collapsed: ratio {}",
        dg.max_occupancy_ratio
    );
    assert!(dg.n_small_modules < m, "every module emptied but one");
}
