use super::*;
use legume_numeric::candle::candle_core::Device;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A small axis: genes are features `0..3`, peaks `3..7`; genes in modules
/// `0, 0, 1`, peaks in modules `2, 2, 3, 3`. Gene 1 has no pairs.
fn small_spec() -> (CisGatesResolved, Vec<u32>) {
    let module_of = vec![0u32, 0, 1, 2, 2, 3, 3];
    let spec = CisGatesResolved {
        gene_feat: vec![0, 0, 0, 2, 2],
        peak_module: vec![2, 3, 2, 3, 2],
        abc: vec![0.5, 0.3, 0.2, 0.6, 0.4],
        z_log_contact: vec![0.3, -1.0, 0.8, 0.1, -0.4],
        source_idx: vec![0, 1, 2, 3, 4],
    };
    (spec, module_of)
}

fn rand_mat(rng: &mut StdRng, n: usize, h: usize) -> Vec<f32> {
    (0..n * h).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

struct Tables {
    mu: Vec<f32>,
    b_m: Vec<f32>,
    r: Vec<f32>,
    h: usize,
}

fn tables(seed: u64, n_m: usize, n_f: usize, h: usize) -> Tables {
    let mut rng = StdRng::seed_from_u64(seed);
    Tables {
        mu: rand_mat(&mut rng, n_m, h),
        b_m: rand_mat(&mut rng, n_m, 1),
        r: rand_mat(&mut rng, n_f, h),
        h,
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The pool's inputs on device: `μ`, `b_m`, and the cis genes' residual rows.
fn on_device(cis: &CisGateParams, t: &Tables, dev: &Device) -> (Tensor, Tensor, Tensor) {
    let n_m = t.b_m.len();
    let mu = Tensor::from_vec(t.mu.clone(), (n_m, t.h), dev).unwrap();
    let b_m = Tensor::from_vec(t.b_m.clone(), n_m, dev).unwrap();
    let r = Tensor::from_vec(t.r.clone(), (t.r.len() / t.h, t.h), dev).unwrap();
    let r_c = gather_rows(&r, cis.compact_feat()).unwrap();
    (mu, b_m, r_c)
}

/// `w_gp`, by hand, at the current scalars.
fn hand_w(cis: &CisGateParams, spec: &CisGatesResolved, module_of: &[u32], t: &Tables) -> Vec<f32> {
    let th = |v: &Var| v.as_tensor().to_scalar::<f32>().unwrap();
    let (t0, t1, t3) = (th(&cis.theta0), th(&cis.theta1), th(&cis.theta3));
    let h = t.h;
    (0..spec.n_pairs())
        .map(|k| {
            let g = spec.gene_feat[k] as usize;
            let m = spec.peak_module[k] as usize;
            let mg = module_of[g] as usize;
            let rho: Vec<f32> = (0..h).map(|d| t.mu[mg * h + d] + t.r[g * h + d]).collect();
            let agree = dot(&rho, &t.mu[m * h..(m + 1) * h]);
            let s = t0 + t1 * spec.z_log_contact[k] + t3 * agree;
            spec.abc[k] * s.max(0.0)
        })
        .collect()
}

/// `w`, `ν_g = Σ_p w_gp μ_{m(p)}` and `c_g = Σ_p w_gp b_{m(p)}` agree with a
/// loop over the pairs; a gene without pairs reads the zero row.
#[test]
fn the_pool_is_the_pairwise_sum() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, module_of.len(), &dev).unwrap();
    let t = tables(1, 4, module_of.len(), 3);
    let (mu, b_m, r_c) = on_device(&cis, &t, &dev);
    let pool = cis.pool(&mu, &b_m, &r_c).unwrap();

    let w = hand_w(&cis, &spec, &module_of, &t);
    let got_w = pool.w.to_vec1::<f32>().unwrap();
    for (a, b) in got_w.iter().zip(&w) {
        assert!((a - b).abs() < 1e-5, "w {got_w:?} vs {w:?}");
    }
    let nu = pool.nu.to_vec2::<f32>().unwrap();
    let c = pool.c.to_vec1::<f32>().unwrap();
    assert_eq!(nu.len(), cis.n_compact + 1);
    let h = t.h;
    for (g, row) in [0usize, 2].iter().zip(0..) {
        let mut want_nu = vec![0f32; h];
        let mut want_c = 0f32;
        for ((&gf, &m), &wk) in spec.gene_feat.iter().zip(&spec.peak_module).zip(&w) {
            if gf as usize == *g {
                let m = m as usize;
                for (acc, &mu) in want_nu.iter_mut().zip(&t.mu[m * h..(m + 1) * h]) {
                    *acc += wk * mu;
                }
                want_c += wk * t.b_m[m];
            }
        }
        for (got, want) in nu[row].iter().zip(&want_nu) {
            assert!((got - want).abs() < 1e-5, "ν row {row}");
        }
        assert!((c[row] - want_c).abs() < 1e-5, "c row {row}");
    }
    assert!(nu[cis.n_compact].iter().all(|&x| x == 0.0));
    assert_eq!(c[cis.n_compact], 0.0);
}

/// `η_ug = γ₂(⟨e_u, r_g⟩ + b_g) + γ₁ Σ_p w_gp (⟨e_u, μ_{m(p)}⟩ + b_{m(p)})`
/// through the mixed gene rows, for a cis gene and a gene without pairs.
#[test]
fn mixed_gene_rows_score_the_model() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, module_of.len(), &dev).unwrap();
    cis.raw_gamma1
        .set(&Tensor::new(0.3f32, &dev).unwrap())
        .unwrap();
    let t = tables(2, 4, module_of.len(), 3);
    let (mu, b_m, r_c) = on_device(&cis, &t, &dev);
    let mix = cis.mix(&mu, &b_m, &r_c).unwrap();
    let (g1, g2) = (
        mix.gamma1.to_scalar::<f32>().unwrap(),
        mix.gamma2.to_scalar::<f32>().unwrap(),
    );

    let h = t.h;
    let e = [0.4f32, -0.7, 1.1];
    let b_g = [0.2f32, -0.1, 0.5];
    let genes = [0u32, 1, 2];
    let g_ids = Tensor::from_vec(genes.to_vec(), 3, &dev).unwrap();
    let r_all = Tensor::from_vec(t.r.clone(), (module_of.len(), h), &dev).unwrap();
    let r = gather_rows(&r_all, &g_ids).unwrap();
    let bias = Tensor::from_vec(b_g.to_vec(), 3, &dev).unwrap();
    let (r_mix, b_mix) = mix.gene_rows(&g_ids, &r, &bias).unwrap();
    let eta = Tensor::from_vec(e.to_vec(), (1, h), &dev)
        .unwrap()
        .matmul(&r_mix.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap()
        .add(&b_mix)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let w = hand_w(&cis, &spec, &module_of, &t);
    for (j, &g) in genes.iter().enumerate() {
        let g = g as usize;
        let rho = dot(&e, &t.r[g * h..(g + 1) * h]) + b_g[j];
        let a: f32 = (0..spec.n_pairs())
            .filter(|&k| spec.gene_feat[k] as usize == g)
            .map(|k| {
                let m = spec.peak_module[k] as usize;
                w[k] * (dot(&e, &t.mu[m * h..(m + 1) * h]) + t.b_m[m])
            })
            .sum();
        let want = g2 * rho + g1 * a;
        assert!(
            (eta[j] - want).abs() < 1e-4,
            "gene {g}: {} vs {want}",
            eta[j]
        );
    }
}

/// At the default start the links carry a small share (`γ₁ ≈ 0.01`, `γ₂ ≈ 1`)
/// and the gate scalars take a gradient that is not vanishing — and that
/// gradient is the finite difference of the loss.
#[test]
fn the_gates_take_a_live_gradient_at_the_default_start() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, module_of.len(), &dev).unwrap();
    let (g1, g2) = cis.gammas().unwrap();
    let g1 = g1.to_scalar::<f32>().unwrap();
    let g2 = g2.to_scalar::<f32>().unwrap();
    assert!((g1 - 0.01).abs() < 1e-4, "γ₁ = {g1}");
    assert!((g2 - 1.0).abs() < 1e-4, "γ₂ = {g2}");

    let t = tables(3, 4, module_of.len(), 3);
    let (mu, b_m, r_c) = on_device(&cis, &t, &dev);
    let e = Tensor::from_vec(vec![0.4f32, -0.7, 1.1, 0.9, 0.2, -0.3], (2, 3), &dev).unwrap();
    let g_ids = Tensor::from_vec(vec![0u32, 2], 2, &dev).unwrap();
    let r_all = Tensor::from_vec(t.r.clone(), (module_of.len(), 3), &dev).unwrap();
    let loss = |cis: &CisGateParams| -> Tensor {
        let mix = cis.mix(&mu, &b_m, &r_c).unwrap();
        let r = gather_rows(&r_all, &g_ids).unwrap();
        let bias = Tensor::zeros(2, DType::F32, &dev).unwrap();
        let (rm, bm) = mix.gene_rows(&g_ids, &r, &bias).unwrap();
        e.matmul(&rm.t().unwrap())
            .unwrap()
            .broadcast_add(&bm)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
    };
    let grads = loss(&cis).backward().unwrap();
    for (name, v) in [
        ("θ₃", &cis.theta3),
        ("θ₁", &cis.theta1),
        ("raw γ₁", &cis.raw_gamma1),
    ] {
        let g = grads.get(v).unwrap().to_scalar::<f32>().unwrap();
        assert!(g.abs() > 1e-4, "{name}: gradient {g} vanishes at the start");
        let x = v.as_tensor().to_scalar::<f32>().unwrap();
        let eps = 1e-2f32;
        v.set(&Tensor::new(x + eps, &dev).unwrap()).unwrap();
        let up = loss(&cis).to_scalar::<f32>().unwrap();
        v.set(&Tensor::new(x - eps, &dev).unwrap()).unwrap();
        let down = loss(&cis).to_scalar::<f32>().unwrap();
        v.set(&Tensor::new(x, &dev).unwrap()).unwrap();
        let fd = (up - down) / (2.0 * eps);
        assert!(
            (g - fd).abs() < 1e-2 * fd.abs().max(1.0),
            "{name}: autograd {g} vs finite difference {fd}"
        );
    }
}

/// Differentiating through detached leaves and then back through the pool
/// gives the gradients of differentiating straight through: the CPU step
/// builds the pool once and its slices share the leaves.
#[test]
fn backprop_through_detached_leaves_is_the_direct_gradient() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, module_of.len(), &dev).unwrap();
    let t = tables(4, 4, module_of.len(), 3);
    let mu = Var::from_vec(t.mu.clone(), (4, 3), &dev).unwrap();
    let b_m = Var::from_vec(t.b_m.clone(), 4, &dev).unwrap();
    let r_all = Var::from_vec(t.r.clone(), (module_of.len(), 3), &dev).unwrap();
    let e = Tensor::from_vec(vec![0.4f32, -0.7, 1.1, 0.9, 0.2, -0.3], (2, 3), &dev).unwrap();
    let g_ids = Tensor::from_vec(vec![0u32, 2], 2, &dev).unwrap();
    let graph = || {
        let r_c = gather_rows(r_all.as_tensor(), cis.compact_feat()).unwrap();
        cis.mix(mu.as_tensor(), b_m.as_tensor(), &r_c).unwrap()
    };
    let loss = |mix: &CisMix| -> Tensor {
        let r = gather_rows(r_all.as_tensor(), &g_ids).unwrap();
        let bias = Tensor::zeros(2, DType::F32, &dev).unwrap();
        let (rm, bm) = mix.gene_rows(&g_ids, &r, &bias).unwrap();
        e.matmul(&rm.t().unwrap())
            .unwrap()
            .broadcast_add(&bm)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
    };
    let direct = loss(&graph()).backward().unwrap();

    let g = graph();
    let leaves = g.detached().unwrap();
    let mut via = loss(&leaves.mix).backward().unwrap();
    let through = leaves
        .backprop(&g, &via)
        .unwrap()
        .expect("the leaves took a gradient");
    for &id in through.get_ids() {
        let x = through.get_id(id).unwrap();
        let sum = match via.get_id(id) {
            Some(a) => (a + x).unwrap(),
            None => x.clone(),
        };
        via.insert_id(id, sum);
    }
    let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (name, v) in [
        ("θ₀", cis.theta0.as_tensor()),
        ("θ₃", cis.theta3.as_tensor()),
        ("raw γ₁", cis.raw_gamma1.as_tensor()),
        ("raw γ₂", cis.raw_gamma2.as_tensor()),
        ("μ", mu.as_tensor()),
        ("b_m", b_m.as_tensor()),
        ("r", r_all.as_tensor()),
    ] {
        let a = flat(direct.get(v).unwrap());
        let b = flat(via.get(v).unwrap());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "{name}: {a:?} vs {b:?}");
        }
    }
}

/// Pairs of a module-only gene are dropped: that gene has no gene level for
/// the links to feed.
#[test]
fn pairs_of_module_only_genes_are_dropped() {
    let (spec, _) = small_spec();
    let mut is_module_only = vec![false; 7];
    is_module_only[2] = true;
    let kept = spec.without_genes(&is_module_only);
    assert_eq!(kept.gene_feat, vec![0, 0, 0]);
    assert_eq!(kept.source_idx, vec![0, 1, 2]);
    assert_eq!(kept.abc, vec![0.5, 0.3, 0.2]);
}

/// Pairs whose peak sits in a background module (near-empty or scattered
/// peaks) are dropped: such a peak carries no link signal.
#[test]
fn pairs_to_background_peak_modules_are_dropped() {
    let (spec, _) = small_spec();
    // Modules 0..4; module 3 is background.
    let kept = spec.without_peak_modules(&[false, false, false, true]);
    assert_eq!(kept.peak_module, vec![2, 2, 2]);
    assert_eq!(kept.source_idx, vec![0, 2, 4]);
}
