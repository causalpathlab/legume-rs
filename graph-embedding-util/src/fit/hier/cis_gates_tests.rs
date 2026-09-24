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

/// The cis genes of [`small_spec`], in compact order.
const CIS_GENES: [usize; 2] = [0, 2];

fn rand_mat(rng: &mut StdRng, n: usize, h: usize) -> Vec<f32> {
    (0..n * h).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

struct Tables {
    mu: Vec<f32>,
    r: Vec<f32>,
    /// Unit embeddings `[n_u × h]`.
    e: Vec<f32>,
    /// Module biases `[n_m]`.
    b_m: Vec<f32>,
    n_u: usize,
    h: usize,
}

fn tables(seed: u64, n_m: usize, n_f: usize, h: usize) -> Tables {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_u = 6;
    Tables {
        mu: rand_mat(&mut rng, n_m, h),
        r: rand_mat(&mut rng, n_f, h),
        e: rand_mat(&mut rng, n_u, h),
        b_m: rand_mat(&mut rng, n_m, 1),
        n_u,
        h,
    }
}

/// The module biases on device.
fn b_m_on_device(t: &Tables, dev: &Device) -> Tensor {
    Tensor::from_vec(t.b_m.clone(), t.b_m.len(), dev).unwrap()
}

/// The pool over a table set's module rows and biases.
fn pool_of(cis: &CisGateParams, t: &Tables, dev: &Device) -> CisPool {
    let n_m = t.mu.len() / t.h;
    let mu = Tensor::from_vec(t.mu.clone(), (n_m, t.h), dev).unwrap();
    cis.pool(&mu, &b_m_on_device(t, dev)).unwrap()
}

/// The share of each cis gene's score its cis peaks take in the mixture tests.
const ALPHA: f32 = 0.3;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// On device: the module table `μ`, the cis genes' residual rows `r_c` and
/// the unit embeddings.
fn on_device(cis: &CisGateParams, t: &Tables, dev: &Device) -> (Tensor, Tensor, Tensor) {
    let n_m = t.mu.len() / t.h;
    let mu = Tensor::from_vec(t.mu.clone(), (n_m, t.h), dev).unwrap();
    let r = Tensor::from_vec(t.r.clone(), (t.r.len() / t.h, t.h), dev).unwrap();
    let r_c = gather_rows(&r, cis.compact_feat()).unwrap();
    let e = Tensor::from_vec(t.e.clone(), (t.n_u, t.h), dev).unwrap();
    (mu, r_c, e)
}

/// The gene's share of each pair, by hand at the current scalars:
/// `w_gp = abc · max(0, θ₀ + θ₁ z)` normalised over the gene's pairs.
fn hand_w(cis: &CisGateParams, spec: &CisGatesResolved) -> Vec<f32> {
    let th = |v: &Var| v.as_tensor().to_scalar::<f32>().unwrap();
    let (t0, t1) = (th(&cis.theta0), th(&cis.theta1));
    let raw: Vec<f32> = (0..spec.n_pairs())
        .map(|k| spec.abc[k] * (t0 + t1 * spec.z_log_contact[k]).max(0.0))
        .collect();
    (0..spec.n_pairs())
        .map(|k| {
            let tot: f32 = (0..spec.n_pairs())
                .filter(|&q| spec.gene_feat[q] == spec.gene_feat[k])
                .map(|q| raw[q])
                .sum();
            if tot > 0.0 {
                raw[k] / tot
            } else {
                0.0
            }
        })
        .collect()
}

/// `ν̃_g = Σ_p w̃_gp μ_{m(p)}` by hand.
fn hand_nu(spec: &CisGatesResolved, w: &[f32], t: &Tables, g: usize) -> Vec<f32> {
    let h = t.h;
    let mut nu = vec![0f32; h];
    for k in (0..spec.n_pairs()).filter(|&k| spec.gene_feat[k] as usize == g) {
        let m = spec.peak_module[k] as usize;
        for (acc, &mu) in nu.iter_mut().zip(&t.mu[m * h..(m + 1) * h]) {
            *acc += w[k] * mu;
        }
    }
    nu
}

/// The gap by hand: per cis gene, the mean over units of the centred
/// `⟨e_u, ρ_g − ν̃_g⟩²`, `ρ_g = μ_{m(g)} + r_g`; averaged over the genes.
fn hand_gap(cis: &CisGateParams, spec: &CisGatesResolved, module_of: &[u32], t: &Tables) -> f32 {
    let h = t.h;
    let w = hand_w(cis, spec);
    let per_gene: Vec<f32> = CIS_GENES
        .iter()
        .map(|&g| {
            let nu = hand_nu(spec, &w, t, g);
            let mg = module_of[g] as usize;
            let d: Vec<f32> = (0..h)
                .map(|k| t.mu[mg * h + k] + t.r[g * h + k] - nu[k])
                .collect();
            let s: Vec<f32> = (0..t.n_u)
                .map(|u| dot(&t.e[u * h..(u + 1) * h], &d))
                .collect();
            let mean = s.iter().sum::<f32>() / t.n_u as f32;
            s.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / t.n_u as f32
        })
        .collect();
    per_gene.iter().sum::<f32>() / per_gene.len() as f32
}

/// `w̃`, `ν̃_g = Σ_p w̃_gp μ_{m(p)}` and `c̃_g = Σ_p w̃_gp b_{m(p)}` agree with a
/// loop over the pairs.
#[test]
fn the_pool_is_the_pairwise_sum() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let t = tables(1, 4, module_of.len(), 3);
    let pool = pool_of(&cis, &t, &dev);

    let w = hand_w(&cis, &spec);
    let got_w = pool.w.to_vec1::<f32>().unwrap();
    for (a, b) in got_w.iter().zip(&w) {
        assert!((a - b).abs() < 1e-5, "w {got_w:?} vs {w:?}");
    }
    let nu = pool.nu.to_vec2::<f32>().unwrap();
    let c = pool.c.to_vec1::<f32>().unwrap();
    assert_eq!(nu.len(), cis.n_compact);
    for (row, &g) in CIS_GENES.iter().enumerate() {
        for (got, want) in nu[row].iter().zip(&hand_nu(&spec, &w, &t, g)) {
            assert!((got - want).abs() < 1e-5, "ν̃ row {row}");
        }
        let want_c: f32 = (0..spec.n_pairs())
            .filter(|&k| spec.gene_feat[k] as usize == g)
            .map(|k| w[k] * t.b_m[spec.peak_module[k] as usize])
            .sum();
        assert!((c[row] - want_c).abs() < 1e-5, "c̃ row {row}");
    }
}

/// The gate is a distance prior: it reads contact only, so new module
/// tables leave every share where it was.
#[test]
fn the_gate_reads_contact_only() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let w_of = |seed: u64| {
        pool_of(&cis, &tables(seed, 4, module_of.len(), 3), &dev)
            .w
            .to_vec1::<f32>()
            .unwrap()
    };
    assert_eq!(w_of(5), w_of(6));
}

/// Only the gate's shape matters: scaling `(θ₀, θ₁)` together leaves the
/// shares where they were.
#[test]
fn the_shares_ignore_the_gate_scale() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let t = tables(11, 4, module_of.len(), 3);
    let before = pool_of(&cis, &t, &dev).w.to_vec1::<f32>().unwrap();
    for v in [&cis.theta0, &cis.theta1] {
        let x = v.as_tensor().to_scalar::<f32>().unwrap();
        v.set(&Tensor::new(3.0 * x, &dev).unwrap()).unwrap();
    }
    let after = pool_of(&cis, &t, &dev).w.to_vec1::<f32>().unwrap();
    for (a, b) in before.iter().zip(&after) {
        assert!((a - b).abs() < 1e-6, "{before:?} vs {after:?}");
    }
}

/// A gene whose gates all close has nothing pooled: its row is zero, not
/// `NaN`.
#[test]
fn a_gene_with_every_gate_closed_pools_nothing() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    cis.theta0.set(&Tensor::new(-10f32, &dev).unwrap()).unwrap();
    cis.theta1.set(&Tensor::new(0f32, &dev).unwrap()).unwrap();
    let pool = pool_of(&cis, &tables(12, 4, module_of.len(), 3), &dev);
    assert!(pool.w.to_vec1::<f32>().unwrap().iter().all(|&w| w == 0.0));
    let nu = pool.nu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(nu.iter().all(|&x| x == 0.0), "{nu:?}");
}

/// The alignment gap: per cis gene, the variance over the units of
/// `⟨e_u, ρ_g − ν̃_g⟩` — how far the gene's own profile sits from its
/// ATAC-guided activity — averaged over the genes. Biases drop out.
#[test]
fn the_gap_is_the_profile_variance_of_gene_minus_pooled() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let t = tables(2, 4, module_of.len(), 3);
    let (mu, r_c, e) = on_device(&cis, &t, &dev);
    let got = cis
        .align_gap(&pool_of(&cis, &t, &dev), &mu, &r_c, &e)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let want = hand_gap(&cis, &spec, &module_of, &t);
    assert!((got - want).abs() < 1e-4, "{got} vs {want}");
}

/// A gene whose own row IS its pooled row has no gap, whatever the units.
#[test]
fn a_gene_equal_to_its_pooled_row_has_no_gap() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let mut t = tables(3, 4, module_of.len(), 3);
    let w = hand_w(&cis, &spec);
    let h = t.h;
    for &g in &CIS_GENES {
        let nu = hand_nu(&spec, &w, &t, g);
        let mg = module_of[g] as usize;
        let mu_g = t.mu[mg * h..(mg + 1) * h].to_vec();
        for ((r, &n), &m) in t.r[g * h..(g + 1) * h].iter_mut().zip(&nu).zip(&mu_g) {
            *r = n - m;
        }
    }
    let (mu, r_c, e) = on_device(&cis, &t, &dev);
    let gap = cis
        .align_gap(&pool_of(&cis, &t, &dev), &mu, &r_c, &e)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(gap.abs() < 1e-6, "{gap}");
}

/// The gap trains the gates, `μ` and `r` — the finite difference of the gap —
/// and never the units: flattening `e_u` must not be a way to close it.
#[test]
fn the_gap_moves_features_and_gates_not_units() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let t = tables(4, 4, module_of.len(), 3);
    let (mu0, r_c0, e0) = on_device(&cis, &t, &dev);
    let mu = Var::from_tensor(&mu0).unwrap();
    let r_c = Var::from_tensor(&r_c0).unwrap();
    let e = Var::from_tensor(&e0).unwrap();
    let b_m = b_m_on_device(&t, &dev);
    let gap = |cis: &CisGateParams| {
        let pool = cis.pool(mu.as_tensor(), &b_m).unwrap();
        cis.align_gap(&pool, mu.as_tensor(), r_c.as_tensor(), e.as_tensor())
            .unwrap()
    };
    let grads = gap(&cis).backward().unwrap();
    assert!(grads.get(&e).is_none(), "the gap reached the units");
    for v in [&mu, &r_c] {
        let g = grads.get(v).expect("the gap reaches the feature rows");
        let n = g
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(n > 1e-8, "vanishing feature gradient");
    }
    for (name, v) in [("θ₀", &cis.theta0), ("θ₁", &cis.theta1)] {
        let g = grads.get(v).unwrap().to_scalar::<f32>().unwrap();
        let x = v.as_tensor().to_scalar::<f32>().unwrap();
        let eps = 1e-2f32;
        v.set(&Tensor::new(x + eps, &dev).unwrap()).unwrap();
        let up = gap(&cis).to_scalar::<f32>().unwrap();
        v.set(&Tensor::new(x - eps, &dev).unwrap()).unwrap();
        let down = gap(&cis).to_scalar::<f32>().unwrap();
        v.set(&Tensor::new(x, &dev).unwrap()).unwrap();
        let fd = (up - down) / (2.0 * eps);
        assert!(g.abs() > 1e-5, "{name}: gradient {g} vanishes");
        assert!(
            (g - fd).abs() < 1e-2 * fd.abs().max(1e-2),
            "{name}: autograd {g} vs finite difference {fd}"
        );
    }
}

/// Without a mixture there is nothing to fold into the gene level.
#[test]
fn without_a_mixture_the_gene_level_is_left_alone() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let t = tables(20, 4, module_of.len(), 3);
    assert!(cis.mix(&pool_of(&cis, &t, &dev)).unwrap().is_none());
}

/// The mixture through the mixed gene rows: a cis gene scores
/// `η_ug = (1 − α)(⟨e_u, r_g⟩ + b_g) + α Σ_p w̃_gp (⟨e_u, μ_{m(p)}⟩ + b_{m(p)})`,
/// and a gene without pairs keeps its own score.
#[test]
fn mixed_gene_rows_score_the_mixture() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let n_f = module_of.len();
    let cis = CisGateParams::new(&spec, &module_of, &dev)
        .unwrap()
        .with_mix(ALPHA, n_f, &dev)
        .unwrap();
    let t = tables(21, 4, n_f, 3);
    let mix = cis.mix(&pool_of(&cis, &t, &dev)).unwrap().unwrap();

    let h = t.h;
    let e = [0.4f32, -0.7, 1.1];
    let b_g = [0.2f32, -0.1, 0.5];
    let genes = [0u32, 1, 2];
    let g_ids = Tensor::from_vec(genes.to_vec(), 3, &dev).unwrap();
    let r_all = Tensor::from_vec(t.r.clone(), (n_f, h), &dev).unwrap();
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

    let w = hand_w(&cis, &spec);
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
        let want = if CIS_GENES.contains(&g) {
            (1.0 - ALPHA) * rho + ALPHA * a
        } else {
            rho
        };
        assert!(
            (eta[j] - want).abs() < 1e-4,
            "gene {g}: {} vs {want}",
            eta[j]
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
    let n_f = module_of.len();
    let cis = CisGateParams::new(&spec, &module_of, &dev)
        .unwrap()
        .with_mix(ALPHA, n_f, &dev)
        .unwrap();
    let t = tables(22, 4, n_f, 3);
    let mu = Var::from_vec(t.mu.clone(), (4, 3), &dev).unwrap();
    let b_m = Var::from_vec(t.b_m.clone(), 4, &dev).unwrap();
    let r_all = Var::from_vec(t.r.clone(), (n_f, 3), &dev).unwrap();
    let e = Tensor::from_vec(vec![0.4f32, -0.7, 1.1, 0.9, 0.2, -0.3], (2, 3), &dev).unwrap();
    let g_ids = Tensor::from_vec(vec![0u32, 2], 2, &dev).unwrap();
    let graph = || {
        let pool = cis.pool(mu.as_tensor(), b_m.as_tensor()).unwrap();
        cis.mix(&pool).unwrap().unwrap()
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
        ("θ₁", cis.theta1.as_tensor()),
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

/// Phase 2 sees the row phase 1 scored: under the mixture a cis gene's
/// dictionary row `μ_{m(g)} + r_g` becomes
/// `μ_{m(g)} + (1 − α) r_g + α Σ_p w̃_gp μ_{m(p)}`, its bias alike; a gene
/// without pairs keeps its row. Without a mixture there is no blend.
#[test]
fn the_dictionary_carries_the_mixed_gene_rows() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let n_f = module_of.len();
    let t = tables(23, 4, n_f, 3);
    let h = t.h;
    let b_g: Vec<f32> = (0..n_f).map(|f| 0.1 * f as f32 - 0.2).collect();
    let plain = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let (_, r_c, _) = on_device(&plain, &t, &dev);
    let b_c = gather_rows(
        &Tensor::from_vec(b_g.clone(), n_f, &dev).unwrap(),
        plain.compact_feat(),
    )
    .unwrap();
    let pool = pool_of(&plain, &t, &dev);
    assert!(plain.dictionary_blend(&pool, &r_c, &b_c).unwrap().is_none());

    let cis = plain.with_mix(ALPHA, n_f, &dev).unwrap();
    let blend = cis
        .dictionary_blend(&pool, &r_c, &b_c)
        .unwrap()
        .expect("a blend under a mixture");
    let mut rho = nalgebra::DMatrix::<f32>::from_fn(n_f, h, |f, d| {
        t.mu[module_of[f] as usize * h + d] + t.r[f * h + d]
    });
    let mut b: Vec<f32> = (0..n_f)
        .map(|f| t.b_m[module_of[f] as usize] + b_g[f])
        .collect();
    let (rho0, b0) = (rho.clone(), b.clone());
    blend.apply(&mut rho, &mut b);

    let w = hand_w(&cis, &spec);
    for f in 0..n_f {
        if !CIS_GENES.contains(&f) {
            assert_eq!(rho.row(f), rho0.row(f), "feature {f} moved");
            assert_eq!(b[f], b0[f], "feature {f} bias moved");
            continue;
        }
        let nu = hand_nu(&spec, &w, &t, f);
        let m = module_of[f] as usize;
        for d in 0..h {
            let want = t.mu[m * h + d] + (1.0 - ALPHA) * t.r[f * h + d] + ALPHA * nu[d];
            assert!((rho[(f, d)] - want).abs() < 1e-5, "row {f} col {d}");
        }
        let pooled_b: f32 = (0..spec.n_pairs())
            .filter(|&k| spec.gene_feat[k] as usize == f)
            .map(|k| w[k] * t.b_m[spec.peak_module[k] as usize])
            .sum();
        let want_b = t.b_m[m] + (1.0 - ALPHA) * b_g[f] + ALPHA * pooled_b;
        assert!(
            (b[f] - want_b).abs() < 1e-5,
            "bias {f}: {} vs {want_b}",
            b[f]
        );
    }
}

/// Pearson correlation of two equal-length series.
fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let (mx, my) = (x.iter().sum::<f32>() / n, y.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0f32, 0f32, 0f32);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx * syy).sqrt()
}

/// The readout's link evidence: per pair, the correlation across units of
/// the peak module's score `⟨e_u, μ_{m(p)}⟩` and the gene's own score
/// `⟨e_u, ρ_g⟩`; it also reports the shares and the gap.
#[test]
fn the_readout_correlates_peak_and_gene_scores_across_units() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let t = tables(7, 4, module_of.len(), 3);
    let h = t.h;
    let (mu, r_c, e) = on_device(&cis, &t, &dev);
    let out = cis
        .readout(&pool_of(&cis, &t, &dev), &mu, &r_c, &e)
        .unwrap();

    for (a, b) in out.w.iter().zip(&hand_w(&cis, &spec)) {
        assert!((a - b).abs() < 1e-6, "w {a} vs {b}");
    }
    assert!((out.align_gap - hand_gap(&cis, &spec, &module_of, &t)).abs() < 1e-4);
    assert_eq!(out.corr.len(), spec.n_pairs());
    for k in 0..spec.n_pairs() {
        let g = spec.gene_feat[k] as usize;
        let m = spec.peak_module[k] as usize;
        let mg = module_of[g] as usize;
        let rho: Vec<f32> = (0..h).map(|d| t.mu[mg * h + d] + t.r[g * h + d]).collect();
        let (x, y): (Vec<f32>, Vec<f32>) = (0..t.n_u)
            .map(|u| {
                let e = &t.e[u * h..(u + 1) * h];
                (dot(e, &t.mu[m * h..(m + 1) * h]), dot(e, &rho))
            })
            .unzip();
        let want = pearson(&x, &y);
        assert!(
            (out.corr[k] - want).abs() < 1e-4,
            "pair {k}: {} vs {want}",
            out.corr[k]
        );
    }
}

/// A peak module whose score does not vary across units has no evidence
/// either way: correlation `0`, not `NaN`.
#[test]
fn a_flat_peak_module_scores_zero_evidence() {
    let dev = Device::Cpu;
    let (spec, module_of) = small_spec();
    let cis = CisGateParams::new(&spec, &module_of, &dev).unwrap();
    let mut t = tables(9, 4, module_of.len(), 3);
    t.mu[2 * 3..3 * 3].fill(0.0);
    let (mu, r_c, e) = on_device(&cis, &t, &dev);
    let out = cis
        .readout(&pool_of(&cis, &t, &dev), &mu, &r_c, &e)
        .unwrap();
    for k in 0..spec.n_pairs() {
        if spec.peak_module[k] == 2 {
            assert_eq!(out.corr[k], 0.0, "pair {k}");
        } else {
            assert!(out.corr[k].is_finite() && out.corr[k] != 0.0, "pair {k}");
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
