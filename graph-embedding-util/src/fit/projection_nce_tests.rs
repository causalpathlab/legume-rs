//! Fidelity test for the frozen-feature NCE projection on synthetic Poisson data
//! with a KNOWN dictionary. Covers BOTH passes gem uses:
//!   θ (identity) from spliced edges, and δ (velocity increment) from unspliced
//!   edges with the base offset θ. Asserts each recovers its analytic solve
//!   (`solve_one_cell` / `solve_cell_increment`) and its planted direction —
//!   and, crucially, that the contrastive δ does NOT collapse onto `−θ` the way
//!   the analytic per-cell increment does. Deterministic (CPU, seeded).

use super::{fit_cell_block, NceProjectionOpts};
use crate::cell_projection::{solve_cell_increment, solve_one_cell};
use crate::loss::NceObjective;
use candle_util::candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson};

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn rand_dir(h: usize, norm: f32, normal: &Normal<f64>, rng: &mut StdRng) -> Vec<f32> {
    let raw: Vec<f32> = (0..h).map(|_| normal.sample(rng) as f32).collect();
    let n = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    raw.iter().map(|x| x / n * norm).collect()
}

#[test]
fn nce_block_fit_recovers_theta_and_delta() {
    let dev = Device::Cpu;
    let (h, n_genes, ncell, ntypes) = (16usize, 100usize, 40usize, 4usize);
    let mut rng = StdRng::seed_from_u64(7);
    let normal = Normal::new(0.0f64, 1.0 / (h as f64).sqrt()).unwrap();

    // Feature layout: row 2g = spliced (β_g), row 2g+1 = unspliced (β_g + δ_g).
    let nfeat = 2 * n_genes;
    let mut e_feat = vec![0f32; nfeat * h];
    let mut unspliced_rows = vec![false; nfeat];
    for g in 0..n_genes {
        let beta: Vec<f32> = (0..h).map(|_| normal.sample(&mut rng) as f32).collect();
        let delta_g: Vec<f32> = (0..h).map(|_| 0.3 * normal.sample(&mut rng) as f32).collect();
        for d in 0..h {
            e_feat[(2 * g) * h + d] = beta[d];
            e_feat[(2 * g + 1) * h + d] = beta[d] + delta_g[d];
        }
        unspliced_rows[2 * g + 1] = true;
    }
    let b_feat = vec![0f32; nfeat];

    // Per-type identity θ (norm 2) and a forward velocity ψ (norm 1), independent
    // directions — ψ is deliberately NOT aligned with −θ.
    let type_theta: Vec<Vec<f32>> = (0..ntypes)
        .map(|_| rand_dir(h, 2.0, &normal, &mut rng))
        .collect();
    let type_psi: Vec<Vec<f32>> = (0..ntypes)
        .map(|_| rand_dir(h, 1.0, &normal, &mut rng))
        .collect();

    // Poisson counts: spliced ~ exp(⟨β_g, θ⟩+b), unspliced ~ exp(⟨β_g+δ_g, θ+ψ⟩+b).
    let (mut sp, mut un): (Vec<(Vec<u32>, Vec<f32>)>, Vec<(Vec<u32>, Vec<f32>)>) =
        (Vec::new(), Vec::new());
    let (mut true_theta, mut true_psi): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
    for c in 0..ncell {
        let t = c % ntypes;
        let theta = &type_theta[t];
        let psi = &type_psi[t];
        true_theta.push(theta.clone());
        true_psi.push(psi.clone());
        let np: Vec<f32> = theta.iter().zip(psi).map(|(a, b)| a + b).collect();
        let (mut sf, mut sc, mut uf, mut uc) = (vec![], vec![], vec![], vec![]);
        for g in 0..n_genes {
            let sr = &e_feat[(2 * g) * h..(2 * g + 1) * h];
            let ur = &e_feat[(2 * g + 1) * h..(2 * g + 2) * h];
            let ss: f64 = sr.iter().zip(theta).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() + 1.5;
            let us: f64 = ur.iter().zip(&np).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() + 1.5;
            let sn = Poisson::new(ss.clamp(-20.0, 5.0).exp()).unwrap().sample(&mut rng);
            let un_ = Poisson::new(us.clamp(-20.0, 5.0).exp()).unwrap().sample(&mut rng);
            if sn > 0.5 {
                sf.push(2 * g as u32);
                sc.push(sn as f32);
            }
            if un_ > 0.5 {
                uf.push((2 * g + 1) as u32);
                uc.push(un_ as f32);
            }
        }
        sp.push((sf, sc));
        un.push((uf, uc));
    }

    let e_feat_t = Tensor::from_vec(e_feat.clone(), (nfeat, h), &dev).unwrap();
    let b_feat_t = Tensor::from_vec(b_feat.clone(), nfeat, &dev).unwrap();
    let opts = NceProjectionOpts {
        objective: NceObjective::Softmax,
        block_size: ncell,
        epochs: 500,
        batch_size: 256,
        n_negatives: 10,
        learning_rate: 0.1,
        seed: 123,
    };

    // Pass 1: θ from spliced (base = None).
    let sp_block: Vec<(&[u32], &[f32])> =
        sp.iter().map(|(f, c)| (f.as_slice(), c.as_slice())).collect();
    let (theta_nce, _) =
        fit_cell_block(&e_feat_t, &b_feat_t, &sp_block, h, &opts, 123, &dev, false, None).unwrap();

    // Pass 2: δ from unspliced with base = θ_nce (cell-side θ+δ).
    let theta_t = Tensor::from_vec(theta_nce.clone(), (ncell, h), &dev).unwrap();
    let un_block: Vec<(&[u32], &[f32])> =
        un.iter().map(|(f, c)| (f.as_slice(), c.as_slice())).collect();
    let (delta_nce, _) =
        fit_cell_block(&e_feat_t, &b_feat_t, &un_block, h, &opts, 456, &dev, false, Some(&theta_t))
            .unwrap();

    // Per-cell comparisons.
    let (mut t_an, mut t_tr) = (0.0f32, 0.0f32); // θ: nce~analytic, nce~true
    let (mut d_an, mut d_tr, mut d_theta) = (0.0f32, 0.0f32, 0.0f32); // δ: ~analytic, ~true ψ, ~θ
    for c in 0..ncell {
        let sp_edges: Vec<(u32, f32)> = sp[c].0.iter().copied().zip(sp[c].1.iter().copied()).collect();
        let un_edges: Vec<(u32, f32)> = un[c].0.iter().copied().zip(un[c].1.iter().copied()).collect();
        let th_n = &theta_nce[c * h..(c + 1) * h];
        let dl_n = &delta_nce[c * h..(c + 1) * h];
        let (th_a, _) = solve_one_cell(&sp_edges, &e_feat, &b_feat, h, 1.0);
        let (dl_a, _) = solve_cell_increment(&un_edges, th_n, &e_feat, &b_feat, h, 1.0);
        t_an += cos(th_n, &th_a);
        t_tr += cos(th_n, &true_theta[c]);
        d_an += cos(dl_n, &dl_a);
        d_tr += cos(dl_n, &true_psi[c]);
        d_theta += cos(dl_n, th_n);
    }
    let n = ncell as f32;
    let (t_an, t_tr, d_an, d_tr, d_theta) = (t_an / n, t_tr / n, d_an / n, d_tr / n, d_theta / n);
    eprintln!(
        "θ: nce~analytic={t_an:.3} nce~true={t_tr:.3} | δ: nce~analytic={d_an:.3} nce~ψ={d_tr:.3} nce~θ={d_theta:.3}"
    );

    // θ recovers the analytic MAP and the planted identity.
    assert!(t_an > 0.6, "θ_nce should match analytic θ (got {t_an:.3})");
    assert!(t_tr > 0.5, "θ_nce should recover planted θ (got {t_tr:.3})");
    // δ tracks the analytic increment and the planted forward velocity ψ.
    assert!(d_an > 0.4, "δ_nce should track analytic increment (got {d_an:.3})");
    assert!(d_tr > 0.3, "δ_nce should recover planted ψ (got {d_tr:.3})");
    // The contrastive δ should not be MORE shrunk onto −θ than the analytic δ_c
    // (whose real-data signature is cos≈−0.65). This is a loose guard; the real
    // anti-shrinkage assessment is on data (compare the cos(δ,θ) distribution to
    // analytic δ_c — plan step 5), not a synthetic unit invariant.
    assert!(d_theta > -0.6, "δ_nce more shrunk than analytic δ_c (cos(δ,θ)={d_theta:.3})");
}
