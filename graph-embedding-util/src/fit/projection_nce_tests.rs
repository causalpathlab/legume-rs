//! Fidelity test for the frozen-feature NCE projection: on synthetic Poisson
//! data with a KNOWN feature dictionary and planted per-cell θ, the block NCE
//! fit should recover both the analytical Poisson-MAP direction and the planted
//! θ direction. Deterministic (CPU, seeded) — no GPU-noise confound, unlike a
//! cross-run comparison.

use super::{fit_cell_block, NceProjectionOpts};
use crate::cell_projection::solve_one_cell;
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

#[test]
fn nce_block_fit_recovers_analytic_and_planted_direction() {
    let dev = Device::Cpu;
    let (h, nfeat, ncell, ntypes) = (16usize, 200usize, 40usize, 4usize);
    let mut rng = StdRng::seed_from_u64(7);
    let normal = Normal::new(0.0f64, 1.0 / (h as f64).sqrt()).unwrap();

    // Planted feature dictionary [nfeat × h] row-major.
    let e_feat_flat: Vec<f32> = (0..nfeat * h)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();
    let b_feat = vec![0f32; nfeat];

    // `ntypes` cell types, each a random θ direction of norm ~2 (real signal).
    let type_theta: Vec<Vec<f32>> = (0..ntypes)
        .map(|_| {
            let raw: Vec<f32> = (0..h).map(|_| normal.sample(&mut rng) as f32).collect();
            let n = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            raw.iter().map(|x| x / n * 2.0).collect()
        })
        .collect();

    // Poisson counts per cell from its type θ + a library intercept.
    let mut cells: Vec<(Vec<u32>, Vec<f32>)> = Vec::new();
    let mut true_theta: Vec<Vec<f32>> = Vec::new();
    for c in 0..ncell {
        let theta = &type_theta[c % ntypes];
        true_theta.push(theta.clone());
        let mut feats = Vec::new();
        let mut counts = Vec::new();
        for g in 0..nfeat {
            let s: f64 = (0..h)
                .map(|d| f64::from(e_feat_flat[g * h + d]) * f64::from(theta[d]))
                .sum::<f64>()
                + 1.5; // library intercept → moderate counts
            let rate = s.clamp(-20.0, 5.0).exp();
            let cnt = Poisson::new(rate).unwrap().sample(&mut rng);
            if cnt > 0.5 {
                feats.push(g as u32);
                counts.push(cnt as f32);
            }
        }
        cells.push((feats, counts));
    }

    // NCE block fit against the frozen dictionary.
    let e_feat_t = Tensor::from_vec(e_feat_flat.clone(), (nfeat, h), &dev).unwrap();
    let b_feat_t = Tensor::from_vec(b_feat.clone(), nfeat, &dev).unwrap();
    let block: Vec<(&[u32], &[f32])> = cells
        .iter()
        .map(|(f, c)| (f.as_slice(), c.as_slice()))
        .collect();
    let opts = NceProjectionOpts {
        objective: NceObjective::Softmax,
        block_size: ncell,
        epochs: 500, // batch ≥ block ⇒ 1 epoch = 1 step ⇒ ~3.2k samples/cell
        batch_size: 256,
        n_negatives: 10,
        learning_rate: 0.1,
        seed: 123,
    };
    let e_nce = fit_cell_block(&e_feat_t, &b_feat_t, &block, h, &opts, 123, &dev, false).unwrap();

    // Compare per cell: nce vs analytic, nce vs planted θ, analytic vs planted θ.
    let (mut c_na, mut c_nt, mut c_at) = (0.0f32, 0.0f32, 0.0f32);
    for c in 0..ncell {
        let edges: Vec<(u32, f32)> = cells[c]
            .0
            .iter()
            .copied()
            .zip(cells[c].1.iter().copied())
            .collect();
        let (e_a, _b) = solve_one_cell(&edges, &e_feat_flat, &b_feat, h, 1.0);
        let e_n = &e_nce[c * h..(c + 1) * h];
        c_na += cos(e_n, &e_a);
        c_nt += cos(e_n, &true_theta[c]);
        c_at += cos(&e_a, &true_theta[c]);
    }
    let n = ncell as f32;
    let (m_na, m_nt, m_at) = (c_na / n, c_nt / n, c_at / n);
    eprintln!("mean cos: nce~analytic={m_na:.3}  nce~θ={m_nt:.3}  analytic~θ={m_at:.3}");

    // The analytic MAP is the gold reference — it should track planted θ tightly.
    assert!(m_at > 0.7, "analytic should recover planted θ (got {m_at:.3})");
    // The NCE fit should align with both the planted θ and the analytic solve.
    assert!(m_nt > 0.5, "nce should recover planted θ (got {m_nt:.3})");
    assert!(m_na > 0.6, "nce should match analytic direction (got {m_na:.3})");
}
