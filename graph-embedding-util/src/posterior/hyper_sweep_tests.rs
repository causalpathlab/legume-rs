//! The interleaved Tier-1 joint Gibbs recovers a planted `σ₀²` with a
//! well-mixing (non-stalled, non-funneled) σ²-chain — the "is interleaving too
//! flaky?" gate.

use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, StandardNormal};

const H: usize = 8;
const N_GENES: usize = 200;
const N_CELL: usize = 60;

fn randn(sd: f32, rng: &mut StdRng) -> f32 {
    let z: f64 = StandardNormal.sample(rng);
    z as f32 * sd
}

#[test]
fn interleaved_tier1_recovers_sigma0_and_mixes() {
    let mut rng = StdRng::seed_from_u64(20260724);

    // Frozen cell side (random, modest bias floor).
    let e_cell: Vec<f32> = (0..N_CELL * H).map(|_| randn(0.7, &mut rng)).collect();
    let b_cell: Vec<f32> = (0..N_CELL).map(|_| -0.5).collect();

    // Planted global slab variance σ₀²; every gene's effect ~ N(0, σ₀²·I).
    let sigma0_sq_true = 0.30f64;
    let sd = sigma0_sq_true.sqrt() as f32;
    let mut e_gene = vec![0.0f32; N_GENES * H];
    let b_gene: Vec<f32> = (0..N_GENES).map(|_| -0.3).collect();
    for v in &mut e_gene {
        *v = randn(sd, &mut rng);
    }

    // Poisson counts against the frozen cell side.
    let mut pos: Vec<Vec<(u32, f32)>> = vec![Vec::new(); N_GENES];
    for g in 0..N_GENES {
        let eg = &e_gene[g * H..(g + 1) * H];
        for c in 0..N_CELL {
            let ec = &e_cell[c * H..(c + 1) * H];
            let dot: f64 = eg
                .iter()
                .zip(ec)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let s = dot + f64::from(b_gene[g]) + f64::from(b_cell[c]);
            let n = Poisson::new(s.exp()).unwrap().sample(&mut rng);
            if n > 0.0 {
                pos[g].push((c as u32, n as f32));
            }
        }
    }

    let side = FrozenSide {
        e: &e_cell,
        b: &b_cell,
        h: H,
    };
    let all_cells: Vec<u32> = (0..N_CELL as u32).collect();
    let nodes: Vec<NodeTerm> = pos
        .iter()
        .map(|p| NodeTerm::new(p, &all_cells, 1.0))
        .collect();
    let inits: Vec<Vec<f32>> = (0..N_GENES).map(|_| vec![0.0f32; H + 1]).collect();

    let cfg = HyperSweepConfig::new(300, 100, 7);
    let res = hyper_sweep(&nodes, &inits, &side, &cfg);

    // (1) Recovery: σ₀² posterior mean near the plant (mild shrinkage tolerated).
    let rel = (res.effect_var_mean - sigma0_sq_true).abs() / sigma0_sq_true;
    assert!(
        rel < 0.35,
        "interleaved σ₀² should recover the plant: got {:.4}, true {sigma0_sq_true:.4}",
        res.effect_var_mean
    );

    // (2) NOT flaky: the σ²-chain mixes — real ESS, not stalled at a value.
    assert!(
        res.sigma_diag.stuck_fraction < 0.2,
        "σ²-chain should not be stuck (funnel/fallback): {:.3}",
        res.sigma_diag.stuck_fraction
    );
    assert!(
        res.sigma_diag.min_ess > 0.1 * res.effect_var_chain.len() as f32,
        "σ²-chain ESS should be a real fraction of the retained draws: {:.1}/{}",
        res.sigma_diag.min_ess,
        res.effect_var_chain.len()
    );

    // (3) The per-gene θ still tracks truth (score cosine), so the hyper draw is
    // not being fed garbage effects.
    let mut cos_sum = 0.0f64;
    for g in 0..N_GENES {
        let th = &res.theta_mean[g];
        let (eg_hat, bg_hat) = (&th[..H], th[H]);
        let eg = &e_gene[g * H..(g + 1) * H];
        let (mut ts, mut hs) = (Vec::new(), Vec::new());
        for c in 0..N_CELL {
            let ec = &e_cell[c * H..(c + 1) * H];
            let dot_t: f64 = eg
                .iter()
                .zip(ec)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let dot_h: f64 = eg_hat
                .iter()
                .zip(ec)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            ts.push(dot_t + f64::from(b_gene[g]) + f64::from(b_cell[c]));
            hs.push(dot_h + f64::from(bg_hat) + f64::from(b_cell[c]));
        }
        let dot: f64 = ts.iter().zip(&hs).map(|(a, b)| a * b).sum();
        let na: f64 = ts.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = hs.iter().map(|x| x * x).sum::<f64>().sqrt();
        cos_sum += dot / (na * nb);
    }
    let mean_cos = cos_sum / N_GENES as f64;
    assert!(
        mean_cos > 0.9,
        "per-gene θ should track truth under the joint chain (mean cos={mean_cos:.3})"
    );
}
