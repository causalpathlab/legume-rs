//! The spike-and-slab Tier-1 recovers both σ₀² and π₀ and separates included from
//! null genes.

use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, StandardNormal};

const H: usize = 8;
const N_GENES: usize = 300;
const N_CELL: usize = 80;

fn randn(sd: f32, rng: &mut StdRng) -> f32 {
    let z: f64 = StandardNormal.sample(rng);
    z as f32 * sd
}

#[test]
fn spike_slab_tier1_recovers_sigma0_and_pi0() {
    let mut rng = StdRng::seed_from_u64(20260724);

    // Frozen cell side (dense → informative likelihood, so null genes clearly
    // prefer z=0 rather than reverting to the prior inclusion).
    let e_cell: Vec<f32> = (0..N_CELL * H).map(|_| randn(0.7, &mut rng)).collect();
    let b_cell: Vec<f32> = (0..N_CELL).map(|_| 0.2).collect(); // higher floor ⇒ more counts

    // Plant: the first (1-π₀) fraction are INCLUDED (effect ~ N(0,σ₀²)); the rest
    // are NULL (zero effect).
    let pi0_true = 0.40f64;
    let n_null_true = (pi0_true * N_GENES as f64) as usize; // 120
    let sigma0_sq_true = 0.40f64;
    let sd = sigma0_sq_true.sqrt() as f32;
    let mut e_gene = vec![0.0f32; N_GENES * H];
    let b_gene: Vec<f32> = (0..N_GENES).map(|_| 0.0).collect();
    for g in 0..(N_GENES - n_null_true) {
        for k in 0..H {
            e_gene[g * H + k] = randn(sd, &mut rng);
        }
    }

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

    let cfg = HyperSsConfig::new(400, 150, 7);
    let res = hyper_ss(&nodes, &side, &cfg);

    eprintln!(
        "σ₀²={:.3} (true {:.3}), π₀={:.3} (true {:.3}); σ-ESS {:.0} π-ESS {:.0}",
        res.sigma2_mean,
        sigma0_sq_true,
        res.pi0_mean,
        pi0_true,
        res.sigma_diag.min_ess,
        res.pi0_diag.min_ess
    );

    // σ₀² recovered from the included slab.
    let rel = (res.sigma2_mean - sigma0_sq_true).abs() / sigma0_sq_true;
    assert!(
        rel < 0.4,
        "σ₀² off: got {:.3}, true {sigma0_sq_true:.3}",
        res.sigma2_mean
    );

    // π₀ recovered (directionally): moved off the 0.9 prior toward the true 0.40.
    assert!(
        (res.pi0_mean - pi0_true).abs() < 0.18,
        "π₀ off: got {:.3}, true {pi0_true:.3}",
        res.pi0_mean
    );

    // Inclusion separates: truly-included genes score higher than truly-null.
    let incl_mean: f32 = (0..(N_GENES - n_null_true))
        .map(|g| res.inclusion_prob[g])
        .sum::<f32>()
        / (N_GENES - n_null_true) as f32;
    let null_mean: f32 = ((N_GENES - n_null_true)..N_GENES)
        .map(|g| res.inclusion_prob[g])
        .sum::<f32>()
        / n_null_true as f32;
    assert!(
        incl_mean > null_mean + 0.3,
        "inclusion should separate: included {incl_mean:.2} vs null {null_mean:.2}"
    );

    // Both chains mix.
    assert!(res.sigma_diag.stuck_fraction < 0.3 && res.pi0_diag.stuck_fraction < 0.3);
}
