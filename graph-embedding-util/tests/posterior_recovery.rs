//! Stage-A validation for `graph_embedding_util::posterior`.
//!
//! Plants a known feature/cell embedding, generates Poisson counts at the exact
//! model rate, FREEZES the cell side at truth, and samples the feature-side
//! posterior. With one side fixed at a known basis there is no rotational gauge
//! freedom, so we can check recovery and interval coverage directly.
//!
//! Two assertions gate the approach:
//!   1. recovery — the posterior-mean reconstructed score `⟨e_f, e_c⟩` matches the
//!      planted score (high cosine);
//!   2. coverage — per-(feature,cell) central credible intervals cover the planted
//!      score at ≈ the nominal rate. This is what turns the sampler's SD from "a
//!      lower bound" into a calibrated statement.

use graph_embedding_util::posterior::{sweep_side, FrozenSide, NodeTerm, SweepConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, StandardNormal};

const H: usize = 4;
const N_FEAT: usize = 60;
const N_CELL: usize = 40;

/// Row-major `[rows × H]` of N(0, sd²) draws (seeded).
fn randn_rows(rows: usize, sd: f32, rng: &mut SmallRng) -> Vec<f32> {
    (0..rows * H)
        .map(|_| {
            let z: f64 = StandardNormal.sample(rng);
            z as f32 * sd
        })
        .collect()
}

fn score(e_a: &[f32], b_a: f32, e_o: &[f32], b_o: f32) -> f64 {
    let dot: f64 = e_a
        .iter()
        .zip(e_o)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    dot + f64::from(b_a) + f64::from(b_o)
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn feature_posterior_recovers_and_covers() {
    let mut rng = SmallRng::seed_from_u64(20260723);

    // Planted truth. Small embeddings + a negative bias floor keep the Poisson
    // rates modest (counts in the low single digits, like real sparse data).
    let e_feat = randn_rows(N_FEAT, 0.6, &mut rng);
    let e_cell = randn_rows(N_CELL, 0.6, &mut rng);
    let b_feat: Vec<f32> = (0..N_FEAT).map(|_| -0.5).collect();
    let b_cell: Vec<f32> = (0..N_CELL).map(|_| -0.5).collect();

    // Observed counts n_fc ~ Poisson(exp(s_fc)); keep each feature's nonzero edges.
    let mut pos_by_feat: Vec<Vec<(u32, f32)>> = vec![Vec::new(); N_FEAT];
    for f in 0..N_FEAT {
        let ef = &e_feat[f * H..(f + 1) * H];
        for c in 0..N_CELL {
            let ec = &e_cell[c * H..(c + 1) * H];
            let s = score(ef, b_feat[f], ec, b_cell[c]);
            let lam = s.exp();
            let n = Poisson::new(lam).unwrap().sample(&mut rng);
            if n > 0.0 {
                pos_by_feat[f].push((c as u32, n as f32));
            }
        }
    }

    // Freeze the cell side at truth; the exact partition is every cell.
    let side = FrozenSide {
        e: &e_cell,
        b: &b_cell,
        h: H,
    };
    let all_cells: Vec<u32> = (0..N_CELL as u32).collect();
    let nodes: Vec<NodeTerm> = pos_by_feat
        .iter()
        .map(|pos| NodeTerm::new(pos, &all_cells, 1.0))
        .collect();

    // Warm start at zero (no MAP available in the unit test); ESS finds the mode.
    let inits: Vec<Vec<f32>> = (0..N_FEAT).map(|_| vec![0.0f32; H + 1]).collect();

    let cfg = SweepConfig::new(3000, 1000, 0.6, 7);
    let post = sweep_side(&nodes, &inits, &side, &cfg);

    // (1) Recovery: posterior-mean reconstructed scores vs planted, per feature.
    let mut cos_sum = 0.0;
    for (f, np) in post.iter().enumerate() {
        let (ef_hat, bf_hat) = (np.e_mean(), np.b_mean());
        let ef = &e_feat[f * H..(f + 1) * H];
        let mut true_s = Vec::with_capacity(N_CELL);
        let mut hat_s = Vec::with_capacity(N_CELL);
        for c in 0..N_CELL {
            let ec = &e_cell[c * H..(c + 1) * H];
            true_s.push(score(ef, b_feat[f], ec, b_cell[c]));
            hat_s.push(score(ef_hat, bf_hat, ec, b_cell[c]));
        }
        cos_sum += cosine(&true_s, &hat_s);
    }
    let mean_cos = cos_sum / N_FEAT as f64;
    assert!(
        mean_cos > 0.95,
        "posterior mean should reconstruct the score matrix (mean cos={mean_cos:.3})"
    );

    // (2) Coverage: 90% central credible interval per (feature, cell) score.
    let (mut covered, mut total) = (0usize, 0usize);
    for (f, np) in post.iter().enumerate() {
        if np.samples.is_empty() {
            continue;
        }
        let ef = &e_feat[f * H..(f + 1) * H];
        for c in 0..N_CELL {
            let ec = &e_cell[c * H..(c + 1) * H];
            let truth = score(ef, b_feat[f], ec, b_cell[c]);
            // Reconstructed score under each retained draw → its 5%/95% quantiles.
            let mut draws: Vec<f64> = np
                .samples
                .iter()
                .map(|th| {
                    let (e_a, b_a) = (&th[..H], th[H]);
                    score(e_a, b_a, ec, b_cell[c])
                })
                .collect();
            draws.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = draws[(0.05 * draws.len() as f64) as usize];
            let hi = draws[((0.95 * draws.len() as f64) as usize).min(draws.len() - 1)];
            if truth >= lo && truth <= hi {
                covered += 1;
            }
            total += 1;
        }
    }
    let cov = covered as f64 / total as f64;
    assert!(
        (0.83..=0.97).contains(&cov),
        "90% credible interval coverage should be ≈ nominal, got {cov:.3}"
    );
}
