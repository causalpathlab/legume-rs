//! The pair projection solves a known problem: with the dictionary frozen and the
//! counts generated from a known `e_uv`, the MAP is that `e_uv`. These tests
//! generate exactly that and check the solver lands on it — through both the
//! exhaustive and the sampled partition — plus the two properties the design
//! rests on: `β_uv` absorbs pooled depth, and a pair with no counts stays at
//! the origin rather than being handed a fabricated direction.

use crate::cell_activity_graph_embedding::pair_projection::{PairDictionary, ProjectionArgs};
use crate::util::common::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

const N_GENES: usize = 240;
const DIM: usize = 4;
const N_CELLS: usize = 100;

/// Deterministic spread in `[-0.2, 0.2)`, so the off-block dimensions are not
/// all identical and the design matrix is not rank-4-with-4-distinct-rows.
fn jitter(seed: usize) -> f32 {
    let h = (seed.wrapping_mul(2_654_435_761)) % 1000;
    (h as f32 / 1000.0 - 0.5) * 0.4
}

/// `[G × D]` frozen dictionary: gene `g` loads mainly on dim `g % D`.
fn dictionary_matrix() -> Mat {
    let mut e = Mat::zeros(N_GENES, DIM);
    for g in 0..N_GENES {
        for j in 0..DIM {
            e[(g, j)] = if j == g % DIM {
                1.0 + 0.1 * ((g / DIM) % 5) as f32
            } else {
                jitter(g * DIM + j)
            };
        }
    }
    e
}

/// Log gene abundance the offsets are built from, and the totals that imply it.
fn abundances() -> (Vec<f32>, Vec<f64>) {
    let b: Vec<f32> = (0..N_GENES).map(|g| (5.0 + (g % 7) as f32).ln()).collect();
    let totals: Vec<f64> = b.iter().map(|&x| x.exp() as f64 * N_CELLS as f64).collect();
    (b, totals)
}

/// Pooled counts a pair at `theta` with intercept `beta` would produce, exactly
/// (no Poisson draw), so the MAP is `theta` up to the ridge.
fn counts_from(e: &Mat, b: &[f32], theta: &[f32], beta: f32) -> Vec<(u32, f32)> {
    (0..N_GENES)
        .map(|g| {
            let s: f32 = (0..DIM).map(|j| e[(g, j)] * theta[j]).sum::<f32>() + b[g] + beta;
            (g as u32, s.exp())
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-8)
}

fn norm(a: &[f32]) -> f32 {
    a.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn args(steps: usize, gene_sample: usize) -> ProjectionArgs {
    ProjectionArgs {
        // Deliberately near-zero: these tests check that the *likelihood*
        // recovers the truth, and a working ridge would bias the norm down.
        ridge: 1e-4,
        steps,
        gene_sample,
    }
}

#[test]
fn projection_recovers_known_pair_embedding() {
    let e = dictionary_matrix();
    let (b, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");
    assert_eq!(dict.n_active(), N_GENES);

    let truth = [0.6f32, -0.4, 0.25, 0.0];
    let beta_truth = 2.0f32.ln();
    let obs = counts_from(&e, &b, &truth, beta_truth);

    let mut rng = SmallRng::seed_from_u64(1);
    let (theta, beta) = dict.project(&obs, &args(1500, 0), &mut rng);

    assert!(
        cosine(&theta, &truth) > 0.98,
        "direction off: cos = {}, theta = {theta:?}",
        cosine(&theta, &truth)
    );
    let rel = (norm(&theta) - norm(&truth)).abs() / norm(&truth);
    assert!(rel < 0.15, "scale off: ‖θ̂‖ = {}, rel = {rel}", norm(&theta));
    assert!(
        (beta - beta_truth).abs() < 0.1,
        "intercept off: {beta} vs {beta_truth}"
    );
}

#[test]
fn sampled_partition_recovers_the_same_direction() {
    let e = dictionary_matrix();
    let (b, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");

    let truth = [0.5f32, -0.3, 0.2, 0.1];
    let obs = counts_from(&e, &b, &truth, 0.0);

    let mut rng = SmallRng::seed_from_u64(2);
    // A third of the gene axis per step: the proposal cancels `exp(b_g)`, so
    // what is left is unbiased and only mildly noisy.
    let (theta, _) = dict.project(&obs, &args(1500, 80), &mut rng);

    assert!(
        cosine(&theta, &truth) > 0.95,
        "sampled partition drifted: cos = {}, theta = {theta:?}",
        cosine(&theta, &truth)
    );
}

#[test]
fn intercept_absorbs_pooled_depth() {
    let e = dictionary_matrix();
    let (b, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");

    let truth = [0.4f32, -0.5, 0.1, 0.2];
    let shallow = counts_from(&e, &b, &truth, 0.0);
    // Same composition, ten times the depth.
    let deep: Vec<(u32, f32)> = shallow.iter().map(|&(g, n)| (g, n * 10.0)).collect();

    let mut rng = SmallRng::seed_from_u64(3);
    let (theta_shallow, beta_shallow) = dict.project(&shallow, &args(1500, 0), &mut rng);
    let mut rng = SmallRng::seed_from_u64(3);
    let (theta_deep, beta_deep) = dict.project(&deep, &args(1500, 0), &mut rng);

    // Depth lands entirely on the intercept…
    assert!(
        (beta_deep - beta_shallow - 10.0f32.ln()).abs() < 0.05,
        "β did not track depth: {beta_shallow} → {beta_deep}"
    );
    // …and leaves the embedding alone, which is the whole point of fitting it.
    assert!(
        cosine(&theta_shallow, &theta_deep) > 0.999,
        "depth moved the latent: cos = {}",
        cosine(&theta_shallow, &theta_deep)
    );
}

#[test]
fn empty_profile_stays_at_the_origin() {
    let e = dictionary_matrix();
    let (_, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");

    let mut rng = SmallRng::seed_from_u64(4);
    let (theta, beta) = dict.project(&[], &args(100, 0), &mut rng);
    assert_eq!(theta, vec![0.0; DIM]);
    assert_eq!(beta, 0.0);

    // A gene that carries no counts anywhere is not on the partition axis, so a
    // profile made only of such genes is empty too — not a direction.
    let mut totals_with_dead = totals.clone();
    totals_with_dead[0] = 0.0;
    let dict = PairDictionary::new(&e, &totals_with_dead, N_CELLS).expect("dictionary");
    assert_eq!(dict.n_active(), N_GENES - 1);
    let (theta, _) = dict.project(&[(0, 12.0)], &args(100, 0), &mut rng);
    assert_eq!(theta, vec![0.0; DIM]);
}
