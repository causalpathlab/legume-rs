//! The pair projection solves a known problem: with the dictionary frozen and the
//! counts generated from a known `e_uv`, the MAP is that `e_uv`. These tests
//! generate exactly that and check the solver lands on it — through both the
//! exhaustive and the sampled partition — plus the two properties the design
//! rests on: `β_uv` absorbs pooled depth, and a pair with no counts stays at
//! the origin rather than being handed a fabricated direction.

use super::fixture::*;
use crate::cell_activity_graph_embedding::pair_projection::{PairDictionary, ProjectionArgs};
use rand::rngs::SmallRng;
use rand::SeedableRng;

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

//////////////////////////////
// Hold-out split of a pair //
//////////////////////////////

#[test]
fn newton_polish_lands_where_the_converged_solve_lands() {
    let e = dictionary_matrix();
    let (b, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");
    let truth = [0.6f32, -0.4, 0.25, 0.0];
    let obs = counts_from(&e, &b, &truth, 0.3);
    let mut rng = SmallRng::seed_from_u64(1);
    let (theta, beta) = dict.project(&obs, &args(1500, 0), &mut rng);
    // From a start well off the optimum.
    let start = [0.1f32, 0.1, -0.1, 0.2];
    let (polished, beta_polished, certificate) = dict.polish(&obs, 1e-4, &start, 8);
    // The bound is `‖∇‖²/(2λ)`, so a near-zero ridge inflates it; small
    // against the pair's thousands of nats of likelihood is what settled means.
    assert!(
        certificate < 1.0,
        "certificate {certificate} at the optimum"
    );
    assert!(
        cosine(&polished, &theta) > 0.9999,
        "direction: {polished:?} vs {theta:?}"
    );
    assert!(
        (norm(&polished) - norm(&theta)).abs() < 1e-3,
        "scale: {} vs {}",
        norm(&polished),
        norm(&theta)
    );
    assert!(
        (beta_polished - beta).abs() < 1e-3,
        "intercept: {beta_polished} vs {beta}"
    );
}
