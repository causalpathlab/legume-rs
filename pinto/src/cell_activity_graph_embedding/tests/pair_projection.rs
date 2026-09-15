//! The pair projection solves a known problem: with the dictionary frozen and the
//! counts generated from a known `e_uv`, the MAP is that `e_uv`. These tests
//! generate exactly that and check the exact solver lands on it — from the
//! origin and from a warm start — plus the two properties the design rests on:
//! `β_uv` absorbs pooled depth, and a pair with no counts stays at the origin
//! rather than being handed a fabricated direction.

use super::fixture::*;
use crate::cell_activity_graph_embedding::pair_projection::PairDictionary;

/// Deliberately near-zero: these tests check that the *likelihood* recovers
/// the truth, and a working ridge would bias the norm down.
const RIDGE: f32 = 1e-4;

#[test]
fn projection_recovers_known_pair_embedding() {
    let e = dictionary_matrix();
    let (b, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");
    assert_eq!(dict.n_active(), N_GENES);

    let truth = [0.6f32, -0.4, 0.25, 0.0];
    let beta_truth = 2.0f32.ln();
    let obs = counts_from(&e, &b, &truth, beta_truth);

    let (theta, beta, gap) = dict.solve(&obs, RIDGE);

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
    // The certificate is `‖∇‖²/(2λ)`, so a near-zero ridge inflates it; small
    // against the pair's thousands of nats of likelihood is what settled means.
    assert!(gap < 1.0, "{gap} nats above the optimum at the optimum");
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

    let (theta_shallow, beta_shallow, _) = dict.solve(&shallow, RIDGE);
    let (theta_deep, beta_deep, _) = dict.solve(&deep, RIDGE);

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

    let (theta, beta, gap) = dict.solve(&[], RIDGE);
    assert_eq!(theta, vec![0.0; DIM]);
    assert_eq!(beta, 0.0);
    assert_eq!(gap, 0.0);

    // A gene that carries no counts anywhere is not on the partition axis, so a
    // profile made only of such genes is empty too — not a direction.
    let mut totals_with_dead = totals.clone();
    totals_with_dead[0] = 0.0;
    let dict = PairDictionary::new(&e, &totals_with_dead, N_CELLS).expect("dictionary");
    assert_eq!(dict.n_active(), N_GENES - 1);
    let (theta, _, _) = dict.solve(&[(0, 12.0)], RIDGE);
    assert_eq!(theta, vec![0.0; DIM]);
}

//////////////////////////////////////
// Newton from a warm or wild start //
//////////////////////////////////////

#[test]
fn newton_polish_lands_where_the_solve_from_the_origin_lands() {
    let e = dictionary_matrix();
    let (b, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");
    let truth = [0.6f32, -0.4, 0.25, 0.0];
    let obs = counts_from(&e, &b, &truth, 0.3);
    let (theta, beta, _) = dict.solve(&obs, RIDGE);
    // From a start well off the optimum.
    let start = [0.1f32, 0.1, -0.1, 0.2];
    let (polished, beta_polished, gap) = dict.polish(&obs, RIDGE, &start, 8);
    assert!(gap < 1.0, "{gap} nats above the optimum after polishing");
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

#[test]
fn a_wild_start_is_walked_back_to_the_same_optimum() {
    // Far outside the quadratic regime, in a direction the counts contradict:
    // every Newton step from here is long, so each is line-searched, and the
    // solve must still end where the solve from the origin ends.
    let e = dictionary_matrix();
    let (b, totals) = abundances();
    let dict = PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary");
    let truth = [0.6f32, -0.4, 0.25, 0.0];
    let obs = counts_from(&e, &b, &truth, 0.0);
    let (theta, beta, _) = dict.solve(&obs, RIDGE);
    let wild = [-12.0f32, 9.0, -7.0, 11.0];
    let (back, beta_back, gap) = dict.polish(&obs, RIDGE, &wild, 64);
    assert!(
        gap < 1.0,
        "{gap} nats above the optimum: the wild start did not settle"
    );
    assert!(
        cosine(&back, &theta) > 0.9999,
        "direction: {back:?} vs {theta:?}"
    );
    assert!(
        (norm(&back) - norm(&theta)).abs() < 1e-3,
        "scale: {} vs {}",
        norm(&back),
        norm(&theta)
    );
    assert!(
        (beta_back - beta).abs() < 1e-3,
        "intercept: {beta_back} vs {beta}"
    );
}
