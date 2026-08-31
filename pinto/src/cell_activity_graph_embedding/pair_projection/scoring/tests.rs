//! The scorer's contract: the null is the abundance multinomial, the model
//! reduces to exactly that null when the latent is off, and a latent aligned
//! with a gene has to earn its likelihood on a profile concentrated there.

use super::*;
use crate::util::common::Mat;

/// Three genes, two latent dimensions. Gene 0 loads on dim 0, gene 1 on dim 1,
/// gene 2 on neither — so a θ can be pointed at one gene at a time.
fn fixture() -> PairDictionary {
    let e_feat = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let totals = vec![300.0, 200.0, 500.0];
    PairDictionary::new(&e_feat, &totals, 100).expect("dictionary")
}

#[test]
fn a_zero_latent_scores_exactly_the_abundance_null() {
    // At θ = 0 the log-rate IS b, and the partition is Σ exp(b_g), which is what
    // `log_z` holds. The two columns must therefore agree — if they ever drift,
    // the null and the model are normalising over different axes and the
    // reported gain is meaningless.
    let dict = fixture();
    let obs = [(0u32, 7.0f32), (2, 3.0)];
    let s = dict.score(&obs, &[0.0, 0.0], &dict.eval_axis(None));
    assert!(
        (s.llik - s.null_llik).abs() < 1e-4,
        "llik {} vs null {}",
        s.llik,
        s.null_llik
    );
    assert_eq!(s.total, 10.0);
}

#[test]
fn a_latent_aimed_at_the_observed_gene_beats_the_null() {
    let dict = fixture();
    // Everything observed on gene 0, and θ points at gene 0's dimension.
    let obs = [(0u32, 10.0f32)];
    let good = dict.score(&obs, &[2.0, 0.0], &dict.eval_axis(None));
    let bad = dict.score(&obs, &[-2.0, 0.0], &dict.eval_axis(None));
    assert!(good.llik > good.null_llik, "aligned latent must gain");
    assert!(bad.llik < bad.null_llik, "opposed latent must lose");
}

#[test]
fn the_likelihood_is_a_proper_multinomial() {
    // Scores are log-probabilities over the active axis, so every one is ≤ 0.
    let dict = fixture();
    let obs = [(0u32, 4.0f32), (1, 6.0)];
    for theta in [[0.0, 0.0], [3.0, -1.0], [-5.0, 5.0]] {
        let s = dict.score(&obs, &theta, &dict.eval_axis(None));
        assert!(s.llik <= 0.0, "llik {} must be ≤ 0", s.llik);
        assert!(s.null_llik <= 0.0);
    }
}

#[test]
fn an_empty_profile_scores_nothing_rather_than_nan_likelihood() {
    let dict = fixture();
    let s = dict.score(&[], &[1.0, 1.0], &dict.eval_axis(None));
    assert_eq!(s.total, 0.0);
    assert_eq!(s.llik, 0.0);
    assert!(s.agreement.spearman.is_nan());
}

#[test]
fn agreement_needs_an_evaluation_axis() {
    let dict = fixture();
    let obs = [(0u32, 9.0f32), (1, 1.0)];
    assert!(dict
        .score(&obs, &[1.0, 0.0], &dict.eval_axis(None))
        .agreement
        .spearman
        .is_nan());

    // Given one, the correlation is over that axis with the zeros densified in.
    let axis: Vec<u32> = vec![0, 1, 2];
    let s = dict.score(&obs, &[3.0, -3.0], &dict.eval_axis(Some(axis.clone())));
    assert!(
        s.agreement.spearman.is_finite(),
        "an eval axis must yield a real correlation"
    );
}

#[test]
fn eval_axis_drops_names_that_carry_no_counts() {
    // Gene 3 has zero total, so it never enters the active list and cannot be
    // scored — naming it must not shift the other genes' positions.
    let e_feat = Mat::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    let dict = PairDictionary::new(&e_feat, &[300.0, 200.0, 500.0, 0.0], 100).expect("dictionary");
    let names: Vec<Box<str>> = vec!["a".into(), "b".into(), "c".into(), "dead".into()];
    // Deliberately cased differently from the data: the panel and the data may
    // disagree on case while naming the same genes.
    let wanted: Vec<Box<str>> = vec!["A".into(), "DEAD".into()];
    assert_eq!(dict.eval_positions(&names, &wanted), vec![0]);
}

#[test]
fn an_eval_axis_restricts_the_likelihood_and_the_total() {
    // Comparability with senna depends on this: given `--eval-features`, the
    // likelihood is the CONDITIONAL multinomial over the scored genes, and counts
    // outside that set are in neither the numerator nor the denominator.
    //
    // The expected value is the two-gene conditional worked out by hand from the
    // fixture's b = ln(total / n_cells) and theta = (1, -1):
    //   logit = (1 + ln3, -1 + ln2), z = ln(e^logit0 + e^logit1)
    //   llik  = 4(logit0 - z) + 6(logit1 - z), over 10 counts.
    let dict = fixture();
    let obs = [(0u32, 4.0f32), (1, 6.0), (2, 90.0)];
    let axis: Vec<u32> = vec![0, 1];

    let s = dict.score(&obs, &[1.0, -1.0], &dict.eval_axis(Some(axis.clone())));
    assert_eq!(s.total, 10.0, "gene 2's counts are outside the scored set");
    assert!(
        (s.llik / s.total - -1.529_661_8).abs() < 1e-4,
        "got {}",
        s.llik / s.total
    );
}

#[test]
fn the_restricted_score_ignores_everything_off_its_axis() {
    // What "restricted" has to mean: nothing outside the scored genes may move
    // the number. Without this the correlations and the likelihood would drift
    // with a gene the user deliberately excluded, and two methods carrying
    // different unscored genes would not be comparable after all.
    let dict = fixture();
    let axis: Vec<u32> = vec![0, 1];
    let theta = [0.6f32, -0.2];

    let ax = dict.eval_axis(Some(axis.clone()));
    let a = dict.score(&[(0, 4.0), (1, 6.0), (2, 1.0)], &theta, &ax);
    let b = dict.score(&[(0, 4.0), (1, 6.0), (2, 9_000.0)], &theta, &ax);
    let c = dict.score(&[(0, 4.0), (1, 6.0)], &theta, &ax);

    for other in [&b, &c] {
        assert_eq!(a.total, other.total);
        assert!((a.llik - other.llik).abs() < 1e-4);
        assert!((a.null_llik - other.null_llik).abs() < 1e-4);
        assert!((a.agreement.spearman - other.agreement.spearman).abs() < 1e-6);
    }
}

#[test]
fn the_restricted_null_normalises_over_the_same_genes_as_the_model() {
    // If model and null used different partitions their difference would be an
    // artifact of that mismatch rather than of the latent.
    let dict = fixture();
    let obs = [(0u32, 5.0f32), (1, 5.0)];
    let axis: Vec<u32> = vec![0, 1];
    let s = dict.score(&obs, &[0.0, 0.0], &dict.eval_axis(Some(axis.clone())));
    assert!(
        (s.llik - s.null_llik).abs() < 1e-4,
        "at theta = 0 the model IS the null, on any axis: {} vs {}",
        s.llik,
        s.null_llik
    );
}

/// A model that puts ~no mass on an observed gene must be charged the same
/// penalty in both engines. senna floors the probability at LOG_PROB_FLOOR
/// nats; before this test, pinto's logit clamp let the charge run to roughly
/// twice that, so `eval_llik_per_count` — the documented cross-engine ranking
/// column — punished the same event differently depending on the binary.
#[test]
fn a_starved_gene_is_charged_the_shared_floor() {
    let dict = fixture();
    // theta drives gene 0's logit to the clamp floor while gene 1 takes all the
    // mass; every observed count sits on the starved gene.
    let s = dict.score(&[(0u32, 10.0f32)], &[-100.0, 100.0], &dict.eval_axis(None));
    let per_count = f64::from(s.llik) / f64::from(s.total);
    assert!(
        (per_count - matrix_util::agreement::LOG_PROB_FLOOR).abs() < 1e-3,
        "starved-gene charge {per_count} nats/count; the shared floor is {}",
        matrix_util::agreement::LOG_PROB_FLOOR
    );
}
