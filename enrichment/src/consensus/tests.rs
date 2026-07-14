use super::*;

/// Build a `[n × k]` row-major tally from per-item vote counts.
fn tally(rows: &[Vec<f32>]) -> (Vec<f32>, usize, usize) {
    let n = rows.len();
    let k = rows[0].len();
    let mut post = Vec::with_capacity(n * k);
    for r in rows {
        assert_eq!(r.len(), k);
        post.extend_from_slice(r);
    }
    (post, n, k)
}

fn cfg(abstain: Abstain) -> AbstainConfig {
    AbstainConfig {
        abstain,
        set_coverage: 0.8,
        max_set_size: 3,
    }
}

#[test]
fn the_sign_test_is_the_binomial_upper_tail() {
    // P(X >= k) for X ~ Binomial(m, 1/2). Exact, hand-checkable values.
    assert!((binom_half_upper_tail(0, 0) - 1.0).abs() < 1e-12);
    // m=1: P(X>=1) = 1/2
    assert!((binom_half_upper_tail(1, 1) - 0.5).abs() < 1e-12);
    // m=2: P(X>=2) = 1/4
    assert!((binom_half_upper_tail(2, 2) - 0.25).abs() < 1e-12);
    // m=4: P(X>=3) = (4 + 1)/16 = 5/16
    assert!((binom_half_upper_tail(4, 3) - 5.0 / 16.0).abs() < 1e-12);
    // The whole tail is 1.
    assert!((binom_half_upper_tail(10, 0) - 1.0).abs() < 1e-12);
}

#[test]
fn support_is_not_scale_free_but_separable_is() {
    // The core argument for `Separable`. A fixed support bar means a DIFFERENT test on panels of
    // different size, because chance agreement is 1/C — but the sign test does not move with C.
    let bar = Abstain::Support(0.5);
    // 0.5 is exactly chance on a 2-type panel and 12x chance on a 24-type one; the same flag
    // accepts both.
    assert!(bar.allows(0.5, 0.5, 100));

    let sep = Abstain::Separable(0.05);
    // It refuses a 50/50 split no matter how many replicates: the leaders are indistinguishable.
    assert!(!sep.allows(0.5, 0.5, 100));
    assert!(sep.allows(0.9, 0.1, 100));

    // …and it asks for EVIDENCE, not just a ratio. 3-vs-1 out of 4 replicates is
    // P(X>=3 | m=4) = 5/16 = 0.31 — not significant. The same ratio out of 40 is overwhelming.
    assert!(!sep.allows(0.75, 0.25, 4));
    assert!(sep.allows(0.75, 0.25, 40));
}

#[test]
fn the_credible_set_says_what_the_replicates_said() {
    // Two leaders splitting 0.5/0.35 reach 0.8 coverage together, and neither alone does.
    let row = [0.5f32, 0.35, 0.1, 0.05];
    let set = credible_set(&row, 0.8, 3).expect("two labels cover 0.85");
    assert_eq!(set, vec![0, 1]);

    // One dominant label needs no company.
    let row = [0.9f32, 0.05, 0.05];
    assert_eq!(credible_set(&row, 0.8, 3).expect("one label"), vec![0]);
}

#[test]
fn a_set_too_wide_to_mean_anything_is_no_call() {
    // A flat 5-way split cannot reach 0.8 within 3 labels — that is not an annotation.
    let row = [0.2f32; 5];
    assert!(credible_set(&row, 0.8, 3).is_none());
}

#[test]
fn unassigned_mass_makes_coverage_harder_to_reach() {
    // `credible_set` is handed only the TYPE columns, but the shares stay shares of ALL the
    // replicates — so replicates that declined to call keep sitting in the denominator.
    // Types get 0.3/0.3, and `unassigned` took the other 0.4.
    let row = [0.3f32, 0.3, 0.0, 0.4]; // last column is `unassigned`
    let c = 3;
    // Over the type columns alone, the best two only reach 0.6 < 0.8 → no set.
    assert!(credible_set(&row[..c], 0.8, 3).is_none());
}

#[test]
fn a_declined_item_does_not_become_a_label() {
    // One item; `unassigned` (last column) wins outright. It must not be called.
    let (post, n, k) = tally(&[vec![10.0, 20.0, 70.0]]); // c = 2 types, col 2 = unassigned
    let con = summarize_consensus(post, n, k, 100, &cfg(Abstain::Support(0.5)));
    assert_eq!(con.label[0], UNASSIGNED);
    assert!(
        (con.support[0] - 0.7).abs() < 1e-6,
        "support is the max share"
    );
    // And it gets no set either: the types only reach 0.3.
    assert!(con.label_set[0].is_empty());
}

#[test]
fn a_clean_winner_is_called_with_its_support() {
    let (post, n, k) = tally(&[vec![95.0, 3.0, 2.0]]); // types 0,1; col 2 unassigned
    let con = summarize_consensus(post, n, k, 100, &cfg(Abstain::Support(0.5)));
    assert_eq!(con.label[0], 0);
    assert!((con.support[0] - 0.95).abs() < 1e-6);
    assert_eq!(con.label_set[0], vec![0]);
    assert!((con.set_support[0] - 0.95).abs() < 1e-6);
    // Nearly all the mass on one label ⇒ low normalized entropy (0.21 here; a flat 3-way split
    // would be 1.0).
    assert!(con.entropy[0] < 0.25, "entropy was {}", con.entropy[0]);
}

#[test]
fn an_inseparable_pair_abstains_but_still_gets_a_set() {
    // 45/45/10 — the two leaders are a coin flip, so `Separable` refuses to pick…
    let (post, n, k) = tally(&[vec![45.0, 45.0, 10.0]]);
    let con = summarize_consensus(post, n, k, 100, &cfg(Abstain::Separable(0.05)));
    assert_eq!(con.label[0], UNASSIGNED, "a 45/45 split is not separable");
    // …but the honest answer is the pair, not a shrug: they cover 0.9 together.
    assert_eq!(con.label_set[0], vec![0, 1]);
    assert!((con.set_support[0] - 0.9).abs() < 1e-6);
}

#[test]
fn support_is_normalized_by_replicates_that_completed_not_requested() {
    // 30 of 200 replicates finished; the item won 24 of the 30 that ran. The support is 24/30,
    // NOT 24/200 — an interrupted bootstrap is a smaller bootstrap, not a broken one.
    let (post, n, k) = tally(&[vec![24.0, 6.0, 0.0]]);
    let con = summarize_consensus(post, n, k, 30, &cfg(Abstain::Support(0.5)));
    assert!(
        (con.support[0] - 0.8).abs() < 1e-6,
        "got {}",
        con.support[0]
    );
    assert_eq!(con.label[0], 0);
}

#[test]
fn label_support_includes_the_unassigned_column() {
    // The support is the max over EVERYTHING, `unassigned` included — a run whose replicates
    // agreed to decline is a confident decline. (The support null and the observed run had
    // drifted apart on exactly this.)
    let row = [0.2f32, 0.1, 0.7]; // last is `unassigned`
    assert!((label_support(&row) - 0.7).abs() < 1e-6);
}

#[test]
fn keyed_rng_depends_only_on_the_key() {
    use rand::RngExt;
    let draw = |seed: u64, d: usize, item: u64| -> u64 {
        let mut r = keyed_rng(seed, d, item);
        r.random::<u64>()
    };
    // Same key ⇒ same stream, regardless of when or on which thread it is built.
    assert_eq!(draw(42, 7, 3), draw(42, 7, 3));
    // Every coordinate of the key separates the stream.
    assert_ne!(draw(42, 7, 3), draw(43, 7, 3));
    assert_ne!(draw(42, 7, 3), draw(42, 8, 3));
    assert_ne!(draw(42, 7, 3), draw(42, 7, 4));
}
