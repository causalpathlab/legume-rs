//! `masked-vae` stores a raw Gaussian `z`, not `log θ`. Consumers must convert
//! through the head rather than assuming `exp()`.
//!
//! Regression for the contract gap where every masked head wrote
//! `kind: "itopic"`, so downstream applied the log-θ reading to masked-vae and
//! produced unnormalized "proportions".

use candle_util::vae::masked_topic::LatentHead;
use senna::embed_common::{latent_to_theta, softmax_rows_inplace, Mat};
use senna::run_manifest::{CellSpace, RunKind};

fn rows_sum_to_one(m: &Mat) {
    for (i, row) in m.row_iter().enumerate() {
        let s: f32 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "row {i} sums to {s}, expected 1.0 — not on the simplex"
        );
    }
}

#[test]
fn gaussian_latent_becomes_proportions_via_softmax() {
    // Raw `z` with positive entries — `exp` alone would give values > 1 and
    // rows summing to far more than 1.
    let z = Mat::from_row_slice(2, 3, &[2.0, 0.0, -1.0, 8.0, 8.0, -8.0]);

    let naive_sum: f32 = z.row(1).iter().map(|x| x.exp()).sum();
    assert!(
        naive_sum > 100.0,
        "test premise: bare exp(z) should be wildly off-simplex, got {naive_sum}"
    );

    let theta = latent_to_theta(&z, LatentHead::Gaussian);
    rows_sum_to_one(&theta);
    assert!(theta.iter().all(|&x| (0.0..=1.0).contains(&x)));
    // Order is preserved: the largest logit keeps the largest share.
    assert!(theta[(0, 0)] > theta[(0, 1)] && theta[(0, 1)] > theta[(0, 2)]);
}

#[test]
fn simplex_latents_are_exponentiated_not_renormalized() {
    // A genuine log-softmax row: exp already sums to 1.
    let third = (1.0f32 / 3.0).ln();
    let log_theta = Mat::from_row_slice(1, 3, &[third, third, third]);
    for head in [LatentHead::Softmax, LatentHead::StickBreaking] {
        let theta = latent_to_theta(&log_theta, head);
        rows_sum_to_one(&theta);
        assert!((theta[(0, 0)] - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn softmax_rows_normalizes_recovers_theta_and_zeros_degenerate() {
    // Gaussian-style row: softmax → simplex, order preserved.
    let mut g = Mat::from_row_slice(1, 3, &[2.0, 0.0, -1.0]);
    softmax_rows_inplace(&mut g);
    rows_sum_to_one(&g);
    assert!(g[(0, 0)] > g[(0, 1)] && g[(0, 1)] > g[(0, 2)]);

    // log θ row: exp recovers θ, renorm is a no-op → 1/3 each.
    let third = (1.0f32 / 3.0).ln();
    let mut lt = Mat::from_row_slice(1, 3, &[third, third, third]);
    softmax_rows_inplace(&mut lt);
    assert!(lt.iter().all(|&x| (x - 1.0 / 3.0).abs() < 1e-6));

    // Degenerate all-`-inf` row → zeroed, not left holding NaN.
    let mut deg = Mat::from_row_slice(1, 2, &[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    softmax_rows_inplace(&mut deg);
    assert!(
        deg.iter().all(|&x| x == 0.0),
        "all -inf row must zero: {deg:?}"
    );

    // NaN-poisoned row → zeroed, NaN not propagated.
    let mut nan = Mat::from_row_slice(1, 3, &[1.0, f32::NAN, 2.0]);
    softmax_rows_inplace(&mut nan);
    assert!(
        nan.iter().all(|&x| x == 0.0),
        "NaN row must be zeroed: {nan:?}"
    );
}

#[test]
fn masked_vae_is_topic_family_but_not_log_simplex() {
    assert!(RunKind::MaskedVae.is_topic_family(), "its β is a simplex");
    assert!(
        !RunKind::MaskedVae.latent_is_log_simplex(),
        "its latent is a raw Gaussian z — this is the whole point of the kind"
    );
    for k in [RunKind::Topic, RunKind::Itopic, RunKind::JointTopic] {
        assert!(k.latent_is_log_simplex(), "{k} should store log θ");
    }
    for k in [RunKind::Svd, RunKind::JointSvd, RunKind::Bge, RunKind::Fne] {
        assert!(!k.latent_is_log_simplex(), "{k} should not store log θ");
    }
}

#[test]
fn masked_vae_kind_round_trips_through_json() {
    let json = serde_json::to_string(&RunKind::MaskedVae).unwrap();
    assert_eq!(json, "\"masked-vae\"");
    let back: RunKind = serde_json::from_str(&json).unwrap();
    assert_eq!(back, RunKind::MaskedVae);
    // Existing manifests must keep parsing.
    let legacy: RunKind = serde_json::from_str("\"itopic\"").unwrap();
    assert_eq!(legacy, RunKind::Itopic);
}

/// Every kind's cell space, pinned.
///
/// `cell_space` is the single declaration five geometry decisions now read —
/// whether to `exp()`, cosine vs z-scored Euclidean kNN, raw-embedding layout,
/// direct-cells layout. Those used to be `matches!(kind, Bge | Fne)` and `_ =>`
/// at five sites, so a new variant compiled and silently took the fallback at
/// every one. The `match` in `cell_space` makes that a compile error; this test
/// makes the ANSWER a test failure rather than a silent reclassification.
#[test]
fn every_kind_declares_its_cell_space() {
    for k in [
        RunKind::Bge,
        RunKind::Fne,
        RunKind::ResolveEmbeddingSpace,
        RunKind::Gem,
        RunKind::GemEncoder,
    ] {
        assert_eq!(
            k.cell_space(),
            CellSpace::Embedding,
            "{k}'s geometry table is a Euclidean embedding, so it wants angular kNN"
        );
    }
    for k in [RunKind::Topic, RunKind::Itopic, RunKind::JointTopic] {
        assert_eq!(
            k.cell_space(),
            CellSpace::LogSimplex,
            "{k} lays out on log θ"
        );
    }
    for k in [
        RunKind::MaskedVae,
        RunKind::Vae,
        RunKind::Svd,
        RunKind::JointSvd,
    ] {
        assert_eq!(k.cell_space(), CellSpace::Signed, "{k} is signed scores");
    }
}

/// The gem kinds, whose absence from these predicates was the original defect.
///
/// `gem` writes a Euclidean `cell_embedding` and no latent at all, so `exp()`-ing
/// it is meaningless. `gem-encoder` is the first kind that is log-simplex in the
/// LATENT sense while its GEOMETRY table is Euclidean — the two predicates stopped
/// nesting there, which is exactly what a consumer reading one and assuming the
/// other gets wrong.
#[test]
fn gem_kinds_are_classified_on_both_axes() {
    assert!(
        !RunKind::Gem.latent_is_log_simplex(),
        "gem writes no latent; exp() of its cell embedding is meaningless"
    );
    assert!(!RunKind::Gem.is_topic_family(), "gem has no topic axis");

    assert!(
        RunKind::GemEncoder.latent_is_log_simplex(),
        "gem-encoder's latent.parquet IS log θ"
    );
    assert!(
        RunKind::GemEncoder.is_topic_family(),
        "its log_softmax-over-genes dictionary is a simplex β"
    );
    assert_eq!(
        RunKind::GemEncoder.cell_space(),
        CellSpace::Embedding,
        "but geometry_latent hands back cell_embedding (θ·α), NOT the simplex — \
         this is the pair that stopped nesting"
    );
}

/// The wire strings are a compatibility surface: renaming a variant without
/// changing them orphans every manifest already on disk.
#[test]
fn gem_kinds_round_trip_through_json() {
    for (k, wire) in [(RunKind::Gem, "gem"), (RunKind::GemEncoder, "gem-encoder")] {
        assert_eq!(k.as_str(), wire);
        let json = serde_json::to_string(&k).unwrap();
        assert!(
            json.contains(wire),
            "{k} serializes as {json}, expected {wire}"
        );
        let back: RunKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, k);
    }
}
