//! Unit tests for the pure transforms enrichment mode applies to the
//! gem-encoder tables before handing them to `enrichment::annotate`: the
//! gene-axis reconciliation between the dictionary and the pseudobulk profile,
//! and the two log → simplex conversions (θ and β).
//!
//! All three are **silent** when wrong, which is the whole reason they are
//! tested rather than eyeballed. A mis-ordered gene axis pairs gene *i*'s β with
//! gene *j*'s pseudobulk profile and every score downstream still looks
//! well-formed; a θ read on the wrong scale produces a plausible annotation
//! rather than an error; and a dictionary left on the log scale clamps to zero
//! inside `SpecificityMode::Simplex`, making every gene ranking an arbitrary tie
//! with nothing raised anywhere.

use super::*;

fn names(xs: &[&str]) -> Vec<Box<str>> {
    xs.iter().map(|s| Box::from(*s)).collect()
}

//////////////////////////////
// gene-axis reconciliation //
//////////////////////////////

#[test]
fn strip_track_suffix_normalizes_both_key_styles() {
    // pb_gene rows carry the gem feature suffix, dictionary rows do not.
    assert_eq!(
        strip_track_suffix("ENSG1_CFH/count/spliced", Track::Mature).as_ref(),
        "ENSG1_CFH"
    );
    assert_eq!(
        strip_track_suffix("ENSG1_CFH", Track::Mature).as_ref(),
        "ENSG1_CFH"
    );
    // Only the pass's OWN suffix is stripped. An unspliced row is a different
    // axis and must not silently collide with its spliced twin — which is why
    // the stripper is scoped to a track rather than trying both.
    assert_eq!(
        strip_track_suffix("ENSG1_CFH/count/unspliced", Track::Mature).as_ref(),
        "ENSG1_CFH/count/unspliced"
    );
    assert_eq!(
        strip_track_suffix("ENSG1_CFH/count/unspliced", Track::Nascent).as_ref(),
        "ENSG1_CFH"
    );
    assert_eq!(
        strip_track_suffix("ENSG1_CFH/count/spliced", Track::Nascent).as_ref(),
        "ENSG1_CFH/count/spliced"
    );
}

#[test]
fn align_gene_axis_permutes_pb_rows_onto_the_dictionary_order() {
    let dict = names(&["A", "B", "C"]);
    let pb = names(&["C", "A", "B"]);
    let order = align_gene_axis(&dict, &pb, "d.parquet", "p.parquet").unwrap();
    assert_eq!(order, vec![1, 2, 0]);

    // Applying it really does put pb_gene on the dictionary's axis: row values
    // encode their source gene, so a positional zip would fail this.
    let pb_mat = Mat::from_row_slice(3, 1, &[3.0, 1.0, 2.0]); // C, A, B
    let aligned = pb_mat.select_rows(&order);
    assert_eq!(
        aligned.column(0).iter().copied().collect::<Vec<f32>>(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn align_gene_axis_is_identity_when_the_axes_already_agree() {
    let dict = names(&["A", "B", "C"]);
    let order = align_gene_axis(&dict, &dict, "d.parquet", "p.parquet").unwrap();
    assert_eq!(order, vec![0, 1, 2]);
}

#[test]
fn align_gene_axis_bails_when_a_dictionary_gene_has_no_pseudobulk_row() {
    let dict = names(&["A", "B", "C"]);
    let pb = names(&["A", "C"]);
    let err = align_gene_axis(&dict, &pb, "d.parquet", "p.parquet")
        .expect_err("a missing gene must not be silently dropped");
    let msg = err.to_string();
    assert!(
        msg.contains('B'),
        "the error should name the missing gene: {msg}"
    );
    assert!(
        msg.contains("d.parquet") && msg.contains("p.parquet"),
        "{msg}"
    );
}

#[test]
fn align_gene_axis_tolerates_a_wider_pseudobulk_axis() {
    // Extra pb rows are fine — they are simply not selected. Only genes the
    // dictionary asks for and pb cannot supply are fatal.
    let dict = names(&["B", "A"]);
    let pb = names(&["A", "X", "B", "Y"]);
    let order = align_gene_axis(&dict, &pb, "d.parquet", "p.parquet").unwrap();
    assert_eq!(order, vec![2, 0]);
}

///////////////
// log θ → θ //
///////////////

/// Row-normalize a `[·, K]` matrix — the softmax `exp_log_theta` deliberately
/// does NOT apply, kept here so the tests can measure against it.
fn softmax_rows(log_theta: &Mat) -> Mat {
    let mut out = log_theta.map(f32::exp);
    for r in 0..out.nrows() {
        let denom: f32 = out.row(r).iter().sum();
        if denom > 0.0 {
            for c in 0..out.ncols() {
                out[(r, c)] /= denom;
            }
        }
    }
    out
}

#[test]
fn exp_log_theta_recovers_the_simplex_exactly() {
    // Two cells on the simplex, stored as log θ (the `latent.parquet` contract).
    let theta = [0.2f32, 0.3, 0.5, 0.7, 0.2, 0.1];
    let log_theta = Mat::from_row_slice(2, 3, &theta.map(f32::ln));

    let got = exp_log_theta(
        &log_theta,
        "latent.parquet",
        Some(manifest::Latent::LogTheta),
    );
    for (i, want) in theta.iter().enumerate() {
        let (r, c) = (i / 3, i % 3);
        assert!(
            (got[(r, c)] - want).abs() < 1e-6,
            "exp(log theta)[{r},{c}] = {} but theta = {want}",
            got[(r, c)]
        );
    }
    for r in 0..2 {
        let sum: f32 = got.row(r).iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row {r} sums to {sum}, not 1");
    }

    // On a row that honours the contract a softmax agrees exactly — which is
    // WHY it must not be used: it cannot help here, it can only hide the cases
    // below.
    let soft = softmax_rows(&log_theta);
    assert!((soft - &got).amax() < 1e-6);
}

#[test]
fn exp_log_theta_leaves_a_broken_contract_visible_where_softmax_would_hide_it() {
    // (a) raw logits — what runs before 2026-07-21 stored under the same shape.
    let logits = Mat::from_row_slice(1, 3, &[2.0f32, -1.0, 0.5]);
    let got = exp_log_theta(&logits, "latent.parquet", None);
    let sum: f32 = got.row(0).iter().sum();
    assert!(
        (sum - 1.0).abs() > 0.5,
        "raw logits must NOT come back looking like a distribution (sum {sum})"
    );
    // A softmax would have returned a perfectly plausible θ instead.
    let soft = softmax_rows(&logits);
    assert!((soft.row(0).iter().sum::<f32>() - 1.0).abs() < 1e-5);

    // (b) a pseudobulk that observed no cells: `write_pseudobulk_tables` leaves
    // its row all-zero, so exp() is all-ones and sums to K, not 1.
    let empty_pb = Mat::zeros(1, 4);
    let got = exp_log_theta(&empty_pb, "pb_latent.parquet", None);
    assert!((got.row(0).iter().sum::<f32>() - 4.0).abs() < 1e-5);
}

#[test]
fn exp_log_theta_leaves_an_empty_table_alone() {
    let empty = Mat::zeros(0, 4);
    assert_eq!(exp_log_theta(&empty, "latent.parquet", None).nrows(), 0);
}

/// The stamped contract steers DIAGNOSTICS ONLY — it must never change a number.
///
/// This is the guard on reading the manifest at all: the point of consulting it
/// was to stop telling the user to go open a file the code could read itself,
/// not to give the manifest a vote on what θ is. If a stamp could alter the
/// values, a manifest copied from the wrong run would silently rewrite the
/// annotation instead of merely mis-describing it.
#[test]
fn the_stated_contract_never_changes_the_values() {
    let logits = Mat::from_row_slice(2, 3, &[2.0f32, -1.0, 0.5, 0.1, 0.2, 0.3]);
    let baseline = exp_log_theta(&logits, "latent.parquet", None);
    for stated in [
        Some(manifest::Latent::LogTheta),
        Some(manifest::Latent::Embedding),
    ] {
        let got = exp_log_theta(&logits, "latent.parquet", stated);
        assert!(
            (&got - &baseline).amax() < 1e-9,
            "{stated:?} changed the values; it may only change what is reported"
        );
    }
}

///////////////
// log β → β //
///////////////

#[test]
fn exp_log_beta_gives_simplex_columns_which_specificity_needs() {
    // `dictionary.parquet` is log_softmax over GENES, so columns (not rows) are
    // the distributions and every entry is <= 0.
    let beta = [0.6f32, 0.1, 0.3, 0.6, 0.1, 0.3]; // G=3, K=2, column-major values
    let log_beta = Mat::from_column_slice(3, 2, &beta.map(f32::ln));

    let got = exp_log_beta(&log_beta, "dictionary.parquet");
    for j in 0..2 {
        let sum: f32 = got.column(j).iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "column {j} sums to {sum}, not 1");
    }

    // The trap this conversion exists for: `SpecificityMode::Simplex` clamps at
    // max(0.0), so the file AS WRITTEN zeroes out completely and every factor's
    // gene ranking becomes an arbitrary tie — with no error raised anywhere.
    let raw = enrichment::compute_specificity(&log_beta, enrichment::SpecificityMode::Simplex);
    assert_eq!(
        raw.iter().copied().fold(0.0f32, f32::max),
        0.0,
        "log beta must clamp to an all-zero specificity — that is the failure being guarded"
    );
    let exped = enrichment::compute_specificity(&got, enrichment::SpecificityMode::Simplex);
    assert!(
        exped.iter().copied().fold(0.0f32, f32::max) > 0.0,
        "exp(log beta) must produce a usable specificity"
    );
}

//////////////////////
// per-track tables //
//////////////////////

/// Each track must name a COMPLETE, self-consistent set: the null recomputes
/// `β̃ = pb_gene · pb_membership[π]` and compares it against `profile_gk`, so a
/// pb_gene from one track paired with a dictionary from the other would
/// reconstruct the wrong object and calibrate the ranking against it — silently,
/// since both files have identical shape.
#[test]
fn each_track_names_its_own_dictionary_latent_and_pseudobulk() {
    let (d_m, l_m, p_m) = track_tables(Track::Mature);
    let (d_n, l_n, p_n) = track_tables(Track::Nascent);

    assert_eq!(
        (d_m, l_m, p_m),
        ("dictionary.parquet", "latent.parquet", "pb_gene.parquet")
    );
    assert_eq!(
        (d_n, l_n, p_n),
        (
            "dictionary_nascent.parquet",
            "latent_nascent.parquet",
            "pb_gene_nascent.parquet"
        )
    );
    // No file is shared between the tracks. `pb_latent` is the one table both
    // read, and it is deliberately NOT in this list: the model has a single θ,
    // so the pseudobulk membership is track-independent by construction.
    for (a, b) in [(d_m, d_n), (l_m, l_n), (p_m, p_n)] {
        assert_ne!(a, b, "{a} must not serve both tracks");
    }
}

/// Mature keeps the bare tag so existing output paths and downstream readers are
/// untouched; nascent takes a suffix so `--track both` leaves two complete
/// result sets under one prefix instead of one overwriting the other.
#[test]
fn the_two_tracks_write_to_different_output_tags() {
    assert_eq!(track_tag(Track::Mature), "enrichment");
    assert_eq!(track_tag(Track::Nascent), "enrichment.nascent");
    assert_ne!(track_tag(Track::Mature), track_tag(Track::Nascent));
}

/// The gene axis has to reconcile ACROSS tracks: `pb_gene_nascent` rows carry
/// the unspliced suffix while `dictionary_nascent` rows are bare, and
/// `align_gene_axis` matches them by name. If the stripper only knew the mature
/// suffix, every nascent gene would fail to match and the nascent pass would
/// refuse to run.
#[test]
fn strip_track_suffix_normalizes_the_nascent_axis_too() {
    // A nascent pb_gene reorders onto the bare-keyed nascent dictionary.
    let dict = names(&["A", "B"]);
    let pb = names(&["B/count/unspliced", "A/count/unspliced"]);
    let pb_keys: Vec<Box<str>> = pb
        .iter()
        .map(|r| strip_track_suffix(r, Track::Nascent))
        .collect();
    let order = align_gene_axis(&dict, &pb_keys, "d.parquet", "p.parquet").unwrap();
    assert_eq!(order, vec![1, 0]);
}
