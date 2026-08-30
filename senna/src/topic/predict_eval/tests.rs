//! The accumulator's own behaviour. The correlations it calls are tested where
//! they live, in `matrix_util::agreement`.

use super::*;

/// The accumulator's per-gene axis is across CELLS: a gene that tracks the
/// prediction cell-to-cell scores high even when each individual cell's profile
/// does not, which is exactly the distinction the two axes exist to separate.
#[test]
fn the_per_gene_axis_is_across_cells() {
    let mut ev = PredictEval::new(vec![0, 1], true);
    // Two cells; gene 0 rises with the prediction, gene 1 moves against it.
    for (o, p) in [([1.0f32, 9.0], [1.0f32, 1.0]), ([5.0, 1.0], [5.0, 9.0])] {
        ev.keep(&o, &p);
    }
    let per_gene = ev.per_gene();
    assert_eq!(per_gene.len(), 2);
    assert!(
        (per_gene[0].1 - 1.0).abs() < 1e-5,
        "gene 0 tracks: {:?}",
        per_gene[0]
    );
    assert!(
        (per_gene[1].1 + 1.0).abs() < 1e-5,
        "gene 1 anti-tracks: {:?}",
        per_gene[1]
    );
    assert!((per_gene[0].3 - 3.0).abs() < 1e-5, "mean observed");
}

#[test]
fn without_kept_values_there_is_no_per_gene_table() {
    let mut ev = PredictEval::new(vec![0, 1], false);
    ev.keep(&[1.0, 1.0], &[1.0, 1.0]);
    assert!(ev.per_gene().is_empty());
}

/// A composition, for comparing the two families' rates on equal footing.
fn normalized(m: &Mat, col: usize) -> Vec<f32> {
    let c = m.column(col);
    let z = c.sum();
    c.iter().map(|v| v / z).collect()
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < tol, "{a:?} vs {b:?}");
    }
}

#[test]
fn the_embedding_rate_is_a_softmax_of_the_logits() {
    // ρ·θ + b, exponentiated and normalised. Written out longhand here so the
    // max-shift inside `rate` is checked against the definition, not against
    // itself.
    let rho = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let b = [0.1f32, -0.2, 0.3];
    let theta = Mat::from_row_slice(1, 2, &[0.7, -0.4]);
    let recon = Reconstruction::Embedding {
        rho_dh: rho.clone(),
        b_feat: &b,
        theta_nh: &theta,
    };

    let logits: Vec<f32> = (0..3)
        .map(|d| rho[(d, 0)] * 0.7 + rho[(d, 1)] * -0.4 + b[d])
        .collect();
    let m = logits.iter().copied().fold(f32::MIN, f32::max);
    let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let z: f32 = e.iter().sum();
    let want: Vec<f32> = e.iter().map(|v| v / z).collect();

    close(&normalized(&recon.rate(&theta), 0), &want, 1e-6);
}

#[test]
fn the_max_shift_does_not_change_the_composition() {
    // The guard that matters for a real ρ: logits are unbounded, and the shift
    // must be exactly invisible after normalising. Adding a constant to every
    // feature bias shifts every logit equally.
    let rho = Mat::from_row_slice(3, 2, &[2.0, -1.0, 0.0, 3.0, 1.0, 1.0]);
    let theta = Mat::from_row_slice(1, 2, &[4.0, -6.0]);
    let small = [0.0f32, 0.0, 0.0];
    let huge = [80.0f32, 80.0, 80.0];

    let a = Reconstruction::Embedding {
        rho_dh: rho.clone(),
        b_feat: &small,
        theta_nh: &theta,
    };
    let b = Reconstruction::Embedding {
        rho_dh: rho,
        b_feat: &huge,
        theta_nh: &theta,
    };
    close(
        &normalized(&a.rate(&theta), 0),
        &normalized(&b.rate(&theta), 0),
        1e-6,
    );
}

#[test]
fn the_null_is_a_composition_over_the_scored_genes_only() {
    // Mass outside the evaluation set must not reach the null: the likelihood
    // renormalises over the scored genes, so a null carrying weight elsewhere
    // would be a different distribution than the model it is differenced against.
    let totals = vec![10.0f64, 30.0, 999.0, 60.0];
    let out = normalize_over(&totals, &[0, 1, 3], 4);
    assert_eq!(out[2], 0.0, "an unscored gene keeps no mass");
    assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    // 10 : 30 : 60 over a denominator of 100 — the unscored 999 is not in it.
    assert!((out[0] - 0.1).abs() < 1e-6);
    assert!((out[1] - 0.3).abs() < 1e-6);
    assert!((out[3] - 0.6).abs() < 1e-6);
}

#[test]
fn a_null_over_genes_with_no_counts_is_all_zero_rather_than_nan() {
    // Reachable under ablation: a hidden gene set that the test half happens not
    // to express. The scoring loop only reads the null where a count exists, so
    // zeros are never logged — but they must not be NaN either, or they would
    // poison anything that inspects the composition.
    let out = normalize_over(&[0.0, 0.0, 5.0], &[0, 1], 3);
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(out, vec![0.0, 0.0, 0.0]);
}
