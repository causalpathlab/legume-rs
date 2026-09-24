//! Per-cluster link tables: `c_gpk = w_gp λ_pk / Σ_q w_gq λ_qk`, with `w` the
//! trained gate weights and, as the baseline, the fixed ABC contact, and
//! `λ_pk` peak `p`'s accessibility in cluster `k`.

use chickpea::p2g::cis::CisPairs;
use chickpea::p2g::context::for_each_context_share;
use nalgebra::DMatrix;

/// GENE1 with peaks 0 and 1; GENE2 with peak 2 only.
fn pairs() -> CisPairs {
    CisPairs {
        gene_ptr: vec![0, 2, 3],
        peak: vec![0, 1, 2],
        dist: vec![0, 5_000, 1_000],
        weight: vec![0.6, 0.4, 1.0],
        n_genes_placed: 2,
        n_unparsed_peaks: 0,
        n_unreached_peaks: 0,
    }
}

/// Accessibility `[peaks × clusters]`: cluster 0 favours peak 1, cluster 1
/// closes peak 1 entirely.
fn lambda() -> DMatrix<f32> {
    DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 0.0, 5.0, 5.0])
}

/// Dense `[pairs × clusters]` gate and ABC shares.
fn shares(pairs: &CisPairs, gate: &[f32], lambda: &DMatrix<f32>) -> (DMatrix<f32>, DMatrix<f32>) {
    let mut a = DMatrix::zeros(pairs.n_pairs(), lambda.ncols());
    let mut b = a.clone();
    for_each_context_share(pairs, gate, lambda, |k, j, ga, ab| {
        a[(k, j)] = ga;
        b[(k, j)] = ab;
    })
    .unwrap();
    (a, b)
}

#[test]
fn shares_match_a_hand_computed_toy() {
    let pairs = pairs();
    // Gate = ABC here, so both tables agree.
    let (c, abc) = shares(&pairs, &pairs.weight, &lambda());
    // cluster 0: 0.6·1 and 0.4·3 → 1/3, 2/3; cluster 1: 0.6·2 and 0 → 1, 0.
    let want = [[1.0 / 3.0, 1.0], [2.0 / 3.0, 0.0], [1.0, 1.0]];
    for (k, row) in want.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!(
                (c[(k, j)] - v).abs() < 1e-6,
                "gate pair {k} cluster {j}: {}",
                c[(k, j)]
            );
            assert!(
                (abc[(k, j)] - v).abs() < 1e-6,
                "abc pair {k} cluster {j}: {}",
                abc[(k, j)]
            );
        }
    }
}

#[test]
fn every_gene_sums_to_one_in_every_cluster() {
    let pairs = pairs();
    let (c, abc) = shares(&pairs, &[0.9, 0.1, 1.0], &lambda());
    for g in 0..2 {
        for j in 0..2 {
            let s: f32 = pairs.gene(g).map(|k| c[(k, j)]).sum();
            assert!(
                (s - 1.0).abs() < 1e-6,
                "gate: gene {g} cluster {j} sums to {s}"
            );
            let s: f32 = pairs.gene(g).map(|k| abc[(k, j)]).sum();
            assert!(
                (s - 1.0).abs() < 1e-6,
                "abc: gene {g} cluster {j} sums to {s}"
            );
        }
    }
}

#[test]
fn a_gene_whose_peaks_are_all_closed_in_a_cluster_gets_zeros_not_nans() {
    let pairs = pairs();
    let lam = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]);
    let (c, _) = shares(&pairs, &pairs.weight, &lam);
    assert_eq!((c[(0, 0)], c[(1, 0)]), (0.0, 0.0));
    assert!((c[(2, 0)] - 1.0).abs() < 1e-6);
}

#[test]
fn a_gate_weight_per_pair_is_required() {
    let pairs = pairs();
    assert!(for_each_context_share(&pairs, &[0.5, 0.5], &lambda(), |_, _, _, _| {}).is_err());
}
