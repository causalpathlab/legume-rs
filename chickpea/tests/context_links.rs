//! Per-cluster link tables: `c_gpk = w_gp λ_pk / Σ_q w_gq λ_qk`, with `w` the
//! learned attention shares (or the fixed ABC contact, as the baseline) and
//! `λ_pk` peak `p`'s accessibility in cluster `k`.

use chickpea::p2g::cis::CisPairs;
use chickpea::p2g::context::context_shares;
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

#[test]
fn shares_match_a_hand_computed_toy() {
    let pairs = pairs();
    let c = context_shares(&pairs, &pairs.weight, &lambda()).unwrap();
    assert_eq!(c.shape(), (3, 2));
    // cluster 0: 0.6·1 and 0.4·3 → 1/3, 2/3; cluster 1: 0.6·2 and 0 → 1, 0.
    let want = [[1.0 / 3.0, 1.0], [2.0 / 3.0, 0.0], [1.0, 1.0]];
    for (k, row) in want.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!(
                (c[(k, j)] - v).abs() < 1e-6,
                "pair {k} cluster {j}: {} vs {v}",
                c[(k, j)]
            );
        }
    }
}

#[test]
fn every_gene_sums_to_one_in_every_cluster() {
    let pairs = pairs();
    let pi = [0.9f32, 0.1, 1.0];
    let c = context_shares(&pairs, &pi, &lambda()).unwrap();
    for g in 0..2 {
        for j in 0..2 {
            let s: f32 = pairs.gene(g).map(|k| c[(k, j)]).sum();
            assert!((s - 1.0).abs() < 1e-6, "gene {g} cluster {j} sums to {s}");
        }
    }
}

#[test]
fn a_gene_whose_peaks_are_all_closed_in_a_cluster_gets_zeros_not_nans() {
    let pairs = pairs();
    let lam = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]);
    let c = context_shares(&pairs, &pairs.weight, &lam).unwrap();
    assert_eq!((c[(0, 0)], c[(1, 0)]), (0.0, 0.0));
    assert!((c[(2, 0)] - 1.0).abs() < 1e-6);
}

#[test]
fn a_weight_per_pair_is_required() {
    let pairs = pairs();
    assert!(context_shares(&pairs, &[0.5, 0.5], &lambda()).is_err());
}
