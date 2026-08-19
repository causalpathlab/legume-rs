//! What a row means, and the fold each answer implies.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::<str>::from(*s)).collect()
}

/// Two channel rows per gene, deliberately NOT adjacent and not in gene order,
/// because nothing guarantees a producer emits them that way.
fn channelized() -> Vec<Box<str>> {
    names(&[
        "GENE1/count/spliced",
        "GENE2/count/unspliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
    ])
}

#[test]
fn a_matrix_without_channels_resolves_to_the_identity_axis() {
    let rows = names(&["GENE1", "GENE2", "GENE3"]);
    let axis = GeneAxis::resolve(&rows).unwrap();

    assert!(!axis.is_channelized());
    assert_eq!(axis.n_genes(), 3);
    assert_eq!(axis.gene_names(), rows.as_slice());
    for r in 0..3 {
        assert_eq!(axis.gene_of_row(r), r);
        assert!(!axis.row_is_nascent(r));
    }
}

#[test]
fn a_fully_channelized_matrix_pairs_the_two_tracks_of_each_gene() {
    let axis = GeneAxis::resolve(&channelized()).unwrap();

    assert!(axis.is_channelized());
    assert_eq!(axis.n_genes(), 2);
    assert_eq!(axis.gene_names(), names(&["GENE1", "GENE2"]).as_slice());
    assert_eq!(axis.gene_of_row(0), axis.gene_of_row(2));
    assert_eq!(axis.gene_of_row(1), axis.gene_of_row(3));
    assert!(axis.row_is_nascent(1) && axis.row_is_nascent(2));
    assert!(!axis.row_is_nascent(0) && !axis.row_is_nascent(3));
}

/// The failure `cage` cannot absorb. Break it by falling back to the identity
/// axis on a mixed matrix and the `total` row becomes a third gene whose counts
/// are already inside the other two.
#[test]
fn a_mixed_matrix_is_a_hard_error_naming_the_offenders() {
    let rows = names(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE1/count/total",
    ]);
    let err = GeneAxis::resolve(&rows).unwrap_err().to_string();
    assert!(err.contains("GENE1/count/total"), "{err}");
    assert!(err.contains("total"), "{err}");
}

/// The regression Stage 0 exists to prevent, stated as an equality: a
/// two-channel matrix must fold to exactly the single-channel matrix whose
/// counts are its per-gene sums.
#[test]
fn pooling_two_channels_equals_the_single_channel_matrix() {
    let axis = GeneAxis::resolve(&channelized()).unwrap();

    // rows: G1/s, G2/u, G1/u, G2/s   over three columns
    let two = Mat::from_row_slice(
        4,
        3,
        &[
            1.0, 2.0, 3.0, // GENE1 spliced
            10.0, 20.0, 30.0, // GENE2 unspliced
            4.0, 5.0, 6.0, // GENE1 unspliced
            40.0, 50.0, 60.0, // GENE2 spliced
        ],
    );
    let pooled = axis.pool_rows_opt(&two).expect("channelized axis folds");
    let expect = Mat::from_row_slice(2, 3, &[5.0, 7.0, 9.0, 50.0, 70.0, 90.0]);
    assert_eq!(pooled, expect);

    assert_eq!(
        axis.pool_totals(vec![1.0, 10.0, 4.0, 40.0]),
        vec![5.0, 50.0]
    );

    // Same property on a sparse profile: the two rows of GENE1 merge into one
    // entry, and the result is ascending by gene id.
    let obs = axis.pool_profile(vec![(3, 40.0), (0, 1.0), (2, 4.0)]);
    assert_eq!(obs, vec![(0, 5.0), (1, 40.0)]);
}

#[test]
fn the_identity_axis_folds_are_pass_throughs() {
    let rows = names(&["GENE1", "GENE2"]);
    let axis = GeneAxis::resolve(&rows).unwrap();
    let m = Mat::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

    // `None`, not a copy: the identity axis must not allocate a fold at all.
    assert!(axis.pool_rows_opt(&m).is_none());
    assert_eq!(axis.pool_totals(vec![1.0, 2.0]), vec![1.0, 2.0]);
    // Unsorted input stays untouched — an identity fold must not even reorder.
    assert_eq!(
        axis.pool_profile(vec![(1, 2.0), (0, 1.0)]),
        vec![(1, 2.0), (0, 1.0)]
    );
}

/// HVG ranks ROWS, so it can pick one track of a gene and drop the other. Break
/// the promotion and the projection sees half a gene.
#[test]
fn hvg_weights_are_promoted_to_whole_genes() {
    let axis = GeneAxis::resolve(&channelized()).unwrap();
    // only GENE1's spliced row was selected
    let mut w = vec![1.0, 0.0, 0.0, 0.0];
    let n_genes = axis.promote_row_weights(&mut w);

    assert_eq!(n_genes, 1, "one gene carries weight, not one row");
    assert_eq!(w, vec![1.0, 0.0, 1.0, 0.0], "GENE1's nascent row joins it");
}

/// `δ` is identified only by the contrast, so a gene with counts on one track
/// is not identified — and neither is any gene at all when there are no
/// channels to contrast.
#[test]
fn delta_is_identified_only_where_both_tracks_carry_counts() {
    let axis = GeneAxis::resolve(&channelized()).unwrap();
    //          G1/s   G2/u  G1/u  G2/s
    let totals = [7.0, 0.0, 3.0, 9.0];
    assert_eq!(axis.delta_identified(&totals), vec![true, false]);

    let flat = GeneAxis::resolve(&names(&["GENE1", "GENE2"])).unwrap();
    assert_eq!(flat.delta_identified(&[1.0, 1.0]), vec![false, false]);
}
