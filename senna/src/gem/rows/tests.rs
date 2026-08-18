//! Tests for the gene-count row grammar shared by `gem` and `gem-encoder`.
//!
//! These moved here from `gem_encoder` when the parser did. They belong to the
//! grammar, not to either model — which is the whole point of the module.
//!
//! The splitter's own tests moved one crate further down with it, to
//! `auxiliary-data/src/feature_rows/tests.rs`. What is left here is the part
//! that is gem's alone: interning, and the warn-and-keep policy for a row the
//! splitter rejects.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::<str>::from(*s)).collect()
}

/// A gene's two rows must intern to ONE gene id. This is the pairing the whole
/// model rests on: if the tracks landed on different ids, `ρ` and `ρ + δ` would
/// describe unrelated genes.
#[test]
fn both_tracks_of_a_gene_share_one_id() {
    let rows = names(&[
        "A/count/spliced",
        "B/count/unspliced",
        "A/count/unspliced",
        "B/count/spliced",
        "C/count/spliced",
    ]);
    let (map, genes) = build_gene_track_map(&rows);

    assert_eq!(map.n_genes, 3);
    assert_eq!(genes.as_slice(), names(&["A", "B", "C"]).as_slice());
    assert_eq!(map.row_to_gene, vec![0, 1, 0, 1, 2]);
    assert_eq!(
        map.row_is_nascent,
        vec![false, true, true, false, false],
        "nascent flags must follow the /count/unspliced suffix"
    );
}

/// The regression this module exists to prevent.
///
/// faba's gene counter emits a pooled `{gene}/count/total` track alongside the
/// two splice tracks. Under `gem`'s old `rsplit_once("/count/")` it parsed as
/// `("A", suffix != "unspliced")` = a SECOND MATURE ROW of gene A, so a gene's
/// spliced signal was counted twice while `gem-encoder`, on the same file,
/// rejected the row. Same input, two different fits.
///
/// Break the fix by restoring the old body and this fails on both asserts:
/// `n_genes` collapses to 1 and `total` joins A's id.
#[test]
fn a_total_row_is_never_folded_into_the_spliced_track() {
    let rows = names(&["A/count/spliced", "A/count/unspliced", "A/count/total"]);
    let (map, genes) = build_gene_track_map(&rows);

    assert_eq!(
        map.n_genes, 2,
        "the total row must get its own id, not join gene A"
    );
    assert_ne!(
        map.row_to_gene[2], map.row_to_gene[0],
        "a `total` row summed into the spliced track double-counts the gene"
    );
    assert_eq!(genes[0].as_ref(), "A");
    assert_eq!(
        genes[1].as_ref(),
        "A/count/total",
        "an unpairable row keeps its full name so it is identifiable in the warning"
    );
    assert!(
        !map.row_is_nascent[2],
        "an unpairable row is never flagged nascent — it must not reach the velocity side"
    );
}

/// Every row keeps its index. The map is used to subset matrices positionally,
/// so dropping a rejected row instead of re-homing it would shift every row
/// after it onto the wrong gene.
#[test]
fn rejected_rows_stay_on_the_axis() {
    let rows = names(&[
        "A/count/spliced",
        "junk",
        "A/count/unspliced",
        "B/m6a/methylated",
    ]);
    let (map, _) = build_gene_track_map(&rows);

    assert_eq!(map.row_to_gene.len(), rows.len());
    assert_eq!(map.row_is_nascent.len(), rows.len());
    assert_eq!(
        map.row_to_gene[0], map.row_to_gene[2],
        "A pairs across tracks"
    );
    let distinct: std::collections::HashSet<u32> = map.row_to_gene.iter().copied().collect();
    assert_eq!(distinct.len(), 3, "A, junk and the m6a row are three genes");
}
