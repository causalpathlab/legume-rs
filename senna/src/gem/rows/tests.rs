//! Tests for the gene-count row grammar shared by `gem` and `gem-encoder`.
//!
//! These moved here from `gem_encoder` when the parser did. They belong to the
//! grammar, not to either model — which is the whole point of the module.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::<str>::from(*s)).collect()
}

#[test]
fn splits_gem_rows_into_gene_and_track() {
    assert_eq!(
        split_count_row("ENSG001_BRCA2/count/spliced"),
        Some(("ENSG001_BRCA2", false))
    );
    assert_eq!(
        split_count_row("ENSG001_BRCA2/count/unspliced"),
        Some(("ENSG001_BRCA2", true))
    );
}

/// A row that is not a gene-level count row must be REJECTED, not silently
/// absorbed as a spliced one.
///
/// The old `rsplit_once("/count/")` could not tell the two apart — both fell to
/// the same branch — so `BRCA2/m6a/methylated` became a mature gene literally
/// named `BRCA2/m6a/methylated`, and a per-site row became a mature row of the
/// right gene. Neither errored, and the `n_nascent > 0` guard does not catch
/// contamination, only a wholly spliced input.
#[test]
fn non_count_rows_are_rejected_not_silently_called_spliced() {
    // wrong modality
    assert_eq!(split_count_row("ENSG001_BRCA2/m6a/methylated"), None);
    // right modality, wrong channel
    assert_eq!(split_count_row("ENSG001_BRCA2/count/total"), None);
    // sub-gene resolution: this model is gene-level, so a site row is not pairable
    assert_eq!(
        split_count_row("ENSG001_BRCA2/count/chr1:100/spliced"),
        None
    );
    // not a feature row at all
    assert_eq!(split_count_row("weird_name"), None);
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
