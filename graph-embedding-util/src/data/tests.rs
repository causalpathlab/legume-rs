use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

/// A single-panel load passes no suffixes, so there is no partition and every
/// negative draw stays exactly as it was.
#[test]
fn no_suffix_list_is_a_single_panel() {
    assert!(feature_modality_from_suffix(&names(&["CD3E", "LYZ"]), None).is_none());
}

/// One modality across several samples is still one panel.
#[test]
fn one_distinct_suffix_is_a_single_panel() {
    let n = names(&["CD3E/scRNA", "LYZ/scRNA"]);
    assert!(feature_modality_from_suffix(&n, Some(&names(&["scRNA", "scRNA"]))).is_none());
}

#[test]
fn rows_are_keyed_by_the_suffix_they_were_namespaced_with() {
    let n = names(&["CD3E/scRNA", "ADT-CD3/scADT", "LYZ/scRNA", "ADT-CD14/scADT"]);
    let m = feature_modality_from_suffix(&n, Some(&names(&["scRNA", "scADT"]))).expect("two");
    assert_eq!(m[0], m[2]);
    assert_eq!(m[1], m[3]);
    assert_ne!(m[0], m[1]);
}

/// The regression this function exists to prevent. faba writes, and `senna
/// gem` reads, `{gene}/{modality}/{channel}` — whose LAST field is the splice
/// channel, not a modality. A trailing-field rule reads that axis as two
/// modalities and silently re-scopes gem's negative and module pools. gem
/// passes no suffix list, so the answer must be `None`.
#[test]
fn a_gem_splice_axis_is_not_two_modalities() {
    let n = names(&[
        "G1/count/spliced",
        "G1/count/unspliced",
        "G2/count/spliced",
        "G2/count/unspliced",
    ]);
    assert!(feature_modality_from_suffix(&n, None).is_none());
    // Even were a suffix list somehow present, a row whose trailing field is
    // not one of the suffixes is not classifiable — refuse rather than guess.
    assert!(feature_modality_from_suffix(&n, Some(&names(&["scRNA", "scADT"]))).is_none());
}

/// A row that carries no suffix at all cannot be placed, so the whole axis
/// declines rather than silently putting it in panel 0.
#[test]
fn an_unsuffixed_row_declines_the_whole_axis() {
    let n = names(&["CD3E/scRNA", "LYZ"]);
    assert!(feature_modality_from_suffix(&n, Some(&names(&["scRNA", "scADT"]))).is_none());
}
