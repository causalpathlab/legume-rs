use super::*;

fn must_train(names: &[&str]) -> MustTrainFeatures {
    MustTrainFeatures {
        names: names.iter().map(|&n| n.into()).collect(),
        source: "test".into(),
    }
}

fn vocab(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&n| n.into()).collect()
}

#[test]
fn resolve_matches_plain_symbols() {
    let v = vocab(&["ACTB", "CD8A", "MS4A1"]);
    assert_eq!(must_train(&["CD8A", "MS4A1"]).resolve(&v), vec![1, 2]);
}

/// A curated panel lists bare symbols; the data may carry `ENSG…_SYMBOL`. Both
/// directions have to reconcile or the flag silently rescues nothing.
#[test]
fn resolve_matches_symbol_against_ensembl_composite() {
    let v = vocab(&["ENSG00000075624_ACTB", "ENSG00000153563_CD8A"]);
    assert_eq!(must_train(&["CD8A"]).resolve(&v), vec![1]);
    assert_eq!(must_train(&["ENSG00000075624"]).resolve(&v), vec![0]);
}

#[test]
fn resolve_is_case_insensitive() {
    let v = vocab(&["ACTB", "CD8A"]);
    assert_eq!(must_train(&["cd8a"]).resolve(&v), vec![1]);
}

/// Marker panels routinely name genes an assay never captured — those are
/// reported and skipped, never fatal.
#[test]
fn resolve_skips_unmatched_names() {
    let v = vocab(&["ACTB", "CD8A"]);
    assert_eq!(must_train(&["CD8A", "NOT_A_GENE"]).resolve(&v), vec![1]);
    assert!(must_train(&["NOT_A_GENE"]).resolve(&v).is_empty());
}

#[test]
fn resolve_dedupes_names_hitting_the_same_row() {
    let v = vocab(&["ENSG00000153563_CD8A"]);
    let got = must_train(&["CD8A", "ENSG00000153563", "cd8a"]).resolve(&v);
    assert_eq!(got, vec![0]);
}

/// gem resolves against interned gene keys, so a rescued gene keeps both splice
/// tracks. Resolving against the raw count rows instead would return only the
/// first — the regression this guards.
#[test]
fn resolve_against_gene_keys_not_count_rows() {
    let gene_keys = vocab(&["ACTB", "CD8A"]);
    assert_eq!(must_train(&["CD8A"]).resolve(&gene_keys), vec![1]);

    // The row axis a gene key expands to; GeneIndex strips the `/count/...`
    // suffix, so a row-level resolve would match only the first of the pair.
    let rows = vocab(&[
        "ACTB/count/spliced",
        "ACTB/count/unspliced",
        "CD8A/count/spliced",
        "CD8A/count/unspliced",
    ]);
    assert_eq!(must_train(&["CD8A"]).resolve(&rows), vec![2]);
}

#[test]
fn resolve_quiet_agrees_with_resolve() {
    let v = vocab(&["ACTB", "CD8A", "MS4A1"]);
    let mk = must_train(&["MS4A1", "CD8A", "NOPE"]);
    assert_eq!(mk.resolve(&v), mk.resolve_quiet(&v));
}

#[test]
fn union_indices_merges_sorted_and_deduped() {
    let mut selected = vec![0, 5, 9];
    let added = union_indices(&mut selected, &[5, 2, 7]);
    assert_eq!(selected, vec![0, 2, 5, 7, 9]);
    assert_eq!(added, 2, "5 was already selected");
}

#[test]
fn union_indices_with_nothing_to_add_is_a_no_op() {
    let mut selected = vec![1, 2, 3];
    assert_eq!(union_indices(&mut selected, &[]), 0);
    assert_eq!(selected, vec![1, 2, 3]);
    assert_eq!(union_indices(&mut selected, &[2]), 0);
    assert_eq!(selected, vec![1, 2, 3]);
}

/// With selection off, every feature is kept already — the flag has nothing to
/// rescue and must not pretend it did.
#[test]
fn load_must_train_is_none_when_no_selection_runs() {
    assert!(load_must_train(Some("/nonexistent"), false)
        .unwrap()
        .is_none());
    assert!(load_must_train(None, true).unwrap().is_none());
}

#[test]
fn load_must_train_errors_on_a_missing_file_when_selection_is_on() {
    assert!(load_must_train(Some("/nonexistent/panel.tsv"), true).is_err());
}
