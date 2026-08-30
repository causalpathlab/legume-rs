//! Ablation is the flag that turns the score from a reconstruction into a
//! prediction, so its gate has to be exact: a gene named for hiding must leave
//! the encoder's view, and nothing else may move.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

fn hide_file(dir: &std::path::Path, lines: &[&str]) -> String {
    let p = dir.join("hide.txt");
    std::fs::write(&p, lines.join("\n")).expect("write");
    p.to_string_lossy().into_owned()
}

#[test]
fn ablation_hides_exactly_the_named_features() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = names(&["a", "b", "c", "d"]);
    let path = hide_file(dir.path(), &["b", "d"]);

    // No prior remap: the axes matched, so an identity one is materialised and
    // only the named rows differ from it.
    let out = apply_ablation(None, &genes, &path)
        .expect("ablation")
        .expect("a remap");
    assert_eq!(
        out.new_to_train,
        vec![Some(0), None, Some(2), None],
        "only the named genes may be hidden"
    );
    assert_eq!(out.n_mapped, 2);
}

#[test]
fn ablation_composes_with_an_existing_remap() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = names(&["a", "b", "c"]);
    let path = hide_file(dir.path(), &["c"]);

    // Query row 1 already had no model gene; hiding row 2 must not resurrect it
    // or renumber the survivor.
    let prior = GeneRemap {
        new_to_train: vec![Some(5), None, Some(7)],
        d_train: 9,
        n_mapped: 2,
    };
    let out = apply_ablation(Some(prior), &genes, &path)
        .expect("ablation")
        .expect("a remap");
    assert_eq!(out.new_to_train, vec![Some(5), None, None]);
    assert_eq!(out.n_mapped, 1);
    assert_eq!(out.d_train, 9, "the model axis is untouched");
}

#[test]
fn a_name_that_matches_nothing_is_an_error_not_a_silent_reconstruction() {
    // The failure mode this guards: a typo'd file leaves every gene visible, the
    // run succeeds, and the reported number is a plain reconstruction wearing the
    // ablation's name.
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = names(&["a", "b"]);
    let path = hide_file(dir.path(), &["zzz"]);
    assert!(apply_ablation(None, &genes, &path).is_err());
}

#[test]
fn hiding_every_feature_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = names(&["a", "b"]);
    let path = hide_file(dir.path(), &["a", "b"]);
    assert!(apply_ablation(None, &genes, &path).is_err());
}

#[test]
fn blank_lines_and_padding_are_ignored() {
    let dir = tempfile::tempdir().expect("tempdir");
    let genes = names(&["a", "b", "c"]);
    let path = hide_file(dir.path(), &["", "  b  ", "", "c"]);
    let out = apply_ablation(None, &genes, &path)
        .expect("ablation")
        .expect("a remap");
    assert_eq!(out.new_to_train, vec![Some(0), None, None]);
}
