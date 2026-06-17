//! Unit tests for the filename → sample-id strip inference helpers in
//! [`super`] (`longest_common_underscore_suffix`,
//! `longest_underscore_prefix_in`, `longest_shared_underscore_prefix`,
//! `infer_satellite_strip`).

use super::*;

fn bx(s: &[&str]) -> Vec<Box<str>> {
    s.iter().map(|x| (*x).into()).collect()
}

fn ids(s: &[&str]) -> FxHashSet<Box<str>> {
    s.iter().map(|x| (*x).into()).collect()
}

#[test]
fn lcs_picks_longest_aligned_suffix() {
    // Two genes files → shared `_genes` suffix.
    assert_eq!(
        &*longest_common_underscore_suffix(&bx(&["rep1_wt_genes", "rep1_mut_genes"])),
        "_genes"
    );
    // Shared deeper suffix wins (greedy longest).
    assert_eq!(
        &*longest_common_underscore_suffix(&bx(&["a_rep1_genes", "b_rep1_genes"])),
        "_rep1_genes"
    );
    // Single input → no suffix to diff against.
    assert_eq!(
        &*longest_common_underscore_suffix(&bx(&["rep1_wt_genes"])),
        ""
    );
    // No shared `_`-suffix.
    assert_eq!(
        &*longest_common_underscore_suffix(&bx(&["foo_a", "bar_b"])),
        ""
    );
}

#[test]
fn prefix_match_is_underscore_bounded() {
    let cand = ids(&["rep1_wt", "rep1_mut"]);
    assert_eq!(
        longest_underscore_prefix_in("rep1_wt_m6a_mixture", &cand).as_deref(),
        Some("rep1_wt")
    );
    // No partial-token match: `rep1_wtX` must not match `rep1_wt`.
    assert_eq!(longest_underscore_prefix_in("rep1_wtX_m6a", &cand), None);
    // Longest candidate wins when several align.
    let nested = ids(&["rep1", "rep1_wt"]);
    assert_eq!(
        longest_underscore_prefix_in("rep1_wt_m6a", &nested).as_deref(),
        Some("rep1_wt")
    );
}

#[test]
fn shared_prefix_recovers_single_genes_sample_id() {
    // The single-genes-file case: recover `rep1_wt` from the satellite.
    assert_eq!(
        longest_shared_underscore_prefix("rep1_wt_genes", &bx(&["rep1_wt_m6a_mixture"])).as_deref(),
        Some("rep1_wt")
    );
    // Multiple satellites must all share the prefix.
    assert_eq!(
        longest_shared_underscore_prefix(
            "rep1_wt_genes",
            &bx(&["rep1_wt_m6a_mixture", "rep1_wt_apa_mixture"])
        )
        .as_deref(),
        Some("rep1_wt")
    );
    // No shared `_`-segment at all → None.
    assert_eq!(
        longest_shared_underscore_prefix("rep1_wt_genes", &bx(&["other_m6a"])),
        None
    );
}

#[test]
fn satellite_strip_single_file_against_two_genes_samples() {
    // The user's real layout: genes {rep1_wt, rep1_mut}, one m6A file.
    let genes_ids = ids(&["rep1_wt", "rep1_mut"]);
    let strip =
        infer_satellite_strip(&bx(&["rep1_wt_m6a_mixture"]), &genes_ids, "dartseq").unwrap();
    assert_eq!(&*strip, "_m6a_mixture");
}

#[test]
fn satellite_strip_multi_file_lcs_path() {
    // Two m6A files, one per genes sample → LCS path returns `_m6a_mixture`.
    let genes_ids = ids(&["rep1_wt", "rep1_mut"]);
    let strip = infer_satellite_strip(
        &bx(&["rep1_wt_m6a_mixture", "rep1_mut_m6a_mixture"]),
        &genes_ids,
        "dartseq",
    )
    .unwrap();
    assert_eq!(&*strip, "_m6a_mixture");
}

#[test]
fn satellite_strip_errors_on_no_overlap() {
    let genes_ids = ids(&["rep1_wt", "rep1_mut"]);
    let err = infer_satellite_strip(&bx(&["sampleZ_m6a_mixture"]), &genes_ids, "dartseq")
        .unwrap_err()
        .to_string();
    assert!(err.contains("don't overlap"), "got: {err}");
    // Mentions both flags so the fix is actionable.
    assert!(err.contains("--genes-sample-strip"), "got: {err}");
    assert!(err.contains("--dartseq-sample-strip"), "got: {err}");
}

#[test]
fn satellite_strip_empty_files_is_noop() {
    let genes_ids = ids(&["rep1_wt"]);
    assert_eq!(
        &*infer_satellite_strip(&[], &genes_ids, "dartseq").unwrap(),
        ""
    );
}
