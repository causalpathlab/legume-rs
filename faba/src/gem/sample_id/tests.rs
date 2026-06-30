//! Unit tests for [`super`] — sample-id suffix inference.

use super::*;

fn bx(s: &[&str]) -> Vec<Box<str>> {
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
fn strip_sample_id_handles_empty_and_match() {
    assert_eq!(&*strip_sample_id("rep1_wt_genes", "_genes"), "rep1_wt");
    assert_eq!(&*strip_sample_id("rep1_wt_genes", ""), "rep1_wt_genes");
    // Non-matching strip keeps the full basename.
    assert_eq!(&*strip_sample_id("rep1_wt_genes", "_m6a"), "rep1_wt_genes");
}
