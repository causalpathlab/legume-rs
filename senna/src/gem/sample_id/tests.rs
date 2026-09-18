//! Unit tests for [`super`] — sample-id suffix inference.

use super::*;

/// Gene files carry `_count` now and `_genes` from before the rename; both
/// resolve to the same sample id, so old and new outputs load alike.
#[test]
fn gene_file_default_strips_count_then_legacy_genes() {
    let both = [COUNT_SUFFIX, LEGACY_COUNT_SUFFIX];
    assert_eq!(&*strip_any_suffix("rep1_wt_count", &both), "rep1_wt");
    assert_eq!(&*strip_any_suffix("rep1_wt_genes", &both), "rep1_wt");
    assert_eq!(&*strip_any_suffix("rep1_wt", &both), "rep1_wt");
}
