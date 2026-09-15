//! Unit tests for [`super`] — sample-id suffix inference.

use super::*;

#[test]
fn strip_sample_id_handles_empty_and_match() {
    assert_eq!(&*strip_sample_id("rep1_wt_genes", "_genes"), "rep1_wt");
    assert_eq!(&*strip_sample_id("rep1_wt_genes", ""), "rep1_wt_genes");
    // Non-matching strip keeps the full basename.
    assert_eq!(&*strip_sample_id("rep1_wt_genes", "_m6a"), "rep1_wt_genes");
}

#[test]
fn file_sample_id_strips_the_basename() {
    assert_eq!(
        &*file_sample_id("out/s1_genes.zarr.zip", "_genes").unwrap(),
        "s1"
    );
}
