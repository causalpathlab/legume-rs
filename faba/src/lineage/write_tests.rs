//! Tests for the output-table helpers.

use super::*;

#[test]
fn numbered_names() {
    let got = numbered("node_", 3);
    assert_eq!(got, vec!["node_0".into(), "node_1".into(), "node_2".into()]);
    assert!(numbered("x", 0).is_empty());
}
