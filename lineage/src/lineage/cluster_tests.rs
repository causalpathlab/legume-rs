//! Tests for K selection.

use super::*;

#[test]
fn choose_k_defaults_and_clamps() {
    assert_eq!(choose_k(100, None), 10); // N/10
    assert_eq!(choose_k(15, None), 2); // clamped up to 2
    assert_eq!(choose_k(5000, None), 200); // clamped down to 200
    assert_eq!(choose_k(100, Some(30)), 30); // explicit
    assert_eq!(choose_k(20, Some(50)), 20); // capped at N
}
