use super::*;

#[test]
fn choose_k_defaults_and_clamps() {
    assert_eq!(choose_k(100, None), 10); // N/10
    assert_eq!(choose_k(15, None), 2); // clamped up to 2
    assert_eq!(choose_k(5000, None), 200); // clamped down to 200
    assert_eq!(choose_k(100, Some(30)), 30); // explicit
    assert_eq!(choose_k(20, Some(50)), 20); // capped at N
}

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

#[test]
fn resolve_root_node_override() {
    let (nm, lab) = (names(&["a", "b", "c"]), vec![0usize, 1, 1]);
    assert_eq!(resolve_root(Some(2), None, &nm, &lab, 3, None).unwrap(), 2);
    assert!(resolve_root(Some(5), None, &nm, &lab, 3, None).is_err()); // out of range
}

#[test]
fn resolve_root_cell_maps_to_its_node() {
    let (nm, lab) = (names(&["a", "b", "c"]), vec![0usize, 1, 1]);
    assert_eq!(
        resolve_root(None, Some("b"), &nm, &lab, 3, None).unwrap(),
        1
    );
    assert!(resolve_root(None, Some("zzz"), &nm, &lab, 3, None).is_err());
}

#[test]
fn resolve_root_falls_back_to_velocity_then_zero() {
    let (nm, lab) = (names(&["a", "b"]), vec![0usize, 1]);
    assert_eq!(resolve_root(None, None, &nm, &lab, 2, Some(1)).unwrap(), 1);
    assert_eq!(resolve_root(None, None, &nm, &lab, 2, None).unwrap(), 0);
}

#[test]
fn numbered_names() {
    let got = numbered("node_", 3);
    assert_eq!(got, vec!["node_0".into(), "node_1".into(), "node_2".into()]);
    assert!(numbered("x", 0).is_empty());
}
