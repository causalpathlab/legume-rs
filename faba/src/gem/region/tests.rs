//! Unit tests for [`super::RegionMap`] — per-component slot lookup + clamping.

use super::*;

#[test]
fn lookup_clamps_to_n_regions_minus_one() {
    let map = RegionMap::new(5);
    // Within-range components return their own index.
    assert_eq!(map.lookup(0), 0);
    assert_eq!(map.lookup(1), 1);
    assert_eq!(map.lookup(4), 4);
    // Beyond-range components clamp to R - 1.
    assert_eq!(map.lookup(5), 4);
    assert_eq!(map.lookup(100), 4);
}

#[test]
fn zero_n_regions_clamped_to_one() {
    let map = RegionMap::new(0);
    assert_eq!(map.n_regions, 1);
    assert_eq!(map.lookup(0), 0);
    assert_eq!(map.lookup(7), 0);
}
