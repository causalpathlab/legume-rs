use super::slab_end;

fn by_col(entries: &[(u64, u64)]) -> Vec<(u64, u64, f32)> {
    let mut t: Vec<(u64, u64, f32)> = entries.iter().map(|&(r, c)| (r, c, 1.0)).collect();
    t.sort_unstable_by_key(|&(r, c, _)| (c, r));
    t
}

/// A slab boundary never falls inside a column, however small the slab.
#[test]
fn a_slab_never_splits_a_column() {
    let t = by_col(&[(0, 0), (1, 0), (2, 0), (0, 1), (5, 1), (3, 2)]);
    assert_eq!(
        slab_end(&t, 0, 1, 3, |t| t.1),
        (3, 1),
        "column 0 has three entries"
    );
    assert_eq!(slab_end(&t, 3, 1, 3, |t| t.1), (5, 2));
}

/// The final slab runs to the axis length, so trailing empties are tiled too.
#[test]
fn the_last_slab_extends_to_the_axis_length() {
    let t = by_col(&[(0, 0), (1, 2)]);
    assert_eq!(
        slab_end(&t, 0, 1, 7, |t| t.1),
        (1, 2),
        "column 1 is empty: next band starts there"
    );
    assert_eq!(
        slab_end(&t, 1, 1, 7, |t| t.1),
        (2, 7),
        "columns 3..7 are empty and belong to the last band"
    );
}

#[test]
fn a_slab_larger_than_the_matrix_is_one_band() {
    let t = by_col(&[(0, 0), (1, 1), (2, 2)]);
    assert_eq!(slab_end(&t, 0, 1 << 20, 3, |t| t.1), (3, 3));
}

/// The same walker serves CSR once the vector is sorted row-major.
#[test]
fn the_walker_follows_the_row_axis_for_csr() {
    let mut t = by_col(&[(2, 0), (2, 1), (0, 3), (4, 2)]);
    t.sort_unstable_by_key(|&(r, c, _)| (r, c));
    assert_eq!(slab_end(&t, 0, 1, 6, |t| t.0), (1, 2), "row 1 is empty");
    assert_eq!(
        slab_end(&t, 1, 1, 6, |t| t.0),
        (3, 4),
        "row 2 has two entries, row 3 is empty"
    );
    assert_eq!(
        slab_end(&t, 3, 1, 6, |t| t.0),
        (4, 6),
        "row 5 is empty and belongs to the last band"
    );
}
