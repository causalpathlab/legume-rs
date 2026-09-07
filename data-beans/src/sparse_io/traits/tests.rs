use super::csc_slab_end;

fn sorted(entries: &[(u64, u64)]) -> Vec<(u64, u64, f32)> {
    let mut t: Vec<(u64, u64, f32)> = entries.iter().map(|&(r, c)| (r, c, 1.0)).collect();
    t.sort_unstable_by_key(|&(r, c, _)| (c, r));
    t
}

/// A slab boundary never falls inside a column, however small the slab.
#[test]
fn a_slab_never_splits_a_column() {
    let t = sorted(&[(0, 0), (1, 0), (2, 0), (0, 1), (5, 1), (3, 2)]);
    let (end, band_end_col) = csc_slab_end(&t, 0, 1, 3);
    assert_eq!((end, band_end_col), (3, 1), "column 0 has three entries");
    let (end, band_end_col) = csc_slab_end(&t, 3, 1, 3);
    assert_eq!((end, band_end_col), (5, 2));
}

/// The final slab runs to `ncol`, so trailing empty columns are tiled too.
#[test]
fn the_last_slab_extends_to_ncol() {
    let t = sorted(&[(0, 0), (1, 2)]);
    let (end, band_end_col) = csc_slab_end(&t, 0, 1, 7);
    assert_eq!(
        (end, band_end_col),
        (1, 2),
        "column 1 is empty: next band starts there"
    );
    let (end, band_end_col) = csc_slab_end(&t, 1, 1, 7);
    assert_eq!(
        (end, band_end_col),
        (2, 7),
        "columns 3..7 are empty and belong to the last band"
    );
}

#[test]
fn a_slab_larger_than_the_matrix_is_one_band() {
    let t = sorted(&[(0, 0), (1, 1), (2, 2)]);
    assert_eq!(csc_slab_end(&t, 0, 1 << 20, 3), (3, 3));
}
