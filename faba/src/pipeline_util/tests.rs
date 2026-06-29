//! Tests for the mitochondrial-fraction elbow cutoff.

use super::mito_elbow_cutoff;

/// Build an ascending-sorted MT-fraction vector: `n_low` cells at `low`,
/// then `n_high` cells at `high`.
fn dist(n_low: usize, low: f64, n_high: usize, high: f64) -> Vec<f64> {
    let mut v = vec![low; n_low];
    v.extend(std::iter::repeat_n(high, n_high));
    v // already ascending since low < high
}

#[test]
fn elbow_cuts_a_clear_minority_burst_tail() {
    // 900 low-MT cells + 100 high-MT "burst" cells.
    let fracs = dist(900, 0.02, 100, 0.6);
    let cut = mito_elbow_cutoff(&fracs).expect("a clear tail should yield a cutoff");
    // Cutoff sits at the top of the bulk, so the 100 burst cells are dropped and
    // the 900 bulk cells are kept.
    assert!(
        cut >= 0.02 && cut < 0.6,
        "cutoff {cut} should separate bulk (0.02) from tail (0.6)"
    );
    let dropped = fracs.iter().filter(|&&f| f > cut).count();
    assert_eq!(dropped, 100, "should drop exactly the 100 burst cells");
}

#[test]
fn flat_distribution_yields_no_cut() {
    // No mito genes / no spread → nothing to cut.
    let fracs = vec![0.0; 1000];
    assert!(mito_elbow_cutoff(&fracs).is_none());
    let uniform = vec![0.03; 500];
    assert!(mito_elbow_cutoff(&uniform).is_none());
}

#[test]
fn too_few_cells_yields_no_cut() {
    let fracs = dist(40, 0.02, 9, 0.6); // 49 < 50
    assert!(mito_elbow_cutoff(&fracs).is_none());
}

#[test]
fn majority_high_is_not_filtered() {
    // Over-filtering guard: if the "tail" is actually the majority (elbow in the
    // lower half), don't cut — there is no clear minority burst population.
    let fracs = dist(100, 0.02, 900, 0.6);
    assert!(
        mito_elbow_cutoff(&fracs).is_none(),
        "should not cut when the high-MT group is the majority"
    );
}
