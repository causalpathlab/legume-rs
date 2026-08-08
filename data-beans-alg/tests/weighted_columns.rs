//! A column may stand for more than one observation.
//!
//! Two properties, and the first is the one that protects everyone else:
//!
//! 1. **Unit weights change nothing.** Every existing caller registers no
//!    multiplicity, so every statistic must be bit-for-bit what it was.
//! 2. **`m` identical cells == one column of their mean with weight `m`.**
//!    This is the identity that lets a carried pseudobulk stand in for the
//!    cells it was built from. If it does not hold exactly, a reference
//!    pseudobulk silently mis-weights the cohort it represents.

use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::CollapsingOps;

const D: usize = 6;

/// `cells` as (gene, value) columns → an in-memory backend in a temp dir.
fn backend(tag: &str, cells: &[Vec<f32>]) -> SparseIoVec {
    let dir = std::env::temp_dir().join(format!("dba_wcol_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join(format!("{tag}.zarr"));
    let _ = std::fs::remove_dir_all(&path);

    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (j, col) in cells.iter().enumerate() {
        for (g, &v) in col.iter().enumerate() {
            if v != 0.0 {
                triplets.push((g as u64, j as u64, v));
            }
        }
    }
    let shape = (D, cells.len(), triplets.len());
    let mut b = create_sparse_from_triplets(
        &triplets,
        shape,
        Some(path.to_str().expect("utf8")),
        Some(&data_beans::sparse_io::SparseIoBackend::Zarr),
    )
    .expect("create backend");
    b.register_row_names_vec(
        &(0..D)
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..cells.len())
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );

    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

/// Collapse everything into one group and return `(observed_sum, size)`.
fn one_group(v: &mut SparseIoVec, weights: Option<&[f32]>) -> (Vec<f32>, f32) {
    let n = v.num_columns();
    v.register_batch_membership(&vec!["b0"; n]);
    if let Some(w) = weights {
        v.register_column_multiplicity(w).expect("weights");
    }
    v.assign_groups(&vec!["g0".to_string(); n], None);

    let out = v
        .collapse_columns(None, None, None, Some(1))
        .expect("collapse");
    // With a single batch there is no δ, so `mu_observed` is exactly
    // `Σy / n` — the quantity both properties are about.
    let mu = matrix_param::traits::Inference::posterior_mean(&out.mu_observed);
    let rates: Vec<f32> = (0..D).map(|g| mu[(g, 0)]).collect();
    (rates, n as f32)
}

#[test]
fn unit_weights_change_nothing() {
    let cells: Vec<Vec<f32>> = (0..8)
        .map(|j| (0..D).map(|g| ((g + j) % 5) as f32).collect())
        .collect();

    let (plain, _) = one_group(&mut backend("plain", &cells), None);
    let ones = vec![1.0f32; cells.len()];
    let (weighted, _) = one_group(&mut backend("ones", &cells), Some(&ones));

    assert_eq!(
        plain, weighted,
        "registering all-ones multiplicity must be bit-for-bit inert"
    );
}

/// The identity the carried-pseudobulk encoding rests on.
#[test]
fn one_weighted_column_equals_the_cells_it_summarizes() {
    const M: usize = 20;
    let profile: Vec<f32> = (0..D).map(|g| 1.0 + g as f32).collect();

    // M identical cells...
    let many: Vec<Vec<f32>> = (0..M).map(|_| profile.clone()).collect();
    let (from_cells, n_cells) = one_group(&mut backend("many", &many), None);

    // ...versus one column holding their (identical) mean, weighing M.
    let (from_summary, n_summary) = one_group(
        &mut backend("one", std::slice::from_ref(&profile)),
        Some(&[M as f32]),
    );

    assert_eq!(n_cells, M as f32);
    assert_eq!(n_summary, 1.0, "one physical column");

    for g in 0..D {
        assert!(
            (from_cells[g] - from_summary[g]).abs() < 1e-4,
            "gene {g}: {M} cells gave {}, the weighted summary gave {}",
            from_cells[g],
            from_summary[g]
        );
    }
}

/// A weight of zero would drop a column from the denominator while leaving its
/// counts in the numerator — an inflated rate, silently.
#[test]
fn non_positive_and_mismatched_weights_are_refused() {
    let cells: Vec<Vec<f32>> = (0..3).map(|_| vec![1.0; D]).collect();
    let mut v = backend("bad", &cells);

    let err = v
        .register_column_multiplicity(&[1.0, 1.0])
        .expect_err("wrong length must be refused");
    assert!(err.to_string().contains("3 columns"), "{err}");

    let err = v
        .register_column_multiplicity(&[1.0, 0.0, 1.0])
        .expect_err("zero weight must be refused");
    assert!(err.to_string().contains("multiplicity"), "{err}");

    let err = v
        .register_column_multiplicity(&[1.0, f32::NAN, 1.0])
        .expect_err("NaN weight must be refused");
    assert!(err.to_string().contains("multiplicity"), "{err}");
}
