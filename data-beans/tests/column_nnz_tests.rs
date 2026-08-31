//! Per-column nnz from the resident indptr — the accessor that makes streaming
//! writes of column subsets possible without a counting pass.
//!
//! The indptr vector is loaded at `open()` and lives in memory either way; what
//! these tests pin is that the accessor reads it faithfully, that a subset sums
//! exactly, and that absence is reported as absence rather than as zeros —
//! `read_column_indptr` silently does nothing when the array is missing, and an
//! accessor that turned that into "every column is empty" would let a streaming
//! writer declare nnz = 0 and produce a corrupt backend that opens cleanly.

use data_beans::sparse_io::*;
use matrix_util::traits::SampleOps;

fn fixture() -> (Box<dyn SparseIo<IndexIter = Vec<usize>>>, Vec<u64>) {
    // Hand-built sparsity with empty leading, interior, and trailing columns —
    // the shapes an even split actually produces.
    let mut arr = ndarray::Array2::<f32>::zeros((5, 7));
    arr[(0, 1)] = 1.0;
    arr[(3, 1)] = 2.0;
    arr[(2, 3)] = 3.0;
    arr[(0, 4)] = 4.0;
    arr[(1, 4)] = 5.0;
    arr[(4, 4)] = 6.0;
    arr[(2, 5)] = 7.0;
    let per_column = vec![0u64, 2, 0, 1, 3, 1, 0];
    let data = create_sparse_from_ndarray(&arr, None, None).expect("fixture backend");
    (data, per_column)
}

#[test]
fn per_column_nnz_matches_a_triplet_count() -> anyhow::Result<()> {
    let (data, expected) = fixture();
    for (col, &want) in expected.iter().enumerate() {
        assert_eq!(
            data.column_nnz(col),
            Some(want),
            "column {col}: indptr-derived nnz must equal the triplet count"
        );
    }
    assert_eq!(data.column_nnz(99), None, "out of range is None, not 0");
    Ok(())
}

#[test]
fn a_subset_sum_is_exact_whatever_the_order() -> anyhow::Result<()> {
    let (data, expected) = fixture();
    // Reversed and gappy on purpose: a streaming writer's column selection is
    // sorted, but the accessor must not depend on that.
    let picked = vec![5usize, 1, 4];
    let want: u64 = picked.iter().map(|&c| expected[c]).sum();
    assert_eq!(data.columns_nnz(&picked), Some(want));
    // The whole matrix agrees with num_non_zeros.
    let every: Vec<usize> = (0..7).collect();
    let all: u64 = data.columns_nnz(&every).expect("resident indptr");
    assert_eq!(all as usize, data.num_non_zeros().expect("nnz"));
    Ok(())
}

#[test]
fn an_out_of_range_column_poisons_the_sum_rather_than_counting_zero() -> anyhow::Result<()> {
    let (data, _) = fixture();
    assert_eq!(
        data.columns_nnz(&[1usize, 99]),
        None,
        "a silent 0 here would understate a streaming writer's declared nnz"
    );
    Ok(())
}

#[test]
fn random_matrix_totals_agree() -> anyhow::Result<()> {
    // Dense runif fixture: every column count is nrow; totals must line up.
    let arr = ndarray::Array2::<f32>::runif(4, 9);
    let data = create_sparse_from_ndarray(&arr, None, None)?;
    let every: Vec<usize> = (0..9).collect();
    assert_eq!(data.columns_nnz(&every), Some(36));
    assert_eq!(data.column_nnz(8), Some(4));
    Ok(())
}

/// `--preload-data` is a request, not a command: preloading is 12 bytes per
/// non-zero with no size check anywhere, and no consumer ever releases it. The
/// budget turns an OOM into a warning plus the cold read path, which every
/// reader already handles — `csc_column_arrays()` returning `None` IS the
/// documented "not preloaded" state.
#[test]
fn a_preload_over_budget_is_skipped_not_obeyed() -> anyhow::Result<()> {
    let arr = ndarray::Array2::<f32>::from_elem((50, 40), 1.0);
    let mut data = create_sparse_from_ndarray(&arr, None, None)?;

    // 2000 nnz x 12 B = 24 kB; a 1 kB budget must refuse.
    std::env::set_var("LEGUME_PRELOAD_BUDGET_BYTES", "1024");
    let result = data.preload_columns();
    std::env::remove_var("LEGUME_PRELOAD_BUDGET_BYTES");
    result?;
    assert!(
        data.csc_column_arrays().is_none(),
        "over budget: the arrays must not be resident"
    );
    // and the cold path still serves reads
    let (_, _, t) = data.read_triplets_by_columns((0..40).collect())?;
    assert_eq!(t.len(), 2000);

    // under budget (default): the request is honoured
    data.preload_columns()?;
    assert!(data.csc_column_arrays().is_some(), "under budget: loaded");
    Ok(())
}
