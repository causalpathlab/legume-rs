//! Tests for the zarr-backed `SparseMtxData` read path that *does not* preload
//! columns into memory. The sibling `sparse_io_vector_tests.rs` always calls
//! `preload_columns()` so it never exercises the zarr storage reads — these
//! tests cover the merge-and-cache path used by `read_triplets_by_columns` and
//! `read_triplets_by_rows`.

use data_beans::sparse_io::*;
use matrix_util::traits::SampleOps;
use ndarray::Array2;

/// Reference matrix → CSC triplets the slow way, so each test owns a
/// hand-checkable expectation independent of the zarr code path.
fn dense_to_triplets(mat: &Array2<f32>, columns: &[usize]) -> Vec<(u64, u64, f32)> {
    let mut triplets = Vec::new();
    for (jj, &j) in columns.iter().enumerate() {
        for (i, &v) in mat.column(j).iter().enumerate() {
            if v != 0.0 {
                triplets.push((i as u64, jj as u64, v));
            }
        }
    }
    triplets
}

/// Same idea for row-wise reads.
fn dense_to_triplets_by_rows(mat: &Array2<f32>, rows: &[usize]) -> Vec<(u64, u64, f32)> {
    let mut triplets = Vec::new();
    for (ii, &i) in rows.iter().enumerate() {
        for (j, &v) in mat.row(i).iter().enumerate() {
            if v != 0.0 {
                triplets.push((ii as u64, j as u64, v));
            }
        }
    }
    triplets
}

fn sort_triplets(mut t: Vec<(u64, u64, f32)>) -> Vec<(u64, u64, f32)> {
    t.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
    t
}

fn make_sparse_no_preload(data: &Array2<f32>) -> Box<dyn SparseIo<IndexIter = Vec<usize>>> {
    create_sparse_from_ndarray(data, None, None).unwrap()
}

fn assert_triplets_match(actual: Vec<(u64, u64, f32)>, expected: Vec<(u64, u64, f32)>) {
    let a = sort_triplets(actual);
    let e = sort_triplets(expected);
    assert_eq!(a.len(), e.len(), "triplet count mismatch");
    for ((ai, aj, av), (ei, ej, ev)) in a.iter().zip(e.iter()) {
        assert_eq!(ai, ei, "row mismatch");
        assert_eq!(aj, ej, "col mismatch");
        assert!((av - ev).abs() < 1e-6, "value mismatch: {av} vs {ev}");
    }
}

// ─────────────────────────────────────────────────────
// Column reads
// ─────────────────────────────────────────────────────

#[test]
fn columns_contiguous_block() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(8, 16);
    let sp = make_sparse_no_preload(&raw);

    let sel: Vec<usize> = (3..11).collect();
    let (nrow, ncol_out, triplets) = sp.read_triplets_by_columns(sel.clone())?;

    assert_eq!(nrow, 8);
    assert_eq!(ncol_out, sel.len());
    assert_triplets_match(triplets, dense_to_triplets(&raw, &sel));
    Ok(())
}

#[test]
fn columns_sparse_with_gaps() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(6, 20);
    let sp = make_sparse_no_preload(&raw);

    // Picks columns with gaps so adjacent indptr ranges do NOT abut after sorting.
    let sel = vec![0, 5, 11, 17];
    let (_, _, triplets) = sp.read_triplets_by_columns(sel.clone())?;
    assert_triplets_match(triplets, dense_to_triplets(&raw, &sel));
    Ok(())
}

#[test]
fn columns_reversed_preserves_output_order() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(5, 12);
    let sp = make_sparse_no_preload(&raw);

    // Reversed: input order != indptr.start order, so the sort+merge logic
    // must remap output positions correctly.
    let sel: Vec<usize> = (0..12).rev().collect();
    let (_, _, triplets) = sp.read_triplets_by_columns(sel.clone())?;
    assert_triplets_match(triplets, dense_to_triplets(&raw, &sel));
    Ok(())
}

#[test]
fn columns_duplicates() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(4, 10);
    let sp = make_sparse_no_preload(&raw);

    // Same column requested multiple times — both output positions must be filled
    // even though they collapse to a single merged read.
    let sel = vec![3, 3, 7, 3];
    let (_, ncol_out, triplets) = sp.read_triplets_by_columns(sel.clone())?;
    assert_eq!(ncol_out, 4);
    assert_triplets_match(triplets, dense_to_triplets(&raw, &sel));
    Ok(())
}

#[test]
fn columns_empty() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(3, 5);
    let sp = make_sparse_no_preload(&raw);

    let (nrow, ncol_out, triplets) = sp.read_triplets_by_columns(vec![])?;
    assert_eq!(nrow, 3);
    assert_eq!(ncol_out, 0);
    assert!(triplets.is_empty());
    Ok(())
}

#[test]
fn columns_single() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(4, 8);
    let sp = make_sparse_no_preload(&raw);

    let (_, ncol_out, triplets) = sp.read_triplets_by_columns(vec![5])?;
    assert_eq!(ncol_out, 1);
    assert_triplets_match(triplets, dense_to_triplets(&raw, &[5]));
    Ok(())
}

#[test]
fn columns_preload_and_zarr_paths_agree() -> anyhow::Result<()> {
    // Read the same columns through both code paths and ensure they produce
    // equivalent triplets (modulo ordering).
    let raw = Array2::<f32>::runif(7, 15);
    let sel = vec![1, 4, 4, 9, 0, 14];

    let zarr_only = make_sparse_no_preload(&raw);
    let mut preloaded = create_sparse_from_ndarray(&raw, None, None)?;
    preloaded.preload_columns()?;

    let (_, _, t_zarr) = zarr_only.read_triplets_by_columns(sel.clone())?;
    let (_, _, t_pre) = preloaded.read_triplets_by_columns(sel.clone())?;
    assert_triplets_match(t_zarr, t_pre);
    Ok(())
}

// ─────────────────────────────────────────────────────
// Row reads
// ─────────────────────────────────────────────────────

#[test]
fn rows_contiguous_block() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(20, 8);
    let sp = make_sparse_no_preload(&raw);

    let sel: Vec<usize> = (5..15).collect();
    let (nrow_out, ncol, triplets) = sp.read_triplets_by_rows(sel.clone())?;
    assert_eq!(nrow_out, sel.len());
    assert_eq!(ncol, 8);
    assert_triplets_match(triplets, dense_to_triplets_by_rows(&raw, &sel));
    Ok(())
}

#[test]
fn rows_sparse_with_gaps_and_duplicates() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(25, 6);
    let sp = make_sparse_no_preload(&raw);

    let sel = vec![0, 7, 7, 12, 24];
    let (_, _, triplets) = sp.read_triplets_by_rows(sel.clone())?;
    assert_triplets_match(triplets, dense_to_triplets_by_rows(&raw, &sel));
    Ok(())
}

#[test]
fn rows_preload_and_zarr_paths_agree() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(15, 7);
    let sel = vec![1, 4, 4, 9, 0, 14];

    let zarr_only = make_sparse_no_preload(&raw);
    let mut preloaded = create_sparse_from_ndarray(&raw, None, None)?;
    preloaded.preload_rows()?;

    let (_, _, t_zarr) = zarr_only.read_triplets_by_rows(sel.clone())?;
    let (_, _, t_pre) = preloaded.read_triplets_by_rows(sel.clone())?;
    assert_triplets_match(t_zarr, t_pre);
    Ok(())
}

#[test]
fn rows_reversed() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(10, 5);
    let sp = make_sparse_no_preload(&raw);

    let sel: Vec<usize> = (0..10).rev().collect();
    let (_, _, triplets) = sp.read_triplets_by_rows(sel.clone())?;
    assert_triplets_match(triplets, dense_to_triplets_by_rows(&raw, &sel));
    Ok(())
}

// ─────────────────────────────────────────────────────
// Persistent cache reuse (cross-call correctness)
// ─────────────────────────────────────────────────────

#[test]
fn columns_repeated_calls_warm_cache() -> anyhow::Result<()> {
    // The decoded-chunk LRU is stored on `SparseMtxData`, so a second call
    // with the same selection must hit the persisted cache and still return
    // identical triplets. Also covers a *third* call with a different
    // selection that touches overlapping chunks — exercises eviction +
    // partial-hit behaviour.
    let raw = Array2::<f32>::runif(12, 30);
    let sp = make_sparse_no_preload(&raw);

    let sel_a: Vec<usize> = (0..30).collect();
    let sel_b: Vec<usize> = vec![2, 4, 4, 17, 29];

    let (_, _, t_a1) = sp.read_triplets_by_columns(sel_a.clone())?;
    let (_, _, t_a2) = sp.read_triplets_by_columns(sel_a.clone())?;
    let (_, _, t_b) = sp.read_triplets_by_columns(sel_b.clone())?;
    let (_, _, t_a3) = sp.read_triplets_by_columns(sel_a.clone())?;

    assert_triplets_match(t_a1, dense_to_triplets(&raw, &sel_a));
    assert_triplets_match(t_a2, dense_to_triplets(&raw, &sel_a));
    assert_triplets_match(t_b, dense_to_triplets(&raw, &sel_b));
    assert_triplets_match(t_a3, dense_to_triplets(&raw, &sel_a));
    Ok(())
}

#[test]
fn rows_repeated_calls_warm_cache() -> anyhow::Result<()> {
    let raw = Array2::<f32>::runif(40, 6);
    let sp = make_sparse_no_preload(&raw);

    let sel_a: Vec<usize> = (0..40).collect();
    let sel_b: Vec<usize> = vec![3, 3, 19, 27, 38];

    let (_, _, t_a1) = sp.read_triplets_by_rows(sel_a.clone())?;
    let (_, _, t_b) = sp.read_triplets_by_rows(sel_b.clone())?;
    let (_, _, t_a2) = sp.read_triplets_by_rows(sel_a.clone())?;

    assert_triplets_match(t_a1, dense_to_triplets_by_rows(&raw, &sel_a));
    assert_triplets_match(t_b, dense_to_triplets_by_rows(&raw, &sel_b));
    assert_triplets_match(t_a2, dense_to_triplets_by_rows(&raw, &sel_a));
    Ok(())
}

#[test]
fn columns_concurrent_calls_share_cache() -> anyhow::Result<()> {
    // Hammer the same backend from many threads to verify the moka-backed
    // cache is safe under concurrent first-callers (race on `OnceLock::set`)
    // and that no thread sees a corrupt or partial result.
    use std::sync::Arc;
    use std::thread;

    let raw = Array2::<f32>::runif(15, 25);
    let sp: Arc<dyn SparseIo<IndexIter = Vec<usize>>> =
        Arc::from(create_sparse_from_ndarray(&raw, None, None)?);

    let sel: Vec<usize> = (0..25).collect();
    let expected = sort_triplets(dense_to_triplets(&raw, &sel));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let sp = Arc::clone(&sp);
            let sel = sel.clone();
            let expected = expected.clone();
            thread::spawn(move || -> anyhow::Result<()> {
                for _ in 0..4 {
                    let (_, _, t) = sp.read_triplets_by_columns(sel.clone())?;
                    let got = sort_triplets(t);
                    assert_eq!(got.len(), expected.len());
                    for ((ai, aj, av), (ei, ej, ev)) in got.iter().zip(expected.iter()) {
                        assert_eq!((ai, aj), (ei, ej));
                        assert!((av - ev).abs() < 1e-6);
                    }
                }
                Ok(())
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked")?;
    }
    Ok(())
}
