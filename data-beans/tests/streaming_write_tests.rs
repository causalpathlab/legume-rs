//! The streaming CSC write path, tested for the first time.
//!
//! The API places nine obligations on a caller (exact nnz, contiguous tiling,
//! sorted rows within columns, ...) and — before these tests — validated none of
//! them. A violation did not error: unwritten regions read back as the zarr
//! fill value, so a gap produced a file that opens and reads cleanly while
//! carrying poisoned or duplicated entries. Each violation test below asserts
//! an `Err`, because "corrupt but readable" is the failure mode a data layer
//! must not have.

use data_beans::sparse_io::*;

type Shape = (usize, usize, usize);

/// Column-major slabs of a fixed 4x5 matrix with an empty interior column.
fn reference() -> (Vec<(u64, u64, f32)>, Shape) {
    let triplets = vec![
        (0u64, 0u64, 1.0f32),
        (2, 0, 2.0),
        (1, 1, 3.0),
        // column 2 empty
        (0, 3, 4.0),
        (1, 3, 5.0),
        (3, 3, 6.0),
        (2, 4, 7.0),
    ];
    (triplets, (4, 5, 7))
}

/// Split `(col, row)`-sorted triplets into per-column-band CSC slabs.
fn slab(triplets: &[(u64, u64, f32)], col_lo: u64, col_hi: u64) -> (Vec<u64>, Vec<u64>, Vec<f32>) {
    let band: Vec<_> = triplets
        .iter()
        .filter(|(_, j, _)| (col_lo..col_hi).contains(j))
        .collect();
    let mut colptr = Vec::new();
    let mut rows = Vec::new();
    let mut vals = Vec::new();
    for c in col_lo..col_hi {
        colptr.push(rows.len() as u64);
        for &&(i, j, x) in &band {
            if j == c {
                rows.push(i);
                vals.push(x);
            }
        }
    }
    (colptr, rows, vals)
}

fn stream_write(
    path: &str,
    shape: Shape,
    bands: &[(u64, u64)],
    triplets: &[(u64, u64, f32)],
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    let mut out = create_sparse_streaming_empty(Some(path), Some(&SparseIoBackend::Zarr))?;
    out.begin_streaming_csc(shape)?;
    let mut nnz_offset = 0u64;
    for &(lo, hi) in bands {
        let (colptr, rows, vals) = slab(triplets, lo, hi);
        out.append_csc_slab(lo, nnz_offset, &colptr, &rows, &vals)?;
        nnz_offset += vals.len() as u64;
    }
    out.finalize_streaming_csc()?;
    out.build_csr_from_csc_streaming()?;
    Ok(out)
}

fn sorted(mut t: Vec<(u64, u64, f32)>) -> Vec<(u64, u64, f32)> {
    t.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
    t
}

#[test]
fn a_streamed_matrix_equals_the_triplet_path() -> anyhow::Result<()> {
    let (triplets, shape) = reference();
    let dir = tempfile::tempdir()?;

    let streamed_path = dir.path().join("s.zarr");
    let streamed = stream_write(
        streamed_path.to_str().expect("utf8"),
        shape,
        &[(0, 2), (2, 5)],
        &triplets,
    )?;

    let reference_backend = create_sparse_from_triplets(
        &triplets,
        shape,
        Some(dir.path().join("t.zarr").to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )?;

    for backend in [&streamed, &reference_backend] {
        assert_eq!(backend.num_rows(), Some(4));
        assert_eq!(backend.num_columns(), Some(5));
        assert_eq!(backend.num_non_zeros(), Some(7));
    }
    let (_, _, a) = streamed.read_triplets_by_columns((0..5).collect())?;
    let (_, _, b) = reference_backend.read_triplets_by_columns((0..5).collect())?;
    assert_eq!(
        sorted(a),
        sorted(b),
        "the two write paths must agree entry-for-entry"
    );
    // the CSR side too — the transpose is part of the contract
    let (_, _, ar) = streamed.read_triplets_by_rows((0..4).collect())?;
    assert_eq!(
        sorted(ar),
        sorted(triplets),
        "CSR disagrees with what was written"
    );
    Ok(())
}

#[test]
fn an_nnz_gap_is_an_error_not_a_poisoned_file() -> anyhow::Result<()> {
    let (triplets, shape) = reference();
    let dir = tempfile::tempdir()?;
    let mut out = create_sparse_streaming_empty(
        Some(dir.path().join("g.zarr").to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )?;
    out.begin_streaming_csc(shape)?;
    let (colptr, rows, vals) = slab(&triplets, 0, 2);
    out.append_csc_slab(0, 0, &colptr, &rows, &vals)?;
    // second band starts 2 entries late: a hole of fill values
    let (colptr, rows, vals) = slab(&triplets, 2, 5);
    let late = out
        .append_csc_slab(2, vals.len() as u64 + 5, &colptr, &rows, &vals)
        .and_then(|()| out.finalize_streaming_csc());
    assert!(late.is_err(), "a gap in the nnz tiling must be refused");
    Ok(())
}

#[test]
fn an_overdeclared_nnz_is_an_error_at_finalize() -> anyhow::Result<()> {
    let (triplets, (nrow, ncol, _)) = reference();
    let dir = tempfile::tempdir()?;
    let mut out = create_sparse_streaming_empty(
        Some(dir.path().join("o.zarr").to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )?;
    // declares 9, writes 7 — the tail would read back as fill
    out.begin_streaming_csc((nrow, ncol, 9))?;
    let (colptr, rows, vals) = slab(&triplets, 0, 5);
    out.append_csc_slab(0, 0, &colptr, &rows, &vals)?;
    assert!(
        out.finalize_streaming_csc().is_err(),
        "declared nnz 9 but wrote 7; the phantom tail must be refused"
    );
    Ok(())
}

#[test]
fn unsorted_rows_within_a_column_are_refused() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut out = create_sparse_streaming_empty(
        Some(dir.path().join("u.zarr").to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )?;
    out.begin_streaming_csc((4, 2, 3))?;
    // column 0 carries rows [2, 0] — descending
    let refused = out.append_csc_slab(0, 0, &[0, 2], &[2, 0, 1], &[1.0, 2.0, 3.0]);
    assert!(
        refused.is_err(),
        "readers document ascending rows as an invariant"
    );
    Ok(())
}

#[test]
fn a_row_index_at_or_past_nrow_is_refused() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut out = create_sparse_streaming_empty(
        Some(dir.path().join("r.zarr").to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )?;
    out.begin_streaming_csc((3, 1, 1))?;
    let refused = out.append_csc_slab(0, 0, &[0], &[3], &[1.0]);
    assert!(
        refused.is_err(),
        "row 3 in a 3-row matrix lands outside the CSR"
    );
    Ok(())
}

#[test]
fn a_nonmonotone_colptr_is_refused() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut out = create_sparse_streaming_empty(
        Some(dir.path().join("m.zarr").to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )?;
    out.begin_streaming_csc((4, 3, 4))?;
    let refused = out.append_csc_slab(0, 0, &[0, 3, 2], &[0, 1, 2, 3], &[1.0; 4]);
    assert!(
        refused.is_err(),
        "a decreasing colptr makes a column claim a negative span"
    );
    Ok(())
}
