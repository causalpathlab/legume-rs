//! Storage-agnostic helpers shared by the `zarr` and `hdf5` sparse backends.
//!
//! These capture the *algorithmic* duplication between the two `SparseIo`
//! impls (range coalescing and MTX header emission) without touching the
//! *structural* divergences (zarr's decoded-chunk caches and `Element` string
//! storage vs hdf5's datasets and `VarLenUnicode`). Each backend supplies its
//! own retrieval primitive via a closure or keeps its own dataset-write tail;
//! only the format-agnostic math lives here. (Name-file parsing is shared too,
//! but lives in `utilities::io_helpers::parse_name_file` since it is general
//! I/O, not backend-specific.)

use std::io::Write;

/// Coalesce abutting/overlapping ranges in `tagged` (already sorted by
/// `start`), retrieve each merged span once through `retrieve`, and emit
/// triplets via `make_triplet(out_tag, inner_idx, value)`.
///
/// Used by both CSC (`out_tag` = output column, `inner_idx` = row) and CSR
/// (`out_tag` = output row, `inner_idx` = column) read paths. `retrieve`
/// returns the `(data, indices)` slabs covering `[start, end)`; abstracting it
/// keeps each backend's cache / dataset handle backend-private (zarr fetches
/// through its decoded-chunk cache, hdf5 through `read_slice_1d`). The buffers
/// are only indexed, so any `Index<usize>`-able owner works (`Vec`,
/// `ndarray::Array1`, …).
pub(crate) fn coalesce_and_emit<R, F, D, I>(
    tagged: &[(u64, u64, u64)],
    inner_bound: usize,
    make_triplet: F,
    mut retrieve: R,
) -> anyhow::Result<Vec<(u64, u64, f32)>>
where
    R: FnMut(u64, u64) -> anyhow::Result<(D, I)>,
    D: std::ops::Index<usize, Output = f32>,
    I: std::ops::Index<usize, Output = u64>,
    F: Fn(u64, u64, f32) -> (u64, u64, f32),
{
    let total: u64 = tagged.iter().map(|&(_, s, e)| e - s).sum();
    let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity(total as usize);

    let mut i = 0;
    while i < tagged.len() {
        let merged_start = tagged[i].1;
        let mut merged_end = tagged[i].2;
        let mut j = i + 1;
        while j < tagged.len() && tagged[j].1 <= merged_end {
            merged_end = merged_end.max(tagged[j].2);
            j += 1;
        }

        let (data_buf, indices_buf) = retrieve(merged_start, merged_end)?;

        for &(tag, start, end) in &tagged[i..j] {
            let off = (start - merged_start) as usize;
            let len = (end - start) as usize;
            for k in 0..len {
                let inner = indices_buf[off + k];
                let val = data_buf[off + k];
                debug_assert!((inner as usize) < inner_bound);
                ret.push(make_triplet(tag, inner, val));
            }
        }

        i = j;
    }
    Ok(ret)
}

/// Write the MatrixMarket coordinate header: the format magic line followed by
/// the `rows  cols  nnz` size line. The per-nonzero body stays per-backend
/// (zarr block-streams the CSC arrays, hdf5 reads per column).
pub(crate) fn write_mtx_header<W: Write>(
    buf: &mut W,
    nrow: usize,
    ncol: usize,
    nnz: usize,
) -> anyhow::Result<()> {
    writeln!(buf, "%%MatrixMarket matrix coordinate real general")?;
    writeln!(buf, "{}\t{}\t{}", nrow, ncol, nnz)?;
    Ok(())
}
