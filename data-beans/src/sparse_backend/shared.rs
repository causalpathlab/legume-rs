//! Storage-agnostic helpers shared by the `zarr` and `hdf5` sparse backends.
//!
//! These capture the *algorithmic* duplication between the two `SparseIo`
//! impls (range coalescing, name-file parsing, MTX header emission) without
//! touching the *structural* divergences (zarr's decoded-chunk caches and
//! `Element` string storage vs hdf5's datasets and `VarLenUnicode`). Each
//! backend supplies its own retrieval primitive via a closure or keeps its
//! own dataset-write tail; only the format-agnostic math lives here.

use matrix_util::common_io::*;
use std::io::Write;
use std::ops::Range;

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

/// Parse a names file into one joined string per line.
///
/// Reads `name_file` as lines of whitespace-separated words, then for each
/// line joins the words at the column indices in `name_columns` with
/// `name_sep`. The backend-specific dataset write (zarr `Element` string vs
/// hdf5 `VarLenUnicode`) stays in the caller.
pub(crate) fn parse_name_file(
    name_file: &str,
    name_columns: Range<usize>,
    name_sep: &str,
) -> anyhow::Result<Vec<String>> {
    let name_data = read_lines_of_words(name_file, -1)?;
    let name_columns: Vec<usize> = name_columns.collect();

    let names: Vec<String> = name_data
        .lines
        .iter()
        .map(|line| {
            name_columns
                .iter()
                .filter_map(|&i| line.get(i))
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(name_sep)
        })
        .collect();
    Ok(names)
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
