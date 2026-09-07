use crate::common_io::*;
use rayon::prelude::*;
use std::io::{BufRead, Write};

#[cfg(test)]
mod tests;

/// Write the triplets into a MatrixMarket file with 1-based indices
/// * `triplets` - the triplets to write
/// * `nrow` - number of rows
/// * `ncol` - number of columns
/// * `mtx_file` - the output file (e.g., "matrix.mtx.gz")
pub fn write_mtx_triplets<T>(
    triplets: &Vec<(T, T, f32)>,
    nrow: usize,
    ncol: usize,
    mtx_file: &str,
) -> anyhow::Result<()>
where
    T: std::fmt::Display + Copy + std::ops::Add<Output = T> + num_traits::FromPrimitive,
{
    mkdir_parent(mtx_file)?;

    let mut buf = open_buf_writer(mtx_file)?;

    // write the header
    let nnz = triplets.len();
    writeln!(buf, "%%MatrixMarket matrix coordinate real general")?;
    writeln!(buf, "{}\t{}\t{}", nrow, ncol, nnz)?;

    let one_idx = T::from_usize(1).ok_or(anyhow::anyhow!("failed to have 1 value"))?;
    // write them with 1-based indices
    for &(row, col, val) in triplets {
        let r = row + one_idx;
        let c = col + one_idx;
        writeln!(buf, "{}\t{}\t{}", r, c, val)?;
    }

    buf.flush()?;
    Ok(())
}

/// Read only the shape header of a MatrixMarket file — returns
/// `(nrow, ncol, nnz)` without parsing any triplets.
///
/// Skips `%`-prefixed comment lines and blank lines, then parses the
/// first remaining line as the coordinate-format shape triple. Useful
/// for streaming merge paths that need to pre-size backend datasets
/// before reading the body.
pub fn read_mtx_header(mtx_file: &str) -> anyhow::Result<(usize, usize, usize)> {
    let mut reader = open_buf_reader(mtx_file)?;
    read_shape(&mut *reader, mtx_file)
}

/// Lines parsed per batch. Small enough that a batch's text is still cache
/// resident when the parallel parse reads it back; the triplet vector itself
/// is sized once from the header.
const TRIPLET_BATCH_LINES: usize = 1 << 16;

/// Comments and blank lines, which the coordinate format allows anywhere.
fn is_skippable(line: &str) -> bool {
    matches!(
        line.trim_start().as_bytes().first(),
        None | Some(b'%') | Some(b'#')
    )
}

/// Consume lines up to and including the `nrow ncol nnz` shape line.
fn read_shape<R: BufRead + ?Sized>(
    reader: &mut R,
    mtx_file: &str,
) -> anyhow::Result<(usize, usize, usize)> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(anyhow::anyhow!("no shape header found in {}", mtx_file));
        }
        if !is_skippable(&line) {
            return parse_shape_line(&line, mtx_file);
        }
    }
}

fn parse_shape_line(line: &str, mtx_file: &str) -> anyhow::Result<(usize, usize, usize)> {
    let mut parts = line.split_whitespace();
    let mut field = |what: &str| -> anyhow::Result<usize> {
        parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("mtx header missing {} in {}", what, mtx_file))?
            .parse()
            .map_err(|_| anyhow::anyhow!("mtx header has a bad {} in {}", what, mtx_file))
    };
    Ok((field("nrow")?, field("ncol")?, field("nnz")?))
}

/// Parse one `row col value` body line into a 0-based triplet.
///
/// Every way a line can fail to be an entry is an error, never a dropped line:
/// a body that quietly parses to fewer entries than the header declares is the
/// exact shape of the corruption this whole path is guarding against.
fn parse_triplet_line(
    line: &str,
    nrow: usize,
    ncol: usize,
    mtx_file: &str,
) -> anyhow::Result<(u64, u64, f32)> {
    let bad = |what: &str| anyhow::anyhow!("{}: bad {} in `{}`", mtx_file, what, line.trim());

    let mut parts = line.split_whitespace();
    let (row, col, val) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(row), Some(col), Some(val), None) => (row, col, val),
        _ => return Err(bad("entry, expected `row col value`")),
    };
    let row: u64 = row.parse().map_err(|_| bad("row index"))?;
    let col: u64 = col.parse().map_err(|_| bad("column index"))?;
    let val: f32 = val.parse().map_err(|_| bad("value"))?;

    // coordinate format indices are 1-based
    match (row.checked_sub(1), col.checked_sub(1)) {
        (Some(r), Some(c)) if (r as usize) < nrow && (c as usize) < ncol => Ok((r, c, val)),
        _ => Err(anyhow::anyhow!(
            "{}: entry `{}` is outside the declared {} x {} shape (indices are 1-based)",
            mtx_file,
            line.trim(),
            nrow,
            ncol
        )),
    }
}

/// Read a matrix market file into `(row, col, value)` triplets in file order,
/// together with its `(nrow, ncol, nnz)` shape. Callers that need an order
/// sort for it themselves -- every one in this workspace does -- so sorting
/// here would only be a second full pass over the vector.
///
/// The body is streamed into a vector sized once from the header, 24 bytes per
/// entry. Going through one owned string per line plus one per field instead
/// costs an order of magnitude more, which is what put a matrix with hundreds
/// of millions of entries out of reach of an ordinary machine.
///
/// The entry count is checked against the header on the way out. A short body
/// -- the signature of a decoder that stopped at the first gzip member, or of
/// a truncated file -- fails here rather than becoming a smaller matrix that
/// looks like a successful build.
#[allow(clippy::type_complexity)]
pub fn read_mtx_triplets(
    mtx_file: &str,
) -> anyhow::Result<(Vec<(u64, u64, f32)>, (usize, usize, usize))> {
    let mut reader = open_buf_reader(mtx_file)?;
    let shape = read_shape(&mut *reader, mtx_file)?;
    let (nrow, ncol, nnz) = shape;

    if nnz > nrow.saturating_mul(ncol) {
        return Err(anyhow::anyhow!(
            "{}: header declares {} nonzero entries, more than a {} x {} matrix can hold",
            mtx_file,
            nnz,
            nrow,
            ncol
        ));
    }

    let mut mtx_triplets: Vec<(u64, u64, f32)> = Vec::new();
    mtx_triplets.try_reserve_exact(nnz).map_err(|_| {
        anyhow::anyhow!(
            "{}: cannot allocate room for the declared {} nonzero entries",
            mtx_file,
            nnz
        )
    })?;

    // One reusable line buffer per slot, refilled batch after batch, so a body
    // of a billion lines does not mean a billion live allocations.
    let mut batch: Vec<String> = Vec::new();
    batch.resize_with(TRIPLET_BATCH_LINES, String::new);

    loop {
        let mut filled = 0_usize;
        while filled < TRIPLET_BATCH_LINES {
            let slot = &mut batch[filled];
            slot.clear();
            if reader.read_line(slot)? == 0 {
                break;
            }
            if !is_skippable(slot) {
                filled += 1;
            }
        }
        if filled == 0 {
            break;
        }

        let base = mtx_triplets.len();
        if base + filled > nnz {
            return Err(anyhow::anyhow!(
                "{}: the body holds more entries than the {} the header declares",
                mtx_file,
                nnz
            ));
        }

        // Parse straight into the reserved tail: no per-batch vector, no copy.
        mtx_triplets.resize(base + filled, (0, 0, 0.0));
        mtx_triplets[base..]
            .par_iter_mut()
            .zip(batch[..filled].par_iter())
            .try_for_each(|(dst, line)| {
                *dst = parse_triplet_line(line, nrow, ncol, mtx_file)?;
                Ok::<(), anyhow::Error>(())
            })?;
    }

    if mtx_triplets.len() != nnz {
        return Err(anyhow::anyhow!(
            "{}: header declares {} nonzero entries but the body holds {}; a truncated file or a decoder that stopped at the first gzip member is the usual cause",
            mtx_file,
            nnz,
            mtx_triplets.len()
        ));
    }

    Ok((mtx_triplets, shape))
}
