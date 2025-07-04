#![allow(dead_code)]

use crate::common_io::*;
use rayon::prelude::*;
use std::io::Write;

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
    if let Some(parent_dir) = std::path::Path::new(mtx_file).parent() {
        std::fs::create_dir_all(parent_dir)?;
    }

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

/// Read a matrix market file and return a vector of triplets (row, col, val)
/// * `mtx_file` - Path to the matrix market file
pub fn read_mtx_triplets(
    mtx_file: &str,
) -> anyhow::Result<(Vec<(u64, u64, f32)>, Option<(usize, usize, usize)>)> {
    let mtx_hdr_position = 0;

    let ReadLinesOut {
        lines: mtx_data_lines,
        header: mtx_data_hdr,
    } = read_lines_of_words(mtx_file, mtx_hdr_position)?;

    // Convert a triplet of strings to a triplet of usize, usize, f32
    fn parse_row_col_val(triplet: &[Box<str>]) -> Option<(u64, u64, f32)> {
        if triplet.len() != 3 {
            return None;
        }

        // f32 should be enough for most cases
        let val = triplet[2].parse::<f32>().ok()?;

        // convert 1-based to 0-based
        let row = (triplet[0].parse::<u64>().ok()?) - 1;
        let col = (triplet[1].parse::<u64>().ok()?) - 1;

        Some((row, col, val))
    }

    let mtx_shape = if mtx_data_hdr.len() == 3 {
        let nrow = mtx_data_hdr[0].parse::<usize>()?;
        let ncol = mtx_data_hdr[1].parse::<usize>()?;
        let nnz = mtx_data_hdr[2].parse::<usize>()?;

        Some((nrow, ncol, nnz))
    } else {
        None
    };

    // #allow(unused_variables)
    if let Some((nrow, ncol, nnz)) = mtx_shape {
        debug_assert!(nrow > 0);
        debug_assert!(ncol > 0);
        debug_assert!(nnz > 0);
    } else {
        return Err(anyhow::anyhow!("Failed to parse mtx header"));
    };

    let mut mtx_triplets = mtx_data_lines
        .iter()
        .par_bridge()
        .filter_map(|x| parse_row_col_val(x))
        .collect::<Vec<_>>();

    mtx_triplets.par_sort_by_key(|&(row, _, _)| row);
    mtx_triplets.par_sort_by_key(|&(_, col, _)| col);
    Ok((mtx_triplets, mtx_shape))
}
