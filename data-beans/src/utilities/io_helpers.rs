use crate::sparse_io::{COLUMN_SEP, ROW_SEP};
use matrix_util::common_io::*;

/// Read row names from a file, joining multi-column names with ROW_SEP
pub fn read_row_names(
    row_file: Box<str>,
    max_row_name_idx: usize,
) -> anyhow::Result<Vec<Box<str>>> {
    let _names = read_lines_of_words(&row_file, -1)?.lines;
    Ok(_names
        .into_iter()
        .map(|x| {
            let s = (0..x.len().min(max_row_name_idx))
                .filter_map(|i| x.get(i))
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(ROW_SEP)
                .parse::<String>()
                .expect("invalid row name");
            s.into_boxed_str()
        })
        .collect())
}

/// Read column names from a file, joining multi-column names with COLUMN_SEP
pub fn read_col_names(
    col_file: Box<str>,
    max_column_name_idx: usize,
) -> anyhow::Result<Vec<Box<str>>> {
    let _names = read_lines_of_words(&col_file, -1)?.lines;
    Ok(_names
        .into_iter()
        .map(|x| {
            let s = (0..x.len().min(max_column_name_idx))
                .filter_map(|i| x.get(i))
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(COLUMN_SEP)
                .parse::<String>()
                .expect("invalid col name");
            s.into_boxed_str()
        })
        .collect())
}

// Re-export constants for convenience
pub use crate::sparse_io::{MAX_COLUMN_NAME_IDX, MAX_ROW_NAME_IDX};

/// Target ~1 MB per chunk: big enough for good compression, small enough for
/// fast random-access reads of individual columns/rows.
const TARGET_CHUNK_BYTES: usize = 1024 * 1024;
/// Never fewer than this many elements in a chunk.
const MIN_CHUNK_ELEMS: usize = 8192;

/// Compute the chunk size (in elements) for a 1-D array of `nelem` elements,
/// each `elem_bytes` wide. Returns at least 1 even for empty arrays.
pub fn chunk_elems(nelem: usize, elem_bytes: usize) -> usize {
    (TARGET_CHUNK_BYTES / elem_bytes.max(1))
        .max(MIN_CHUNK_ELEMS)
        .min(nelem.max(1))
}
