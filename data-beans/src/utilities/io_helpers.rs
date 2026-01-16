use matrix_util::common_io::*;
use crate::sparse_io::{ROW_SEP, COLUMN_SEP};

/// Read row names from a file, joining multi-column names with ROW_SEP
pub fn read_row_names(row_file: Box<str>, max_row_name_idx: usize) -> anyhow::Result<Vec<Box<str>>> {
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
pub fn read_col_names(col_file: Box<str>, max_column_name_idx: usize) -> anyhow::Result<Vec<Box<str>>> {
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
pub use crate::sparse_io::{MAX_ROW_NAME_IDX, MAX_COLUMN_NAME_IDX};
