use crate::sparse_io::{COLUMN_SEP, ROW_SEP};
use matrix_util::common_io::*;
use std::ops::Range;

/// Parse a names file into one joined string per line.
///
/// Reads `name_file` as lines of whitespace-separated words, then for each line
/// joins the words at the column indices in `name_columns` (clamped to the
/// line's word count) with `name_sep`. Single source of truth shared by the
/// row/column name readers below and by the zarr/hdf5 backends'
/// `register_names_file` (the backend-specific dataset write stays per backend).
pub fn parse_name_file(
    name_file: &str,
    name_columns: Range<usize>,
    name_sep: &str,
) -> anyhow::Result<Vec<String>> {
    let name_data = read_lines_of_words(name_file, -1)?;
    let names: Vec<String> = name_data
        .lines
        .iter()
        .map(|line| {
            // Clamp the upper bound to the line's word count so a large
            // `name_columns.end` (e.g. a user-supplied word count) iterates at
            // most `line.len()` times rather than materializing a huge range.
            (name_columns.start..name_columns.end.min(line.len()))
                .filter_map(|i| line.get(i))
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(name_sep)
        })
        .collect();
    Ok(names)
}

/// Read row names from a file, joining multi-column names with ROW_SEP
pub fn read_row_names(
    row_file: Box<str>,
    max_row_name_idx: usize,
) -> anyhow::Result<Vec<Box<str>>> {
    Ok(parse_name_file(&row_file, 0..max_row_name_idx, ROW_SEP)?
        .into_iter()
        .map(String::into_boxed_str)
        .collect())
}

/// Read column names from a file, joining multi-column names with COLUMN_SEP
pub fn read_col_names(
    col_file: Box<str>,
    max_column_name_idx: usize,
) -> anyhow::Result<Vec<Box<str>>> {
    Ok(
        parse_name_file(&col_file, 0..max_column_name_idx, COLUMN_SEP)?
            .into_iter()
            .map(String::into_boxed_str)
            .collect(),
    )
}

/// Parse an index specification into an explicit, ordered list of indices.
///
/// Accepts comma-separated single indices and inclusive ranges, e.g.
/// `0,2,5`, `1-10`, or `1-20,50-55`. Whitespace around tokens is ignored.
/// Ranges are inclusive on both ends (`1-10` -> 1,2,...,10).
pub fn parse_index_spec(spec: &str) -> anyhow::Result<Vec<usize>> {
    let mut out = Vec::new();
    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        match token.split_once('-') {
            Some((start, end)) => {
                let start: usize = start
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid range start in `{}`", token))?;
                let end: usize = end
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid range end in `{}`", token))?;
                if start > end {
                    anyhow::bail!("range start {} exceeds end {} in `{}`", start, end, token);
                }
                out.extend(start..=end);
            }
            None => {
                let idx: usize = token
                    .parse()
                    .map_err(|_| anyhow::anyhow!("invalid index `{}`", token))?;
                out.push(idx);
            }
        }
    }
    if out.is_empty() {
        anyhow::bail!("no valid indices parsed from `{}`", spec);
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::parse_index_spec;

    #[test]
    fn singles_and_ranges() {
        assert_eq!(parse_index_spec("0,2,5").unwrap(), vec![0, 2, 5]);
        assert_eq!(parse_index_spec("1-5").unwrap(), vec![1, 2, 3, 4, 5]);
        assert_eq!(
            parse_index_spec("1-3,10,20-22").unwrap(),
            vec![1, 2, 3, 10, 20, 21, 22]
        );
        // single-element range and surrounding whitespace
        assert_eq!(parse_index_spec(" 7 - 7 , 9 ").unwrap(), vec![7, 9]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse_index_spec("5-1").is_err());
        assert!(parse_index_spec("abc").is_err());
        assert!(parse_index_spec("").is_err());
    }
}
