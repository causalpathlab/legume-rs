use crate::sparse_io::{COLUMN_SEP, ROW_SEP};
use matrix_util::common_io::*;

/// Read row names from a file, joining multi-column names with ROW_SEP
pub fn read_row_names(
    row_file: Box<str>,
    max_row_name_idx: usize,
) -> anyhow::Result<Vec<Box<str>>> {
    let names = read_lines_of_words(&row_file, -1)?.lines;
    Ok(names
        .into_iter()
        .map(|x| {
            let end = x.len().min(max_row_name_idx.saturating_add(1));
            let s = (0..end)
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
    let names = read_lines_of_words(&col_file, -1)?.lines;
    Ok(names
        .into_iter()
        .map(|x| {
            let end = x.len().min(max_column_name_idx.saturating_add(1));
            let s = (0..end)
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
        if token.starts_with('-') {
            anyhow::bail!("negative indices are not supported: `{}`", token);
        }
        if let Ok(idx) = token.parse::<usize>() {
            out.push(idx);
            continue;
        }
        if token.contains('-') {
            let parts: Vec<&str> = token.split('-').collect();
            if parts.len() != 2 || parts[0].trim().is_empty() || parts[1].trim().is_empty() {
                anyhow::bail!("invalid range `{}`", token);
            }
            let start: usize = parts[0]
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid range start in `{}`", token))?;
            let end: usize = parts[1]
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid range end in `{}`", token))?;
            if start > end {
                anyhow::bail!("range start {} exceeds end {} in `{}`", start, end, token);
            }
            out.extend(start..=end);
        } else {
            anyhow::bail!("invalid index `{}`", token);
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
        assert!(parse_index_spec("-5").is_err());
        assert!(parse_index_spec("3--5").is_err());
    }
}
