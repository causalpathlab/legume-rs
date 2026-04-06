use log::info;
use matrix_util::common_io::{open_buf_reader, open_buf_writer};
use rustc_hash::FxHashSet;
use std::io::{BufRead, Write};

/// Read a gzipped ATAC fragments BED file and optionally filter to a set of barcodes.
///
/// Fragment files from 10x Multiome have tab-separated columns:
///   chr \t start \t end \t barcode \t count
///
/// Lines starting with `#` are treated as comments and passed through.
///
/// Returns the number of fragments written.
pub fn filter_fragments(
    input: &str,
    output: &str,
    barcodes: Option<&FxHashSet<Box<str>>>,
) -> anyhow::Result<usize> {
    let reader = open_buf_reader(input)?;
    let mut writer = open_buf_writer(output)?;

    let mut n_written = 0usize;
    let mut n_total = 0usize;

    for line in reader.lines() {
        let line = line?;

        // Pass through comment/header lines
        if line.starts_with('#') {
            writeln!(writer, "{}", line)?;
            continue;
        }

        n_total += 1;

        if let Some(barcode_set) = barcodes {
            // Barcode is the 4th column (index 3)
            let barcode = line
                .split('\t')
                .nth(3)
                .ok_or_else(|| anyhow::anyhow!("line {} has fewer than 4 columns", n_total))?;

            if !barcode_set.contains(barcode) {
                continue;
            }
        }

        writeln!(writer, "{}", line)?;
        n_written += 1;
    }

    info!(
        "Fragments: {} / {} retained ({})",
        n_written, n_total, output
    );

    Ok(n_written)
}
