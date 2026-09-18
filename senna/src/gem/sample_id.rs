//! Per-file `@sample` barcode-tagging convention for `senna gem`.
//!
//! Multiple input files of one experiment reuse the same 10x barcode
//! whitelist, so a bare barcode is ambiguous across samples. `senna gem` tags
//! each file's barcodes with `{COLUMN_SEP}{sample_id}` (COLUMN_SEP = `@`)
//! under `ColumnAlignment::Union`, where the sample id is the file's basename
//! with a suffix stripped so every file of one sample collapses to the same id
//! (`rep1_wt_count` → `rep1_wt`).

/// The suffix `faba count` gives its gene-count matrices (`{batch}_count`).
pub const COUNT_SUFFIX: &str = "_count";
/// The suffix `faba genes` gave them before the subcommand became `count`;
/// still stripped so older outputs keep loading.
pub const LEGACY_COUNT_SUFFIX: &str = "_genes";

/// Strip whichever of `suffixes` the basename ends with, first match wins;
/// none matching keeps the full basename.
pub fn strip_any_suffix(base: &str, suffixes: &[&str]) -> Box<str> {
    suffixes
        .iter()
        .find_map(|s| base.strip_suffix(s))
        .unwrap_or(base)
        .into()
}

#[cfg(test)]
mod tests;
