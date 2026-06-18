//! Per-file `@sample` barcode-tagging convention shared by `faba gem` and
//! `faba gem-summary`.
//!
//! Multiple input files of one experiment reuse the same 10x barcode
//! whitelist, so a bare barcode is ambiguous across samples. `faba gem` tags
//! each file's barcodes with `{COLUMN_SEP}{sample_id}` (COLUMN_SEP = `@`)
//! under `ColumnAlignment::Union`, where the sample id is the file's basename
//! with a per-modality suffix stripped so every modality of one sample
//! collapses to the same id (`rep1_wt_genes`, `rep1_wt_m6a_ratio` → `rep1_wt`).
//! `gem-summary` reuses the exact same helpers so its tagged matrix columns
//! line up with the `@sample`-tagged cell membership `gem-annotate` writes.

use matrix_util::common_io::basename;

/// Strip the per-flag `strip` suffix from an already-computed basename.
/// Empty (or non-matching) `strip` keeps the full basename, so two modality
/// files of one sample merge only when their stripped basenames agree.
pub fn strip_sample_id(base: &str, strip: &str) -> Box<str> {
    if strip.is_empty() {
        base.into()
    } else {
        base.strip_suffix(strip).unwrap_or(base).into()
    }
}

/// Per-file sample id: the file's basename (sparse-data extension stripped)
/// with the per-flag `strip` suffix removed. `rep1_wt_genes.zarr.zip` with
/// `strip = "_genes"` → `rep1_wt`.
pub fn file_sample_id(file: &str, strip: &str) -> anyhow::Result<Box<str>> {
    Ok(strip_sample_id(basename(file)?.as_ref(), strip))
}

/// Longest `_`-aligned suffix shared by every basename in `names`.
/// Returns "" with fewer than two inputs, or when no `_`-prefixed suffix
/// (e.g. `_genes`, `_m6a_mixture`) is common to all. Greedy from the
/// longest candidate down — picks the longest `_`-prefixed suffix of
/// `names[0]` that's a suffix of every other entry.
pub fn longest_common_underscore_suffix(names: &[Box<str>]) -> Box<str> {
    if names.len() < 2 {
        return "".into();
    }
    // Candidate `_`-aligned suffixes of the first basename, longest first.
    let first = names[0].as_ref();
    let mut candidates: Vec<&str> = first.match_indices('_').map(|(i, _)| &first[i..]).collect();
    candidates.sort_by_key(|s| std::cmp::Reverse(s.len()));
    for cand in candidates {
        if names[1..].iter().all(|n| n.ends_with(cand)) {
            return cand.into();
        }
    }
    "".into()
}
