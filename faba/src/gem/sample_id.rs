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
use rustc_hash::FxHashSet;

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

/// Longest `_`-aligned prefix of `name` that is a member of `candidates`.
/// The match is `_`-bounded: a candidate `rep1_wt` matches
/// `rep1_wt_m6a_mixture` (because the char after the prefix is `_`) but
/// does NOT match `rep1_wtX_...`. Returns `None` when no candidate aligns.
fn longest_underscore_prefix_in(name: &str, candidates: &FxHashSet<Box<str>>) -> Option<Box<str>> {
    let mut best: Option<&str> = None;
    for sid in candidates.iter() {
        let s = sid.as_ref();
        if !name.starts_with(s) {
            continue;
        }
        // Boundary: full equality or next char is `_`.
        if name.len() != s.len() && !name[s.len()..].starts_with('_') {
            continue;
        }
        if best.is_none_or(|b| s.len() > b.len()) {
            best = Some(s);
        }
    }
    best.map(|s| s.into())
}

/// Longest `_`-aligned prefix of `name` that is also a `_`-aligned prefix
/// of *every* entry in `others`. `_`-aligned = the prefix is the whole
/// string or is immediately followed by `_`. Returns `None` when not even
/// the first `_`-segment is shared. Used to recover a **single** genes
/// file's sample id from the satellite basenames it must co-sample with
/// (e.g. `rep1_wt_genes` + `rep1_wt_m6a_mixture` → `rep1_wt`), so the genes
/// strip can be inferred without a second genes file to diff against.
pub fn longest_shared_underscore_prefix(name: &str, others: &[Box<str>]) -> Option<Box<str>> {
    // `_`-aligned prefixes of `name`, longest first: the whole name, then
    // each truncation at a `_`.
    let mut cands: Vec<&str> = std::iter::once(name)
        .chain(name.match_indices('_').map(|(i, _)| &name[..i]))
        .collect();
    cands.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let aligned =
        |hay: &str, p: &str| hay == p || hay.strip_prefix(p).is_some_and(|r| r.starts_with('_'));
    cands
        .into_iter()
        .find(|p| others.iter().all(|o| aligned(o.as_ref(), p)))
        .map(|p| p.into())
}

/// Infer the strip suffix for one satellite modality from its `basenames`.
///
/// Algorithm:
///   1. If ≥2 files: try LCS-at-`_` across this modality's basenames.
///      Accept when every stripped name is a known genes sample id.
///   2. Else (1 file, or LCS failed): per file, find the longest
///      `_`-aligned prefix in `genes_ids` and take the tail as the strip.
///      Accept when every file yields the *same* tail.
///   3. Otherwise: hard error with both sample-id sets surfaced so the
///      user can set `--{label}-sample-strip` explicitly.
pub fn infer_satellite_strip(
    basenames: &[Box<str>],
    genes_ids: &FxHashSet<Box<str>>,
    label: &str,
) -> anyhow::Result<Box<str>> {
    // No files → no strip (the caller already short-circuits empty modalities;
    // this keeps the `tails[0]` access below sound for any future caller).
    if basenames.is_empty() {
        return Ok("".into());
    }

    // Genes ids, sorted, for any error message (the source is a HashSet, so
    // sort to keep the diagnostic deterministic run-to-run / test-stable).
    let sorted_genes_ids = || {
        let mut v: Vec<&str> = genes_ids.iter().map(|s| s.as_ref()).collect();
        v.sort_unstable();
        v
    };

    // Path 1: LCS across this modality's basenames, validated against genes ids.
    if basenames.len() >= 2 {
        let lcs = longest_common_underscore_suffix(basenames);
        if !lcs.is_empty() {
            let ok = basenames.iter().all(|b| {
                b.as_ref()
                    .strip_suffix(lcs.as_ref())
                    .is_some_and(|s| genes_ids.contains(s))
            });
            if ok {
                return Ok(lcs);
            }
        }
    }

    // Path 2: per-file LCP against genes ids; tails must agree.
    let mut tails: Vec<Box<str>> = Vec::with_capacity(basenames.len());
    for b in basenames {
        match longest_underscore_prefix_in(b.as_ref(), genes_ids) {
            Some(sid) => {
                let tail = &b.as_ref()[sid.len()..];
                tails.push(tail.into());
            }
            None => {
                anyhow::bail!(
                    "{label} sample ids don't overlap with genes sample ids.\n  \
                     genes sample ids: {:?}\n  \
                     {label} basenames: {:?}\n  \
                     No `_`-aligned suffix or prefix lines up. Set --genes-sample-strip \
                     and/or --{label}-sample-strip so both reduce to the same sample id.",
                    sorted_genes_ids(),
                    basenames.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
                );
            }
        }
    }
    let first = tails[0].clone();
    if tails.iter().all(|t| *t == first) {
        return Ok(first);
    }
    anyhow::bail!(
        "{label} files imply inconsistent sample-id strips.\n  \
         genes sample ids: {:?}\n  \
         per-file inferred strips: {:?}\n  \
         Pass --{label}-sample-strip <SUFFIX> to override.",
        sorted_genes_ids(),
        tails.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(s: &[&str]) -> Vec<Box<str>> {
        s.iter().map(|x| (*x).into()).collect()
    }

    fn ids(s: &[&str]) -> FxHashSet<Box<str>> {
        s.iter().map(|x| (*x).into()).collect()
    }

    #[test]
    fn lcs_picks_longest_aligned_suffix() {
        // Two genes files → shared `_genes` suffix.
        assert_eq!(
            &*longest_common_underscore_suffix(&bx(&["rep1_wt_genes", "rep1_mut_genes"])),
            "_genes"
        );
        // Shared deeper suffix wins (greedy longest).
        assert_eq!(
            &*longest_common_underscore_suffix(&bx(&["a_rep1_genes", "b_rep1_genes"])),
            "_rep1_genes"
        );
        // Single input → no suffix to diff against.
        assert_eq!(
            &*longest_common_underscore_suffix(&bx(&["rep1_wt_genes"])),
            ""
        );
        // No shared `_`-suffix.
        assert_eq!(
            &*longest_common_underscore_suffix(&bx(&["foo_a", "bar_b"])),
            ""
        );
    }

    #[test]
    fn prefix_match_is_underscore_bounded() {
        let cand = ids(&["rep1_wt", "rep1_mut"]);
        assert_eq!(
            longest_underscore_prefix_in("rep1_wt_m6a_mixture", &cand).as_deref(),
            Some("rep1_wt")
        );
        // No partial-token match: `rep1_wtX` must not match `rep1_wt`.
        assert_eq!(longest_underscore_prefix_in("rep1_wtX_m6a", &cand), None);
        // Longest candidate wins when several align.
        let nested = ids(&["rep1", "rep1_wt"]);
        assert_eq!(
            longest_underscore_prefix_in("rep1_wt_m6a", &nested).as_deref(),
            Some("rep1_wt")
        );
    }

    #[test]
    fn shared_prefix_recovers_single_genes_sample_id() {
        // The single-genes-file case: recover `rep1_wt` from the satellite.
        assert_eq!(
            longest_shared_underscore_prefix("rep1_wt_genes", &bx(&["rep1_wt_m6a_mixture"]))
                .as_deref(),
            Some("rep1_wt")
        );
        // Multiple satellites must all share the prefix.
        assert_eq!(
            longest_shared_underscore_prefix(
                "rep1_wt_genes",
                &bx(&["rep1_wt_m6a_mixture", "rep1_wt_apa_mixture"])
            )
            .as_deref(),
            Some("rep1_wt")
        );
        // No shared `_`-segment at all → None.
        assert_eq!(
            longest_shared_underscore_prefix("rep1_wt_genes", &bx(&["other_m6a"])),
            None
        );
    }

    #[test]
    fn satellite_strip_single_file_against_two_genes_samples() {
        // The user's real layout: genes {rep1_wt, rep1_mut}, one m6A file.
        let genes_ids = ids(&["rep1_wt", "rep1_mut"]);
        let strip =
            infer_satellite_strip(&bx(&["rep1_wt_m6a_mixture"]), &genes_ids, "dartseq").unwrap();
        assert_eq!(&*strip, "_m6a_mixture");
    }

    #[test]
    fn satellite_strip_multi_file_lcs_path() {
        // Two m6A files, one per genes sample → LCS path returns `_m6a_mixture`.
        let genes_ids = ids(&["rep1_wt", "rep1_mut"]);
        let strip = infer_satellite_strip(
            &bx(&["rep1_wt_m6a_mixture", "rep1_mut_m6a_mixture"]),
            &genes_ids,
            "dartseq",
        )
        .unwrap();
        assert_eq!(&*strip, "_m6a_mixture");
    }

    #[test]
    fn satellite_strip_errors_on_no_overlap() {
        let genes_ids = ids(&["rep1_wt", "rep1_mut"]);
        let err = infer_satellite_strip(&bx(&["sampleZ_m6a_mixture"]), &genes_ids, "dartseq")
            .unwrap_err()
            .to_string();
        assert!(err.contains("don't overlap"), "got: {err}");
        // Mentions both flags so the fix is actionable.
        assert!(err.contains("--genes-sample-strip"), "got: {err}");
        assert!(err.contains("--dartseq-sample-strip"), "got: {err}");
    }

    #[test]
    fn satellite_strip_empty_files_is_noop() {
        let genes_ids = ids(&["rep1_wt"]);
        assert_eq!(
            &*infer_satellite_strip(&[], &genes_ids, "dartseq").unwrap(),
            ""
        );
    }
}
