use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

/// Make duplicate names unique by appending `-1`, `-2`, etc. to repeated entries.
/// Similar to scanpy's `var_names_make_unique()`.
pub fn make_names_unique(names: &mut [Box<str>]) -> usize {
    let mut counts: HashMap<Box<str>, usize> = HashMap::default();
    let mut num_duped = 0usize;
    for name in names.iter_mut() {
        if let Some(count) = counts.get_mut(name.as_ref()) {
            if *count == 1 {
                num_duped += 1;
            }
            *name = format!("{}-{}", name, count).into_boxed_str();
            *count += 1;
        } else {
            counts.insert(name.clone(), 1);
        }
    }
    if num_duped > 0 {
        log::warn!(
            "{} names had duplicates and were made unique with -N suffixes",
            num_duped
        );
    }
    num_duped
}

/// Combine feature IDs and names into composite `id_name` strings.
/// If a name is empty or already equals the ID (e.g. 10x ATAC peaks where
/// both `features/id` and `features/name` are `chr1:1000-2000`), the ID is
/// used as-is to avoid `chr1:1000-2000_chr1:1000-2000` duplication.
pub fn compose_id_name(ids: Vec<Box<str>>, names: Vec<Box<str>>) -> Vec<Box<str>> {
    ids.into_iter()
        .zip(names)
        .map(|(id, name)| {
            if name.is_empty() || name.as_ref() == id.as_ref() {
                id
            } else {
                format!("{}_{}", id, name).into_boxed_str()
            }
        })
        .collect()
}

/// Comma-separated case-insensitive substring filter, parsed once and matched
/// many times. Used by `--select-row-type` / `--remove-row-type` /
/// `--hto-row-type` so callers can pass e.g. `"gene,peak"` to match either
/// "Gene Expression" or "Peaks".
pub struct RowTypeFilter {
    patterns: Vec<Box<str>>,
}

impl RowTypeFilter {
    pub fn parse(s: &str) -> Self {
        let patterns = s
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.to_ascii_lowercase().into_boxed_str())
            .collect();
        Self { patterns }
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// True if any pattern is an ASCII-case-insensitive substring of `s`.
    /// Bytewise scan — does not allocate, so callers can pass row types
    /// straight from the backend without an intermediate lowercase copy.
    pub fn matches(&self, s: &str) -> bool {
        self.patterns
            .iter()
            .any(|p| contains_ignore_ascii_case(s, p))
    }
}

/// Bytewise case-insensitive substring search. ASCII only; non-ASCII bytes
/// compare verbatim. Allocation-free.
pub fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    let n = needle.len();
    if n == 0 {
        return true;
    }
    let h = haystack.as_bytes();
    if h.len() < n {
        return false;
    }
    h.windows(n)
        .any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Return indices of rows whose type passes select/remove filtering.
/// - `select`: comma-separated patterns; row passes if any pattern is a
///   case-insensitive substring of the row type. Empty keeps all rows.
/// - `remove`: comma-separated patterns; row is dropped if any pattern matches.
pub fn filter_row_indices_by_type(
    row_types: &[Box<str>],
    select: &str,
    remove: &str,
) -> Vec<usize> {
    let sel = RowTypeFilter::parse(select);
    let rem = RowTypeFilter::parse(remove);
    if sel.is_empty() && rem.is_empty() {
        return (0..row_types.len()).collect();
    }
    row_types
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let selected = sel.is_empty() || sel.matches(x);
            let removed = !rem.is_empty() && rem.matches(x);
            if selected && !removed {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Flexible gene name matching (case-insensitive, underscore-delimited)
/// Returns true if `query` matches `target` with these rules:
/// - Exact match (case-insensitive)
/// - Suffix match: target ends with `_query`
/// - Prefix match: target starts with `query_`
/// - Segment match: target contains `_query_`
///
/// Example: "CD8A" matches "ENSG00000153563_CD8A", "CD8A_variant1", "chr1_CD8A_isoform2"
#[allow(dead_code)]
pub fn flexible_name_match(query: &str, target: &str) -> bool {
    let q = query.to_lowercase();
    let t = target.to_lowercase();
    t == q
        || t.ends_with(&format!("_{}", q))
        || t.starts_with(&format!("{}_", q))
        || t.contains(&format!("_{}_", q))
}

/// Pre-built index over a gene-name vocabulary for fast marker→row matching.
/// Resolves a query gene in three tiers, returning the first matching row:
///   1. exact (case-insensitive) full-name match,
///   2. last `_`-segment symbol match (`CD8A` → `ENSG…_CD8A`),
///   3. fallback to the general [`flexible_name_match`] (prefix / `_x_` segment).
///
/// Tiers 1–2 are O(1) hash lookups; only the rare fallback scans the
/// vocabulary. Build once, match many — replaces the O(genes × markers)
/// `.position(flexible_name_match)` scan. Tier 1 preferring an exact match
/// over an earlier-indexed suffix match is the one intended refinement vs a
/// pure positional scan.
#[allow(dead_code)] // consumed by downstream crates (geu, senna), not the data-beans bin
pub struct GeneIndex {
    lowered: Vec<String>,
    exact: HashMap<String, usize>,
    symbol: HashMap<String, usize>,
}

#[allow(dead_code)] // consumed by downstream crates (geu, senna), not the data-beans bin
impl GeneIndex {
    /// Build the index from the dictionary's gene-name order. The first row
    /// wins on duplicate keys (matching positional-scan semantics).
    #[must_use]
    pub fn build(gene_names: &[Box<str>]) -> Self {
        let lowered: Vec<String> = gene_names.par_iter().map(|g| g.to_lowercase()).collect();
        let mut exact: HashMap<String, usize> = HashMap::default();
        let mut symbol: HashMap<String, usize> = HashMap::default();
        for (i, low) in lowered.iter().enumerate() {
            exact.entry(low.clone()).or_insert(i);
            // Strip a faba-style aux suffix first (`SYMBOL/count/spliced` →
            // symbol is the leading `/`-segment), then an Ensembl-style prefix
            // (`ENSG…_CD8A` → symbol is the trailing `_`-segment). Handles
            // either convention, or both combined (`ENSG…_CD8A/count/spliced`).
            let core = low.split('/').next().unwrap_or(low);
            if let Some(sym) = core.rsplit('_').next() {
                symbol.entry(sym.to_string()).or_insert(i);
            }
        }
        Self {
            lowered,
            exact,
            symbol,
        }
    }

    /// Row index for `gene`, or `None` if unmatched (tiers above).
    #[must_use]
    pub fn match_gene(&self, gene: &str) -> Option<usize> {
        let gl = gene.to_lowercase();
        if let Some(&i) = self.exact.get(&gl) {
            return Some(i);
        }
        if let Some(&i) = self.symbol.get(&gl) {
            return Some(i);
        }
        self.lowered
            .iter()
            .position(|t| flexible_name_match(&gl, t))
    }
}

/// Inverse-document-frequency marker weight `ln(C / df)`: a gene claimed by
/// all `C` types gets weight 0 (removed from scoring), a type-exclusive gene
/// the maximum `ln(C)`.
#[allow(dead_code)] // consumed by downstream crates (geu, senna), not the data-beans bin
#[must_use]
pub fn idf_weight(n_types: usize, df: usize) -> f32 {
    (n_types as f32 / df.max(1) as f32).ln()
}

/// Match names by substring queries and return matched indices and names
///
/// # Arguments
/// * `all_names` - All available names to search through
/// * `queries` - Substring queries to match against
/// * `entity_type` - Description of what's being matched (e.g., "column", "row") for error messages
///
/// # Returns
/// A tuple of (matched_indices, matched_names)
pub fn match_by_substring(
    all_names: &[Box<str>],
    queries: &[Box<str>],
    entity_type: &str,
) -> anyhow::Result<(Vec<usize>, Vec<Box<str>>)> {
    let mut matched_indices = Vec::new();

    for query in queries.iter() {
        for (idx, name) in all_names.iter().enumerate() {
            if name.contains(query.as_ref()) {
                matched_indices.push(idx);
            }
        }
    }

    if matched_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No {} names matched the provided queries",
            entity_type
        ));
    }

    let matched_names: Vec<Box<str>> = matched_indices
        .iter()
        .map(|&i| all_names[i].clone())
        .collect();

    Ok((matched_indices, matched_names))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<Box<str>> {
        v.iter().map(|s| Box::from(*s)).collect()
    }

    #[test]
    fn gene_index_tiers_and_idf() {
        let dict = names(&["ENSG00000153563_CD8A", "MS4A1", "A_FOO", "FOO"]);
        let idx = GeneIndex::build(&dict);

        // exact (case-insensitive)
        assert_eq!(idx.match_gene("ms4a1"), Some(1));
        // symbol = last `_`-segment
        assert_eq!(idx.match_gene("CD8A"), Some(0));
        // exact preferred over an earlier-indexed suffix match (the refinement)
        assert_eq!(idx.match_gene("FOO"), Some(3));
        // unmatched
        assert_eq!(idx.match_gene("ZZZ9"), None);

        // IDF: ubiquitous (df == C) → 0; exclusive → ln(C)
        assert_eq!(idf_weight(4, 4), 0.0);
        assert!(idf_weight(4, 1) > 1.38 && idf_weight(4, 1) < 1.39);
    }
}
