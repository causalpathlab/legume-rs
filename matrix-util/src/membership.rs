//! Membership mapping utilities for grouping items by category.
//!
//! Supports reading membership files (TSV/CSV) and matching with prefix support.

use crate::common_io::{read_lines_of_words_delim, ReadLinesOut};
use log::info;
use rustc_hash::FxHashMap as HashMap;

/// A membership mapping from keys to groups/categories.
///
/// Supports exact and prefix matching for flexible key lookup.
#[derive(Clone)]
pub struct Membership {
    /// Direct key -> group mapping
    map: HashMap<Box<str>, Box<str>>,
    /// Base key (before delimiter) -> group mapping for delimiter-based matching
    base_map: HashMap<Box<str>, Box<str>>,
    /// Keys stored for prefix matching
    keys: Vec<Box<str>>,
    /// Whether prefix matching is enabled
    allow_prefix: bool,
    /// Delimiter for extracting base keys (e.g., "@")
    delimiter: Option<char>,
}

/// Statistics about membership matching
#[derive(Debug, Default, Clone)]
pub struct MatchStats {
    pub exact: usize,
    pub base_key: usize,
    pub prefix: usize,
    pub unmatched: usize,
}

impl MatchStats {
    pub fn total_matched(&self) -> usize {
        self.exact + self.base_key + self.prefix
    }

    pub fn total(&self) -> usize {
        self.exact + self.base_key + self.prefix + self.unmatched
    }
}

/// Extract base key by splitting on delimiter and taking the first part
fn extract_base_key(key: &str, delimiter: Option<char>) -> Box<str> {
    match delimiter {
        Some(d) => key.split(d).next().unwrap_or(key).into(),
        None => key.into(),
    }
}

impl Membership {
    /// Create a new empty membership
    pub fn new(allow_prefix: bool) -> Self {
        Self {
            map: HashMap::default(),
            base_map: HashMap::default(),
            keys: Vec::new(),
            allow_prefix,
            delimiter: None,
        }
    }

    /// Create membership from key-value pairs
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = (Box<str>, Box<str>)>,
        allow_prefix: bool,
    ) -> Self {
        let mut map = HashMap::default();
        let mut keys = Vec::new();

        for (key, value) in pairs {
            keys.push(key.clone());
            map.insert(key, value);
        }

        Self {
            map,
            base_map: HashMap::default(),
            keys,
            allow_prefix,
            delimiter: None,
        }
    }

    /// Set delimiter for base key extraction (e.g., '@' to match "ACGT-1@suffix" with "ACGT-1@other")
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = Some(delimiter);
        // Rebuild base_map
        self.base_map = self
            .map
            .iter()
            .map(|(k, v)| (extract_base_key(k, self.delimiter), v.clone()))
            .collect();
        self
    }

    /// Load membership from file (TSV, CSV, or gzipped variants)
    ///
    /// # Arguments
    /// * `file_path` - Path to membership file
    /// * `key_col` - Column index for keys (0-based)
    /// * `value_col` - Column index for values (0-based)
    /// * `allow_prefix` - Enable prefix matching
    pub fn from_file(
        file_path: &str,
        key_col: usize,
        value_col: usize,
        allow_prefix: bool,
    ) -> anyhow::Result<Self> {
        let delim = detect_delimiter(file_path);

        let ReadLinesOut { lines, header: _ } = read_lines_of_words_delim(file_path, delim, -1)?;

        if lines.is_empty() {
            anyhow::bail!("Membership file is empty: {}", file_path);
        }

        let max_col = key_col.max(value_col);
        if lines[0].len() <= max_col {
            anyhow::bail!(
                "Membership file has {} columns but requested column index {}",
                lines[0].len(),
                max_col
            );
        }

        let mut map = HashMap::default();
        let mut keys = Vec::with_capacity(lines.len());

        for line in lines {
            if line.len() <= max_col {
                log::warn!("Skipping malformed line with {} columns", line.len());
                continue;
            }

            let key = line[key_col].clone();
            let value = line[value_col].clone();

            keys.push(key.clone());
            map.insert(key, value);
        }

        info!("Loaded {} entries from {}", map.len(), file_path);

        Ok(Self {
            map,
            base_map: HashMap::default(),
            keys,
            allow_prefix,
            delimiter: None,
        })
    }

    /// Load membership from file with single column (values only)
    ///
    /// Keys will be assigned positionally from the provided key list.
    pub fn from_file_positional(file_path: &str, keys: &[Box<str>]) -> anyhow::Result<Self> {
        let delim = detect_delimiter(file_path);

        let ReadLinesOut { lines, header: _ } = read_lines_of_words_delim(file_path, delim, -1)?;

        if lines.is_empty() {
            anyhow::bail!("Membership file is empty: {}", file_path);
        }

        if lines.len() != keys.len() {
            anyhow::bail!(
                "Membership file has {} lines but {} keys provided",
                lines.len(),
                keys.len()
            );
        }

        let mut map = HashMap::default();
        let mut key_vec = Vec::with_capacity(keys.len());

        for (key, line) in keys.iter().zip(lines.iter()) {
            let value = line[0].clone();
            key_vec.push(key.clone());
            map.insert(key.clone(), value);
        }

        info!("Loaded {} positional entries from {}", map.len(), file_path);

        Ok(Self {
            map,
            base_map: HashMap::default(),
            keys: key_vec,
            allow_prefix: false,
            delimiter: None,
        })
    }

    /// Get the group for a key using exact, base-key, or prefix matching
    pub fn get(&self, key: &str) -> Option<&str> {
        // Try exact match first
        if let Some(value) = self.map.get(key) {
            return Some(value.as_ref());
        }

        // Try base key match (if delimiter is set)
        if self.delimiter.is_some() {
            let base_key = extract_base_key(key, self.delimiter);
            if let Some(value) = self.base_map.get(&base_key) {
                return Some(value.as_ref());
            }
        }

        if !self.allow_prefix {
            return None;
        }

        // Try prefix match: stored key is prefix of query
        for stored_key in &self.keys {
            if key.starts_with(stored_key.as_ref()) {
                return self.map.get(stored_key).map(|v| v.as_ref());
            }
        }

        // Try reverse prefix: query is prefix of stored key
        for stored_key in &self.keys {
            if stored_key.starts_with(key) {
                return self.map.get(stored_key).map(|v| v.as_ref());
            }
        }

        None
    }

    /// Build a mapping for a list of query keys, with match statistics
    ///
    /// Returns a HashMap from query keys to their matched groups.
    /// Uses parallel processing for large key sets.
    pub fn match_keys(&self, query_keys: &[Box<str>]) -> (HashMap<Box<str>, Box<str>>, MatchStats) {
        use indicatif::ParallelProgressIterator;
        use rayon::prelude::*;

        // Match type: 1 = exact, 2 = base-key, 3 = prefix
        let matches: Vec<_> = query_keys
            .par_iter()
            .progress_count(query_keys.len() as u64)
            .map(|key| {
                // Try exact match first
                if let Some(value) = self.map.get(key) {
                    return Some((key.clone(), value.clone(), 1u8));
                }

                // Try base key match (if delimiter is set)
                if self.delimiter.is_some() {
                    let base_key = extract_base_key(key, self.delimiter);
                    if let Some(value) = self.base_map.get(&base_key) {
                        return Some((key.clone(), value.clone(), 2u8));
                    }
                }

                if !self.allow_prefix {
                    return None;
                }

                // Forward prefix: stored key is prefix of query
                for stored_key in &self.keys {
                    if key.starts_with(stored_key.as_ref()) {
                        if let Some(value) = self.map.get(stored_key) {
                            return Some((key.clone(), value.clone(), 3u8));
                        }
                    }
                }

                // Reverse prefix: query is prefix of stored key
                for stored_key in &self.keys {
                    if stored_key.starts_with(key.as_ref()) {
                        if let Some(value) = self.map.get(stored_key) {
                            return Some((key.clone(), value.clone(), 3u8));
                        }
                    }
                }

                None
            })
            .collect();

        // Aggregate results
        let mut result = HashMap::default();
        let mut stats = MatchStats::default();

        for m in matches {
            match m {
                Some((key, value, 1)) => {
                    result.insert(key, value);
                    stats.exact += 1;
                }
                Some((key, value, 2)) => {
                    result.insert(key, value);
                    stats.base_key += 1;
                }
                Some((key, value, 3)) => {
                    result.insert(key, value);
                    stats.prefix += 1;
                }
                _ => {
                    stats.unmatched += 1;
                }
            }
        }

        (result, stats)
    }

    /// Get all unique groups/values
    pub fn unique_groups(&self) -> Vec<Box<str>> {
        let mut groups: Vec<_> = self.map.values().cloned().collect();
        groups.sort();
        groups.dedup();
        groups
    }

    /// Number of entries in the membership
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if membership is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get sample keys for debugging
    pub fn sample_keys(&self, n: usize) -> Vec<&str> {
        self.keys.iter().take(n).map(|k| k.as_ref()).collect()
    }

    /// Iterate over all key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&Box<str>, &Box<str>)> {
        self.map.iter()
    }
}

/// Detect delimiter from file extension
pub fn detect_delimiter(file_path: &str) -> &'static str {
    if file_path.ends_with(".csv") || file_path.ends_with(".csv.gz") {
        ","
    } else {
        "\t"
    }
}

/// Gene-name → row-index resolver used across crates.
///
/// Gene row names in spatial/scRNA backends commonly arrive as compound
/// strings like `ENSG00000105329_TGFB1` (`{id}_{symbol}`) or
/// `TGFB1:ENSG00000105329`. A user-supplied gene list (e.g. an LR pair TSV,
/// a gene module file, or an external interaction network) almost never uses
/// the exact compound form, so we match generously:
///
/// - exact match on the full compound name wins first,
/// - when `delimiter` is set, every split component (on that char) is
///   registered as an alternate alias, so either `"ENSG…"` or `"TGFB1"`
///   resolves to the same index,
/// - when `allow_prefix` is `true`, the underlying `Membership` also falls
///   back to forward / reverse prefix matching.
///
/// Ambiguity: if two genes share an alias component, the last registered one
/// wins (consistent with `HashMap::insert`). Callers that need strict exact-
/// only matching should pass `delimiter = None, allow_prefix = false`.
#[derive(Clone)]
pub struct GeneIndexResolver {
    mem: Membership,
    n_genes: usize,
}

impl GeneIndexResolver {
    /// Build a resolver over `gene_names`. `delimiter` enables split-component
    /// aliasing (pass `Some('_')` for `ENSG..._SYMBOL` row names). Passing
    /// `None` disables aliasing and only the full name is indexed.
    pub fn build(gene_names: &[Box<str>], delimiter: Option<char>, allow_prefix: bool) -> Self {
        let mut pairs: Vec<(Box<str>, Box<str>)> = Vec::with_capacity(gene_names.len() * 2);
        for (i, name) in gene_names.iter().enumerate() {
            let idx_str: Box<str> = i.to_string().into_boxed_str();
            pairs.push((name.clone(), idx_str.clone()));
            if let Some(d) = delimiter {
                for part in name.split(d) {
                    if !part.is_empty() && part != name.as_ref() {
                        pairs.push((part.into(), idx_str.clone()));
                    }
                }
            }
        }
        Self {
            mem: Membership::from_pairs(pairs, allow_prefix),
            n_genes: gene_names.len(),
        }
    }

    /// Resolve a query gene name to its 0-based row index, if any.
    pub fn resolve(&self, query: &str) -> Option<usize> {
        self.mem.get(query)?.parse::<usize>().ok()
    }

    /// Number of genes registered.
    pub fn n_genes(&self) -> usize {
        self.n_genes
    }
}

/// Genomic-locus → row-index resolver, sibling of [`GeneIndexResolver`].
///
/// Locus row names appear in many flavors across pipelines:
///
/// - `chr1:1000-2000`, `1:1000-2000` (UCSC-style with optional `chr`)
/// - `chr1_1000_2000`, `1_1000_2000` (filename-safe underscore form)
/// - `chr1-1000-2000` (dash form some tools emit)
///
/// We canonicalize on read by stripping the `chr` prefix and folding any
/// of `:`, `_`, `-` to a single `_`, so `chr1:1000-2000` and `1_1000_2000`
/// both resolve to the same locus. Strand suffixes (`:+`, `:-`) attached
/// after the second coordinate get folded into the same separator scheme;
/// callers that care about strand should keep it out of the row name.
///
/// Ambiguity behavior: identical to [`GeneIndexResolver`] — last
/// registered wins on collision (consistent with `HashMap::insert`).
#[derive(Clone)]
pub struct LocusIndexResolver {
    mem: Membership,
    n_loci: usize,
}

/// Canonicalize a locus string: strip leading `chr` (case-insensitive)
/// and fold `:`, `-` to `_`. Empty strings pass through unchanged.
/// Allocations are avoided when the input is already canonical.
pub fn canon_locus(name: &str) -> Box<str> {
    if name.is_empty() {
        return name.into();
    }
    let stripped = name
        .strip_prefix("chr")
        .or_else(|| name.strip_prefix("CHR"))
        .or_else(|| name.strip_prefix("Chr"))
        .unwrap_or(name);
    if !stripped.contains(':') && !stripped.contains('-') {
        // Hot-path skip: most peak names already use `_`, so a per-row
        // canonicalize call (~36k × N files) avoids the chars/collect
        // allocation entirely.
        return stripped.into();
    }
    let folded: String = stripped
        .chars()
        .map(|c| if c == ':' || c == '-' { '_' } else { c })
        .collect();
    folded.into_boxed_str()
}

impl LocusIndexResolver {
    /// Build a resolver over `loci`. Each input name is registered both
    /// raw and in canonical form; lookups are tried raw first, then
    /// canonical. `allow_prefix` enables the same forward / reverse
    /// prefix fallback that [`GeneIndexResolver`] uses.
    pub fn build(loci: &[Box<str>], allow_prefix: bool) -> Self {
        let mut pairs: Vec<(Box<str>, Box<str>)> = Vec::with_capacity(loci.len() * 2);
        for (i, name) in loci.iter().enumerate() {
            let idx_str: Box<str> = i.to_string().into_boxed_str();
            pairs.push((name.clone(), idx_str.clone()));
            let canon = canon_locus(name);
            if canon.as_ref() != name.as_ref() {
                pairs.push((canon, idx_str));
            }
        }
        Self {
            mem: Membership::from_pairs(pairs, allow_prefix),
            n_loci: loci.len(),
        }
    }

    /// Resolve a query locus to its 0-based row index, if any. Tries
    /// raw match first, then canonical (chr-stripped + separator-folded).
    pub fn resolve(&self, query: &str) -> Option<usize> {
        if let Some(s) = self.mem.get(query) {
            if let Ok(i) = s.parse::<usize>() {
                return Some(i);
            }
        }
        let canon = canon_locus(query);
        self.mem.get(canon.as_ref())?.parse::<usize>().ok()
    }

    pub fn n_loci(&self) -> usize {
        self.n_loci
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "AAACCT\tgroup_A").unwrap();
        writeln!(file, "BBBCCT\tgroup_B").unwrap();
        writeln!(file, "CCCCCT\tgroup_A").unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_exact_match() {
        let file = create_test_file();
        let membership = Membership::from_file(file.path().to_str().unwrap(), 0, 1, false).unwrap();

        assert_eq!(membership.get("AAACCT"), Some("group_A"));
        assert_eq!(membership.get("BBBCCT"), Some("group_B"));
        assert_eq!(membership.get("AAACCT@suffix"), None); // No prefix matching
    }

    #[test]
    fn test_prefix_match() {
        let file = create_test_file();
        let membership = Membership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

        // Forward prefix: stored "AAACCT" is prefix of query "AAACCT@suffix"
        assert_eq!(membership.get("AAACCT@suffix"), Some("group_A"));

        // Reverse prefix: query "AAA" is prefix of stored "AAACCT"
        assert_eq!(membership.get("AAA"), Some("group_A"));
    }

    #[test]
    fn test_match_keys() {
        let file = create_test_file();
        let membership = Membership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

        let queries: Vec<Box<str>> = vec!["AAACCT".into(), "BBBCCT@1".into(), "UNKNOWN".into()];

        let (matched, stats) = membership.match_keys(&queries);

        assert_eq!(stats.exact, 1);
        assert_eq!(stats.prefix, 1);
        assert_eq!(stats.unmatched, 1);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_unique_groups() {
        let file = create_test_file();
        let membership = Membership::from_file(file.path().to_str().unwrap(), 0, 1, false).unwrap();

        let groups = membership.unique_groups();
        assert_eq!(groups, vec![Box::from("group_A"), Box::from("group_B")]);
    }

    #[test]
    fn gene_resolver_matches_full_and_suffix() {
        let names: Vec<Box<str>> = vec![
            "ENSG00000105329_TGFB1".into(),
            "ENSG00000121966_CXCR4".into(),
            "APOE".into(),
        ];
        let r = GeneIndexResolver::build(&names, Some('_'), false);
        // Full compound name resolves.
        assert_eq!(r.resolve("ENSG00000105329_TGFB1"), Some(0));
        // Symbol suffix resolves.
        assert_eq!(r.resolve("TGFB1"), Some(0));
        // Ensembl id prefix resolves.
        assert_eq!(r.resolve("ENSG00000121966"), Some(1));
        // Plain symbol without any delimiter-composition still resolves.
        assert_eq!(r.resolve("APOE"), Some(2));
        // Missing gene.
        assert_eq!(r.resolve("UNKNOWN"), None);
    }

    #[test]
    fn gene_resolver_exact_only_without_delimiter() {
        let names: Vec<Box<str>> = vec!["ENSG00000105329_TGFB1".into()];
        let r = GeneIndexResolver::build(&names, None, false);
        assert_eq!(r.resolve("ENSG00000105329_TGFB1"), Some(0));
        assert_eq!(r.resolve("TGFB1"), None);
    }

    #[test]
    fn locus_resolver_handles_chr_prefix_and_separators() {
        let loci: Vec<Box<str>> = vec![
            "chr1:1000-2000".into(),
            "chr2_3000_4000".into(),
            "X:5000-6000".into(),
        ];
        let r = LocusIndexResolver::build(&loci, false);
        // Exact match still wins.
        assert_eq!(r.resolve("chr1:1000-2000"), Some(0));
        assert_eq!(r.resolve("chr2_3000_4000"), Some(1));
        // chr-stripped + dash → underscore canonical form.
        assert_eq!(r.resolve("1_1000_2000"), Some(0));
        // Already-canonical query against a chr-prefixed row.
        assert_eq!(r.resolve("2_3000_4000"), Some(1));
        // X chromosome (no chr prefix in the row name).
        assert_eq!(r.resolve("X_5000_6000"), Some(2));
        assert_eq!(r.resolve("chrX_5000_6000"), Some(2));
        // Unknown locus.
        assert_eq!(r.resolve("Y:1-100"), None);
    }

    #[test]
    fn locus_resolver_case_variants_of_chr() {
        let loci: Vec<Box<str>> = vec!["CHR1:100-200".into()];
        let r = LocusIndexResolver::build(&loci, false);
        // All chr/CHR/Chr variants strip identically.
        assert_eq!(r.resolve("chr1:100-200"), Some(0));
        assert_eq!(r.resolve("Chr1_100_200"), Some(0));
        assert_eq!(r.resolve("1:100-200"), Some(0));
    }
}
