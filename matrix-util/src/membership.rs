//! Membership mapping utilities for grouping items by category.
//!
//! Supports reading membership files (TSV/CSV) and matching with prefix support.

use crate::common_io::{read_lines_of_words_delim, ReadLinesOut};
use fnv::FnvHashMap as HashMap;
use log::info;

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

        assert_eq!(
            membership.get("AAACCT").map(|s| s.as_ref()),
            Some("group_A")
        );
        assert_eq!(
            membership.get("BBBCCT").map(|s| s.as_ref()),
            Some("group_B")
        );
        assert_eq!(membership.get("AAACCT@suffix"), None); // No prefix matching
    }

    #[test]
    fn test_prefix_match() {
        let file = create_test_file();
        let membership = Membership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

        // Forward prefix: stored "AAACCT" is prefix of query "AAACCT@suffix"
        assert_eq!(
            membership.get("AAACCT@suffix").map(|s| s.as_ref()),
            Some("group_A")
        );

        // Reverse prefix: query "AAA" is prefix of stored "AAACCT"
        assert_eq!(membership.get("AAA").map(|s| s.as_ref()), Some("group_A"));
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
}
