#![allow(dead_code)]

use crate::data::sam::CellBarcode;
use fnv::{FnvHashMap as HashMap, FnvHashSet as HashSet};
use std::sync::Mutex;
use anyhow::{Result, Context};
use matrix_util::common_io::{read_lines_of_words_delim, ReadLinesOut};

/// Cell membership data structure for filtering BAM records by cell barcode
/// Supports exact and prefix matching with caching for performance
pub struct CellMembership {
    // Fast exact lookup (O(1))
    exact_barcodes: HashSet<Box<str>>,

    // For prefix matching when exact fails
    prefix_barcodes: Vec<Box<str>>,

    // Map barcode -> cell type
    barcode_to_celltype: HashMap<Box<str>, Box<str>>,

    // Cache for matched BAM barcodes (thread-safe for parallel processing)
    match_cache: Mutex<HashMap<Box<str>, Option<Box<str>>>>,

    // Statistics tracking
    stats: Mutex<MatchStats>,

    // Configuration
    allow_prefix_matching: bool,
}

#[derive(Default)]
struct MatchStats {
    matched: usize,
    total_checked: usize,
}

impl CellMembership {
    /// Load cell membership from file (TSV, CSV, or Parquet)
    ///
    /// # Arguments
    /// * `file_path` - Path to membership file (.tsv, .csv, .parquet, or .gz variants)
    /// * `barcode_col` - Column index for cell barcodes (0-based)
    /// * `celltype_col` - Column index for cell types (0-based)
    /// * `allow_prefix` - Enable prefix matching (membership barcodes as prefixes of BAM barcodes)
    ///
    /// # Returns
    /// * `Ok(CellMembership)` on success
    /// * `Err` if file cannot be read or parsed
    pub fn from_file(
        file_path: &str,
        barcode_col: usize,
        celltype_col: usize,
        allow_prefix: bool,
    ) -> Result<Self> {
        // Determine delimiter from file extension
        let delim = if file_path.ends_with(".csv") || file_path.ends_with(".csv.gz") {
            ","
        } else {
            "\t"
        };

        // Load file using matrix-util
        let ReadLinesOut { lines, header: _ } = read_lines_of_words_delim(file_path, delim, -1)
            .with_context(|| format!("Failed to read membership file: {}", file_path))?;

        // Validate column indices
        if lines.is_empty() {
            anyhow::bail!("Membership file is empty: {}", file_path);
        }

        let max_col = barcode_col.max(celltype_col);
        if lines[0].len() <= max_col {
            anyhow::bail!(
                "Membership file has {} columns but requested column index {}",
                lines[0].len(),
                max_col
            );
        }

        // Build data structures
        let mut exact_barcodes = HashSet::default();
        let mut prefix_barcodes = Vec::new();
        let mut barcode_to_celltype = HashMap::default();

        for line in lines {
            if line.len() <= max_col {
                log::warn!("Skipping malformed line with {} columns", line.len());
                continue;
            }

            let barcode = line[barcode_col].clone();
            let celltype = line[celltype_col].clone();

            exact_barcodes.insert(barcode.clone());
            prefix_barcodes.push(barcode.clone());
            barcode_to_celltype.insert(barcode, celltype);
        }

        log::info!(
            "Loaded {} unique cell barcodes from {}",
            exact_barcodes.len(),
            file_path
        );

        Ok(Self {
            exact_barcodes,
            prefix_barcodes,
            barcode_to_celltype,
            match_cache: Mutex::new(HashMap::default()),
            stats: Mutex::new(MatchStats::default()),
            allow_prefix_matching: allow_prefix,
        })
    }

    /// Check if a BAM barcode matches membership
    ///
    /// Uses two-tier lookup:
    /// 1. Try exact match first (O(1))
    /// 2. If that fails and prefix matching is enabled, try prefix match (O(n))
    /// 3. Cache results for performance
    ///
    /// # Arguments
    /// * `barcode` - Cell barcode from BAM file
    ///
    /// # Returns
    /// * `Some(celltype)` if barcode matches membership
    /// * `None` if no match found or barcode is Missing
    pub fn matches_barcode(&self, barcode: &CellBarcode) -> Option<Box<str>> {
        let bam_bc = match barcode {
            CellBarcode::Barcode(bc) => bc,
            CellBarcode::Missing => {
                // Update stats but don't cache Missing
                let mut stats = self.stats.lock().unwrap();
                stats.total_checked += 1;
                return None;
            }
        };

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_checked += 1;
        }

        // Check cache first
        {
            let cache = self.match_cache.lock().unwrap();
            if let Some(result) = cache.get(bam_bc) {
                if result.is_some() {
                    let mut stats = self.stats.lock().unwrap();
                    stats.matched += 1;
                }
                return result.clone();
            }
        }

        // Try exact match first (O(1))
        if self.exact_barcodes.contains(bam_bc) {
            let result = self.barcode_to_celltype.get(bam_bc).cloned();
            self.match_cache
                .lock()
                .unwrap()
                .insert(bam_bc.clone(), result.clone());
            if result.is_some() {
                let mut stats = self.stats.lock().unwrap();
                stats.matched += 1;
            }
            return result;
        }

        // Fall back to prefix matching if enabled (O(n))
        if self.allow_prefix_matching {
            for prefix in &self.prefix_barcodes {
                if bam_bc.starts_with(prefix.as_ref()) {
                    let result = self.barcode_to_celltype.get(prefix).cloned();
                    self.match_cache
                        .lock()
                        .unwrap()
                        .insert(bam_bc.clone(), result.clone());
                    if result.is_some() {
                        let mut stats = self.stats.lock().unwrap();
                        stats.matched += 1;
                    }
                    return result;
                }
            }
        }

        // No match found - cache the negative result
        self.match_cache.lock().unwrap().insert(bam_bc.clone(), None);
        None
    }

    /// Get total number of membership cells
    pub fn num_cells(&self) -> usize {
        self.exact_barcodes.len()
    }

    /// Get match statistics for logging
    ///
    /// # Returns
    /// * `(matched, total_checked)` - Number of matched barcodes and total checked
    pub fn match_stats(&self) -> (usize, usize) {
        let stats = self.stats.lock().unwrap();
        (stats.matched, stats.total_checked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_membership_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "AAACCTGAGAAACCAT\tT_cell").unwrap();
        writeln!(file, "AAACCTGAGCCCAATT\tB_cell").unwrap();
        writeln!(file, "AAACCTGCATACTCTT\tMonocyte").unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_exact_matching() {
        let file = create_test_membership_file();
        let membership = CellMembership::from_file(
            file.path().to_str().unwrap(),
            0,
            1,
            false, // exact matching only
        )
        .unwrap();

        assert_eq!(membership.num_cells(), 3);

        // Test exact match
        let barcode = CellBarcode::Barcode("AAACCTGAGAAACCAT".into());
        assert_eq!(
            membership.matches_barcode(&barcode),
            Some("T_cell".into())
        );

        // Test no match (prefix would match but exact matching is disabled)
        let barcode_with_suffix = CellBarcode::Barcode("AAACCTGAGAAACCAT-1".into());
        assert_eq!(membership.matches_barcode(&barcode_with_suffix), None);
    }

    #[test]
    fn test_prefix_matching() {
        let file = create_test_membership_file();
        let membership = CellMembership::from_file(
            file.path().to_str().unwrap(),
            0,
            1,
            true, // enable prefix matching
        )
        .unwrap();

        // Test exact match still works
        let barcode = CellBarcode::Barcode("AAACCTGAGCCCAATT".into());
        assert_eq!(
            membership.matches_barcode(&barcode),
            Some("B_cell".into())
        );

        // Test prefix match with suffix
        let barcode_with_suffix = CellBarcode::Barcode("AAACCTGAGAAACCAT-1".into());
        assert_eq!(
            membership.matches_barcode(&barcode_with_suffix),
            Some("T_cell".into())
        );

        // Test no match
        let unknown = CellBarcode::Barcode("ZZZZZZZZZZZZZZZ".into());
        assert_eq!(membership.matches_barcode(&unknown), None);
    }

    #[test]
    fn test_missing_barcode() {
        let file = create_test_membership_file();
        let membership =
            CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

        let missing = CellBarcode::Missing;
        assert_eq!(membership.matches_barcode(&missing), None);
    }

    #[test]
    fn test_caching() {
        let file = create_test_membership_file();
        let membership =
            CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

        let barcode = CellBarcode::Barcode("AAACCTGAGAAACCAT-1".into());

        // First call - should cache the result
        let result1 = membership.matches_barcode(&barcode);
        assert_eq!(result1, Some("T_cell".into()));

        // Second call - should use cache
        let result2 = membership.matches_barcode(&barcode);
        assert_eq!(result2, Some("T_cell".into()));

        // Verify both returned the same value
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_stats() {
        let file = create_test_membership_file();
        let membership =
            CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

        let barcode1 = CellBarcode::Barcode("AAACCTGAGAAACCAT".into());
        let barcode2 = CellBarcode::Barcode("UNKNOWN".into());
        let missing = CellBarcode::Missing;

        membership.matches_barcode(&barcode1); // match
        membership.matches_barcode(&barcode2); // no match
        membership.matches_barcode(&missing); // missing

        let (matched, total) = membership.match_stats();
        assert_eq!(matched, 1);
        assert_eq!(total, 3);
    }
}
