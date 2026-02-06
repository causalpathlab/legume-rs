#![allow(dead_code)]

use genomic_data::sam::CellBarcode;
use fnv::FnvHashMap as HashMap;
use matrix_util::membership::Membership;
use std::sync::Mutex;

/// Cell membership data structure for filtering BAM records by cell barcode
/// Supports exact and prefix matching with caching for performance
pub struct CellMembership {
    /// Core membership from matrix-util
    inner: Membership,

    /// Cache for matched BAM barcodes (thread-safe for parallel processing)
    match_cache: Mutex<HashMap<Box<str>, Option<Box<str>>>>,

    /// Statistics tracking
    stats: Mutex<MatchStats>,
}

#[derive(Default)]
struct MatchStats {
    matched: usize,
    total_checked: usize,
}

impl CellMembership {
    /// Load cell membership from file (TSV, CSV, or gzipped variants)
    ///
    /// # Arguments
    /// * `file_path` - Path to membership file (.tsv, .csv, or .gz variants)
    /// * `barcode_col` - Column index for cell barcodes (0-based)
    /// * `celltype_col` - Column index for cell types (0-based)
    /// * `allow_prefix` - Enable prefix matching (membership barcodes as prefixes of BAM barcodes)
    pub fn from_file(
        file_path: &str,
        barcode_col: usize,
        celltype_col: usize,
        allow_prefix: bool,
    ) -> anyhow::Result<Self> {
        let inner = Membership::from_file(file_path, barcode_col, celltype_col, allow_prefix)?;

        Ok(Self {
            inner,
            match_cache: Mutex::new(HashMap::default()),
            stats: Mutex::new(MatchStats::default()),
        })
    }

    /// Check if a BAM barcode matches membership
    ///
    /// Uses two-tier lookup:
    /// 1. Try exact match first (O(1))
    /// 2. If that fails and prefix matching is enabled, try prefix match (O(n))
    /// 3. Cache results for performance
    ///
    /// # Returns
    /// * `Some(celltype)` if barcode matches membership
    /// * `None` if no match found or barcode is Missing
    pub fn matches_barcode(&self, barcode: &CellBarcode) -> Option<Box<str>> {
        let bam_bc = match barcode {
            CellBarcode::Barcode(bc) => bc,
            CellBarcode::Missing => {
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

        // Use inner membership for lookup
        let result = self.inner.get(bam_bc).cloned();

        // Cache the result
        self.match_cache
            .lock()
            .unwrap()
            .insert(bam_bc.clone(), result.clone());

        if result.is_some() {
            let mut stats = self.stats.lock().unwrap();
            stats.matched += 1;
        }

        result
    }

    /// Get total number of membership cells
    pub fn num_cells(&self) -> usize {
        self.inner.len()
    }

    /// Get match statistics for logging
    ///
    /// # Returns
    /// * `(matched, total_checked)` - Number of matched barcodes and total checked
    pub fn match_stats(&self) -> (usize, usize) {
        let stats = self.stats.lock().unwrap();
        (stats.matched, stats.total_checked)
    }

    /// Get all unique cell types in the membership
    pub fn cell_types(&self) -> Vec<Box<str>> {
        self.inner.unique_groups()
    }

    /// Check if a barcode matches a specific cell type
    pub fn matches_celltype(&self, barcode: &CellBarcode, target_celltype: &str) -> bool {
        self.matches_barcode(barcode)
            .map(|ct| ct.as_ref() == target_celltype)
            .unwrap_or(false)
    }

    /// Create cell membership from cluster assignments
    pub fn from_clusters(
        barcodes: &[Box<str>],
        assignments: &[usize],
        allow_prefix: bool,
    ) -> Self {
        let pairs = barcodes.iter().zip(assignments.iter()).map(|(barcode, &cluster)| {
            let celltype: Box<str> = format!("cluster_{}", cluster).into();
            (barcode.clone(), celltype)
        });

        let inner = Membership::from_pairs(pairs, allow_prefix);

        log::info!(
            "Created membership from {} cells in {} clusters",
            inner.len(),
            assignments.iter().max().map(|x| x + 1).unwrap_or(0)
        );

        Self {
            inner,
            match_cache: Mutex::new(HashMap::default()),
            stats: Mutex::new(MatchStats::default()),
        }
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
            false,
        )
        .unwrap();

        assert_eq!(membership.num_cells(), 3);

        let barcode = CellBarcode::Barcode("AAACCTGAGAAACCAT".into());
        assert_eq!(
            membership.matches_barcode(&barcode),
            Some("T_cell".into())
        );

        // No prefix matching
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
            true,
        )
        .unwrap();

        let barcode = CellBarcode::Barcode("AAACCTGAGCCCAATT".into());
        assert_eq!(
            membership.matches_barcode(&barcode),
            Some("B_cell".into())
        );

        let barcode_with_suffix = CellBarcode::Barcode("AAACCTGAGAAACCAT-1".into());
        assert_eq!(
            membership.matches_barcode(&barcode_with_suffix),
            Some("T_cell".into())
        );

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

        let result1 = membership.matches_barcode(&barcode);
        assert_eq!(result1, Some("T_cell".into()));

        let result2 = membership.matches_barcode(&barcode);
        assert_eq!(result2, Some("T_cell".into()));

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

        membership.matches_barcode(&barcode1);
        membership.matches_barcode(&barcode2);
        membership.matches_barcode(&missing);

        let (matched, total) = membership.match_stats();
        assert_eq!(matched, 1);
        assert_eq!(total, 3);
    }
}
