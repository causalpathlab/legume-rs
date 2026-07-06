use genomic_data::sam::CellBarcode;
use matrix_util::membership::Membership;
use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Cell membership data structure for filtering BAM records by cell barcode
/// Supports exact and prefix matching with caching for performance
pub struct CellMembership {
    /// Core membership from matrix-util
    inner: Membership,

    /// Cache for matched BAM barcodes (thread-safe for parallel processing)
    match_cache: Mutex<HashMap<std::sync::Arc<str>, Option<Box<str>>>>,

    /// Statistics tracking (lock-free)
    matched: AtomicUsize,
    total_checked: AtomicUsize,
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
            matched: AtomicUsize::new(0),
            total_checked: AtomicUsize::new(0),
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
                self.total_checked.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        self.total_checked.fetch_add(1, Ordering::Relaxed);

        // Check cache (short lock)
        {
            let cache = self.match_cache.lock().unwrap();
            if let Some(result) = cache.get(bam_bc) {
                if result.is_some() {
                    self.matched.fetch_add(1, Ordering::Relaxed);
                }
                return result.clone();
            }
        }

        // Compute outside lock (prefix scan can be O(n))
        let result = self.inner.get(bam_bc).map(Box::from);

        // Insert into cache
        self.match_cache
            .lock()
            .unwrap()
            .insert(bam_bc.clone(), result.clone());

        if result.is_some() {
            self.matched.fetch_add(1, Ordering::Relaxed);
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
        (
            self.matched.load(Ordering::Relaxed),
            self.total_checked.load(Ordering::Relaxed),
        )
    }

    /// Get all unique cell types in the membership
    pub fn cell_types(&self) -> Vec<Box<str>> {
        self.inner.unique_groups()
    }

    /// Create cell membership from cluster assignments
    pub fn from_clusters(barcodes: &[Box<str>], assignments: &[usize], allow_prefix: bool) -> Self {
        let pairs = barcodes
            .iter()
            .zip(assignments.iter())
            .map(|(barcode, &cluster)| {
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
            matched: AtomicUsize::new(0),
            total_checked: AtomicUsize::new(0),
        }
    }
}

#[cfg(test)]
mod tests;
