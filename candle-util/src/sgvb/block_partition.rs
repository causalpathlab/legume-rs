/// Block partition for recursive multilevel SuSiE.
///
/// Divides p features into contiguous blocks at one level of the hierarchy.
/// Used by `collapse_with_alpha` to define which features are grouped together
/// when collapsing the design matrix X into a coarser representation.
///
/// For genetics applications, blocks typically correspond to LD blocks or
/// fixed genomic windows. For general use, `regular()` creates equal sized
/// contiguous blocks, and `build_hierarchy()` recursively nests them until
/// the top level has at most `block_size` groups.
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct BlockPartition {
    /// Block boundaries: block b spans features block_ranges[b].start..block_ranges[b].end
    pub block_ranges: Vec<Range<usize>>,
}

impl BlockPartition {
    /// Number of blocks B.
    pub fn num_blocks(&self) -> usize {
        self.block_ranges.len()
    }

    /// Total features p at this level.
    pub fn num_features(&self) -> usize {
        self.block_ranges.last().map_or(0, |r| r.end)
    }

    /// Map each feature index to its block index. O(p).
    pub fn feature_to_block(&self) -> Vec<u32> {
        let p = self.num_features();
        let mut mapping = vec![0u32; p];
        for (b, range) in self.block_ranges.iter().enumerate() {
            for j in range.clone() {
                mapping[j] = b as u32;
            }
        }
        mapping
    }
}

impl BlockPartition {
    /// Create equal sized contiguous blocks.
    ///
    /// Last block may be smaller if p is not divisible by block_size.
    pub fn regular(num_features: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(num_features > 0, "num_features must be > 0");

        let num_blocks = num_features.div_ceil(block_size);
        let block_ranges: Vec<Range<usize>> = (0..num_blocks)
            .map(|b| {
                let start = b * block_size;
                let end = ((b + 1) * block_size).min(num_features);
                start..end
            })
            .collect();

        Self { block_ranges }
    }

    /// Create from explicit boundary positions.
    ///
    /// `boundaries` contains the start position of each block (must include 0).
    /// The last block extends to `num_features`.
    pub fn from_boundaries(boundaries: &[usize], num_features: usize) -> Self {
        assert!(!boundaries.is_empty(), "need at least one boundary");
        assert_eq!(boundaries[0], 0, "first boundary must be 0");

        let num_blocks = boundaries.len();
        let block_ranges: Vec<Range<usize>> = (0..num_blocks)
            .map(|b| {
                let start = boundaries[b];
                let end = if b + 1 < num_blocks {
                    boundaries[b + 1]
                } else {
                    num_features
                };
                start..end
            })
            .collect();

        Self { block_ranges }
    }

    /// Build a recursive hierarchy of partitions.
    ///
    /// Starting from `num_features`, repeatedly partitions into blocks of
    /// `block_size` until the number of groups fits in a single softmax
    /// (i.e., <= `block_size`). Returns an empty vec when p <= block_size
    /// (no collapse needed, caller uses a flat SusieVar).
    ///
    /// Example: p=1M, block_size=100
    ///   Level 0: 1M features -> 10K groups
    ///   Level 1: 10K features -> 100 groups
    ///   100 <= 100, done. Returns 2 partitions.
    ///
    /// Returns partitions from finest (level 0, over p features) to coarsest.
    pub fn build_hierarchy(num_features: usize, block_size: usize) -> Vec<Self> {
        assert!(block_size > 1, "block_size must be > 1");

        let mut levels = Vec::new();
        let mut current = num_features;

        while current > block_size {
            let partition = Self::regular(current, block_size);
            let next = partition.num_blocks();
            levels.push(partition);
            current = next;
        }

        levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_even() {
        let part = BlockPartition::regular(100, 10);
        assert_eq!(part.num_blocks(), 10);
        assert_eq!(part.num_features(), 100);
        assert_eq!(part.block_ranges[0], 0..10);
        assert_eq!(part.block_ranges[9], 90..100);
    }

    #[test]
    fn test_regular_uneven() {
        let part = BlockPartition::regular(103, 10);
        assert_eq!(part.num_blocks(), 11);
        assert_eq!(part.block_ranges[10], 100..103);
    }

    #[test]
    fn test_from_boundaries() {
        let part = BlockPartition::from_boundaries(&[0, 20, 50, 80], 100);
        assert_eq!(part.num_blocks(), 4);
        assert_eq!(part.block_ranges[0], 0..20);
        assert_eq!(part.block_ranges[1], 20..50);
        assert_eq!(part.block_ranges[2], 50..80);
        assert_eq!(part.block_ranges[3], 80..100);
    }

    #[test]
    fn test_build_hierarchy_two_levels() {
        // p=1000, block_size=100 → [1000→10 blocks] → 10 ≤ 100, done
        let levels = BlockPartition::build_hierarchy(1000, 100);
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].num_features(), 1000);
        assert_eq!(levels[0].num_blocks(), 10);
    }

    #[test]
    fn test_build_hierarchy_three_levels() {
        // p=10000, block_size=100 → [10000→100 blocks] → [100→1 block] → done
        let levels = BlockPartition::build_hierarchy(10000, 100);
        assert_eq!(levels.len(), 1); // 100 groups ≤ 100, so only 1 collapse level
    }

    #[test]
    fn test_build_hierarchy_deep() {
        // p=1000000, block_size=100
        // Level 0: 1M → 10000 blocks
        // Level 1: 10000 → 100 blocks
        // 100 ≤ 100, done → 2 levels
        let levels = BlockPartition::build_hierarchy(1_000_000, 100);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].num_features(), 1_000_000);
        assert_eq!(levels[0].num_blocks(), 10_000);
        assert_eq!(levels[1].num_features(), 10_000);
        assert_eq!(levels[1].num_blocks(), 100);
    }

    #[test]
    fn test_build_hierarchy_small_p() {
        // p=50, block_size=100 → no collapse needed
        let levels = BlockPartition::build_hierarchy(50, 100);
        assert!(levels.is_empty());
    }

    #[test]
    fn test_contiguous_coverage() {
        let part = BlockPartition::regular(57, 10);
        // Verify all features are covered exactly once
        let mut covered = vec![false; 57];
        for range in &part.block_ranges {
            for j in range.clone() {
                assert!(!covered[j], "feature {} covered twice", j);
                covered[j] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "not all features covered");
    }
}
