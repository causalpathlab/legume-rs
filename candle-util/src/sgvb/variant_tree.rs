use super::block_partition::BlockPartition;

/// Hierarchical tree structure for grouping variants in multi-level SuSiE.
///
/// Variants (leaves) are organized into groups at multiple levels.
/// Each level maps every variant to a (group, child_index) pair,
/// enabling efficient hierarchical softmax via gather operations.
///
/// One level of the variant tree.
#[derive(Debug, Clone)]
pub struct TreeLevel {
    /// Number of groups at this level
    pub num_groups: usize,
    /// Maximum number of children across all groups (for padding)
    pub max_children: usize,
    /// Validity mask: mask[group][child] = true if this child slot is occupied
    pub mask: Vec<Vec<bool>>,
    /// For each variant j: flat_path_indices[j] = group[j] * max_children + child_index[j]
    /// Used for efficient gather via index_select after reshaping logits to (L, G*C, k)
    pub flat_path_indices: Vec<usize>,
}

/// Hierarchical tree of variant groupings.
#[derive(Debug, Clone)]
pub struct VariantTree {
    /// Number of tree levels (not counting the leaves themselves)
    pub depth: usize,
    /// Number of leaf variants
    pub num_variants: usize,
    /// One TreeLevel per depth level
    pub levels: Vec<TreeLevel>,
}

impl VariantTree {
    /// Create a tree with equal-sized contiguous blocks.
    ///
    /// Delegates hierarchy construction to `BlockPartition::build_hierarchy`,
    /// then converts range-based partitions to per-variant assignments.
    ///
    /// # Arguments
    /// * `num_variants` - Total number of leaf variants (p)
    /// * `block_size` - Number of children per group at the finest level
    pub fn regular(num_variants: usize, block_size: usize) -> Self {
        assert!(block_size > 1, "block_size must be > 1");
        assert!(num_variants > 0, "num_variants must be > 0");

        let partitions = BlockPartition::build_hierarchy(num_variants, block_size);

        if partitions.is_empty() {
            // Single group containing everything
            let assignments = vec![vec![0usize; num_variants]];
            return Self::from_assignments(num_variants, &assignments);
        }

        // Convert range-based partitions to per-variant assignments.
        // Level 0 partition operates over p features directly.
        // Level d partition operates over num_blocks(d-1) items.
        // We compose through levels to get per-variant assignments.
        let mut assignments = Vec::with_capacity(partitions.len());

        // Finest level: directly over variants
        let mut assign = vec![0usize; num_variants];
        for (block_idx, range) in partitions[0].block_ranges.iter().enumerate() {
            for j in range.clone() {
                assign[j] = block_idx;
            }
        }
        assignments.push(assign);

        // Coarser levels: compose through previous
        for d in 1..partitions.len() {
            let prev_assign = &assignments[d - 1];
            let mut block_to_group = vec![0usize; partitions[d - 1].num_blocks()];
            for (group_idx, range) in partitions[d].block_ranges.iter().enumerate() {
                for b in range.clone() {
                    block_to_group[b] = group_idx;
                }
            }
            let assign: Vec<usize> = (0..num_variants)
                .map(|j| block_to_group[prev_assign[j]])
                .collect();
            assignments.push(assign);
        }

        // Add root level if top still has > 1 group
        let top_groups = partitions.last().unwrap().num_blocks();
        if top_groups > 1 {
            assignments.push(vec![0usize; num_variants]);
        }

        // Reverse so level 0 is coarsest (VariantTree convention)
        assignments.reverse();

        Self::from_assignments(num_variants, &assignments)
    }

    /// Create a tree from per-level group assignments.
    ///
    /// # Arguments
    /// * `assignments` - assignments[d][j] = group index of variant j at level d.
    ///   Level 0 is the coarsest (fewest groups), last level is finest.
    ///   Each level must have `num_variants` entries.
    pub fn from_assignments(num_variants: usize, assignments: &[Vec<usize>]) -> Self {
        assert!(!assignments.is_empty(), "need at least one level");
        for (d, a) in assignments.iter().enumerate() {
            assert_eq!(
                a.len(),
                num_variants,
                "level {} has {} assignments but expected {}",
                d,
                a.len(),
                num_variants
            );
        }

        let depth = assignments.len();
        let mut levels = Vec::with_capacity(depth);

        for d in 0..depth {
            let assign = &assignments[d];
            let num_groups = *assign.iter().max().unwrap() + 1;

            if d < depth - 1 {
                // Non-leaf level: children are the distinct next-level groups
                // within each current-level group.
                let next_assign = &assignments[d + 1];

                // For each group g at level d, collect the set of distinct
                // next-level group ids that its variants map to.
                let mut group_children: Vec<std::collections::BTreeSet<usize>> =
                    vec![std::collections::BTreeSet::new(); num_groups];
                for j in 0..num_variants {
                    group_children[assign[j]].insert(next_assign[j]);
                }

                // Map each (group, next_group) to a child index
                let mut group_child_map: Vec<std::collections::BTreeMap<usize, usize>> =
                    vec![std::collections::BTreeMap::new(); num_groups];
                let mut max_children = 0;
                for g in 0..num_groups {
                    for (idx, &next_g) in group_children[g].iter().enumerate() {
                        group_child_map[g].insert(next_g, idx);
                    }
                    max_children = max_children.max(group_children[g].len());
                }

                // Build mask
                let mut mask = vec![vec![false; max_children]; num_groups];
                for g in 0..num_groups {
                    for idx in 0..group_children[g].len() {
                        mask[g][idx] = true;
                    }
                }

                // Build flat path indices: for variant j, look up its child index
                // within its level-d group based on its level-(d+1) group
                let flat_path_indices: Vec<usize> = (0..num_variants)
                    .map(|j| {
                        let g = assign[j];
                        let child = group_child_map[g][&next_assign[j]];
                        g * max_children + child
                    })
                    .collect();

                levels.push(TreeLevel {
                    num_groups,
                    max_children,
                    mask,
                    flat_path_indices,
                });
            } else {
                // Leaf level: children are individual variants
                let mut group_child_count = vec![0usize; num_groups];
                let mut child_indices = vec![0usize; num_variants];

                for j in 0..num_variants {
                    let g = assign[j];
                    child_indices[j] = group_child_count[g];
                    group_child_count[g] += 1;
                }

                let max_children = *group_child_count.iter().max().unwrap();

                let mut mask = vec![vec![false; max_children]; num_groups];
                for j in 0..num_variants {
                    mask[assign[j]][child_indices[j]] = true;
                }

                let flat_path_indices: Vec<usize> = (0..num_variants)
                    .map(|j| assign[j] * max_children + child_indices[j])
                    .collect();

                levels.push(TreeLevel {
                    num_groups,
                    max_children,
                    mask,
                    flat_path_indices,
                });
            }
        }

        VariantTree {
            depth,
            num_variants,
            levels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_basic() {
        // 100 variants, block_size=10: root → 10 groups → 10 variants each
        let tree = VariantTree::regular(100, 10);
        assert_eq!(tree.num_variants, 100);
        // depth=2: root level + leaf level
        assert_eq!(tree.depth, 2);

        // Level 0 (root): 1 group with 10 children (the 10 groups)
        assert_eq!(tree.levels[0].num_groups, 1);
        assert_eq!(tree.levels[0].max_children, 10);

        // Level 1 (leaf): 10 groups, each with 10 children
        assert_eq!(tree.levels[1].num_groups, 10);
        assert_eq!(tree.levels[1].max_children, 10);

        // All paths should be valid
        for j in 0..100 {
            for level in &tree.levels {
                let idx = level.flat_path_indices[j];
                let g = idx / level.max_children;
                let c = idx % level.max_children;
                assert!(level.mask[g][c], "variant {} invalid", j);
            }
        }
    }

    #[test]
    fn test_regular_uneven() {
        // 103 variants with block_size=10:
        // Bottom: 11 groups (10 of 10, 1 of 3), 11 > 10 so another level
        // Mid: 2 groups (1 of 10, 1 of 1), 2 <= 10 so stop
        // Root added since 2 > 1
        let tree = VariantTree::regular(103, 10);
        assert!(tree.depth >= 2);

        // Leaf level should have 11 groups
        let finest = tree.levels.last().unwrap();
        assert_eq!(finest.num_groups, 11);

        // Last group should have 3 valid children
        let last_group = &finest.mask[10];
        let valid_count = last_group.iter().filter(|&&b| b).count();
        assert_eq!(valid_count, 3);

        // All paths valid
        for j in 0..103 {
            for level in &tree.levels {
                let idx = level.flat_path_indices[j];
                let g = idx / level.max_children;
                let c = idx % level.max_children;
                assert!(level.mask[g][c], "variant {} invalid at level", j);
            }
        }
    }

    #[test]
    fn test_regular_large() {
        // 2500 variants, block_size=50: root → 50 groups → 50 variants
        let tree = VariantTree::regular(2500, 50);
        assert_eq!(tree.depth, 2);
        assert_eq!(tree.levels[0].num_groups, 1);
        assert_eq!(tree.levels[0].max_children, 50);
        assert_eq!(tree.levels[1].num_groups, 50);
        assert_eq!(tree.levels[1].max_children, 50);

        // All paths valid
        for j in 0..2500 {
            for level in &tree.levels {
                let idx = level.flat_path_indices[j];
                let g = idx / level.max_children;
                let c = idx % level.max_children;
                assert!(level.mask[g][c]);
            }
        }
    }

    #[test]
    fn test_from_assignments() {
        // 10 variants, 2 groups of 5, then each group split into sub-groups
        let level0 = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]; // 2 groups
        let level1 = vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 3]; // 4 groups

        let tree = VariantTree::from_assignments(10, &[level0, level1]);
        assert_eq!(tree.depth, 2);
        assert_eq!(tree.levels[0].num_groups, 2);
        assert_eq!(tree.levels[1].num_groups, 4);

        // Level 0: group 0 has 2 children (sub-groups 0,1), group 1 has 2 children (sub-groups 2,3)
        assert_eq!(tree.levels[0].max_children, 2);

        // Level 1 (leaf): groups have 2, 3, 2, 3 children
        assert_eq!(tree.levels[1].max_children, 3);
        assert_eq!(tree.levels[1].mask[0].iter().filter(|&&b| b).count(), 2);
        assert_eq!(tree.levels[1].mask[1].iter().filter(|&&b| b).count(), 3);

        // All paths valid
        for j in 0..10 {
            for level in &tree.levels {
                let idx = level.flat_path_indices[j];
                let g = idx / level.max_children;
                let c = idx % level.max_children;
                assert!(level.mask[g][c], "variant {} invalid", j);
            }
        }
    }

    #[test]
    fn test_paths_unique_at_leaf() {
        // Every variant should have a unique path at the leaf level
        let tree = VariantTree::regular(25, 5);
        let leaf = tree.levels.last().unwrap();
        let mut seen = rustc_hash::FxHashSet::default();
        for j in 0..25 {
            assert!(
                seen.insert(leaf.flat_path_indices[j]),
                "duplicate at variant {}",
                j
            );
        }
    }
}
