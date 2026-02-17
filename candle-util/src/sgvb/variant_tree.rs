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
    /// For depth=1 with block_size=50 and p=2500: 50 groups of 50.
    /// For depth=2: recursively splits groups into sub-blocks.
    ///
    /// # Arguments
    /// * `num_variants` - Total number of leaf variants (p)
    /// * `block_size` - Number of children per group at the finest level
    pub fn regular(num_variants: usize, block_size: usize) -> Self {
        assert!(block_size > 1, "block_size must be > 1");
        assert!(num_variants > 0, "num_variants must be > 0");

        let mut current_count = num_variants;

        // Build levels bottom-up: each level groups `current_count` items into blocks
        let mut level_assignments_stack = Vec::new();

        while current_count > block_size {
            let num_groups = current_count.div_ceil(block_size);
            // Assignment: item i -> group i / block_size
            let assignments: Vec<usize> = (0..current_count).map(|i| i / block_size).collect();
            level_assignments_stack.push((current_count, assignments));
            current_count = num_groups;
        }

        // If we still have > 1 items at the top, add a root level grouping them all
        if current_count > 1 {
            let assignments: Vec<usize> = (0..current_count).map(|_| 0).collect();
            level_assignments_stack.push((current_count, assignments));
        }

        // Now build tree levels from top to bottom
        // level_assignments_stack is bottom-up, we need top-down
        level_assignments_stack.reverse();

        // Convert stacked assignments into TreeLevels
        // We need to map variant-level indices through each level
        Self::build_from_stacked_assignments(num_variants, &level_assignments_stack)
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

    /// Build tree from stacked (count, assignments) pairs.
    /// Each pair describes grouping at one level of the hierarchy.
    fn build_from_stacked_assignments(
        num_variants: usize,
        stacked: &[(usize, Vec<usize>)],
    ) -> Self {
        if stacked.is_empty() {
            // Trivial: single level with one group containing everything
            let assignments = vec![vec![0usize; num_variants]];
            return Self::from_assignments(num_variants, &assignments);
        }

        // For a multi-level tree built from regular blocks, we need to compose
        // the assignments. stacked[0] groups the items at the top level,
        // stacked[1] groups items within those groups, etc.
        //
        // We need to convert this into per-variant assignments at each level.
        // stacked[0] has `stacked[0].0` items (which are groups from stacked[1])
        // The final stacked level has `num_variants` items.

        let depth = stacked.len();
        let mut per_variant_assignments = Vec::with_capacity(depth);

        if depth == 1 {
            // Simple case: one level of grouping directly over variants
            per_variant_assignments.push(stacked[0].1.clone());
        } else {
            // Multi-level: compose assignments
            // stacked[last] maps variants to groups at the finest level
            // stacked[last-1] maps those groups to coarser groups, etc.

            // Start from the finest level and propagate upward
            // stacked[d] maps items at level d+1 to groups at level d
            // For the finest level (last), items are variants
            let finest = &stacked[depth - 1].1; // maps variants to finest groups
            per_variant_assignments.push(finest.clone());

            // For coarser levels, compose: variant -> finest_group -> coarser_group -> ...
            for d in (0..depth - 1).rev() {
                let coarser = &stacked[d].1; // maps items at level d+1 to groups at level d
                                             // The items at level d+1 are the groups from level d+1
                                             // We need: for each variant j, find its group at level d
                                             // = coarser[variant_group_at_d+1[j]]
                let finer_assignments = per_variant_assignments.last().unwrap();

                // finer_assignments[j] = group of variant j at the next-finer level
                // coarser maps those groups to coarser groups
                let composed: Vec<usize> = (0..num_variants)
                    .map(|j| coarser[finer_assignments[j]])
                    .collect();
                per_variant_assignments.push(composed);
            }

            // Reverse so level 0 is coarsest
            per_variant_assignments.reverse();
        }

        Self::from_assignments(num_variants, &per_variant_assignments)
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
        let mut seen = std::collections::HashSet::new();
        for j in 0..25 {
            assert!(
                seen.insert(leaf.flat_path_indices[j]),
                "duplicate at variant {}",
                j
            );
        }
    }
}
