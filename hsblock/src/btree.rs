//! Fixed-depth binary tree with O(1) lowest-common-ancestor (LCA) queries.
//!
//! The tree is stored in a 1-indexed heap layout:
//! - Root is node 1
//! - Left child of node i is 2*i, right child is 2*i + 1
//! - Parent of node i is i / 2
//! - Leaves occupy indices `num_leaves..2*num_leaves` (1-indexed)
//!
//! For a tree of depth D, there are `2^(D-1)` leaf clusters and `2^D - 1` total nodes.

/// Gamma-Poisson conjugate parameters stored at each tree node.
#[derive(Debug, Clone)]
pub struct GammaPoissonParam {
    /// log(shape) parameter
    pub ln_a0: f64,
    /// log(rate) parameter
    pub ln_b0: f64,
}

/// A fixed-depth binary tree encoding hierarchical block model structure.
///
/// Generic over the per-node parameter type `P`. Each node stores a value
/// of type `P`, which can be model-specific parameters or statistics.
#[derive(Debug, Clone)]
pub struct BTree<P: Clone> {
    depth: usize,
    num_leaves: usize,
    num_nodes: usize,
    /// Per-node data (1-indexed, params[0] unused)
    params: Vec<P>,
}

impl<P: Clone> BTree<P> {
    /// Create a new binary tree with the given depth and a default parameter for all nodes.
    ///
    /// * `depth` - Tree depth (must be >= 2). Number of leaf clusters K = 2^(depth-1).
    /// * `default_param` - Initial parameter value cloned to every node.
    pub fn new(depth: usize, default_param: P) -> Self {
        assert!(depth >= 2, "Tree depth must be at least 2");
        let num_leaves = 1 << (depth - 1); // 2^(D-1)
        let num_nodes = (1 << depth) - 1; // 2^D - 1

        // 1-indexed: allocate num_nodes + 1 elements, index 0 is unused
        let params = vec![default_param; num_nodes + 1];

        BTree {
            depth,
            num_leaves,
            num_nodes,
            params,
        }
    }

    /// Number of leaf clusters K = 2^(depth-1)
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Total number of tree nodes (internal + leaves)
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Tree depth
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Convert a cluster index (0-based) to a 1-indexed leaf position in the heap.
    ///
    /// Leaf nodes occupy positions `num_leaves..2*num_leaves` in the 1-indexed heap.
    #[inline]
    pub fn leaf_index(&self, cluster: usize) -> usize {
        debug_assert!(cluster < self.num_leaves);
        self.num_leaves + cluster
    }

    /// Compute the lowest common ancestor (LCA) of two leaf nodes.
    ///
    /// Takes cluster indices (0-based), returns a 1-indexed tree node.
    #[inline]
    pub fn lca(&self, cluster_i: usize, cluster_j: usize) -> usize {
        let mut a = self.leaf_index(cluster_i);
        let mut b = self.leaf_index(cluster_j);
        while a != b {
            if a > b {
                a >>= 1;
            } else {
                b >>= 1;
            }
        }
        a
    }

    /// Get a reference to the parameter at a tree node (1-indexed).
    #[inline]
    pub fn param(&self, node: usize) -> &P {
        debug_assert!(node >= 1 && node <= self.num_nodes);
        &self.params[node]
    }

    /// Get a mutable reference to the parameter at a tree node (1-indexed).
    #[inline]
    pub fn param_mut(&mut self, node: usize) -> &mut P {
        debug_assert!(node >= 1 && node <= self.num_nodes);
        &mut self.params[node]
    }

    /// Iterate over ancestors of a leaf, from the leaf's parent up to the root.
    ///
    /// Takes a cluster index (0-based), yields 1-indexed node indices.
    pub fn ancestors(&self, cluster: usize) -> AncestorIter {
        let leaf = self.leaf_index(cluster);
        AncestorIter { current: leaf >> 1 }
    }

    /// Parent of a 1-indexed node (returns 0 for root)
    #[inline]
    pub fn parent(node: usize) -> usize {
        node >> 1
    }

    /// Check if a 1-indexed node is a leaf
    #[inline]
    pub fn is_leaf(&self, node: usize) -> bool {
        node >= self.num_leaves
    }
}

// ─── Gamma-Poisson specific methods ───

impl BTree<GammaPoissonParam> {
    /// Create a new binary tree with Gamma-Poisson parameters.
    ///
    /// * `depth` - Tree depth (must be >= 2). Number of leaf clusters K = 2^(depth-1).
    /// * `init_a0` - Initial shape parameter a0 (stored as ln(a0))
    /// * `init_b0` - Initial rate parameter b0 (stored as ln(b0))
    pub fn with_gamma_poisson(depth: usize, init_a0: f64, init_b0: f64) -> Self {
        let default = GammaPoissonParam {
            ln_a0: init_a0.ln(),
            ln_b0: init_b0.ln(),
        };
        Self::new(depth, default)
    }

    /// Get the (a0, b0) parameters at a tree node (1-indexed).
    #[inline]
    pub fn node_params(&self, node: usize) -> (f64, f64) {
        debug_assert!(node >= 1 && node <= self.num_nodes());
        let p = &self.params[node];
        (p.ln_a0.exp(), p.ln_b0.exp())
    }

    /// Get the raw log parameters at a tree node (1-indexed).
    #[inline]
    pub fn node_ln_params(&self, node: usize) -> (f64, f64) {
        debug_assert!(node >= 1 && node <= self.num_nodes());
        let p = &self.params[node];
        (p.ln_a0, p.ln_b0)
    }

    /// Set the log parameters at a tree node (1-indexed).
    #[inline]
    pub fn set_node_ln_params(&mut self, node: usize, ln_a0: f64, ln_b0: f64) {
        debug_assert!(node >= 1 && node <= self.num_nodes());
        let p = &mut self.params[node];
        p.ln_a0 = ln_a0;
        p.ln_b0 = ln_b0;
    }
}

/// Iterator over ancestors of a tree node, walking from parent to root.
pub struct AncestorIter {
    current: usize,
}

impl Iterator for AncestorIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= 1 {
            let node = self.current;
            self.current >>= 1;
            Some(node)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_sizes() {
        let tree = BTree::with_gamma_poisson(2, 1.0, 1.0);
        assert_eq!(tree.num_leaves(), 2);
        assert_eq!(tree.num_nodes(), 3);

        let tree = BTree::with_gamma_poisson(3, 1.0, 1.0);
        assert_eq!(tree.num_leaves(), 4);
        assert_eq!(tree.num_nodes(), 7);

        let tree = BTree::with_gamma_poisson(4, 1.0, 1.0);
        assert_eq!(tree.num_leaves(), 8);
        assert_eq!(tree.num_nodes(), 15);
    }

    #[test]
    fn test_leaf_index() {
        let tree = BTree::with_gamma_poisson(3, 1.0, 1.0);
        // K=4 leaves, so leaves are at 1-indexed positions 4,5,6,7
        assert_eq!(tree.leaf_index(0), 4);
        assert_eq!(tree.leaf_index(1), 5);
        assert_eq!(tree.leaf_index(2), 6);
        assert_eq!(tree.leaf_index(3), 7);
    }

    #[test]
    fn test_lca_same_cluster() {
        let tree = BTree::with_gamma_poisson(3, 1.0, 1.0);
        // LCA of a cluster with itself is the leaf itself
        for c in 0..tree.num_leaves() {
            assert_eq!(tree.lca(c, c), tree.leaf_index(c));
        }
    }

    #[test]
    fn test_lca_siblings() {
        let tree = BTree::with_gamma_poisson(3, 1.0, 1.0);
        // Depth 3, K=4: tree structure (1-indexed):
        //         1
        //       /   \
        //      2     3
        //     / \   / \
        //    4   5 6   7   <- leaves (clusters 0,1,2,3)

        // Siblings share a direct parent
        assert_eq!(tree.lca(0, 1), 2); // leaves 4,5 -> parent 2
        assert_eq!(tree.lca(2, 3), 3); // leaves 6,7 -> parent 3
    }

    #[test]
    fn test_lca_cross_subtree() {
        let tree = BTree::with_gamma_poisson(3, 1.0, 1.0);
        // Clusters in different subtrees -> LCA is root
        assert_eq!(tree.lca(0, 2), 1); // leaves 4,6 -> root 1
        assert_eq!(tree.lca(0, 3), 1); // leaves 4,7 -> root 1
        assert_eq!(tree.lca(1, 2), 1); // leaves 5,6 -> root 1
        assert_eq!(tree.lca(1, 3), 1); // leaves 5,7 -> root 1
    }

    #[test]
    fn test_lca_depth4() {
        let tree = BTree::with_gamma_poisson(4, 1.0, 1.0);
        // K=8 leaves at positions 8..15
        //              1
        //          /       \
        //        2           3
        //       / \         / \
        //      4   5       6   7
        //     /\ /\       /\ /\
        //    8 9 10 11  12 13 14 15

        assert_eq!(tree.lca(0, 1), 4); // leaves 8,9
        assert_eq!(tree.lca(2, 3), 5); // leaves 10,11
        assert_eq!(tree.lca(0, 2), 2); // leaves 8,10
        assert_eq!(tree.lca(0, 4), 1); // leaves 8,12 -> root
        assert_eq!(tree.lca(4, 5), 6); // leaves 12,13
        assert_eq!(tree.lca(6, 7), 7); // leaves 14,15
        assert_eq!(tree.lca(4, 6), 3); // leaves 12,14
    }

    #[test]
    fn test_ancestors() {
        let tree = BTree::with_gamma_poisson(3, 1.0, 1.0);
        // Leaf 0 (position 4): ancestors are 2, 1
        let ancestors: Vec<usize> = tree.ancestors(0).collect();
        assert_eq!(ancestors, vec![2, 1]);

        // Leaf 3 (position 7): ancestors are 3, 1
        let ancestors: Vec<usize> = tree.ancestors(3).collect();
        assert_eq!(ancestors, vec![3, 1]);
    }

    #[test]
    fn test_node_params() {
        let tree = BTree::with_gamma_poisson(2, 2.0, 3.0);
        let (a0, b0) = tree.node_params(1);
        assert!((a0 - 2.0).abs() < 1e-10);
        assert!((b0 - 3.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Tree depth must be at least 2")]
    fn test_depth_too_small() {
        BTree::with_gamma_poisson(1, 1.0, 1.0);
    }

    #[test]
    fn test_generic_btree_with_custom_type() {
        #[derive(Debug, Clone, PartialEq)]
        struct MyParam {
            value: i32,
        }

        let mut tree = BTree::new(2, MyParam { value: 0 });
        assert_eq!(tree.num_leaves(), 2);
        assert_eq!(tree.num_nodes(), 3);

        // Write and read back
        tree.param_mut(1).value = 42;
        assert_eq!(tree.param(1).value, 42);
        assert_eq!(tree.param(2).value, 0); // untouched

        // Structural methods still work
        assert_eq!(tree.lca(0, 1), 1);
        assert_eq!(tree.leaf_index(0), 2);
        assert!(tree.is_leaf(2));
        assert!(!tree.is_leaf(1));
    }
}
