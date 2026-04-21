//! Disjoint-set union (DSU) with union-by-rank and path halving.
//!
//! Kept in-tree (not delegated to `petgraph`/`disjoint_sets`) because this
//! coarsener needs the per-root **size** (member count) for weighted centroid
//! merging — a field none of the ecosystem crates expose.

/// Union-Find (disjoint set) with path halving and union by rank.
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    #[inline]
    pub(crate) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Union the sets containing `a` and `b`; returns the surviving root.
    #[inline]
    pub(crate) fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        big
    }

    /// Member count of the set whose root is `x` (caller passes a root, typically
    /// `uf.find(...)` — accepting any node index would require an extra find).
    #[inline]
    pub(crate) fn size(&self, x: usize) -> usize {
        self.size[x]
    }

    /// Direct parent pointer — valid as a final representative only after
    /// `flatten` has been called. Used in the multilevel extraction hot loop
    /// to avoid per-call `find` after a batch of unions.
    #[inline]
    pub(crate) fn parent(&self, x: usize) -> usize {
        self.parent[x]
    }

    /// Path-compress every node so `parent[i]` is its final representative.
    /// Call after a batch of unions if subsequent lookups use [`parent`] directly.
    pub(crate) fn flatten(&mut self) {
        for i in 0..self.parent.len() {
            self.parent[i] = self.find(i);
        }
    }
}
