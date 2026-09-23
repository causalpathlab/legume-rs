//! Phase 2 on a **collapsed** feature axis: every module-only module becomes
//! one row.
//!
//! A module-only feature's row is its module's row, `e_g = μ_m`, and its bias
//! `b_g` its share. So for any `θ` the module's features score
//! `⟨θ, μ_m⟩ + b_g`, and
//!
//! ```text
//! lse_{g ∈ m}(⟨θ, μ_m⟩ + b_g) = ⟨θ, μ_m⟩ + lse_{g ∈ m} b_g
//! Σ_{g ∈ m} n_g (⟨θ, μ_m⟩ + b_g) = n_m ⟨θ, μ_m⟩ + Σ_{g ∈ m} n_g b_g
//! ```
//!
//! One row with embedding `μ_m`, bias `lse_{g∈m} b_g` and the cell's summed
//! count `n_m` therefore gives the same log-partition, and a data term off by
//! a constant in `θ`: the per-cell solve, `θ` and the intercept are exactly
//! those of the full axis. Residual rows keep a row of their own. Counts are
//! batch-folded BEFORE they are summed, since the fold is per feature.

/// Full feature row → collapsed row.
#[derive(Clone, Debug)]
pub struct RowCollapse {
    /// `[n_features]`, each full row's collapsed row.
    pub row_of: Vec<u32>,
    /// Collapsed rows: the residual rows plus one per occupied module.
    pub n_rows: usize,
}

impl RowCollapse {
    /// The collapse a saved row map describes: `row_of[g]` is full row `g`'s
    /// collapsed row, numbered densely from 0.
    pub fn from_row_map(row_of: Vec<u32>) -> Self {
        let n_rows = row_of.iter().copied().max().map_or(0, |r| r as usize + 1);
        Self { row_of, n_rows }
    }

    /// The collapse of a one-track axis with per-row `module_only` flags and
    /// module `labels`. Rows are numbered in order of first appearance.
    /// `None` when no row is module-only (nothing to collapse).
    pub fn from_modules(module_only: &[bool], labels: &[u32]) -> Option<Self> {
        if !module_only.iter().any(|&b| b) {
            return None;
        }
        let n_modules = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut module_row = vec![u32::MAX; n_modules];
        let mut next = 0u32;
        let row_of = module_only
            .iter()
            .zip(labels)
            .map(|(&mo, &m)| {
                let fresh = |next: &mut u32| {
                    let r = *next;
                    *next += 1;
                    r
                };
                if mo {
                    if module_row[m as usize] == u32::MAX {
                        module_row[m as usize] = fresh(&mut next);
                    }
                    module_row[m as usize]
                } else {
                    fresh(&mut next)
                }
            })
            .collect();
        Some(Self {
            row_of,
            n_rows: next as usize,
        })
    }

    /// The collapsed dictionary `([n_rows × h] row-major, [n_rows])`: a
    /// collapsed row's embedding is any member's (they are equal), its bias
    /// the log-sum-exp of the members' biases.
    pub fn reduce_dictionary(&self, feat: &[f32], b: &[f32], h: usize) -> (Vec<f32>, Vec<f32>) {
        let mut out = vec![0f32; self.n_rows * h];
        let mut seen = vec![false; self.n_rows];
        let mut mx = vec![f32::NEG_INFINITY; self.n_rows];
        for (g, &r) in self.row_of.iter().enumerate() {
            let r = r as usize;
            if !seen[r] {
                out[r * h..(r + 1) * h].copy_from_slice(&feat[g * h..(g + 1) * h]);
                seen[r] = true;
            }
            mx[r] = mx[r].max(b[g]);
        }
        let mut sum = vec![0f64; self.n_rows];
        for (g, &r) in self.row_of.iter().enumerate() {
            sum[r as usize] += f64::from(b[g] - mx[r as usize]).exp();
        }
        let bias = mx
            .iter()
            .zip(&sum)
            .map(|(&m, &s)| m + s.ln() as f32)
            .collect();
        (out, bias)
    }

    /// One cell's (already batch-folded) edges on the collapsed axis: counts
    /// summed per collapsed row, rows ascending.
    pub fn reduce_edges(&self, feats: &[u32], counts: &[f32]) -> (Vec<u32>, Vec<f32>) {
        let mut pairs: Vec<(u32, f32)> = feats
            .iter()
            .zip(counts)
            .map(|(&f, &c)| (self.row_of[f as usize], c))
            .collect();
        pairs.sort_unstable_by_key(|&(r, _)| r);
        let mut rows: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut sums: Vec<f32> = Vec::with_capacity(pairs.len());
        for (r, c) in pairs {
            if rows.last() == Some(&r) {
                *sums.last_mut().expect("paired with rows") += c;
            } else {
                rows.push(r);
                sums.push(c);
            }
        }
        (rows, sums)
    }
}

#[cfg(test)]
#[path = "collapse_tests.rs"]
mod collapse_tests;
