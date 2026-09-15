//! Hard gene→module partition and each unit's view through it.

use super::units::UnitTable;

/// One module per GENE (not per feature row): tracks of the same gene share a
/// module. `module_of` is indexed by gene id, `members[m]` lists that module's
/// gene ids in ascending order.
pub struct Partition {
    pub module_of: Vec<u32>,
    pub members: Vec<Vec<u32>>,
}

/// Hard labels from a soft membership `[D × M]`: the argmax column per row
/// (ties → the lowest index), `0` for an all-zero row.
#[must_use]
pub fn labels_from_membership(pi: &nalgebra::DMatrix<f32>) -> Vec<u32> {
    pi.row_iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..row.ncols() {
                let v = row[j];
                if v > best_val {
                    best_val = v;
                    best = j;
                }
            }
            best as u32
        })
        .collect()
}

impl Partition {
    /// `labels` is one module per GENE (`labels.len() == n_genes`).
    pub fn from_labels(labels: &[u32], n_modules: usize) -> Self {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_modules];
        for (g, &m) in labels.iter().enumerate() {
            members[m as usize].push(g as u32);
        }
        for m in &mut members {
            m.sort_unstable();
        }
        Self {
            module_of: labels.to_vec(),
            members,
        }
    }

    pub fn n_modules(&self) -> usize {
        self.members.len()
    }

    pub fn slot_of(&self) -> Vec<u32> {
        let mut slot = vec![0u32; self.module_of.len()];
        for m in &self.members {
            for (s, &g) in m.iter().enumerate() {
                slot[g as usize] = s as u32;
            }
        }
        slot
    }
}

/// One unit's buckets: per `(track, module)` key, that unit's `(slot, count)`
/// pairs in the module on that track.
pub type UnitBuckets = Vec<((u32, u32), Vec<(u32, f32)>)>;

/// Each unit's view through the gene partition, per TRACK.
///
/// Invariants: `q` and `n_um` are `[n_units × n_tracks × n_modules]`, indexed
/// by [`UnitModules::idx`]; `q[idx(u, t, ·)]` sums to 1 when unit `u` has any
/// counts on track `t` and is all-zero otherwise; `by_module[u]` is sorted by
/// `(track, module)` and holds only the `(t, m)` pairs the unit has counts in.
pub struct UnitModules {
    pub n_tracks: usize,
    pub n_modules: usize,
    /// `[n_units × n_tracks × n_modules]`, index `(u*T + t)*M + m`.
    pub q: Vec<f32>,
    pub n_um: Vec<f32>,
    /// Per unit, sorted by `(track, module)`: that unit's (slot, count) pairs
    /// in the module on that track. `slot` is the gene's position in
    /// `Partition::members[m]`.
    pub by_module: Vec<UnitBuckets>,
}

impl UnitModules {
    /// Flat index of `(unit, track, module)` into [`Self::q`] / [`Self::n_um`].
    #[must_use]
    pub fn idx(&self, u: usize, t: usize, m: usize) -> usize {
        (u * self.n_tracks + t) * self.n_modules + m
    }

    pub fn new(units: &UnitTable, part: &Partition) -> Self {
        let (n_u, m) = (units.n_units(), part.n_modules());
        let n_t = units.n_tracks();
        let slot = part.slot_of();
        let mut n_um = vec![0f32; n_u * n_t * m];
        let mut by_module: Vec<UnitBuckets> = Vec::with_capacity(n_u);
        // One bucket per (track, module), indexed directly as `t*M + m`, so the
        // kept entries come out sorted by `(track, module)`. Within a bucket,
        // `feats` is ascending and a track's rows follow its genes' order, so
        // the slots come out ascending too.
        let mut buckets: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_t * m];
        for u in 0..n_u {
            for (&row, &c) in units.feats[u].iter().zip(&units.counts[u]) {
                let t = units.tracks.track_of_row[row as usize] as usize;
                let g = units.tracks.gene_of_row[row as usize] as usize;
                let mm = part.module_of[g] as usize;
                n_um[(u * n_t + t) * m + mm] += c;
                buckets[t * m + mm].push((slot[g], c));
            }
            by_module.push(
                buckets
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(k, v)| (((k / m) as u32, (k % m) as u32), std::mem::take(v)))
                    .collect(),
            );
        }
        let mut q = vec![0f32; n_u * n_t * m];
        for u in 0..n_u {
            for t in 0..n_t {
                let tot = units.total_of(u, t);
                for k in 0..m {
                    let i = (u * n_t + t) * m + k;
                    q[i] = if tot > 0.0 { n_um[i] / tot } else { 0.0 };
                }
            }
        }
        Self {
            n_tracks: n_t,
            n_modules: m,
            q,
            n_um,
            by_module,
        }
    }
}

#[cfg(test)]
#[path = "partition_tests.rs"]
mod partition_tests;
