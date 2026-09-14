//! Hard gene→module partition and each unit's view through it.

use super::units::UnitTable;

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

pub struct UnitModules {
    pub q: Vec<f32>,
    pub n_um: Vec<f32>,
    pub by_module: Vec<Vec<(u32, Vec<(u32, f32)>)>>,
}

impl UnitModules {
    pub fn new(units: &UnitTable, part: &Partition) -> Self {
        let (n_u, m) = (units.n_units(), part.n_modules());
        let slot = part.slot_of();
        let mut n_um = vec![0f32; n_u * m];
        let mut by_module: Vec<Vec<(u32, Vec<(u32, f32)>)>> = Vec::with_capacity(n_u);
        // One bucket per module, indexed directly; `members` are sorted by gene
        // and `feats` are too, so each bucket's slots come out ascending.
        let mut buckets: Vec<Vec<(u32, f32)>> = vec![Vec::new(); m];
        for u in 0..n_u {
            for (&g, &c) in units.feats[u].iter().zip(&units.counts[u]) {
                let mm = part.module_of[g as usize] as usize;
                n_um[u * m + mm] += c;
                buckets[mm].push((slot[g as usize], c));
            }
            by_module.push(
                buckets
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(k, v)| (k as u32, std::mem::take(v)))
                    .collect(),
            );
        }
        let mut q = vec![0f32; n_u * m];
        for u in 0..n_u {
            let tot = units.total[u];
            for k in 0..m {
                q[u * m + k] = if tot > 0.0 {
                    n_um[u * m + k] / tot
                } else {
                    0.0
                };
            }
        }
        Self { q, n_um, by_module }
    }
}

#[cfg(test)]
#[path = "partition_tests.rs"]
mod partition_tests;
