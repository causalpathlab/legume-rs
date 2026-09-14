//! Hard gene→module partition and each unit's view through it.

use super::units::UnitTable;

pub struct Partition {
    pub module_of: Vec<u32>,
    pub members: Vec<Vec<u32>>,
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
        for u in 0..n_u {
            // `feats` are sorted by gene, so group by module with a map keyed on module.
            let mut groups: Vec<(u32, Vec<(u32, f32)>)> = Vec::new();
            for (&g, &c) in units.feats[u].iter().zip(&units.counts[u]) {
                let mm = part.module_of[g as usize];
                n_um[u * m + mm as usize] += c;
                match groups.iter_mut().find(|(k, _)| *k == mm) {
                    Some((_, v)) => v.push((slot[g as usize], c)),
                    None => groups.push((mm, vec![(slot[g as usize], c)])),
                }
            }
            groups.sort_unstable_by_key(|(k, _)| *k);
            for (_, v) in &mut groups {
                v.sort_unstable_by_key(|&(s, _)| s);
            }
            by_module.push(groups);
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
