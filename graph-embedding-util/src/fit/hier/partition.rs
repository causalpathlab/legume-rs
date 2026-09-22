//! Hard gene→module partition and each unit's view through it.

use super::units::UnitTable;
use crate::fit::config::TrackSpec;

/// Shared bipartite module partition over genes and peaks.
///
/// One module index `m = 0..M−1`. Each gene and each peak has exactly one hard
/// label. `module_of` / `members` are the gene axis (kept for existing callers);
/// `module_of_peak` / `peak_members` are the peak axis. Peaks are **not** a
/// [`TrackSpec`] track of genes.
pub struct Partition {
    /// Module label per gene id (= [`Self::module_of_gene`]).
    pub module_of: Vec<u32>,
    /// Gene members per module, ascending (= [`Self::gene_members`]).
    pub members: Vec<Vec<u32>>,
    /// Module label per peak id.
    pub module_of_peak: Vec<u32>,
    /// Peak members per module, ascending.
    pub peak_members: Vec<Vec<u32>>,
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
    /// Gene-only labels (`labels.len() == n_genes`); peak side is empty.
    pub fn from_labels(labels: &[u32], n_modules: usize) -> Self {
        Self::from_gene_peak_labels(labels, &[], n_modules)
    }

    /// Hard labels for genes and peaks into a shared module index.
    ///
    /// `gene_labels.len()` is `n_genes`; `peak_labels.len()` is `n_peaks`.
    /// Each label must be `< n_modules`. Members of each module are sorted.
    pub fn from_gene_peak_labels(
        gene_labels: &[u32],
        peak_labels: &[u32],
        n_modules: usize,
    ) -> Self {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_modules];
        for (g, &m) in gene_labels.iter().enumerate() {
            members[m as usize].push(g as u32);
        }
        let mut peak_members: Vec<Vec<u32>> = vec![Vec::new(); n_modules];
        for (p, &m) in peak_labels.iter().enumerate() {
            peak_members[m as usize].push(p as u32);
        }
        for m in &mut members {
            m.sort_unstable();
        }
        for m in &mut peak_members {
            m.sort_unstable();
        }
        Self {
            module_of: gene_labels.to_vec(),
            members,
            module_of_peak: peak_labels.to_vec(),
            peak_members,
        }
    }

    pub fn n_modules(&self) -> usize {
        self.members.len()
    }

    /// Module label per gene id.
    #[must_use]
    pub fn module_of_gene(&self) -> &[u32] {
        &self.module_of
    }

    /// Gene members per module (ascending).
    #[must_use]
    pub fn gene_members(&self) -> &[Vec<u32>] {
        &self.members
    }

    /// Position of each gene in its module's member list.
    pub fn slot_of(&self) -> Vec<u32> {
        let mut slot = vec![0u32; self.module_of.len()];
        for m in &self.members {
            for (s, &g) in m.iter().enumerate() {
                slot[g as usize] = s as u32;
            }
        }
        slot
    }

    /// Position of each peak in its module's peak-member list.
    pub fn peak_slot_of(&self) -> Vec<u32> {
        let mut slot = vec![0u32; self.module_of_peak.len()];
        for m in &self.peak_members {
            for (s, &p) in m.iter().enumerate() {
                slot[p as usize] = s as u32;
            }
        }
        slot
    }
}

/// Each track's SUPPORT through the gene partition: the genes it has a row
/// for, and the modules those genes put it in.
///
/// A track's rows ARE its feature axis — a gene with no row on track `t` is
/// OUTSIDE that track's axis, not a gene the track observed zero times — so
/// track `t`'s softmaxes run over its support only:
///
/// ```text
/// S_t = { g : gene g has a row on track t }
/// M_t = { m : m ∩ S_t ≠ ∅ }
/// ```
///
/// A track that has a row for EVERY gene is `full` and carries no restriction
/// at all: it keeps the plain model's columns, empty modules included, exactly
/// as a one-track axis has them. `modules_of` / `slots_of` / `local_of` return
/// empty slices for such a track — the caller takes the unrestricted path and
/// reads [`Partition::members`] directly.
///
/// Built once per fit from the [`TrackSpec`] and the partition; nothing here
/// depends on a unit, a step or a plan.
pub struct TrackSupport {
    n_modules: usize,
    full: Vec<bool>,
    /// Per track, ascending; empty when that track is `full`.
    modules: Vec<Vec<u32>>,
    /// Per `(track, module)` at `t * M + m`, ascending; empty when `full`.
    slots: Vec<Vec<u32>>,
    /// Per `(track, module)`: full member slot → its position in `slots`, or
    /// `u32::MAX` when the track has no row for that member. Empty when `full`.
    local: Vec<Vec<u32>>,
}

impl TrackSupport {
    pub fn new(tracks: &TrackSpec, part: &Partition) -> Self {
        let (n_t, n_m) = (tracks.n_tracks(), part.n_modules());
        let n_g = part.module_of.len();
        let mut has = vec![false; n_t * n_g];
        for (&t, &g) in tracks.track_of_row.iter().zip(&tracks.gene_of_row) {
            has[t as usize * n_g + g as usize] = true;
        }
        let mut full = vec![false; n_t];
        let mut modules: Vec<Vec<u32>> = vec![Vec::new(); n_t];
        let mut slots: Vec<Vec<u32>> = vec![Vec::new(); n_t * n_m];
        let mut local: Vec<Vec<u32>> = vec![Vec::new(); n_t * n_m];
        for t in 0..n_t {
            let row = &has[t * n_g..(t + 1) * n_g];
            full[t] = row.iter().all(|&b| b);
            if full[t] {
                continue;
            }
            for (m, members) in part.members.iter().enumerate() {
                let mut sup = Vec::new();
                let mut loc = vec![u32::MAX; members.len()];
                for (j, &g) in members.iter().enumerate() {
                    if row[g as usize] {
                        loc[j] = sup.len() as u32;
                        sup.push(j as u32);
                    }
                }
                if !sup.is_empty() {
                    modules[t].push(m as u32);
                }
                slots[t * n_m + m] = sup;
                local[t * n_m + m] = loc;
            }
        }
        Self {
            n_modules: n_m,
            full,
            modules,
            slots,
            local,
        }
    }

    /// Does track `t` have a row for every gene? Then it carries no restriction.
    #[must_use]
    pub fn is_full(&self, t: usize) -> bool {
        self.full.get(t).copied().unwrap_or(true)
    }

    /// The modules track `t` is scored in, ascending. EMPTY for a `full` track,
    /// whose modules are `0..M`.
    #[must_use]
    pub fn modules_of(&self, t: usize) -> &[u32] {
        &self.modules[t]
    }

    /// Module `m`'s member slots track `t` has a row for, ascending. EMPTY for
    /// a `full` track, whose slots are `0..members[m].len()`.
    #[must_use]
    pub fn slots_of(&self, t: usize, m: usize) -> &[u32] {
        &self.slots[t * self.n_modules + m]
    }

    /// Full member slot → its position in [`Self::slots_of`], `u32::MAX` when
    /// the track has no row there. EMPTY for a `full` track (the identity).
    #[must_use]
    pub fn local_of(&self, t: usize, m: usize) -> &[u32] {
        &self.local[t * self.n_modules + m]
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

    /// The `(slot, count)` pairs of unit `u` in module `m` on track `t`; empty
    /// when the unit has no counts there. `by_module[u]` is sorted by
    /// `(track, module)`, so this is a binary search.
    #[must_use]
    pub fn counts_of(&self, u: usize, t: usize, m: usize) -> &[(u32, f32)] {
        let key = (t as u32, m as u32);
        match self.by_module[u].binary_search_by_key(&key, |(k, _)| *k) {
            Ok(i) => self.by_module[u][i].1.as_slice(),
            Err(_) => &[],
        }
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
