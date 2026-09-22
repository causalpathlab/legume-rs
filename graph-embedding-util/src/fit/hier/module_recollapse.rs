//! Merge-only re-collapse of one feature partition from **frozen unit profiles**.
//!
//! Whole modules merge (never split) by average-linkage cosine of their
//! members' unit profiles. Not `μ` fusion, not composed `ρ`.
//!
//! Average linkage needs no per-feature table: with `f̂` the unit-normalized
//! profile of feature `f`, the mean pairwise cosine between modules `i` and `j`
//! is
//!
//! ```text
//! sim(i, j) = (1 / |i||j|) Σ_{f∈i} Σ_{g∈j} f̂·ĝ = (S_i · S_j) / (|i| |j|),   S_m = Σ_{f∈m} f̂
//! ```
//!
//! so one `[M × observations]` table of module sums, built by streaming the
//! sparse unit counts twice (feature norms, then sums), is exact. Merging `i`
//! and `j` adds their sums and sizes. On axis 0 an observation is a
//! `(unit, track)` pair and a row's feature is its gene; on every other axis a
//! row IS its feature and the observation is the unit.

use super::partition::Partition;
use super::units::UnitTable;
use legume_numeric::candle::candle_core::{Device, Result as CResult, Tensor, Var};
use legume_numeric::candle::convert::to_host;

/// For each old module index, its new compact module id after re-collapse, or
/// [`Self::DROPPED`] for an old module that had no member (it has no new row).
#[derive(Debug, Clone)]
pub struct ModuleRecollapseMap {
    pub old_to_new: Vec<u32>,
    pub n_modules: usize,
}

impl ModuleRecollapseMap {
    /// The `old_to_new` entry of an old module with no member.
    pub const DROPPED: u32 = u32::MAX;

    /// Fewer modules than before: `μ` / bias rows must be compacted with the map.
    #[must_use]
    pub fn merged(&self) -> bool {
        self.n_modules < self.old_to_new.len()
    }
}

/// `(observation, feature, count)` view of one axis: axis 0 maps a row to its
/// `(track, gene)`, every other axis is a plain feature axis on one track.
fn for_each_count(units: &UnitTable, axis: usize, mut f: impl FnMut(usize, usize, f32)) {
    let ax = &units.axes[axis];
    let n_t = if axis == 0 { units.n_tracks() } else { 1 };
    for (u, (rows, counts)) in ax.feats.iter().zip(&ax.counts).enumerate() {
        for (&row, &c) in rows.iter().zip(counts) {
            let (t, feat) = if axis == 0 {
                (
                    units.tracks.track_of_row[row as usize] as usize,
                    units.tracks.gene_of_row[row as usize] as usize,
                )
            } else {
                (0, row as usize)
            };
            f(u * n_t + t, feat, c);
        }
    }
}

/// Merge-only collapse of whole modules on one axis: greedy average linkage
/// until the best pair's similarity drops below `min_cosine`. Modules with no
/// member are dropped. Returns `None` when the partition would come back with
/// the same module count.
pub fn recollapse_modules(
    units: &UnitTable,
    axis: usize,
    part: &Partition,
    min_cosine: f32,
) -> Option<(Partition, ModuleRecollapseMap)> {
    let from_m = part.n_modules();
    let n_feat = part.module_of.len();
    if from_m < 2 || min_cosine <= -1.0 {
        return None;
    }
    let n_t = if axis == 0 { units.n_tracks() } else { 1 };
    let n_obs = units.n_units() * n_t;

    // Pass 1: each feature's profile norm over the observations.
    let mut norm2 = vec![0f32; n_feat];
    for_each_count(units, axis, |_, f, c| norm2[f] += c * c);
    let inv_norm: Vec<f32> = norm2
        .iter()
        .map(|&n2| if n2 > 0.0 { n2.sqrt().recip() } else { 0.0 })
        .collect();
    // Pass 2: per-module sums of the unit-normalized profiles, `[M × n_obs]`.
    let mut sums = vec![0f32; from_m * n_obs];
    for_each_count(units, axis, |obs, f, c| {
        let m = part.module_of[f] as usize;
        sums[m * n_obs + obs] += c * inv_norm[f];
    });

    // Clusters start as the non-empty modules; a member with a zero profile
    // still counts in the size (its cosines are zero).
    struct Cluster {
        old: Vec<usize>,
        size: usize,
        sum: Vec<f32>,
    }
    let mut clusters: Vec<Cluster> = (0..from_m)
        .filter(|&m| !part.members[m].is_empty())
        .map(|m| Cluster {
            old: vec![m],
            size: part.members[m].len(),
            sum: sums[m * n_obs..(m + 1) * n_obs].to_vec(),
        })
        .collect();
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    while clusters.len() >= 2 {
        let mut best = (f32::NEG_INFINITY, 0usize, 1usize);
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let sim = dot(&clusters[i].sum, &clusters[j].sum)
                    / (clusters[i].size * clusters[j].size) as f32;
                if sim > best.0 {
                    best = (sim, i, j);
                }
            }
        }
        if best.0 < min_cosine {
            break;
        }
        let (_, i, j) = best;
        let mut cj = clusters.remove(j);
        let ci = &mut clusters[i];
        ci.old.append(&mut cj.old);
        ci.size += cj.size;
        for (a, b) in ci.sum.iter_mut().zip(&cj.sum) {
            *a += b;
        }
    }
    let n_modules = clusters.len();
    if n_modules == from_m {
        return None;
    }

    let mut old_to_new = vec![ModuleRecollapseMap::DROPPED; from_m];
    for (new_m, c) in clusters.iter().enumerate() {
        for &old_m in &c.old {
            old_to_new[old_m] = new_m as u32;
        }
    }
    let module_of: Vec<u32> = part
        .module_of
        .iter()
        .map(|&m| old_to_new[m as usize])
        .collect();
    Some((
        Partition::from_labels(&module_of, n_modules),
        ModuleRecollapseMap {
            old_to_new,
            n_modules,
        },
    ))
}

/// Average `μ` / `b_m` host rows when old modules map into fewer modules;
/// dropped old modules contribute nothing.
pub fn merge_module_rows_host(
    mu: &[f32],
    b_m: &[f32],
    n_m: usize,
    h: usize,
    map: &ModuleRecollapseMap,
) -> (Vec<f32>, Vec<f32>) {
    let n_new = map.n_modules;
    let mut mu_out = vec![0f32; n_new * h];
    let mut b_out = vec![0f32; n_new];
    let mut counts = vec![0usize; n_new];
    for old in 0..n_m {
        let new = map.old_to_new[old] as usize;
        if new >= n_new {
            continue;
        }
        counts[new] += 1;
        for k in 0..h {
            mu_out[new * h + k] += mu[old * h + k];
        }
        b_out[new] += b_m[old];
    }
    for new in 0..n_new {
        let c = counts[new].max(1) as f32;
        for k in 0..h {
            mu_out[new * h + k] /= c;
        }
        b_out[new] /= c;
    }
    (mu_out, b_out)
}

/// Replace `μ` and module biases after a re-collapse map (host-averaged rows).
pub fn merge_module_rows(
    mu: &mut Var,
    b_m: &mut Var,
    map: &ModuleRecollapseMap,
    dev: &Device,
) -> CResult<()> {
    if !map.merged() {
        return Ok(());
    }
    let h = mu.dims()[1];
    let n_m = mu.dims()[0];
    let mu_h = to_host(mu.as_tensor())?;
    let b_h = to_host(b_m.as_tensor())?;
    let (mu_new, b_new) = merge_module_rows_host(&mu_h, &b_h, n_m, h, map);
    *mu = Var::from_tensor(&Tensor::from_vec(mu_new, (map.n_modules, h), dev)?)?;
    *b_m = Var::from_tensor(&Tensor::from_vec(b_new, map.n_modules, dev)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "module_recollapse_tests.rs"]
mod module_recollapse_tests;
