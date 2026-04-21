//! BBKNN + Poisson DC-SBM refinement over multi-level super-cell partitions.
//!
//! Given a hash-initialized hierarchy of super-cell → group mappings and a
//! `SuperCellLayout` (HNSW over centroids, batch assignments), refine each
//! level from coarsest to finest by proposing moves that keep each super-cell
//! under the same parent group (sibling-constrained) and are drawn from the
//! batch-balanced KNN neighborhood. Moves are scored by an IDF-weighted
//! Poisson-Gamma DC-SBM log-likelihood.
//!
//! Entry point: [`refine_assignments`].

#![allow(dead_code)]

use crate::collapse_data::SuperCellLayout;
use log::{debug, info};
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rustc_hash::FxHashMap as HashMap;

/// Additive floor to keep `ln(.)` finite when a block or feature has zero
/// mass. Matches the smoothing used in the existing pinto reference.
pub const LOG_EPS: f64 = 1e-9;

pub use crate::collapse_data::GeneSums;

////////////////////////////////////////////////////////////////////////////////
// Public configuration
////////////////////////////////////////////////////////////////////////////////

/// Feature representation used to score super-cell → block moves.
#[derive(Clone, Debug)]
pub enum ProfileSource {
    /// Sparse super-cell × gene counts (the existing `gene_sums` format).
    /// IDF weighting carries its standard "housekeeping gene down-weighting"
    /// semantics.
    Raw,
    /// Dense projection: `basis * indicator(sc)` per super-cell. `basis` is
    /// `proj_dim × num_cells`-shaped; the super-cell profile is the sum over
    /// its member cells' projection columns. IDF is skipped because the
    /// feature axis is no longer "genes".
    Projected { basis: DMatrix<f32> },
}

/// Parameters controlling the refinement pass.
#[derive(Clone, Debug)]
pub struct RefineParams {
    /// Gibbs sweeps per level (0 disables Gibbs; greedy still runs).
    pub num_gibbs: usize,
    /// Greedy sweeps per level (early-exits on zero moves).
    pub num_greedy: usize,
    /// Apply IDF reweighting on the profile (only meaningful for `Raw`).
    pub idf_weighting: bool,
    /// Seed for Gibbs RNG.
    pub seed: u64,
    /// Gibbs stagnation threshold (fraction of entities moving per sweep).
    /// Breaks early when three consecutive sweeps are below this bound.
    /// `0.0` disables early exit.
    pub gibbs_stagnation: f64,
    /// Feature source. Defaults to `Raw`.
    pub profile_source: ProfileSource,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            num_gibbs: 20,
            num_greedy: 10,
            idf_weighting: true,
            seed: 42,
            gibbs_stagnation: 0.005,
            profile_source: ProfileSource::Raw,
        }
    }
}

/// Per-level refined super-cell → group mapping.
///
/// `sc_to_group[level][sc_idx] = refined group id at that level`. Level order
/// matches the existing collapse convention (`level_dims[0]` = finest).
#[derive(Debug, Clone)]
pub struct RefinedAssignment {
    pub sc_to_group: Vec<Vec<usize>>,
    pub num_groups_per_level: Vec<usize>,
}

////////////////////////////////////////////////////////////////////////////////
// Internal profile representation
////////////////////////////////////////////////////////////////////////////////

/// Sparse row-per-entity profile, owned by the refinement pass.
///
/// Materialized from either the existing `gene_sums` (`Raw`) or a projected
/// centroid matrix (`Projected`). Values are stored sorted by feature index
/// within each row so that downstream code can rely on the invariant when
/// merging with `DcSbmStats` accumulators.
struct Profiles {
    rows: Vec<Vec<(u32, f32)>>,
    size_factor: Vec<f32>,
    num_entities: usize,
    num_features: usize,
}

impl Profiles {
    fn from_gene_sums(gene_sums: &[Vec<(usize, f32)>], num_features: usize) -> Self {
        let num_entities = gene_sums.len();
        let mut rows: Vec<Vec<(u32, f32)>> = Vec::with_capacity(num_entities);
        let mut size_factor = Vec::with_capacity(num_entities);
        for row in gene_sums {
            let mut out: Vec<(u32, f32)> = row
                .iter()
                .filter(|(_, v)| *v > 0.0)
                .map(|(g, v)| (*g as u32, *v))
                .collect();
            out.sort_unstable_by_key(|&(g, _)| g);
            let sf = out.iter().map(|(_, v)| *v).sum();
            rows.push(out);
            size_factor.push(sf);
        }
        Self {
            rows,
            size_factor,
            num_entities,
            num_features,
        }
    }

    /// Build profiles from a dense projection.
    ///
    /// `basis` is `proj_dim × num_cells`; each super-cell profile is the sum
    /// of basis columns over the cells constituting it. IDF is not applied.
    fn from_projection(basis: &DMatrix<f32>, super_cell_to_cells: &[Vec<usize>]) -> Self {
        let num_features = basis.nrows();
        let num_entities = super_cell_to_cells.len();
        let mut rows: Vec<Vec<(u32, f32)>> = Vec::with_capacity(num_entities);
        let mut size_factor = Vec::with_capacity(num_entities);
        for cells in super_cell_to_cells {
            let mut acc = vec![0f32; num_features];
            for &c in cells {
                for d in 0..num_features {
                    acc[d] += basis[(d, c)];
                }
            }
            // Dense → sparse: keep entries where the accumulated score is
            // strictly positive so Poisson scoring stays well-defined.
            let mut out = Vec::with_capacity(num_features);
            let mut sf = 0f32;
            for (d, &v) in acc.iter().enumerate() {
                if v > 0.0 {
                    out.push((d as u32, v));
                    sf += v;
                }
            }
            rows.push(out);
            size_factor.push(sf);
        }
        Self {
            rows,
            size_factor,
            num_entities,
            num_features,
        }
    }

    /// Empirical marginal p_g = Σ_e y_eg / Σ_{e,g} y_eg.
    fn empirical_marginal(&self) -> Vec<f64> {
        let mut bg = vec![0f64; self.num_features];
        let mut total = 0f64;
        for row in &self.rows {
            for &(g, v) in row {
                bg[g as usize] += v as f64;
                total += v as f64;
            }
        }
        if total > 0.0 {
            for x in &mut bg {
                *x /= total;
            }
        }
        bg
    }

    /// In-place IDF reweighting: y_eg ← y_eg · (−ln(bg_g + ε)). Recomputes
    /// per-row size factors.
    fn weight_by_idf(&mut self, bg: &[f64]) {
        debug_assert_eq!(bg.len(), self.num_features);
        let w: Vec<f32> = bg.iter().map(|&p| (-(p + LOG_EPS).ln()) as f32).collect();
        for (row, sf) in self.rows.iter_mut().zip(self.size_factor.iter_mut()) {
            let mut new_sf = 0f32;
            for (g, v) in row.iter_mut() {
                *v *= w[*g as usize];
                new_sf += *v;
            }
            *sf = new_sf;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Poisson-Gamma DC-SBM sufficient statistics
////////////////////////////////////////////////////////////////////////////////

/// Sufficient statistics with cached log quantities for O(K · nnz(row)) scoring.
///
/// `gene_sum[k·M + g]` = Σ_{e : z_e = k} y_eg; `size_sum[k]` = Σ_{e : z_e = k} s_e.
/// Log caches let `compute_log_probs_restricted` avoid any `ln()` inside the
/// hot loop; only `delta_move` calls `ln()` on two rows / two scalars per move.
///
/// Log caches are `f32` — values live in roughly `[-20, +25]` (dominated by
/// `-ln(LOG_EPS)` at the floor), and scoring is noise-dominated so f32's
/// ~7-digit precision is ample. Halves the largest allocation versus f64.
#[derive(Clone)]
struct DcSbmStats {
    k: usize,
    num_features: usize,
    membership: Vec<usize>,
    gene_sum: Vec<f64>,
    size_sum: Vec<f64>,
    log_gene: Vec<f32>,
    log_size_offset: Vec<f32>,
}

impl DcSbmStats {
    fn from_profiles(profiles: &Profiles, k: usize, membership: &[usize]) -> Self {
        let m = profiles.num_features;
        let mut gene_sum = vec![0f64; k * m];
        let mut size_sum = vec![0f64; k];
        for (e, row) in profiles.rows.iter().enumerate() {
            let z = membership[e];
            debug_assert!(z < k, "membership[{}]={} out of range 0..{}", e, z, k);
            let base = z * m;
            for &(g, v) in row {
                gene_sum[base + g as usize] += v as f64;
            }
            size_sum[z] += profiles.size_factor[e] as f64;
        }
        let mut log_gene = vec![0f32; k * m];
        for i in 0..k * m {
            log_gene[i] = (gene_sum[i] + LOG_EPS).ln() as f32;
        }
        let m_eps = m as f64 * LOG_EPS;
        let log_size_offset: Vec<f32> = size_sum
            .iter()
            .map(|&s| -((s + m_eps).ln()) as f32)
            .collect();
        Self {
            k,
            num_features: m,
            membership: membership.to_vec(),
            gene_sum,
            size_sum,
            log_gene,
            log_size_offset,
        }
    }

    /// Apply an entity's reassignment and refresh only the affected log rows.
    fn delta_move(&mut self, e: usize, k_from: usize, k_to: usize, profiles: &Profiles) {
        if k_from == k_to {
            return;
        }
        let m = self.num_features;
        let m_eps = m as f64 * LOG_EPS;

        let base_from = k_from * m;
        let base_to = k_to * m;
        for &(g, v) in &profiles.rows[e] {
            let gi = g as usize;
            self.gene_sum[base_from + gi] -= v as f64;
            self.gene_sum[base_to + gi] += v as f64;
            self.log_gene[base_from + gi] = (self.gene_sum[base_from + gi] + LOG_EPS).ln() as f32;
            self.log_gene[base_to + gi] = (self.gene_sum[base_to + gi] + LOG_EPS).ln() as f32;
        }
        let sf = profiles.size_factor[e] as f64;
        self.size_sum[k_from] -= sf;
        self.size_sum[k_to] += sf;
        self.log_size_offset[k_from] = -((self.size_sum[k_from] + m_eps).ln()) as f32;
        self.log_size_offset[k_to] = -((self.size_sum[k_to] + m_eps).ln()) as f32;

        self.membership[e] = k_to;
    }

    /// Full recompute from current `membership` (slow path, used for tests).
    #[cfg(test)]
    fn recompute(&mut self, profiles: &Profiles) {
        let k = self.k;
        let m = self.num_features;
        self.gene_sum.iter_mut().for_each(|x| *x = 0.0);
        self.size_sum.iter_mut().for_each(|x| *x = 0.0);
        for (e, row) in profiles.rows.iter().enumerate() {
            let z = self.membership[e];
            let base = z * m;
            for &(g, v) in row {
                self.gene_sum[base + g as usize] += v as f64;
            }
            self.size_sum[z] += profiles.size_factor[e] as f64;
        }
        for i in 0..k * m {
            self.log_gene[i] = (self.gene_sum[i] + LOG_EPS).ln() as f32;
        }
        let m_eps = m as f64 * LOG_EPS;
        for i in 0..k {
            self.log_size_offset[i] = -((self.size_sum[i] + m_eps).ln()) as f32;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Scoring kernels
////////////////////////////////////////////////////////////////////////////////

/// Log-probability of placing entity `e` into each of the `allowed` blocks.
///
/// Slots outside `allowed` are set to `f64::NEG_INFINITY`. The caller reuses
/// the same `log_probs` buffer across sweeps to avoid allocation.
///
/// Score (Poisson plug-in MAP, up to a common constant):
/// `  s(e, k) = Σ_{g : y_eg > 0} y_eg · log_gene[k, g] + size_factor[e] · log_size_offset[k]`
fn compute_log_probs_restricted(
    e: usize,
    stats: &DcSbmStats,
    profiles: &Profiles,
    allowed: &[usize],
    log_probs: &mut [f64],
) {
    log_probs.iter_mut().for_each(|x| *x = f64::NEG_INFINITY);
    let m = stats.num_features;
    let sf = profiles.size_factor[e] as f64;
    let row = &profiles.rows[e];
    for &k in allowed {
        let base = k * m;
        let mut acc = sf * stats.log_size_offset[k] as f64;
        for &(g, v) in row {
            acc += v as f64 * stats.log_gene[base + g as usize] as f64;
        }
        log_probs[k] = acc;
    }
}

/// Unrestricted variant used only from tests.
#[cfg(test)]
fn compute_log_probs(e: usize, stats: &DcSbmStats, profiles: &Profiles, log_probs: &mut [f64]) {
    let m = stats.num_features;
    let sf = profiles.size_factor[e] as f64;
    let row = &profiles.rows[e];
    for (k, slot) in log_probs.iter_mut().enumerate().take(stats.k) {
        let base = k * m;
        let mut acc = sf * stats.log_size_offset[k] as f64;
        for &(g, v) in row {
            acc += v as f64 * stats.log_gene[base + g as usize] as f64;
        }
        *slot = acc;
    }
}

/// Gumbel-max categorical sampling restricted to finite-valued slots.
fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
    let mut best_key = f64::NEG_INFINITY;
    let mut best_idx = usize::MAX;
    for (i, &lp) in log_probs.iter().enumerate() {
        if lp.is_finite() {
            let u: f64 = rng.random_range(1e-12..1.0_f64);
            let g = -(-u.ln()).ln();
            let key = lp + g;
            if key > best_key {
                best_key = key;
                best_idx = i;
            }
        }
    }
    debug_assert!(
        best_idx != usize::MAX,
        "sample_categorical_log: no finite slot"
    );
    best_idx
}

fn argmax_log_restricted(log_probs: &[f64], allowed: &[usize]) -> usize {
    let mut best = allowed[0];
    let mut best_val = log_probs[best];
    for &k in &allowed[1..] {
        if log_probs[k] > best_val {
            best = k;
            best_val = log_probs[k];
        }
    }
    best
}

////////////////////////////////////////////////////////////////////////////////
// Candidate set construction (siblings ∩ BBKNN with sibling fallback)
////////////////////////////////////////////////////////////////////////////////

/// Per-super-cell BBKNN proposals: for every non-own batch, query that
/// batch's cell-level HNSW (`SparseIoVec::batch_knn_lookup`) with the
/// super-cell's centroid, dedup returned cells to their owning super-cells
/// via `layout.cell_to_sc`, and keep up to `knn` distinct super-cells per
/// other batch.
///
/// Total candidate set per super-cell is up to `knn · (num_batches − 1)`.
fn build_bbknn_neighbors(
    layout: &SuperCellLayout,
    batch_knn_lookup: &[matrix_util::knn_match::ColumnDict<usize>],
    knn: usize,
) -> anyhow::Result<Vec<Vec<usize>>> {
    use matrix_util::knn_match::MakeVecPoint;
    let num_sc = layout.cell_counts.len();
    let cell_oversample = (knn * 4 + 1).max(knn);

    let mut result = Vec::with_capacity(num_sc);
    for sc in 0..num_sc {
        let sc_batch = layout.super_cell_to_batch[sc];
        let centroid = layout.centroids.column(sc).to_vp();
        let mut all_scs: Vec<usize> = Vec::new();
        for (b, bknn) in batch_knn_lookup.iter().enumerate() {
            if b == sc_batch {
                continue;
            }
            let (cells, _dists) = bknn.search_by_query_data(&centroid, cell_oversample)?;
            let mut seen: Vec<usize> = Vec::new();
            for &c in &cells {
                let other_sc = layout.cell_to_sc[c];
                if other_sc == usize::MAX || other_sc == sc {
                    continue;
                }
                if !seen.contains(&other_sc) {
                    seen.push(other_sc);
                    if seen.len() >= knn {
                        break;
                    }
                }
            }
            all_scs.extend(seen);
        }
        result.push(all_scs);
    }
    Ok(result)
}

/// For each super-cell, siblings at `level` = groups at `level` sharing the
/// same parent group at `level + 1`. At the coarsest level (index
/// `num_levels - 1`), the virtual root groups every sc together → siblings =
/// all groups.
fn compute_sibling_sets(
    refined: &[Vec<usize>],
    level: usize,
    num_groups_at_level: usize,
) -> Vec<Vec<usize>> {
    let num_levels = refined.len();
    let num_sc = refined[level].len();

    if level + 1 >= num_levels {
        let all_groups: Vec<usize> = (0..num_groups_at_level).collect();
        return vec![all_groups; num_sc];
    }

    // parent_group -> sorted unique set of child groups at `level`
    let mut parent_to_children: HashMap<usize, Vec<usize>> = HashMap::default();
    for (child, parent) in refined[level].iter().zip(refined[level + 1].iter()) {
        let entry = parent_to_children.entry(*parent).or_default();
        if !entry.contains(child) {
            entry.push(*child);
        }
    }
    for v in parent_to_children.values_mut() {
        v.sort_unstable();
    }

    (0..num_sc)
        .map(|sc| {
            let parent = refined[level + 1][sc];
            parent_to_children.get(&parent).cloned().unwrap_or_default()
        })
        .collect()
}

/// Candidate set per super-cell = siblings ∩ (groups of BBKNN neighbors).
/// Falls back to full siblings when the intersection is empty.
fn build_candidate_sets(
    siblings: &[Vec<usize>],
    bbknn: &[Vec<usize>],
    sc_to_group_at_level: &[usize],
) -> Vec<Vec<usize>> {
    let num_sc = siblings.len();
    let mut out = Vec::with_capacity(num_sc);
    for sc in 0..num_sc {
        let sib = &siblings[sc];
        if sib.is_empty() {
            out.push(Vec::new());
            continue;
        }
        if sib.len() == 1 {
            // Nothing to choose; skip neighbor lookup.
            out.push(sib.clone());
            continue;
        }
        let mut neighbor_groups: Vec<usize> =
            bbknn[sc].iter().map(|&j| sc_to_group_at_level[j]).collect();
        neighbor_groups.sort_unstable();
        neighbor_groups.dedup();

        let intersect: Vec<usize> = sib
            .iter()
            .copied()
            .filter(|g| neighbor_groups.binary_search(g).is_ok())
            .collect();
        // Always include the current group so a sc can stay put.
        let mut cand = if intersect.is_empty() {
            sib.clone()
        } else {
            let current = sc_to_group_at_level[sc];
            if intersect.contains(&current) {
                intersect
            } else {
                let mut c = intersect;
                c.push(current);
                c.sort_unstable();
                c
            }
        };
        cand.dedup();
        out.push(cand);
    }
    out
}

////////////////////////////////////////////////////////////////////////////////
// Refinement driver
////////////////////////////////////////////////////////////////////////////////

/// Compact a group-index vector to 0..K (preserving relative order of first
/// appearance). Returns the new labels and the new K.
pub fn compact_labels(labels: &[usize]) -> (Vec<usize>, usize) {
    let mut map: HashMap<usize, usize> = HashMap::default();
    let mut next = 0usize;
    let mut out = Vec::with_capacity(labels.len());
    for &g in labels {
        let new = *map.entry(g).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        out.push(new);
    }
    (out, next)
}

fn refine_one_level(
    profiles: &Profiles,
    sc_to_group: &mut [usize],
    k: usize,
    candidates: &[Vec<usize>],
    params: &RefineParams,
    rng: &mut SmallRng,
    level_label: &str,
) -> usize {
    use indicatif::{ProgressBar, ProgressStyle};
    let num_sc = sc_to_group.len();
    let mut stats = DcSbmStats::from_profiles(profiles, k, sc_to_group);
    let mut log_probs = vec![f64::NEG_INFINITY; k];
    let mut total_moves = 0usize;

    let max_sweeps = (params.num_gibbs + params.num_greedy) as u64;
    let pb = ProgressBar::new(max_sweeps).with_style(
        ProgressStyle::with_template(&format!(
            "{} {{bar:40}} {{pos}}/{{len}} sweeps ({{eta}})",
            level_label
        ))
        .unwrap()
        .progress_chars("##-"),
    );

    if params.num_gibbs > 0 {
        let mut order: Vec<usize> = (0..num_sc).collect();
        let mut low_sweeps = 0usize;
        for _sweep in 0..params.num_gibbs {
            order.shuffle(rng);
            let mut moves = 0usize;
            for &e in &order {
                let cand = &candidates[e];
                if cand.len() < 2 {
                    continue;
                }
                compute_log_probs_restricted(e, &stats, profiles, cand, &mut log_probs);
                let new = sample_categorical_log(&log_probs, rng);
                let old = stats.membership[e];
                if new != old {
                    stats.delta_move(e, old, new, profiles);
                    moves += 1;
                }
            }
            total_moves += moves;
            pb.inc(1);
            if params.gibbs_stagnation > 0.0 {
                if (moves as f64) < params.gibbs_stagnation * (num_sc as f64) {
                    low_sweeps += 1;
                    if low_sweeps >= 3 {
                        break;
                    }
                } else {
                    low_sweeps = 0;
                }
            }
        }
    }

    for _sweep in 0..params.num_greedy {
        let mut moves = 0usize;
        for (e, cand) in candidates.iter().enumerate() {
            if cand.len() < 2 {
                continue;
            }
            compute_log_probs_restricted(e, &stats, profiles, cand, &mut log_probs);
            let new = argmax_log_restricted(&log_probs, cand);
            let old = stats.membership[e];
            if new != old {
                stats.delta_move(e, old, new, profiles);
                moves += 1;
            }
        }
        total_moves += moves;
        pb.inc(1);
        if moves == 0 {
            break;
        }
    }
    pb.finish_and_clear();

    sc_to_group.copy_from_slice(&stats.membership);
    total_moves
}

/// Fixed inputs describing *what* to refine. `batch_knn_lookup` is the
/// per-batch HNSW over cells, typically obtained from
/// `SparseIoVec::batch_knn_lookup()` after `build_hnsw_per_batch`.
/// `k_per_batch` is the BBKNN fan-out — up to this many distinct
/// super-cells are drawn from **each** non-own batch as move candidates.
#[derive(Clone, Copy)]
pub struct RefineInputs<'a> {
    pub layout: &'a SuperCellLayout,
    pub gene_sums: &'a GeneSums,
    pub num_genes: usize,
    pub super_cell_to_cells: &'a [Vec<usize>],
    pub batch_knn_lookup: &'a [matrix_util::knn_match::ColumnDict<usize>],
    pub k_per_batch: usize,
    pub initial_sc_to_group_per_level: &'a [Vec<usize>],
}

/// Top-down BBKNN + Poisson DC-SBM refinement.
///
/// Levels in `inputs.initial_sc_to_group_per_level` follow the existing
/// finest → coarsest convention. The returned `RefinedAssignment` uses the
/// same ordering.
pub fn refine_assignments(
    inputs: &RefineInputs<'_>,
    params: &RefineParams,
) -> anyhow::Result<RefinedAssignment> {
    let RefineInputs {
        layout,
        gene_sums,
        num_genes,
        super_cell_to_cells,
        batch_knn_lookup,
        k_per_batch,
        initial_sc_to_group_per_level,
    } = *inputs;
    let num_levels = initial_sc_to_group_per_level.len();
    if num_levels == 0 {
        return Err(anyhow::anyhow!("no levels"));
    }
    let num_sc = layout.cell_counts.len();
    for (i, lvl) in initial_sc_to_group_per_level.iter().enumerate() {
        if lvl.len() != num_sc {
            return Err(anyhow::anyhow!(
                "level {} has {} entries, expected {}",
                i,
                lvl.len(),
                num_sc
            ));
        }
    }

    // Build profiles once.
    let mut profiles = match &params.profile_source {
        ProfileSource::Raw => Profiles::from_gene_sums(gene_sums, num_genes),
        ProfileSource::Projected { basis } => Profiles::from_projection(basis, super_cell_to_cells),
    };
    if params.idf_weighting && matches!(params.profile_source, ProfileSource::Raw) {
        let bg = profiles.empirical_marginal();
        profiles.weight_by_idf(&bg);
    }

    // BBKNN proposals via the shared per-batch cell HNSW.
    let bbknn = build_bbknn_neighbors(layout, batch_knn_lookup, k_per_batch)?;

    // Compact each level's initial labels to a dense 0..K range.
    let mut refined: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    let mut ks: Vec<usize> = Vec::with_capacity(num_levels);
    for lvl in initial_sc_to_group_per_level {
        let (compact, k) = compact_labels(lvl);
        refined.push(compact);
        ks.push(k);
    }

    let mut rng = SmallRng::seed_from_u64(params.seed);

    // Walk coarsest → finest (highest level index down to 0).
    for level in (0..num_levels).rev() {
        let k = ks[level];
        debug!("refining level {} (k={}, num_sc={})", level, k, num_sc);
        let siblings = compute_sibling_sets(&refined, level, k);

        let candidates = build_candidate_sets(&siblings, &bbknn, &refined[level]);

        let sc_to_group = &mut refined[level];
        let label = format!("Refine L{}/{}", num_levels - level, num_levels);
        let moves = refine_one_level(
            &profiles,
            sc_to_group,
            k,
            &candidates,
            params,
            &mut rng,
            &label,
        );
        info!("  level {} refined: {} moves; k={} groups", level, moves, k);

        // Compact in case greedy emptied groups (keeps K monotone without gaps).
        let (compact, new_k) = compact_labels(sc_to_group);
        *sc_to_group = compact;
        ks[level] = new_k;
    }

    Ok(RefinedAssignment {
        sc_to_group: refined,
        num_groups_per_level: ks,
    })
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_gene_sums(
        num_entities: usize,
        num_features: usize,
        labels: &[usize],
        rng_seed: u64,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut rng = SmallRng::seed_from_u64(rng_seed);
        let num_blocks = *labels.iter().max().unwrap() + 1;
        let per_block = num_features / num_blocks;
        (0..num_entities)
            .map(|e| {
                let c = labels[e];
                let start = c * per_block;
                let end = ((c + 1) * per_block).min(num_features);
                let mut row: Vec<(usize, f32)> = (start..end)
                    .map(|g| (g, 5.0 + rng.random_range(0.0..3.0_f32)))
                    .collect();
                // Add a little noise elsewhere.
                for _ in 0..3 {
                    let g: usize = rng.random_range(0..num_features);
                    row.push((g, rng.random_range(0.0..1.0_f32)));
                }
                row.sort_unstable_by_key(|&(g, _)| g);
                row.dedup_by_key(|&mut (g, _)| g);
                row
            })
            .collect()
    }

    fn make_profiles(gene_sums: &[Vec<(usize, f32)>], num_features: usize) -> Profiles {
        Profiles::from_gene_sums(gene_sums, num_features)
    }

    #[test]
    fn test_log_probs_match_after_delta_moves() {
        let n = 24;
        let m = 16;
        let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
        let gs = toy_gene_sums(n, m, &labels, 1);
        let profiles = make_profiles(&gs, m);

        let mut stats = DcSbmStats::from_profiles(&profiles, 4, &labels);
        let mut rng = SmallRng::seed_from_u64(7);
        // Random moves.
        for _ in 0..50 {
            let e: usize = rng.random_range(0..n);
            let to: usize = rng.random_range(0..4);
            let from = stats.membership[e];
            stats.delta_move(e, from, to, &profiles);
        }

        let kept_mem = stats.membership.clone();
        let mut fresh = DcSbmStats::from_profiles(&profiles, 4, &kept_mem);
        fresh.recompute(&profiles);

        for i in 0..stats.gene_sum.len() {
            assert!((stats.gene_sum[i] - fresh.gene_sum[i]).abs() < 1e-6);
            assert!((stats.log_gene[i] - fresh.log_gene[i]).abs() < 1e-6);
        }
        for i in 0..stats.size_sum.len() {
            assert!((stats.size_sum[i] - fresh.size_sum[i]).abs() < 1e-6);
            assert!((stats.log_size_offset[i] - fresh.log_size_offset[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_restricted_matches_unrestricted_on_allowed() {
        let n = 20;
        let m = 12;
        let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
        let gs = toy_gene_sums(n, m, &labels, 2);
        let profiles = make_profiles(&gs, m);
        let stats = DcSbmStats::from_profiles(&profiles, 4, &labels);

        let mut full = vec![0f64; 4];
        let mut restricted = vec![0f64; 4];
        compute_log_probs(3, &stats, &profiles, &mut full);

        let allowed = vec![1usize, 3usize];
        compute_log_probs_restricted(3, &stats, &profiles, &allowed, &mut restricted);
        for &k in &allowed {
            assert!((full[k] - restricted[k]).abs() < 1e-9);
        }
        for (k, &lp) in restricted.iter().enumerate() {
            if !allowed.contains(&k) {
                assert!(lp.is_infinite() && lp < 0.0);
            }
        }
    }

    #[test]
    fn test_idf_downweights_housekeeping() {
        // Two super-cells, two features. Feature 0 is ubiquitous; feature 1
        // is super-cell-specific. After IDF, feature 1 should dominate the
        // log-scores.
        let gs = vec![
            vec![(0usize, 10.0f32), (1, 10.0)],
            vec![(0usize, 10.0f32), (1, 0.1)],
        ];
        let mut profiles = Profiles::from_gene_sums(&gs, 2);
        let bg = profiles.empirical_marginal();
        profiles.weight_by_idf(&bg);
        // p_0 ≈ 20/30.1 large → small weight; p_1 ≈ 10.1/30.1 → larger weight.
        let ratio = profiles.rows[0].iter().find(|(g, _)| *g == 1).unwrap().1
            / profiles.rows[0].iter().find(|(g, _)| *g == 0).unwrap().1;
        assert!(
            ratio > 1.0,
            "IDF should boost the specific feature relative to housekeeping (got ratio={})",
            ratio
        );
    }

    #[test]
    fn test_empty_block_finite() {
        // Block 0 has no members → size_sum[0]=0 → log_size_offset[0] finite.
        let gs = vec![vec![(0usize, 3.0f32)], vec![(1usize, 4.0f32)]];
        let profiles = Profiles::from_gene_sums(&gs, 2);
        let stats = DcSbmStats::from_profiles(&profiles, 3, &[1, 2]);
        assert!(stats.log_size_offset[0].is_finite());
        assert_eq!(stats.size_sum[0], 0.0);
    }

    #[test]
    fn test_compact_labels() {
        let (c, k) = compact_labels(&[5, 5, 2, 7, 2, 7, 5]);
        assert_eq!(k, 3);
        assert_eq!(c, vec![0, 0, 1, 2, 1, 2, 0]);
    }

    #[test]
    fn test_compute_sibling_sets_at_coarsest_gives_all() {
        let refined = vec![
            vec![0usize, 1, 2, 3], // finest
            vec![0usize, 0, 1, 1], // coarsest
        ];
        let top = refined.len() - 1;
        let sibs = compute_sibling_sets(&refined, top, 2);
        assert!(sibs.iter().all(|s| s == &vec![0usize, 1]));
    }

    #[test]
    fn test_compute_sibling_sets_respects_parent() {
        let refined = vec![
            vec![0usize, 1, 2, 3], // finest: 4 groups
            vec![0usize, 0, 1, 1], // parent: {0,1}→0, {2,3}→1
        ];
        let sibs = compute_sibling_sets(&refined, 0, 4);
        assert_eq!(sibs[0], vec![0, 1]);
        assert_eq!(sibs[1], vec![0, 1]);
        assert_eq!(sibs[2], vec![2, 3]);
        assert_eq!(sibs[3], vec![2, 3]);
    }

    #[test]
    fn test_build_candidate_sets_fallback() {
        // sc 0 has siblings [0,1] but BBKNN lands only on a non-sibling group.
        let siblings = vec![vec![0usize, 1], vec![0, 1]];
        let bbknn = vec![vec![1usize], vec![0]];
        let sc_to_group = vec![0usize, 1];
        // Neighbor of sc0 is sc1 (group 1) → intersection {1}.
        // sc0 is in group 0; intersection lacks 0 so 0 is appended.
        let cand = build_candidate_sets(&siblings, &bbknn, &sc_to_group);
        assert_eq!(cand[0], vec![0, 1]);
        assert_eq!(cand[1], vec![0, 1]);
    }
}
