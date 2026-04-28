//! Degree-corrected Poisson refinement core.
//!
//! Shared Poisson-scoring machinery for top-down multi-level refinement.
//! Given per-entity sparse profiles (gene sums or a dense projection) and
//! current group labels, refinement reassigns entities between groups by
//! scoring moves under a Poisson likelihood with size-factor offsets.
//! Candidate-set construction is delegated to [`CandidateProposer`] so
//! different front-ends (cross-batch BBKNN, spatial KNN graph, ...) can
//! plug in their own neighborhood definition.
//!
//! The name "DC-Poisson" is deliberate: we are *not* inferring a stochastic
//! block model — the blocks are given externally (by a prior coarsening
//! step) and we only score membership moves. Compare to the full DC-SBM
//! where blocks themselves are latent variables.
//!
//! Entry points:
//! - [`refine_with_candidates`] — lowest-level sweep driver given pre-built candidates.
//! - [`refine_with_proposer`] — generic driver that first asks a [`CandidateProposer`].
//!
//! See the `BbknnProposer` in `refine_multilevel` for the data-beans-alg
//! front-end, and pinto's `GraphProposer` for the spatial-graph front-end.

use log::info;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

/// Additive floor to keep `ln(.)` finite when a block or feature has zero
/// mass.
pub const LOG_EPS: f64 = 1e-9;

////////////////////////////////////////////////////////////////////////////////
// Public configuration
////////////////////////////////////////////////////////////////////////////////

/// Feature representation used to score entity → block moves.
#[derive(Clone, Debug)]
pub enum ProfileSource {
    /// Sparse entity × gene counts (the existing `gene_sums` format).
    /// [`GeneWeighting`] is applied per-gene on the sparse profile.
    Raw,
    /// Dense projection: `basis * indicator(entity)` per entity. `basis` is
    /// `proj_dim × num_cells`-shaped; the per-entity profile is the sum over
    /// its member cells' projection columns. [`GeneWeighting`] is skipped
    /// because the feature axis is no longer "genes".
    Projected { basis: DMatrix<f32> },
}

/// Per-feature (gene) weighting applied to the sparse profile before DC-Poisson
/// scoring. All variants multiply each nonzero entry `y_{e,g}` by a scalar
/// `w_g` and recompute per-row size factors; only the formula for `w_g`
/// differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneWeighting {
    /// No reweighting: `w_g = 1`. Recovers proper DC-Poisson with entity-level
    /// degree correction (Karrer–Newman MAP under `Gamma(α, 0)` on `θ`).
    None,
    /// Fisher-information weight under the fitted NB trend:
    /// `w_g = 1 / (1 + π_g · s̄ · φ(μ_g))`. Bounded in `(0, 1]`,
    /// attenuates high-mean / high-dispersion features, recovers
    /// `w_g = 1` in the Poisson limit (`φ → 0`). Current default.
    FisherInfoNb,
}

/// Parameters controlling the refinement pass.
#[derive(Clone, Debug)]
pub struct RefineParams {
    /// Gibbs sweeps per level (0 disables Gibbs; greedy still runs).
    pub num_gibbs: usize,
    /// Greedy sweeps per level (early-exits on zero moves).
    pub num_greedy: usize,
    /// Per-feature weighting scheme (only meaningful for `Raw` profile source).
    pub gene_weighting: GeneWeighting,
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
            gene_weighting: GeneWeighting::FisherInfoNb,
            seed: 42,
            gibbs_stagnation: 0.005,
            profile_source: ProfileSource::Raw,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sparse row-per-entity profiles
////////////////////////////////////////////////////////////////////////////////

/// Sparse row-per-entity profile, owned by the refinement pass.
///
/// Materialized from either a sparse gene-sum format (`from_gene_sums`) or a
/// projected centroid matrix (`from_projection`). Values are stored sorted
/// by feature index within each row so downstream accumulators can rely on
/// the invariant.
pub struct Profiles {
    pub rows: Vec<Vec<(u32, f32)>>,
    pub size_factor: Vec<f32>,
    pub num_entities: usize,
    pub num_features: usize,
}

impl Profiles {
    pub fn from_gene_sums(gene_sums: &[Vec<(usize, f32)>], num_features: usize) -> Self {
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
    /// `basis` is `proj_dim × num_cells`; each entity profile is the sum of
    /// basis columns over the cells constituting it. Gene weighting is not
    /// applied — projection dims aren't gene-aligned.
    pub fn from_projection(basis: &DMatrix<f32>, entity_to_cells: &[Vec<usize>]) -> Self {
        let num_features = basis.nrows();
        let num_entities = entity_to_cells.len();
        let (rows, size_factor): (Vec<Vec<(u32, f32)>>, Vec<f32>) = entity_to_cells
            .par_iter()
            .map(|cells| {
                let mut acc = vec![0f32; num_features];
                for &c in cells {
                    for d in 0..num_features {
                        acc[d] += basis[(d, c)];
                    }
                }
                let mut out = Vec::with_capacity(num_features);
                let mut sf = 0f32;
                for (d, &v) in acc.iter().enumerate() {
                    if v > 0.0 {
                        out.push((d as u32, v));
                        sf += v;
                    }
                }
                (out, sf)
            })
            .unzip();
        Self {
            rows,
            size_factor,
            num_entities,
            num_features,
        }
    }

    /// In-place reweighting by a caller-supplied per-feature weight vector.
    /// Used by [`Profiles::apply_gene_weighting`] for the NB Fisher-info path.
    pub fn weight_by_vec(&mut self, w: &[f32]) {
        debug_assert_eq!(w.len(), self.num_features);
        for (row, sf) in self.rows.iter_mut().zip(self.size_factor.iter_mut()) {
            let mut new_sf = 0f32;
            for (g, v) in row.iter_mut() {
                *v *= w[*g as usize];
                new_sf += *v;
            }
            *sf = new_sf;
        }
    }

    /// Apply the chosen gene weighting in place.
    ///
    /// [`GeneWeighting::None`] is a no-op; [`GeneWeighting::FisherInfoNb`] fits
    /// an NB dispersion trend from the current profiles and reweights each
    /// feature by `1 / (1 + π_g · s̄ · φ(μ_g))`.
    pub fn apply_gene_weighting(&mut self, method: GeneWeighting) {
        match method {
            GeneWeighting::None => {}
            GeneWeighting::FisherInfoNb => {
                let w = self.nb_fisher_weights();
                self.weight_by_vec(&w);
            }
        }
    }

    /// Compute per-feature Fisher-info weights under an NB trend fit from
    /// the current profile contents. Returned vector has length `num_features`
    /// and is suitable for [`Profiles::weight_by_vec`].
    pub fn nb_fisher_weights(&self) -> Vec<f32> {
        use crate::nb_dispersion::DispersionTrend;
        use matrix_util::sparse_stat::SparseRunningStatistics;
        use matrix_util::traits::RunningStatOps;

        // One pass: accumulate per-feature sum and sum-of-squares over all
        // entities. Reuses one scratch buffer per row.
        let mut stats = SparseRunningStatistics::<f32>::new(self.num_features);
        let mut col_rows: Vec<usize> = Vec::new();
        let mut col_vals: Vec<f32> = Vec::new();
        for row in &self.rows {
            col_rows.clear();
            col_vals.clear();
            for &(g, v) in row {
                col_rows.push(g as usize);
                col_vals.push(v);
            }
            stats.add_sparse_column(&col_rows, &col_vals);
        }

        // `π_g = sum[g] / Σ sum` is derived directly from the stats without
        // a second sparse traversal. `mean_g` uses the entity-count denominator
        // that `SparseRunningStatistics` already provides.
        let trend = DispersionTrend::from_sparse_stats(&stats);
        let means = stats.mean();
        let sums = stats.sum();
        let total_mass: f64 = sums.iter().map(|&s| s as f64).sum();
        let avg_s = if self.num_entities > 0 {
            (total_mass / self.num_entities as f64) as f32
        } else {
            1.0
        };
        let inv_total = if total_mass > 0.0 {
            1.0 / total_mass as f32
        } else {
            0.0
        };
        (0..self.num_features)
            .map(|g| trend.fisher_weight(sums[g] * inv_total, avg_s, means[g]))
            .collect()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Poisson sufficient statistics
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
pub struct DcPoissonStats {
    pub k: usize,
    pub num_features: usize,
    pub membership: Vec<usize>,
    pub gene_sum: Vec<f64>,
    pub size_sum: Vec<f64>,
    pub log_gene: Vec<f32>,
    pub log_size_offset: Vec<f32>,
}

impl DcPoissonStats {
    pub fn from_profiles(profiles: &Profiles, k: usize, membership: &[usize]) -> Self {
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
    pub fn delta_move(&mut self, e: usize, k_from: usize, k_to: usize, profiles: &Profiles) {
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
pub fn compute_log_probs_restricted(
    e: usize,
    stats: &DcPoissonStats,
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
fn compute_log_probs(e: usize, stats: &DcPoissonStats, profiles: &Profiles, log_probs: &mut [f64]) {
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
pub fn sample_categorical_log(log_probs: &[f64], rng: &mut SmallRng) -> usize {
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

/// Argmax over `allowed` indices.
pub fn argmax_log_restricted(log_probs: &[f64], allowed: &[usize]) -> usize {
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
// Hierarchy / relabeling helpers
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

/// For each entity, siblings at `level` = groups at `level` sharing the
/// same parent group at `level + 1`.
///
/// `refined` follows the finest → coarsest convention: `refined[0]` is the
/// finest level, `refined[refined.len() - 1]` is the coarsest. At the
/// coarsest level (index `num_levels - 1`), the virtual root groups every
/// entity together → siblings = all groups.
pub fn compute_sibling_sets(
    refined: &[Vec<usize>],
    level: usize,
    num_groups_at_level: usize,
) -> Vec<Vec<usize>> {
    let num_levels = refined.len();
    let num_entities = refined[level].len();

    if level + 1 >= num_levels {
        let all_groups: Vec<usize> = (0..num_groups_at_level).collect();
        return vec![all_groups; num_entities];
    }

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

    (0..num_entities)
        .map(|e| {
            let parent = refined[level + 1][e];
            parent_to_children.get(&parent).cloned().unwrap_or_default()
        })
        .collect()
}

////////////////////////////////////////////////////////////////////////////////
// Candidate proposer + move guard traits
////////////////////////////////////////////////////////////////////////////////

/// Propose candidate group labels each entity may move into on one sweep.
///
/// The returned slice `candidates[e]` MUST include `labels[e]` (staying put
/// is always legal) and MUST index into the current 0..k label space.
pub trait CandidateProposer {
    fn propose(&self, labels: &[usize]) -> Vec<Vec<usize>>;
}

/// Veto individual moves after they've been picked by the sweep.
///
/// Called once per *proposed* accepted move (`from != to`) before the move
/// is applied to the sufficient statistics. The destination label is
/// intentionally not passed — guards so far only need the source cluster
/// (e.g. articulation tests ask "does removing this entity disconnect
/// `from`?"). Add a destination-aware variant if a concrete use case
/// surfaces.
pub trait MoveGuard {
    fn accept_move(&self, entity: usize, from: usize, labels: &[usize]) -> bool;
}

/// Trivial guard that accepts every move. Zero-cost default used by the
/// unguarded [`refine_with_candidates`] and [`refine_with_proposer`].
pub struct NoGuard;

impl MoveGuard for NoGuard {
    #[inline(always)]
    fn accept_move(&self, _e: usize, _from: usize, _labels: &[usize]) -> bool {
        true
    }
}

/// Intersect a per-entity sibling set with a per-entity neighbor-group set,
/// with sibling fallback when the intersection is empty and guaranteed
/// inclusion of the entity's current label so staying put is always a
/// legal move.
///
/// Shared between [`crate::refine_multilevel::BbknnProposer`] and pinto's
/// `GraphProposer` — both proposers differ only in how they gather
/// `neighbor_groups`, not in how they combine with siblings.
///
/// * `siblings` — sorted, deduped sibling group list.
/// * `neighbor_groups` — sorted, deduped group list from the proposer's
///   spatial/structural neighborhood.
/// * `current` — the entity's current label; appended if the intersection
///   excluded it.
pub fn intersect_with_siblings_fallback(
    siblings: &[usize],
    neighbor_groups: &[usize],
    current: usize,
) -> Vec<usize> {
    if siblings.is_empty() {
        return Vec::new();
    }
    if siblings.len() == 1 {
        return siblings.to_vec();
    }
    let intersect: Vec<usize> = siblings
        .iter()
        .copied()
        .filter(|g| neighbor_groups.binary_search(g).is_ok())
        .collect();
    if intersect.is_empty() {
        return siblings.to_vec();
    }
    if intersect.contains(&current) {
        intersect
    } else {
        let mut c = intersect;
        c.push(current);
        c.sort_unstable();
        c
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sweep drivers
////////////////////////////////////////////////////////////////////////////////

/// Refine `labels` in place for one level by running Gibbs + greedy sweeps
/// over `profiles` with the given pre-built `candidates` and a custom
/// [`MoveGuard`]. Returns total accepted moves across all sweeps.
///
/// A proposed move `from → to` is applied only when `guard.accept_move(...)`
/// returns `true`. In Gibbs sweeps, a vetoed move leaves the entity in its
/// current group for this sweep. Greedy sweeps likewise skip the move.
///
/// Shared sweep-driver context bundling the readonly inputs that every
/// `refine_with_*` variant needs. Kept as a struct so the driver fns don't
/// exceed clippy's argument-count threshold.
pub struct RefineContext<'a> {
    pub profiles: &'a Profiles,
    pub k: usize,
    pub params: &'a RefineParams,
    pub level_label: &'a str,
}

/// This is the lowest-level driver. Pass [`NoGuard`] to get the unguarded
/// behavior; most callers should prefer the convenience wrappers
/// [`refine_with_candidates`] / [`refine_with_proposer`] /
/// [`refine_with_proposer_guarded`].
pub fn refine_with_candidates_guarded<G: MoveGuard>(
    labels: &mut [usize],
    candidates: &[Vec<usize>],
    guard: &G,
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    use indicatif::{ProgressBar, ProgressStyle};
    let RefineContext {
        profiles,
        k,
        params,
        level_label,
    } = *ctx;
    let num_entities = labels.len();
    let mut stats = DcPoissonStats::from_profiles(profiles, k, labels);
    let mut log_probs = vec![f64::NEG_INFINITY; k];
    let mut total_moves = 0usize;
    let mut total_vetoed = 0usize;

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
        let mut order: Vec<usize> = (0..num_entities).collect();
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
                    if guard.accept_move(e, old, &stats.membership) {
                        stats.delta_move(e, old, new, profiles);
                        moves += 1;
                    } else {
                        total_vetoed += 1;
                    }
                }
            }
            total_moves += moves;
            pb.inc(1);
            if params.gibbs_stagnation > 0.0 {
                if (moves as f64) < params.gibbs_stagnation * (num_entities as f64) {
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
                if guard.accept_move(e, old, &stats.membership) {
                    stats.delta_move(e, old, new, profiles);
                    moves += 1;
                } else {
                    total_vetoed += 1;
                }
            }
        }
        total_moves += moves;
        pb.inc(1);
        if moves == 0 {
            break;
        }
    }
    pb.finish_and_clear();

    if total_vetoed > 0 {
        log::debug!(
            "{}: {} moves, {} vetoed by MoveGuard",
            level_label,
            total_moves,
            total_vetoed
        );
    }

    labels.copy_from_slice(&stats.membership);
    total_moves
}

/// Unguarded variant of [`refine_with_candidates_guarded`].
pub fn refine_with_candidates(
    labels: &mut [usize],
    candidates: &[Vec<usize>],
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    refine_with_candidates_guarded(labels, candidates, &NoGuard, rng, ctx)
}

/// Generic driver: ask `proposer` for candidates, then run the guarded sweep.
pub fn refine_with_proposer_guarded<P: CandidateProposer, G: MoveGuard>(
    labels: &mut [usize],
    proposer: &P,
    guard: &G,
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    let candidates = proposer.propose(labels);
    let moves = refine_with_candidates_guarded(labels, &candidates, guard, rng, ctx);
    info!("  {}: {} DC-Poisson moves", ctx.level_label, moves);
    moves
}

/// Unguarded generic driver: ask `proposer` for candidates, then run the
/// sweep with [`NoGuard`].
pub fn refine_with_proposer<P: CandidateProposer>(
    labels: &mut [usize],
    proposer: &P,
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    let candidates = proposer.propose(labels);
    let moves = refine_with_candidates(labels, &candidates, rng, ctx);
    info!("  {}: {} DC-Poisson moves", ctx.level_label, moves);
    moves
}

////////////////////////////////////////////////////////////////////////////////
// Tests (algorithm core)
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

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

        let mut stats = DcPoissonStats::from_profiles(&profiles, 4, &labels);
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..50 {
            let e: usize = rng.random_range(0..n);
            let to: usize = rng.random_range(0..4);
            let from = stats.membership[e];
            stats.delta_move(e, from, to, &profiles);
        }

        let kept_mem = stats.membership.clone();
        let mut fresh = DcPoissonStats::from_profiles(&profiles, 4, &kept_mem);
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
        let stats = DcPoissonStats::from_profiles(&profiles, 4, &labels);

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
    fn test_empty_block_finite() {
        let gs = vec![vec![(0usize, 3.0f32)], vec![(1usize, 4.0f32)]];
        let profiles = Profiles::from_gene_sums(&gs, 2);
        let stats = DcPoissonStats::from_profiles(&profiles, 3, &[1, 2]);
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
}
