//! Bayesian Hierarchical Clustering over K clusters with gene-mass sufficient stats.
//!
//! Post-hoc bottom-up merge of K clusters driven by a pairwise Dirichlet-Multinomial
//! marginal-likelihood Bayes factor under an **empirical-Bayes asymmetric Dirichlet
//! prior** centered on the pooled empirical marginal `bg[g] = (Σ_k T_k,g) / (Σ_k S_k)`.
//! The prior expects every cluster to look like `bg`; the BF measures deviation from
//! that baseline. Housekeeping-shaped clusters merge cheaply; clusters with distinct
//! structure stay separate.
//!
//! Node log marginal under Dir(γ · bg) prior:
//!
//!   f(T, S) = lgamma(γ) − lgamma(γ + S)
//!           + Σ_g [lgamma(γ·bg[g] + T_g) − lgamma(γ·bg[g])]
//!
//! Pairwise log Bayes factor:
//!
//!   log_BF(i, j) = f(T_i + T_j, S_i + S_j) − f(T_i, S_i) − f(T_j, S_j)
//!
//! ## Interpreting the scores
//!
//! `log_BF` is the **log of the ratio of evidences**, H₁ (the two clusters came
//! from one underlying distribution) vs H₀ (they came from two distinct ones):
//!
//!   log_BF > 0 — data favor merging (H₁ wins)
//!   log_BF < 0 — data favor keeping separate (H₀ wins)
//!   log_BF = 0 — model is indifferent (natural cut point)
//!
//! Magnitude carries BIC-like meaning: at high counts,
//! `log_BF ≈ −n · KL(p_i ‖ p_j)`, where n is the merged sample size and KL is
//! computed on the proportion vectors `p_k = T_k / S_k`. Rough calibration
//! (symmetric about zero):
//!
//!   |log_BF| < 1       weak — effectively a toss-up
//!   1 ≤ |log_BF| < 3   moderate evidence (≈ e≈2.7× to 20× odds)
//!   3 ≤ |log_BF| < 5   strong (20× to 150× odds)
//!   |log_BF| ≥ 5       decisive (>150× odds)
//!
//! Compare only WITHIN ONE run: absolute magnitudes are not probabilities (the
//! data-dependent normalizer `log Γ(y+1)` is dropped since it cancels across any
//! partition of the same edges). Cross-run comparisons are not meaningful.
//!
//! In the bottom-up agglomeration `bhc_merge` picks the largest `log_BF` at each
//! step, so scores are monotonically non-increasing along the merge sequence.
//! The crossing point `log_BF = 0` is the natural "consensus cut" threshold
//! exposed via `bhc_cut(merges, k, 0.0)`.
//!
//! **Effective sample size is caller-specified.** The caller passes `effective_size`
//! per cluster and the internal `(T, S)` gets rescaled so that `S_eff = effective_size[k]`,
//! preserving shape `T_g / S`. Without rescaling, large raw `S` values (e.g.
//! 10⁴–10⁶ of projection mass on Visium scale, or Σ_n θ_{n,k}·depth_n at
//! 100k-cell scale) make the DM posterior absurdly sharp and blow up log-BF
//! magnitudes. Typical choices:
//!   - link-community: `effective_size[k] = edge_count[k]` (the natural N).
//!   - topic model: `effective_size[k] = (Σ_n θ_{n,k})² / Σ_n θ_{n,k}²` (ESS of the soft assignment).
//!
//! A Heller-Ghahramani DP tree prior can be slotted in later: extend with a
//! `BhcConfig { dp_alpha, ... }` and replace the pure pairwise log-BF rule
//! with the full posterior `r_k` recursion.
//!
//! Runs over K clusters (not N samples). K²·M per merge, K·K²·M overall;
//! trivially fast for K ≤ few hundred. Single-threaded, no RNG.

use crate::union_find::UnionFind;
use special::Gamma as SpecialGamma;

/// Borrowed view over the per-cluster sufficient statistics needed by `bhc_merge`.
///
/// Callers stash their own cluster struct (e.g. `LinkCommunityStats`, topic-model
/// posterior stats) and construct this view without copying.
pub struct BhcInput<'a> {
    /// Number of clusters.
    pub k: usize,
    /// Feature dimension (genes).
    pub m: usize,
    /// Row-major `[k × m]` gene-mass per cluster. `gene_sum[c * m + g]` = T_{c,g}.
    pub gene_sum: &'a [f64],
    /// Per-cluster total size `S_k = Σ_g T_{k,g}` (or an equivalent scale; the
    /// only requirement is `T_k / S_k` is the cluster's proportion vector).
    pub size_sum: &'a [f64],
    /// Per-cluster effective sample size used for DM posterior sharpness.
    /// `(T_g, S)` is rescaled so `S_eff = effective_size[k]` before evaluating
    /// the Bayes factor. A cluster with `effective_size == 0` is skipped.
    pub effective_size: &'a [usize],
}

/// One merge event in the dendrogram.
#[derive(Debug, Clone)]
pub struct BhcMerge {
    /// New node id. Starts at K and increments by 1 per merge.
    pub id: i32,
    /// Child id (< `id`). `left < right` by convention.
    pub left: i32,
    /// Child id (< `id`).
    pub right: i32,
    /// Pairwise log Bayes factor under the empirical-Bayes Dirichlet-Multinomial
    /// model: `log p(D_merged | H₁) − log p(D_i | H₀) − log p(D_j | H₀)`.
    /// Positive → data favor merging; negative → favor keeping separate.
    /// Magnitude at high counts is ~`n · KL(p_i ‖ p_j)` (BIC-like).
    /// Only meaningful within one run — not an absolute probability.
    pub log_bf: f64,
    /// Total effective sample size under this node (Σ over children).
    pub n_samples: i32,
}

/// Sufficient stats for one node in the agglomeration arena.
struct NodeStats {
    t_gene: Vec<f64>,
    s_size: f64,
    n_samples: i32,
    /// Cached f(T, S) for this node; re-used when this node participates in any pairwise BF.
    f_cache: f64,
    /// Public id (original cluster id for leaves, K .. 2K−2 for internal nodes).
    id: i32,
}

#[inline]
fn order_pair(a: i32, b: i32) -> (i32, i32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Empirical-Bayes prior constants. Precomputed once per `bhc_merge` call.
struct Prior {
    /// Total Dirichlet concentration γ.
    gamma: f64,
    /// Per-gene prior mass `γ · bg[g]`.
    gamma_bg: Vec<f64>,
    /// Σ_g lgamma(γ · bg[g]).
    sum_lgamma_gamma_bg: f64,
    /// lgamma(γ).
    lgamma_gamma: f64,
}

/// Node log marginal under Dir(γ · bg):
///
///   f(T, S) = lgamma(γ) − lgamma(γ + S)
///           + Σ_g [lgamma(γ·bg[g] + T_g) − lgamma(γ·bg[g])]
#[inline]
fn f_node(t_gene: &[f64], s_size: f64, prior: &Prior) -> f64 {
    let mut sum_new = 0.0f64;
    for (&t, &gb) in t_gene.iter().zip(prior.gamma_bg.iter()) {
        sum_new += SpecialGamma::ln_gamma(gb + t).0;
    }
    prior.lgamma_gamma - SpecialGamma::ln_gamma(prior.gamma + s_size).0 + sum_new
        - prior.sum_lgamma_gamma_bg
}

/// Pairwise log BF; evaluates `f(merged)` inline and subtracts children's cached f's.
#[inline]
fn pairwise_log_bf(a: &NodeStats, b: &NodeStats, prior: &Prior) -> f64 {
    let mut merged_sum = 0.0f64;
    for ((&ta, &tb), &gb) in a
        .t_gene
        .iter()
        .zip(b.t_gene.iter())
        .zip(prior.gamma_bg.iter())
    {
        merged_sum += SpecialGamma::ln_gamma(gb + ta + tb).0;
    }
    let s_merged = a.s_size + b.s_size;
    let f_merged = prior.lgamma_gamma - SpecialGamma::ln_gamma(prior.gamma + s_merged).0
        + merged_sum
        - prior.sum_lgamma_gamma_bg;
    f_merged - a.f_cache - b.f_cache
}

/// Bottom-up agglomerative merge by pairwise log Bayes factor.
///
/// Returns one `BhcMerge` per merge event (K_eff − 1 rows, where K_eff is the
/// number of non-empty clusters — those with `effective_size > 0`). Empty
/// clusters are silently skipped and never appear in the tree.
pub fn bhc_merge(input: BhcInput<'_>, gamma: f64) -> Vec<BhcMerge> {
    const BG_EPS: f64 = 1e-9;

    let k = input.k;
    let m = input.m;
    let gamma = gamma.max(1e-6);
    assert_eq!(input.gene_sum.len(), k * m, "gene_sum must be k × m");
    assert_eq!(input.size_sum.len(), k);
    assert_eq!(input.effective_size.len(), k);

    // Pooled empirical marginal bg[g] = (Σ_k T_k,g) / (Σ_k S_k). Floor at BG_EPS so
    // lgamma(γ·bg) stays finite for genes with no mass anywhere (those genes then
    // cancel out of any Bayes factor — see module docstring).
    let mut bg = vec![0.0f64; m];
    let mut total_mass = 0.0f64;
    for c in 0..k {
        if input.effective_size[c] == 0 {
            continue;
        }
        let row = &input.gene_sum[c * m..(c + 1) * m];
        for (dst, &src) in bg.iter_mut().zip(row.iter()) {
            *dst += src;
        }
        total_mass += input.size_sum[c];
    }
    let total_mass = total_mass.max(BG_EPS);
    for p in bg.iter_mut() {
        *p = (*p / total_mass).max(BG_EPS);
    }

    let gamma_bg: Vec<f64> = bg.iter().map(|&p| gamma * p).collect();
    let sum_lgamma_gamma_bg: f64 = gamma_bg
        .iter()
        .map(|&gb| SpecialGamma::ln_gamma(gb).0)
        .sum();
    let prior = Prior {
        gamma,
        gamma_bg,
        sum_lgamma_gamma_bg,
        lgamma_gamma: SpecialGamma::ln_gamma(gamma).0,
    };

    // Rescale (T, S) so effective sample size S_eff = effective_size. Preserves
    // proportions T_g / S; puts log BF magnitudes in an interpretable range
    // (see module docstring).
    let mut arena: Vec<NodeStats> = Vec::with_capacity(k);
    let mut active: Vec<usize> = Vec::with_capacity(k);
    for c in 0..k {
        if input.effective_size[c] == 0 {
            continue;
        }
        let s_raw = input.size_sum[c];
        let n_eff = input.effective_size[c] as f64;
        let scale = if s_raw > 0.0 { n_eff / s_raw } else { 0.0 };
        let t: Vec<f64> = input.gene_sum[c * m..(c + 1) * m]
            .iter()
            .map(|&x| x * scale)
            .collect();
        let f = f_node(&t, n_eff, &prior);
        arena.push(NodeStats {
            t_gene: t,
            s_size: n_eff,
            n_samples: input.effective_size[c] as i32,
            f_cache: f,
            id: c as i32,
        });
        active.push(arena.len() - 1);
    }

    let k_eff = active.len();
    if k_eff < 2 {
        return Vec::new();
    }
    let mut merges: Vec<BhcMerge> = Vec::with_capacity(k_eff - 1);
    let mut next_id: i32 = k as i32;

    while active.len() >= 2 {
        // Maximize log BF; tiebreak on ascending (min_id, max_id).
        let mut best: Option<(usize, usize, f64, (i32, i32))> = None;
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let a = &arena[active[i]];
                let b = &arena[active[j]];
                let bf = pairwise_log_bf(a, b, &prior);
                let key = order_pair(a.id, b.id);
                let pick = match best {
                    None => true,
                    Some((_, _, best_bf, best_key)) => {
                        bf > best_bf || (bf == best_bf && key < best_key)
                    }
                };
                if pick {
                    best = Some((i, j, bf, key));
                }
            }
        }

        let (i, j, log_bf, _) = best.expect("at least one pair exists");
        let ai = active[i];
        let aj = active[j];
        let (left_id, right_id) = order_pair(arena[ai].id, arena[aj].id);

        let new_t: Vec<f64> = arena[ai]
            .t_gene
            .iter()
            .zip(arena[aj].t_gene.iter())
            .map(|(x, y)| x + y)
            .collect();
        let new_s = arena[ai].s_size + arena[aj].s_size;
        let new_n = arena[ai].n_samples + arena[aj].n_samples;
        let new_f = f_node(&new_t, new_s, &prior);

        arena.push(NodeStats {
            t_gene: new_t,
            s_size: new_s,
            n_samples: new_n,
            f_cache: new_f,
            id: next_id,
        });
        let new_arena_idx = arena.len() - 1;

        merges.push(BhcMerge {
            id: next_id,
            left: left_id,
            right: right_id,
            log_bf,
            n_samples: new_n,
        });
        next_id += 1;

        // Remove j first (j > i) so the index for i stays valid.
        active.remove(j);
        active.remove(i);
        active.push(new_arena_idx);
    }

    merges
}

/// Walk the merge tree and collapse children whose `log_bf >= cutoff`.
///
/// Returns dense consensus labels indexed by original cluster id (length `k`).
/// Clusters that were empty (`effective_size == 0` during `bhc_merge` and
/// therefore never appear in any merge) get label `-1`.
///
/// A cutoff of `0.0` is the natural Bayesian break point: positive log BF means
/// the data prefers merging; negative means separating is preferred.
pub fn bhc_cut(merges: &[BhcMerge], k: usize, cutoff: f64) -> Vec<i32> {
    let mut uf = UnionFind::new(k);
    // Representative leaf id per node, indexed by BhcMerge.id. Ids are contiguous
    // K .. K + merges.len() − 1, so we size once up front.
    let mut rep: Vec<usize> = Vec::with_capacity(k + merges.len());
    rep.extend(0..k);
    rep.resize(k + merges.len(), 0);
    // Non-empty originals appear as a direct leaf in exactly one merge.
    let mut referenced = vec![false; k];

    for m in merges {
        if (m.left as usize) < k {
            referenced[m.left as usize] = true;
        }
        if (m.right as usize) < k {
            referenced[m.right as usize] = true;
        }
        let l_rep = rep[m.left as usize];
        let r_rep = rep[m.right as usize];
        if m.log_bf >= cutoff {
            uf.union(l_rep, r_rep);
        }
        rep[m.id as usize] = l_rep.min(r_rep);
    }

    let mut labels = vec![-1i32; k];
    let mut root_to_dense: Vec<Option<i32>> = vec![None; k];
    let mut next_dense: i32 = 0;
    for c in 0..k {
        if !referenced[c] {
            continue;
        }
        let root = uf.find(c);
        let label = match root_to_dense[root] {
            Some(l) => l,
            None => {
                let l = next_dense;
                root_to_dense[root] = Some(l);
                next_dense += 1;
                l
            }
        };
        labels[c] = label;
    }
    labels
}
