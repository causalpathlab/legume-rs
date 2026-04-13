//! Multi-level coarsening tree for tree-structured CNV state inference.
//!
//! Builds a tree from multi-level genomic coarsening where:
//! - Leaves = finest-level blocks
//! - Internal nodes = coarser blocks (unions of finer blocks)
//! - Parent-child relationships follow block containment
//!
//! Provides belief propagation (upward + downward) for state inference,
//! replacing the chain HMM with approximate tree inference in O(nodes × K²).
//! The downward pass uses the full parent marginal (no cavity correction),
//! which slightly overestimates each child's influence on its parent.

use log::info;
use nalgebra::DMatrix;

use crate::genomic_coarsening::{greedy_coarsen, GenomicCoarsening};

// ---------------------------------------------------------------------------
// Tree structure
// ---------------------------------------------------------------------------

/// A multi-level coarsening tree.
#[derive(Debug, Clone)]
pub struct CoarseningTree {
    /// Coarsenings at each level (index 0 = finest, last = coarsest).
    pub levels: Vec<GenomicCoarsening>,
    /// Parent mapping: `parent_map[l][i]` = index of parent block at level `l+1`
    /// for block `i` at level `l`. Length = n_levels - 1.
    pub parent_map: Vec<Vec<usize>>,
    /// Block-level aggregated signal at each level: `[n_blocks_l × n_samples]`.
    pub block_signals: Vec<DMatrix<f32>>,
    /// Chromosome boundaries at the finest block level (for output).
    pub finest_chr_block_bounds: Vec<(usize, usize)>,
}

impl CoarseningTree {
    /// Build a coarsening tree from genome-ordered signal at multiple thresholds.
    ///
    /// # Arguments
    /// * `ordered_signal` — `[G_ordered × S]` log(mu_residual) in genome order
    /// * `chr_bounds` — chromosome boundaries: `[(chr_name, start, end)]`
    /// * `corr_thresholds` — correlation thresholds, **decreasing** (finest first).
    ///   E.g., `[0.7, 0.4]` → level 0 at 0.7 (more blocks), level 1 at 0.4 (fewer blocks).
    pub fn build(
        ordered_signal: &DMatrix<f32>,
        chr_bounds: &[(Box<str>, usize, usize)],
        corr_thresholds: &[f32],
    ) -> Self {
        assert!(
            !corr_thresholds.is_empty(),
            "need at least one correlation threshold"
        );

        // Build coarsenings at each level
        let mut levels = Vec::with_capacity(corr_thresholds.len());
        let mut block_signals = Vec::with_capacity(corr_thresholds.len());

        for &threshold in corr_thresholds {
            let coarsening = greedy_coarsen(ordered_signal, chr_bounds, threshold);
            let signal = coarsening.aggregate_to_blocks(ordered_signal);
            block_signals.push(signal);
            levels.push(coarsening);
        }

        // Build parent maps between consecutive levels
        let mut parent_map = Vec::with_capacity(levels.len().saturating_sub(1));
        for l in 0..levels.len().saturating_sub(1) {
            let fine = &levels[l];
            let coarse = &levels[l + 1];
            let mapping = build_parent_mapping(fine, coarse);
            parent_map.push(mapping);
        }

        let finest_chr_block_bounds = levels[0]
            .chr_block_boundaries()
            .iter()
            .map(|(_, s, e)| (*s, *e))
            .collect();

        info!(
            "Coarsening tree: {} levels, blocks = [{}]",
            levels.len(),
            levels
                .iter()
                .map(|l| l.num_blocks().to_string())
                .collect::<Vec<_>>()
                .join(" → "),
        );

        CoarseningTree {
            levels,
            parent_map,
            block_signals,
            finest_chr_block_bounds,
        }
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn n_finest_blocks(&self) -> usize {
        self.levels[0].num_blocks()
    }

    pub fn n_samples(&self) -> usize {
        self.block_signals[0].ncols()
    }

    /// Children of block `parent_idx` at level `l+1`, returned as indices into level `l`.
    pub fn children_of(&self, level_coarse: usize, parent_idx: usize) -> Vec<usize> {
        if level_coarse == 0 {
            return vec![];
        }
        let fine_level = level_coarse - 1;
        self.parent_map[fine_level]
            .iter()
            .enumerate()
            .filter(|(_, &p)| p == parent_idx)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Map each fine-level block to its parent coarse-level block.
///
/// A fine block `[start, end)` is contained in the coarse block whose range
/// covers `start`. Since both levels are genome-ordered and non-overlapping,
/// this is a simple sweep.
fn build_parent_mapping(fine: &GenomicCoarsening, coarse: &GenomicCoarsening) -> Vec<usize> {
    let mut mapping = Vec::with_capacity(fine.num_blocks());
    let mut coarse_idx = 0;

    for fine_block in &fine.blocks {
        // Find the coarse block that contains this fine block's start
        while coarse_idx < coarse.num_blocks() {
            let cb = &coarse.blocks[coarse_idx];
            if fine_block.start >= cb.start && fine_block.start < cb.end {
                break;
            }
            coarse_idx += 1;
        }
        assert!(
            coarse_idx < coarse.num_blocks(),
            "fine block [{}, {}) not contained in any coarse block",
            fine_block.start,
            fine_block.end,
        );
        mapping.push(coarse_idx);
        // Don't increment coarse_idx — next fine block might be in the same coarse block
    }

    mapping
}

// ---------------------------------------------------------------------------
// Tree belief propagation for state inference
// ---------------------------------------------------------------------------

/// Result of tree state inference for one factor.
pub struct TreeStateResult {
    /// Posterior state probabilities at finest level: `[n_finest_blocks × K]`.
    pub posteriors: DMatrix<f32>,
    /// MAP state at each finest-level block.
    pub map_states: Vec<usize>,
}

/// Run tree belief propagation for one factor.
///
/// Computes per-block per-state evidence, then propagates up and down the
/// coarsening tree to get regularized posteriors at the finest level.
///
/// # Arguments
/// * `tree` — the coarsening tree
/// * `partial_residuals` — per-sample, per-block partial residual at finest level: `[B_finest × S]`
/// * `loadings` — per-sample loading for this factor: length S
/// * `emission_means` — per-state emission means: length K (neutral pinned at 0)
/// * `sigma_sq` — noise variance
/// * `transition_prob` — off-diagonal transition probability for parent→child
pub fn tree_state_inference(
    tree: &CoarseningTree,
    partial_residuals: &DMatrix<f32>,
    loadings: &[f32],
    emission_means: &[f32],
    sigma_sq: f32,
    transition_prob: f32,
) -> TreeStateResult {
    let n_levels = tree.n_levels();
    let k = emission_means.len();
    let n_samples = loadings.len();

    // --- Compute per-block per-state local log-evidence at each level ---
    // evidence[l][b, k] = Σ_s Σ_{g ∈ block} log N(R[g,s]; L[s]*μ[k], σ²)
    let inv_sigma_sq = 1.0 / sigma_sq;
    let log_norm = -0.5 * (sigma_sq.ln() + std::f32::consts::TAU.ln());

    // Finest level: compute from partial_residuals directly
    let n_finest = tree.n_finest_blocks();
    let mut finest_evidence = DMatrix::<f32>::zeros(n_finest, k);
    for b in 0..n_finest {
        for kk in 0..k {
            let mut ll = 0.0f32;
            for s in 0..n_samples {
                let diff = partial_residuals[(b, s)] - loadings[s] * emission_means[kk];
                ll += log_norm - 0.5 * diff * diff * inv_sigma_sq;
            }
            finest_evidence[(b, kk)] = ll;
        }
    }

    if n_levels == 1 {
        // Single level: no tree, just use local evidence
        let (posteriors, map_states) = evidence_to_posteriors(&finest_evidence, k);
        return TreeStateResult {
            posteriors,
            map_states,
        };
    }

    // Coarser levels: aggregate from block_signals
    let mut level_evidence: Vec<DMatrix<f32>> = Vec::with_capacity(n_levels);
    level_evidence.push(finest_evidence);
    for l in 1..n_levels {
        let n_blocks = tree.levels[l].num_blocks();
        let mut evidence = DMatrix::<f32>::zeros(n_blocks, k);
        let signal = &tree.block_signals[l];
        for b in 0..n_blocks {
            for kk in 0..k {
                let mut ll = 0.0f32;
                for s in 0..n_samples {
                    let diff = signal[(b, s)] - loadings[s] * emission_means[kk];
                    ll += log_norm - 0.5 * diff * diff * inv_sigma_sq;
                }
                evidence[(b, kk)] = ll;
            }
        }
        level_evidence.push(evidence);
    }

    // --- Build log transition matrix ---
    let log_self = (1.0 - (k as f32 - 1.0) * transition_prob).ln();
    let log_switch = transition_prob.ln();

    // --- Upward pass: finest → coarsest ---
    // up_msg[l][b, k_parent] = logsumexp_{k_child}(log P(k_child | k_parent) + evidence_child)
    let mut up_msgs: Vec<DMatrix<f32>> = Vec::with_capacity(n_levels - 1);
    #[allow(clippy::needless_range_loop)]
    for l in 0..(n_levels - 1) {
        let n_fine = tree.levels[l].num_blocks();
        let n_coarse = tree.levels[l + 1].num_blocks();
        let mut msg = DMatrix::<f32>::zeros(n_coarse, k);

        let mut terms = vec![f32::NEG_INFINITY; k];
        for fine_b in 0..n_fine {
            let parent_b = tree.parent_map[l][fine_b];
            // Compute message from fine_b to parent_b
            for k_parent in 0..k {
                for k_child in 0..k {
                    let log_trans = if k_child == k_parent {
                        log_self
                    } else {
                        log_switch
                    };
                    terms[k_child] = log_trans + level_evidence[l][(fine_b, k_child)];
                }
                msg[(parent_b, k_parent)] += logsumexp(&terms);
            }
        }
        up_msgs.push(msg);
    }

    // --- Downward pass: coarsest → finest ---
    // At each level, compute the "context" from above (parent's marginal minus this child's contribution)
    // Then combine with local evidence to get marginal.

    // Start at coarsest level: marginal = local evidence + upward messages from children
    let top = n_levels - 1;
    let n_top = tree.levels[top].num_blocks();
    let mut top_marginal = DMatrix::<f32>::zeros(n_top, k);
    for b in 0..n_top {
        for kk in 0..k {
            top_marginal[(b, kk)] = level_evidence[top][(b, kk)] + up_msgs[top - 1][(b, kk)];
        }
    }

    // Propagate down
    let mut marginals: Vec<DMatrix<f32>> = vec![DMatrix::zeros(0, 0); n_levels];
    marginals[top] = top_marginal;

    let mut down_terms = vec![0.0f32; k];
    for l in (0..n_levels - 1).rev() {
        let n_fine = tree.levels[l].num_blocks();
        let mut fine_marginal = DMatrix::<f32>::zeros(n_fine, k);

        for fine_b in 0..n_fine {
            let parent_b = tree.parent_map[l][fine_b];

            // Downward message: parent's marginal passed through transition
            // Approximate: uses full parent marginal (slightly overestimates this child's influence)
            for k_child in 0..k {
                for k_parent in 0..k {
                    let log_trans = if k_child == k_parent {
                        log_self
                    } else {
                        log_switch
                    };
                    let parent_ctx = marginals[l + 1][(parent_b, k_parent)];
                    down_terms[k_parent] = log_trans + parent_ctx;
                }
                let down_msg = logsumexp(&down_terms);
                fine_marginal[(fine_b, k_child)] = level_evidence[l][(fine_b, k_child)] + down_msg;
            }
        }

        marginals[l] = fine_marginal;
    }

    // --- Extract posteriors at finest level ---
    let (posteriors, map_states) = evidence_to_posteriors(&marginals[0], k);

    TreeStateResult {
        posteriors,
        map_states,
    }
}

/// Convert log-evidence matrix to normalized posteriors and MAP states.
fn evidence_to_posteriors(log_evidence: &DMatrix<f32>, k: usize) -> (DMatrix<f32>, Vec<usize>) {
    let n = log_evidence.nrows();
    let mut posteriors = DMatrix::<f32>::zeros(n, k);
    let mut map_states = Vec::with_capacity(n);

    for b in 0..n {
        let row: Vec<f32> = (0..k).map(|kk| log_evidence[(b, kk)]).collect();
        let lse = logsumexp(&row);
        let mut best_k = 0;
        let mut best_val = f32::NEG_INFINITY;
        for kk in 0..k {
            let p = (row[kk] - lse).exp();
            posteriors[(b, kk)] = p;
            if row[kk] > best_val {
                best_val = row[kk];
                best_k = kk;
            }
        }
        map_states.push(best_k);
    }

    (posteriors, map_states)
}

fn logsumexp(vals: &[f32]) -> f32 {
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        return f32::NEG_INFINITY;
    }
    let sum: f32 = vals.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_parent_mapping() {
        use crate::genomic_coarsening::GenomicBlock;

        let fine = GenomicCoarsening {
            blocks: vec![
                GenomicBlock {
                    start: 0,
                    end: 5,
                    chromosome: "chr1".into(),
                },
                GenomicBlock {
                    start: 5,
                    end: 10,
                    chromosome: "chr1".into(),
                },
                GenomicBlock {
                    start: 10,
                    end: 15,
                    chromosome: "chr1".into(),
                },
                GenomicBlock {
                    start: 15,
                    end: 20,
                    chromosome: "chr1".into(),
                },
            ],
        };
        let coarse = GenomicCoarsening {
            blocks: vec![
                GenomicBlock {
                    start: 0,
                    end: 10,
                    chromosome: "chr1".into(),
                },
                GenomicBlock {
                    start: 10,
                    end: 20,
                    chromosome: "chr1".into(),
                },
            ],
        };

        let mapping = build_parent_mapping(&fine, &coarse);
        assert_eq!(mapping, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_tree_single_level() {
        // Single level = no tree, just local evidence
        let n_blocks = 10;
        let n_samples = 3;

        // Block signal: first 5 blocks neutral, last 5 gain
        let mut signal = DMatrix::<f32>::zeros(n_blocks, n_samples);
        for b in 5..10 {
            for s in 0..n_samples {
                signal[(b, s)] = 0.5;
            }
        }

        let chr_bounds: Vec<(Box<str>, usize, usize)> = vec![("chr1".into(), 0, n_blocks)];
        let tree = CoarseningTree::build(&signal, &chr_bounds, &[0.5]);

        let loadings = vec![1.0f32; n_samples];
        let means = vec![-0.5, 0.0, 0.4];

        let result = tree_state_inference(&tree, &signal, &loadings, &means, 0.1, 1e-6);

        // First 5 blocks should be neutral (state 1)
        for b in 0..5 {
            assert_eq!(result.map_states[b], 1, "block {} should be neutral", b);
        }
    }
}
