//! Which exposure effects can be estimated at all.
//!
//! Some genes carry no information on the effect: no counts in a topic, a
//! level whose individuals all have zero counts (the effect is unbounded),
//! or a level with too few individuals in a topic. Other failures hit the
//! whole run: an exposure arm with almost no propensity overlap, or a
//! permutation null that barely moves the labels. Such genes are flagged
//! and left out (NA), which costs power but keeps the p-values of the
//! remaining genes on their null distribution. The flags are fixed from the
//! observed data and applied the same way to every permutation draw.

use crate::common::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// Individuals a level needs in a topic.
pub const MIN_INDIVIDUALS_PER_LEVEL: usize = 3;

/// Propensity effective sample size an arm needs.
pub const MIN_ARM_ESS: f32 = 3.0;

/// Distinct relabelings the permutation null needs.
pub const MIN_DISTINCT_PERMUTATIONS: usize = 20;

/// Why an effect was left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flag {
    /// no counts in the topic
    NoCounts,
    /// every individual of some level has zero counts
    LevelZero,
    /// some level has too few individuals with cells in the topic
    LevelSparse,
    /// some exposure arm has too little propensity overlap
    WeakOverlap,
    /// the permutation draws barely move the labels
    DegenerateNull,
}

impl Flag {
    pub fn as_str(&self) -> &'static str {
        match self {
            Flag::NoCounts => "no_counts",
            Flag::LevelZero => "level_zero",
            Flag::LevelSparse => "level_sparse",
            Flag::WeakOverlap => "weak_overlap",
            Flag::DegenerateNull => "degenerate_null",
        }
    }
}

/// Per-gene flags of one topic.
///
/// * `y_di` - counts, gene x individual
/// * `used` - individuals with a level and cells in this topic
/// * `exposure` - level per individual (`n_levels` or more: unassigned)
pub fn topic_flags(
    y_di: &Mat,
    used: &[bool],
    exposure: &[usize],
    n_levels: usize,
    min_per_level: usize,
) -> Vec<Option<Flag>> {
    let mut per_level = vec![0usize; n_levels];
    for (i, &x) in exposure.iter().enumerate() {
        if used[i] && x < n_levels {
            per_level[x] += 1;
        }
    }
    if per_level.iter().any(|&c| c < min_per_level) {
        return vec![Some(Flag::LevelSparse); y_di.nrows()];
    }
    (0..y_di.nrows())
        .map(|d| {
            let mut level_sum = vec![0f32; n_levels];
            for (i, &x) in exposure.iter().enumerate() {
                if used[i] && x < n_levels {
                    level_sum[x] += y_di[(d, i)];
                }
            }
            if level_sum.iter().all(|&s| s <= 0.0) {
                Some(Flag::NoCounts)
            } else if level_sum.iter().any(|&s| s <= 0.0) {
                Some(Flag::LevelZero)
            } else {
                None
            }
        })
        .collect()
}

/// True when the relabelings hold fewer than `min_distinct` distinct ones.
pub fn permutations_degenerate(draws: &[Vec<usize>], min_distinct: usize) -> bool {
    let distinct: HashSet<&Vec<usize>> = draws.iter().collect();
    distinct.len() < min_distinct
}

/// Which (gene, topic) effects enter the contrast.
pub struct EstimableMask {
    /// per topic, per gene
    topic_flags: Vec<Vec<Option<Flag>>>,
    /// run-level flag covering every gene
    run_flag: Option<Flag>,
}

impl EstimableMask {
    pub fn from_topic_flags(topic_flags: &[Vec<Option<Flag>>]) -> Self {
        Self {
            topic_flags: topic_flags.to_vec(),
            run_flag: None,
        }
    }

    /// Flag every gene (a run-level failure).
    pub fn flag_all(&mut self, flag: Flag) {
        self.run_flag.get_or_insert(flag);
    }

    /// Flag of (gene, topic); `None` when it enters the contrast.
    pub fn flag(&self, gene: usize, topic: usize) -> Option<Flag> {
        self.run_flag.or(self.topic_flags[topic][gene])
    }

    /// Flag of a gene: set when no topic is estimable, the first topic's
    /// reason.
    pub fn gene_flag(&self, gene: usize) -> Option<Flag> {
        if self.run_flag.is_some() {
            return self.run_flag;
        }
        let mut reasons = self.topic_flags.iter().map(|f| f[gene]);
        if reasons.clone().any(|f| f.is_none()) {
            None
        } else {
            reasons.next().flatten()
        }
    }

    /// psi(x) = log tau(x) - log tau(0) averaged over the estimable topics
    /// of each gene, gene x (K - 1) non-reference levels; NaN for a gene
    /// with none.
    pub fn mean_log_effect<'a>(&self, psis: impl ExactSizeIterator<Item = &'a Mat>) -> Mat {
        let mut sum: Option<Mat> = None;
        let mut count: Vec<f32> = Vec::new();
        for (k, psi) in psis.enumerate() {
            let (n_genes, n_cols) = psi.shape();
            let s = sum.get_or_insert_with(|| Mat::zeros(n_genes, n_cols - 1));
            count.resize(n_genes, 0.0);
            for d in 0..n_genes {
                if self.flag(d, k).is_none() {
                    for l in 1..n_cols {
                        s[(d, l - 1)] += psi[(d, l)];
                    }
                    count[d] += 1.0;
                }
            }
        }
        let mut out = sum.expect("at least one topic");
        for (d, &c) in count.iter().enumerate() {
            let mut row = out.row_mut(d);
            if c > 0.0 {
                row /= c;
            } else {
                row.fill(f32::NAN);
            }
        }
        out
    }

    /// Genes flagged per reason, for the log.
    pub fn summary(&self, n_genes: usize) -> Vec<(Flag, usize)> {
        let mut counts: HashMap<Flag, usize> = HashMap::default();
        for d in 0..n_genes {
            if let Some(f) = self.gene_flag(d) {
                *counts.entry(f).or_default() += 1;
            }
        }
        let mut out: Vec<(Flag, usize)> = counts.into_iter().collect();
        out.sort_by_key(|(f, _)| f.as_str());
        out
    }
}

#[cfg(test)]
mod tests;
