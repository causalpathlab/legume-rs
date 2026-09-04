//! Posterior jitter of the pseudobulk profiles between phase-1 rounds.
//!
//! The collapse fits a Gamma posterior per `(gene, pseudobulk)`, and phase 1
//! normally trains on its MEAN, discarding the variance. Redrawing every
//! profile from its posterior between rounds injects structured,
//! magnitude-aware noise — a dropout analogue whose scale is set by the data
//! rather than by a hyperparameter — and is the direct test of the
//! oversmoothing account of what a pseudobulk loses.
//!
//! A draw changes only the sampling WEIGHTS: a count reaches the objective
//! solely through the positive and degree-negative pickers, and a Gamma draw is
//! positive wherever the mean is, so the support is fixed. Each round therefore
//! refreshes the samplers in place ([`StratifiedSampler::rejitter`]) rather
//! than rebuilding them: `nnz` seeded draws per level, no triplet list, no
//! bucketing. The triplets themselves stay on the mean — phase 2 projects
//! against them, and jitter is a phase-1 device.

use super::axes::AxisData;
use super::setup::gather_to_unified_axis;
use crate::loss::StratifiedSampler;
use data_beans_alg::collapse_data::CollapsedOut;
use matrix_util::rand_util::mix_seed;

/// What one jitter round did, for the log line.
pub(super) struct JitterReport {
    pub levels: usize,
    /// Mean over levels of `Σ|new − old| / Σ old` over the sampling weights.
    pub mean_rel_change: f64,
    pub secs: f64,
}

/// Epochs round `round` of `rounds` gets out of `total`: an even split with the
/// remainder on the first rounds, so the rounds sum to `total` exactly and a
/// single round is the whole budget.
#[must_use]
pub(super) fn epochs_for_round(total: usize, rounds: usize, round: usize) -> usize {
    if rounds == 0 {
        return total;
    }
    total / rounds + usize::from(round < total % rounds)
}

/// Redraw every level's pseudobulk profiles from their Gamma posterior and
/// refresh that level's sampler in place. `seed` should already carry the
/// round; each level salts it once more.
pub(super) fn rejitter_levels(
    ax: &mut AxisData,
    collapsed_levels: &[CollapsedOut],
    n_features: usize,
    feature_to_backend: &[usize],
    seed: u64,
) -> anyhow::Result<JitterReport> {
    let t0 = std::time::Instant::now();
    anyhow::ensure!(
        ax.level_axes.len() == collapsed_levels.len(),
        "jitter: {} sampler level(s) against {} collapse level(s)",
        ax.level_axes.len(),
        collapsed_levels.len()
    );
    let mut acc = 0.0f64;
    for (level, (collapsed, (_, sampler))) in collapsed_levels
        .iter()
        .zip(ax.level_axes.iter_mut())
        .enumerate()
    {
        // The same table phase 1 trained on: the batch-adjusted profile when
        // the collapse produced one, else the observed one.
        let param = collapsed
            .mu_adjusted
            .as_ref()
            .unwrap_or(&collapsed.mu_observed);
        anyhow::ensure!(
            param.has_shape_stat(),
            "jitter needs the collapse's shape statistics at level {level}; the \
             pseudobulks were built without `keep_shape_stats`"
        );
        let drawn = param.posterior_sample_seeded(mix_seed(seed, level as u64))?;
        let drawn = gather_to_unified_axis(&drawn, n_features, feature_to_backend);
        acc += StratifiedSampler::rejitter(sampler, n_features, &|pb, row| {
            drawn[(row as usize, pb as usize)]
        });
    }
    let levels = collapsed_levels.len();
    Ok(JitterReport {
        levels,
        mean_rel_change: if levels == 0 {
            0.0
        } else {
            acc / levels as f64
        },
        secs: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests;
