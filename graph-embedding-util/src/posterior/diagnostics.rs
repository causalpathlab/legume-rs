//! Mixing diagnostics for a posterior sweep — mostly **wiring** over
//! `mcmc_util`'s ready-made chain diagnostics, plus the one thing they don't
//! provide: a bracket-fallback proxy.
//!
//! Retained draws arrive as per-node chains of `θ = [e ; b]`.
//! [`chain_diagnostics`] reports, per node:
//!   * **min ESS** across coordinates (`mcmc_util::ess`) — the honest summary; a
//!     healthy chain has ESS a real fraction of the retained count.
//!   * **stuck fraction** — the share of adjacent draws that are byte-identical.
//!     Elliptical slice falls back to the *current* state when the bracket
//!     collapses (`elliptical_slice.rs` `MAX_BRACKET_ITERS`), so a run of
//!     identical draws is that fallback showing through. `mcmc_util` exposes no
//!     counter for it, but the output reveals it — a stalled chain is visible,
//!     not silent.
//!
pub use mcmc_util::engine::ess;

/// Per-node mixing summary.
#[derive(Clone, Copy, Debug)]
pub struct ChainDiag {
    /// Worst-coordinate effective sample size over the retained draws.
    pub min_ess: f32,
    /// Fraction of adjacent draws identical to their predecessor (fallback proxy).
    pub stuck_fraction: f32,
}

/// Effective sample size of one coordinate's series across a node's draws.
/// `draws[t][k]` is coordinate `k` at retained step `t`.
fn coord_ess(draws: &[Vec<f32>], k: usize) -> f32 {
    let series: Vec<f32> = draws.iter().map(|d| d[k]).collect();
    ess(&series)
}

/// Diagnose one node's retained draws (`draws[t]` = `θ` at step `t`, all equal
/// length). Returns `min_ess = draws.len()` and `stuck_fraction = 0` for a chain
/// with `< 2` draws (nothing to assess).
#[must_use]
pub fn chain_diagnostics(draws: &[Vec<f32>]) -> ChainDiag {
    let t = draws.len();
    if t < 2 {
        return ChainDiag {
            min_ess: t as f32,
            stuck_fraction: 0.0,
        };
    }
    let dim = draws[0].len();
    let min_ess = (0..dim)
        .map(|k| coord_ess(draws, k))
        .fold(f32::INFINITY, f32::min);

    let stuck = draws.windows(2).filter(|w| w[0] == w[1]).count();
    ChainDiag {
        min_ess,
        stuck_fraction: stuck as f32 / (t - 1) as f32,
    }
}

/// Diagnose a **scalar** chain directly (e.g. a σ²/π₀ hyperparameter series) —
/// the min-over-coordinates fold of [`chain_diagnostics`] is a no-op on one
/// coordinate, so the interleaved sweeps call this rather than wrapping the chain
/// as a `Vec<Vec<f32>>`.
#[must_use]
pub fn scalar_diagnostics(chain: &[f64]) -> ChainDiag {
    let t = chain.len();
    if t < 2 {
        return ChainDiag {
            min_ess: t as f32,
            stuck_fraction: 0.0,
        };
    }
    let series: Vec<f32> = chain.iter().map(|&v| v as f32).collect();
    let stuck = chain.windows(2).filter(|w| w[0] == w[1]).count();
    ChainDiag {
        min_ess: ess(&series),
        stuck_fraction: stuck as f32 / (t - 1) as f32,
    }
}

/// Aggregate diagnostics over many nodes' chains: the worst (minimum) ESS and the
/// worst (maximum) stuck fraction across nodes — the summary that must be surfaced
/// per run so a stalled chain cannot hide behind healthy neighbours.
#[must_use]
pub fn worst_case(diags: &[ChainDiag]) -> ChainDiag {
    diags.iter().fold(
        ChainDiag {
            min_ess: f32::INFINITY,
            stuck_fraction: 0.0,
        },
        |acc, d| ChainDiag {
            min_ess: acc.min_ess.min(d.min_ess),
            stuck_fraction: acc.stuck_fraction.max(d.stuck_fraction),
        },
    )
}

#[cfg(test)]
#[path = "diagnostics_tests.rs"]
mod diagnostics_tests;
