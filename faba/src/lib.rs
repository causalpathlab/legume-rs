//! Library surface for `faba`.
//!
//! The crate ships as the `faba` binary (see `main.rs`); this surface exists
//! only so the pure-function statistics below can be driven from integration
//! tests under `tests/`. Nothing else is exported, and nothing else should be:
//! faba's product is the per-cell feature matrices its subcommands write, and
//! every downstream consumer reads those as files rather than linking to them.

/// The two-sample and single-sample statistics the editing caller is built on:
/// beta-binomial and Fisher-exact p-values, and the log odds ratio.
///
/// Deliberately NO FDR adjustment. Neighbouring conversion sites are covered by
/// the same reads, and a read converted at one site is evidence against its
/// unconverted neighbour, so the dependence is not even reliably positive and
/// Benjamini-Hochberg's assumption fails. Editing selects on a marginal p-value
/// and claims no FDR guarantee. Callers whose units genuinely are independent
/// use [`matrix_util::hypothesis::benjamini_hochberg`].
pub mod hypothesis_tests;
