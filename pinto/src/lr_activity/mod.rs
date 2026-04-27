//! Posthoc directional ligand→receptor activity test per link community.
//!
//! Consumes the output of `pinto lc` (edge→community assignments) and a
//! user-supplied directional ligand→receptor list, and tests whether each
//! LR pair shows coherent activity within each community.
//!
//! ## Design
//!
//! 1. **Pseudobulk samples** = `(batch × propensity-bin)`. The propensity
//!    bin is the sign-LSH binary code of an SVD'd random projection of the
//!    full expression matrix (`data_beans_alg::binary_sort_columns`); cells
//!    with similar broad expression land in the same bin.
//! 2. **Soft community membership per role.** From `lc` edges, each cell
//!    `i` gets `p_send[i, c]` (fraction of `i`'s incident edges in
//!    community `c` on which `i` is sender) and `p_recv[i, c]` (receiver
//!    analogue), row-normalised per cell.
//! 3. **Role-weighted pseudobulk per (community, sample).** For each LR
//!    gene we accumulate `Σ_i x[g, i] · p_role[i, c] · 1{sample(i)=s}`
//!    plus the matching weight `w_role[c, s]`. Sender and receiver
//!    pseudobulks are computed separately and matched by sample id.
//! 4. **Statistic.** Per `(batch, community, LR pair)` a weighted Spearman
//!    rank correlation between `L_send_s` and `R_recv_s` across the
//!    stratum's samples, sample-weighted by `√(w_send · w_recv)`.
//! 5. **Null.** Sample-level permutation of `L` within propensity-stratified
//!    buckets (top `shuffle_stratify_dim` bits of the propensity code).
//!    Stratification preserves the cell-population marginal — without it,
//!    free shuffles across populations pick up cell-type-marginal
//!    correlations and become anti-conservative.
//! 6. **p-values.** Both empirical (`1/(n+1)`-floored) and parametric
//!    (one-sided Gaussian tail of `z`); BH within batch is applied to the
//!    parametric p so q-values aren't pinned at the resolution floor.
//!
//! ## Statistical justification
//!
//! Pseudobulk-then-test follows the SC-DE consensus that cell-level
//! inference inflates type-I error (Squair 2021 *Nat Commun* 12:5692;
//! Crowell 2020 *Nat Commun* 11:6077; muscat). Stratified permutation is
//! a textbook restricted-exchangeability null (Good 2005 *Permutation
//! Tests*; Pesarin & Salmaso 2010 *Permutation Tests for Complex Data*).
//! Stratifying samples by a coarse expression-propensity binary code is
//! structurally similar to propensity-score stratification in observational
//! causal inference (Rosenbaum & Rubin 1983 *Biometrika* 70:41–55; with
//! the over-conditioning caveat in Rosenbaum 1984 *J R Stat Soc A*
//! 147:656–666 — mitigated here by random projection over all genes vs
//! the few-hundred LR genes in the test, plus masking to broad-stroke top
//! bits). The propensity codes themselves are sign-random-projection LSH
//! on SVD axes (Charikar 2002 *STOC '02*). The motivation is that
//! cell-label permutation nulls in the standard CCC stack (CellChat,
//! CellPhoneDB, Squidpy, NicheNet, LIANA, COMMOT) pick up
//! cell-type-marginal correlations as significant L→R coupling — see
//! Dimitrov 2022 *Nat Commun* 13:3224 for the field benchmark.

pub mod args;
pub mod fit;
pub mod io;
pub mod outputs;

pub use args::SrtLrActivityArgs;
pub use fit::fit_srt_lr_activity;
