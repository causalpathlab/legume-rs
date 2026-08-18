//! The block-level machinery both sweeps call: the per-block RNG seed, the pooled gene-side
//! warm start, the profiled intercept, the convergence report, one `dim_block` invocation,
//! and the anchor → feature-row scatters. They live together because the two sweeps have to
//! agree on them exactly — a second copy of any one is a silent divergence between the paths.

use super::{BetaTerm, PbGibbsConfig, SpliceTracks};
use crate::posterior::diagnostics::ChainDiag;
use crate::posterior::dim_block::{dim_block, DimBlockConfig, DimBlockResult, HyperState};
use crate::posterior::lnpdf::{FrozenSide, NodeTerm};
use crate::posterior::pb_index::{AnchorMap, FeatureSide};
use log::info;

/// The intercept the profile likelihood maximised out, recovered for a finished
/// loading: `b_a* = ln(T_a) − ln(scale · Σ_o exp(⟨e_a + off, e_o⟩ + b_o))`.
///
/// Max-shifted and accumulated in `f64` — the exponents span the whole bias range,
/// which is exactly where a naive sum loses its low bits. An anchor with no counts
/// has no rate to match and is parked at `-SCORE_CLAMP` (rate ≈ 0), matching
/// [`super::index::ContrastiveIndex::calibrate_anchor_bias`].
/// Report what a run's chains are actually worth, at `info`, plus a `warn` when they are
/// not worth much.
///
/// Both numbers exist because they fail independently. `min_ess` says how much the draws
/// are WORTH; `rhat` says whether they are draws from one distribution at all — a chain
/// that never left its starting region can have a healthy ESS and an R̂ of 3, because it
/// is mixing efficiently inside the wrong place. The bracket-fallback fraction is the
/// third: a fallback is the slice sampler's analogue of a rejected move, and a run where
/// most coordinates stall still returns a full table of plausible-looking numbers.
///
/// This matters more now than it used to. With phase 1 SAMPLED rather than trained there
/// is no point estimate to fall back on, so an unconverged chain is not a caveat on the
/// output — it is the output.
pub(super) fn report_convergence(
    label: &str,
    sigma_diag: &[ChainDiag],
    pi0_diag: &[ChainDiag],
    // `None` = this line does not own a fallback count. Deliberately not `(0, 0)`:
    // "0 of 0" prints as 0.00% and reads as "nothing stalled", which is the opposite of
    // "not measured here". A reporting path that exists to stop bad numbers being believed
    // must not itself invent a clean one.
    bracket: Option<(usize, usize)>,
) {
    // Split-R̂ over the hyper chains. `1.01` is the conventional pass; past `1.1` the
    // chain has not converged and the per-dim numbers should not be read as a posterior.
    const RHAT_WARN: f32 = 1.1;
    let worst = |d: &[ChainDiag]| d.iter().map(|c| c.rhat).fold(1.0f32, f32::max);
    let failing = |d: &[ChainDiag]| d.iter().filter(|c| c.rhat > RHAT_WARN).count();
    let min_ess = |d: &[ChainDiag]| d.iter().map(|c| c.min_ess).fold(f32::INFINITY, f32::min);

    // The denominator is transitions ATTEMPTED, not every `(anchor, dim)`: an excluded
    // coordinate takes a prior draw with no bracket and cannot fall back, so counting it
    // would divide by `1/(1−π₀)` too many and make a stalled run read as healthy — at a
    // measured `π₀ ≈ 0.885` that is a factor of nearly nine.
    let frac = bracket.and_then(|(f, n)| (n > 0).then(|| f as f64 / n as f64));
    let bracket_note = match (bracket, frac) {
        (Some((f, n)), Some(p)) => format!("bracket fallbacks {f}/{n} ({:.2}%)", 100.0 * p),
        (Some((f, _)), None) => format!("bracket fallbacks {f}/0 (no transitions attempted)"),
        (None, _) => "bracket fallbacks counted on the run's other line".to_string(),
    };
    info!(
        "{label}: split-R̂ worst {:.3} (σ₀²) / {:.3} (π₀), {} of {} dims over {RHAT_WARN}; \
         min ESS {:.1} / {:.1}; {bracket_note}",
        worst(sigma_diag),
        worst(pi0_diag),
        failing(sigma_diag) + failing(pi0_diag),
        sigma_diag.len() + pi0_diag.len(),
        min_ess(sigma_diag),
        min_ess(pi0_diag),
    );
    let n_bad = failing(sigma_diag) + failing(pi0_diag);
    if n_bad > 0 {
        log::warn!(
            "{n_bad} hyper chain(s) have split-R̂ above {RHAT_WARN}, so those dims are NOT \
             stationary — their σ₀²/π₀ and every PIP that conditioned on them describe where \
             the chain happened to be, not a posterior. Treat them as unidentified rather \
             than as measurements, and raise --posterior N before reading them."
        );
    }
    if frac.is_some_and(|p| p > 0.25) {
        log::warn!(
            "{:.0}% of slice transitions exhausted their bracket and fell back to the current \
             value, so most coordinates are not moving. The tables will still be fully \
             populated; they are not a sample.",
            100.0 * frac.unwrap_or(0.0),
        );
    }
}

pub(super) fn profiled_bias(
    total: f64,
    e_a: &[f32],
    off: Option<&[f32]>,
    partition: &[u32],
    scale: f64,
    side: &FrozenSide<'_>,
) -> f32 {
    if total <= 0.0 {
        return -(crate::cell_projection::SCORE_CLAMP as f32);
    }
    let h = side.h;
    let sc = |o: u32| -> f64 {
        let e_o = &side.e[o as usize * h..(o as usize + 1) * h];
        let dot: f64 = match off {
            None => e_a
                .iter()
                .zip(e_o)
                .map(|(a, v)| f64::from(*a) * f64::from(*v))
                .sum(),
            Some(f) => e_a
                .iter()
                .zip(f)
                .zip(e_o)
                .map(|((a, b), v)| (f64::from(*a) + f64::from(*b)) * f64::from(*v))
                .sum(),
        };
        dot + f64::from(side.b[o as usize])
    };
    let m = partition
        .iter()
        .map(|&o| sc(o))
        .fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return 0.0;
    }
    let s: f64 = partition.iter().map(|&o| (sc(o) - m).exp()).sum();
    let log_part = m + (scale * s).max(f64::MIN_POSITIVE).ln();
    (total.ln() - log_part).clamp(
        -(crate::cell_projection::SCORE_CLAMP),
        crate::cell_projection::SCORE_CLAMP,
    ) as f32
}

/// Per-block RNG seed that stays distinct at every sweep, including sweep 0.
///
/// The obvious `seed ^ (sweep * salt)` collapses to `seed` for every salt when
/// `sweep == 0`, so on the sweep that seeds the whole chain the β, δ and pb blocks
/// would draw the identical stream of normals and uniforms — the two gates'
/// proposals perfectly correlated exactly where it matters most.
pub fn block_seed(seed: u64, salt: u64, sweep: usize) -> u64 {
    seed.rotate_left(17)
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (sweep as u64)
            .wrapping_add(1)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// Gene-side warm start `[n_anchors × h]`, pooled from the phase-1 feature rows.
///
/// Under a grouping several rows share an anchor (gem's spliced + unspliced both
/// carry `β_g`), so the anchor takes the FIRST mapped row rather than a sum —
/// they are copies of one parameter, and adding them would double it.
pub(super) fn warm_start_genes(
    feat: &FeatureSide<'_>,
    anchors: Option<&AnchorMap<'_>>,
    n_anchors: usize,
    h: usize,
) -> Vec<f32> {
    match anchors {
        None => feat.e_feat.to_vec(),
        Some(a) => {
            let mut out = vec![0.0f32; n_anchors * h];
            let mut seen = vec![false; n_anchors];
            for (uid, &an) in a.row_to_anchor.iter().enumerate() {
                if an == u32::MAX || seen[an as usize] {
                    continue;
                }
                let (s, d) = (an as usize * h, uid * h);
                out[s..s + h].copy_from_slice(&feat.e_feat[d..d + h]);
                seen[an as usize] = true;
            }
            out
        }
    }
}

/// One gene-side block: build the anchor terms against the current frozen side, run a
/// single `dim_block` sweep, hand back the draw.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_block(
    pos: &[Vec<(u32, f32)>],
    partition: &[u32],
    partition_scale: f64,
    side: &FrozenSide<'_>,
    warm: &[f32],
    offset: Option<&[f32]>,
    z_allowed: Option<Vec<bool>>,
    init_z: Option<Vec<bool>>,
    hyper: Option<HyperState>,
    cfg: &PbGibbsConfig,
    sweep: usize,
    salt: u64,
    label: &str,
) -> DimBlockResult {
    let h = side.h;
    let nodes: Vec<NodeTerm> = pos
        .iter()
        .enumerate()
        .map(|(a, p)| {
            let mut n = NodeTerm::new(p, partition, partition_scale);
            n.offset = offset.map(|o| &o[a * h..(a + 1) * h]);
            n
        })
        .collect();
    let mut bcfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, salt, sweep))
        .with_init_beta(warm.to_vec())
        .with_label(label)
        .quiet();
    bcfg.transitions_per_dim = cfg.transitions_per_dim;
    bcfg.stick_alpha = cfg.stick_alpha;
    if let Some(mask) = z_allowed {
        bcfg = bcfg.with_z_allowed(mask);
    }
    if let Some(z) = init_z {
        bcfg = bcfg.with_init_z(z);
    }
    // One sweep per call, so the slab variance and its half-Cauchy auxiliary only
    // form a chain if the driver hands them back — see `DimBlockConfig::init_hyper`.
    if let Some(hs) = hyper {
        bcfg = bcfg.with_init_hyper(hs);
    }
    dim_block(&nodes, side, &bcfg)
}

/// Effective per-row loading for the pb block: `β_g` on a spliced row, `β_g + δ_g`
/// on an unspliced one, and the phase-1 MAP for any row no gene claims.
pub(super) fn scatter_splice_to_rows(
    e_beta: &[f32],
    e_delta: &[f32],
    tracks: &SpliceTracks<'_>,
    map_fallback: &[f32],
    h: usize,
    out: &mut [f32],
) {
    out.copy_from_slice(map_fallback);
    for (uid, (&g, &unspliced)) in tracks
        .row_to_gene
        .iter()
        .zip(tracks.unspliced_rows)
        .enumerate()
    {
        if g == u32::MAX {
            continue;
        }
        let (s, dst) = (g as usize * h, uid * h);
        for k in 0..h {
            out[dst + k] = e_beta[s + k] + if unspliced { e_delta[s + k] } else { 0.0 };
        }
    }
}

/// Scatter anchor loadings out to the feature-row axis, which is what the pb
/// block scores against.
///
/// Without a grouping the two axes are the same and this is a copy. With one,
/// every row mapped to an anchor takes that anchor's current draw, and every
/// dropped row keeps `map_fallback` — its phase-1 MAP. Dropped rows are not
/// sampled but they ARE observed, so they belong in the pb conditional.
/// Scatter each gate's profiled intercept onto the rows it describes.
///
/// A row takes δ's intercept when it is unspliced and β's SPLICED term otherwise, because
/// those are the blocks that last moved each track. Both are read through
/// [`DimBlockResult::intercept`], so the term stride lives on the result and not here.
///
/// **No `u32::MAX` sentinel to skip, unlike the anchor-map scatters.** `row_to_gene` comes
/// from `intern_gene_keys`, which assigns `gid = gene_ids.len()` and pushes one for EVERY
/// row unconditionally — so it is dense in `0..n_genes` by construction and has no code
/// path that emits a sentinel. The sentinels live only in the per-track maps derived from
/// it (`spliced_map` / `unspliced_map`), which is why the sibling scatters do check. The
/// assertion below is the invariant, not the prose.
pub(super) fn scatter_splice_bias_to_rows(
    beta: &DimBlockResult,
    delta: &DimBlockResult,
    tracks: &SpliceTracks<'_>,
    fallback: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(
        beta.n_terms,
        BetaTerm::COUNT,
        "the β block should carry one term per splice track"
    );
    debug_assert!(
        tracks
            .row_to_gene
            .iter()
            .all(|&g| (g as usize) < tracks.n_genes),
        "row_to_gene must be dense in 0..n_genes — a sentinel here would index the wrong \
         gene's intercept rather than be skipped"
    );
    out.copy_from_slice(fallback);
    for (row, &g) in tracks.row_to_gene.iter().enumerate() {
        let g = g as usize;
        out[row] = if tracks.unspliced_rows[row] {
            delta.intercept(g, 0)
        } else {
            beta.intercept(g, BetaTerm::SPLICED)
        };
    }
}

pub(super) fn scatter_to_rows(
    e_anchor: &[f32],
    anchors: Option<&AnchorMap<'_>>,
    map_fallback: &[f32],
    h: usize,
    out: &mut [f32],
) {
    match anchors {
        None => out.copy_from_slice(e_anchor),
        Some(a) => {
            out.copy_from_slice(map_fallback);
            for (uid, &an) in a.row_to_anchor.iter().enumerate() {
                if an == u32::MAX {
                    continue;
                }
                let (s, d) = (an as usize * h, uid * h);
                out[d..d + h].copy_from_slice(&e_anchor[s..s + h]);
            }
        }
    }
}
