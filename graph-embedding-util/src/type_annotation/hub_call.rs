//! **Hub call** — catching the features the co-embedding never actually learned.
//!
//! # The laundering
//!
//! [`crate::null_call::live_row`] is the contract every consumer here relies on: an untrained
//! feature row is *deliberately zeroed*, and must be read as **missing data, not an observation of
//! zero**. It rests on an invariant — "a row the model actually trained is never exactly zero" —
//! and that invariant is sound for the raw dictionary `ρ`.
//!
//! But the SIMBA-style co-embedding ([`crate::feature_coembedding`]) is what
//! `{out}.feature_embedding.parquet` actually holds, and it **destroys the zero**:
//!
//! ```text
//! e_g = Σ_i softmax(⟨e_cell_i, ρ_g⟩ / T) · e_cell_i
//! ```
//!
//! For a gene the model never moved off its init, `ρ_g` is noise, the softmax over cells is flat,
//! and the weighted average collapses to the **plain mean of every cell** — the *hub* of the cell
//! cloud. The row comes out with a healthy unit-ish norm and a perfectly plausible coordinate, and
//! `live_row` cannot tell it from a real one.
//!
//! This is not hypothetical. `senna bge` trains on the features that survive its null-QC refit and
//! then **post-hoc projects the rest back in, so downstream tools get a full feature set** (which is
//! the right call — a dropped gene should still have a coordinate). Measured on BMMNC, H=32: of
//! 36,591 rows, 11,096 were projected, and they land as a *degenerate point mass* —
//! 0.0019 from the cell-cloud centre, coordinate spread 0.0069, norms 0.957 ± 0.0024 — while
//! genuinely trained genes sit 0.603 away.
//!
//! # Why it matters
//!
//! A term centroid is the mean of its markers' rows. Let a few hub-parked rows into it and the
//! centroid is dragged to the hub — and the hub is the **worst possible place for a centroid**,
//! because `argmin_t ‖x − c_t‖` makes a centroid that sits at the centre of the cell cloud close to
//! *every* cell at once. It becomes a magnet and swallows the dataset. That is the same failure as
//! the all-zero-signature bug (a zero centroid is at constant distance from everything), arriving by
//! a different route.
//!
//! # The call
//!
//! Measure every gene's distance to the hub **in units of the cell cloud's own radius**
//! `R = mean‖x − hub‖`, and zero the rows that sit within [`HUB_RADIUS_FRAC`] of the centre.
//!
//! The scale is the point of it. A gene can only tell one cell from another if its coordinate is an
//! appreciable distance from the middle of them all; at `d ≪ R` it is very nearly equidistant from
//! every cell and discriminates nothing. On BMMNC the two populations are not close: the projected
//! rows sit at `0.008·R` and the trained ones at `2.4·R`, three orders of magnitude apart.
//!
//! It does **not** matter whether such a gene got there by never being trained or by being
//! genuinely expressed in every cell at the same level — neither can localize a cell type, and both
//! are the wrong thing to average into a centroid. Zeroing is correct either way.
//!
//! A *relative* threshold rather than a fitted null is deliberate. The obvious move is to hand
//! `‖e_g − hub‖²` to [`crate::null_call::chi2_null_call`], and it is **wrong**: that call asks which
//! rows are significantly *above* a fitted null bulk, so on a healthy embedding — where every gene
//! is legitimately far from the hub and there is no low spike at all — nothing is above the bulk,
//! nothing is called live, and it zeroes the entire gene space. The test
//! `an_embedding_with_nothing_at_the_hub_is_left_alone` exists because that is exactly what happened.
//! The rule here degenerates safely: no point mass ⇒ nothing zeroed.
//!
//! Rows that are zeroed restore the `live_row` invariant, and every existing consumer (the
//! centroids, the marker bootstrap's live panel, the panel null's pool, the support null's pool, the
//! norm strata) then treats them as missing data for free, without any of them knowing this module
//! exists. A type that loses too many markers this way falls below `MIN_LIVE_MARKERS` and stops
//! competing — the honest outcome: it was never located.
//!
//! **The cure is upstream.** A marker that lands here was never trained into the embedding, and no
//! downstream statistic can recover it — re-run the fit with `--must-train-features <panel>`.

#[cfg(test)]
mod tests;

use log::{info, warn};
use matrix_util::dmatrix_io::DMatrix;
use rayon::prelude::*;

/// How close to the hub a gene must be, as a fraction of the cell cloud's own radius, before it is
/// judged to carry no localizing information at all.
///
/// Deliberately small. The populations this separates are three orders of magnitude apart on real
/// data (`0.008·R` vs `2.4·R`), so there is no need to cut anywhere near the live genes — and a
/// *false positive* here silently deletes a real marker, which is the one thing this must not do.
const HUB_RADIUS_FRAC: f32 = 0.05;

/// Zero every feature row that the co-embedding parked at the hub of the cell cloud.
///
/// `beta_flat` is `[g × h]` row-major and is modified in place; `cell_emb` is `[n × h]`. Returns
/// how many rows were zeroed. See the module doc — this restores the [`crate::null_call::live_row`]
/// invariant that the co-embedding breaks.
pub(super) fn zero_hub_parked(
    beta_flat: &mut [f32],
    cell_emb: &DMatrix<f32>,
    g: usize,
    h: usize,
) -> usize {
    let n = cell_emb.nrows();
    if n < 2 || g == 0 || h == 0 {
        return 0;
    }

    // The hub: the centre of the cell cloud, which is exactly where a flat softmax lands a gene.
    let hub: Vec<f32> = (0..h)
        .map(|j| cell_emb.column(j).iter().sum::<f32>() / n as f32)
        .collect();

    // The cell cloud's own radius — the scale against which "close to the centre" is judged.
    let radius = (0..n)
        .map(|i| {
            (0..h)
                .map(|j| {
                    let d = cell_emb[(i, j)] - hub[j];
                    d * d
                })
                .sum::<f32>()
                .sqrt()
        })
        .sum::<f32>()
        / n as f32;
    if !(radius > 0.0) {
        // Every cell is at the same point; there is no cloud, so there is no hub to call against.
        return 0;
    }
    let cutoff = HUB_RADIUS_FRAC * radius;

    let parked: Vec<bool> = (0..g)
        .into_par_iter()
        .map(|i| {
            let d2: f32 = beta_flat[i * h..(i + 1) * h]
                .iter()
                .zip(&hub)
                .map(|(&x, &m)| {
                    let d = x - m;
                    d * d
                })
                .sum();
            d2.sqrt() < cutoff
        })
        .collect();

    let n_zeroed = parked.iter().filter(|&&p| p).count();
    if n_zeroed == 0 {
        return 0;
    }
    for (i, &p) in parked.iter().enumerate() {
        if p {
            beta_flat[i * h..(i + 1) * h].fill(0.0);
        }
    }

    let frac = n_zeroed as f64 / g as f64;
    info!(
        "hub call: {n_zeroed}/{g} feature rows ({:.0}%) sit at the centre of the cell cloud — the \
         co-embedding's signature for a gene it never learned. Zeroed, so they are read as missing \
         data rather than averaged into a centroid.",
        100.0 * frac
    );
    if frac > 0.5 {
        warn!(
            "hub call: MORE THAN HALF the feature rows ({:.0}%) are parked at the cell-cloud \
             centre. This embedding has barely learned a gene space at all; a nearest-centroid \
             annotation over it will be noise whatever the markers say.",
            100.0 * frac
        );
    }
    n_zeroed
}
