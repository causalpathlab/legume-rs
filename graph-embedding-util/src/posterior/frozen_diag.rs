//! Read-only geometry of the **frozen other side**, reported alongside every
//! posterior run.
//!
//! # Why this is worth measuring
//!
//! Every sampler here conditions on one side held fixed, so the frozen matrix
//! `Eᵒ` `[n_other × h]` is the design the anchors are regressed against. Two of
//! its properties decide whether a **per-dim** readout — [`super::gate`]'s PIP,
//! [`super::dim_block`]'s inclusion — means anything at all, and neither is
//! visible from the chains:
//!
//! 1. **Collinearity.** If two dims of `Eᵒ` are near-parallel, "which of them does
//!    this anchor load" is not identifiable and independent per-dim marginals will
//!    split the mass between them, looking confident and being wrong on both. The
//!    variance-inflation factor is the standard read: `VIF < 5` means per-dim
//!    marginals are safe, and it is also what says whether a set-valued answer
//!    (SuSiE credible sets, `L > 1`) would buy anything — under low VIF those sets
//!    come out as singletons.
//! 2. **A common mode.** If every frozen row carries a large shared component,
//!    then `⟨e_a, e_o⟩` is dominated for every anchor by its projection onto that
//!    one direction — identical across the other side, so it carries no selection
//!    information while still soaking up the likelihood. Measured on real fits,
//!    [`FrozenSideDiag::common_mode_cos`] runs from 0.16 to **0.96**, so this is
//!    not hypothetical.
//!
//! The two are distinguished by centering: a common mode makes the *raw* Gram
//! look near-rank-1 while the *centered* Gram recovers its rank. Reporting both is
//! what separates "the fit collapsed" from "the fit has a large mean offset",
//! which are different problems with different fixes.
//!
//! This is a diagnostic of the **data**, not of a chain, which is why it lives
//! here and not in [`super::diagnostics`] — every entry point there takes a draw
//! series. The shape follows [`super::index::BiasCalibration`]: a plain struct of
//! measured scalars, computed once and handed to the caller to report.

use super::lnpdf::FrozenSide;
use nalgebra::DMatrix;

/// Geometry of the frozen side, measured once per track before sampling.
///
/// `Serialize` is derived rather than hand-picked into the report: every field
/// here exists *only* to be reported, so a hand-written JSON literal could omit a
/// newly added one and compile clean — the one failure mode a diagnostic must not
/// have. (Contrast [`super::diagnostics::ChainDiag`], which is deliberately
/// hand-picked: the same type is emitted under two prefixes, so the key has to
/// name which chain, and it lives inside a result that also carries the full
/// retained draws.)
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct FrozenSideDiag {
    /// Mean `|cos|` between a frozen row and the mean direction over rows. `→ 1`
    /// means every row is essentially the same direction plus a small residual;
    /// `→ 0` means the rows are spread about the origin.
    pub common_mode_cos: f32,
    /// Participation ratio `(Σλ)² / Σλ²` of the **uncentered** Gram `EᵀE/n`,
    /// in `[1, h]`.
    ///
    /// READ THIS AS VARIANCE CONCENTRATION, NOT USEFUL DIMENSIONALITY. A low value
    /// says the variance is carried by few directions; it does **not** say the
    /// remaining dims are noise. Measured on real 12k BMMC fits at `H=16`, this
    /// reads 2.7 while dims 4–16 *alone* still recover cell type at kNN-purity
    /// 0.676 against a 0.249 baseline. Low-variance directions can carry ample
    /// signal, and reading this field as a capacity estimate is a mistake that has
    /// already been made once.
    pub eff_rank_raw: f32,
    /// Same, for the **column-centered** Gram. Much larger than
    /// [`Self::eff_rank_raw`] ⇒ the apparent low rank is a mean offset (a common
    /// mode), not a genuine collapse.
    pub eff_rank_centered: f32,
    /// Largest `|correlation|` between two distinct dims.
    pub max_abs_corr: f32,
    /// Largest variance-inflation factor `diag(C⁻¹)` over dims (`1` = orthogonal).
    /// Above ~5, per-dim marginals stop being trustworthy on their own.
    pub max_vif: f32,
}

/// Participation ratio `(Σλ)²/Σλ²` of a symmetric PSD matrix's spectrum — a
/// smooth "how many dims are really in use" that needs no eigenvalue cutoff.
fn eff_rank(gram: &DMatrix<f64>) -> f32 {
    let ev = gram.clone().symmetric_eigenvalues();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for l in ev.iter() {
        let l = l.max(0.0); // numerical negatives on a PSD matrix
        s1 += l;
        s2 += l * l;
    }
    if s2 <= 0.0 {
        return 0.0;
    }
    ((s1 * s1) / s2) as f32
}

/// Measure the frozen side's geometry. `O(n·h² + h³)` — negligible against a
/// single sampling sweep, so it runs unconditionally.
///
/// Degenerate inputs return zeros rather than `NaN`: an empty side, or one whose
/// dims are constant, has no geometry to report and a `NaN` would poison the
/// JSON.
#[must_use]
pub fn frozen_side_diag(side: &FrozenSide) -> FrozenSideDiag {
    let h = side.h;
    let n = side.n();
    if n == 0 || h == 0 {
        return FrozenSideDiag::default();
    }
    // f64 throughout: this is a sum over every frozen row, the same widening
    // `cell_projection` uses for its Gram.
    let e = DMatrix::<f64>::from_fn(n, h, |i, j| f64::from(side.e[i * h + j]));
    let inv_n = 1.0 / n as f64;

    let raw_gram = e.tr_mul(&e) * inv_n;
    let mean: Vec<f64> = (0..h).map(|j| e.column(j).sum() * inv_n).collect();
    // Centred Gram in closed form: `Eᶜᵀ Eᶜ / n = EᵀE/n − μμᵀ`. Materialising a
    // centred copy of `E` would cost another `n×h` allocation (12 MB at 100k
    // cells) and a second `O(n·h²)` pass, for an `h×h` rank-one update.
    let ctr_gram = &raw_gram - DMatrix::<f64>::from_fn(h, h, |i, j| mean[i] * mean[j]);

    // Mean |cos| to the shared mean direction. A zero-norm row contributes 0 —
    // it has no direction to agree with, and skipping it would silently change
    // the denominator.
    let mean_norm = mean.iter().map(|m| m * m).sum::<f64>().sqrt();
    let common_mode_cos = if mean_norm <= 0.0 {
        0.0
    } else {
        let acc: f64 = (0..n)
            .map(|i| {
                let row = e.row(i);
                let nrm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                if nrm <= 0.0 {
                    return 0.0;
                }
                let dot: f64 = (0..h).map(|j| row[j] * mean[j]).sum();
                (dot / (nrm * mean_norm)).abs()
            })
            .sum();
        (acc * inv_n) as f32
    };

    // Correlation matrix from the CENTERED Gram (a correlation is centered by
    // definition; using the raw Gram here would report a common mode as
    // collinearity and conflate the two things this struct exists to separate).
    // A constant dim gets `inv_sd = 0`, so it correlates with nothing.
    let inv_sd: Vec<f64> = (0..h)
        .map(|j| {
            let sd = ctr_gram[(j, j)].max(0.0).sqrt();
            if sd > 0.0 {
                1.0 / sd
            } else {
                0.0
            }
        })
        .collect();
    let corr = DMatrix::<f64>::from_fn(h, h, |i, j| match i == j {
        true => 1.0,
        false => ctr_gram[(i, j)] * inv_sd[i] * inv_sd[j],
    });
    let max_abs_corr = (0..h)
        .flat_map(|i| ((i + 1)..h).map(move |j| (i, j)))
        .fold(0.0f64, |m, (i, j)| m.max(corr[(i, j)].abs()));

    // VIF = diag(C⁻¹), via a Cholesky solve against the identity rather than a
    // direct inversion — `cell_projection` states the crate rule ("never an
    // explicit inverse"), and it matters most exactly here, since near-collinear
    // dims are the case this measures. A singular `C` means a dim is an exact
    // combination of the others: infinite inflation, reported as such rather than
    // as a NaN that would read as "not measured".
    let eye = DMatrix::<f64>::identity(h, h);
    let max_vif = corr
        .clone()
        .cholesky()
        .map(|c| c.solve(&eye))
        .or_else(|| corr.lu().solve(&eye))
        .map_or(f32::INFINITY, |inv| {
            (0..h).fold(0.0f64, |m, j| m.max(inv[(j, j)])) as f32
        })
        .max(1.0);

    FrozenSideDiag {
        common_mode_cos,
        eff_rank_raw: eff_rank(&raw_gram),
        eff_rank_centered: eff_rank(&ctr_gram),
        max_abs_corr: max_abs_corr as f32,
        max_vif,
    }
}

#[cfg(test)]
#[path = "frozen_diag_tests.rs"]
mod frozen_diag_tests;
