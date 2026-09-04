//! Read-only geometry of an `[n × h]` embedding table: how many of its `h`
//! directions are actually in use, and how much its rows share one direction.
//!
//! One function serves every table with that shape — a cell embedding, a
//! per-gene loading, a module dictionary — because the question is the same for
//! each: "of the `h` dimensions this was given, how many does it use, and is the
//! apparent answer a genuine collapse or a large mean offset?" The two are
//! distinguished by centering: a common mode makes the *raw* Gram look
//! near-rank-1 while the *centered* Gram recovers its rank. Reporting both is
//! what separates "the fit collapsed" from "the fit has a large mean offset",
//! which are different problems with different fixes.
//!
//! Lives here rather than in any one engine because the number has to mean the
//! same thing across engines for a comparison to be a comparison — and because
//! it was first written inside a sampler diagnostic that has since been removed,
//! where nothing outside that sampler could reach it.

use nalgebra::DMatrix;
use rayon::prelude::*;

/// Geometry of an embedding table, measured once and handed to the caller to
/// report. A struct of measured scalars; nothing here decides anything.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EmbeddingGeometry {
    /// Rows measured (units: cells, genes, modules).
    pub n_rows: usize,
    /// Columns (the embedding dimension `h`).
    pub h: usize,
    /// Mean `|cos|` between a row and the mean direction over rows. `→ 1`
    /// means every row is essentially the same direction plus a small residual;
    /// `→ 0` means the rows are spread about the origin.
    pub common_mode_cos: f32,
    /// SIGNED mean cosine over distinct row pairs, zero-norm rows excluded.
    /// Computed in closed form from the sum of unit rows, so it is one `O(n·h)`
    /// pass rather than `n²` pairs. A balanced cloud reads `−1/(n−1)`, not 0.
    pub mean_pairwise_cos: f32,
    /// Participation ratio `(Σλ)² / Σλ²` of the **uncentered** Gram `EᵀE/n`,
    /// in `[1, h]`.
    ///
    /// READ THIS AS VARIANCE CONCENTRATION, NOT USEFUL DIMENSIONALITY. A low value
    /// says the variance is carried by few directions; it does **not** say the
    /// remaining dims are noise. On real fits this has read under 3 of 16 while
    /// the dims *beyond* the first few still recovered cell type well above a
    /// null. Low-variance directions can carry ample signal, and reading this
    /// field as a capacity estimate is a mistake that has already been made once.
    pub eff_rank_raw: f32,
    /// Same, for the **column-centered** Gram. Much larger than
    /// [`Self::eff_rank_raw`] ⇒ the apparent low rank is a mean offset (a common
    /// mode), not a genuine collapse.
    pub eff_rank_centered: f32,
    /// Largest `|correlation|` between two distinct dims.
    pub max_abs_corr: f32,
    /// Largest variance-inflation factor `diag(C⁻¹)` over dims (`1` =
    /// orthogonal). Above ~5, per-dim readouts stop being trustworthy on their
    /// own.
    pub max_vif: f32,
}

/// Participation ratio `(Σλ)²/Σλ²` of a symmetric PSD matrix's spectrum — a
/// smooth "how many dims are really in use" that needs no eigenvalue cutoff.
/// A zero matrix reports `0`, not NaN.
#[must_use]
pub fn participation_ratio(gram: &DMatrix<f64>) -> f32 {
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

/// Per-thread accumulator for the single row pass: the upper triangle of the
/// raw Gram, the column sums, the sum of unit rows, and how many rows had a
/// direction at all.
struct RowPass {
    gram_upper: Vec<f64>,
    col_sum: Vec<f64>,
    unit_sum: Vec<f64>,
    live: usize,
}

impl RowPass {
    fn zero(h: usize) -> Self {
        Self {
            gram_upper: vec![0.0; h * h],
            col_sum: vec![0.0; h],
            unit_sum: vec![0.0; h],
            live: 0,
        }
    }

    fn add_row(mut self, row: &[f64]) -> Self {
        let h = row.len();
        for (i, &ri) in row.iter().enumerate() {
            self.col_sum[i] += ri;
            // Row `i` of the upper triangle: entries `(i, i..h)`.
            let upper = &mut self.gram_upper[i * h + i..(i + 1) * h];
            for (g, &rj) in upper.iter_mut().zip(&row[i..]) {
                *g += ri * rj;
            }
        }
        let nrm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 0.0 {
            for (u, &r) in self.unit_sum.iter_mut().zip(row) {
                *u += r / nrm;
            }
            self.live += 1;
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.gram_upper.iter_mut().zip(other.gram_upper) {
            *a += b;
        }
        for (a, b) in self.col_sum.iter_mut().zip(other.col_sum) {
            *a += b;
        }
        for (a, b) in self.unit_sum.iter_mut().zip(other.unit_sum) {
            *a += b;
        }
        self.live += other.live;
        self
    }
}

/// Measure the geometry of `e` (`rows` = units, `cols` = dims). `O(n·h² + h³)`,
/// with the `O(n·h²)` row pass parallel over rows.
///
/// Degenerate inputs return zeros rather than `NaN`: an empty table, or one
/// whose dims are constant, has no geometry to report.
#[must_use]
pub fn embedding_geometry(e: &DMatrix<f32>) -> EmbeddingGeometry {
    let (n, h) = (e.nrows(), e.ncols());
    if n == 0 || h == 0 {
        return EmbeddingGeometry {
            n_rows: n,
            h,
            ..Default::default()
        };
    }
    // Row-major f64 copy. nalgebra is column-major, so the transpose's buffer
    // *is* row-major of the original; f64 because every quantity below is a sum
    // over all rows.
    let rm: Vec<f64> = e.transpose().iter().map(|&x| f64::from(x)).collect();
    let inv_n = 1.0 / n as f64;

    let pass = rm
        .par_chunks(h)
        .fold(|| RowPass::zero(h), RowPass::add_row)
        .reduce(|| RowPass::zero(h), RowPass::merge);

    let raw_gram = DMatrix::<f64>::from_fn(h, h, |i, j| {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        pass.gram_upper[a * h + b] * inv_n
    });
    let mean: Vec<f64> = pass.col_sum.iter().map(|c| c * inv_n).collect();
    // Centred Gram in closed form: `Eᶜᵀ Eᶜ / n = EᵀE/n − μμᵀ`. Materialising a
    // centred copy would cost another `n×h` pass for an `h×h` rank-one update.
    let ctr_gram = &raw_gram - DMatrix::<f64>::from_fn(h, h, |i, j| mean[i] * mean[j]);

    // Signed mean over distinct pairs of unit rows, from the sum of unit rows:
    // `Σ_{i≠j} ê_i·ê_j = ‖Σ_i ê_i‖² − m` over the `m` rows that have a direction.
    let m = pass.live;
    let mean_pairwise_cos = if m < 2 {
        0.0
    } else {
        let s2: f64 = pass.unit_sum.iter().map(|x| x * x).sum();
        let m = m as f64;
        ((s2 - m) / (m * (m - 1.0))) as f32
    };

    // Mean |cos| to the shared mean direction. A zero-norm row contributes 0 —
    // it has no direction to agree with, and skipping it would silently change
    // the denominator.
    let mean_norm = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
    let common_mode_cos = if mean_norm <= 0.0 {
        0.0
    } else {
        let acc: f64 = rm
            .par_chunks(h)
            .map(|row| {
                let nrm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                if nrm <= 0.0 {
                    return 0.0;
                }
                let dot: f64 = row.iter().zip(&mean).map(|(r, m)| r * m).sum();
                (dot / (nrm * mean_norm)).abs()
            })
            .sum();
        (acc * inv_n) as f32
    };

    // Correlation matrix from the CENTERED Gram (a correlation is centered by
    // definition; the raw Gram would report a common mode as collinearity and
    // conflate the two things this struct exists to separate). A constant dim
    // gets `inv_sd = 0`, so it correlates with nothing.
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
    let corr = DMatrix::<f64>::from_fn(h, h, |i, j| {
        if i == j {
            1.0
        } else {
            ctr_gram[(i, j)] * inv_sd[i] * inv_sd[j]
        }
    });
    let max_abs_corr = (0..h)
        .flat_map(|i| ((i + 1)..h).map(move |j| (i, j)))
        .fold(0.0f64, |acc, (i, j)| acc.max(corr[(i, j)].abs()));

    // VIF = diag(C⁻¹), via a Cholesky solve against the identity rather than an
    // explicit inverse — it matters most exactly here, since near-collinear dims
    // are the case this measures. A singular `C` means a dim is an exact
    // combination of the others: infinite inflation, reported as such rather
    // than as a NaN that would read as "not measured".
    let eye = DMatrix::<f64>::identity(h, h);
    let max_vif = corr
        .clone()
        .cholesky()
        .map(|c| c.solve(&eye))
        .or_else(|| corr.lu().solve(&eye))
        .map_or(f32::INFINITY, |inv| {
            (0..h).fold(0.0f64, |acc, j| acc.max(inv[(j, j)])) as f32
        })
        .max(1.0);

    EmbeddingGeometry {
        n_rows: n,
        h,
        common_mode_cos,
        mean_pairwise_cos,
        eff_rank_raw: participation_ratio(&raw_gram),
        eff_rank_centered: participation_ratio(&ctr_gram),
        max_abs_corr: max_abs_corr as f32,
        max_vif,
    }
}

#[cfg(test)]
mod tests;
