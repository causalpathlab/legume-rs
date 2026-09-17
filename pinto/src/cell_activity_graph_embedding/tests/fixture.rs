//! The planted-truth fixture the pair-projection tests share: a frozen
//! dictionary, feature abundances, and the counts a known pair latent produces.

use crate::util::common::*;

pub(super) const N_FEATURES: usize = 240;
pub(super) const DIM: usize = 4;
pub(super) const N_CELLS: usize = 100;

/// Deterministic spread in `[-0.2, 0.2)`, so the off-block dimensions are not
/// all identical and the design matrix is not rank-4-with-4-distinct-rows.
pub(super) fn jitter(seed: usize) -> f32 {
    let h = (seed.wrapping_mul(2_654_435_761)) % 1000;
    (h as f32 / 1000.0 - 0.5) * 0.4
}

/// `[G × D]` frozen dictionary: feature `g` loads mainly on dim `g % D`.
pub(super) fn dictionary_matrix() -> Mat {
    let mut e = Mat::zeros(N_FEATURES, DIM);
    for g in 0..N_FEATURES {
        for j in 0..DIM {
            e[(g, j)] = if j == g % DIM {
                1.0 + 0.1 * ((g / DIM) % 5) as f32
            } else {
                jitter(g * DIM + j)
            };
        }
    }
    e
}

/// Log feature abundance the offsets are built from, and the totals that imply it.
pub(super) fn abundances() -> (Vec<f32>, Vec<f64>) {
    let b: Vec<f32> = (0..N_FEATURES)
        .map(|g| (5.0 + (g % 7) as f32).ln())
        .collect();
    let totals: Vec<f64> = b.iter().map(|&x| x.exp() as f64 * N_CELLS as f64).collect();
    (b, totals)
}

/// Pooled counts a pair at `theta` with intercept `beta` would produce, exactly
/// (no Poisson draw), so the MAP is `theta` up to the ridge.
pub(super) fn counts_from(e: &Mat, b: &[f32], theta: &[f32], beta: f32) -> Vec<(u32, f32)> {
    (0..N_FEATURES)
        .map(|g| {
            let s: f32 = (0..DIM).map(|j| e[(g, j)] * theta[j]).sum::<f32>() + b[g] + beta;
            (g as u32, s.exp())
        })
        .collect()
}

pub(super) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-8)
}

pub(super) fn norm(a: &[f32]) -> f32 {
    a.iter().map(|&x| x * x).sum::<f32>().sqrt()
}
