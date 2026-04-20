//! Minimal UMAP-style SGD layout over a weighted edge list. Expects a
//! pre-built fuzzy kNN graph (edges + [0,1] weights); the low-d kernel
//! is `1 / (1 + a·d^(2b))` with the standard `(a, b) ≈ (1.929, 0.7915)`
//! fit for `spread=1, min_dist=0.1`.
//!
//! Inner math uses `Vector2<f32>` (SIMD auto-vec). Edges are processed
//! in parallel via rayon with HOGWILD! benign races on the shared
//! coords buffer — UMAP's SGD is robust to these; the reference numba
//! impl does the same.
//!
//! References:
//! - McInnes, Healy & Melville, *arXiv* 1802.03426 — UMAP.
//! - Recht et al., *NeurIPS* 2011 — HOGWILD! lock-free SGD.

use nalgebra::Vector2;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;

const A: f32 = 1.929;
const B: f32 = 0.7915;

pub struct Umap {
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

impl Default for Umap {
    fn default() -> Self {
        Self {
            n_epochs: 500,
            negative_sample_rate: 5,
            learning_rate: 1.0,
            seed: 42,
        }
    }
}

/// Shared handle for HOGWILD! parallel SGD on `coords`.
struct HogwildCoords {
    ptr: *mut f32,
    n: usize,
}

unsafe impl Sync for HogwildCoords {}
unsafe impl Send for HogwildCoords {}

impl HogwildCoords {
    #[inline]
    fn get(&self, i: usize) -> Vector2<f32> {
        debug_assert!(i < self.n);
        // SAFETY: HOGWILD! allows benign races; each index is 2 f32s.
        unsafe { Vector2::new(*self.ptr.add(i * 2), *self.ptr.add(i * 2 + 1)) }
    }

    #[inline]
    fn add(&self, i: usize, delta: Vector2<f32>) {
        debug_assert!(i < self.n);
        // SAFETY: HOGWILD! tolerates torn updates; values are bounded by clamp.
        unsafe {
            *self.ptr.add(i * 2) += delta.x;
            *self.ptr.add(i * 2 + 1) += delta.y;
        }
    }
}

impl Umap {
    /// Run HOGWILD! SGD on the given undirected edge list.
    ///
    /// * `edges` — `(i, j, weight)` with `weight ∈ (0, 1]`, `i < j`.
    /// * `n` — number of points (rows in `init`/output).
    /// * `init` — row-major `n × 2` initial coords.
    pub fn fit(&self, edges: &[(usize, usize, f32)], n: usize, init: &[f32]) -> Vec<f32> {
        assert_eq!(init.len(), n * 2, "init size mismatch");
        let mut y = init.to_vec();

        let eps = 1e-4_f32;
        let max_weight = edges.iter().map(|e| e.2).fold(0.0_f32, f32::max).max(eps);
        let epochs_per_sample: Vec<f32> = edges
            .iter()
            .map(|&(_, _, w)| {
                if w > 0.0 {
                    max_weight / w
                } else {
                    f32::INFINITY
                }
            })
            .collect();
        let mut next_epoch: Vec<f32> = epochs_per_sample.clone();

        let coords = HogwildCoords {
            ptr: y.as_mut_ptr(),
            n,
        };
        let coords = &coords;
        let n_neg = self.negative_sample_rate;
        let seed = self.seed;

        for epoch in 0..self.n_epochs {
            let epoch_f = epoch as f32;
            let alpha = self.learning_rate * (1.0 - epoch_f / self.n_epochs as f32);

            edges
                .par_iter()
                .zip(next_epoch.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || {
                        let tid = rayon::current_thread_index().unwrap_or(0) as u64;
                        SmallRng::seed_from_u64(seed ^ ((epoch as u64) << 32) ^ tid)
                    },
                    |rng, (e_idx, (&(i, j, _), ne))| {
                        if *ne > epoch_f {
                            return;
                        }
                        apply_attraction(coords, i, j, alpha);

                        for _ in 0..n_neg {
                            let k = rng.random_range(0..n);
                            if k == i {
                                continue;
                            }
                            apply_repulsion(coords, i, k, alpha);
                        }

                        *ne += epochs_per_sample[e_idx];
                    },
                );
        }

        y
    }
}

#[inline]
fn apply_attraction(y: &HogwildCoords, i: usize, j: usize, alpha: f32) {
    let diff = y.get(i) - y.get(j);
    let d2 = diff.norm_squared();
    if d2 <= 0.0 {
        return;
    }
    let d2b = d2.powf(B);
    let coeff = -2.0 * A * B * (d2b / d2) / (A * d2b + 1.0);
    let grad = (diff * coeff).map(clamp4) * alpha;
    y.add(i, grad);
    y.add(j, -grad);
}

#[inline]
fn apply_repulsion(y: &HogwildCoords, i: usize, k: usize, alpha: f32) {
    let diff = y.get(i) - y.get(k);
    let d2 = diff.norm_squared();
    let coeff = if d2 > 0.0 {
        2.0 * B / ((0.001 + d2) * (A * d2.powf(B) + 1.0))
    } else {
        4.0
    };
    let grad = (diff * coeff).map(clamp4) * alpha;
    y.add(i, grad);
}

#[inline]
fn clamp4(x: f32) -> f32 {
    x.clamp(-4.0, 4.0)
}
