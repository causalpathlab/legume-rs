//! Host-side parameter tables and PBG's row-wise Adagrad.

use matrix_util::rand_util::{collect_f32_seeded, mix_seed};
use rand_distr::Normal;

const INIT_STDEV: f32 = 0.1;
const ADAGRAD_EPS: f32 = 1e-10;

/// One non-base track's additive offsets from the base model: `Δ^t_m` on the
/// module dictionary and `δ^t_g` on the gene residual, with their biases.
/// Zero-initialised, so training starts at the base model and the ridge
/// (`FitConfig::offset_l2`) keeps the offsets small.
#[derive(Clone, Debug)]
pub struct TrackOffset {
    /// `[M × H]` row-major.
    pub d_mu: Vec<f32>,
    /// `[M]`.
    pub d_b_m: Vec<f32>,
    /// `[G × H]` row-major.
    pub d_r: Vec<f32>,
    /// `[G]`.
    pub d_b_g: Vec<f32>,
}

impl TrackOffset {
    fn zeros(n_modules: usize, n_genes: usize, h: usize) -> Self {
        Self {
            d_mu: vec![0.0; n_modules * h],
            d_b_m: vec![0.0; n_modules],
            d_r: vec![0.0; n_genes * h],
            d_b_g: vec![0.0; n_genes],
        }
    }
}

/// The base model's tables plus one offset table per non-base track.
///
/// Invariants: `e_u` is `[n_units × h]`, `mu` `[M × h]`, `r` `[G × h]`, all
/// row-major; `offsets` holds tracks `1..T` in order, so `offsets[t - 1]` is
/// track `t`'s and the list is empty on a one-track axis.
pub struct HierParams {
    pub h: usize,
    pub e_u: Vec<f32>,
    pub mu: Vec<f32>,
    pub b_m: Vec<f32>,
    pub r: Vec<f32>,
    pub b_g: Vec<f32>,
    /// Tracks `1..T`; empty at `T == 1`.
    pub offsets: Vec<TrackOffset>,
}

fn randn(n: usize, seed: u64) -> Vec<f32> {
    let dist = Normal::new(0.0f32, INIT_STDEV).expect("finite stdev");
    collect_f32_seeded(n, dist, seed)
}

impl HierParams {
    /// The one-track tables: no offsets.
    pub fn new(n_units: usize, n_modules: usize, n_genes: usize, h: usize, seed: u64) -> Self {
        Self::new_tracked(n_units, n_modules, n_genes, 1, h, seed)
    }

    /// [`Self::new`] plus a zero offset table per non-base track. The offsets
    /// draw nothing from the RNG, so the base tables are identical to
    /// [`Self::new`]'s for the same seed.
    pub fn new_tracked(
        n_units: usize,
        n_modules: usize,
        n_genes: usize,
        n_tracks: usize,
        h: usize,
        seed: u64,
    ) -> Self {
        Self {
            h,
            e_u: randn(n_units * h, mix_seed(seed, 0x4855)),
            mu: randn(n_modules * h, mix_seed(seed, 0x4d55)),
            b_m: vec![0.0; n_modules],
            r: randn(n_genes * h, mix_seed(seed, 0x5253)),
            b_g: vec![0.0; n_genes],
            offsets: (1..n_tracks)
                .map(|_| TrackOffset::zeros(n_modules, n_genes, h))
                .collect(),
        }
    }

    /// Track `t`'s offsets; `None` for the base track and for an unknown one.
    #[must_use]
    pub fn offset(&self, t: usize) -> Option<&TrackOffset> {
        self.offsets.get(t.checked_sub(1)?)
    }
}

pub struct RowAdagrad {
    pub acc: Vec<f32>,
    pub lr: f32,
    pub eps: f32,
}

impl RowAdagrad {
    pub fn new(n_rows: usize, lr: f32) -> Self {
        Self {
            acc: vec![0.0; n_rows],
            lr,
            eps: ADAGRAD_EPS,
        }
    }
    pub fn update(&mut self, r: usize, row: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(row.len(), grad.len());
        let g2 = grad.iter().map(|g| g * g).sum::<f32>() / grad.len().max(1) as f32;
        if g2 == 0.0 {
            return;
        }
        self.acc[r] += g2;
        let step = self.lr / (self.acc[r].sqrt() + self.eps);
        for (x, g) in row.iter_mut().zip(grad) {
            *x -= step * g;
        }
    }

    /// [`Self::update`] for a row that carries a scalar bias alongside it: the
    /// accumulator sees the mean of `grad²` over the row AND the bias, and both
    /// move by the same row step. What the callers did by concatenating the
    /// bias onto a copy of the row, without the copy.
    pub fn update_with_bias(
        &mut self,
        r: usize,
        row: &mut [f32],
        bias: &mut f32,
        grad: &[f32],
        gbias: f32,
    ) {
        debug_assert_eq!(row.len(), grad.len());
        let n = (grad.len() + 1) as f32;
        let g2 = (grad.iter().map(|g| g * g).sum::<f32>() + gbias * gbias) / n;
        if g2 == 0.0 {
            return;
        }
        self.acc[r] += g2;
        let step = self.lr / (self.acc[r].sqrt() + self.eps);
        for (x, g) in row.iter_mut().zip(grad) {
            *x -= step * g;
        }
        *bias -= step * gbias;
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod params_tests;
