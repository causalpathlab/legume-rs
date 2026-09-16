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
    /// Per gene, whether its residual row is pinned (see [`Self::preset`]).
    /// Empty when nothing is frozen, so the plain model pays no lookup.
    pub frozen_gene: Vec<bool>,
    /// The pinned rows as given, `[G × H]` row-major with zeros on free genes:
    /// what a pinned gene's composed row IS, kept verbatim so the output owes
    /// nothing to the `μ + r` round trip. Empty when nothing is frozen.
    pub frozen_rows: Vec<f32>,
    /// Whether the module dictionary `μ` is pinned. Set together with
    /// `frozen_gene`; the biases `b_m` / `b_g` always train.
    pub mu_frozen: bool,
}

/// Gene rows handed to phase 1 from outside: `rows` is `[gene.len() × H]`
/// row-major, one row per entry of `gene`, which indexes the gene axis. With
/// `freeze` every listed gene keeps its row for the whole fit; without it the
/// rows are the starting point and train on. Unlisted genes train freely
/// either way.
#[derive(Clone, Debug)]
pub struct PresetGenes {
    pub gene: Vec<u32>,
    pub rows: Vec<f32>,
    pub freeze: bool,
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
            frozen_gene: Vec::new(),
            frozen_rows: Vec::new(),
            mu_frozen: false,
        }
    }

    /// Set the listed genes' composed rows `μ_{m(g)} + r_g` to `preset.rows`.
    ///
    /// `μ` becomes the mean of the given rows in each module (a module with no
    /// given member keeps its random `μ`, which its free members' residuals
    /// absorb), and each given gene's residual is `r_g = row − μ_m`, so the
    /// row composes back exactly. Under `preset.freeze` both `μ` and those
    /// residuals are then pinned; otherwise they train on from there. Free
    /// genes keep their random residual either way; every bias keeps training.
    pub fn preset(&mut self, frozen: &PresetGenes, module_of: &[u32]) -> anyhow::Result<()> {
        let (h, n_genes, n_modules) = (self.h, module_of.len(), self.b_m.len());
        anyhow::ensure!(
            frozen.rows.len() == frozen.gene.len() * h,
            "frozen rows are {} values for {} genes at H={h}",
            frozen.rows.len(),
            frozen.gene.len()
        );
        let mut count = vec![0usize; n_modules];
        let mut sum = vec![0f32; n_modules * h];
        for (i, &g) in frozen.gene.iter().enumerate() {
            let g = g as usize;
            anyhow::ensure!(
                g < n_genes,
                "frozen gene {g} is outside the {n_genes}-gene axis"
            );
            let m = module_of[g] as usize;
            count[m] += 1;
            for k in 0..h {
                sum[m * h + k] += frozen.rows[i * h + k];
            }
        }
        for m in 0..n_modules {
            if count[m] > 0 {
                let inv = 1.0 / count[m] as f32;
                for k in 0..h {
                    self.mu[m * h + k] = sum[m * h + k] * inv;
                }
            }
        }
        for (i, &g) in frozen.gene.iter().enumerate() {
            let g = g as usize;
            let m = module_of[g] as usize;
            for k in 0..h {
                self.r[g * h + k] = frozen.rows[i * h + k] - self.mu[m * h + k];
            }
        }
        if frozen.freeze {
            self.frozen_gene = vec![false; n_genes];
            self.frozen_rows = vec![0.0; n_genes * h];
            for (i, &g) in frozen.gene.iter().enumerate() {
                let g = g as usize;
                self.frozen_gene[g] = true;
                self.frozen_rows[g * h..(g + 1) * h]
                    .copy_from_slice(&frozen.rows[i * h..(i + 1) * h]);
            }
            self.mu_frozen = true;
        }
        Ok(())
    }

    #[inline]
    pub fn is_frozen_gene(&self, g: usize) -> bool {
        self.frozen_gene.get(g).copied().unwrap_or(false)
    }

    /// Gene `g`'s composed base row: the pinned row when it has one, else
    /// `μ_{m(g)} + r_g`.
    pub fn base_row(&self, g: usize, m: usize, out: &mut [f32]) {
        let h = self.h;
        if self.is_frozen_gene(g) {
            out.copy_from_slice(&self.frozen_rows[g * h..(g + 1) * h]);
        } else {
            for (k, x) in out.iter_mut().enumerate() {
                *x = self.mu[m * h + k] + self.r[g * h + k];
            }
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
