//! The phase-1 tables as candle `Var`s: unit embeddings, the module
//! dictionary, the per-gene residuals, their biases, and one offset table per
//! non-base track. Rows given from outside are pinned through gradient masks
//! (a masked row's gradient is zero, so its Adagrad step is zero) and, for the
//! output, kept verbatim so a pinned row owes nothing to the `μ + r` round
//! trip. A LoRA residual on the gene rows is the shared
//! [`candle_util::lora`] primitive, read through one gather per step.

pub use crate::preset_mode::PresetMode;
use candle_util::candle_core::{DType, Device, Result as CResult, Tensor, Var};
use candle_util::lora::PinnedLora;
use matrix_util::rand_util::{collect_f32_seeded, mix_seed};
use nalgebra::DMatrix;
use rand_distr::Normal;

/// Spread of every random init.
pub const INIT_STDEV: f32 = 0.1;

/// One non-base track's additive offsets from the base tables.
pub struct TrackOffset {
    /// `[M, H]`.
    pub d_mu: Var,
    /// `[M]`.
    pub d_b_m: Var,
    /// `[G, H]`.
    pub d_r: Var,
    /// `[G]`.
    pub d_b_g: Var,
}

impl TrackOffset {
    fn zeros(n_modules: usize, n_genes: usize, h: usize, dev: &Device) -> CResult<Self> {
        Ok(Self {
            d_mu: Var::zeros((n_modules, h), DType::F32, dev)?,
            d_b_m: Var::zeros(n_modules, DType::F32, dev)?,
            d_r: Var::zeros((n_genes, h), DType::F32, dev)?,
            d_b_g: Var::zeros(n_genes, DType::F32, dev)?,
        })
    }
}

/// Two residuals of one rank on the two pinned tables ([`candle_util::lora`]):
/// `μ_m = μ₀_m + a_m · V_M` moves a module's genes together, and
/// `r_g = r₀_g + u_g · V_G` moves a gene on its own, so a pinned gene's row is
/// `ρ₀_g + a_{m(g)} · V_M + u_g · V_G`. The module residual is on every module
/// (all of `μ` is pinned); the gene residual is masked to the pinned genes.
pub struct HierLora {
    pub module: PinnedLora,
    pub gene: PinnedLora,
    /// Per-epoch ridge on each residual's mean row norm² (see
    /// [`crate::preset_mode::PresetMode::Lora`]); the trainer spreads it over
    /// the epoch's steps like the offset ridge.
    pub ridge: f32,
    /// How many gene rows the gene residual reaches: the ridge's divisor.
    pub n_pinned: usize,
}

/// The base model's tables plus one offset table per non-base track.
///
/// Invariants: `e_u` is `[n_units, H]`, `mu` `[M, H]`, `r` `[G, H]`;
/// `offsets` holds tracks `1..T` in order, so `offsets[t - 1]` is track `t`'s
/// and the list is empty on a one-track axis.
pub struct HierParams {
    pub h: usize,
    pub dev: Device,
    pub e_u: Var,
    pub mu: Var,
    pub b_m: Var,
    pub r: Var,
    pub b_g: Var,
    /// Tracks `1..T`; empty at `T == 1`.
    pub offsets: Vec<TrackOffset>,
    /// `[G, 1]` gradient mask on `r`, `0` on pinned genes, and `[M, 1]` on
    /// `μ`, all zero (the whole dictionary is pinned with the rows). `None`
    /// when nothing is pinned, so the plain model pays no multiply. The biases
    /// `b_m` / `b_g` always train.
    pub r_mask: Option<Tensor>,
    pub mu_mask: Option<Tensor>,
    /// Per gene, whether its row is pinned (see [`Self::preset`]). Empty when
    /// nothing is pinned.
    pub frozen_gene: Vec<bool>,
    /// The pinned rows as given, `[G × H]` row-major with zeros on free genes:
    /// what a pinned gene's composed row IS. Empty when nothing is pinned.
    pub frozen_rows: Vec<f32>,
    /// The low-rank residual on the pinned genes, under [`PresetMode::Lora`].
    pub lora: Option<HierLora>,
    /// The seed the tables were drawn from; the LoRA row factors draw from it too.
    pub seed: u64,
}

pub use crate::preset_mode::PresetRows;

/// Phase 1's name for [`PresetRows`]: the ids index the gene axis.
pub type PresetGenes = PresetRows;

fn randn(n: usize, stdev: f32, seed: u64) -> Vec<f32> {
    let dist = Normal::new(0.0f32, stdev).expect("finite stdev");
    collect_f32_seeded(n, dist, seed)
}

/// A `[rows, cols]` `Var` from row-major host data.
fn var2(data: Vec<f32>, rows: usize, cols: usize, dev: &Device) -> CResult<Var> {
    Var::from_tensor(&Tensor::from_vec(data, (rows, cols), dev)?)
}

/// One non-base track's offsets on the host: `(d_mu, d_b_m, d_r, d_b_g)`.
pub type HostOffset = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Row-major host copy of a 2-D tensor.
pub fn to_host2(t: &Tensor) -> CResult<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

/// Host copy of a 1-D tensor.
pub fn to_host1(t: &Tensor) -> CResult<Vec<f32>> {
    t.to_vec1::<f32>()
}

impl HierParams {
    /// The one-track tables: no offsets.
    pub fn new(
        n_units: usize,
        n_modules: usize,
        n_genes: usize,
        h: usize,
        seed: u64,
        dev: &Device,
    ) -> CResult<Self> {
        Self::new_tracked(n_units, n_modules, n_genes, 1, h, seed, dev)
    }

    /// Base tables drawn from `seed` (the same draws whatever `n_tracks` is),
    /// plus a zero offset table per non-base track.
    pub fn new_tracked(
        n_units: usize,
        n_modules: usize,
        n_genes: usize,
        n_tracks: usize,
        h: usize,
        seed: u64,
        dev: &Device,
    ) -> CResult<Self> {
        Ok(Self {
            h,
            dev: dev.clone(),
            e_u: var2(
                randn(n_units * h, INIT_STDEV, mix_seed(seed, 0x4855)),
                n_units,
                h,
                dev,
            )?,
            mu: var2(
                randn(n_modules * h, INIT_STDEV, mix_seed(seed, 0x4d55)),
                n_modules,
                h,
                dev,
            )?,
            b_m: Var::zeros(n_modules, DType::F32, dev)?,
            r: var2(
                randn(n_genes * h, INIT_STDEV, mix_seed(seed, 0x5253)),
                n_genes,
                h,
                dev,
            )?,
            b_g: Var::zeros(n_genes, DType::F32, dev)?,
            offsets: (1..n_tracks)
                .map(|_| TrackOffset::zeros(n_modules, n_genes, h, dev))
                .collect::<CResult<_>>()?,
            r_mask: None,
            mu_mask: None,
            frozen_gene: Vec::new(),
            frozen_rows: Vec::new(),
            lora: None,
            seed,
        })
    }

    /// Set the listed genes' composed rows `μ_{m(g)} + r_g` to `preset.rows`.
    ///
    /// `μ` becomes the mean of the given rows in each module (a module with no
    /// given member keeps its random `μ`, which its free members' residuals
    /// absorb), and each given gene's residual is `r_g = row − μ_m`, so the
    /// row composes back exactly. Under `Freeze` both `μ` and those residuals
    /// are then pinned; under `Lora` they are pinned too and a low-rank
    /// residual trains on top; under `Init` they train on from there. Free
    /// genes keep their random residual either way; every bias keeps training.
    pub fn preset(&mut self, frozen: &PresetGenes, module_of: &[u32]) -> anyhow::Result<()> {
        let (h, n_genes, n_modules) = (self.h, module_of.len(), self.b_m.dims()[0]);
        frozen.mode.validate(h)?;
        // A preset is a statement about the base rows; the offset tracks would
        // train against a residual the output never carries.
        anyhow::ensure!(
            self.offsets.is_empty(),
            "preset gene rows need a single-track feature axis"
        );
        anyhow::ensure!(
            frozen.rows.len() == frozen.ids.len() * h,
            "frozen rows are {} values for {} genes at H={h}",
            frozen.rows.len(),
            frozen.ids.len()
        );
        let mut mu = to_host2(self.mu.as_tensor())?;
        let mut r = to_host2(self.r.as_tensor())?;
        let mut count = vec![0usize; n_modules];
        let mut sum = vec![0f32; n_modules * h];
        for (i, &g) in frozen.ids.iter().enumerate() {
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
                    mu[m * h + k] = sum[m * h + k] * inv;
                }
            }
        }
        for (i, &g) in frozen.ids.iter().enumerate() {
            let g = g as usize;
            let m = module_of[g] as usize;
            for k in 0..h {
                r[g * h + k] = frozen.rows[i * h + k] - mu[m * h + k];
            }
        }
        self.mu
            .set(&Tensor::from_vec(mu, (n_modules, h), &self.dev)?)?;
        self.r.set(&Tensor::from_vec(r, (n_genes, h), &self.dev)?)?;
        if frozen.mode.pins() {
            self.frozen_gene = vec![false; n_genes];
            self.frozen_rows = vec![0.0; n_genes * h];
            let mut keep = vec![1f32; n_genes];
            for (i, &g) in frozen.ids.iter().enumerate() {
                let g = g as usize;
                self.frozen_gene[g] = true;
                keep[g] = 0.0;
                self.frozen_rows[g * h..(g + 1) * h]
                    .copy_from_slice(&frozen.rows[i * h..(i + 1) * h]);
            }
            self.r_mask = Some(Tensor::from_vec(keep, (n_genes, 1), &self.dev)?);
            self.mu_mask = Some(Tensor::zeros((n_modules, 1), DType::F32, &self.dev)?);
            if let Some(spec) = frozen.mode.lora() {
                let (rank, lr_ratio) = (spec.rank, spec.lr_ratio);
                let all_modules: Vec<u32> = (0..n_modules as u32).collect();
                self.lora = Some(HierLora {
                    module: PinnedLora::new(
                        n_modules,
                        h,
                        rank,
                        &all_modules,
                        lr_ratio,
                        mix_seed(self.seed, 0x4c4f_524d),
                        &self.dev,
                    )?,
                    gene: PinnedLora::new(
                        n_genes,
                        h,
                        rank,
                        &frozen.ids,
                        lr_ratio,
                        self.seed,
                        &self.dev,
                    )?,
                    ridge: spec.ridge,
                    n_pinned: frozen.ids.len(),
                });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn is_frozen_gene(&self, g: usize) -> bool {
        self.frozen_gene.get(g).copied().unwrap_or(false)
    }

    /// Track `t`'s offsets; `None` for the base track and for an unknown one.
    #[must_use]
    pub fn offset(&self, t: usize) -> Option<&TrackOffset> {
        self.offsets.get(t.checked_sub(1)?)
    }

    /// The composed dictionary, one row per FEATURE ROW: for row `f` naming
    /// gene `g` on track `t`, `ρ_f = μ_{m(g)} + r_g (+ Δ^t_{m(g)} + δ^t_g)` and
    /// `b_f = b_{m(g)} + b_g (+ β^t_{m(g)} + γ^t_g)`. A pinned gene's base row
    /// is the given row verbatim, plus its module's and its own LoRA shift
    /// when there is one (a preset needs one track, so no offset applies).
    pub fn compose(
        &self,
        track_of_row: &[u32],
        gene_of_row: &[u32],
        module_of: &[u32],
    ) -> anyhow::Result<(DMatrix<f32>, Vec<f32>)> {
        let h = self.h;
        let n_features = track_of_row.len();
        let mu = to_host2(self.mu.as_tensor())?;
        let r = to_host2(self.r.as_tensor())?;
        let b_m = to_host1(self.b_m.as_tensor())?;
        let b_g = to_host1(self.b_g.as_tensor())?;
        // Per module and per gene, the residual shifts on the host.
        let (mod_shift, gene_shift): (Option<Vec<f32>>, Option<Vec<f32>>) = match self.lora.as_ref()
        {
            Some(l) => (
                Some(to_host2(&l.module.residual()?)?),
                Some(to_host2(&l.gene.residual()?)?),
            ),
            None => (None, None),
        };
        let offsets: Vec<HostOffset> = self
            .offsets
            .iter()
            .map(|o| {
                Ok((
                    to_host2(o.d_mu.as_tensor())?,
                    to_host1(o.d_b_m.as_tensor())?,
                    to_host2(o.d_r.as_tensor())?,
                    to_host1(o.d_b_g.as_tensor())?,
                ))
            })
            .collect::<CResult<_>>()?;
        let mut rho = DMatrix::<f32>::zeros(n_features, h);
        let mut b_feat = vec![0f32; n_features];
        for row in 0..n_features {
            let t = track_of_row[row] as usize;
            let g = gene_of_row[row] as usize;
            let m = module_of[g] as usize;
            match t.checked_sub(1).and_then(|i| offsets.get(i)) {
                None => {
                    if self.is_frozen_gene(g) {
                        for k in 0..h {
                            rho[(row, k)] = self.frozen_rows[g * h + k]
                                + mod_shift.as_ref().map_or(0.0, |d| d[m * h + k])
                                + gene_shift.as_ref().map_or(0.0, |d| d[g * h + k]);
                        }
                    } else {
                        // A free gene in a pinned module still sits on the
                        // module's shifted mean.
                        for k in 0..h {
                            rho[(row, k)] = mu[m * h + k]
                                + mod_shift.as_ref().map_or(0.0, |d| d[m * h + k])
                                + r[g * h + k];
                        }
                    }
                    b_feat[row] = b_m[m] + b_g[g];
                }
                Some((d_mu, d_b_m, d_r, d_b_g)) => {
                    for k in 0..h {
                        rho[(row, k)] =
                            mu[m * h + k] + r[g * h + k] + d_mu[m * h + k] + d_r[g * h + k];
                    }
                    b_feat[row] = b_m[m] + b_g[g] + d_b_m[m] + d_b_g[g];
                }
            }
        }
        Ok((rho, b_feat))
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod params_tests;
