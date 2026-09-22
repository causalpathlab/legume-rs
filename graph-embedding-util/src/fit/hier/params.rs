//! The phase-1 tables as candle `Var`s: unit embeddings, the module
//! dictionary, the per-gene residuals, their biases, and one offset per
//! non-base track. Rows given from outside are pinned through gradient masks
//! (a masked row's gradient is zero, so its Adagrad step is zero) and, for the
//! output, kept verbatim so a pinned row owes nothing to the `μ + r` round
//! trip. A LoRA residual on the gene rows, and every track's per-gene offset,
//! is the shared [`legume_numeric::candle::lora`] primitive, read through one gather per
//! step.

pub use crate::preset_mode::{LoraSpec, PresetMode, PresetOffsets};
use legume_numeric::candle::candle_core::{DType, Device, Result as CResult, Tensor, Var};
use legume_numeric::candle::convert::to_host;
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::lora::PinnedLora;
use legume_numeric::matrix::rand_util::{mix_seed, normal_f32_seeded};
use nalgebra::DMatrix;

/// Spread of every random init.
pub const INIT_STDEV: f32 = 0.1;

/// Floor on a count total before its log, so an unseen feature or an empty
/// module gets a finite share instead of `ln 0`.
const MODULE_SHARE_FLOOR: f32 = 1e-6;

/// Seed salt of track `t`'s offset residual: `OFFSET_SALT + t`.
const OFFSET_SALT: u64 = 0x4f46_4653;

/// One non-base track's additive offsets from the base tables. The module
/// offset `Δ` is a full `[M, H]` table; the gene offset is a LOW-RANK residual
/// `δ_g = δ₀_g + u_g · V` ([`legume_numeric::candle::lora`], every gene a row) on top of
/// an optional given base `δ₀` (see [`HierParams::preset_offsets`]). So a
/// track's row of a gene is the gene's base row moved inside the
/// `rank`-dimensional subspace `V` spans — one subspace per track, shared by
/// every gene on it.
pub struct TrackOffset {
    /// `[M, H]`.
    pub d_mu: Var,
    /// `[M]`.
    pub d_b_m: Var,
    /// The trained part of the gene offset: `u` `[G, rank]`, `V` `[rank, H]`.
    pub d_r: PinnedLora,
    /// The given part `δ₀`, `[G, H]` with zeros off the given genes; `None`
    /// when nothing was given. A given gene the residual skips (its `u` row
    /// masked off, see [`Self::pinned_genes`]) is pinned: its composed track
    /// row is `base row + δ₀` verbatim.
    pub d_r_given: Option<Tensor>,
    /// `[G]`.
    pub d_b_g: Var,
}

impl TrackOffset {
    fn new(
        n_modules: usize,
        n_genes: usize,
        h: usize,
        rank: usize,
        seed: u64,
        dev: &Device,
    ) -> CResult<Self> {
        let all: Vec<u32> = (0..n_genes as u32).collect();
        Ok(Self {
            d_mu: Var::zeros((n_modules, h), DType::F32, dev)?,
            d_b_m: Var::zeros(n_modules, DType::F32, dev)?,
            d_r: PinnedLora::new(n_genes, h, rank, &all, seed, dev)?,
            d_r_given: None,
            d_b_g: Var::zeros(n_genes, DType::F32, dev)?,
        })
    }

    /// The gene offset on the rows named by `ids`, `[ids, H]`:
    /// `δ₀[ids] + u[ids] · V`. In the graph, so the factors take gradient.
    pub fn residual_rows(&self, ids: &Tensor) -> CResult<Tensor> {
        let r = self.d_r.residual_rows(ids)?;
        match self.d_r_given.as_ref() {
            Some(b) => r + gather_rows(b, ids)?,
            None => Ok(r),
        }
    }

    /// The whole gene offset `[G, H]` on the host, row-major: `δ₀ + u · V`.
    pub fn delta_host(&self) -> CResult<Vec<f32>> {
        let r = self.d_r.residual()?;
        let t = match self.d_r_given.as_ref() {
            Some(b) => (r + b)?,
            None => r,
        };
        to_host(&t)
    }

    /// Per gene, whether its offset is pinned: read off the residual's own
    /// mask, which is what silences the gradient (a masked-off `u` row is
    /// exactly a given `δ₀` under freeze).
    pub fn pinned_genes(&self) -> CResult<Vec<bool>> {
        Ok(to_host(&self.d_r.u_mask)?
            .into_iter()
            .map(|active| active == 0.0)
            .collect())
    }
}

/// Two residuals of one rank on the two pinned tables ([`legume_numeric::candle::lora`]):
/// `μ_m = μ₀_m + a_m · V_M` moves a module's genes together, and
/// `r_g = r₀_g + u_g · V_G` moves a gene on its own, so a pinned gene's row is
/// `ρ₀_g + a_{m(g)} · V_M + u_g · V_G`. The module residual is on every module
/// (all of `μ` is pinned); the gene residual is masked to the pinned genes.
pub struct HierLora {
    pub module: PinnedLora,
    pub gene: PinnedLora,
    /// LoRA+: the shared factors' learning rate over the row factors'.
    pub lr_ratio: f32,
    /// Per-epoch ridge weight per row on each residual (see
    /// [`LoraSpec::ridge`]); the trainer spreads it over the epoch's steps
    /// like the offset ridge.
    pub ridge: f32,
}

/// The base model's tables plus one offset table per non-base track.
///
/// Invariants: `e_u` is `[n_units, H]`, `mu` `[M, H]`, `r` `[G, H]`;
/// `offsets` holds tracks `1..T` in order, so `offsets[t - 1]` is track `t`'s
/// and the list is empty on a one-track axis.
///
/// Multi-partition fits keep one shared `e_u` here (axis 0 / gene tables) and
/// hold further axes in [`Self::extra`] — never a second unit table.
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
    /// `[G, 1]` gradient mask on `r`, `0` on pinned genes. `None` when nothing
    /// is pinned, so the plain model pays no multiply.
    pub r_mask: Option<Tensor>,
    /// Whether the module dictionary `μ` is pinned, as a whole, with the rows.
    /// The biases `b_m` / `b_g` always train.
    pub mu_frozen: bool,
    /// Per gene, whether its row is pinned (see [`Self::preset`]). Empty when
    /// nothing is pinned.
    pub frozen_gene: Vec<bool>,
    /// The pinned rows as given, `[G × H]` row-major with zeros on free genes:
    /// what a pinned gene's composed row IS. Empty when nothing is pinned.
    pub frozen_rows: Vec<f32>,
    /// The low-rank residual on the pinned genes, under [`PresetMode::Lora`].
    pub lora: Option<HierLora>,
    /// Rank of every track offset's gene residual (see [`TrackOffset`]).
    pub offset_rank: usize,
    /// LoRA+ ratio of the track offsets' shared factors.
    pub offset_lr_ratio: f32,
    /// The seed the tables were drawn from; the LoRA row factors draw from it too.
    pub seed: u64,
    /// Extra feature partitions (axes `1..`), each with its own `μ`/`r`. Empty
    /// on gene-only fits. Share [`Self::e_u`]; never duplicate the unit table.
    pub extra: Vec<ExtraAxisParams>,
}

/// What a feature adds to its module's row on a non-gene partition.
pub enum FeatureTerm {
    /// A trained residual row `r_f` and bias `b_f` (the within-module softmax).
    Residual { r: Var, b_g: Var },
    /// Nothing: the row IS the module row, and the bias is the feature's
    /// closed-form share of its module's counts, `ln(total_f / total_m)`.
    /// No within-module term, no per-feature table.
    ModuleOnly { total: Vec<f32> },
}

/// One non-gene feature partition's tables (`μ`, biases, and the per-feature
/// term). Shares the fit's unit embedding; no TrackSpec offsets.
pub struct ExtraAxisParams {
    pub mu: Var,
    pub b_m: Var,
    pub features: FeatureTerm,
}

impl ExtraAxisParams {
    /// `module_only`: the per-feature count totals when the features carry no
    /// residual; `None` allocates a residual table.
    pub fn new(
        n_modules: usize,
        n_features: usize,
        h: usize,
        seed: u64,
        module_only: Option<Vec<f32>>,
        dev: &Device,
    ) -> CResult<Self> {
        let features = match module_only {
            Some(total) => {
                assert_eq!(total.len(), n_features, "one total per feature");
                FeatureTerm::ModuleOnly { total }
            }
            None => FeatureTerm::Residual {
                r: var2(
                    randn(n_features * h, INIT_STDEV, mix_seed(seed, 0x5253)),
                    n_features,
                    h,
                    dev,
                )?,
                b_g: Var::zeros(n_features, DType::F32, dev)?,
            },
        };
        Ok(Self {
            mu: var2(
                randn(n_modules * h, INIT_STDEV, mix_seed(seed, 0x4d55)),
                n_modules,
                h,
                dev,
            )?,
            b_m: Var::zeros(n_modules, DType::F32, dev)?,
            features,
        })
    }

    /// The residual tables, `None` on a module-only partition.
    pub fn residual(&self) -> Option<(&Var, &Var)> {
        match &self.features {
            FeatureTerm::Residual { r, b_g } => Some((r, b_g)),
            FeatureTerm::ModuleOnly { .. } => None,
        }
    }

    /// Composed rows `μ_{m(f)} + r_f` and biases `b_m + b_g` for every feature;
    /// on a module-only partition `μ_{m(f)}` and `b_m + ln(total_f / total_m)`.
    pub fn compose(&self, module_of: &[u32]) -> anyhow::Result<(DMatrix<f32>, Vec<f32>)> {
        let h = self.mu.dims()[1];
        let n_features = module_of.len();
        let mu = to_host(self.mu.as_tensor())?;
        let b_m = to_host(self.b_m.as_tensor())?;
        let (r, b_g): (Option<Vec<f32>>, Vec<f32>) = match &self.features {
            FeatureTerm::Residual { r, b_g } => {
                (Some(to_host(r.as_tensor())?), to_host(b_g.as_tensor())?)
            }
            FeatureTerm::ModuleOnly { total } => {
                anyhow::ensure!(
                    total.len() == n_features,
                    "{} feature totals for {n_features} features",
                    total.len()
                );
                let mut module_total = vec![0f32; self.mu.dims()[0]];
                for (f, &m) in module_of.iter().enumerate() {
                    module_total[m as usize] += total[f];
                }
                let share = |f: usize| {
                    (total[f].max(MODULE_SHARE_FLOOR)
                        / module_total[module_of[f] as usize].max(MODULE_SHARE_FLOOR))
                    .ln()
                };
                (None, (0..n_features).map(share).collect())
            }
        };
        let mut rho = DMatrix::<f32>::zeros(n_features, h);
        let mut b_feat = vec![0f32; n_features];
        for f in 0..n_features {
            let m = module_of[f] as usize;
            for k in 0..h {
                rho[(f, k)] = mu[m * h + k] + r.as_ref().map_or(0.0, |r| r[f * h + k]);
            }
            b_feat[f] = b_m[m] + b_g[f];
        }
        Ok((rho, b_feat))
    }

    /// Host copy of `μ`, `[M × H]` row-major into a matrix.
    pub fn mu_host(&self) -> CResult<DMatrix<f32>> {
        let (n_m, h) = (self.mu.dims()[0], self.mu.dims()[1]);
        let data = to_host(self.mu.as_tensor())?;
        Ok(DMatrix::from_row_slice(n_m, h, &data))
    }
}

pub use crate::preset_mode::PresetRows;

/// Phase 1's name for [`PresetRows`]: the ids index the gene axis.
pub type PresetGenes = PresetRows;

fn randn(n: usize, stdev: f32, seed: u64) -> Vec<f32> {
    normal_f32_seeded(n, stdev, seed)
}

/// A `[rows, cols]` `Var` from row-major host data.
fn var2(data: Vec<f32>, rows: usize, cols: usize, dev: &Device) -> CResult<Var> {
    Var::from_tensor(&Tensor::from_vec(data, (rows, cols), dev)?)
}

/// One non-base track's offsets on the host: `(Δ, β, δ, γ)`, with `δ` the
/// dense gene offset `δ₀ + u · V` ([`TrackOffset::delta_host`]).
pub type HostOffset = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

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
        Self::new_tracked(n_units, n_modules, n_genes, 1, h, 1, seed, dev)
    }

    /// Base tables drawn from `seed` (the same draws whatever `n_tracks` is),
    /// plus a zero offset per non-base track, its gene residual of rank
    /// `offset_rank` (unused at one track; the trainer checks it against `h`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_tracked(
        n_units: usize,
        n_modules: usize,
        n_genes: usize,
        n_tracks: usize,
        h: usize,
        offset_rank: usize,
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
                .map(|t| {
                    TrackOffset::new(
                        n_modules,
                        n_genes,
                        h,
                        offset_rank,
                        mix_seed(seed, OFFSET_SALT + t as u64),
                        dev,
                    )
                })
                .collect::<CResult<_>>()?,
            r_mask: None,
            mu_frozen: false,
            frozen_gene: Vec::new(),
            frozen_rows: Vec::new(),
            lora: None,
            offset_rank,
            offset_lr_ratio: LoraSpec::default().lr_ratio,
            seed,
            extra: Vec::new(),
        })
    }

    /// Append one plain feature partition's tables (axes beyond the TrackSpec
    /// gene axis). Shares [`Self::e_u`].
    pub fn push_extra_axis(
        &mut self,
        n_modules: usize,
        n_features: usize,
        axis_salt: u64,
        module_only: Option<Vec<f32>>,
    ) -> CResult<()> {
        let ax = ExtraAxisParams::new(
            n_modules,
            n_features,
            self.h,
            mix_seed(self.seed, axis_salt),
            module_only,
            &self.dev,
        )?;
        self.extra.push(ax);
        Ok(())
    }

    /// Host copy of axis-0 `μ`, `[M × H]`.
    pub fn mu_host(&self) -> CResult<DMatrix<f32>> {
        let (n_m, h) = (self.mu.dims()[0], self.h);
        let data = to_host(self.mu.as_tensor())?;
        Ok(DMatrix::from_row_slice(n_m, h, &data))
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
        anyhow::ensure!(
            frozen.rows.len() == frozen.ids.len() * h,
            "frozen rows are {} values for {} genes at H={h}",
            frozen.rows.len(),
            frozen.ids.len()
        );
        let mut mu = to_host(self.mu.as_tensor())?;
        let mut r = to_host(self.r.as_tensor())?;
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
            for (i, &g) in frozen.ids.iter().enumerate() {
                let g = g as usize;
                self.frozen_gene[g] = true;
                self.frozen_rows[g * h..(g + 1) * h]
                    .copy_from_slice(&frozen.rows[i * h..(i + 1) * h]);
            }
            let keep: Vec<f32> = self
                .frozen_gene
                .iter()
                .map(|&pinned| if pinned { 0.0 } else { 1.0 })
                .collect();
            self.r_mask = Some(Tensor::from_vec(keep, (n_genes, 1), &self.dev)?);
            self.mu_frozen = true;
            if let Some(spec) = frozen.mode.lora() {
                let (rank, lr_ratio) = (spec.rank, spec.lr_ratio);
                let all_modules: Vec<u32> = (0..n_modules as u32).collect();
                self.lora = Some(HierLora {
                    module: PinnedLora::new(
                        n_modules,
                        h,
                        rank,
                        &all_modules,
                        mix_seed(self.seed, 0x4c4f_524d),
                        &self.dev,
                    )?,
                    gene: PinnedLora::new(n_genes, h, rank, &frozen.ids, self.seed, &self.dev)?,
                    lr_ratio,
                    ridge: spec.ridge,
                });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn is_frozen_gene(&self, g: usize) -> bool {
        self.frozen_gene.get(g).copied().unwrap_or(false)
    }

    /// Give `δ₀` on the non-base tracks: for each entry, `rows[i]` becomes the
    /// base of gene `ids[i]`'s offset on `track`. Under `Freeze` the residual
    /// skips those genes and their composed track rows are `base row + δ₀`
    /// verbatim, so a pinned offset needs the gene's base row pinned too
    /// ([`Self::preset`] first); under `Lora` and `Init` the residual trains
    /// on top of `δ₀` on every gene. One entry per track.
    pub fn preset_offsets(
        &mut self,
        entries: &[PresetOffsets],
        mode: PresetMode,
    ) -> anyhow::Result<()> {
        let (h, n_genes) = (self.h, self.b_g.dims()[0]);
        let is_freeze = matches!(mode, PresetMode::Freeze);
        let mut track_given = vec![false; self.offsets.len() + 1];
        for entry in entries {
            let t = entry.track as usize;
            anyhow::ensure!(
                (1..=self.offsets.len()).contains(&t),
                "offset rows on track {t}: the axis has {} non-base track(s)",
                self.offsets.len()
            );
            anyhow::ensure!(!track_given[t], "offset rows for track {t} given twice");
            track_given[t] = true;
            anyhow::ensure!(
                entry.rows.len() == entry.ids.len() * h,
                "offset rows on track {t} are {} values for {} genes at H={h}",
                entry.rows.len(),
                entry.ids.len()
            );
            let mut delta0 = vec![0f32; n_genes * h];
            let mut gene_given = vec![false; n_genes];
            for (i, &g) in entry.ids.iter().enumerate() {
                let g = g as usize;
                anyhow::ensure!(
                    g < n_genes,
                    "offset gene {g} is outside the {n_genes}-gene axis"
                );
                anyhow::ensure!(!gene_given[g], "offset gene {g} given twice on track {t}");
                anyhow::ensure!(
                    !is_freeze || self.is_frozen_gene(g),
                    "a pinned offset on gene {g} needs the gene's base row pinned too"
                );
                gene_given[g] = true;
                delta0[g * h..(g + 1) * h].copy_from_slice(&entry.rows[i * h..(i + 1) * h]);
            }
            let o = &mut self.offsets[t - 1];
            o.d_r_given = Some(Tensor::from_vec(delta0, (n_genes, h), &self.dev)?);
            if is_freeze {
                // The residual skips the given genes: rebuilt with only the
                // others active, so their `u` rows are masked off.
                let active: Vec<u32> = (0..n_genes as u32)
                    .filter(|&g| !gene_given[g as usize])
                    .collect();
                o.d_r = PinnedLora::new(
                    n_genes,
                    h,
                    self.offset_rank,
                    &active,
                    mix_seed(self.seed, OFFSET_SALT + t as u64),
                    &self.dev,
                )?;
            }
        }
        Ok(())
    }

    /// Track `t`'s offsets; `None` for the base track and for an unknown one.
    #[must_use]
    pub fn offset(&self, t: usize) -> Option<&TrackOffset> {
        self.offsets.get(t.checked_sub(1)?)
    }

    /// The composed dictionary, one row per FEATURE ROW: for row `f` naming
    /// gene `g` on track `t`, `ρ_f = base_g (+ Δ^t_{m(g)} + δ^t_g)` and
    /// `b_f = b_{m(g)} + b_g (+ β^t_{m(g)} + γ^t_g)`, where the base row is
    /// `μ_{m(g)} + r_g`, or for a pinned gene the given row verbatim plus its
    /// module's and its own LoRA shift when there is one. A pinned offset
    /// (see [`Self::preset_offsets`]) makes the track row `base_g + δ₀_g`
    /// verbatim, without the module offset.
    pub fn compose(
        &self,
        track_of_row: &[u32],
        gene_of_row: &[u32],
        module_of: &[u32],
    ) -> anyhow::Result<(DMatrix<f32>, Vec<f32>)> {
        let h = self.h;
        let n_features = track_of_row.len();
        let mu = to_host(self.mu.as_tensor())?;
        let r = to_host(self.r.as_tensor())?;
        let b_m = to_host(self.b_m.as_tensor())?;
        let b_g = to_host(self.b_g.as_tensor())?;
        // Per module and per gene, the residual shifts on the host.
        let (mod_shift, gene_shift): (Option<Vec<f32>>, Option<Vec<f32>>) = match self.lora.as_ref()
        {
            Some(l) => (
                Some(to_host(&l.module.residual()?)?),
                Some(to_host(&l.gene.residual()?)?),
            ),
            None => (None, None),
        };
        let offsets: Vec<HostOffset> = self
            .offsets
            .iter()
            .map(|o| {
                Ok((
                    to_host(o.d_mu.as_tensor())?,
                    to_host(o.d_b_m.as_tensor())?,
                    o.delta_host()?,
                    to_host(o.d_b_g.as_tensor())?,
                ))
            })
            .collect::<CResult<_>>()?;
        let pinned_offset: Vec<Vec<bool>> = self
            .offsets
            .iter()
            .map(TrackOffset::pinned_genes)
            .collect::<CResult<_>>()?;
        // The base row of gene `g` in module `m`, column `k`.
        let base = |g: usize, m: usize, k: usize| -> f32 {
            let shift = mod_shift.as_ref().map_or(0.0, |d| d[m * h + k]);
            if self.is_frozen_gene(g) {
                self.frozen_rows[g * h + k]
                    + shift
                    + gene_shift.as_ref().map_or(0.0, |d| d[g * h + k])
            } else {
                // A free gene in a pinned module still sits on the module's
                // shifted mean.
                mu[m * h + k] + shift + r[g * h + k]
            }
        };
        let mut rho = DMatrix::<f32>::zeros(n_features, h);
        let mut b_feat = vec![0f32; n_features];
        for row in 0..n_features {
            let t = track_of_row[row] as usize;
            let g = gene_of_row[row] as usize;
            let m = module_of[g] as usize;
            match t.checked_sub(1).map(|i| (&offsets[i], &pinned_offset[i])) {
                None => {
                    for k in 0..h {
                        rho[(row, k)] = base(g, m, k);
                    }
                    b_feat[row] = b_m[m] + b_g[g];
                }
                Some(((d_mu, d_b_m, d_r, d_b_g), pinned)) => {
                    let module_shift = if pinned[g] { 0.0 } else { 1.0 };
                    for k in 0..h {
                        rho[(row, k)] =
                            base(g, m, k) + module_shift * d_mu[m * h + k] + d_r[g * h + k];
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
