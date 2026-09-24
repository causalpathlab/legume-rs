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
    /// Per module, whether `μ_m` is pinned: a module with a given member under a
    /// pinning mode. Empty when nothing is pinned. The biases `b_m` / `b_g`
    /// always train.
    pub mu_pinned: Vec<bool>,
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
    /// Optional cis gates mixed into RNA gene scores.
    pub cis: Option<super::cis_gates::CisGateParams>,
    /// Optional per-unit intercept per module group (see [`GroupIntercepts`]).
    pub group: Option<GroupIntercepts>,
}

/// One intercept per unit and module GROUP (a modality, on a multiome axis),
/// added to the unit's module scores:
///
/// ```text
/// logit_um = ⟨e_u, μ_m⟩ + b_m + β_{u, k(m)}        β_{u, 0} = 0
/// ```
///
/// A unit's split of counts across groups (its ATAC:RNA ratio) is then
/// absorbed by `β_u` instead of being written into `e_u` and `μ` as a group
/// axis. Group 0 is the reference; the module softmax is shift-invariant, so
/// only `K − 1` intercepts per unit are free. Constant within a module, `β`
/// never reaches the gene level.
///
/// `β` alone is not identified: a direction `μ̄_k` shared by group `k`'s rows
/// gives `⟨e_u, μ̄_k⟩`, another per-(unit, group) intercept. So `μ` is held
/// centred within each group ([`Self::centre`]): `Σ_{m ∈ k} μ_m = 0`. The
/// removed term is exactly what `β` covers, so the model loses nothing.
pub struct GroupIntercepts {
    /// `[n_units, K − 1]`, groups `1..K`.
    pub beta: Var,
    pub n_groups: usize,
    /// `[K − 1, M]` one-hot of every module's non-reference group.
    onehot: Tensor,
    /// `[K, M]`, `1 / n_k` on group `k`'s modules: `avg · μ` is the group means.
    avg: Tensor,
    /// `[M]` group of every module, `u32`.
    group_ids: Tensor,
}

impl GroupIntercepts {
    /// `None` when fewer than two groups are named. The group tables are fixed
    /// for the fit, so they are built here once.
    pub fn new(n_units: usize, module_group: &[u32], dev: &Device) -> CResult<Option<Self>> {
        let n_groups = module_group.iter().max().map_or(0, |&k| k as usize + 1);
        if n_groups < 2 {
            return Ok(None);
        }
        let n_m = module_group.len();
        let mut size = vec![0f32; n_groups];
        for &g in module_group {
            size[g as usize] += 1.0;
        }
        let (mut onehot, mut avg) = (vec![0f32; (n_groups - 1) * n_m], vec![0f32; n_groups * n_m]);
        for (m, &g) in module_group.iter().enumerate() {
            let g = g as usize;
            if g > 0 {
                onehot[(g - 1) * n_m + m] = 1.0;
            }
            avg[g * n_m + m] = 1.0 / size[g];
        }
        Ok(Some(Self {
            beta: Var::zeros((n_units, n_groups - 1), DType::F32, dev)?,
            n_groups,
            onehot: Tensor::from_vec(onehot, (n_groups - 1, n_m), dev)?,
            avg: Tensor::from_vec(avg, (n_groups, n_m), dev)?,
            group_ids: Tensor::from_vec(module_group.to_vec(), n_m, dev)?,
        }))
    }

    /// `β_b · G` for the units `unit_ids` over the modules `mods`: `[B, |mods|]`,
    /// `G` the one-hot of the non-reference groups (all modules on a full track).
    pub fn scores(&self, unit_ids: &Tensor, mods: &[u32]) -> CResult<Tensor> {
        let beta = gather_rows(self.beta.as_tensor(), unit_ids)?;
        if mods.len() == self.onehot.dim(1)? {
            return beta.matmul(&self.onehot);
        }
        let cols = Tensor::from_vec(mods.to_vec(), mods.len(), unit_ids.device())?;
        beta.matmul(&self.onehot.index_select(&cols, 1)?)
    }

    /// `μ_m ← μ_m − mean_{m' ∈ k(m)} μ_{m'}`: every group's rows sum to zero.
    pub fn centre(&self, mu: &Var) -> CResult<()> {
        let means = self.avg.matmul(mu.as_tensor())?;
        mu.set(&(mu.as_tensor() - gather_rows(&means, &self.group_ids)?)?)
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
            mu_pinned: Vec::new(),
            frozen_gene: Vec::new(),
            frozen_rows: Vec::new(),
            lora: None,
            offset_rank,
            offset_lr_ratio: LoraSpec::default().lr_ratio,
            seed,
            cis: None,
            group: None,
        })
    }

    /// Set the listed genes' composed rows `μ_{m(g)} + r_g` to `preset.rows`.
    ///
    /// `μ` becomes the mean of the given rows in each module (a module with no
    /// given member keeps its random start), and each given gene's residual is
    /// `r_g = row − μ_m`, so the row composes back exactly. Under `Freeze` the
    /// `μ` of every module with a given member and those residuals are then
    /// pinned; under `Lora` they are pinned too and a low-rank residual trains
    /// on top; under `Init` they train on from there. A module with no given
    /// member always trains. Free genes keep their random residual either way;
    /// every bias keeps training.
    ///
    /// A **module-only** gene (`module_only[g]`; empty = none) carries no
    /// residual, so its row IS its module's: its given row enters the module
    /// mean, and a pinning mode holds that mean rather than the row itself.
    pub fn preset(
        &mut self,
        frozen: &PresetGenes,
        module_of: &[u32],
        module_only: &[bool],
    ) -> anyhow::Result<()> {
        let is_module_only = |g: usize| module_only.get(g).copied().unwrap_or(false);
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
                r[g * h + k] = if is_module_only(g) {
                    0.0
                } else {
                    frozen.rows[i * h + k] - mu[m * h + k]
                };
            }
        }
        self.mu
            .set(&Tensor::from_vec(mu, (n_modules, h), &self.dev)?)?;
        self.r.set(&Tensor::from_vec(r, (n_genes, h), &self.dev)?)?;
        if frozen.mode.pins() {
            // Row-pinned genes: the given ones that carry a residual.
            let row_pinned: Vec<u32> = frozen
                .ids
                .iter()
                .copied()
                .filter(|&g| !is_module_only(g as usize))
                .collect();
            self.frozen_gene = vec![false; n_genes];
            self.frozen_rows = vec![0.0; n_genes * h];
            for (i, &g) in frozen.ids.iter().enumerate() {
                let g = g as usize;
                if is_module_only(g) {
                    continue;
                }
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
            self.mu_pinned = count.iter().map(|&c| c > 0).collect();
            if let Some(spec) = frozen.mode.lora() {
                let (rank, lr_ratio) = (spec.rank, spec.lr_ratio);
                let pinned_modules: Vec<u32> = (0..n_modules as u32)
                    .filter(|&m| self.mu_pinned[m as usize])
                    .collect();
                self.lora = Some(HierLora {
                    module: PinnedLora::new(
                        n_modules,
                        h,
                        rank,
                        &pinned_modules,
                        mix_seed(self.seed, 0x4c4f_524d),
                        &self.dev,
                    )?,
                    gene: PinnedLora::new(n_genes, h, rank, &row_pinned, self.seed, &self.dev)?,
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
