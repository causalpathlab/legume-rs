//! Optional cis peak→gene coupling that aligns gene rows with their cis peaks.
//!
//! ```text
//! w_gp  = abc · max(0, θ₀ + θ₁ z_log_contact)       w̃_gp = w_gp / Σ_q w_gq
//! λ_u,m = ⟨e_u, μ_m⟩ + b_m
//! ã_ug  = Σ_p w̃_gp · λ_u,m(p)                      the gene's ATAC-guided activity
//! η_ug  = (1 − α) ρ_ug + α ã_ug                     a cis gene's likelihood score
//! gap   = mean_g  var_u(ρ_ug − ã_ug)                ρ_ug = ⟨e_u, μ_{m(g)} + r_g⟩ + b
//! ```
//! Two couplings, set on [`CisCoupling`]. The **mixture** takes a global share
//! `α` of each cis gene's likelihood score from its cis peaks; genes without
//! pairs keep `ρ_ug`, and phase 2 sees the mixed rows
//! ([`CisGateParams::dictionary_blend`]). The **alignment** adds
//! `align_weight · n_units · gap` to phase 1's loss over the step's units: it
//! pulls each cis gene's own profile across the units toward the predicted
//! accessibility of its cis peaks, and those peaks' module rows toward the
//! gene, so the two modalities share one feature space. `θ` (shared by every
//! gene) trains through both: the gate's shape is the distance profile under
//! which a gene's cis peaks track the gene.
//!
//! The gate is a distance prior: it reads contact only. `w̃` is each gene's
//! share of its pairs, so only the gate's shape `θ₁/θ₀` matters; a gene whose
//! gates all close pools nothing.
//!
//! The data's evidence for a link is read out after training
//! ([`CisGateParams::readout`]): the correlation across units of the peak
//! module's score and the gene's.
//!
//! # How it is computed
//!
//! Both scores are linear in `e_u`, so the pooled term is a gene row plus a
//! gene bias, and the gap is a quadratic form in the units' covariance `Σ`
//! (centring across units drops every bias):
//!
//! ```text
//! ã_ug = ⟨e_u, ν̃_g⟩ + c̃_g      ν̃_g = Σ_p w̃_gp μ_{m(p)}      c̃_g = Σ_p w̃_gp b_{m(p)}
//! var_u(ρ_ug − ã_ug) = d_gᵀ Σ d_g                   d_g = μ_{m(g)} + r_g − ν̃_g
//! ```
//!
//! [`CisGateParams::pool`] scatters each pair's `w̃ μ_{m(p)}` and `w̃ b_{m(p)}`
//! onto its gene (`[n_pairs × H]` work, no gene × module block), once per
//! step; the mixture and the gap both read that pool. A step split over
//! threads hands the slices the mixture as shared leaves
//! ([`CisMix::detached`]) and carries their summed gradient back through the
//! pool once ([`CisMixLeaves::backprop`]). `Σ` is taken on the units
//! detached: the gap moves features and gates, never the units, so flattening
//! the units is not a way to close it.

use legume_numeric::candle::candle_core::backprop::GradStore;
use legume_numeric::candle::candle_core::{DType, Device, Result as CResult, Tensor, Var};
use legume_numeric::candle::fast_index::{gather_rows, index_add_rows};
use nalgebra::DMatrix;
use rustc_hash::FxHashMap;

/// Floor on a gene's gate total before dividing its shares: a gene whose
/// gates all close pools nothing instead of `0/0`.
const GATE_TOTAL_FLOOR: f64 = 1e-12;

/// Parallel cis pairs on the unified feature axis (one-track multiome).
#[derive(Clone, Debug, Default)]
pub struct CisGates {
    /// RNA feature index per pair.
    pub gene_feat: Vec<u32>,
    /// ATAC **feature** index per pair (resolved to a module after the partition).
    pub peak_feat: Vec<u32>,
    pub abc: Vec<f32>,
    /// z-scored log-contact features.
    pub z_log_contact: Vec<f32>,
}

impl CisGates {
    #[must_use]
    pub fn n_pairs(&self) -> usize {
        self.gene_feat.len()
    }

    pub fn validate(&self, n_features: usize) -> anyhow::Result<()> {
        let n = self.n_pairs();
        anyhow::ensure!(
            self.peak_feat.len() == n && self.abc.len() == n && self.z_log_contact.len() == n,
            "cis gate pair arrays must share length"
        );
        for (&g, &p) in self.gene_feat.iter().zip(&self.peak_feat) {
            anyhow::ensure!(
                (g as usize) < n_features,
                "cis gene feat {g} out of range ({n_features})"
            );
            anyhow::ensure!(
                (p as usize) < n_features,
                "cis peak feat {p} out of range ({n_features})"
            );
        }
        Ok(())
    }

    /// Resolve peak features to module ids.
    #[must_use]
    pub fn with_peak_modules(&self, module_of: &[u32]) -> CisGatesResolved {
        CisGatesResolved {
            gene_feat: self.gene_feat.clone(),
            peak_module: self
                .peak_feat
                .iter()
                .map(|&p| module_of[p as usize])
                .collect(),
            abc: self.abc.clone(),
            z_log_contact: self.z_log_contact.clone(),
            source_idx: (0..self.n_pairs() as u32).collect(),
        }
    }
}

/// The cis pairs and how strongly they couple the two modalities.
#[derive(Clone, Debug, Default)]
pub struct CisCoupling {
    pub pairs: CisGates,
    /// Weight of the alignment gap per unit, next to the likelihood.
    pub align_weight: f32,
    /// Share `α ∈ [0, 1)` of each cis gene's likelihood score taken from its
    /// cis peaks: `η = (1 − α) ρ + α ã`. `0` leaves the likelihood exact.
    pub mix: f32,
}

impl CisCoupling {
    pub fn validate(&self, n_features: usize) -> anyhow::Result<()> {
        self.pairs.validate(n_features)?;
        anyhow::ensure!(
            self.align_weight.is_finite() && self.align_weight >= 0.0,
            "cis alignment weight must be finite and non-negative, got {}",
            self.align_weight
        );
        anyhow::ensure!(
            (0.0..1.0).contains(&self.mix),
            "cis mixture share must be in [0, 1), got {}",
            self.mix
        );
        Ok(())
    }
}

/// Pairs with peak modules resolved.
#[derive(Clone, Debug)]
pub struct CisGatesResolved {
    pub gene_feat: Vec<u32>,
    pub peak_module: Vec<u32>,
    pub abc: Vec<f32>,
    pub z_log_contact: Vec<f32>,
    /// Index into the caller's original [`CisGates`] pair order.
    pub source_idx: Vec<u32>,
}

impl CisGatesResolved {
    #[must_use]
    pub fn n_pairs(&self) -> usize {
        self.gene_feat.len()
    }

    /// The pairs at `keep`, in that order.
    fn select(&self, keep: &[usize]) -> Self {
        Self {
            gene_feat: keep.iter().map(|&i| self.gene_feat[i]).collect(),
            peak_module: keep.iter().map(|&i| self.peak_module[i]).collect(),
            abc: keep.iter().map(|&i| self.abc[i]).collect(),
            z_log_contact: keep.iter().map(|&i| self.z_log_contact[i]).collect(),
            source_idx: keep.iter().map(|&i| self.source_idx[i]).collect(),
        }
    }

    /// Drop the pairs of genes flagged in `drop_gene` (by feature id): a
    /// module-only gene has no row of its own to align.
    #[must_use]
    pub fn without_genes(&self, drop_gene: &[bool]) -> Self {
        let keep: Vec<usize> = (0..self.n_pairs())
            .filter(|&i| {
                !drop_gene
                    .get(self.gene_feat[i] as usize)
                    .copied()
                    .unwrap_or(false)
            })
            .collect();
        self.select(&keep)
    }

    /// Drop the pairs whose peak sits in a module flagged in `background` (by
    /// module id): flat or near-empty peaks carry no link signal.
    #[must_use]
    pub fn without_peak_modules(&self, background: &[bool]) -> Self {
        let keep: Vec<usize> = (0..self.n_pairs())
            .filter(|&i| {
                !background
                    .get(self.peak_module[i] as usize)
                    .copied()
                    .unwrap_or(false)
            })
            .collect();
        self.select(&keep)
    }
}

/// Trainable scalars + device copies of the pair tables. Genes are indexed
/// on the COMPACT axis of the genes that have pairs.
pub struct CisGateParams {
    pub theta0: Var,
    pub theta1: Var,
    /// Feature id of every compact gene, `[n_compact]`.
    compact_feat: Tensor,
    /// Module of every compact gene, `[n_compact]`: `ρ_g = μ_{m(g)} + r_g`.
    compact_gene_module: Tensor,
    /// Compact gene of every pair, `[n_pairs]`.
    pair_gene: Tensor,
    /// Module of every pair's peak, `[n_pairs]`.
    pair_module: Tensor,
    abc: Tensor,
    z_log_contact: Tensor,
    pub n_compact: usize,
    /// Index into the caller's original [`CisGates`] order (after dropping
    /// pairs of module-only genes and background peaks).
    pub source_idx: Vec<u32>,
    /// The mixture, when on (see [`Self::with_mix`]).
    mixture: Option<Mixture>,
}

/// The mixture's tables: its share and the gene-axis lookup.
struct Mixture {
    alpha: f32,
    /// Feature → compact gene id; a gene without pairs maps to `n_compact`.
    feat_to_compact: Tensor,
    /// Each gene's own share: `1 − α` per compact gene, `1` on the trailing
    /// row for genes without pairs, `[n_compact + 1]`.
    self_share: Tensor,
}

/// Per item its compact id (first-seen order), and the distinct items.
fn compact_ids(ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut map: FxHashMap<u32, u32> = FxHashMap::default();
    let mut order = Vec::new();
    let per_item = ids
        .iter()
        .map(|&x| {
            *map.entry(x).or_insert_with(|| {
                order.push(x);
                order.len() as u32 - 1
            })
        })
        .collect();
    (per_item, order)
}

/// The units' covariance `[H, H]`, detached: `(1/n) Σ_u (e_u − ē)(e_u − ē)ᵀ`.
fn unit_covariance(e: &Tensor) -> CResult<Tensor> {
    let e = e.detach();
    let n = e.dim(0)?.max(1) as f64;
    let e_c = e.broadcast_sub(&e.mean_keepdim(0)?)?;
    e_c.t()?.matmul(&e_c)? / n
}

impl CisGateParams {
    /// `module_of` is the partition of the whole feature axis: it gives each
    /// cis gene's own module for `ρ_g = μ_{m(g)} + r_g`.
    pub fn new(spec: &CisGatesResolved, module_of: &[u32], dev: &Device) -> CResult<Self> {
        let n = spec.n_pairs();
        let scalar = |x: f64| -> CResult<Var> { Var::from_tensor(&Tensor::new(x as f32, dev)?) };
        let (gene_c, genes) = compact_ids(&spec.gene_feat);
        let n_compact = genes.len();
        let gene_module: Vec<u32> = genes.iter().map(|&g| module_of[g as usize]).collect();

        Ok(Self {
            theta0: scalar(1.0)?,
            theta1: scalar(0.5)?,
            compact_feat: Tensor::from_vec(genes, n_compact, dev)?,
            compact_gene_module: Tensor::from_vec(gene_module, n_compact, dev)?,
            pair_gene: Tensor::from_vec(gene_c, n, dev)?,
            pair_module: Tensor::from_vec(spec.peak_module.clone(), n, dev)?,
            abc: Tensor::from_vec(spec.abc.clone(), n, dev)?,
            z_log_contact: Tensor::from_vec(spec.z_log_contact.clone(), n, dev)?,
            n_compact,
            source_idx: spec.source_idx.clone(),
            mixture: None,
        })
    }

    /// Mix a share `alpha` of the cis peaks' pooled activity into each cis
    /// gene's likelihood score (`alpha = 0`: no mixture). `n_features` is the
    /// length of the feature axis the gene level indexes.
    pub fn with_mix(mut self, alpha: f32, n_features: usize, dev: &Device) -> CResult<Self> {
        if alpha == 0.0 {
            self.mixture = None;
            return Ok(self);
        }
        let n_c = self.n_compact;
        let mut feat_to_compact = vec![n_c as u32; n_features];
        for (c, g) in self.compact_feat.to_vec1::<u32>()?.into_iter().enumerate() {
            feat_to_compact[g as usize] = c as u32;
        }
        let mut self_share = vec![1.0 - alpha; n_c];
        self_share.push(1.0);
        self.mixture = Some(Mixture {
            alpha,
            feat_to_compact: Tensor::from_vec(feat_to_compact, n_features, dev)?,
            self_share: Tensor::from_vec(self_share, n_c + 1, dev)?,
        });
        Ok(self)
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![self.theta0.clone(), self.theta1.clone()]
    }

    /// Feature ids of the cis genes, `[n_compact]`: the rows of `r` and `b_g`
    /// the callers pass as `r_c` and `b_c`.
    #[must_use]
    pub fn compact_feat(&self) -> &Tensor {
        &self.compact_feat
    }

    /// Each gene's pair shares `w̃` `[n_pairs]`, the pooled rows `ν̃`
    /// `[n_compact, H]` and biases `c̃` `[n_compact]`, from the module table
    /// `mu` `[M, H]` and biases `b_m` `[M]`: every pair's `w̃ μ_{m(p)}` and
    /// `w̃ b_{m(p)}` scattered onto its gene.
    pub fn pool(&self, mu: &Tensor, b_m: &Tensor) -> CResult<CisPool> {
        let n_c = self.n_compact;
        let h = mu.dim(1)?;
        let dev = mu.device();
        let s = self
            .theta0
            .as_tensor()
            .broadcast_add(&self.z_log_contact.broadcast_mul(self.theta1.as_tensor())?)?;
        let w_raw = (&self.abc * s.relu()?)?;
        let total = index_add_rows(
            &Tensor::zeros((n_c, 1), DType::F32, dev)?,
            &self.pair_gene,
            &w_raw.unsqueeze(1)?,
        )?;
        let w = (&w_raw
            / gather_rows(&total, &self.pair_gene)?
                .squeeze(1)?
                .maximum(GATE_TOTAL_FLOOR)?)?;
        let w_col = w.unsqueeze(1)?;
        let nu = index_add_rows(
            &Tensor::zeros((n_c, h), DType::F32, dev)?,
            &self.pair_gene,
            &gather_rows(mu, &self.pair_module)?.broadcast_mul(&w_col)?,
        )?;
        let c = index_add_rows(
            &Tensor::zeros((n_c, 1), DType::F32, dev)?,
            &self.pair_gene,
            &(gather_rows(b_m, &self.pair_module)?.unsqueeze(1)? * &w_col)?,
        )?
        .squeeze(1)?;
        Ok(CisPool { w, nu, c })
    }

    /// What the gene level mixes in over a built `pool`, or `None` without a
    /// mixture: `ν̃` and `c̃` on the compact gene axis, a zero row appended for
    /// genes without pairs.
    pub fn mix(&self, pool: &CisPool) -> CResult<Option<CisMix>> {
        let Some(mx) = self.mixture.as_ref() else {
            return Ok(None);
        };
        let dev = pool.nu.device();
        let h = pool.nu.dim(1)?;
        Ok(Some(CisMix {
            feat_to_compact: mx.feat_to_compact.clone(),
            nu: Tensor::cat(&[&pool.nu, &Tensor::zeros((1, h), DType::F32, dev)?], 0)?,
            c: Tensor::cat(&[&pool.c, &Tensor::zeros(1, DType::F32, dev)?], 0)?,
            self_share: mx.self_share.clone(),
        }))
    }

    /// Under a mixture, what it adds over a built `pool` to each cis gene's
    /// composed dictionary row `μ_{m(g)} + r_g` and bias `b_{m(g)} + b_g`, so
    /// phase 2 projects against the scores phase 1 fitted: `α (ν̃_g − r_g)` and
    /// `α (c̃_g − b_g)`. `r_c`, `b_c` are the cis genes' residual rows and
    /// biases. `None` without a mixture.
    pub fn dictionary_blend(
        &self,
        pool: &CisPool,
        r_c: &Tensor,
        b_c: &Tensor,
    ) -> CResult<Option<CisBlend>> {
        let Some(mx) = self.mixture.as_ref() else {
            return Ok(None);
        };
        let alpha = f64::from(mx.alpha);
        Ok(Some(CisBlend {
            feat: self.compact_feat.to_vec1::<u32>()?,
            row: ((&pool.nu - r_c)? * alpha)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            bias: ((&pool.c - b_c)? * alpha)?.to_vec1::<f32>()?,
        }))
    }

    /// The cis genes' own rows `ρ_c = μ_{m(g)} + r_c`, `[n_compact, H]`.
    fn gene_rows(&self, mu: &Tensor, r_c: &Tensor) -> CResult<Tensor> {
        gather_rows(mu, &self.compact_gene_module)? + r_c
    }

    /// The alignment gap over the units `e` `[n_u, H]`: per cis gene
    /// `var_u(⟨e_u, ρ_g − ν̃_g⟩) = d_gᵀ Σ d_g`, averaged over the genes, with
    /// `Σ` detached. `pool` is built over the module table `mu`; `r_c` are
    /// the cis genes' residual rows.
    pub fn align_gap(
        &self,
        pool: &CisPool,
        mu: &Tensor,
        r_c: &Tensor,
        e: &Tensor,
    ) -> CResult<Tensor> {
        let d = (self.gene_rows(mu, r_c)? - &pool.nu)?;
        (d.matmul(&unit_covariance(e)?)? * &d)?.sum(1)?.mean_all()
    }

    /// Host readout after training over a built `pool`: shared θ, pair shares
    /// `w̃`, the gap over the units `e_u` `[n_u, H]`, and each pair's evidence
    /// `corr` — the correlation across the units of the peak module's score
    /// `⟨e_u, μ_{m(p)}⟩` and the gene's own `⟨e_u, ρ_g⟩`. With `Σ` the units'
    /// covariance it is `μᵀΣρ / √(μᵀΣμ · ρᵀΣρ)`; `0` for a score that does
    /// not vary.
    pub fn readout(
        &self,
        pool: &CisPool,
        mu: &Tensor,
        r_c: &Tensor,
        e_u: &Tensor,
    ) -> CResult<CisGateReadout> {
        let sigma = unit_covariance(e_u)?;
        let rho_c = self.gene_rows(mu, r_c)?;
        let rho_s = rho_c.matmul(&sigma)?;
        let var_g = (&rho_s * &rho_c)?.sum(1)?;
        let var_m = (mu.matmul(&sigma)? * mu)?.sum(1)?;
        let cov = (gather_rows(&rho_s, &self.pair_gene)? * gather_rows(mu, &self.pair_module)?)?
            .sum(1)?
            .to_vec1::<f32>()?;
        let den = (gather_rows(&var_g, &self.pair_gene)?
            * gather_rows(&var_m, &self.pair_module)?)?
        .sqrt()?
        .to_vec1::<f32>()?;
        let corr = cov
            .iter()
            .zip(&den)
            .map(|(&c, &d)| {
                if d > 1e-12 {
                    (c / d).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect();
        Ok(CisGateReadout {
            theta0: self.theta0.as_tensor().to_scalar::<f32>()?,
            theta1: self.theta1.as_tensor().to_scalar::<f32>()?,
            w: pool.w.to_vec1::<f32>()?,
            corr,
            align_gap: self.align_gap(pool, mu, r_c, e_u)?.to_scalar::<f32>()?,
            source_idx: self.source_idx.clone(),
        })
    }
}

/// The pool of one step (see [`CisGateParams::pool`]).
pub struct CisPool {
    /// Each gene's share of each pair, `[n_pairs]`.
    pub w: Tensor,
    /// Pooled rows `ν̃`, `[n_compact, H]`.
    pub nu: Tensor,
    /// Pooled biases `c̃`, `[n_compact]`.
    pub c: Tensor,
}

/// What the gene level mixes in: pooled rows and biases on the compact gene
/// axis, and each gene's own share.
pub struct CisMix {
    feat_to_compact: Tensor,
    pub nu: Tensor,
    pub c: Tensor,
    self_share: Tensor,
}

impl CisMix {
    /// `(s r + (1 − s) ν̃_g, s b + (1 − s) c̃_g)` — `s = 1 − α` for a cis gene,
    /// `1` for a gene without pairs — for the genes `g_ids` whose residual
    /// rows `r` `[n, H]` and biases `bias` `[n]` are already gathered.
    pub fn gene_rows(
        &self,
        g_ids: &Tensor,
        r: &Tensor,
        bias: &Tensor,
    ) -> CResult<(Tensor, Tensor)> {
        let compact = self.feat_to_compact.index_select(g_ids, 0)?;
        let nu = gather_rows(&self.nu, &compact)?;
        let c = gather_rows(&self.c, &compact)?;
        let s = self.self_share.index_select(&compact, 0)?;
        let other = s.affine(-1.0, 1.0)?;
        let r = (r.broadcast_mul(&s.unsqueeze(1)?)? + nu.broadcast_mul(&other.unsqueeze(1)?)?)?;
        let bias = ((bias * &s)? + (c * &other)?)?;
        Ok((r, bias))
    }

    /// The same mix on fresh leaves cut from this graph, for slices that
    /// each run their own backward.
    pub fn detached(&self) -> CResult<CisMixLeaves> {
        let leaf = |t: &Tensor| Var::from_tensor(&t.detach());
        let vars = [leaf(&self.nu)?, leaf(&self.c)?];
        Ok(CisMixLeaves {
            mix: CisMix {
                feat_to_compact: self.feat_to_compact.clone(),
                nu: vars[0].as_tensor().clone(),
                c: vars[1].as_tensor().clone(),
                self_share: self.self_share.clone(),
            },
            vars,
        })
    }
}

/// A [`CisMix`] on leaves, and the leaves themselves.
pub struct CisMixLeaves {
    pub mix: CisMix,
    vars: [Var; 2],
}

impl CisMixLeaves {
    /// The gradient `grads` holds on the leaves, carried back through `graph`
    /// (the mix the leaves were cut from): `Σ ⟨graph_i, ∂L/∂leaf_i⟩`
    /// differentiated. `None` when no leaf took a gradient this step.
    pub fn backprop(&self, graph: &CisMix, grads: &GradStore) -> CResult<Option<GradStore>> {
        let outs = [&graph.nu, &graph.c];
        let mut surrogate: Option<Tensor> = None;
        for (v, out) in self.vars.iter().zip(outs) {
            if let Some(g) = grads.get(v) {
                let term = (out * g)?.sum_all()?;
                surrogate = Some(match surrogate {
                    Some(s) => (s + term)?,
                    None => term,
                });
            }
        }
        surrogate.map(|s| s.backward()).transpose()
    }
}

/// What a mixture adds to the cis genes' dictionary rows (see
/// [`CisGateParams::dictionary_blend`]).
pub struct CisBlend {
    /// Feature id of every cis gene.
    feat: Vec<u32>,
    /// Per cis gene, added to its row: `[n × H]` row-major.
    row: Vec<f32>,
    /// Per cis gene, added to its bias.
    bias: Vec<f32>,
}

impl CisBlend {
    /// Apply to a composed dictionary (one row per feature, `b` alike).
    pub fn apply(&self, rho: &mut DMatrix<f32>, b: &mut [f32]) {
        let h = rho.ncols();
        debug_assert_eq!(self.row.len(), self.feat.len() * h);
        for (i, &f) in self.feat.iter().enumerate() {
            let f = f as usize;
            for k in 0..h {
                rho[(f, k)] += self.row[i * h + k];
            }
            b[f] += self.bias[i];
        }
    }
}

/// Trained cis scalars + pair shares (caller pair order via
/// [`Self::source_idx`]).
#[derive(Clone, Debug)]
pub struct CisGateReadout {
    pub theta0: f32,
    pub theta1: f32,
    /// `w̃_gp`, the gene's share of each pair, aligned with [`Self::source_idx`].
    pub w: Vec<f32>,
    /// Per pair, the data's evidence (see [`CisGateParams::readout`]),
    /// aligned with [`Self::source_idx`].
    pub corr: Vec<f32>,
    /// The alignment gap over all units at the end of training.
    pub align_gap: f32,
    /// Indices into the original [`CisGates`] pair arrays passed to the fit.
    pub source_idx: Vec<u32>,
}

#[cfg(test)]
#[path = "cis_gates_tests.rs"]
mod tests;
