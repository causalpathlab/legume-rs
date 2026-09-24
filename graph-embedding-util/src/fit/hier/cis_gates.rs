//! Optional cis peak→gene gates inside phase-1 gene scores.
//!
//! ```text
//! w_gp = abc · max(0, θ₀ + θ₁ z_log_contact + θ₃ ⟨μ_{m(p)}, ρ_g⟩)
//! λ_u,m = ⟨e_u, μ_m⟩ + b_m
//! a_ug  = Σ_p w_gp · λ_u,m(p)
//! η_ug  = γ₂ · ρ_ug + γ₁ · a_ug      (γ = softplus; θ, γ shared by every gene)
//! ```
//! Gene-level log-softmax then uses `η` instead of `ρ_ug = ⟨e,r⟩+b`.
//!
//! # How it is computed
//!
//! `λ` is linear in `e_u`, so the pooled term is a gene row and a gene bias:
//!
//! ```text
//! a_ug = ⟨e_u, ν_g⟩ + c_g        ν_g = Σ_p w_gp μ_{m(p)}      c_g = Σ_p w_gp b_{m(p)}
//! η_ug = ⟨e_u, γ₂ r_g + γ₁ ν_g⟩ + γ₂ b_g + γ₁ c_g
//! ```
//!
//! [`CisGateParams::pool`] builds `ν`, `c` once per step over the cis genes
//! and the modules their peaks fall in (`W` and `⟨ρ, μ⟩`, both
//! `[n_cis_genes × n_peak_modules]`), and [`CisMix::gene_rows`] folds them
//! into the gathered gene rows before the gene-level matmul. No tensor in a
//! gene batch carries a module axis.
//!
//! A step split over threads builds the pool once: [`CisMix::detached`] hands
//! the slices shared leaves, and [`CisMixLeaves::backprop`] carries their
//! summed gradient back through the pool.

use legume_numeric::candle::candle_core::backprop::GradStore;
use legume_numeric::candle::candle_core::{DType, Device, Result as CResult, Tensor, Var};
use legume_numeric::candle::fast_index::{gather_rows, index_add_rows};
use legume_numeric::candle::loss::log_sigmoid;
use rustc_hash::FxHashMap;

/// γ₁ at the start: the links carry a small share and still take a gradient.
const INIT_GAMMA1: f64 = 0.01;
/// γ₂ at the start: the residual score as it is.
const INIT_GAMMA2: f64 = 1.0;

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
    /// module-only gene has no gene level for its links to feed.
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
    /// module id): near-empty or scattered peaks carry no link signal.
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

/// Trainable scalars + device copies of the pair tables, indexed on the
/// COMPACT axes: the genes that have pairs, and the modules their peaks fall in.
pub struct CisGateParams {
    pub theta0: Var,
    pub theta1: Var,
    pub theta3: Var,
    /// softplus⁻¹(γ₁); starts at γ₁ = [`INIT_GAMMA1`].
    pub raw_gamma1: Var,
    /// softplus⁻¹(γ₂); starts at γ₂ = [`INIT_GAMMA2`].
    pub raw_gamma2: Var,
    /// Feature id of every compact gene, `[n_compact]`.
    compact_feat: Tensor,
    /// Module of every compact gene, `[n_compact]`.
    compact_gene_module: Tensor,
    /// Module id of every compact peak module, `[n_peak_modules]`.
    peak_modules: Tensor,
    /// `compact_gene · n_peak_modules + compact_module` per pair, `u32`
    /// (never through f32: the product passes 2^24).
    flat_idx: Tensor,
    abc: Tensor,
    z_log_contact: Tensor,
    /// Feature → compact gene id; a gene without pairs maps to the zero row
    /// `n_compact`.
    feat_to_compact: Tensor,
    pub n_compact: usize,
    pub n_peak_modules: usize,
    /// Index into the caller's original [`CisGates`] order (after dropping
    /// module-only genes' pairs).
    pub source_idx: Vec<u32>,
}

/// `softplus(x) = −log σ(−x)`.
fn softplus(t: &Tensor) -> CResult<Tensor> {
    log_sigmoid(&t.neg()?)?.neg()
}

/// `softplus⁻¹(y) = ln(eʸ − 1)`.
fn inv_softplus(y: f64) -> f64 {
    y.exp_m1().ln()
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

impl CisGateParams {
    /// `module_of` is the partition of the whole feature axis (`n_features`
    /// long): it gives each cis gene's own module for `ρ_g = μ_{m(g)} + r_g`.
    pub fn new(
        spec: &CisGatesResolved,
        module_of: &[u32],
        n_features: usize,
        dev: &Device,
    ) -> CResult<Self> {
        let n = spec.n_pairs();
        let scalar = |x: f64| -> CResult<Var> { Var::from_tensor(&Tensor::new(x as f32, dev)?) };
        let (gene_c, genes) = compact_ids(&spec.gene_feat);
        let (module_c, modules) = compact_ids(&spec.peak_module);
        let (n_compact, n_peak_modules) = (genes.len(), modules.len());
        let mut feat_to_compact = vec![n_compact as u32; n_features];
        for (c, &g) in genes.iter().enumerate() {
            feat_to_compact[g as usize] = c as u32;
        }
        let flat: Vec<u32> = gene_c
            .iter()
            .zip(&module_c)
            .map(|(&g, &m)| g * n_peak_modules as u32 + m)
            .collect();
        let gene_module: Vec<u32> = genes.iter().map(|&g| module_of[g as usize]).collect();

        Ok(Self {
            theta0: scalar(1.0)?,
            theta1: scalar(0.5)?,
            theta3: scalar(0.5)?,
            raw_gamma1: scalar(inv_softplus(INIT_GAMMA1))?,
            raw_gamma2: scalar(inv_softplus(INIT_GAMMA2))?,
            compact_feat: Tensor::from_vec(genes, n_compact, dev)?,
            compact_gene_module: Tensor::from_vec(gene_module, n_compact, dev)?,
            peak_modules: Tensor::from_vec(modules, n_peak_modules, dev)?,
            flat_idx: Tensor::from_vec(flat, n, dev)?,
            abc: Tensor::from_vec(spec.abc.clone(), n, dev)?,
            z_log_contact: Tensor::from_vec(spec.z_log_contact.clone(), n, dev)?,
            feat_to_compact: Tensor::from_vec(feat_to_compact, n_features, dev)?,
            n_compact,
            n_peak_modules,
            source_idx: spec.source_idx.clone(),
        })
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![
            self.theta0.clone(),
            self.theta1.clone(),
            self.theta3.clone(),
            self.raw_gamma1.clone(),
            self.raw_gamma2.clone(),
        ]
    }

    /// Feature ids of the cis genes, `[n_compact]`: the rows of `r` that
    /// [`Self::pool`] takes as `r_c`.
    #[must_use]
    pub fn compact_feat(&self) -> &Tensor {
        &self.compact_feat
    }

    pub fn gammas(&self) -> CResult<(Tensor, Tensor)> {
        Ok((
            softplus(self.raw_gamma1.as_tensor())?,
            softplus(self.raw_gamma2.as_tensor())?,
        ))
    }

    /// Pair weights `w` `[n_pairs]` and the pooled gene rows `ν`
    /// `[n_compact + 1, H]` and biases `c` `[n_compact + 1]` (last row zero,
    /// for genes without pairs). `mu` `[M, H]`, `b_m` `[M]` are the module
    /// tables; `r_c` `[n_compact, H]` the cis genes' residual rows.
    pub fn pool(&self, mu: &Tensor, b_m: &Tensor, r_c: &Tensor) -> CResult<CisPool> {
        let (n_c, n_pm) = (self.n_compact, self.n_peak_modules);
        let h = mu.dim(1)?;
        let dev = mu.device();
        let rho_c = (gather_rows(mu, &self.compact_gene_module)? + r_c)?;
        let mu_p = gather_rows(mu, &self.peak_modules)?;
        let b_p = gather_rows(b_m, &self.peak_modules)?;
        // ⟨μ_{m(p)}, ρ_g⟩ for every pair, read off the dense `[n_c, n_pm]` block.
        let agree = gather_rows(
            &rho_c.matmul(&mu_p.t()?)?.reshape((n_c * n_pm, 1))?,
            &self.flat_idx,
        )?
        .squeeze(1)?;
        let s = self
            .theta0
            .as_tensor()
            .broadcast_add(&self.z_log_contact.broadcast_mul(self.theta1.as_tensor())?)?
            .broadcast_add(&agree.broadcast_mul(self.theta3.as_tensor())?)?;
        let w = (&self.abc * s.relu()?)?;
        let w_gm = index_add_rows(
            &Tensor::zeros((n_c * n_pm, 1), DType::F32, dev)?,
            &self.flat_idx,
            &w.unsqueeze(1)?,
        )?
        .reshape((n_c, n_pm))?;
        let nu = Tensor::cat(
            &[
                &w_gm.matmul(&mu_p)?,
                &Tensor::zeros((1, h), DType::F32, dev)?,
            ],
            0,
        )?;
        let c = Tensor::cat(
            &[
                &w_gm.matmul(&b_p.unsqueeze(1)?)?.squeeze(1)?,
                &Tensor::zeros(1, DType::F32, dev)?,
            ],
            0,
        )?;
        Ok(CisPool { w, nu, c })
    }

    /// [`Self::pool`] and γ: what the gene level mixes in.
    pub fn mix(&self, mu: &Tensor, b_m: &Tensor, r_c: &Tensor) -> CResult<CisMix> {
        let pool = self.pool(mu, b_m, r_c)?;
        let (gamma1, gamma2) = self.gammas()?;
        Ok(CisMix {
            feat_to_compact: self.feat_to_compact.clone(),
            nu: pool.nu,
            c: pool.c,
            gamma1,
            gamma2,
        })
    }

    /// Host readout after training: shared θ, γ and pair weights `w`.
    pub fn readout(&self, mu: &Tensor, b_m: &Tensor, r_c: &Tensor) -> CResult<CisGateReadout> {
        let pool = self.pool(mu, b_m, r_c)?;
        let (g1, g2) = self.gammas()?;
        Ok(CisGateReadout {
            theta0: self.theta0.as_tensor().to_scalar::<f32>()?,
            theta1: self.theta1.as_tensor().to_scalar::<f32>()?,
            theta3: self.theta3.as_tensor().to_scalar::<f32>()?,
            gamma1: g1.to_scalar::<f32>()?,
            gamma2: g2.to_scalar::<f32>()?,
            w: pool.w.to_vec1::<f32>()?,
            source_idx: self.source_idx.clone(),
        })
    }
}

/// The pool of one step (see [`CisGateParams::pool`]).
pub struct CisPool {
    pub w: Tensor,
    pub nu: Tensor,
    pub c: Tensor,
}

/// What the gene level mixes in: pooled rows and biases on the compact gene
/// axis, and the two shared weights.
pub struct CisMix {
    feat_to_compact: Tensor,
    pub nu: Tensor,
    pub c: Tensor,
    pub gamma1: Tensor,
    pub gamma2: Tensor,
}

impl CisMix {
    /// `(γ₂ r + γ₁ ν_g, γ₂ b + γ₁ c_g)` for the genes `g_ids` whose residual
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
        let r = (r.broadcast_mul(&self.gamma2)? + nu.broadcast_mul(&self.gamma1)?)?;
        let bias = (bias.broadcast_mul(&self.gamma2)? + c.broadcast_mul(&self.gamma1)?)?;
        Ok((r, bias))
    }

    /// The same mix on fresh leaves cut from this graph, for slices that
    /// each run their own backward.
    pub fn detached(&self) -> CResult<CisMixLeaves> {
        let leaf = |t: &Tensor| Var::from_tensor(&t.detach());
        let vars = [
            leaf(&self.nu)?,
            leaf(&self.c)?,
            leaf(&self.gamma1)?,
            leaf(&self.gamma2)?,
        ];
        Ok(CisMixLeaves {
            mix: CisMix {
                feat_to_compact: self.feat_to_compact.clone(),
                nu: vars[0].as_tensor().clone(),
                c: vars[1].as_tensor().clone(),
                gamma1: vars[2].as_tensor().clone(),
                gamma2: vars[3].as_tensor().clone(),
            },
            vars,
        })
    }
}

/// A [`CisMix`] on leaves, and the leaves themselves.
pub struct CisMixLeaves {
    pub mix: CisMix,
    vars: [Var; 4],
}

impl CisMixLeaves {
    /// The gradient `grads` holds on the leaves, carried back through `graph`
    /// (the mix the leaves were cut from): `Σ ⟨graph_i, ∂L/∂leaf_i⟩`
    /// differentiated. `None` when no leaf took a gradient this step.
    pub fn backprop(&self, graph: &CisMix, grads: &GradStore) -> CResult<Option<GradStore>> {
        let outs = [&graph.nu, &graph.c, &graph.gamma1, &graph.gamma2];
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

/// Trained cis scalars + pair weights (caller pair order via [`Self::source_idx`]).
#[derive(Clone, Debug)]
pub struct CisGateReadout {
    pub theta0: f32,
    pub theta1: f32,
    pub theta3: f32,
    pub gamma1: f32,
    pub gamma2: f32,
    /// `w_gp` aligned with [`Self::source_idx`].
    pub w: Vec<f32>,
    /// Indices into the original [`CisGates`] pair arrays passed to the fit.
    pub source_idx: Vec<u32>,
}

#[cfg(test)]
#[path = "cis_gates_tests.rs"]
mod tests;
