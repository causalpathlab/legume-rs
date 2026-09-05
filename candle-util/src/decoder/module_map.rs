//! A hard gene → module map for the module-collapsed decoder.
//!
//! The dense masked heads score every gene, which costs one `[K, D]` product
//! and a few dozen `[N, D]` kernels per step. With a map from genes to `M`
//! modules the decoder's dense targets become module totals, its logits
//! `[K, M]`, and its per-gene questions are answered by the query head on a
//! sampled set. The gene-level embedding ρ stays: the module embedding is the
//! within-module mean of ρ, computed each step, so ρ trains through it, and
//! the pinned within-module share `π_{g|m}` splits a module's rate back over
//! its genes wherever a gene is asked about. The identity map (every gene its
//! own module) reproduces the dense heads exactly.

use crate::fast_index::index_add_rows;
use candle_core::{DType, Device, Result, Tensor};
use nalgebra::DMatrix;

pub struct ModuleMap {
    /// `[D]` u32, module of each gene.
    fine_to_coarse_d: Tensor,
    /// `[M, 1]` reciprocal module sizes (0 for an empty module).
    inv_size_m1: Tensor,
    /// `[D]` `log π_{g|m(g)}`.
    log_share_d: Tensor,
    host_fine_to_coarse: Vec<usize>,
    host_log_share: Vec<f32>,
    n_fine: usize,
    n_coarse: usize,
    identity: bool,
}

impl ModuleMap {
    /// `fine_to_coarse[g]` is the module of gene `g`; `share_of_gene[g]` is
    /// the gene's pinned share of its module's rate, summing to one within a
    /// module.
    pub fn new(fine_to_coarse: &[usize], share_of_gene: &[f32], dev: &Device) -> Result<Self> {
        let d = fine_to_coarse.len();
        if share_of_gene.len() != d {
            candle_core::bail!("module map: {} genes but {} shares", d, share_of_gene.len());
        }
        let m = fine_to_coarse.iter().max().map_or(0, |&x| x + 1);
        let mut size = vec![0f32; m];
        for &c in fine_to_coarse {
            size[c] += 1.0;
        }
        let inv_size: Vec<f32> = size
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();
        let log_share: Vec<f32> = share_of_gene.iter().map(|&s| s.max(1e-12).ln()).collect();
        let identity = m == d && fine_to_coarse.iter().enumerate().all(|(g, &c)| g == c);
        Ok(Self {
            fine_to_coarse_d: Tensor::from_vec(
                fine_to_coarse.iter().map(|&c| c as u32).collect(),
                d,
                dev,
            )?,
            inv_size_m1: Tensor::from_vec(inv_size, (m, 1), dev)?,
            log_share_d: Tensor::from_vec(log_share.clone(), d, dev)?,
            host_fine_to_coarse: fine_to_coarse.to_vec(),
            host_log_share: log_share,
            n_fine: d,
            n_coarse: m,
            identity,
        })
    }

    /// Every gene its own module, share one.
    pub fn identity(d: usize, dev: &Device) -> Result<Self> {
        let f2c: Vec<usize> = (0..d).collect();
        Self::new(&f2c, &vec![1.0; d], dev)
    }

    #[must_use]
    pub fn n_fine(&self) -> usize {
        self.n_fine
    }

    #[must_use]
    pub fn n_coarse(&self) -> usize {
        self.n_coarse
    }

    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    #[must_use]
    pub fn host_fine_to_coarse(&self) -> &[usize] {
        &self.host_fine_to_coarse
    }

    #[must_use]
    pub fn host_log_share(&self) -> &[f32] {
        &self.host_log_share
    }

    /// `[M, H]` within-module mean of a `[D, H]` table; ρ itself under the
    /// identity map. Differentiable: each gene gets `1/|m|` of its module's
    /// gradient.
    pub fn coarsen_mean_dh(&self, rho: &Tensor) -> Result<Tensor> {
        if self.identity {
            return Ok(rho.clone());
        }
        let h = rho.dim(1)?;
        let zeros = Tensor::zeros((self.n_coarse, h), rho.dtype(), rho.device())?;
        index_add_rows(&zeros, &self.fine_to_coarse_d, rho)?.broadcast_mul(&self.inv_size_m1)
    }

    /// Module of each gene id, same shape as `ids` (u32).
    pub fn modules_of(&self, ids: &Tensor) -> Result<Tensor> {
        if self.identity {
            return Ok(ids.clone());
        }
        self.fine_to_coarse_d
            .index_select(&ids.flatten_all()?, 0)?
            .reshape(ids.shape())
    }

    /// `log π_{g|m(g)}` at each gene id, same shape as `ids`.
    pub fn log_share_at(&self, ids: &Tensor) -> Result<Tensor> {
        if self.identity {
            return Tensor::zeros(ids.shape(), DType::F32, ids.device());
        }
        self.log_share_d
            .index_select(&ids.flatten_all()?, 0)?
            .reshape(ids.shape())
    }

    /// Broadcast a per-module row `[1, M]` to its genes `[1, D]`.
    pub fn expand_1m_to_1d(&self, x_1m: &Tensor) -> Result<Tensor> {
        if self.identity {
            return Ok(x_1m.clone());
        }
        x_1m.index_select(&self.fine_to_coarse_d, 1)
    }

    /// Sum the columns of a `[P, D]` host matrix into `[P, M]`.
    #[must_use]
    pub fn aggregate_columns_host(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        if self.identity {
            return x.clone();
        }
        let p = x.nrows();
        let mut out = DMatrix::<f32>::zeros(p, self.n_coarse);
        for (g, &c) in self.host_fine_to_coarse.iter().enumerate() {
            let src = x.column(g);
            let mut dst = out.column_mut(c);
            dst += src;
        }
        out
    }
}

#[cfg(test)]
#[path = "module_map_tests.rs"]
mod module_map_tests;
