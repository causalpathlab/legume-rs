//! A hard feature → coarse-feature map for the collapsed decoder.
//!
//! The dense masked heads score every feature, which costs one `[K, D]`
//! product and a few dozen `[N, D]` kernels per step. With a map from features
//! to `M` coarse features the decoder's dense targets become group totals, its
//! logits `[K, M]`, and its per-feature questions are answered by the query
//! head on a sampled set.
//!
//! Two things stay at feature level, which is why coarsening here is not a
//! loss of resolution the way it is for a family that coarsens both sides.
//! The per-feature embedding ρ trains throughout: a group's embedding is the
//! mean of its members' ρ, recomputed every step, so gradient reaches every
//! row. And the pinned within-group share `π_{g|m}` splits a group's rate back
//! over its members wherever one is asked about. That share is fixed from the
//! data, not learned.
//!
//! The identity map (every feature its own group) reproduces the dense heads
//! exactly.
//!
//! The membership here is FIXED: it arrives already computed and is held as
//! plain tensors, never as parameters, so no gradient reaches it. That is what
//! separates a coarsening from a module — a module learns which features group
//! together (see the masked encoder's centroids), a coarsening is told.

use crate::fast_index::index_add_rows;
use candle_core::{DType, Device, Result, Tensor};
use nalgebra::DMatrix;
use rayon::prelude::*;

pub struct CoarseningMap {
    /// `[D]` u32, coarse feature of each gene.
    fine_to_coarse_d: Tensor,
    /// `[M, 1]` reciprocal coarse feature sizes (0 for an empty coarse feature).
    inv_size_m1: Tensor,
    /// `[D]` `log π_{g|m(g)}`.
    log_share_d: Tensor,
    host_fine_to_coarse: Vec<usize>,
    /// Genes of each coarse feature, for the host aggregation.
    host_coarse_to_fine: Vec<Vec<usize>>,
    host_log_share: Vec<f32>,
    n_fine: usize,
    n_coarse: usize,
    identity: bool,
}

impl CoarseningMap {
    /// `fine_to_coarse[g]` is the coarse feature of gene `g`; `share_of_gene[g]` is
    /// the gene's pinned share of its coarse feature's rate, summing to one within a
    /// coarse feature.
    pub fn new(fine_to_coarse: &[usize], share_of_gene: &[f32], dev: &Device) -> Result<Self> {
        let d = fine_to_coarse.len();
        if share_of_gene.len() != d {
            candle_core::bail!("coarse feature map: {} genes but {} shares", d, share_of_gene.len());
        }
        let m = fine_to_coarse.iter().max().map_or(0, |&x| x + 1);
        let mut coarse_to_fine: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (g, &c) in fine_to_coarse.iter().enumerate() {
            coarse_to_fine[c].push(g);
        }
        let inv_size: Vec<f32> = coarse_to_fine
            .iter()
            .map(|genes| {
                if genes.is_empty() {
                    0.0
                } else {
                    1.0 / genes.len() as f32
                }
            })
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
            host_coarse_to_fine: coarse_to_fine,
            host_log_share: log_share,
            n_fine: d,
            n_coarse: m,
            identity,
        })
    }

    /// Every gene its own coarse feature, share one.
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

    /// `[M, H]` within-group mean of a `[D, H]` table; ρ itself under the
    /// identity map. Differentiable: each gene gets `1/|m|` of its coarse feature's
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
    pub fn groups_of(&self, ids: &Tensor) -> Result<Tensor> {
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

    /// `[1, D]` `log π_{g|m(g)}` on the device.
    #[must_use]
    pub fn log_share_1d(&self) -> Tensor {
        self.log_share_d.unsqueeze(0).expect("[D] → [1, D]")
    }

    /// Sum the columns of an `[N, D]` device tensor into `[N, M]`
    /// (differentiable through [`index_add_rows`]); the input itself under
    /// the identity map. Done on the transpose, so the `[D]` map indexes rows
    /// directly and no `[N, D]` index tensor is formed.
    pub fn aggregate_columns(&self, x_nd: &Tensor) -> Result<Tensor> {
        if self.identity {
            return Ok(x_nd.clone());
        }
        let (n, d) = x_nd.dims2()?;
        if d != self.n_fine {
            candle_core::bail!(
                "coarse feature map: {} genes but the input has {d} columns",
                self.n_fine
            );
        }
        let zeros = Tensor::zeros((self.n_coarse, n), x_nd.dtype(), x_nd.device())?;
        index_add_rows(&zeros, &self.fine_to_coarse_d, &x_nd.t()?)?
            .t()?
            .contiguous()
    }

    /// Sum the columns of a `[P, D]` host matrix into `[P, M]`, one coarse feature's
    /// column per worker.
    #[must_use]
    pub fn aggregate_columns_host(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        if self.identity {
            return x.clone();
        }
        let p = x.nrows();
        let mut out = DMatrix::<f32>::zeros(p, self.n_coarse);
        out.as_mut_slice()
            .par_chunks_mut(p.max(1))
            .zip(self.host_coarse_to_fine.par_iter())
            .for_each(|(col, genes)| {
                for &g in genes {
                    for (o, s) in col.iter_mut().zip(x.column(g).iter()) {
                        *o += *s;
                    }
                }
            });
        out
    }
}

#[cfg(test)]
#[path = "coarsening_map_tests.rs"]
mod coarsening_map_tests;
