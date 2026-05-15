//! Sparse residual graph-diffusion block over packed top-K gene
//! representations.
//!
//! Used by `IndexedEmbeddingEncoder` when a feature-feature graph is
//! supplied. Mathematically this is one GCN step (Kipf & Welling 2017):
//!
//! ```text
//!   Ã      = (A + I) row-normalised over each cell's measured top-K
//!   out    = v + γ · (Ã · v)                                  [N, K, H]
//! ```
//!
//! ## Implementation
//!
//! Ã is *never* materialised as a dense `[N, K, K]` tensor. The loader
//! pre-normalises Ã once per cell at attach time and ships a flat
//! [`SparseEdgeBatch`] (`dst_flat`, `src_flat`, `weight`) per
//! minibatch — three `[E]` tensors where `E` is the actual edge count
//! across the batch (`≪ N · K²` for K=512, avg degree ~50). Forward is
//! just:
//!
//! ```text
//!   v_flat       = v.reshape((N*K, H))
//!   v_at_src     = v_flat[src_flat, :]            # index_select
//!   v_weighted   = w · v_at_src                    # broadcast mul
//!   smoothed     = zeros((N*K, H));
//!                  scatter_add(smoothed, dst_flat, v_weighted)
//!   smoothed     = smoothed.reshape((N, K, H))
//!   out          = v + γ · smoothed                # γ inits at 0
//! ```
//!
//! Cost: `O(E · H)` vs the dense `O(N · K² · H)` matmul. For a typical
//! K=512, B=100, avg degree 50, this is ~100× fewer FLOPs and ~25× less
//! per-minibatch memory (4 MB of flat tensors vs 100 MB of dense adj).
//!
//! ## Why this stays stable
//!
//! * γ is the only learnable parameter, **init at zero** ⇒ the block is
//!   the exact identity on `v` at the start of training. Downstream
//!   FC+BN sees the same input distribution as the no-graph encoder; γ
//!   only grows as the gradient finds the graph contribution useful.
//! * Each row of Ã sums to 1 (self-loop + neighbour weights pre-
//!   normalised at cache build), so smoothed slots stay on the same
//!   scale as `v` — sum-pool downstream keeps its native magnitude.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::candle_indexed_data_loader::SparseEdgeBatch;

/// One γ-gated GCN diffusion step. The only learnable parameter is the
/// diffusion scalar `γ` (zero-init ⇒ identity at init).
pub struct GraphAttentionBlock {
    d_model: usize,
    edge_bias_scale: Tensor,
}

impl GraphAttentionBlock {
    pub fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        let edge_bias_scale = vb.get_with_hints((1,), "edge_bias_scale", candle_nn::init::ZERO)?;
        Ok(Self {
            d_model,
            edge_bias_scale,
        })
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Apply the sparse diffusion block.
    ///
    /// * `v`       — `[N, K, H]` per-slot gene representation (value-gated embedding)
    /// * `edges`   — pre-normalised sparse edges scattered from the per-cell cache
    pub fn forward(&self, v: &Tensor, edges: &SparseEdgeBatch) -> Result<Tensor> {
        let (n, k, h) = v.dims3()?;
        debug_assert_eq!(h, self.d_model);

        let v_flat = v.reshape((n * k, h))?;
        let v_at_src = v_flat.index_select(&edges.src_flat, 0)?; // [E, H]
        let v_weighted = v_at_src.broadcast_mul(&edges.weight.unsqueeze(1)?)?; // [E, H]
        let smoothed_flat = Tensor::zeros((n * k, h), v.dtype(), v.device())?.index_add(
            &edges.dst_flat,
            &v_weighted,
            0,
        )?;
        let smoothed = smoothed_flat.reshape((n, k, h))?;

        let gamma = self.edge_bias_scale.reshape((1, 1, 1))?;
        let delta = smoothed.broadcast_mul(&gamma)?;
        v + delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_indexed_data_loader::SparseEdgeBatch;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn small_edges(device: &Device) -> SparseEdgeBatch {
        // Two cells, K=3. Cell 0: real slots 0,1,2 (full graph 0-1-2 chain
        // with self-loops, each row normalised to sum=1). Cell 1: real
        // slots 0,1 only; slot 2 padded (no edges touching it).
        let dst =
            Tensor::from_vec(vec![0u32, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4], (11,), device).unwrap();
        let src =
            Tensor::from_vec(vec![0u32, 1, 0, 1, 2, 1, 2, 3, 4, 3, 4], (11,), device).unwrap();
        let w = Tensor::from_vec(
            vec![
                0.5f32,
                0.5,
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            (11,),
            device,
        )
        .unwrap();
        SparseEdgeBatch {
            dst_flat: dst,
            src_flat: src,
            weight: w,
        }
    }

    #[test]
    fn gat_block_forward_shape_and_finite() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let d = 8usize;
        let block = GraphAttentionBlock::new(d, vb.pp("gat")).unwrap();

        let n = 2usize;
        let k = 3usize;
        let v = Tensor::randn(0.0f32, 1.0, (n, k, d), &device).unwrap();
        let edges = small_edges(&device);

        let out = block.forward(&v, &edges).unwrap();
        assert_eq!(out.dims(), &[n, k, d]);
        for row in out.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!(row.is_finite(), "non-finite output {row}");
        }
    }

    /// At init γ=0, so the block must be the exact identity on `v`.
    #[test]
    fn gat_identity_at_init() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let d = 4usize;
        let block = GraphAttentionBlock::new(d, vb.pp("gat")).unwrap();

        let v = Tensor::randn(0.0f32, 1.0, (2, 3, d), &device).unwrap();
        let edges = small_edges(&device);
        let out = block.forward(&v, &edges).unwrap();
        let v_flat = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let out_flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in v_flat.iter().zip(out_flat.iter()) {
            assert!((a - b).abs() < 1e-6, "γ=0 should give identity: {a} vs {b}");
        }
    }
}
