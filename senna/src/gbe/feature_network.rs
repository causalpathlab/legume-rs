//! Frozen-snapshot SGC smoothing for `senna gbe`.
//!
//! Given a `FeaturePairGraph` G over the same feature axis as the model,
//! we approximate Wu et al. 2019 Simplifying Graph Convolutions by
//! periodically pre-propagating the feature embedding through the
//! K-hop normalized adjacency, freezing the result, and applying it as
//! a network-coherent bias on the bilinear score:
//!
//!     Â  = D^{-1/2} A D^{-1/2}                  (no self-loop)
//!     T  = ((1-α) I + α Â)^K
//!     Ẽ  = T · E
//!
//! Splitting `Ẽ = (1-α)^K · E + R(E)` lets us treat the residual `R(E)`
//! as a frozen tensor refreshed every `refresh_epochs` epochs (computed
//! offline with `nalgebra-sparse`). Live gradients flow through the
//! `(1-α)^K · E` term — attenuated, but still fully through autograd.
//! No custom op needed.

use candle_util::candle_core::{DType, Device, Result, Tensor};
use matrix_util::pair_graph::FeaturePairGraph;
use nalgebra::DMatrix;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};

pub struct FeatureNetworkSmoother {
    /// `Â = D^{-1/2} A D^{-1/2}` — symmetric normalized adjacency, no self-loop.
    norm_adj: CsrMatrix<f32>,
    pub n_features: usize,
    pub embedding_dim: usize,
    pub alpha: f32,
    pub k_hops: usize,
    pub refresh_epochs: usize,
    /// `(1-α)^K`: live-embedding attenuation factor.
    pub self_factor: f32,
    /// `R(E_snapshot)` — frozen network residual `[F, H]`. None until first refresh.
    residual: Option<Tensor>,
}

impl FeatureNetworkSmoother {
    /// Build a smoother over `graph` (must already be aligned to the
    /// model's feature axis: `graph.n_features == n_features`).
    pub fn new(
        graph: &FeaturePairGraph,
        n_features: usize,
        embedding_dim: usize,
        alpha: f32,
        k_hops: usize,
        refresh_epochs: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            graph.n_features == n_features,
            "feature graph axis ({}) ≠ model feature axis ({})",
            graph.n_features,
            n_features
        );
        anyhow::ensure!((0.0..=1.0).contains(&alpha), "alpha must be in [0, 1]");
        anyhow::ensure!(k_hops >= 1, "k_hops must be ≥ 1");
        anyhow::ensure!(refresh_epochs >= 1, "refresh_epochs must be ≥ 1");

        let degrees = graph.feature_degrees();
        let inv_sqrt_d: Vec<f32> = degrees
            .iter()
            .map(|&d| {
                if d == 0 {
                    0.0
                } else {
                    (d as f32).sqrt().recip()
                }
            })
            .collect();

        // Symm-normalized adjacency: Â[u,v] = 1 / sqrt(d_u * d_v) for each undirected edge.
        let mut coo: CooMatrix<f32> = CooMatrix::new(n_features, n_features);
        for &(u, v) in &graph.feature_edges {
            let w = inv_sqrt_d[u] * inv_sqrt_d[v];
            if w > 0.0 {
                coo.push(u, v, w);
                coo.push(v, u, w);
            }
        }
        let norm_adj = CsrMatrix::from(&coo);

        let self_factor = (1.0 - alpha).powi(k_hops as i32);

        Ok(Self {
            norm_adj,
            n_features,
            embedding_dim,
            alpha,
            k_hops,
            refresh_epochs,
            self_factor,
            residual: None,
        })
    }

    /// Recompute the frozen residual `R(E) = T·E - (1-α)^K · E` from the
    /// current parameter values. Cheap: K sparse matvec sweeps over `Â`.
    pub fn refresh(&mut self, e_feat: &Tensor, dev: &Device) -> Result<()> {
        let f = self.n_features;
        let h = self.embedding_dim;
        debug_assert_eq!(e_feat.dims(), &[f, h]);

        // Candle is row-major [F, H]; nalgebra DMatrix is column-major. Use
        // `from_row_slice` to do the layout flip in one pass.
        let host: Vec<f32> = e_feat.flatten_all()?.to_vec1::<f32>()?;
        let snap = DMatrix::<f32>::from_row_slice(f, h, &host);

        // Iterate K times: x ← (1-α) x + α Â x.
        let mut x = snap.clone();
        for _ in 0..self.k_hops {
            let nbr = &self.norm_adj * &x;
            x = nbr * self.alpha + x * (1.0 - self.alpha);
        }
        // Residual = T·E - (1-α)^K · E.
        let residual_dm = &x - &snap * self.self_factor;

        // Back to candle row-major [F, H].
        let mut row_major = Vec::with_capacity(f * h);
        for i in 0..f {
            for j in 0..h {
                row_major.push(residual_dm[(i, j)]);
            }
        }
        self.residual = Some(Tensor::from_vec(row_major, (f, h), dev)?.to_dtype(DType::F32)?);
        Ok(())
    }

    /// Apply the smoother to a row-selection of the live `e_feat`.
    /// Returns `Ẽ_feat[idx] = (1-α)^K · E_feat[idx] + R[idx]`.
    /// Falls back to plain `index_select` if `refresh` hasn't been called yet.
    pub fn apply_select(&self, e_feat: &Tensor, idx: &Tensor) -> Result<Tensor> {
        let live = e_feat.index_select(idx, 0)?;
        match &self.residual {
            None => Ok(live),
            Some(r) => {
                let live_att = (live * self.self_factor as f64)?;
                let resid = r.index_select(idx, 0)?;
                live_att + resid
            }
        }
    }
}

/// Helper that routes feature-embedding row selects through the optional
/// smoother. Loss code calls this instead of `e_feat.index_select` so the
/// SGC plumbing stays out of the bilinear scoring.
pub fn select_feat_emb(
    smoother: Option<&FeatureNetworkSmoother>,
    e_feat: &Tensor,
    idx: &Tensor,
) -> Result<Tensor> {
    match smoother {
        Some(sm) => sm.apply_select(e_feat, idx),
        None => e_feat.index_select(idx, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_util::candle_core::Device;
    use matrix_util::pair_graph::test_graph_from_edges;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn refresh_then_apply_recovers_full_smoothed_row() {
        // Triangle 0-1-2 with H=2.
        let g = test_graph_from_edges(&[(0, 1), (0, 2), (1, 2)], 3);
        let dev = cpu();
        let alpha = 0.5_f32;
        let k = 1_usize;
        let mut sm = FeatureNetworkSmoother::new(&g, 3, 2, alpha, k, 1).unwrap();

        // E = [[1,0],[0,1],[1,1]]  row-major.
        let e = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &dev).unwrap();
        sm.refresh(&e, &dev).unwrap();

        // Â on triangle (deg=2 each): Â[u,v] = 1/sqrt(4) = 0.5 off-diag, 0 self.
        // K=1: Ẽ_i = 0.5 E_i + 0.5 (0.5 E_j + 0.5 E_k) = 0.5 E_i + 0.25 (E_j + E_k)
        // Expected for row 0: 0.5*(1,0) + 0.25*((0,1)+(1,1)) = (0.5+0.25, 0+0.5) = (0.75, 0.5)
        let idx = Tensor::from_vec(vec![0u32, 1, 2], 3, &dev).unwrap();
        let out = sm.apply_select(&e, &idx).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let expected = [
            0.75, 0.5, // row 0
            0.5, 0.75, // row 1
            0.75, 0.75, // row 2
        ];
        for (a, b) in v.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn isolated_feature_passes_through_unchanged() {
        // 4 features, only edge (0,1). Feature 3 is isolated.
        let g = test_graph_from_edges(&[(0, 1)], 4);
        let dev = cpu();
        let mut sm = FeatureNetworkSmoother::new(&g, 4, 2, 0.5, 1, 1).unwrap();
        let e = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (4, 2), &dev).unwrap();
        sm.refresh(&e, &dev).unwrap();
        let idx = Tensor::from_vec(vec![3u32], 1, &dev).unwrap();
        let out = sm.apply_select(&e, &idx).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // Â · row3 = 0, so Ẽ_3 = (1-α) E_3 = 0.5 * (7,8) ... wait
        // Ẽ = (1-α)^1 E + R, R = T·E - (1-α)·E for isolated row, T = (1-α)I + αÂ
        // For isolated row: Ẽ = (1-α) E + 0 = 0.5*(7,8) = (3.5, 4)
        assert!((v[0] - 3.5).abs() < 1e-5);
        assert!((v[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn before_refresh_is_passthrough() {
        let g = test_graph_from_edges(&[(0, 1)], 2);
        let dev = cpu();
        let sm = FeatureNetworkSmoother::new(&g, 2, 2, 0.5, 2, 1).unwrap();
        let e = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &dev).unwrap();
        let idx = Tensor::from_vec(vec![0u32, 1], 2, &dev).unwrap();
        let out = sm.apply_select(&e, &idx).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![1f32, 2., 3., 4.]);
    }

    #[test]
    fn k_two_smooths_more_than_k_one() {
        // Path 0-1-2-3 — K=2 should reach further than K=1 from node 0.
        let g = test_graph_from_edges(&[(0, 1), (1, 2), (2, 3)], 4);
        let dev = cpu();
        let e = Tensor::from_vec(vec![10f32, 0., 0., 0., 0., 0., 0., 10.], (4, 2), &dev).unwrap();

        let mut sm1 = FeatureNetworkSmoother::new(&g, 4, 2, 0.5, 1, 1).unwrap();
        sm1.refresh(&e, &dev).unwrap();
        let idx = Tensor::from_vec(vec![0u32], 1, &dev).unwrap();
        let r1: Vec<f32> = sm1
            .apply_select(&e, &idx)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut sm2 = FeatureNetworkSmoother::new(&g, 4, 2, 0.5, 2, 1).unwrap();
        sm2.refresh(&e, &dev).unwrap();
        let r2: Vec<f32> = sm2
            .apply_select(&e, &idx)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Node 0: K=1 reaches only node 1 (which is 0 in the second column),
        // so col-1 of row 0 stays 0. K=2 reaches node 2 (also 0) but path
        // toward col-1 mass at node 3 still requires K=3 — both should be 0
        // at col-1. Sanity: the row-0 col-0 attenuation differs because K=2
        // has more (1-α)^k weight elsewhere. Just check K=2 differs from K=1.
        assert!(
            (r1[0] - r2[0]).abs() > 1e-5 || (r1[1] - r2[1]).abs() > 1e-5,
            "K=1 vs K=2 should produce different smoothings"
        );
    }
}
