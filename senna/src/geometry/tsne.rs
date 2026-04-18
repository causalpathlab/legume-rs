#![allow(
    clippy::items_after_statements,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

use candle_util::candle_core::*;

/// t-SNE with an Rtsne-aligned update rule: per-parameter adaptive gains
/// (Jacobs 1988), momentum switching, strong early exaggeration, and init
/// rescaling to std = 1e-4. Also supports optional cell-count weighting of
/// the joint P and a densMAP-style density-preservation auxiliary loss.
/// Built on candle for automatic differentiation.
///
/// References:
/// - van der Maaten & Hinton, *JMLR* 2008 — t-SNE.
/// - Jacobs, *Neural Networks* 1988 — increased rate of convergence through
///   learning rate adaptation (delta-bar-delta / adaptive gains).
/// - Narayan, Berger & Cho, *Nat Biotechnol* 2021 — densMAP / den-SNE
///   density-preserving extension.
pub struct TSne {
    perplexity: f32,
    learning_rate: f32,
    initial_momentum: f32,
    final_momentum: f32,
    n_iter: usize,
    early_exaggeration: f32,
    early_exaggeration_iter: usize,
    /// Optional per-point weights. When set, `P_sym`[i,j] is multiplied by
    /// √(`w_i` · `w_j`) before renormalization so that heavier points dominate
    /// the KL objective. Typically `w_i = pb_cell_count[i]` so big
    /// pseudobulks drive the layout geometry.
    weights: Option<Vec<f32>>,
    /// densMAP-style density-preservation strength. When `> 0` and
    /// `weights` is set, an auxiliary loss pins each point's soft low-D
    /// local radius `R_i = Σ_j P_ij · d_ij(Y)` to `∝ 1/√n_i`, so dense
    /// regions (big PBs) shrink and sparse regions (small PBs) spread.
    density_lambda: f32,
}

impl Default for TSne {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            learning_rate: 200.0,
            initial_momentum: 0.5,
            final_momentum: 0.8,
            n_iter: 1000,
            early_exaggeration: 12.0,
            early_exaggeration_iter: 250,
            weights: None,
            density_lambda: 0.0,
        }
    }
}

impl TSne {
    pub fn perplexity(mut self, p: f32) -> Self {
        self.perplexity = p;
        self
    }

    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Attach per-point weights (typically PB cell counts) so the KL
    /// objective upweights pairs of heavy points. Big pseudobulks dominate
    /// the layout geometry; small ones are "along for the ride".
    pub fn weights(mut self, w: &[usize]) -> Self {
        self.weights = Some(w.iter().map(|&x| x as f32).collect());
        self
    }

    /// Enable densMAP-style density preservation with strength `lambda`.
    /// Requires `weights` to be set — the density target is `1/√w_i`.
    /// A good starting value is ~0.1. Set to 0 to disable.
    pub fn density_lambda(mut self, lambda: f32) -> Self {
        self.density_lambda = lambda.max(0.0);
        self
    }

    /// Run t-SNE on a precomputed distance matrix with optional init.
    /// Returns a flat n×2 embedding.
    ///
    /// The update rule follows Rtsne / van der Maaten's reference
    /// implementation: adaptive per-parameter gains, momentum switch at
    /// `early_exaggeration_iter`, init rescaled to std = 1e-4 so early
    /// exaggeration has room to separate clusters, and Y centered every
    /// iteration to prevent drift.
    pub fn fit(&self, distances: &[f32], n: usize, init: Option<&[f32]>) -> Result<Vec<f32>> {
        let device = Device::Cpu;

        // Compute P (high-dimensional affinities) from distances.
        let p = self.compute_joint_probabilities(distances, n)?;
        let p_tensor = Tensor::from_vec(p, (n, n), &device)?;

        // Pre-build the centered log-target tensor for density preservation:
        //   target_i ∝ 1/√w_i   =>   log target_i = -½ log w_i
        let density_target = if self.density_lambda > 0.0 {
            self.weights.as_ref().and_then(|w| {
                if w.len() != n {
                    return None;
                }
                let mut log_t: Vec<f32> = w.iter().map(|&wi| -0.5 * wi.max(1.0).ln()).collect();
                let mean = log_t.iter().sum::<f32>() / n as f32;
                for v in &mut log_t {
                    *v -= mean;
                }
                Tensor::from_vec(log_t, (n,), &device).ok()
            })
        } else {
            None
        };

        // Initial Y: user-provided init or small random uniform in [-0.5, 0.5].
        let mut y_flat: Vec<f32> = if let Some(coords) = init {
            coords.to_vec()
        } else {
            use rand::RngExt;
            let mut rng = rand::rng();
            (0..n * 2).map(|_| rng.random::<f32>() - 0.5).collect()
        };

        // Rescale the init to std = 1e-4 (Rtsne / openTSNE convention).
        // Keeps the *shape* of the init while shrinking the scale so that
        // early exaggeration has room to push clusters apart.
        {
            let mean_x: f32 = (0..n).map(|i| y_flat[i * 2]).sum::<f32>() / n as f32;
            let mean_y: f32 = (0..n).map(|i| y_flat[i * 2 + 1]).sum::<f32>() / n as f32;
            for i in 0..n {
                y_flat[i * 2] -= mean_x;
                y_flat[i * 2 + 1] -= mean_y;
            }
            let var: f32 =
                y_flat.iter().map(|v| v * v).sum::<f32>() / (y_flat.len() as f32).max(1.0);
            let std = var.sqrt().max(1e-12);
            let scale = 1e-4 / std;
            for v in &mut y_flat {
                *v *= scale;
            }
        }

        let y = Var::from_tensor(&Tensor::from_vec(y_flat, (n, 2), &device)?)?;

        // Auto-scale learning rate with n (Kobak & Berens 2019). The fixed
        // lr=200 Rtsne convention diverges for small n: early-exaggeration
        // gradients grow faster than the Student-t tail can damp them,
        // blowing Y up to overflow within ~200 iters at n ≈ 70.
        let lr_auto = (n as f32 / self.early_exaggeration).max(50.0);
        let learning_rate = self.learning_rate.min(lr_auto);

        // Per-parameter velocity and adaptive gains (Jacobs 1988).
        let mut velocity: Vec<f32> = vec![0.0; n * 2];
        let mut gains: Vec<f32> = vec![1.0; n * 2];
        const MIN_GAIN: f32 = 0.01;

        for iter in 0..self.n_iter {
            let p_mult = if iter < self.early_exaggeration_iter {
                self.early_exaggeration
            } else {
                1.0
            };
            let p_scaled = (&p_tensor * f64::from(p_mult))?;

            // KL(+ density). Density term disabled during early exaggeration
            // so gradients through log R don't blow up at tiny distances.
            let kl = self.kl_divergence(&y, &p_scaled, n)?;
            let loss = if iter >= self.early_exaggeration_iter {
                if let Some(log_target) = &density_target {
                    let dens = self.density_loss(&y, &p_tensor, log_target, n)?;
                    (&kl + (dens * f64::from(self.density_lambda))?)?
                } else {
                    kl
                }
            } else {
                kl
            };

            let grad = loss.backward()?;
            let y_grad = grad.get(&y).unwrap();
            let g_flat: Vec<f32> = y_grad.flatten_all()?.to_vec1::<f32>()?;

            // Adaptive gains (van der Maaten reference):
            //   sign(grad) ≠ sign(velocity) (oscillating) → gain += 0.2
            //   sign(grad) == sign(velocity) (steady)      → gain *= 0.8
            for i in 0..gains.len() {
                if (velocity[i] > 0.0) == (g_flat[i] > 0.0) {
                    gains[i] *= 0.8;
                } else {
                    gains[i] += 0.2;
                }
                if gains[i] < MIN_GAIN {
                    gains[i] = MIN_GAIN;
                }
            }

            // Momentum switch at the end of early exaggeration.
            let momentum = if iter < self.early_exaggeration_iter {
                self.initial_momentum
            } else {
                self.final_momentum
            };

            // v ← momentum · v − lr · gains · grad
            for i in 0..velocity.len() {
                velocity[i] = momentum * velocity[i] - learning_rate * gains[i] * g_flat[i];
            }

            // Y ← Y + v, then center so the mean is at the origin.
            let y_vec: Vec<f32> = y.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            let mut y_new: Vec<f32> = y_vec
                .iter()
                .zip(velocity.iter())
                .map(|(yv, vv)| yv + vv)
                .collect();
            let mean_x: f32 = (0..n).map(|i| y_new[i * 2]).sum::<f32>() / n as f32;
            let mean_y: f32 = (0..n).map(|i| y_new[i * 2 + 1]).sum::<f32>() / n as f32;
            for i in 0..n {
                y_new[i * 2] -= mean_x;
                y_new[i * 2 + 1] -= mean_y;
            }
            y.set(&Tensor::from_vec(y_new, (n, 2), &device)?)?;
        }

        let result = y.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        Ok(result)
    }

    /// Compute joint probabilities P from distance matrix using perplexity calibration
    fn compute_joint_probabilities(&self, distances: &[f32], n: usize) -> Result<Vec<f32>> {
        let target_entropy = self.perplexity.ln();
        let mut p = vec![0.0f32; n * n];

        // For each point, find sigma via binary search to match target perplexity
        for i in 0..n {
            let (lo, hi) = (1e-10f32, 1e4f32);
            let sigma = self.binary_search_sigma(i, distances, n, target_entropy, lo, hi);

            // Compute conditional probabilities p(j|i)
            let mut row_sum = 0.0f32;
            for j in 0..n {
                if i != j {
                    let d = distances[i * n + j];
                    let val = (-d * d / (2.0 * sigma * sigma)).exp();
                    p[i * n + j] = val;
                    row_sum += val;
                }
            }
            // Normalize
            if row_sum > 1e-10 {
                for j in 0..n {
                    p[i * n + j] /= row_sum;
                }
            }
        }

        // Symmetrize: P_ij = (P_j|i + P_i|j) / 2n
        let mut p_sym = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                p_sym[i * n + j] = (p[i * n + j] + p[j * n + i]) / (2.0 * n as f32);
            }
        }

        // Optional weighted t-SNE: multiply each pair by √(w_i · w_j) and
        // renormalize so Σ P = 1. Pairs involving heavy points gain weight in
        // the KL objective, pulling the layout toward big-cluster geometry.
        if let Some(w) = self.weights.as_deref() {
            if w.len() == n {
                let sqrt_w: Vec<f32> = w.iter().map(|x| x.max(0.0).sqrt()).collect();
                let mut total = 0.0f32;
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            let v = p_sym[i * n + j] * sqrt_w[i] * sqrt_w[j];
                            p_sym[i * n + j] = v;
                            total += v;
                        }
                    }
                }
                if total > 1e-12 {
                    for v in &mut p_sym {
                        *v /= total;
                    }
                }
            }
        }

        // Ensure minimum probability for numerical stability
        let min_p = 1e-12f32;
        for v in &mut p_sym {
            *v = v.max(min_p);
        }

        Ok(p_sym)
    }

    /// Binary search for sigma to match target entropy (perplexity)
    fn binary_search_sigma(
        &self,
        i: usize,
        distances: &[f32],
        n: usize,
        target: f32,
        mut lo: f32,
        mut hi: f32,
    ) -> f32 {
        for _ in 0..50 {
            let mid = f32::midpoint(lo, hi);
            let entropy = self.compute_entropy(i, distances, n, mid);
            if entropy > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        f32::midpoint(lo, hi)
    }

    /// Compute entropy for a given sigma
    fn compute_entropy(&self, i: usize, distances: &[f32], n: usize, sigma: f32) -> f32 {
        let mut probs = vec![0.0f32; n];
        let mut sum = 0.0f32;

        for j in 0..n {
            if i != j {
                let d = distances[i * n + j];
                probs[j] = (-d * d / (2.0 * sigma * sigma)).exp();
                sum += probs[j];
            }
        }

        if sum < 1e-10 {
            return 0.0;
        }

        let mut entropy = 0.0f32;
        for (j, &prob_j) in probs.iter().enumerate() {
            if i != j && prob_j > 0.0 {
                let p = prob_j / sum;
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Compute KL divergence between P and Q (Student-t in low-dim)
    fn kl_divergence(&self, y: &Var, p: &Tensor, n: usize) -> Result<Tensor> {
        // Compute pairwise squared distances in low-dim
        // ||y_i - y_j||^2 = ||y_i||^2 + ||y_j||^2 - 2 * y_i · y_j
        let y_sq = y.as_tensor().sqr()?.sum(D::Minus1)?; // (n,)
        let y_sq_row = y_sq.reshape((n, 1))?; // (n, 1)
        let y_sq_col = y_sq.reshape((1, n))?; // (1, n)
        let dot = y.as_tensor().matmul(&y.as_tensor().t()?)?; // (n, n)
        let dist_sq = (y_sq_row.broadcast_add(&y_sq_col)? - (&dot * 2.0)?)?;

        // Student-t: q_ij = (1 + ||y_i - y_j||^2)^-1
        let q_unnorm = (&dist_sq + 1.0)?.recip()?;

        // Zero out diagonal
        let mask = Tensor::eye(n, DType::F32, y.as_tensor().device())?;
        let q_unnorm = (&q_unnorm * (1.0 - &mask)?)?;

        // Normalize
        let q_sum = q_unnorm.sum_all()?;
        let q = q_unnorm.broadcast_div(&q_sum)?.clamp(1e-12, 1.0)?;

        // KL divergence: sum(P * log(P/Q))
        let log_pq = p.broadcast_div(&q)?.log()?;
        let kl = (p * log_pq)?.sum_all()?;

        Ok(kl)
    }

    /// densMAP-style density-preservation loss (Narayan, Berger & Cho,
    /// *Nat Biotechnol* 2021). For each point i, compute a soft local radius
    /// in 2D as a P-weighted mean distance,
    ///     `R_i` = `Σ_j` `P_ij` · `d_ij(Y)`,
    /// and penalize the centered log-radius against a centered target
    /// `log_target_i = -½ log(w_i)` (i.e. target radius ∝ `1/√w_i`, so dense
    /// high-D regions stay dense in 2D). Both sides are mean-centered, so no
    /// absolute scale offset needs to be fit.
    fn density_loss(
        &self,
        y: &Var,
        p: &Tensor,
        log_target_centered: &Tensor,
        n: usize,
    ) -> Result<Tensor> {
        // Pairwise Euclidean distances in 2D.
        let y_sq = y.as_tensor().sqr()?.sum(D::Minus1)?;
        let y_sq_row = y_sq.reshape((n, 1))?;
        let y_sq_col = y_sq.reshape((1, n))?;
        let dot = y.as_tensor().matmul(&y.as_tensor().t()?)?;
        let dist_sq = (y_sq_row.broadcast_add(&y_sq_col)? - (&dot * 2.0)?)?;
        let dist = dist_sq.clamp(1e-12f32, f32::INFINITY)?.sqrt()?;

        // R_i = Σ_j P_ij · d_ij, row-wise weighted sum of 2D distances.
        // Floor with 1e-3 so the gradient of log R stays bounded when the
        // layout is still collapsed near the origin.
        let r = (p * &dist)?.sum(D::Minus1)?;
        let log_r = r.clamp(1e-3f32, f32::INFINITY)?.log()?;

        // Center log R so only the pattern (not the absolute scale) is fit.
        let mean_log_r = log_r.mean_all()?;
        let log_r_centered = log_r.broadcast_sub(&mean_log_r)?;

        // Loss = mean squared deviation between centered low-D log-radii
        // and centered target.
        let diff = (&log_r_centered - log_target_centered)?;
        let loss = diff.sqr()?.mean_all()?;
        Ok(loss)
    }
}

/// Convert similarity matrix to distance matrix
pub fn similarity_to_distance(sim: &[f32], _n: usize) -> Vec<f32> {
    sim.iter()
        .map(|&s| (2.0 * (1.0 - s.clamp(-1.0, 1.0))).sqrt()) // Angular distance
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsne_small() {
        // Small test: 5 points
        let n = 5;
        let distances: Vec<f32> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                if i == j {
                    0.0
                } else {
                    (i as f32 - j as f32).abs()
                }
            })
            .collect();

        let tsne = TSne::default().perplexity(2.0).n_iter(100);
        let result = tsne.fit(&distances, n, None).unwrap();
        assert_eq!(result.len(), n * 2);
    }
}
