use candle_util::candle_core::{DType, Device, Result, Tensor, Var, D};

/// t-SNE implementation using candle for automatic differentiation
pub struct TSne {
    perplexity: f32,
    learning_rate: f32,
    momentum: f32,
    n_iter: usize,
    early_exaggeration: f32,
    early_exaggeration_iter: usize,
}

impl Default for TSne {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            learning_rate: 200.0,
            momentum: 0.8,
            n_iter: 1000,
            early_exaggeration: 4.0,
            early_exaggeration_iter: 250,
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

    /// Run t-SNE on distance matrix, with optional initialization
    /// Returns n x 2 embedding
    pub fn fit(&self, distances: &[f32], n: usize, init: Option<&[f32]>) -> Result<Vec<f32>> {
        let device = Device::Cpu;

        // Compute P (high-dimensional affinities) from distances
        let p = self.compute_joint_probabilities(distances, n)?;
        let p_tensor = Tensor::from_vec(p.clone(), (n, n), &device)?;

        // Initialize embedding: use provided init or random
        let y_init: Vec<f32> = match init {
            Some(coords) => coords.to_vec(),
            None => {
                use rand::Rng;
                let mut rng = rand::rng();
                (0..n * 2).map(|_| rng.random::<f32>() * 0.01).collect()
            }
        };

        let y = Var::from_tensor(&Tensor::from_vec(y_init, (n, 2), &device)?)?;
        let mut velocity = Tensor::zeros((n, 2), DType::F32, &device)?;

        // Gradient descent
        for iter in 0..self.n_iter {
            // Early exaggeration
            let p_mult = if iter < self.early_exaggeration_iter {
                self.early_exaggeration
            } else {
                1.0
            };
            let p_scaled = (&p_tensor * p_mult as f64)?;

            // Compute Q (low-dimensional affinities) and KL divergence
            let loss = self.kl_divergence(&y, &p_scaled, n)?;

            // Backward pass
            let grad = loss.backward()?;
            let y_grad = grad.get(&y).unwrap();

            // Momentum update
            velocity =
                ((&velocity * self.momentum as f64)? - (y_grad * self.learning_rate as f64)?)?;
            y.set(&(y.as_tensor() + &velocity)?)?;
        }

        // Extract result
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
            let mid = (lo + hi) / 2.0;
            let entropy = self.compute_entropy(i, distances, n, mid);
            if entropy > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (lo + hi) / 2.0
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
        for j in 0..n {
            if i != j && probs[j] > 0.0 {
                let p = probs[j] / sum;
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
