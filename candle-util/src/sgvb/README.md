# Black Box Variational Inference (SGVB)

Stochastic Gradient Variational Bayes using the score function (REINFORCE) estimator with control variates.

## Features

- [x] Black-box likelihood: no gradients required through likelihood function
- [x] Score function gradient: `∇ELBO ≈ E[(normalized_reward) * ∇log q(θ)]`
- [x] Control variate normalization: `(reward - mean) / std`
- [x] Reparameterized Gaussian: `θ = μ + σ * ε` where `ε ~ N(0, I)`
- [x] Learnable prior scale τ in `p(θ) = N(0, τ²I)`
- [x] Linear regression model: `η = X * θ` where `θ ~ q(θ) = N(μ, σ²I)`
- [x] Multiple eta tensors supported for multimodal likelihoods

## Module Structure

| File | Exports |
|------|---------|
| `traits.rs` | `BlackBoxLikelihood`, `VariationalDistribution`, `Prior` |
| `gaussian.rs` | `GaussianVariational` |
| `prior.rs` | `GaussianPrior`, `FixedGaussianPrior` |
| `sgvb.rs` | `sgvb_loss`, `compute_elbo`, `SGVBConfig` |
| `linear_regression_model.rs` | `LinearRegressionSGVB` |

## Usage

```rust
use candle_util::sgvb::{
    BlackBoxLikelihood, LinearRegressionSGVB, GaussianPrior, SGVBConfig
};

// 1. Define your black-box likelihood
struct MyLikelihood { y: Tensor }

impl BlackBoxLikelihood for MyLikelihood {
    fn log_likelihood(&self, etas: &[&Tensor]) -> Result<Tensor> {
        let eta = etas[0];  // shape (S, n, k)
        // compute log p(y | eta), return shape (S,)
    }
}

// 2. Create model
let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;  // τ = 1.0
let config = SGVBConfig::new(10, true);  // S=10 samples, normalize=true
let model = LinearRegressionSGVB::new(
    vb, x_design, k, likelihood, prior, config
)?;

// 3. Training loop
for _ in 0..num_iters {
    let loss = model.loss()?;
    optimizer.backward_step(&loss)?;
}

// 4. Posterior inference
let coef_mean = model.coef_mean();      // μ_θ: (p, k)
let coef_std = model.coef_std()?;       // σ_θ: (p, k)
let predictions = model.eta_mean()?;    // X @ μ_θ: (n, k)
```

## Tensor Shapes

| Variable | Shape | Description |
|----------|-------|-------------|
| X | (n, p) | Design matrix |
| θ | (S, p, k) | Sampled coefficients |
| μ, σ | (p, k) | Variational parameters |
| η | (S, n, k) | Linear predictor samples |
| reward | (S,) | Per-sample ELBO estimate |

## Algorithm

The ELBO is:
```
ELBO = E_q[log p(y|η) + log p(θ) - log q(θ)]
```

Score function gradient (REINFORCE):
```
∇ELBO ≈ (1/S) Σ_s [ normalized_reward_s * ∇log q(θ_s) ]
```

With control variate:
```
normalized_reward = (reward - mean(reward)) / std(reward)
```
