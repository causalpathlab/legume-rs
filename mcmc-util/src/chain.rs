use crate::traits::{EssParam, EssParamSummary};

/// Collected MCMC samples with log-likelihoods.
pub struct McmcChain<P: EssParam> {
    pub samples: Vec<P>,
    pub log_likelihoods: Vec<f32>,
}

impl<P: EssParam> McmcChain<P> {
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }
}

impl<P: EssParamSummary> McmcChain<P> {
    /// Element-wise posterior mean across samples.
    pub fn posterior_mean(&self) -> Vec<f32> {
        let n = self.n_samples();
        if n == 0 {
            return vec![];
        }
        let d = self.samples[0].dim();
        let mut mean = vec![0.0f32; d];
        for sample in &self.samples {
            let s = sample.as_slice();
            for (m, &v) in mean.iter_mut().zip(s.iter()) {
                *m += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for m in &mut mean {
            *m *= inv_n;
        }
        mean
    }

    /// Element-wise posterior variance across samples.
    pub fn posterior_variance(&self) -> Vec<f32> {
        let n = self.n_samples();
        if n < 2 {
            return vec![];
        }
        let mean = self.posterior_mean();
        let d = mean.len();
        let mut var = vec![0.0f32; d];
        for sample in &self.samples {
            let s = sample.as_slice();
            for i in 0..d {
                let diff = s[i] - mean[i];
                var[i] += diff * diff;
            }
        }
        let inv = 1.0 / (n - 1) as f32;
        for v in &mut var {
            *v *= inv;
        }
        var
    }

    /// Element-wise quantile (0 <= q <= 1) across samples.
    pub fn quantile(&self, q: f32) -> Vec<f32> {
        let n = self.n_samples();
        if n == 0 {
            return vec![];
        }
        let d = self.samples[0].dim();
        let mut result = vec![0.0f32; d];

        // Collect values per dimension
        let mut vals = vec![Vec::with_capacity(n); d];
        for sample in &self.samples {
            let s = sample.as_slice();
            for i in 0..d {
                vals[i].push(s[i]);
            }
        }

        for i in 0..d {
            vals[i].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = (q * (n - 1) as f32).clamp(0.0, (n - 1) as f32);
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            if lo == hi {
                result[i] = vals[i][lo];
            } else {
                let frac = idx - lo as f32;
                result[i] = vals[i][lo] * (1.0 - frac) + vals[i][hi] * frac;
            }
        }
        result
    }
}
