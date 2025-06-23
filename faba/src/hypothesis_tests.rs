#![allow(dead_code)]

use statrs::distribution::{Binomial, DiscreteCDF};

/// Binomial test
/// - `expected (failure, success)`
/// - `observed (failure, success)`
///
pub struct BinomTest<T>
where
    T: num_traits::ToPrimitive,
{
    pub expected: (T, T),
    pub observed: (T, T),
}

impl<T> BinomTest<T>
where
    T: num_traits::ToPrimitive,
{
    pub fn pvalue_greater(&self) -> anyhow::Result<f64> {
        if let (Some(exp1), Some(exp0), Some(obs1), Some(obs0)) = (
            self.expected.1.to_u64(),
            self.expected.0.to_u64(),
            self.observed.1.to_u64(),
            self.observed.0.to_u64(),
        ) {
            let null_pr = (exp1 as f64) / ((exp0 + exp1) as f64).max(1.0);
            let nobs = (obs1 + obs0) as u64;
            let distrib = Binomial::new(null_pr, nobs)?;
            Ok(1.0 - distrib.cdf(obs1.saturating_sub(1)))
        } else {
            Err(anyhow::anyhow!("failed to construct binomial test"))
        }
    }
    pub fn pvalue_less(&self) -> anyhow::Result<f64> {
        if let (Some(exp1), Some(exp0), Some(obs1), Some(obs0)) = (
            self.expected.1.to_u64(),
            self.expected.0.to_u64(),
            self.observed.1.to_u64(),
            self.observed.0.to_u64(),
        ) {
            let null_pr = (exp1 as f64) / ((exp0 + exp1) as f64).max(1.0);
            let nobs = (obs1 + obs0) as u64;
            let distrib = Binomial::new(null_pr, nobs)?;
            Ok(distrib.cdf(obs1))
        } else {
            Err(anyhow::anyhow!("failed to construct binomial test"))
        }
    }
}
