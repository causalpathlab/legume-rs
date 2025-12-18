#![allow(dead_code)]

use statrs::distribution::{Binomial, DiscreteCDF};

/// Counts of failures and successes in a binomial experiment
#[derive(Debug, Clone, Copy)]
pub struct BinomialCounts<T>
where
    T: num_traits::ToPrimitive,
{
    pub num_failure: T,
    pub num_success: T,
}

impl<T> BinomialCounts<T>
where
    T: num_traits::ToPrimitive + std::ops::Add<Output = T> + Copy,
{
    pub fn new(num_failure: T, num_success: T) -> Self {
        Self {
            num_failure,
            num_success,
        }
    }

    /// Add a pseudocount to both failure and success counts for regularization
    pub fn with_pseudocount(num_failure: T, num_success: T, pseudocount: T) -> Self {
        Self {
            num_failure: num_failure + pseudocount,
            num_success: num_success + pseudocount,
        }
    }
}

/// Binomial test
/// - `expected`: expected counts under null hypothesis
/// - `observed`: observed counts in the data
///
pub struct BinomTest<T>
where
    T: num_traits::ToPrimitive,
{
    pub expected: BinomialCounts<T>,
    pub observed: BinomialCounts<T>,
}

impl<T> BinomTest<T>
where
    T: num_traits::ToPrimitive,
{
    /// Test if observed success rate is greater than expected
    /// Returns P(X >= obs_success | X ~ Binomial(n, p_expected))
    pub fn pvalue_greater(&self) -> anyhow::Result<f64> {
        if let (Some(exp_success), Some(exp_failure), Some(obs_success), Some(obs_failure)) = (
            self.expected.num_success.to_u64(),
            self.expected.num_failure.to_u64(),
            self.observed.num_success.to_u64(),
            self.observed.num_failure.to_u64(),
        ) {
            let null_pr = (exp_success as f64) / ((exp_failure + exp_success) as f64).max(1.0);
            let nobs = obs_success + obs_failure;
            let distrib = Binomial::new(null_pr, nobs)?;
            // P(X >= k) = P(X > k-1) = sf(k-1)
            // Using sf() is more numerically stable than 1 - cdf()
            Ok(distrib.sf(obs_success.saturating_sub(1)))
        } else {
            Err(anyhow::anyhow!("failed to construct binomial test"))
        }
    }

    /// Test if observed success rate is less than expected
    /// Returns P(X <= obs_success | X ~ Binomial(n, p_expected))
    pub fn pvalue_less(&self) -> anyhow::Result<f64> {
        if let (Some(exp_success), Some(exp_failure), Some(obs_success), Some(obs_failure)) = (
            self.expected.num_success.to_u64(),
            self.expected.num_failure.to_u64(),
            self.observed.num_success.to_u64(),
            self.observed.num_failure.to_u64(),
        ) {
            let null_pr = (exp_success as f64) / ((exp_failure + exp_success) as f64).max(1.0);
            let nobs = obs_success + obs_failure;
            let distrib = Binomial::new(null_pr, nobs)?;
            Ok(distrib.cdf(obs_success))
        } else {
            Err(anyhow::anyhow!("failed to construct binomial test"))
        }
    }
}
