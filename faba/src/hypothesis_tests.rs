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
    pub fn pvalue_greater(&self) -> anyhow::Result<f32> {
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
            Ok(distrib.sf(obs_success.saturating_sub(1)) as f32)
        } else {
            Err(anyhow::anyhow!("failed to construct binomial test"))
        }
    }

    /// Test if observed success rate is less than expected
    /// Returns P(X <= obs_success | X ~ Binomial(n, p_expected))
    pub fn pvalue_less(&self) -> anyhow::Result<f32> {
        if let (Some(exp_success), Some(exp_failure), Some(obs_success), Some(obs_failure)) = (
            self.expected.num_success.to_u64(),
            self.expected.num_failure.to_u64(),
            self.observed.num_success.to_u64(),
            self.observed.num_failure.to_u64(),
        ) {
            let null_pr = (exp_success as f64) / ((exp_failure + exp_success) as f64).max(1.0);
            let nobs = obs_success + obs_failure;
            let distrib = Binomial::new(null_pr, nobs)?;
            Ok(distrib.cdf(obs_success) as f32)
        } else {
            Err(anyhow::anyhow!("failed to construct binomial test"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pvalue_greater_significant() {
        // Expected 10% rate (90C/10T), observed 50% rate (50C/50T)
        let test = BinomTest {
            expected: BinomialCounts::new(90u64, 10u64),
            observed: BinomialCounts::new(50u64, 50u64),
        };
        let pv = test.pvalue_greater().unwrap();
        assert!(pv < 0.001, "pvalue_greater should be < 0.001, got {pv}");
    }

    #[test]
    fn test_pvalue_greater_nonsignificant() {
        // Expected 10% rate (90C/10T), observed ~12% rate (88C/12T)
        let test = BinomTest {
            expected: BinomialCounts::new(90u64, 10u64),
            observed: BinomialCounts::new(88u64, 12u64),
        };
        let pv = test.pvalue_greater().unwrap();
        assert!(pv > 0.1, "pvalue_greater should be > 0.1, got {pv}");
    }

    #[test]
    fn test_pvalue_less() {
        // Expected 50% rate, observed only 10% (90C/10T)
        let test = BinomTest {
            expected: BinomialCounts::new(50u64, 50u64),
            observed: BinomialCounts::new(90u64, 10u64),
        };
        let pv = test.pvalue_less().unwrap();
        assert!(pv < 0.001, "pvalue_less should be < 0.001, got {pv}");
    }

    #[test]
    fn test_pvalue_at_null() {
        // Expected and observed both 50/50
        let test = BinomTest {
            expected: BinomialCounts::new(50u64, 50u64),
            observed: BinomialCounts::new(50u64, 50u64),
        };
        let pv = test.pvalue_greater().unwrap();
        assert!(
            (pv - 0.5).abs() < 0.1,
            "pvalue_greater at null should be ~0.5, got {pv}"
        );
    }

    #[test]
    fn test_pseudocount() {
        let c1 = BinomialCounts::with_pseudocount(0u64, 0u64, 1u64);
        assert_eq!(c1.num_failure, 1);
        assert_eq!(c1.num_success, 1);

        let c2 = BinomialCounts::with_pseudocount(10u64, 5u64, 2u64);
        assert_eq!(c2.num_failure, 12);
        assert_eq!(c2.num_success, 7);
    }
}
