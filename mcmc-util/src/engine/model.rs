use rand::rngs::SmallRng;

/// Trait for any MCMC model that can be run by the engine.
///
/// Separates model-specific logic (state, sweep, summarization) from
/// the generic MCMC loop (warmup, thinning, sample collection).
pub trait McmcModel {
    /// Full sampler state, mutated each sweep.
    type State;
    /// One collected posterior sample (extracted from state).
    type Sample: Send;
    /// Final result with posterior summaries.
    type Result;

    /// Initialize the sampler state.
    fn init(&self, rng: &mut SmallRng) -> Self::State;

    /// Run one full sweep (e.g. one Gibbs iteration, one ESS step).
    fn sweep(&self, state: &mut Self::State, rng: &mut SmallRng);

    /// Extract a sample from the current state.
    fn collect(&self, state: &Self::State) -> Self::Sample;

    /// Compute posterior summaries from collected samples.
    fn summarize(&self, samples: Vec<Self::Sample>) -> Self::Result;
}
