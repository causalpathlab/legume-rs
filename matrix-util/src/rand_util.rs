//! Deterministic, thread-order-independent random sampling.
//!
//! The `SampleOps` samplers (`rnorm` / `runif` / `rgamma`) historically fed
//! `rand::rng()` (OS entropy) through `into_par_iter().map_init(...)`, which is
//! doubly non-reproducible: the RNG is unseeded, and rayon's work-stealing
//! assigns flat indices to thread-local RNGs in a scheduling-dependent order,
//! so even a fixed seed would not pin the output.
//!
//! [`collect_f32_seeded`] fixes both. It splits the output range into fixed
//! contiguous chunks, seeds each chunk from `(seed, chunk_index)`, and fills it
//! in order. Because the chunk boundaries and per-chunk seeds are pure
//! functions of `(seed, n)` — never of the thread schedule — the result is
//! byte-identical across runs, thread counts, and machines (`StdRng` is
//! ChaCha-based and portable), while still parallelizing across chunks.

use num_traits::NumCast;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::Distribution;
use rayon::prelude::*;

/// Elements generated per deterministic chunk. Large enough that per-chunk
/// `StdRng` seeding is negligible, small enough to keep every core busy.
const CHUNK: usize = 8192;

/// SplitMix64 avalanche of `(base, salt)`. Nearby `(seed, index)` pairs map to
/// well-separated 64-bit values, so per-chunk / per-tensor sub-streams derived
/// from a single user seed do not correlate.
#[inline]
pub fn mix_seed(base: u64, salt: u64) -> u64 {
    let mut z = base ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Base seed mixed with the FNV-1a hash of `name`, giving a stable, portable
/// sub-stream seed keyed by a string identity. Distinct names yield distinct
/// (well-separated) streams automatically, so callers do not have to hand-
/// assign disjoint integer salts.
#[inline]
pub fn name_seed(base: u64, name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for &b in name.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
    }
    mix_seed(base, h)
}

/// Draw `n` i.i.d. `f32` samples from `dist`, deterministically given `seed`
/// and independent of the thread schedule. See the module docs for why this is
/// reproducible where `into_par_iter().map_init(rand::rng, ...)` is not.
pub fn collect_f32_seeded<D>(n: usize, dist: D, seed: u64) -> Vec<f32>
where
    D: Distribution<f32> + Clone + Send + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    let n_chunks = n.div_ceil(CHUNK);
    // `collect` preserves the sequential order (chunk 0, then chunk 1, ...),
    // so the output is fixed regardless of how rayon schedules the chunks.
    (0..n_chunks)
        .into_par_iter()
        .flat_map_iter(move |ci| {
            let start = ci * CHUNK;
            let end = ((ci + 1) * CHUNK).min(n);
            // `+ 1` so chunk 0 never collapses to the bare user seed.
            let mut rng = StdRng::seed_from_u64(mix_seed(seed, ci as u64 + 1));
            let dist = dist.clone();
            (start..end).map(move |_| dist.sample(&mut rng))
        })
        .collect()
}

/// Like [`collect_f32_seeded`] but cast to any numeric `T` (the shared
/// `Vec<f32> → Vec<T>` step the `nalgebra` / `ndarray` `SampleOps` samplers
/// otherwise each open-code).
pub fn collect_seeded<T, D>(n: usize, dist: D, seed: u64) -> Vec<T>
where
    T: NumCast + Send,
    D: Distribution<f32> + Clone + Send + Sync,
{
    collect_f32_seeded(n, dist, seed)
        .into_iter()
        .map(|x| T::from(x).expect("sampled f32 not representable in target type"))
        .collect()
}

/// A fresh entropy-derived seed, for the legacy unseeded samplers that keep
/// their "random every run" contract (simulations, property tests). Routing
/// them through [`collect_f32_seeded`] with this seed still removes the
/// thread-order dependence.
#[inline]
pub fn entropy_seed() -> u64 {
    rand::rng().random()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::StandardNormal;

    #[test]
    fn same_seed_same_output() {
        let a = collect_f32_seeded(100_000, StandardNormal, 42);
        let b = collect_f32_seeded(100_000, StandardNormal, 42);
        assert_eq!(a, b, "same seed must reproduce byte-identical output");
    }

    #[test]
    fn different_seed_different_output() {
        let a = collect_f32_seeded(10_000, StandardNormal, 1);
        let b = collect_f32_seeded(10_000, StandardNormal, 2);
        assert_ne!(a, b, "distinct seeds must diverge");
    }

    #[test]
    fn independent_of_thread_count() {
        // A 1-thread pool and the global pool must agree: the chunk decomposition
        // is scheduling-independent by construction.
        let reference = collect_f32_seeded(200_000, StandardNormal, 7);
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| collect_f32_seeded(200_000, StandardNormal, 7));
        assert_eq!(reference, single);
    }

    #[test]
    fn mix_seed_avalanche() {
        // Adjacent salts must not produce adjacent (correlated) seeds.
        assert_ne!(mix_seed(42, 1), mix_seed(42, 2));
        assert_ne!(mix_seed(0, 0), mix_seed(0, 1));
    }
}
