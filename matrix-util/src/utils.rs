use rand::prelude::SliceRandom;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::hash::Hash;

/// partition membership vector into groups of indexes
/// # Arguments
/// * `membership` - a vector of membership (E.g., cluster assignment)
/// * `nelem_per_group` - number of elements per group (if None, no downsampling)
/// # Returns
/// A hashmap: cluster/group name -> indexes of the elements
pub fn partition_by_membership<T>(
    membership: &[T],
    nelem_per_group: Option<usize>,
) -> HashMap<T, Vec<usize>>
where
    T: Eq + Hash + Clone + Send + Sync,
{
    let mut pb_elems: HashMap<T, Vec<usize>> = HashMap::default();
    for (cell, k) in membership.iter().enumerate() {
        pb_elems.entry(k.clone()).or_default().push(cell);
    }

    // Down sample elements if needed
    pb_elems.par_iter_mut().for_each(|(_k, cells)| {
        let ncells = cells.len();
        if let Some(ntarget) = nelem_per_group {
            if ncells > ntarget {
                let mut rng = rand::rng();
                cells.shuffle(&mut rng);
                cells.truncate(ntarget);
            }
        }
        // dbg!(cells.len());
    });
    pb_elems
}

/// Pick a per-job cell block size given the feature count.
/// Target ≈ `MIN_BLOCK_SIZE × 10 000` cells×features of work per job;
/// clamped to `[MIN_BLOCK_SIZE, MAX_BLOCK_SIZE]`.
/// `None` passed to `generate_minibatch_intervals` / `create_jobs` resolves through this.
pub fn default_block_size(num_features: usize) -> usize {
    const MIN_BLOCK_SIZE: usize = 100;
    const MAX_BLOCK_SIZE: usize = 10_000;
    const TARGET_WORK: usize = MIN_BLOCK_SIZE * 10_000;
    if num_features == 0 {
        return MIN_BLOCK_SIZE;
    }
    (TARGET_WORK / num_features).clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
}

/// Generate minibatch intervals.
/// * `ntot` - number of total samples (columns/cells/edges/...)
/// * `num_features` - number of features per sample; used only when `batch_size` is `None`
///   to derive an adaptive default via [`default_block_size`]
/// * `batch_size` - `Some(n)` to use `n` explicitly, `None` to auto-scale by feature count
pub fn generate_minibatch_intervals(
    ntot: usize,
    num_features: usize,
    batch_size: Option<usize>,
) -> Vec<(usize, usize)> {
    let batch_size = batch_size.unwrap_or_else(|| default_block_size(num_features));
    let num_batches = ntot.div_ceil(batch_size);
    (0..num_batches)
        .map(|b| {
            let lb: usize = b * batch_size;
            let ub: usize = ((b + 1) * batch_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}
