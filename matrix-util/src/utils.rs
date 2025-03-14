use rand::prelude::SliceRandom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

/// partition membership vector into groups of indexes
/// # Arguments
/// * `membership` - a vector of membership (E.g., cluster assignment)
/// * `nelem_per_group` - number of elements per group (if None, no downsampling)
/// # Returns
/// A hashmap: cluster/group name -> indexes of the elements
pub fn partition_by_membership<T>(
    membership: &Vec<T>,
    nelem_per_group: Option<usize>,
) -> HashMap<T, Vec<usize>>
where
    T: Eq + Hash + Clone + Send + Sync,
{
    let mut pb_elems: HashMap<T, Vec<usize>> = HashMap::new();
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
