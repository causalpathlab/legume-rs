use asap_data::sparse_io::*;
use nalgebra::{DMatrix, DVector};
use rand::prelude::SliceRandom;
use std::sync::{Arc, Mutex};

pub const DEFAULT_BLOCK_SIZE: usize = 100;

#[allow(dead_code)]
pub type Mat = DMatrix<f32>;
#[allow(dead_code)]
pub type DVec = DVector<f32>;

#[allow(dead_code)]
pub type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

#[allow(dead_code)]
pub fn create_jobs(block_size: usize, ntot: usize) -> Vec<(usize, usize)> {
    let nblock = (ntot + block_size - 1) / block_size;

    (0..nblock)
        .map(|block| {
            let lb: usize = block * block_size;
            let ub: usize = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}

#[allow(dead_code)]
pub fn partition_to_membership(pb_cells: &HashMap<usize, Vec<usize>>) -> HashMap<usize, usize> {
    let mut sample_membership: HashMap<usize, usize> = HashMap::new();
    let arc = Arc::new(Mutex::new(&mut sample_membership));
    pb_cells.par_iter().for_each(|(&s, cells)| {
        if let Ok(mut sample_membership) = arc.lock() {
            for j in cells {
                sample_membership.insert(*j, s);
            }
        }
    });
    sample_membership
}

#[allow(dead_code)]
pub fn partition_by_membership(
    sample_membership: &Vec<usize>,
    cells_per_sample: Option<usize>,
) -> HashMap<usize, Vec<usize>> {
    // Take care of empty pseudobulk samples
    let mut pb_position: HashMap<usize, usize> = HashMap::new();
    {
        let mut pos = 0_usize;
        for k in sample_membership {
            if !pb_position.contains_key(k) {
                pb_position.insert(*k, pos);
                pos += 1;
            }
        }
    }
    // dbg!(&pb_position);

    let mut pb_cells: HashMap<usize, Vec<usize>> = HashMap::new();
    for (cell, &k) in sample_membership.iter().enumerate() {
        let &s = pb_position.get(&k).expect("failed to get position");
        pb_cells.entry(s).or_default().push(cell);
    }

    // Down sample cells if needed
    pb_cells.par_iter_mut().for_each(|(_, cells)| {
        let ncells = cells.len();
        if let Some(ntarget) = cells_per_sample {
            if ncells > ntarget {
                let mut rng = rand::thread_rng();
                cells.shuffle(&mut rng);
                cells.truncate(ntarget);
            }
        }
        // dbg!(cells.len());
    });
    pb_cells
}
