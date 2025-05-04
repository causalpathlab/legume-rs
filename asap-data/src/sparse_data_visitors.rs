#![allow(dead_code)]
use crate::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const DEFAULT_BLOCK_SIZE: usize = 100;

pub trait VisitColumnsOps {
    fn visit_columns_by_jobs<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: Option<usize>,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn((usize, usize), &Self, &SharedIn, Arc<Mutex<&mut SharedOut>>) + Sync + Send,
        SharedIn: Sync + Send,
        SharedOut: Sync + Send;

    fn visit_column_by_samples<Visitor, SharedIn, SharedOut>(
        &self,
        sample_to_cells: &Vec<Vec<usize>>,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(usize, &Vec<usize>, &SharedIn, Arc<Mutex<&mut SharedOut>>) + Sync + Send,
        SharedIn: Sync + Send,
        SharedOut: Sync + Send;
}

impl VisitColumnsOps for SparseIoVec {
    fn visit_columns_by_jobs<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: Option<usize>,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn((usize, usize), &Self, &SharedIn, Arc<Mutex<&mut SharedOut>>) + Sync + Send,
        SharedIn: Sync + Send,
        SharedOut: Sync + Send,
    {
        let block_size = block_size.unwrap_or(DEFAULT_BLOCK_SIZE);
        let ntot = self.num_columns()?;
        let jobs = create_jobs(ntot, block_size);

        let arc_shared_out = Arc::new(Mutex::new(shared_out));

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .for_each(|&(lb, ub)| {
                visitor((lb, ub), &self, &shared_in, arc_shared_out.clone());
            });

        Ok(())
    }

    fn visit_column_by_samples<Visitor, SharedIn, SharedOut>(
        &self,
        sample_to_cells: &Vec<Vec<usize>>,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_data: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(usize, &Vec<usize>, &SharedIn, Arc<Mutex<&mut SharedOut>>) + Sync + Send,
        SharedIn: Sync + Send,
        SharedOut: Sync + Send,
    {
        let arc_shared_data = Arc::new(Mutex::new(shared_data));
        let num_samples = sample_to_cells.len();
        let num_jobs = num_samples as u64;

        sample_to_cells
            .iter()
            .enumerate()
            .par_bridge()
            .progress_count(num_jobs)
            .for_each(|(sample, cells)| {
                visitor(sample, &cells, shared_in, arc_shared_data.clone());
            });

        Ok(())
    }
}

fn create_jobs(ntot: usize, block_size: usize) -> Vec<(usize, usize)> {
    let nblock = (ntot + block_size - 1) / block_size;

    (0..nblock)
        .map(|block| {
            let lb: usize = block * block_size;
            let ub: usize = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}
