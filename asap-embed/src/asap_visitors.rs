#![allow(dead_code)]
use crate::asap_embed_common::*;
use asap_data::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub trait VisitColumnsOps {
    fn visit_csc_column_jobs<Visitor, SharedIn, SharedOut>(
        &self,
        jobs: Vec<(usize, usize)>,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn((usize, usize), &CscMat, &SharedIn, &mut SharedOut) + Sync + Send,
        SharedIn: Sync + Send,
        SharedOut: Sync + Send;

    fn visit_column_samples<Visitor, SharedOut>(
        &self,
        sample_to_cells: &Vec<Vec<usize>>,
        visitor: &Visitor,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(usize, &Vec<usize>, Arc<Mutex<&mut SharedOut>>) + Sync + Send,
        SharedOut: Sync + Send;
}

impl VisitColumnsOps for SparseIoVec {
    fn visit_csc_column_jobs<Visitor, SharedIn, SharedOut>(
        &self,
        jobs: Vec<(usize, usize)>,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn((usize, usize), &CscMat, &SharedIn, &mut SharedOut) + Sync + Send,
        SharedIn: Sync + Send,
        SharedOut: Sync + Send,
    {
        let arc_shared_out = Arc::new(Mutex::new(shared_out));

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .for_each(|&(lb, ub)| {
                let xx_dm = self
                    .read_columns_csc(lb..ub)
                    .expect("failed to retrieve data");

                let mut shared_job_out = arc_shared_out
                    .lock()
                    .expect("failed to lock shared mutable data");
                visitor((lb, ub), &xx_dm, &shared_in, &mut shared_job_out);
            });

        Ok(())
    }

    fn visit_column_samples<Visitor, SharedOut>(
        &self,
        sample_to_cells: &Vec<Vec<usize>>,
        visitor: &Visitor,
        shared_data: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(usize, &Vec<usize>, Arc<Mutex<&mut SharedOut>>) + Sync + Send,
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
                visitor(sample, &cells, arc_shared_data.clone());
            });

        Ok(())
    }
}
