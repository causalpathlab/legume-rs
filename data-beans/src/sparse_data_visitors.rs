#![allow(dead_code)]

use crate::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const DEFAULT_BLOCK_SIZE: usize = 100;

pub trait VisitColumnsOps {
    /// visit all the columns by sequential blocks.  The visitor
    /// function should take (a) `(lb, ub)` (b) `&Self` (c)
    /// `&SharedIn` (d) `Arc::new(Mutex::new(&mut SharedOut)`
    fn visit_columns_by_block<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: Option<usize>,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn((usize, usize), &Self, &SharedIn, Arc<Mutex<&mut SharedOut>>) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send;

    /// visit all the columns by predefined groups assigned by
    /// `self.assign_groups`. The visitor function should take (a)
    /// `group_index` (b) `&[columns_in_the_group]` (c) `&Self` (d)
    /// `&SharedIn` (e) `Arc::new(Mutex::new(&mut SharedOut)`
    fn visit_columns_by_group<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(usize, &[usize], &Self, &SharedIn, Arc<Mutex<&mut SharedOut>>) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send;
}

impl VisitColumnsOps for SparseIoVec {
    fn visit_columns_by_block<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: Option<usize>,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn((usize, usize), &Self, &SharedIn, Arc<Mutex<&mut SharedOut>>) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send,
    {
        let ntot = self.num_columns();
        let jobs = create_jobs(ntot, block_size);

        let arc_shared_out = Arc::new(Mutex::new(shared_out));

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .map(|&(lb, ub)| visitor((lb, ub), self, shared_in, arc_shared_out.clone()))
            .collect()
    }

    fn visit_columns_by_group<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(usize, &[usize], &Self, &SharedIn, Arc<Mutex<&mut SharedOut>>) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send,
    {
        let group_to_cols = self.take_grouped_columns().ok_or(anyhow::anyhow!(
            "The columns were not assigned before. Call `assign_groups`"
        ))?;

        let arc_shared_out = Arc::new(Mutex::new(shared_out));
        let num_samples = group_to_cols.len();
        let num_jobs = num_samples as u64;

        group_to_cols
            .iter()
            .enumerate()
            .par_bridge()
            .progress_count(num_jobs)
            .map(|(sample, cells)| visitor(sample, cells, self, shared_in, arc_shared_out.clone()))
            .collect()
    }
}

pub fn create_jobs(ntot: usize, block_size: Option<usize>) -> Vec<(usize, usize)> {
    let block_size = block_size.unwrap_or(DEFAULT_BLOCK_SIZE);
    let nblock = ntot.div_ceil(block_size);
    (0..nblock)
        .map(|block| {
            let lb: usize = block * block_size;
            let ub: usize = ((block + 1) * block_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}
