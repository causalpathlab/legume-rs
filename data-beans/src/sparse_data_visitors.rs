#![allow(dead_code)]

use crate::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use matrix_util::utils::generate_minibatch_intervals;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

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

/// Shared bar template used by every par-over-column visitor so progress
/// shows the same `Label {bar:40} {pos}/{len} units ({eta})` format across
/// random projection, collapsing, and downstream column scans.
fn styled_progress_bar(total: u64, unit_label: &str) -> indicatif::ProgressBar {
    use indicatif::{ProgressBar, ProgressStyle};
    let tmpl = format!("{{bar:40}} {{pos}}/{{len}} {} ({{eta}})", unit_label);
    ProgressBar::new(total).with_style(
        ProgressStyle::with_template(&tmpl)
            .unwrap()
            .progress_chars("##-"),
    )
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
        let num_features = self.num_rows();
        let jobs = create_jobs(ntot, num_features, block_size);

        let arc_shared_out = Arc::new(Mutex::new(shared_out));
        let pb = styled_progress_bar(jobs.len() as u64, "blocks");

        let result = jobs
            .par_iter()
            .progress_with(pb.clone())
            .map(|&(lb, ub)| visitor((lb, ub), self, shared_in, arc_shared_out.clone()))
            .collect();
        pb.finish_and_clear();
        result
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
        let pb = styled_progress_bar(num_samples as u64, "groups");

        let result = group_to_cols
            .iter()
            .enumerate()
            .par_bridge()
            .progress_with(pb.clone())
            .map(|(sample, cells)| visitor(sample, cells, self, shared_in, arc_shared_out.clone()))
            .collect();
        pb.finish_and_clear();
        result
    }
}

/// Thin wrapper around [`generate_minibatch_intervals`] kept for in-crate
/// call sites that predate the matrix-util split.
pub fn create_jobs(
    ntot: usize,
    num_features: usize,
    block_size: Option<usize>,
) -> Vec<(usize, usize)> {
    generate_minibatch_intervals(ntot, num_features, block_size)
}
