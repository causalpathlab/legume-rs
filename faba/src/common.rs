#![allow(unused)]

pub use matrix_util::common_io as io;

pub use clap::{ArgAction, Args, Parser, Subcommand};
pub use env_logger;

pub use log::info;
pub use std::path::Path;
pub use std::sync::{Arc, Mutex};
pub use std::thread;

pub use indicatif::ParallelProgressIterator;
pub use rayon::prelude::*;

pub use crate::data::gff::*;
pub use crate::data::sam::*;

use fnv::{FnvHashMap, FnvHashSet};

pub fn format_data_triplets<Feat, Val>(
    stats: Vec<(CellBarcode, Feat, Val)>,
) -> (Vec<(u64, u64, f32)>, Vec<Box<str>>, Vec<Box<str>>)
where
    Feat: std::hash::Hash + std::cmp::Eq + std::cmp::Ord + Clone + Send + ToString,
    Val: Into<f32>,
{
    // identify unique samples and sites
    let mut unique_cells = FnvHashSet::default();
    let mut unique_features = FnvHashSet::default();

    for (cb, g, _) in stats.iter() {
        unique_features.insert(g.clone());
        unique_cells.insert(cb.clone());
    }

    let mut unique_cells = unique_cells.into_iter().collect::<Vec<_>>();
    unique_cells.sort();
    let cell_to_index: FnvHashMap<CellBarcode, usize> = unique_cells
        .into_iter()
        .enumerate()
        .map(|(i, sample)| (sample, i))
        .collect();

    let mut unique_features = unique_features.into_iter().collect::<Vec<_>>();
    unique_features.par_sort();

    let feature_to_index: FnvHashMap<Feat, usize> = unique_features
        .into_iter()
        .enumerate()
        .map(|(i, site)| (site, i))
        .collect();

    // relabel triplets with indices
    let mut relabeled_triplets = Vec::with_capacity(stats.len());
    for (cb, k, value) in stats {
        let row_idx = feature_to_index[&k] as u64;
        let col_idx = cell_to_index[&cb] as u64;
        relabeled_triplets.push((row_idx, col_idx, value.into()));
    }

    let mut cells = vec!["".into(); cell_to_index.len()];
    for (k, j) in cell_to_index {
        cells[j] = k.to_string().into_boxed_str();
    }

    let mut features = vec!["".into(); feature_to_index.len()];
    for (k, j) in feature_to_index {
        features[j] = k.to_string().into_boxed_str();
    }

    (relabeled_triplets, features, cells)
}
