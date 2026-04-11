use ndarray::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use super::DMatrix;

pub fn build_name2index_map(_names: &[Box<str>]) -> HashMap<Box<str>, usize> {
    _names
        .iter()
        .enumerate()
        .map(|(r, name)| (name.clone(), r))
        .collect()
}

pub fn take_subset_indices_names(
    new_indices: &[usize],
    ntot: usize,
    old_names: Vec<Box<str>>,
) -> (HashMap<u64, u64>, Vec<Box<str>>) {
    let mut old2new: HashMap<u64, u64> = Default::default();
    let mut new2old = vec![];
    debug_assert!(ntot == old_names.len());
    let mut k = 0_u64;
    for idx in new_indices.iter() {
        if *idx < ntot {
            old2new.insert(*idx as u64, k);
            new2old.push(*idx);
            k += 1;
        }
    }

    let new_names = new2old
        .iter()
        .map(|&i| old_names[i].clone())
        .collect::<Vec<Box<str>>>();

    (old2new, new_names)
}

pub fn take_subset_indices_names_if_needed(
    new_indices: Option<&Vec<usize>>,
    ntot: Option<usize>,
    old_names: Vec<Box<str>>,
) -> (HashMap<u64, u64>, Vec<Box<str>>) {
    let ntot = ntot.unwrap_or(old_names.len());
    if let Some(new_indices) = new_indices {
        take_subset_indices_names(new_indices, ntot, old_names)
    } else {
        let names = old_names;
        let identity = (0..(ntot as u64))
            .zip(0..(ntot as u64))
            .collect::<HashMap<u64, u64>>();
        (identity, names)
    }
}

pub fn ndarray_to_triplets(array: &Array2<f32>) -> Vec<(u64, u64, f32)> {
    let eps = 1e-6;
    array
        .indexed_iter()
        .filter(|(_, &elem)| elem.abs() > eps)
        .map(|((row, col), &value)| (row as u64, col as u64, value))
        .collect::<Vec<(u64, u64, f32)>>()
}

pub fn dmatrix_to_triplets(matrix: &DMatrix<f32>) -> Vec<(u64, u64, f32)> {
    let (nrow, _) = matrix.shape();
    let eps = 1e-6;
    matrix
        .iter() // column-major
        .enumerate()
        .filter(|(_, &elem)| elem.abs() > eps)
        .map(|(idx, &value)| {
            let row = idx % nrow;
            let col = idx / nrow;
            (row as u64, col as u64, value)
        })
        .collect::<Vec<(u64, u64, f32)>>()
}
