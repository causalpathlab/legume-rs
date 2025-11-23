pub use crate::data::gff::*;
pub use crate::data::sam::*;
pub use clap::{Args, Parser, Subcommand};
pub use data_beans::qc::*;
pub use data_beans::sparse_io::*;
pub use env_logger;
pub use indicatif::ParallelProgressIterator;
pub use log::info;
pub use matrix_util::common_io::*;
pub use rayon::prelude::*;

use fnv::{FnvHashMap, FnvHashSet};

pub struct TripletsRowsCols {
    pub triplets: Vec<(u64, u64, f32)>,
    pub rows: Vec<Box<str>>,
    pub cols: Vec<Box<str>>,
}

pub fn format_data_triplets<Feat, Val>(stats: Vec<(CellBarcode, Feat, Val)>) -> TripletsRowsCols
where
    Feat: std::hash::Hash + std::cmp::Eq + std::cmp::Ord + Clone + Send + ToString,
    Val: Into<f32>,
{
    // identify unique samples and sites
    let mut unique_cells = FnvHashSet::default();
    let mut unique_features = FnvHashSet::default();

    for (cb, f, _) in stats.iter() {
        unique_features.insert(f.clone());
        unique_cells.insert(cb.clone());
    }

    let mut unique_cells = unique_cells.into_iter().collect::<Vec<_>>();
    unique_cells.par_sort();

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
        .map(|(i, f)| (f, i))
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

    TripletsRowsCols {
        triplets: relabeled_triplets,
        rows: features,
        cols: cells,
    }
}

// pub trait ToBed {
//     fn to_bed(&self, file_path: &str) -> anyhow::Result<()>;
// }

// impl ToBed for TripletsRowsCols {
//     fn to_bed(&self, file_path: &str) -> anyhow::Result<()> {
// 	unimplemented!("");
//     }
// }

pub trait ToBackend {
    fn to_backend(
        &self,
        file_path: &str,
    ) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>>;
}

impl ToBackend for TripletsRowsCols {
    fn to_backend(
        &self,
        file_path: &str,
    ) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
        let backend = match file_ext(file_path)?.as_ref() {
            "zarr" => SparseIoBackend::Zarr,
            "h5" => SparseIoBackend::HDF5,
            _ => return Err(anyhow::anyhow!("unknown backend type")),
        };

        let row_names = &self.rows;
        let col_names = &self.cols;
        let triplets = &self.triplets;

        let mtx_shape = (row_names.len(), col_names.len(), triplets.len());

        remove_file(file_path)?;

        let mut data =
            create_sparse_from_triplets(triplets, mtx_shape, Some(file_path), Some(&backend))?;
        data.register_column_names_vec(&col_names);
        data.register_row_names_vec(&row_names);

        Ok(data)
    }
}

pub trait BackendQc {
    fn qc(&self, cutoffs: SqueezeCutoffs) -> anyhow::Result<()>;
}

impl BackendQc for Box<dyn SparseIo<IndexIter = Vec<usize>>> {
    fn qc(&self, cutoffs: SqueezeCutoffs) -> anyhow::Result<()> {
        info!("final Q/C to remove excessive zeros");
        let block_size = 100;
        squeeze_by_nnz(self.as_ref(), cutoffs, block_size)
    }
}
