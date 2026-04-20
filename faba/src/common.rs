pub use clap::{Args, Parser, Subcommand};
pub use data_beans::qc::*;
pub use data_beans::sparse_io::*;
pub use env_logger;
pub use genomic_data::gff::*;
pub use genomic_data::sam::*;
pub use indicatif::ParallelProgressIterator;
pub use log::info;
pub use matrix_util::common_io::*;
pub use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet};

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
    let mut unique_cells = FxHashSet::default();
    let mut unique_features = FxHashSet::default();

    for (cb, f, _) in stats.iter() {
        unique_features.insert(f.clone());
        unique_cells.insert(cb.clone());
    }

    let mut unique_cells = unique_cells.into_iter().collect::<Vec<_>>();
    unique_cells.par_sort();

    let cell_to_index: FxHashMap<CellBarcode, usize> = unique_cells
        .into_iter()
        .enumerate()
        .map(|(i, sample)| (sample, i))
        .collect();

    let mut unique_features = unique_features.into_iter().collect::<Vec<_>>();
    unique_features.par_sort();

    let feature_to_index: FxHashMap<Feat, usize> = unique_features
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

/// Format triplets using pre-specified shared row/col name-to-index mappings.
/// This ensures two matrices end up with identical dimensions.
pub fn format_data_triplets_shared<Feat, Val>(
    stats: Vec<(CellBarcode, Feat, Val)>,
    feature_to_index: &FxHashMap<Feat, usize>,
    cell_to_index: &FxHashMap<CellBarcode, usize>,
    rows: Vec<Box<str>>,
    cols: Vec<Box<str>>,
) -> TripletsRowsCols
where
    Feat: std::hash::Hash + std::cmp::Eq + Clone + ToString,
    Val: Into<f32>,
{
    let mut relabeled_triplets = Vec::with_capacity(stats.len());
    for (cb, k, value) in stats {
        let row_idx = feature_to_index[&k] as u64;
        let col_idx = cell_to_index[&cb] as u64;
        relabeled_triplets.push((row_idx, col_idx, value.into()));
    }

    TripletsRowsCols {
        triplets: relabeled_triplets,
        rows,
        cols,
    }
}

pub struct UnionNames<Feat> {
    pub col_names: Vec<Box<str>>,
    pub cell_to_index: FxHashMap<CellBarcode, usize>,
    pub row_names: Vec<Box<str>>,
    pub feature_to_index: FxHashMap<Feat, usize>,
}

/// Collect the union of cells and features from two sets of triplets,
/// returning sorted name vectors and index maps.
pub fn collect_union_names<Feat>(
    triplets_a: &[(CellBarcode, Feat, f32)],
    triplets_b: &[(CellBarcode, Feat, f32)],
) -> UnionNames<Feat>
where
    Feat: std::hash::Hash + std::cmp::Eq + std::cmp::Ord + Clone + Send + ToString,
{
    let mut unique_cells = FxHashSet::default();
    let mut unique_features = FxHashSet::default();

    for (cb, f, _) in triplets_a.iter().chain(triplets_b.iter()) {
        unique_cells.insert(cb.clone());
        unique_features.insert(f.clone());
    }

    let mut unique_cells = unique_cells.into_iter().collect::<Vec<_>>();
    unique_cells.par_sort();
    let cell_to_index: FxHashMap<CellBarcode, usize> = unique_cells
        .iter()
        .enumerate()
        .map(|(i, cb)| (cb.clone(), i))
        .collect();
    let col_names: Vec<Box<str>> = unique_cells
        .into_iter()
        .map(|cb| cb.to_string().into_boxed_str())
        .collect();

    let mut unique_features = unique_features.into_iter().collect::<Vec<_>>();
    unique_features.par_sort();
    let feature_to_index: FxHashMap<Feat, usize> = unique_features
        .iter()
        .enumerate()
        .map(|(i, f)| (f.clone(), i))
        .collect();
    let row_names: Vec<Box<str>> = unique_features
        .into_iter()
        .map(|f| f.to_string().into_boxed_str())
        .collect();

    UnionNames {
        col_names,
        cell_to_index,
        row_names,
        feature_to_index,
    }
}

/// Generate unique batch names from BAM file paths (using file stems).
pub fn uniq_batch_names(bam_files: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    let batch_names: Vec<Box<str>> = bam_files
        .iter()
        .map(|x| basename(x))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let unique_bams: FxHashSet<_> = bam_files.iter().cloned().collect();
    let unique_names: FxHashSet<_> = batch_names.iter().cloned().collect();

    if unique_names.len() == bam_files.len() && unique_bams.len() == bam_files.len() {
        Ok(batch_names)
    } else {
        info!("bam file (base) names are not unique, appending index");
        Ok(batch_names
            .iter()
            .enumerate()
            .map(|(i, name)| format!("{}_{}", name, i).into_boxed_str())
            .collect())
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
        data.register_column_names_vec(col_names);
        data.register_row_names_vec(row_names);

        Ok(data)
    }
}

pub trait BackendQc {
    fn qc(&self, cutoffs: SqueezeCutoffs) -> anyhow::Result<()>;
}

impl BackendQc for Box<dyn SparseIo<IndexIter = Vec<usize>>> {
    fn qc(&self, cutoffs: SqueezeCutoffs) -> anyhow::Result<()> {
        info!("final Q/C to remove excessive zeros");
        squeeze_by_nnz(self.as_ref(), cutoffs, None, false)
    }
}
