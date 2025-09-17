use clap::ValueEnum;

pub struct ValuesIndicesPointers<'a> {
    pub values: &'a Vec<f32>,
    pub indices: &'a Vec<u64>,
    pub indptr: &'a Vec<u64>,
}

pub struct CooTripletsShape {
    pub triplets: Vec<(u64, u64, f32)>,
    pub shape: TripletsShape,
}

pub struct TripletsShape {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum IndexPointerType {
    Column,
    Row,
}

pub trait SparseTripletsTraits {
    /// convert sparse pointers into (row, column, value) triplets
    fn into_coo(&self, pointer_type: IndexPointerType) -> anyhow::Result<CooTripletsShape>;
}

/////////////////////
// implementations //
/////////////////////

impl<'a> SparseTripletsTraits for ValuesIndicesPointers<'a> {
    fn into_coo(&self, pointer_type: IndexPointerType) -> anyhow::Result<CooTripletsShape> {
        use indicatif::ParallelProgressIterator;
        use rayon::prelude::*;
        use std::sync::{Arc, Mutex};

        let indices = self.indices;
        let indptr = self.indptr;
        let values = self.values;

        let nvectors = indptr.len() - 1;

        let mut triplets = Vec::with_capacity(values.len());
        let arc_triplets = Arc::new(Mutex::new(&mut triplets));

        (0..nvectors)
            .into_par_iter()
            .progress_count(nvectors as u64)
            .for_each(|idx| {
                let j = idx as u64;
                let start = indptr[idx] as usize;
                let end = indptr[idx + 1] as usize;
                let values_slice = &values[start..end];
                let indices_slice = &indices[start..end];

                let triplets_slice: Vec<(u64, u64, f32)> = indices_slice
                    .iter()
                    .zip(values_slice.iter())
                    .map(|(&i, &x_ij)| match pointer_type {
                        IndexPointerType::Column => (i, j, x_ij),
                        _ => (j, i, x_ij),
                    })
                    .collect();

                arc_triplets
                    .lock()
                    .expect("failed to lock triplets")
                    .extend(triplets_slice);
            });

        let nnz = triplets.len();
        let nrows = triplets.iter().map(|&(i, _, _)| i).max().unwrap_or(0_u64) as usize + 1;
        let ncols = triplets.iter().map(|&(_, i, _)| i).max().unwrap_or(0_u64) as usize + 1;

        Ok(CooTripletsShape {
            triplets,
            shape: TripletsShape { nrows, ncols, nnz },
        })
    }

    // fn into_csc_coo(&self) -> anyhow::Result<CooTripletsShape> {
    //     use indicatif::ParallelProgressIterator;
    //     use rayon::prelude::*;
    //     use std::sync::{Arc, Mutex};

    //     let indices = self.indices;
    //     let indptr = self.indptr;
    //     let values = self.values;

    //     let nvectors = indptr.len() - 1;

    //     let mut triplets = Vec::with_capacity(values.len());
    //     let arc_triplets = Arc::new(Mutex::new(&mut triplets));

    //     (0..nvectors)
    //         .into_par_iter()
    //         .progress_count(nvectors as u64)
    //         .for_each(|_idx| {
    //             let j = _idx as u64;
    //             let start = indptr[_idx] as usize;
    //             let end = indptr[_idx + 1] as usize;
    //             let values_slice = &values[start..end];
    //             let indices_slice = &indices[start..end];

    //             // Note: h5ad treats cells as rows, but we treat cells as columns
    //             let triplets_slice: Vec<(u64, u64, f32)> = indices_slice
    //                 .iter()
    //                 .zip(values_slice.iter())
    //                 .map(|(&i, &x_ij)| (i, j, x_ij))
    //                 .collect();

    //             arc_triplets
    //                 .lock()
    //                 .expect("failed to lock triplets")
    //                 .extend(triplets_slice);
    //         });

    //     let nfeatures = triplets.iter().map(|&(i, _, _)| i).max().unwrap_or(0_u64) as usize + 1;
    //     let nnz = triplets.len();

    //     Ok(CooTripletsShape {
    //         triplets,
    //         shape: (nfeatures, nvectors, nnz),
    //     })
    // }
}
