#![allow(dead_code)]

// use asap_data::sparse_io_vector::*;

use candle_core::{Device, Tensor};
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

pub struct MinibatchData {
    pub input: Tensor,
    pub input_null: Option<Tensor>,
    pub coordinate: Option<Tensor>,
    pub output: Option<Tensor>,
}

/// `DataLoader` for minibatch learning
pub trait DataLoader {
    fn minibatch_data(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<MinibatchData>;

    fn num_minibatch(&self) -> usize;

    fn shuffle_minibatch(&mut self, batch_size: usize) -> anyhow::Result<()>;
}

///
/// A simple data loader for in-memory 2d matrix.  Each row will be
/// considered as a feature vector. The number of samples is the
/// number of rows.
///
pub struct InMemoryData {
    input_data: Vec<Tensor>,
    null_data: Option<Vec<Tensor>>,
    coordinate_data: Option<Vec<Tensor>>,
    output_data: Option<Vec<Tensor>>,

    shuffled_input_data: Option<Vec<Tensor>>,
    shuffled_input_null_data: Option<Vec<Tensor>>,
    shuffled_coordinate_data: Option<Vec<Tensor>>,
    shuffled_output_data: Option<Vec<Tensor>>,

    minibatches: Minibatches,
}

impl InMemoryData {
    ///
    /// Create a data loader with the main data tensor `data`
    ///
    pub fn new<D>(data: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        Ok(InMemoryData {
            input_data: data,
            null_data: None,
            coordinate_data: None,
            output_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_coordinate_data: None,
            shuffled_output_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    /// Create a data loader with the main `data` and auxiliary
    /// data `aux`
    ///
    pub fn new_with_null<D>(data: &D, null_data: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let null_data = null_data.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        debug_assert!(data.len() == null_data.len());

        Ok(InMemoryData {
            input_data: data,
            null_data: Some(null_data),
            coordinate_data: None,
            output_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_coordinate_data: None,
            shuffled_output_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    ///
    /// Create a data loader with the main `data` and output `out`
    ///
    pub fn new_with_output<D>(data: &D, out: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let out_data = out.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        debug_assert!(data.len() == out_data.len());

        Ok(InMemoryData {
            input_data: data,
            null_data: None,
            coordinate_data: None,
            output_data: Some(out_data),
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_coordinate_data: None,
            shuffled_output_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    ///
    /// Create a data loader with the main `data` and output `out`
    ///
    pub fn new_with_null_output<D>(data: &D, null_data: &D, out: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let aux_data = null_data.rows_to_tensor_vec();
        let out_data = out.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        debug_assert!(data.len() == out_data.len());

        Ok(InMemoryData {
            input_data: data,
            null_data: Some(aux_data),
            coordinate_data: None,
            output_data: Some(out_data),
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_coordinate_data: None,
            shuffled_output_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    ///
    /// Create a data loader with the main `data` and output `out`
    ///
    pub fn new_with_coord_output<D>(data: &D, coord_data: &D, out: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let coord_data = coord_data.rows_to_tensor_vec();
        let out_data = out.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        debug_assert!(data.len() == out_data.len());

        Ok(InMemoryData {
            input_data: data,
            null_data: None,
            coordinate_data: Some(coord_data),
            output_data: Some(out_data),
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_coordinate_data: None,
            shuffled_output_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    /// Create a data loader with the main `data` (both observed and
    /// null) and output `out`
    ///
    pub fn new_with_coord_null_output<D>(
        data: &D,
        coord_data: &D,
        null_data: &D,
        out: &D,
    ) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let null_data = null_data.rows_to_tensor_vec();
        let coord_data = coord_data.rows_to_tensor_vec();
        let out_data = out.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        debug_assert!(data.len() == out_data.len());

        Ok(InMemoryData {
            input_data: data,
            null_data: Some(null_data),
            coordinate_data: Some(coord_data),
            output_data: Some(out_data),
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_coordinate_data: None,
            shuffled_output_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }
}

impl DataLoader for InMemoryData {
    fn minibatch_data(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<MinibatchData> {
        if let Some(input) =
            take_shuffled(batch_idx, target_device, self.shuffled_input_data.as_ref())?
        {
            let output =
                take_shuffled(batch_idx, target_device, self.shuffled_output_data.as_ref())?;
            let input_null = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_null_data.as_ref(),
            )?;
            let coordinate = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_coordinate_data.as_ref(),
            )?;

            Ok(MinibatchData {
                input,
                input_null,
                coordinate,
                output,
            })
        } else {
            Err(anyhow::anyhow!("need to shuffle data"))
        }
    }

    fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }

    fn shuffle_minibatch(&mut self, batch_size: usize) -> anyhow::Result<()> {
        /////////////////////
        // shuffle indexes //
        /////////////////////

        self.minibatches.shuffle_minibatch(batch_size);

        self.shuffled_input_data = Some(vec![]);

        if self.null_data.is_some() {
            self.shuffled_input_null_data = Some(vec![]);
        }

        if self.output_data.is_some() {
            self.shuffled_output_data = Some(vec![]);
        }

        ///////////////////////////////////
        // preload all the shuffled data //
        ///////////////////////////////////

        for batch_idx in 0..self.num_minibatch() {
            if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
                {
                    let chunk: Vec<Tensor> = samples
                        .into_iter()
                        .map(|&i| self.input_data[i].clone())
                        .collect();

                    if let Some(shuffled_data) = &mut self.shuffled_input_data {
                        let x = Tensor::cat(&chunk, 0)?;
                        shuffled_data.push(x);
                    }
                }
                if let Some(out_data) = self.output_data.as_ref() {
                    let chunk: Vec<Tensor> =
                        samples.into_iter().map(|&i| out_data[i].clone()).collect();

                    if let Some(shuffled_data) = &mut self.shuffled_output_data {
                        let x = Tensor::cat(&chunk, 0)?;
                        shuffled_data.push(x);
                    }
                }

                if let Some(null_data) = self.null_data.as_ref() {
                    let chunk: Vec<Tensor> =
                        samples.into_iter().map(|&i| null_data[i].clone()).collect();

                    if let Some(shuffled_data) = &mut self.shuffled_input_null_data {
                        let x = Tensor::cat(&chunk, 0)?;
                        shuffled_data.push(x);
                    }
                }

                if let Some(coord_data) = self.coordinate_data.as_ref() {
                    let chunk: Vec<Tensor> = samples
                        .into_iter()
                        .map(|&i| coord_data[i].clone())
                        .collect();

                    if let Some(shuffled_data) = &mut self.shuffled_coordinate_data {
                        let x = Tensor::cat(&chunk, 0)?;
                        shuffled_data.push(x);
                    }
                }
            } else {
                return Err(anyhow::anyhow!(
                    "invalid index = {} vs. total # = {}",
                    batch_idx,
                    self.num_minibatch()
                ));
            }
        }

        Ok(())
    }
}

fn take_shuffled(
    batch_idx: usize,
    target_device: &Device,
    data_vec: Option<&Vec<Tensor>>,
) -> anyhow::Result<Option<Tensor>> {
    if let Some(data_vec) = data_vec {
        if data_vec.len() <= batch_idx {
            Err(anyhow::anyhow!(
                "invalid index = {} vs. total # = {}",
                batch_idx,
                data_vec.len()
            ))
        } else {
            Ok(Some(data_vec[batch_idx].to_device(target_device)?))
        }
    } else {
        // if the data vector doesn't exist
        Ok(None)
    }
}

///
/// A helper `struct` for shuffling and creating minibatch indexes;
/// after `shuffle_minibatch` is called, `chunks` partition indexes.
///
pub struct Minibatches {
    samples: Vec<usize>,
    pub chunks: Vec<Vec<usize>>,
}

impl Minibatches {
    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        use rand_distr::{Distribution, Uniform};

        let mut rng = rand::rng();
        self.samples.shuffle(&mut rng);
        let nbatch = (self.size() + batch_size) / batch_size;
        let ntot = nbatch * batch_size;

        let unif = Uniform::new(0, self.size()).expect("unif [0 .. size)");

        let indexes = (0..ntot)
            .into_par_iter()
            .map_init(rand::rng, |rng, _| unif.sample(rng))
            .collect::<Vec<usize>>();

        self.chunks = (0..nbatch)
            .par_bridge()
            .map(|b| {
                let lb = b * batch_size;
                let ub = (b + 1) * batch_size;
                (lb..ub).map(|i| indexes[i]).collect()
            })
            .collect::<Vec<Vec<usize>>>();
    }

    pub fn size(&self) -> usize {
        self.samples.len()
    }
}

///
/// Convert rows of a matrix to a vector of `Tensor`
///
pub trait RowsToTensorVec {
    fn rows_to_tensor_vec(&self) -> Vec<Tensor>;
}

impl RowsToTensorVec for Array2<f32> {
    fn rows_to_tensor_vec(&self) -> Vec<Tensor> {
        let mut idx_data = self
            .axis_iter(ndarray::Axis(0))
            .enumerate()
            .par_bridge()
            .map(|(i, row)| {
                let mut v = Tensor::from_iter(row.iter().map(|x| *x), &Device::Cpu)
                    .expect("failed to create tensor");
                v = v.reshape((1, row.len())).expect("failed to reshape");
                (i, v)
            })
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        idx_data.into_iter().map(|(_, t)| t).collect()
    }
}

impl RowsToTensorVec for DMatrix<f32> {
    fn rows_to_tensor_vec(&self) -> Vec<Tensor> {
        let mut idx_data = self
            .row_iter()
            .enumerate()
            .par_bridge()
            .map(|(i, row)| {
                let mut v = Tensor::from_iter(row.iter().map(|x| *x), &Device::Cpu)
                    .expect("failed to create tensor");
                v = v.reshape((1, row.len())).expect("failed to reshape");
                (i, v)
            })
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        idx_data.into_iter().map(|(_, t)| t).collect()
    }
}

impl RowsToTensorVec for Tensor {
    fn rows_to_tensor_vec(&self) -> Vec<Tensor> {
        let mut idx_data = (0..self.dims()[0])
            .map(|i| (i, self.narrow(0, i, 1).expect("").clone()))
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        idx_data.into_iter().map(|(_, t)| t).collect()
    }
}

// ///
// /// A thin wrapper for `SparseIoVec`. Columns in `SparseIoVec` are
// /// treated as samples.
// ///
// pub struct SparseIoVecData<'a> {
//     data: &'a SparseIoVec,
//     minibatches: Minibatches,
// }

// impl<'a> SparseIoVecData<'a> {
//     pub fn new(data: &'a SparseIoVec) -> anyhow::Result<Self> {
//         let samples = (0..data.num_columns()?).collect();
//         Ok(SparseIoVecData {
//             data,
//             minibatches: Minibatches {
//                 samples,
//                 chunks: vec![],
//             },
//         })
//     }
// }

// impl DataLoader for SparseIoVecData<'_> {
//     fn minibatch_data(&self, batch_idx: usize, target_device: &Device) -> anyhow::Result<Tensor> {
//         if let Some(cols) = self.minibatches.chunks.get(batch_idx) {
//             Ok(self
//                 .data
//                 .read_columns_tensor(cols.iter().cloned())?
//                 .transpose(0, 1)?
//                 .to_device(target_device)?)
//         } else {
//             Err(anyhow::anyhow!(
//                 "invalid index = {} vs. total # = {}",
//                 batch_idx,
//                 self.num_minibatch()
//             ))
//         }
//     }

//     fn minibatch_data_aux(
//         &self,
//         _batch_idx: usize,
//         _target_device: &Device,
//     ) -> anyhow::Result<(Tensor, Option<Tensor>)> {
//         unimplemented!("SparseIoVecData")
//     }

//     fn minibatch_data_output(
//         &self,
//         _batch_idx: usize,
//         _target_device: &Device,
//     ) -> anyhow::Result<(Tensor, Option<Tensor>)> {
//         unimplemented!("SparseIoVecData")
//     }

//     fn minibatch_data_aux_output(
//         &self,
//         _batch_idx: usize,
//         _target_device: &Device,
//     ) -> anyhow::Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
//         unimplemented!("SparseIoVecData")
//     }

//     fn num_minibatch(&self) -> usize {
//         self.minibatches.chunks.len()
//     }

//     fn shuffle_minibatch(&mut self, batch_size: usize) -> anyhow::Result<()> {
//         self.minibatches.shuffle_minibatch(batch_size);
//         Ok(())
//     }
// }
