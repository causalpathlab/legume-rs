#![allow(dead_code)]

use candle_core::{Device, Tensor};
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

pub fn generate_minibatch_intervals(ntot: usize, batch_size: usize) -> Vec<(usize, usize)> {
    let num_batches = ntot.div_ceil(batch_size);
    (0..num_batches)
        .map(|b| {
            let lb: usize = b * batch_size;
            let ub: usize = ((b + 1) * batch_size).min(ntot);
            (lb, ub)
        })
        .collect::<Vec<_>>()
}

pub struct MinibatchData {
    pub input: Tensor,
    pub input_null: Option<Tensor>,
    pub input_matched: Option<Tensor>,
    pub output: Option<Tensor>,
    pub output_matched: Option<Tensor>,
}

/// `DataLoader` for minibatch learning
pub trait DataLoader {
    fn minibatch_shuffled(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<MinibatchData>;

    fn minibatch_ordered(
        &self,
        lb: usize,
        ub: usize,
        target_device: &Device,
    ) -> anyhow::Result<MinibatchData>;

    fn num_data(&self) -> usize;

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
    input_null_data: Option<Vec<Tensor>>,
    input_matched_data: Option<Vec<Tensor>>,
    output_data: Option<Vec<Tensor>>,
    output_matched_data: Option<Vec<Tensor>>,

    shuffled_input_data: Option<Vec<Tensor>>,
    shuffled_input_null_data: Option<Vec<Tensor>>,
    shuffled_input_matched_data: Option<Vec<Tensor>>,
    shuffled_output_data: Option<Vec<Tensor>>,
    shuffled_output_matched_data: Option<Vec<Tensor>>,

    minibatches: Minibatches,
}

impl InMemoryData {
    ///
    /// Create a data loader with the main data tensor `data`
    ///
    pub fn from<D>(data: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_data = data.rows_to_tensor_vec();
        let rows = (0..input_data.len()).collect();

        Ok(InMemoryData {
            input_data,
            input_null_data: None,
            input_matched_data: None,
            output_data: None,
            output_matched_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_input_matched_data: None,
            shuffled_output_data: None,
            shuffled_output_matched_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    /// Create a data loader with the main `data` and auxiliary
    /// data `aux`
    ///
    pub fn new_with_null_input<D>(data: &D, null_data: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_data = data.rows_to_tensor_vec();
        let input_null_data = null_data.rows_to_tensor_vec();
        let rows = (0..input_data.len()).collect();

        debug_assert!(input_data.len() == input_null_data.len());

        Ok(InMemoryData {
            input_data,
            input_null_data: Some(input_null_data),
            input_matched_data: None,
            output_data: None,
            output_matched_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_input_matched_data: None,
            shuffled_output_data: None,
            shuffled_output_matched_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    ///
    /// Create a data loader with the main `data` and output `out`
    ///
    pub fn new_with_input_and_output<D>(data: &D, out: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_data = data.rows_to_tensor_vec();
        let output_data = out.rows_to_tensor_vec();
        let rows = (0..input_data.len()).collect();

        debug_assert!(input_data.len() == output_data.len());

        Ok(InMemoryData {
            input_data,
            input_null_data: None,
            input_matched_data: None,
            output_data: Some(output_data),
            output_matched_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_input_matched_data: None,
            shuffled_output_data: None,
            shuffled_output_matched_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    ///
    /// Create a data loader with the main `data` and output `out`
    ///
    pub fn new_with_null_input_and_output<D>(
        input: &D,
        null_input: &D,
        output: &D,
    ) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_data = input.rows_to_tensor_vec();
        let input_null_data = null_input.rows_to_tensor_vec();
        let output_data = output.rows_to_tensor_vec();
        let rows = (0..input_data.len()).collect();

        debug_assert!(input_data.len() == output_data.len());

        Ok(InMemoryData {
            input_data,
            input_null_data: Some(input_null_data),
            input_matched_data: None,
            output_data: Some(output_data),
            output_matched_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_input_matched_data: None,
            shuffled_output_data: None,
            shuffled_output_matched_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    ///
    /// Create a data loader with the main `data` and output `out`
    ///
    pub fn new_with_matched_input_and_output<D>(
        input: &D,
        input_matched: &D,
        output: &D,
    ) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_data = input.rows_to_tensor_vec();
        let input_matched_data = input_matched.rows_to_tensor_vec();
        let output_data = output.rows_to_tensor_vec();
        let rows = (0..input_data.len()).collect();

        debug_assert!(input_data.len() == output_data.len());

        Ok(InMemoryData {
            input_data,
            input_null_data: None,
            input_matched_data: Some(input_matched_data),
            output_data: Some(output_data),
            output_matched_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_input_matched_data: None,
            shuffled_output_data: None,
            shuffled_output_matched_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    ///
    /// Create a data loader with the main `data` and output `out`
    ///
    pub fn new_with_matched_input_and_matched_output<D>(
        input: &D,
        input_matched: &D,
        out: &D,
        out_matched: &D,
    ) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_data = input.rows_to_tensor_vec();
        let input_matched_data = input_matched.rows_to_tensor_vec();
        let output_data = out.rows_to_tensor_vec();
        let output_matched_data = out_matched.rows_to_tensor_vec();
        let rows = (0..input_data.len()).collect();

        debug_assert!(input_data.len() == output_data.len());

        Ok(InMemoryData {
            input_data,
            input_null_data: None,
            input_matched_data: Some(input_matched_data),
            output_data: Some(output_data),
            output_matched_data: Some(output_matched_data),
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_input_matched_data: None,
            shuffled_output_data: None,
            shuffled_output_matched_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }
}

impl DataLoader for InMemoryData {
    fn minibatch_ordered(
        &self,
        lb: usize,
        ub: usize,
        target_device: &Device,
    ) -> anyhow::Result<MinibatchData> {
        if let Some(input) = take_lb_ub(lb, ub, target_device, Some(&self.input_data))? {
            let output = take_lb_ub(lb, ub, target_device, self.output_data.as_ref())?;
            let output_matched =
                take_lb_ub(lb, ub, target_device, self.output_matched_data.as_ref())?;
            let input_matched =
                take_lb_ub(lb, ub, target_device, self.input_matched_data.as_ref())?;
            let input_null = take_lb_ub(lb, ub, target_device, self.input_null_data.as_ref())?;
            Ok(MinibatchData {
                input,
                input_null,
                input_matched,
                output,
                output_matched,
            })
        } else {
            Err(anyhow::anyhow!("no input data"))
        }
    }

    fn minibatch_shuffled(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<MinibatchData> {
        if let Some(input) =
            take_shuffled(batch_idx, target_device, self.shuffled_input_data.as_ref())?
        {
            let output =
                take_shuffled(batch_idx, target_device, self.shuffled_output_data.as_ref())?;

            let output_matched = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_output_matched_data.as_ref(),
            )?;

            let input_null = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_null_data.as_ref(),
            )?;

            let input_matched = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_matched_data.as_ref(),
            )?;

            Ok(MinibatchData {
                input,
                input_null,
                input_matched,
                output,
                output_matched,
            })
        } else {
            Err(anyhow::anyhow!("need to shuffle data"))
        }
    }

    fn num_data(&self) -> usize {
        self.minibatches.samples.len()
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

        if self.input_null_data.is_some() {
            self.shuffled_input_null_data = Some(vec![]);
        }

        if self.input_matched_data.is_some() {
            self.shuffled_input_matched_data = Some(vec![]);
        }

        if self.output_data.is_some() {
            self.shuffled_output_data = Some(vec![]);
        }

        if self.output_matched_data.is_some() {
            self.shuffled_output_matched_data = Some(vec![]);
        }

        ///////////////////////////////////
        // preload all the shuffled data //
        ///////////////////////////////////

        for batch_idx in 0..self.num_minibatch() {
            if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
                copy_shuffled(
                    &samples,
                    Some(&self.input_data),
                    self.shuffled_input_data.as_mut(),
                )?;
                copy_shuffled(
                    &samples,
                    self.input_matched_data.as_ref(),
                    self.shuffled_input_matched_data.as_mut(),
                )?;

                copy_shuffled(
                    &samples,
                    self.output_data.as_ref(),
                    self.shuffled_output_data.as_mut(),
                )?;
                copy_shuffled(
                    &samples,
                    self.output_matched_data.as_ref(),
                    self.shuffled_output_matched_data.as_mut(),
                )?;
                copy_shuffled(
                    &samples,
                    self.input_null_data.as_ref(),
                    self.shuffled_input_null_data.as_mut(),
                )?;
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

fn take_lb_ub(
    lb: usize,
    ub: usize,
    target_device: &Device,
    data_vec: Option<&Vec<Tensor>>,
) -> anyhow::Result<Option<Tensor>> {
    if let Some(data_vec) = data_vec {
        if lb > ub || ub > data_vec.len() {
            return Err(anyhow::anyhow!(
                "check lb {}, ub {} vs. ntot {}",
                lb,
                ub,
                data_vec.len()
            ));
        }
        if lb == ub {
            return Ok(None);
        }

        let chunk = Tensor::cat(
            &(lb..ub).map(|i| data_vec[i].clone()).collect::<Vec<_>>(),
            0,
        )?;
        Ok(Some(chunk.to_device(target_device)?))
    } else {
        Ok(None)
    }
}

fn copy_shuffled(
    samples: &[usize],
    data: Option<&Vec<Tensor>>,
    shuffled_data: Option<&mut Vec<Tensor>>,
) -> anyhow::Result<()> {
    if let (Some(data), Some(shuffled)) = (data, shuffled_data) {
        let chunk: Vec<Tensor> = samples.iter().map(|&i| data[i].clone()).collect();
        let x = Tensor::cat(&chunk, 0)?;
        shuffled.push(x);
    }
    Ok(())
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

// use asap_data::sparse_io_vector::*;
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
