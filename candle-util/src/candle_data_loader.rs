#![allow(dead_code)]

use asap_data::sparse_io_vector::*;
use matrix_util::traits::MatTriplets;

use candle_core::{Device, Tensor};
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

/// `DataLoader` for minibatch learning
pub trait DataLoader {
    fn minibatch(&self, batch_idx: usize, target_device: &Device) -> anyhow::Result<Tensor>;
    fn minibatch_with_aux(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<(Tensor, Option<Tensor>)>;
    fn num_minibatch(&self) -> usize;
    fn shuffle_minibatch(&mut self, batch_size: usize);
}

///
/// A thin wrapper for `SparseIoVec`. Columns in `SparseIoVec` are
/// treated as samples.
///
pub struct SparseIoVecData<'a> {
    data: &'a SparseIoVec,
    minibatches: Minibatches,
}

impl<'a> SparseIoVecData<'a> {
    pub fn new(data: &'a SparseIoVec) -> anyhow::Result<Self> {
        let samples = (0..data.num_columns()?).collect();
        Ok(SparseIoVecData {
            data,
            minibatches: Minibatches {
                samples,
                chunks: vec![],
            },
        })
    }
}

impl DataLoader for SparseIoVecData<'_> {
    fn minibatch(&self, batch_idx: usize, target_device: &Device) -> anyhow::Result<Tensor> {
        if let Some(cols) = self.minibatches.chunks.get(batch_idx) {
            let (dd, nn, triplets) = self.data.collect_columns_triplets(cols.iter().cloned())?;
            let ret = Tensor::from_nonzero_triplets(dd, nn, triplets)?.transpose(0, 1)?;
            Ok(ret.to_device(target_device)?)
        } else {
            Err(anyhow::anyhow!(
                "invalid index = {} vs. total # = {}",
                batch_idx,
                self.num_minibatch()
            ))
        }
    }

    fn minibatch_with_aux(
        &self,
        _batch_idx: usize,
        _target_device: &Device,
    ) -> anyhow::Result<(Tensor, Option<Tensor>)> {
        unimplemented!("SparseIoVecData")
    }

    fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }

    fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
    }
}

///
/// A simple data loader for in-memory 2d matrix.  Each row will be
/// considered as a feature vector. The number of samples is the
/// number of rows.
///
pub struct InMemoryData {
    data: Vec<Tensor>,
    aux_data: Option<Vec<Tensor>>,
    minibatches: Minibatches,
}

impl InMemoryData {
    pub fn from<D>(data: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        Ok(InMemoryData {
            data,
            aux_data: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    pub fn from_with_aux<D>(data: &D, aux: &D) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let data = data.rows_to_tensor_vec();
        let aux_data = aux.rows_to_tensor_vec();
        let rows = (0..data.len()).collect();

        debug_assert!(data.len() == aux_data.len());

        Ok(InMemoryData {
            data,
            aux_data: Some(aux_data),
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }
}

impl DataLoader for InMemoryData {
    fn minibatch(&self, batch_idx: usize, target_device: &Device) -> anyhow::Result<Tensor> {
        if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
            let chunk: Vec<Tensor> = samples.into_iter().map(|&i| self.data[i].clone()).collect();
            Ok(Tensor::cat(&chunk, 0)?.to_device(target_device)?)
        } else {
            Err(anyhow::anyhow!(
                "invalid index = {} vs. total # = {}",
                batch_idx,
                self.num_minibatch()
            ))
        }
    }

    fn minibatch_with_aux(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<(Tensor, Option<Tensor>)> {
        if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
            let chunk: Vec<Tensor> = samples.into_iter().map(|&i| self.data[i].clone()).collect();
            let data = Tensor::cat(&chunk, 0)?;

            if let Some(ref aux_data) = self.aux_data {
                let chunk: Vec<Tensor> =
                    samples.into_iter().map(|&i| aux_data[i].clone()).collect();
                let aux_data = Tensor::cat(&chunk, 0)?;
                Ok((
                    data.to_device(target_device)?,
                    Some(aux_data.to_device(target_device)?),
                ))
            } else {
                Ok((data.to_device(target_device)?, None))
            }
        } else {
            Err(anyhow::anyhow!(
                "invalid index = {} vs. total # = {}",
                batch_idx,
                self.num_minibatch()
            ))
        }
    }

    fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }

    fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
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
        use rand::distributions::{Distribution, Uniform};

        let mut rng = rand::thread_rng();
        self.samples.shuffle(&mut rng);
        let nbatch = (self.size() + batch_size) / batch_size;
        let ntot = nbatch * batch_size;

        let unif = Uniform::new(0, self.size());

        let indexes = (0..ntot)
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, _| unif.sample(rng))
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
