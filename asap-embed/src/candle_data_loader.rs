#![allow(dead_code)]

use asap_data::sparse_io_vector::*;
use candle_core::{Device, Tensor};
use matrix_util::traits::MatTriplets;
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

pub trait DataLoader {
    fn minibatch(&self, batch_idx: usize, target_device: &Device) -> anyhow::Result<Tensor>;
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
    minibatches: Minibatches,
}

#[allow(dead_code)]
impl InMemoryData {
    pub fn from_tensor(data: &Tensor) -> anyhow::Result<Self> {
        let mut idx_data = (0..data.dims()[0])
            .map(|i| (i, data.narrow(0, i, 1).expect("").clone()))
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        let rows = (0..idx_data.len()).collect();
        let data = idx_data.into_iter().map(|(_, t)| t).collect();

        Ok(InMemoryData {
            data,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    pub fn from_ndarray(data: &Array2<f32>) -> anyhow::Result<Self> {
        let mut idx_data = data
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
        let rows = (0..idx_data.len()).collect();
        let data = idx_data.into_iter().map(|(_, t)| t).collect();

        Ok(InMemoryData {
            data,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    pub fn from_dmatrix(data: &DMatrix<f32>) -> anyhow::Result<Self> {
        let mut idx_data = data
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
        let rows = (0..idx_data.len()).collect();
        let data = idx_data.into_iter().map(|(_, t)| t).collect();
        Ok(InMemoryData {
            data,
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

    fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }

    fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
    }
}

pub struct Minibatches {
    samples: Vec<usize>,
    chunks: Vec<Vec<usize>>,
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
