use candle_core::{Device, Error, Tensor};
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

pub trait DataLoader {
    fn shuffle_minibatch(&mut self, batch_size: usize);
    fn minibatch(&self, batch_idx: usize, target_device: &Device) -> candle_core::Result<Tensor>;
    fn num_minibatch(&self) -> usize;
    fn size(&self) -> usize;
}

impl DataLoader for InMemoryData {
    fn minibatch(&self, batch_idx: usize, target_device: &Device) -> candle_core::Result<Tensor> {
        if let Some(samples) = self.batches.get(batch_idx) {
            let chunk: Vec<Tensor> = samples.into_iter().map(|&i| self.data[i].clone()).collect();
            Tensor::cat(&chunk, 0)?.to_device(target_device)
        } else {
            Err(Error::Msg(format!(
                "invalid index = {} vs. total # = {}",
                batch_idx,
                self.num_minibatch()
            )))
        }
    }

    fn num_minibatch(&self) -> usize {
        self.batches.len()
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn shuffle_minibatch(&mut self, batch_size: usize) {
        use rand::distributions::{Distribution, Uniform};

        let mut rng = rand::thread_rng();
        self.rows.shuffle(&mut rng);
        let nbatch = (self.size() + batch_size) / batch_size;
        let ntot = nbatch * batch_size;

        let unif = Uniform::new(0, self.size());

        let indexes = (0..ntot)
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, _| unif.sample(rng))
            .collect::<Vec<usize>>();

        self.batches = (0..nbatch)
            .par_bridge()
            .map(|b| {
                let lb = b * batch_size;
                let ub = (b + 1) * batch_size;
                (lb..ub).map(|i| indexes[i]).collect()
            })
            .collect::<Vec<Vec<usize>>>();
    }
}



///
/// A simple data loader for in-memory 2d matrix
///
pub struct InMemoryData {
    data: Vec<Tensor>,
    rows: Vec<usize>,
    batches: Vec<Vec<usize>>,
}

#[allow(dead_code)]
impl InMemoryData {
    pub fn from_tensor(data: &Tensor) -> candle_core::Result<Self> {
        let mut idx_data = (0..data.dims()[0])
            .map(|i| (i, data.narrow(0, i, 1).expect("").clone()))
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        let rows = (0..idx_data.len()).collect();
        let data = idx_data.into_iter().map(|(_, t)| t).collect();

        Ok(InMemoryData {
            data,
            rows,
            batches: vec![],
        })
    }

    pub fn from_ndarray(data: &Array2<f32>) -> candle_core::Result<Self> {
        let mut idx_data = data
            .axis_iter(ndarray::Axis(0))
            .enumerate()
            .par_bridge()
            .map(|(i, row)| {
                (
                    i,
                    Tensor::from_iter(row.iter().map(|x| *x), &Device::Cpu).expect(""),
                )
            })
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        let rows = (0..idx_data.len()).collect();
        let data = idx_data.into_iter().map(|(_, t)| t).collect();

        Ok(InMemoryData {
            data,
            rows,
            batches: vec![],
        })
    }

    pub fn from_dmatrix(data: &DMatrix<f32>) -> candle_core::Result<Self> {
        let mut idx_data = data
            .row_iter()
            .enumerate()
            .par_bridge()
            .map(|(i, row)| {
                (
                    i,
                    Tensor::from_iter(row.iter().map(|x| *x), &Device::Cpu)
                        .expect("failed to construct tensor"),
                )
            })
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        let rows = (0..idx_data.len()).collect();
        let data = idx_data.into_iter().map(|(_, t)| t).collect();
        Ok(InMemoryData {
            data,
            rows,
            batches: vec![],
        })
    }
}
