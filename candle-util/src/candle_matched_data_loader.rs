#![allow(dead_code)]

use candle_core::{Device, Tensor};
use nalgebra::DMatrix;
use ndarray::Array2;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

pub struct MinibatchData {
    pub input_marginal_left: Tensor,
    pub input_marginal_right: Tensor,
    pub input_neigh_left: Option<Tensor>,
    pub input_neigh_right: Option<Tensor>,
    pub output_marginal_left: Option<Tensor>,
    pub output_marginal_right: Option<Tensor>,
    pub output_border_left: Option<Tensor>,
    pub output_border_right: Option<Tensor>,
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
    input_marginal_left: Vec<Tensor>,
    input_marginal_right: Vec<Tensor>,

    input_neigh_left: Option<Vec<Tensor>>,
    input_neigh_right: Option<Vec<Tensor>>,

    output_marginal_left: Option<Vec<Tensor>>,
    output_marginal_right: Option<Vec<Tensor>>,

    output_border_left: Option<Vec<Tensor>>,
    output_border_right: Option<Vec<Tensor>>,

    shuffled_input_marginal_left: Option<Vec<Tensor>>,
    shuffled_input_marginal_right: Option<Vec<Tensor>>,

    shuffled_input_neigh_left: Option<Vec<Tensor>>,
    shuffled_input_neigh_right: Option<Vec<Tensor>>,

    shuffled_output_marginal_left: Option<Vec<Tensor>>,
    shuffled_output_marginal_right: Option<Vec<Tensor>>,

    shuffled_output_border_left: Option<Vec<Tensor>>,
    shuffled_output_border_right: Option<Vec<Tensor>>,

    minibatches: Minibatches,
}

/// Just a wrapper to prevent arguments from being swapped
pub struct DataLoaderArgs<'a, D>
where
    D: RowsToTensorVec,
{
    pub input_marginal_left: &'a D,
    pub input_marginal_right: &'a D,
    pub input_neigh_left: Option<&'a D>,
    pub input_neigh_right: Option<&'a D>,
    pub output_marginal_left: Option<&'a D>,
    pub output_marginal_right: Option<&'a D>,
    pub output_border_left: Option<&'a D>,
    pub output_border_right: Option<&'a D>,
}

impl InMemoryData {
    pub fn from<D>(args: DataLoaderArgs<'_, D>) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_marginal_left = args.input_marginal_left.rows_to_tensor_vec();
        let input_marginal_right = args.input_marginal_right.rows_to_tensor_vec();

        let input_neigh_left = args.input_neigh_left.map(|x| x.rows_to_tensor_vec());
        let input_neigh_right = args.input_neigh_right.map(|x| x.rows_to_tensor_vec());

        let output_marginal_left = args.output_marginal_left.map(|x| x.rows_to_tensor_vec());
        let output_marginal_right = args.output_marginal_right.map(|x| x.rows_to_tensor_vec());

        let output_border_left = args.output_border_left.map(|x| x.rows_to_tensor_vec());
        let output_border_right = args.output_border_right.map(|x| x.rows_to_tensor_vec());

        let nsamples = input_marginal_left.len();

        let rows = (0..nsamples).collect();

        Ok(InMemoryData {
            input_marginal_left,
            input_marginal_right,
            input_neigh_left,
            input_neigh_right,
            output_marginal_left,
            output_marginal_right,
            output_border_left,
            output_border_right,
            shuffled_input_marginal_left: None,
            shuffled_input_marginal_right: None,
            shuffled_input_neigh_left: None,
            shuffled_input_neigh_right: None,
            shuffled_output_marginal_left: None,
            shuffled_output_marginal_right: None,
            shuffled_output_border_left: None,
            shuffled_output_border_right: None,
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
        if let (Some(input_marginal_left), Some(input_marginal_right)) = (
            take_lb_ub(lb, ub, target_device, Some(&self.input_marginal_left))?,
            take_lb_ub(lb, ub, target_device, Some(&self.input_marginal_right))?,
        ) {
            let input_neigh_left =
                take_lb_ub(lb, ub, target_device, self.input_neigh_left.as_ref())?;
            let input_neigh_right =
                take_lb_ub(lb, ub, target_device, self.input_neigh_left.as_ref())?;

            let output_marginal_left =
                take_lb_ub(lb, ub, target_device, self.output_marginal_left.as_ref())?;
            let output_marginal_right =
                take_lb_ub(lb, ub, target_device, self.output_marginal_right.as_ref())?;

            let output_border_left =
                take_lb_ub(lb, ub, target_device, self.output_border_left.as_ref())?;
            let output_border_right =
                take_lb_ub(lb, ub, target_device, self.output_border_left.as_ref())?;

            Ok(MinibatchData {
                input_marginal_left,
                input_marginal_right,
                input_neigh_left,
                input_neigh_right,
                output_marginal_left,
                output_marginal_right,
                output_border_left,
                output_border_right,
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
        if let (Some(input_marginal_left), Some(input_marginal_right)) = (
            take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_marginal_left.as_ref(),
            )?,
            take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_marginal_right.as_ref(),
            )?,
        ) {
            let output_marginal_left = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_output_marginal_left.as_ref(),
            )?;

            let output_marginal_right = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_output_marginal_right.as_ref(),
            )?;

            let output_border_left = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_output_border_left.as_ref(),
            )?;

            let output_border_right = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_output_border_right.as_ref(),
            )?;

            let input_neigh_left = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_neigh_left.as_ref(),
            )?;

            let input_neigh_right = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_neigh_right.as_ref(),
            )?;

            Ok(MinibatchData {
                input_marginal_left,
                input_marginal_right,
                input_neigh_left,
                input_neigh_right,
                output_marginal_left,
                output_marginal_right,
                output_border_left,
                output_border_right,
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

        self.shuffled_input_marginal_left = Some(vec![]);

        self.shuffled_input_marginal_right = Some(vec![]);

        if self.input_neigh_left.is_some() {
            self.shuffled_input_neigh_left = Some(vec![]);
        }

        if self.input_neigh_right.is_some() {
            self.shuffled_input_neigh_right = Some(vec![]);
        }

        if self.output_marginal_left.is_some() {
            self.shuffled_output_marginal_left = Some(vec![]);
        }

        if self.output_marginal_right.is_some() {
            self.shuffled_output_marginal_right = Some(vec![]);
        }

        if self.output_border_left.is_some() {
            self.shuffled_output_border_left = Some(vec![]);
        }

        if self.output_border_right.is_some() {
            self.shuffled_output_border_right = Some(vec![]);
        }

        ///////////////////////////////////
        // preload all the shuffled data //
        ///////////////////////////////////

        for batch_idx in 0..self.num_minibatch() {
            if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
                copy_shuffled(
                    samples,
                    Some(&self.input_marginal_left),
                    self.shuffled_input_marginal_left.as_mut(),
                )?;

                copy_shuffled(
                    samples,
                    Some(&self.input_marginal_right),
                    self.shuffled_input_marginal_right.as_mut(),
                )?;

                copy_shuffled(
                    samples,
                    self.input_neigh_left.as_ref(),
                    self.shuffled_input_neigh_left.as_mut(),
                )?;
                copy_shuffled(
                    samples,
                    self.input_neigh_right.as_ref(),
                    self.shuffled_input_neigh_right.as_mut(),
                )?;

                copy_shuffled(
                    samples,
                    self.output_marginal_left.as_ref(),
                    self.shuffled_output_marginal_left.as_mut(),
                )?;
                copy_shuffled(
                    samples,
                    self.output_marginal_right.as_ref(),
                    self.shuffled_output_marginal_right.as_mut(),
                )?;

                copy_shuffled(
                    samples,
                    self.output_border_left.as_ref(),
                    self.shuffled_output_border_left.as_mut(),
                )?;
                copy_shuffled(
                    samples,
                    self.output_border_right.as_ref(),
                    self.shuffled_output_border_right.as_mut(),
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
                let mut v = Tensor::from_iter(row.iter().copied(), &Device::Cpu)
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
                let mut v = Tensor::from_iter(row.iter().copied(), &Device::Cpu)
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
