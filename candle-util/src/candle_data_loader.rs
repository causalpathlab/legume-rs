use crate::candle_data_loader_util::*;

use candle_core::{Device, Tensor};

use matrix_util::traits::{CandleDataLoaderOps, ConvertMatOps};

pub struct MinibatchData {
    pub input: Tensor,
    pub input_null: Option<Tensor>,
    pub output: Option<Tensor>,
    pub output_null: Option<Tensor>,
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

    output_data: Option<Vec<Tensor>>,
    output_null_data: Option<Vec<Tensor>>,

    shuffled_input_data: Option<Vec<Tensor>>,
    shuffled_input_null_data: Option<Vec<Tensor>>,

    shuffled_output_data: Option<Vec<Tensor>>,
    shuffled_output_null_data: Option<Vec<Tensor>>,

    device_input: Option<Tensor>,
    device_input_null: Option<Tensor>,
    device_output: Option<Tensor>,
    device_output_null: Option<Tensor>,

    cached_minibatches: Vec<MinibatchData>,

    minibatches: Minibatches,
}

pub struct InMemoryArgs<'a, D>
where
    D: CandleDataLoaderOps,
{
    pub input: &'a D,
    pub input_null: Option<&'a D>,
    pub output: Option<&'a D>,
    pub output_null: Option<&'a D>,
}

impl InMemoryData {
    pub fn from<D>(args: InMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps,
    {
        let input = args.input;
        let input_null = args.input_null;
        let output = args.output;
        let output_null = args.output_null;

        let input_data = input.rows_to_tensor_vec();
        let input_null_data = input_null.map(|x| x.rows_to_tensor_vec());
        let output_data = output.map(|x| x.rows_to_tensor_vec());
        let output_null_data = output_null.map(|x| x.rows_to_tensor_vec());

        let rows = (0..input_data.len()).collect();

        Ok(InMemoryData {
            input_data,
            input_null_data,
            output_data,
            output_null_data,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_output_data: None,
            shuffled_output_null_data: None,
            device_input: None,
            device_input_null: None,
            device_output: None,
            device_output_null: None,
            cached_minibatches: vec![],
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    /// Build the loader with each input matrix uploaded as a single
    /// `[N, D]` tensor on `target_device`. One host→device transfer
    /// per stream — minibatch construction afterwards is index-select +
    /// narrow on device, with no further transfers.
    pub fn from_device<D>(args: InMemoryArgs<D>, target_device: &Device) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps + ConvertMatOps,
    {
        let device_input = upload_to_device(args.input, target_device)?;
        let n = device_input.dim(0)?;

        let device_input_null = args
            .input_null
            .map(|x| upload_to_device(x, target_device))
            .transpose()?;
        let device_output = args
            .output
            .map(|x| upload_to_device(x, target_device))
            .transpose()?;
        let device_output_null = args
            .output_null
            .map(|x| upload_to_device(x, target_device))
            .transpose()?;

        let rows = (0..n).collect();

        Ok(InMemoryData {
            input_data: vec![],
            input_null_data: None,
            output_data: None,
            output_null_data: None,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_output_data: None,
            shuffled_output_null_data: None,
            device_input: Some(device_input),
            device_input_null,
            device_output,
            device_output_null,
            cached_minibatches: vec![],
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    /// Shuffle and cache minibatches for the device-resident path.
    /// Single index-select on device produces the shuffled `[ntot, D]`
    /// block; per-minibatch tensors are zero-copy `narrow` views.
    pub fn shuffle_minibatch_on_device(&mut self, batch_size: usize) -> anyhow::Result<()> {
        let device_input = self.device_input.as_ref().ok_or_else(|| {
            anyhow::anyhow!("call from_device before shuffle_minibatch_on_device")
        })?;

        let dev = device_input.device();
        let n = self.minibatches.samples.len();
        if n == 0 {
            self.cached_minibatches.clear();
            self.minibatches.chunks.clear();
            return Ok(());
        }
        if batch_size == 0 {
            return Err(anyhow::anyhow!("batch_size must be > 0"));
        }

        let nbatch = n.div_ceil(batch_size);
        let ntot = nbatch * batch_size;
        let idx: Vec<u32> = bootstrap_indices(n, ntot);
        let idx_tensor = Tensor::from_vec(idx, ntot, dev)?;

        let shuffled_input = device_input.index_select(&idx_tensor, 0)?;
        let shuffled_input_null = self
            .device_input_null
            .as_ref()
            .map(|t| t.index_select(&idx_tensor, 0))
            .transpose()?;
        let shuffled_output = self
            .device_output
            .as_ref()
            .map(|t| t.index_select(&idx_tensor, 0))
            .transpose()?;
        let shuffled_output_null = self
            .device_output_null
            .as_ref()
            .map(|t| t.index_select(&idx_tensor, 0))
            .transpose()?;

        let mut chunks: Vec<Vec<usize>> = Vec::with_capacity(nbatch);
        let mut cached: Vec<MinibatchData> = Vec::with_capacity(nbatch);
        for b in 0..nbatch {
            let start = b * batch_size;
            let input = shuffled_input.narrow(0, start, batch_size)?;
            let input_null = shuffled_input_null
                .as_ref()
                .map(|t| t.narrow(0, start, batch_size))
                .transpose()?;
            let output = shuffled_output
                .as_ref()
                .map(|t| t.narrow(0, start, batch_size))
                .transpose()?;
            let output_null = shuffled_output_null
                .as_ref()
                .map(|t| t.narrow(0, start, batch_size))
                .transpose()?;
            cached.push(MinibatchData {
                input,
                input_null,
                output,
                output_null,
            });
            chunks.push(Vec::new());
        }

        self.minibatches.chunks = chunks;
        self.cached_minibatches = cached;
        Ok(())
    }

    /// Retrieve a pre-computed minibatch from the device-resident cache.
    /// Panics if `shuffle_minibatch_on_device` was not called.
    pub fn minibatch_cached(&self, batch_idx: usize) -> &MinibatchData {
        &self.cached_minibatches[batch_idx]
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
            let output_null = take_lb_ub(lb, ub, target_device, self.output_null_data.as_ref())?;

            let input_null = take_lb_ub(lb, ub, target_device, self.input_null_data.as_ref())?;
            Ok(MinibatchData {
                input,
                input_null,
                output,
                output_null,
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

            let output_null = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_output_null_data.as_ref(),
            )?;

            let input_null = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_null_data.as_ref(),
            )?;

            Ok(MinibatchData {
                input,
                input_null,
                output,
                output_null,
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

        if self.output_data.is_some() {
            self.shuffled_output_data = Some(vec![]);
        }

        if self.output_null_data.is_some() {
            self.shuffled_output_null_data = Some(vec![]);
        }

        ///////////////////////////////////
        // preload all the shuffled data //
        ///////////////////////////////////

        for batch_idx in 0..self.num_minibatch() {
            if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
                copy_shuffled(
                    samples,
                    Some(&self.input_data),
                    self.shuffled_input_data.as_mut(),
                )?;
                copy_shuffled(
                    samples,
                    self.input_null_data.as_ref(),
                    self.shuffled_input_null_data.as_mut(),
                )?;

                copy_shuffled(
                    samples,
                    self.output_data.as_ref(),
                    self.shuffled_output_data.as_mut(),
                )?;
                copy_shuffled(
                    samples,
                    self.output_null_data.as_ref(),
                    self.shuffled_output_null_data.as_mut(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn small_mat(n: usize, d: usize) -> DMatrix<f32> {
        DMatrix::<f32>::from_fn(n, d, |i, j| (i * d + j) as f32)
    }

    #[test]
    fn from_device_uploads_full_tensor() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let input = small_mat(7, 3);
        let mut loader = InMemoryData::from_device(
            InMemoryArgs::<DMatrix<f32>> {
                input: &input,
                input_null: None,
                output: None,
                output_null: None,
            },
            &dev,
        )?;

        loader.shuffle_minibatch_on_device(4)?;
        // ceil(7/4) = 2 minibatches
        assert_eq!(loader.num_minibatch(), 2);
        for b in 0..loader.num_minibatch() {
            let mb = loader.minibatch_cached(b);
            assert_eq!(mb.input.dims(), &[4, 3]);
        }
        Ok(())
    }
}
