use crate::candle_data_loader_util::*;

use candle_core::{Device, Tensor};

use matrix_util::traits::{CandleDataLoaderOps, ConvertMatOps};

#[derive(Debug)]
pub struct JointMinibatchData {
    pub input: Vec<Tensor>,
    pub input_null: Vec<Option<Tensor>>,
    pub output: Vec<Option<Tensor>>,
    pub output_null: Vec<Option<Tensor>>,
}

pub trait JointDataLoader {
    fn minibatch_shuffled(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<JointMinibatchData>;

    fn num_data(&self) -> usize;

    fn num_minibatch(&self) -> usize;

    fn shuffle_minibatch(&mut self, batch_size: usize) -> anyhow::Result<()>;
}

pub struct JointInMemoryArgs<'a, D>
where
    D: CandleDataLoaderOps,
{
    pub input: &'a Vec<D>,
    pub input_null: &'a Vec<Option<D>>,
    pub output: &'a Vec<Option<D>>,
    pub output_null: &'a Vec<Option<D>>,
}

/// A simple data loader for in-memory multiple 2d matrices.  Each row
/// will be considered as a feature vector. The number of samples is
/// the number of rows.
///
pub struct JointInMemoryData {
    input_data: Vec<Vec<Tensor>>,
    input_null_data: Vec<Option<Vec<Tensor>>>,

    output_data: Vec<Option<Vec<Tensor>>>,
    output_null_data: Vec<Option<Vec<Tensor>>>,

    shuffled_input_data: Option<Vec<Vec<Tensor>>>,
    shuffled_input_null_data: Option<Vec<Vec<Tensor>>>,

    shuffled_output_data: Option<Vec<Vec<Tensor>>>,
    shuffled_output_null_data: Option<Vec<Vec<Tensor>>>,

    device_input: Vec<Tensor>,
    device_input_null: Vec<Option<Tensor>>,
    device_output: Vec<Option<Tensor>>,
    device_output_null: Vec<Option<Tensor>>,

    cached_minibatches: Vec<JointMinibatchData>,

    minibatches: Minibatches,
}

impl JointInMemoryData {
    pub fn from<D>(args: JointInMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps,
    {
        if args.input.is_empty() {
            return Err(anyhow::anyhow!("empty input in data loader"));
        }

        let input_data: Vec<Vec<Tensor>> =
            args.input.iter().map(|x| x.rows_to_tensor_vec()).collect();

        let nrows = input_data[0].len();

        let input_null_data: Vec<Option<Vec<Tensor>>> = args
            .input_null
            .iter()
            .map(|vv| vv.as_ref().map(|x| x.rows_to_tensor_vec()))
            .collect();

        let output_data: Vec<Option<Vec<Tensor>>> = args
            .output
            .iter()
            .map(|vv| vv.as_ref().map(|x| x.rows_to_tensor_vec()))
            .collect();

        let output_null_data: Vec<Option<Vec<Tensor>>> = args
            .output_null
            .iter()
            .map(|vv| vv.as_ref().map(|x| x.rows_to_tensor_vec()))
            .collect();

        let rows = (0..nrows).collect();

        if input_null_data.len() != input_data.len() {
            return Err(anyhow::anyhow!(
                "input_null: Found different number of data matrices"
            ));
        }

        if output_data.len() != input_data.len() {
            return Err(anyhow::anyhow!(
                "output: Found different number of data matrices"
            ));
        }

        if output_null_data.len() != input_data.len() {
            return Err(anyhow::anyhow!(
                "output_null: Found different number of data matrices"
            ));
        }

        Ok(JointInMemoryData {
            input_data,
            input_null_data,
            output_data,
            output_null_data,
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_output_data: None,
            shuffled_output_null_data: None,
            device_input: vec![],
            device_input_null: vec![],
            device_output: vec![],
            device_output_null: vec![],
            cached_minibatches: vec![],
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    /// Build the loader by uploading every modality's input/null/output as
    /// a single `[N, D_m]` device tensor. After this call,
    /// `shuffle_minibatch_on_device` produces zero-copy minibatch views.
    pub fn from_device<D>(
        args: JointInMemoryArgs<D>,
        target_device: &Device,
    ) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps + ConvertMatOps,
    {
        if args.input.is_empty() {
            return Err(anyhow::anyhow!("empty input in data loader"));
        }
        let ntypes = args.input.len();
        if args.input_null.len() != ntypes
            || args.output.len() != ntypes
            || args.output_null.len() != ntypes
        {
            return Err(anyhow::anyhow!(
                "joint loader: input/input_null/output/output_null length mismatch"
            ));
        }

        let device_input: Vec<Tensor> = args
            .input
            .iter()
            .map(|x| upload_to_device(x, target_device))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let nrows = device_input[0].dim(0)?;

        let device_input_null: Vec<Option<Tensor>> = args
            .input_null
            .iter()
            .map(|vv| {
                vv.as_ref()
                    .map(|x| upload_to_device(x, target_device))
                    .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let device_output: Vec<Option<Tensor>> = args
            .output
            .iter()
            .map(|vv| {
                vv.as_ref()
                    .map(|x| upload_to_device(x, target_device))
                    .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let device_output_null: Vec<Option<Tensor>> = args
            .output_null
            .iter()
            .map(|vv| {
                vv.as_ref()
                    .map(|x| upload_to_device(x, target_device))
                    .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let rows = (0..nrows).collect();

        Ok(JointInMemoryData {
            input_data: vec![],
            input_null_data: vec![],
            output_data: vec![],
            output_null_data: vec![],
            shuffled_input_data: None,
            shuffled_input_null_data: None,
            shuffled_output_data: None,
            shuffled_output_null_data: None,
            device_input,
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
    pub fn shuffle_minibatch_on_device(&mut self, batch_size: usize) -> anyhow::Result<()> {
        if self.device_input.is_empty() {
            return Err(anyhow::anyhow!(
                "call from_device before shuffle_minibatch_on_device"
            ));
        }

        let dev = self.device_input[0].device().clone();
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
        let idx_tensor = Tensor::from_vec(idx, ntot, &dev)?;

        let shuffled_input: Vec<Tensor> = self
            .device_input
            .iter()
            .map(|t| Ok(t.index_select(&idx_tensor, 0)?))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let shuffled_input_null: Vec<Option<Tensor>> = self
            .device_input_null
            .iter()
            .map(|t| {
                t.as_ref()
                    .map(|t| t.index_select(&idx_tensor, 0).map_err(anyhow::Error::from))
                    .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let shuffled_output: Vec<Option<Tensor>> = self
            .device_output
            .iter()
            .map(|t| {
                t.as_ref()
                    .map(|t| t.index_select(&idx_tensor, 0).map_err(anyhow::Error::from))
                    .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let shuffled_output_null: Vec<Option<Tensor>> = self
            .device_output_null
            .iter()
            .map(|t| {
                t.as_ref()
                    .map(|t| t.index_select(&idx_tensor, 0).map_err(anyhow::Error::from))
                    .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut chunks: Vec<Vec<usize>> = Vec::with_capacity(nbatch);
        let mut cached: Vec<JointMinibatchData> = Vec::with_capacity(nbatch);

        for b in 0..nbatch {
            let start = b * batch_size;

            let input: Vec<Tensor> = shuffled_input
                .iter()
                .map(|t| Ok(t.narrow(0, start, batch_size)?))
                .collect::<anyhow::Result<Vec<_>>>()?;

            let input_null: Vec<Option<Tensor>> = shuffled_input_null
                .iter()
                .map(|t| {
                    t.as_ref()
                        .map(|t| t.narrow(0, start, batch_size).map_err(anyhow::Error::from))
                        .transpose()
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let output: Vec<Option<Tensor>> = shuffled_output
                .iter()
                .map(|t| {
                    t.as_ref()
                        .map(|t| t.narrow(0, start, batch_size).map_err(anyhow::Error::from))
                        .transpose()
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let output_null: Vec<Option<Tensor>> = shuffled_output_null
                .iter()
                .map(|t| {
                    t.as_ref()
                        .map(|t| t.narrow(0, start, batch_size).map_err(anyhow::Error::from))
                        .transpose()
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            cached.push(JointMinibatchData {
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
    pub fn minibatch_cached(&self, batch_idx: usize) -> &JointMinibatchData {
        &self.cached_minibatches[batch_idx]
    }
}

impl JointDataLoader for JointInMemoryData {
    fn minibatch_shuffled(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<JointMinibatchData> {
        fn extract(
            shuffled: &[Vec<Tensor>],
            batch_idx: usize,
            target_device: &Device,
        ) -> Vec<Option<Tensor>> {
            shuffled
                .iter()
                .map(|vec_each_type| {
                    if vec_each_type.is_empty() {
                        return None;
                    }

                    Some(
                        vec_each_type[batch_idx]
                            .clone()
                            .to_device(target_device)
                            .unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        }

        if let (Some(input), Some(input_null), Some(output), Some(output_null)) = (
            self.shuffled_input_data.as_ref(),
            self.shuffled_input_null_data.as_ref(),
            self.shuffled_output_data.as_ref(),
            self.shuffled_output_null_data.as_ref(),
        ) {
            let input = extract(input, batch_idx, target_device)
                .into_iter()
                .map(|x| x.unwrap())
                .collect::<Vec<_>>();

            let input_null = extract(input_null, batch_idx, target_device);

            let output = extract(output, batch_idx, target_device);

            let output_null = extract(output_null, batch_idx, target_device);

            Ok(JointMinibatchData {
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
        self.minibatches.shuffle_minibatch(batch_size);

        let ntypes = self.input_data.len();

        self.shuffled_input_data = Some(vec![Vec::with_capacity(self.num_minibatch()); ntypes]);
        self.shuffled_input_null_data =
            Some(vec![Vec::with_capacity(self.num_minibatch()); ntypes]);
        self.shuffled_output_data = Some(vec![Vec::with_capacity(self.num_minibatch()); ntypes]);
        self.shuffled_output_null_data =
            Some(vec![Vec::with_capacity(self.num_minibatch()); ntypes]);

        ///////////////////////////////////
        // preload all the shuffled data //
        ///////////////////////////////////

        for batch_idx in 0..self.num_minibatch() {
            if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
                for d in 0..ntypes {
                    copy_shuffled(
                        samples,
                        Some(&self.input_data[d]),
                        mut_vec_at(&mut self.shuffled_input_data, d),
                    )?;

                    copy_shuffled(
                        samples,
                        self.output_data[d].as_ref(),
                        mut_vec_at(&mut self.shuffled_output_data, d),
                    )?;

                    copy_shuffled(
                        samples,
                        self.input_null_data[d].as_ref(),
                        mut_vec_at(&mut self.shuffled_input_null_data, d),
                    )?;

                    copy_shuffled(
                        samples,
                        self.output_null_data[d].as_ref(),
                        mut_vec_at(&mut self.shuffled_output_null_data, d),
                    )?;
                }
            } else {
                return Err(anyhow::anyhow!(
                    "invalid index = {} vs. total # = {}",
                    batch_idx,
                    self.num_minibatch()
                ));
            }
        }

        fn mut_vec_at(data: &mut Option<Vec<Vec<Tensor>>>, d: usize) -> Option<&mut Vec<Tensor>> {
            data.as_mut().map(|v| v[d].as_mut())
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn small_mat(n: usize, d: usize, off: f32) -> DMatrix<f32> {
        DMatrix::<f32>::from_fn(n, d, |i, j| (i * d + j) as f32 + off)
    }

    #[test]
    fn from_device_caches_minibatches() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let mods: Vec<DMatrix<f32>> = vec![small_mat(6, 4, 0.0), small_mat(6, 5, 100.0)];
        let nulls: Vec<Option<DMatrix<f32>>> = vec![None, None];
        let outs: Vec<Option<DMatrix<f32>>> = vec![None, None];
        let onulls: Vec<Option<DMatrix<f32>>> = vec![None, None];

        let mut loader = JointInMemoryData::from_device(
            JointInMemoryArgs::<DMatrix<f32>> {
                input: &mods,
                input_null: &nulls,
                output: &outs,
                output_null: &onulls,
            },
            &dev,
        )?;
        loader.shuffle_minibatch_on_device(3)?;
        assert_eq!(loader.num_minibatch(), 2);
        for b in 0..loader.num_minibatch() {
            let mb = loader.minibatch_cached(b);
            assert_eq!(mb.input.len(), 2);
            assert_eq!(mb.input[0].dims(), &[3, 4]);
            assert_eq!(mb.input[1].dims(), &[3, 5]);
        }
        Ok(())
    }
}
