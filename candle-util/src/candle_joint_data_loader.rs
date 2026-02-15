use crate::candle_data_loader_util::*;

use candle_core::{Device, Tensor};

use matrix_util::traits::CandleDataLoaderOps;

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
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
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
