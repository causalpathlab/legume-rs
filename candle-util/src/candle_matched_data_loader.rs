use crate::candle_data_loader_util::*;
use anyhow::{anyhow, Context};
use candle_core::{Device, Tensor};

pub struct MinibatchData {
    pub input_left: Tensor,
    pub input_right: Tensor,
    pub input_aux_left: Option<Tensor>,
    pub input_aux_right: Option<Tensor>,
    pub output_left: Option<Tensor>,
    pub output_right: Option<Tensor>,
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
    input_left: Vec<Tensor>,
    input_right: Vec<Tensor>,

    input_aux_left: Option<Vec<Tensor>>,
    input_aux_right: Option<Vec<Tensor>>,

    output_left: Option<Vec<Tensor>>,
    output_right: Option<Vec<Tensor>>,

    shuffled_input_left: Option<Vec<Tensor>>,
    shuffled_input_right: Option<Vec<Tensor>>,

    shuffled_input_aux_left: Option<Vec<Tensor>>,
    shuffled_input_aux_right: Option<Vec<Tensor>>,

    shuffled_output_left: Option<Vec<Tensor>>,
    shuffled_output_right: Option<Vec<Tensor>>,

    minibatches: Minibatches,
}

/// Just a wrapper to prevent arguments from being swapped
pub struct DataLoaderArgs<'a, D>
where
    D: RowsToTensorVec,
{
    pub input_left: &'a D,
    pub input_right: &'a D,
    pub input_aux_left: Option<&'a D>,
    pub input_aux_right: Option<&'a D>,
    pub output_left: Option<&'a D>,
    pub output_right: Option<&'a D>,
}

impl InMemoryData {
    pub fn from<D>(args: DataLoaderArgs<'_, D>) -> anyhow::Result<Self>
    where
        D: RowsToTensorVec,
    {
        let input_left = args.input_left.rows_to_tensor_vec();
        let input_right = args.input_right.rows_to_tensor_vec();

        let input_aux_left = args.input_aux_left.map(|x| x.rows_to_tensor_vec());
        let input_aux_right = args.input_aux_right.map(|x| x.rows_to_tensor_vec());

        let output_left = args.output_left.map(|x| x.rows_to_tensor_vec());
        let output_right = args.output_right.map(|x| x.rows_to_tensor_vec());

        let nsamples = input_left.len();

        let rows = (0..nsamples).collect();

        Ok(InMemoryData {
            input_left,
            input_right,
            input_aux_left,
            input_aux_right,
            output_left,
            output_right,
            shuffled_input_left: None,
            shuffled_input_right: None,
            shuffled_input_aux_left: None,
            shuffled_input_aux_right: None,
            shuffled_output_left: None,
            shuffled_output_right: None,
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
        if let (Some(input_left), Some(input_right)) = (
            take_lb_ub(lb, ub, target_device, Some(&self.input_left))?,
            take_lb_ub(lb, ub, target_device, Some(&self.input_right))?,
        ) {
            let input_aux_left = take_lb_ub(lb, ub, target_device, self.input_aux_left.as_ref())?;
            let input_aux_right = take_lb_ub(lb, ub, target_device, self.input_aux_left.as_ref())?;

            let output_left = take_lb_ub(lb, ub, target_device, self.output_left.as_ref())?;
            let output_right = take_lb_ub(lb, ub, target_device, self.output_right.as_ref())?;

            Ok(MinibatchData {
                input_left,
                input_right,
                input_aux_left,
                input_aux_right,
                output_left,
                output_right,
            })
        } else {
            Err(anyhow!("no input data"))
        }
    }

    fn minibatch_shuffled(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<MinibatchData> {
        if let (Some(input_left), Some(input_right)) = (
            take_shuffled(batch_idx, target_device, self.shuffled_input_left.as_ref())?,
            take_shuffled(batch_idx, target_device, self.shuffled_input_right.as_ref())?,
        ) {
            let output_left =
                take_shuffled(batch_idx, target_device, self.shuffled_output_left.as_ref())?;

            let output_right = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_output_right.as_ref(),
            )?;

            let input_aux_left = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_aux_left.as_ref(),
            )?;

            let input_aux_right = take_shuffled(
                batch_idx,
                target_device,
                self.shuffled_input_aux_right.as_ref(),
            )?;

            Ok(MinibatchData {
                input_left,
                input_right,
                input_aux_left,
                input_aux_right,
                output_left,
                output_right,
            })
        } else {
            Err(anyhow!("need to shuffle data"))
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

        self.shuffled_input_left = Some(vec![]);

        self.shuffled_input_right = Some(vec![]);

        if self.input_aux_left.is_some() {
            self.shuffled_input_aux_left = Some(vec![]);
        }

        if self.input_aux_right.is_some() {
            self.shuffled_input_aux_right = Some(vec![]);
        }

        if self.output_left.is_some() {
            self.shuffled_output_left = Some(vec![]);
        }

        if self.output_right.is_some() {
            self.shuffled_output_right = Some(vec![]);
        }

        ///////////////////////////////////
        // preload all the shuffled data //
        ///////////////////////////////////

        for batch_idx in 0..self.num_minibatch() {
            if let Some(samples) = self.minibatches.chunks.get(batch_idx) {
                copy_shuffled(
                    samples,
                    Some(&self.input_left),
                    self.shuffled_input_left.as_mut(),
                )
                .context("input left")?;
                copy_shuffled(
                    samples,
                    Some(&self.input_right),
                    self.shuffled_input_right.as_mut(),
                )
                .context("input right")?;

                copy_shuffled(
                    samples,
                    self.input_aux_left.as_ref(),
                    self.shuffled_input_aux_left.as_mut(),
                )
                .context("aux left")?;
                copy_shuffled(
                    samples,
                    self.input_aux_right.as_ref(),
                    self.shuffled_input_aux_right.as_mut(),
                )
                .context("aux right")?;

                copy_shuffled(
                    samples,
                    self.output_left.as_ref(),
                    self.shuffled_output_left.as_mut(),
                )
                .context("out left")?;
                copy_shuffled(
                    samples,
                    self.output_right.as_ref(),
                    self.shuffled_output_right.as_mut(),
                )
                .context("out right")?;
            } else {
                return Err(anyhow!(
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
            return Err(anyhow!(
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
            Err(anyhow!(
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
