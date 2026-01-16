use crate::candle_data_loader_util::*;

use candle_core::{Device, Tensor};

use matrix_util::traits::CandleDataLoaderOps;

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
//         let samples = (0..data.num_columns()).collect();
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
