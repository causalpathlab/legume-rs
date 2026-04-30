use anyhow::anyhow;
use candle_core::{Device, Tensor};
use matrix_util::traits::ConvertMatOps;
use rand::prelude::SliceRandom;
use rayon::prelude::*;

/// Upload a matrix to `device` as a contiguous `[N, D]` tensor.
/// `to_tensor` returns a transposed view; `contiguous()` makes it
/// layout-compatible with `index_select`.
pub(crate) fn upload_to_device<D: ConvertMatOps>(x: &D, device: &Device) -> anyhow::Result<Tensor> {
    Ok(x.to_tensor(device)?.contiguous()?)
}

/// Bootstrap-sample `ntot` indices from `[0, n)` with replacement, in
/// parallel. Shared by `Minibatches::shuffle_minibatch` (CPU path,
/// `usize`) and the device-resident loaders (`u32`).
pub(crate) fn bootstrap_indices<I>(n: usize, ntot: usize) -> Vec<I>
where
    I: TryFrom<usize> + Send,
    <I as TryFrom<usize>>::Error: std::fmt::Debug,
{
    use rand_distr::{Distribution, Uniform};
    let unif = Uniform::new(0usize, n).expect("unif [0 .. n)");
    (0..ntot)
        .into_par_iter()
        .map_init(rand::rng, |rng, _| {
            I::try_from(unif.sample(rng)).expect("index fits in target type")
        })
        .collect()
}

///
/// A helper `struct` for shuffling and creating minibatch indexes;
/// after `shuffle_minibatch` is called, `chunks` partition indexes.
///
pub struct Minibatches {
    pub samples: Vec<usize>,
    pub chunks: Vec<Vec<usize>>,
}

impl Minibatches {
    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        let mut rng = rand::rng();
        self.samples.shuffle(&mut rng);

        let nbatch = (self.size() + batch_size) / batch_size;
        let ntot = nbatch * batch_size;

        let indexes: Vec<usize> = bootstrap_indices(self.size(), ntot);

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

pub fn take_lb_ub(
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

pub fn copy_shuffled(
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

pub fn take_shuffled(
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
