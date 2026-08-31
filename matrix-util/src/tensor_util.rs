use crate::rand_util::{collect_f32_seeded, entropy_seed};
use crate::traits::*;
use candle_core::{CpuStorage, DType, Device, InplaceOp2, Layout, Tensor};
use rand_distr::{Gamma, StandardNormal, Uniform};
use rayon::prelude::*;

impl SampleOps for Tensor {
    type Mat = Self;
    type Scalar = f32;

    fn runif(nrow: usize, ncol: usize) -> Self::Mat {
        Self::runif_seeded(nrow, ncol, entropy_seed())
    }

    fn rnorm(nrow: usize, ncol: usize) -> Self::Mat {
        Self::rnorm_seeded(nrow, ncol, entropy_seed())
    }

    fn rgamma(nrow: usize, ncol: usize, param: (f32, f32)) -> Self::Mat {
        Self::rgamma_seeded(nrow, ncol, param, entropy_seed())
    }

    fn runif_seeded(nrow: usize, ncol: usize, seed: u64) -> Self::Mat {
        let u01 = Uniform::new(0_f32, 1_f32).expect("failed to create uniform distribution");
        let data = collect_f32_seeded(nrow * ncol, u01, seed);
        Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor runif_seeded")
    }

    fn rnorm_seeded(nrow: usize, ncol: usize, seed: u64) -> Self::Mat {
        // Candle's own `Tensor::randn` on the CPU backend draws from
        // `rand::rng()` (OS entropy) and its `Device::set_seed` errors out, so
        // there is no way to seed it through candle. Generate the sample
        // host-side instead, where the seed is honored.
        let data = collect_f32_seeded(nrow * ncol, StandardNormal, seed);
        Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor rnorm_seeded")
    }

    fn rgamma_seeded(nrow: usize, ncol: usize, param: (f32, f32), seed: u64) -> Self::Mat {
        let (shape, scale) = param;
        let pdf = Gamma::new(shape, scale).unwrap();
        let data = collect_f32_seeded(nrow * ncol, pdf, seed);
        Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)
            .expect("failed to create Tensor rgamma_seeded")
    }
}

impl MatTriplets for Tensor {
    type Mat = Self;
    type Scalar = f32;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: &[(I, I, Self::Scalar)],
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut data = vec![0_f32; ncol * nrow];
        for &(ii, jj, x_ij) in triplets {
            let ii: usize = ii.try_into().expect("failed to convert index ii");
            let jj: usize = jj.try_into().expect("failed to convert index jj");
            data[ii * ncol + jj] = x_ij;
        }
        Ok(Tensor::from_vec(data, (nrow, ncol), &Device::Cpu)?)
    }

    fn to_nonzero_triplets(&self) -> anyhow::Result<NRowNColTriplets<Self::Scalar>> {
        if let Ok((nrow, ncol)) = self.dims2() {
            let eps = 1e-6;
            let mut ret = vec![];
            let xx: Vec<Vec<Self::Scalar>> = self.to_vec2()?;
            for (i, x_i) in xx.iter().enumerate() {
                for (j, &x_ij) in x_i.iter().enumerate() {
                    if x_ij.abs() > eps {
                        ret.push((i, j, x_ij));
                    }
                }
            }

            Ok(NRowNColTriplets {
                nrow,
                ncol,
                triplets: ret,
            })
        } else {
            anyhow::bail!("not a 2D Tensor");
        }
    }
}

// impl CandleDataLoaderOps for Tensor {
//     fn rows_to_tensor_vec(&self) -> Vec<Tensor> {
//         let mut idx_data = (0..self.dims()[0])
//             .map(|i| (i, self.narrow(0, i, 1).expect("").clone()))
//             .collect::<Vec<_>>();

//         idx_data.sort_by_key(|(i, _)| *i);
//         idx_data.into_iter().map(|(_, t)| t).collect()
//     }
// }

///////////////////////////////////
// Fused elementwise CPU kernels //
///////////////////////////////////

impl FusedTensorOps for Tensor {
    fn clamped_exp_add_inplace(self, offset: &Tensor, ceiling: f64) -> anyhow::Result<Self> {
        // One eligibility test, not a shape test: contiguity and dtype are this
        // kernel's requirements, not the math's, so anything it cannot walk flatly
        // takes the op chain rather than silently reading the wrong elements.
        let broadcast = match (self.dims(), offset.dims()) {
            (&[n, f], &[n_off, f_off]) => Broadcast::of(n, f, n_off, f_off),
            _ => None,
        };
        let fused = matches!(self.device(), Device::Cpu)
            && self.dtype() == DType::F32
            && offset.dtype() == DType::F32
            && self.is_contiguous()
            && offset.is_contiguous();

        let (Some(broadcast), true) = (broadcast, fused) else {
            return Ok(self.broadcast_add(offset)?.minimum(ceiling)?.exp()?);
        };

        self.inplace_op2(
            offset,
            &ClampedExpAdd {
                broadcast,
                // f64 → f32 once here rather than per element, by the same cast
                // `Tensor::minimum` applies to a scalar bound — so the fused path and
                // the op chain agree bitwise.
                ceiling: ceiling as f32,
            },
        )?;
        Ok(self)
    }
}

/// How the `[N, F]` receiver reads its offset. The three shapes
/// `Tensor::broadcast_add` accepts here, resolved once outside the row loop.
#[derive(Clone, Copy)]
enum Broadcast {
    /// `[N, F]` — each row reads its own.
    Element,
    /// `[1, F]` — every row reads the same one.
    Column,
    /// `[N, 1]` — each row reads a single scalar.
    Row,
}

impl Broadcast {
    fn of(n: usize, f: usize, n_off: usize, f_off: usize) -> Option<Self> {
        match (n_off, f_off) {
            _ if n_off == n && f_off == f => Some(Self::Element),
            (1, _) if f_off == f => Some(Self::Column),
            (_, 1) if n_off == n => Some(Self::Row),
            _ => None,
        }
    }
}

/// The fused kernel behind [`FusedTensorOps::clamped_exp_add_inplace`]. CPU only by
/// construction: the device forwards `InplaceOp2` supplies default to an error, and the
/// caller never reaches them.
struct ClampedExpAdd {
    broadcast: Broadcast,
    ceiling: f32,
}

/// Elements a rayon task should carry at minimum. Rows are the natural unit — each is
/// a contiguous run — but a thin panel (`probe` against a few hundred genes) makes a
/// row too small to pay for a `join`, so short rows are batched up to this.
const MIN_FUSED_TASK_ELEMS: usize = 4096;

impl InplaceOp2 for ClampedExpAdd {
    fn name(&self) -> &'static str {
        "clamped-exp-add"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<()> {
        let (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) = (s1, s2) else {
            candle_core::bail!("clamped-exp-add: expected f32 storage on both operands");
        };
        let (n, f) = l1.shape().dims2()?;

        // Both layouts were checked contiguous, so a slice from the start offset is the
        // whole logical tensor in row-major order.
        let lhs = &mut lhs[l1.start_offset()..l1.start_offset() + n * f];
        let rhs = &rhs[l2.start_offset()..l2.start_offset() + l2.shape().elem_count()];

        let (ceiling, broadcast) = (self.ceiling, self.broadcast);
        let rows_per_task = MIN_FUSED_TASK_ELEMS.div_ceil(f.max(1)).max(1);
        lhs.par_chunks_mut(rows_per_task * f)
            .enumerate()
            .for_each(|(t, block)| {
                for (r, row) in block.chunks_mut(f).enumerate() {
                    let i = t * rows_per_task + r;
                    match broadcast {
                        Broadcast::Element => apply(row, &rhs[i * f..(i + 1) * f], ceiling),
                        Broadcast::Column => apply(row, rhs, ceiling),
                        Broadcast::Row => {
                            let b = rhs[i];
                            for x in row.iter_mut() {
                                *x = (*x + b).min(ceiling).exp();
                            }
                        }
                    }
                }
            });
        Ok(())
    }
}

/// `row[j] <- exp(min(row[j] + off[j], ceiling))`.
#[inline]
fn apply(row: &mut [f32], off: &[f32], ceiling: f32) {
    for (x, &b) in row.iter_mut().zip(off) {
        *x = (*x + b).min(ceiling).exp();
    }
}

#[cfg(test)]
#[path = "tensor_util_tests.rs"]
mod tests;
