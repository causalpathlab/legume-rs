//! Row gather with a backward that runs in parallel.
//!
//! `Tensor::index_select` on a `[D, H]` table is the right forward for every
//! embedding lookup in the workspace — the encoder's context tokens, the
//! query decoder's keys, values and queries, the bipartite NCE edges. Its
//! backward is `index_add`, and candle's CUDA kernel for that parallelises
//! only over the `H` columns while walking every id serially in each thread:
//! at fifty thousand ids per step the backward costs fifty times the forward
//! and dominates the training step of every trainer that gathers rows.
//!
//! [`gather_rows`] is the same forward with its own backward,
//! [`index_add_rows`], whose CUDA kernel runs one thread per `(id, column)`
//! element with an atomic add, and whose own backward is a forward gather.
//! On the CPU both are plain loops; the results are identical to candle's ops.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

/// `table[ids]` → `[n, H]` for a `[D, H]` table and `[n]` u32 ids, with a
/// row-parallel backward. A `[D]` table (a per-row bias) gathers to `[n]`.
pub fn gather_rows(table: &Tensor, ids: &Tensor) -> Result<Tensor> {
    if table.rank() == 1 {
        return table.unsqueeze(1)?.apply_op2(ids, GatherRows)?.squeeze(1);
    }
    table.apply_op2(ids, GatherRows)
}

/// `dst` with `src[i]` added into row `ids[i]`, for `dst [D, H]`, `ids [n]`
/// u32, `src [n, H]`. Duplicate ids accumulate.
pub fn index_add_rows(dst: &Tensor, ids: &Tensor, src: &Tensor) -> Result<Tensor> {
    dst.apply_op3(ids, src, IndexAddRows)
}

fn contiguous_range(l: &Layout, what: &str) -> Result<(usize, usize)> {
    l.contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg(format!("{what}: expected a contiguous tensor")))
}

//////////////////
// gather_rows  //
//////////////////

struct GatherRows;

impl CustomOp2 for GatherRows {
    fn name(&self) -> &'static str {
        "gather-rows"
    }

    fn cpu_fwd(
        &self,
        table: &CpuStorage,
        table_l: &Layout,
        ids: &CpuStorage,
        ids_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let out = table.index_select(ids, table_l, ids_l, 0)?;
        Ok((out, gathered_shape(table_l, ids_l)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        table: &candle_core::CudaStorage,
        table_l: &Layout,
        ids: &candle_core::CudaStorage,
        ids_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let out = table.index_select(ids, table_l, ids_l, 0)?;
        Ok((out, gathered_shape(table_l, ids_l)))
    }

    fn bwd(
        &self,
        table: &Tensor,
        ids: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let grad_table = index_add_rows(&table.zeros_like()?, ids, grad_res)?;
        Ok((Some(grad_table), None))
    }
}

fn gathered_shape(table_l: &Layout, ids_l: &Layout) -> Shape {
    let mut dims = table_l.shape().dims().to_vec();
    dims[0] = ids_l.shape().elem_count();
    Shape::from_dims(&dims)
}

////////////////////
// index_add_rows //
////////////////////

struct IndexAddRows;

impl IndexAddRows {
    fn check(dst_l: &Layout, ids_l: &Layout, src_l: &Layout) -> Result<(usize, usize, usize)> {
        let (d, h) = dst_l.shape().dims2()?;
        let n = ids_l.shape().elem_count();
        let (n_src, h_src) = src_l.shape().dims2()?;
        if n != n_src || h != h_src {
            candle_core::bail!(
                "index_add_rows: dst [{d}, {h}], ids [{n}], src [{n_src}, {h_src}] do not agree"
            );
        }
        Ok((d, h, n))
    }
}

impl CustomOp3 for IndexAddRows {
    fn name(&self) -> &'static str {
        "index-add-rows"
    }

    fn cpu_fwd(
        &self,
        dst: &CpuStorage,
        dst_l: &Layout,
        ids: &CpuStorage,
        ids_l: &Layout,
        src: &CpuStorage,
        src_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (d, h, n) = Self::check(dst_l, ids_l, src_l)?;
        let (d0, d1) = contiguous_range(dst_l, "index_add_rows dst")?;
        let (i0, i1) = contiguous_range(ids_l, "index_add_rows ids")?;
        let (s0, s1) = contiguous_range(src_l, "index_add_rows src")?;
        let (CpuStorage::F32(dst), CpuStorage::U32(ids), CpuStorage::F32(src)) = (dst, ids, src)
        else {
            candle_core::bail!("index_add_rows: expected f32 dst/src and u32 ids");
        };
        let mut out = dst[d0..d1].to_vec();
        let ids = &ids[i0..i1];
        let src = &src[s0..s1];
        for (i, &row) in ids.iter().enumerate().take(n) {
            let row = row as usize;
            if row >= d {
                candle_core::bail!("index_add_rows: id {row} out of range for {d} rows");
            }
            let (o, s) = (&mut out[row * h..(row + 1) * h], &src[i * h..(i + 1) * h]);
            for (x, y) in o.iter_mut().zip(s) {
                *x += y;
            }
        }
        Ok((CpuStorage::F32(out), dst_l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        dst: &candle_core::CudaStorage,
        dst_l: &Layout,
        ids: &candle_core::CudaStorage,
        ids_l: &Layout,
        src: &candle_core::CudaStorage,
        src_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (_d, h, n) = Self::check(dst_l, ids_l, src_l)?;
        let (d0, d1) = contiguous_range(dst_l, "index_add_rows dst")?;
        let (i0, i1) = contiguous_range(ids_l, "index_add_rows ids")?;
        let (s0, s1) = contiguous_range(src_l, "index_add_rows src")?;
        let dev = dst.device();
        let dst_slice = dst.as_cuda_slice::<f32>()?.slice(d0..d1);
        let ids_slice = ids.as_cuda_slice::<u32>()?.slice(i0..i1);
        let src_slice = src.as_cuda_slice::<f32>()?.slice(s0..s1);
        // Out = a copy of dst, then every (id, column) adds its own element.
        let mut out = unsafe { dev.alloc::<f32>(d1 - d0)? };
        dev.memcpy_dtod(&dst_slice, &mut out)?;
        let total = n * h;
        if total > 0 {
            let func = dev.get_or_load_custom_func(
                "index_add_rows_f32",
                "legume_fast_index",
                cuda_ptx()?,
            )?;
            let cfg = LaunchConfig::for_num_elems(total as u32);
            let mut builder = func.builder();
            builder.arg(&ids_slice);
            builder.arg(&src_slice);
            builder.arg(&out);
            candle_core::builder_arg!(builder, total as u32);
            candle_core::builder_arg!(builder, h as u32);
            // SAFETY: the kernel reads `ids`/`src` and adds into `out`, all sized
            // by the checks above.
            unsafe { builder.launch(cfg) }.w()?;
        }
        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::F32(out),
                device: dev.clone(),
            },
            dst_l.shape().clone(),
        ))
    }

    fn bwd(
        &self,
        _dst: &Tensor,
        ids: &Tensor,
        _src: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        // ∂/∂dst is the identity; ∂/∂src[i] is the gradient at row ids[i].
        Ok((
            Some(grad_res.clone()),
            None,
            Some(gather_rows(grad_res, ids)?),
        ))
    }
}

#[cfg(feature = "cuda")]
const CUDA_SRC: &str = r#"
extern "C" __global__ void index_add_rows_f32(
    const unsigned int *ids, const float *src, float *out,
    const unsigned int total, const unsigned int h)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int row = i / h;
    unsigned int col = i - row * h;
    atomicAdd(out + ids[row] * h + col, src[i]);
}
"#;

/// The kernel's PTX, compiled once per process through nvrtc.
#[cfg(feature = "cuda")]
fn cuda_ptx() -> Result<&'static str> {
    use std::sync::OnceLock;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    match PTX.get_or_init(|| {
        candle_core::cuda_backend::cudarc::nvrtc::compile_ptx(CUDA_SRC)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(e) => Err(candle_core::Error::Msg(format!(
            "index_add_rows: nvrtc failed: {e}"
        ))),
    }
}

#[cfg(test)]
#[path = "fast_index_tests.rs"]
mod fast_index_tests;
