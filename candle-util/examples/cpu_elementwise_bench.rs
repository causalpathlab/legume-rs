//! How long candle's CPU elementwise ops take at the masked trainer's shapes.
//!
//! A masked step gathers `[N·K, H]` tokens and runs a handful of elementwise
//! passes over them; the matmuls are threaded and cost single-digit
//! milliseconds, so whatever these passes cost on one core is the CPU step.
//! Run before and after threading candle's CPU maps; on CUDA the numbers are
//! a control, since the patch does not touch that backend.
//!
//! `cargo run --release -p candle-util --example cpu_elementwise_bench [cpu|cuda]`

use candle_core::{DType, Device, Result, Tensor, Var};

// Half of a CPU backward's cost is page-faulting fresh multi-megabyte buffers
// that glibc unmaps on free and maps again on the next allocation; an
// allocator that keeps large blocks around removes it.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use std::time::Instant;

const N: usize = 100;
const K: usize = 512;
const H: usize = 128;

fn time(dev: &Device, label: &str, mut f: impl FnMut() -> Result<()>) {
    let reps = 20;
    for _ in 0..3 {
        f().expect(label);
    }
    dev.synchronize().ok();
    let t0 = Instant::now();
    for _ in 0..reps {
        f().expect(label);
    }
    dev.synchronize().ok();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("{label:<52} {ms:8.2} ms/step");
}

fn main() -> Result<()> {
    let dev = match std::env::args().nth(1).as_deref() {
        Some("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    println!(
        "device {:?}, rayon threads {}, [N·K, H] = [{}, {}]",
        dev,
        rayon::current_num_threads(),
        N * K,
        H
    );
    let x = Tensor::rand(-1f32, 1.0, (N, K, H), &dev)?;
    let y = Tensor::rand(-1f32, 1.0, (N, K, H), &dev)?;
    let gate = Tensor::rand(0f32, 1.0, (N, K, 1), &dev)?;
    let row = Tensor::rand(0f32, 1.0, (1, 1, H), &dev)?;

    time(&dev, "exp", || x.exp().map(|_| ()));
    time(&dev, "mul, same shape", || (&x * &y).map(|_| ()));
    time(&dev, "broadcast_mul by [N,K,1] (the gate)", || {
        x.broadcast_mul(&gate).map(|_| ())
    });
    time(&dev, "broadcast_add of a [1,1,H] row", || {
        x.broadcast_add(&row).map(|_| ())
    });
    time(&dev, "affine", || x.affine(2.0, 1.0).map(|_| ()));
    time(&dev, "gt 0 (cmp) + to_dtype f32", || {
        x.gt(0.0)?.to_dtype(DType::F32).map(|_| ())
    });
    time(&dev, "clamp", || x.clamp(-0.5, 0.5).map(|_| ()));
    time(&dev, "sum_all", || x.sum_all().map(|_| ()));
    time(&dev, "sum over H (keepdim)", || {
        x.sum_keepdim(2).map(|_| ())
    });
    time(&dev, "transpose(1,2).contiguous()", || {
        x.transpose(1, 2)?.contiguous().map(|_| ())
    });
    // The pieces of a backward pass as candle composes them: a fresh zero
    // accumulator, the reduce's broadcast gradient added into it, and the
    // elementwise gradient of `exp`.
    let res = x.exp()?;
    let ones_b = Tensor::ones((), DType::F32, &dev)?.broadcast_as(x.shape())?;
    time(&dev, "bwd piece: zeros_like", || x.zeros_like().map(|_| ()));
    time(&dev, "bwd piece: zeros + broadcast scalar grad", || {
        (x.zeros_like()? + &ones_b).map(|_| ())
    });
    time(&dev, "bwd piece: broadcast scalar grad * res", || {
        (&ones_b * &res).map(|_| ())
    });
    time(&dev, "bwd piece: contiguous grad * res", || {
        (&y * &res).map(|_| ())
    });
    time(&dev, "bwd piece: ones_like().contiguous()", || {
        x.ones_like()?.contiguous().map(|_| ())
    });
    // Each link of the encoder's gate chain with its backward, then the
    // chain: what a training step pays per token pass.
    let v = Var::from_tensor(&x)?;
    time(&dev, "exp, fwd+bwd", || {
        let _ = v.as_tensor().exp()?.sum_all()?.backward()?;
        Ok(())
    });
    time(&dev, "broadcast_mul by the gate, fwd+bwd", || {
        let _ = v.as_tensor().broadcast_mul(&gate)?.sum_all()?.backward()?;
        Ok(())
    });
    time(&dev, "broadcast_add of the row, fwd+bwd", || {
        let _ = v.as_tensor().broadcast_add(&row)?.sum_all()?.backward()?;
        Ok(())
    });
    time(&dev, "sum_all alone, fwd+bwd", || {
        let _ = v.as_tensor().sum_all()?.backward()?;
        Ok(())
    });
    time(
        &dev,
        "gate chain: (x·gate + row).exp().sum, fwd+bwd",
        || {
            let z = v
                .as_tensor()
                .broadcast_mul(&gate)?
                .broadcast_add(&row)?
                .exp()?;
            let _ = z.sum_all()?.backward()?;
            Ok(())
        },
    );
    Ok(())
}
