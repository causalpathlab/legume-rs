//! Host ↔ device one-liners that every engine otherwise spells out.

use candle_core::{Device, Result, Tensor, WithDType};

/// A 1-D device tensor from a host slice.
pub fn to_1d<T: WithDType>(v: &[T], dev: &Device) -> Result<Tensor> {
    Tensor::from_slice(v, v.len(), dev)
}

/// Row-major host copy of a tensor of any rank.
pub fn to_host(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

/// `acc += x`, starting from nothing.
pub fn add_into(acc: &mut Option<Tensor>, x: Tensor) -> Result<()> {
    *acc = Some(match acc.take() {
        None => x,
        Some(a) => (a + x)?,
    });
    Ok(())
}
