//! Frozen feature-side primitives shared by `graph-embedding-util` and
//! the senna topic-model encoders.
//!
//! The host loader lives in `auxiliary-data::frozen_features` (it knows
//! about parquet I/O and gene-name canonicalization). This module turns
//! the host-side `DMatrix` / `Vec<f32>` it returns into device tensors
//! suitable for two distinct freeze mechanisms:
//!
//! - **gbe-style freeze**: register as plain `Tensor`s outside any
//!   `VarMap`. Constructed via [`FrozenFeatureSide::from_parts`] and
//!   handed to `JointEmbedModel` directly. `AdamW::new(varmap.all_vars())`
//!   never sees them.
//! - **topic-style freeze**: register as `Var`s in the same `VarMap` the
//!   encoder uses, so the frozen weights round-trip through safetensors
//!   save/load. Installed via [`install_frozen_var_2d`], and the caller
//!   excludes the var name from the optimizer's parameter set via
//!   [`trainable_vars`].

use candle_core::{Device, Result, Tensor, Var};
use candle_nn::VarMap;
use nalgebra::DMatrix;
use rustc_hash::FxHashSet;

/// Frozen feature-side tensors ready to be cloned into every axis head
/// (gbe) or to seed an encoder's `feature_embeddings` slot (topic). Both
/// tensors are constructed outside any `VarMap` and `.detach()`-ed so a
/// stray autograd attachment can't propagate gradients into them.
pub struct FrozenFeatureSide {
    /// `[D, H]` row-major.
    pub e_feat: Tensor,
    /// `[D]`.
    pub b_feat: Tensor,
    pub h: usize,
}

impl FrozenFeatureSide {
    /// Convert a host-side `(e_feat, b_feat)` pair (typically the output
    /// of `auxiliary_data::frozen_features::load_frozen_feature_host`)
    /// into device tensors. `e_feat` is a column-major `nalgebra::DMatrix`,
    /// so this flips to candle's row-major `[D, H]` in one pass.
    pub fn from_parts(e_feat: &DMatrix<f32>, b_feat: &[f32], dev: &Device) -> Result<Self> {
        let d = e_feat.nrows();
        let h = e_feat.ncols();
        assert_eq!(
            b_feat.len(),
            d,
            "FrozenFeatureSide: e_feat has {d} rows but b_feat has {} entries",
            b_feat.len()
        );
        let mut row_major = Vec::with_capacity(d * h);
        for i in 0..d {
            for j in 0..h {
                row_major.push(e_feat[(i, j)]);
            }
        }
        let e = Tensor::from_vec(row_major, (d, h), dev)?.detach();
        let b = Tensor::from_slice(b_feat, d, dev)?.detach();
        Ok(Self {
            e_feat: e,
            b_feat: b,
            h,
        })
    }
}

/// Install `data` as a fresh `Var` named `name` in `varmap` and return
/// its underlying Tensor. Used to seed an encoder's `feature_embeddings`
/// slot with pre-trained values so the `VarMap → safetensors` round-trip
/// preserves them. The caller must also pass `name` to [`trainable_vars`]
/// when constructing AdamW, otherwise the optimizer will treat it as a
/// normal trainable parameter.
pub fn install_frozen_var_2d(
    varmap: &VarMap,
    name: &str,
    data: &DMatrix<f32>,
    dev: &Device,
) -> Result<Tensor> {
    let d = data.nrows();
    let h = data.ncols();
    let mut row_major = Vec::with_capacity(d * h);
    for i in 0..d {
        for j in 0..h {
            row_major.push(data[(i, j)]);
        }
    }
    let var = Var::from_tensor(&Tensor::from_vec(row_major, (d, h), dev)?)?;
    {
        let mut tbl = varmap.data().lock().unwrap();
        tbl.insert(name.to_string(), var.clone());
    }
    Ok(var.as_tensor().clone())
}

/// Overwrite the data of an EXISTING `Var` registered in `varmap` under
/// `fully_qualified_name`. Used by topic-model freeze flows where the
/// encoder constructor already created a randn-initialized `Var` for
/// `feature.embeddings` via `vb.get_with_hints`; we then replace its
/// contents with the frozen tensor without changing the Var's identity,
/// so any `Tensor` reference the encoder already cloned out of the
/// VarBuilder stays valid (Var has interior mutability — same underlying
/// storage).
pub fn overwrite_var_2d(
    varmap: &VarMap,
    fully_qualified_name: &str,
    data: &DMatrix<f32>,
    dev: &Device,
) -> Result<()> {
    let var = {
        let tbl = varmap.data().lock().unwrap();
        tbl.get(fully_qualified_name).cloned().ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "overwrite_var_2d: var '{fully_qualified_name}' not found in VarMap \
                 (have: {:?})",
                tbl.keys().collect::<Vec<_>>()
            ))
        })?
    };
    let d = data.nrows();
    let h = data.ncols();
    {
        let existing = var.as_tensor();
        let dims = existing.dims();
        if dims != [d, h] {
            return Err(candle_core::Error::Msg(format!(
                "overwrite_var_2d: shape mismatch for '{fully_qualified_name}' \
                 (existing {dims:?}, new [{d}, {h}])"
            )));
        }
    }
    let mut row_major = Vec::with_capacity(d * h);
    for i in 0..d {
        for j in 0..h {
            row_major.push(data[(i, j)]);
        }
    }
    let t = Tensor::from_vec(row_major, (d, h), dev)?;
    var.set(&t)?;
    Ok(())
}

/// Every `Var` in `varmap` whose name is NOT in `frozen_names`. Pass the
/// result to `AdamW::new(...)` instead of `varmap.all_vars()` to keep
/// the named slots out of the optimizer.
///
/// Var name semantics follow candle's `VarBuilder` prefix path: a Var
/// registered under prefix `enc` for slot `feature.embeddings` ends up
/// as `enc.feature.embeddings` in the VarMap.
pub fn trainable_vars(varmap: &VarMap, frozen_names: &[&str]) -> Vec<Var> {
    let frozen: FxHashSet<&str> = frozen_names.iter().copied().collect();
    let tbl = varmap.data().lock().unwrap();
    tbl.iter()
        .filter(|(name, _)| !frozen.contains(name.as_str()))
        .map(|(_, v)| v.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    fn from_parts_roundtrips_row_major() {
        let dev = dev();
        // 3 rows × 2 cols
        let m = DMatrix::<f32>::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias = vec![0.1, 0.2, 0.3];
        let frozen = FrozenFeatureSide::from_parts(&m, &bias, &dev).unwrap();
        assert_eq!(frozen.h, 2);
        assert_eq!(frozen.e_feat.dims(), &[3, 2]);
        let flat: Vec<f32> = frozen.e_feat.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b: Vec<f32> = frozen.b_feat.to_vec1().unwrap();
        assert_eq!(b, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn install_then_trainable_partition() {
        let dev = dev();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);

        // Two regular Vars
        let _w1 = vb
            .get_with_hints((2, 3), "layer1.weight", candle_nn::Init::Const(0.0))
            .unwrap();
        let _w2 = vb
            .get_with_hints((3, 4), "layer2.weight", candle_nn::Init::Const(0.0))
            .unwrap();

        // One installed frozen Var
        let m = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let installed = install_frozen_var_2d(&varmap, "feature.embeddings", &m, &dev).unwrap();
        let flat: Vec<f32> = installed.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);

        // VarMap holds all 3
        assert_eq!(varmap.all_vars().len(), 3);

        // trainable_vars excludes the frozen one
        let trainable = trainable_vars(&varmap, &["feature.embeddings"]);
        assert_eq!(trainable.len(), 2);
    }

    #[test]
    fn overwrite_var_propagates_to_existing_tensor_reference() {
        let dev = dev();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        // Encoder-like: create a Var and keep the Tensor it returns.
        let t_ref = vb
            .get_with_hints((2, 2), "feature.embeddings", candle_nn::Init::Const(0.0))
            .unwrap();
        // Initial values are zeros.
        let v0: Vec<f32> = t_ref.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v0, vec![0.0, 0.0, 0.0, 0.0]);

        // Overwrite via the helper.
        let m = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        overwrite_var_2d(&varmap, "feature.embeddings", &m, &dev).unwrap();

        // The Tensor reference the "encoder" holds now sees the new data
        // (interior mutability via Var → same underlying storage).
        let v1: Vec<f32> = t_ref.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn overwrite_var_missing_name_errors() {
        let dev = dev();
        let varmap = VarMap::new();
        let m = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let err = overwrite_var_2d(&varmap, "missing", &m, &dev).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn trainable_vars_no_frozen_returns_all() {
        let dev = dev();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        let _w = vb
            .get_with_hints((2, 2), "w", candle_nn::Init::Const(0.0))
            .unwrap();
        let trainable = trainable_vars(&varmap, &[]);
        assert_eq!(trainable.len(), 1);
    }
}
