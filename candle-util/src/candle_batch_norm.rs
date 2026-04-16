//! Custom BatchNorm whose running statistics live in the VarMap.
//!
//! Unlike `candle_nn::BatchNorm` which creates detached `Var` copies
//! (breaking device transfer via VarMap), this implementation retrieves
//! the VarMap-registered `Var`s after `vb.get_with_hints()` so that
//! moving VarMap tensors to another device automatically moves the
//! running stats too.

use candle_core::{DType, Result, Tensor, Var};
use candle_nn::{Init, ModuleT, VarBuilder, VarMap};

pub struct BatchNorm {
    running_mean: Var,
    running_var: Var,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f64,
    momentum: f64,
}

pub struct BatchNormConfig {
    pub eps: f64,
    pub remove_mean: bool,
    pub affine: bool,
    pub momentum: f64,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        }
    }
}

/// Create a BatchNorm layer whose running stats are the *same* Vars
/// that `vb` registers in the VarMap.
///
/// Requires the VarMap so we can retrieve the Vars after registration.
pub fn batch_norm(
    num_features: usize,
    config: BatchNormConfig,
    varmap: &VarMap,
    vb: VarBuilder,
) -> Result<BatchNorm> {
    // Register in VarMap via VarBuilder
    let _rm_tensor = vb.get_with_hints(num_features, "running_mean", Init::Const(0.))?;
    let _rv_tensor = vb.get_with_hints(num_features, "running_var", Init::Const(1.))?;

    // Retrieve the SAME Vars from VarMap (not detached copies)
    let prefix = vb.prefix();
    let data = varmap.data().lock().expect("VarMap lock");
    let rm_key = if prefix.is_empty() {
        "running_mean".into()
    } else {
        format!("{prefix}.running_mean")
    };
    let rv_key = if prefix.is_empty() {
        "running_var".into()
    } else {
        format!("{prefix}.running_var")
    };
    let rm_var = data
        .get(&rm_key)
        .expect("running_mean must be in VarMap")
        .clone();
    let rv_var = data
        .get(&rv_key)
        .expect("running_var must be in VarMap")
        .clone();
    drop(data);

    let (weight, bias) = if config.affine {
        let w = vb.get_with_hints(num_features, "weight", Init::Const(1.))?;
        let b = vb.get_with_hints(num_features, "bias", Init::Const(0.))?;
        (Some(w), Some(b))
    } else {
        (None, None)
    };

    Ok(BatchNorm {
        running_mean: rm_var,
        running_var: rv_var,
        weight,
        bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
        momentum: config.momentum,
    })
}

impl BatchNorm {
    fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        let num_features = self.running_mean.as_tensor().dim(0)?;
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        if x.rank() < 2 {
            candle_core::bail!(
                "batch-norm input must have at least two dimensions ({:?})",
                x.shape()
            );
        }
        if x.dim(1)? != num_features {
            candle_core::bail!(
                "batch-norm dim mismatch ({:?} vs {})",
                x.shape(),
                num_features
            );
        }
        let x = x.to_dtype(internal_dtype)?;
        let x = x.transpose(0, 1)?;
        let x_dims = x.dims().to_vec();
        let x = x.flatten_from(1)?.contiguous()?;

        let x = if self.remove_mean {
            let mean_x = x.mean_keepdim(1)?;
            let updated = ((self.running_mean.as_tensor() * (1.0 - self.momentum))?
                + (mean_x.flatten_all()? * self.momentum)?)?;
            self.running_mean.set(&updated)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };

        let norm_x = x.sqr()?.mean_keepdim(1)?;
        let updated_var = {
            let batch_size = x.dim(1)? as f64;
            let rw = 1.0 - self.momentum;
            let nw = self.momentum * batch_size / (batch_size - 1.0);
            ((self.running_var.as_tensor() * rw)? + (&norm_x.flatten_all()? * nw)?)?
        };
        self.running_var.set(&updated_var)?;

        let x = x
            .broadcast_div(&(norm_x + self.eps)?.sqrt()?)?
            .to_dtype(x_dtype)?;

        let x = match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let w = w.reshape(((), 1))?;
                let b = b.reshape(((), 1))?;
                x.broadcast_mul(&w)?.broadcast_add(&b)?
            }
            _ => x,
        };
        x.reshape(x_dims.as_slice())?.transpose(0, 1)
    }

    fn forward_eval(&self, x: &Tensor) -> Result<Tensor> {
        let target_shape: Vec<usize> = x
            .dims()
            .iter()
            .enumerate()
            .map(|(idx, v)| if idx == 1 { *v } else { 1 })
            .collect();

        let x = x
            .broadcast_sub(
                &self
                    .running_mean
                    .as_detached_tensor()
                    .reshape(target_shape.as_slice())?,
            )?
            .broadcast_div(
                &(self
                    .running_var
                    .as_detached_tensor()
                    .reshape(target_shape.as_slice())?
                    + self.eps)?
                    .sqrt()?,
            )?;

        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let w = w.reshape(target_shape.as_slice())?;
                let b = b.reshape(target_shape.as_slice())?;
                x.broadcast_mul(&w)?.broadcast_add(&b)
            }
            _ => Ok(x),
        }
    }
}

impl ModuleT for BatchNorm {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            self.forward_train(x)
        } else {
            self.forward_eval(x)
        }
    }
}
