//! Seeded initialisation of a `VarMap`'s linear weights.

use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use matrix_util::rand_util::name_seed;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Re-draw every var in `varmap` — uniform in `±1/√fan_in` for weights,
/// zero for anything named `*.bias` — each from its own name-keyed
/// sub-stream of `seed`, so a run replays and adding a var never shifts
/// another's draw. Vars for which `skip(name)` holds keep their current
/// values (a normalisation layer's affine and statistics, say). candle's
/// `VarBuilder` initialises from an unseeded stream, which is why this
/// exists.
pub fn seed_uniform_vars(varmap: &VarMap, seed: u64, skip: impl Fn(&str) -> bool) -> Result<()> {
    let tbl = varmap.data().lock().unwrap();
    for (name, var) in tbl.iter() {
        if skip(name) {
            continue;
        }
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let draw: Vec<f32> = if name.ends_with(".bias") {
            vec![0f32; n]
        } else {
            let bound = (1.0 / *dims.last().unwrap_or(&1) as f64).sqrt();
            let mut rng = StdRng::seed_from_u64(name_seed(seed, name));
            (0..n)
                .map(|_| ((rng.random::<f64>() * 2.0 - 1.0) * bound) as f32)
                .collect()
        };
        var.set(&Tensor::from_vec(draw, dims.as_slice(), var.device())?)?;
    }
    Ok(())
}
