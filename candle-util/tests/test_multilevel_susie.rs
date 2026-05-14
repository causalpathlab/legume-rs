use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use matrix_util::traits::IoOps;

use candle_util::sgvb::BlackBoxLikelihood;
use candle_util::sgvb::{
    multilevel_loss, GaussianPrior, MultilevelParams, MultilevelPartitionParams,
    MultilevelSusieSGVB, SGVBConfig,
};

struct GaussianLik {
    y: Tensor,
}

impl BlackBoxLikelihood for GaussianLik {
    fn log_likelihood(&self, etas: &[&Tensor]) -> candle_core::Result<Tensor> {
        let eta = etas[0];
        let diff_sq = eta.broadcast_sub(&self.y)?.sqr()?;
        let log_prob = (diff_sq * (-0.5))?;
        log_prob.sum(2)?.sum(1)
    }
}

/// Test on the susie_example data (n=2490, p=2577, causal: 949, 1484, 1756)
#[test]
fn test_susie_example_multilevel() -> Result<()> {
    let device = Device::Cpu;

    let x_path = "temp/susie_example/x_scaled.tsv.gz";
    let y_path = "temp/susie_example/y.tsv";

    // Check files exist
    if !std::path::Path::new(x_path).exists() {
        eprintln!("Skipping: {} not found", x_path);
        return Ok(());
    }

    let x = Tensor::from_tsv(x_path, None)?
        .to_dtype(DType::F32)?
        .to_device(&device)?;
    let y = Tensor::from_tsv(y_path, None)?
        .to_dtype(DType::F32)?
        .to_device(&device)?;

    let (n, p) = x.dims2()?;
    let k = y.dim(1)?;
    println!("susie_example: n={}, p={}, k={}", n, p, k);

    let l = 5;
    let block_size = 100; // 2577 → 26 groups → terminal: 2 levels

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
    let config = SGVBConfig::new(30);

    let params = MultilevelParams {
        base: MultilevelPartitionParams {
            num_components: l,
            k,
            config,
            gate_epsilon: None,
            has_null: false,
        },
        block_size,
    };
    let model = MultilevelSusieSGVB::new(vb.pp("ml"), x, prior, params)?;

    println!(
        "Multilevel: {} levels, L={}, block_size={}",
        model.num_levels(),
        l,
        block_size
    );

    let likelihood = GaussianLik { y };
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 0.01)?;

    for i in 0..500 {
        let loss = multilevel_loss(&model, &likelihood, 30, 1.0)?;
        optimizer.backward_step(&loss)?;

        if i % 50 == 0 {
            let lv: f32 = loss.to_scalar()?;
            let pip = model.joint_pip()?;
            let pip_949: f32 = pip.get(949)?.get(0)?.to_scalar()?;
            let pip_1484: f32 = pip.get(1484)?.get(0)?.to_scalar()?;
            let pip_1756: f32 = pip.get(1756)?.get(0)?.to_scalar()?;
            println!(
                "iter {:4}: loss={:10.2}, PIP[949]={:.4}, PIP[1484]={:.4}, PIP[1756]={:.4}",
                i, lv, pip_949, pip_1484, pip_1756
            );
        }
    }

    let pip = model.joint_pip()?;
    let causal = [949, 1484, 1756];
    println!("\nFinal PIPs for causal SNPs:");
    for &idx in &causal {
        let pip_val: f32 = pip.get(idx)?.get(0)?.to_scalar()?;
        println!("  PIP[{}] = {:.4}", idx, pip_val);
    }

    // Mean PIP of non-causal
    let mut other_sum = 0.0f32;
    for j in 0..p {
        if !causal.contains(&j) {
            other_sum += pip.get(j)?.get(0)?.to_scalar::<f32>()?;
        }
    }
    let other_mean = other_sum / (p - causal.len()) as f32;
    println!("  Others mean = {:.6}", other_mean);

    // At least one causal SNP should be clearly above background
    let max_causal_pip = causal
        .iter()
        .map(|&j| {
            pip.get(j)
                .unwrap()
                .get(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        })
        .fold(0.0f32, f32::max);

    assert!(
        max_causal_pip > other_mean * 5.0,
        "Best causal PIP {} should be > 5x background {}",
        max_causal_pip,
        other_mean
    );

    Ok(())
}
