use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use clap::{Args, ValueEnum};
use matrix_util::traits::IoOps;
use std::path::{Path, PathBuf};

use crate::sgvb::{
    direct_elbo_loss, compute_elbo, sgvb_loss,
    GaussianPrior, LinearModelSGVB, LinearRegressionSGVB,
    SGVBConfig, SparseVariationalOutput, SusieVar, VariationalOutput,
};

use super::regression_likelihood::{GaussianLikelihood, PoissonLikelihood};

/// Extract the format extension from a path, handling .gz compression.
/// e.g., "file.csv.gz" -> "csv", "file.tsv" -> "tsv", "file.parquet" -> "parquet"
fn get_format_ext(path: &Path) -> String {
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Check for .gz suffix and get the extension before it
    let name_lower = name.to_lowercase();
    if name_lower.ends_with(".gz") {
        // Strip .gz and get the next extension
        let stem = &name[..name.len() - 3];
        Path::new(stem)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
    } else {
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
    }
}

/// Load a tensor from file, detecting format from extension.
/// Supports: .tsv, .txt (tab), .csv (comma), .parquet, .pq
/// Also handles .gz compression (e.g., .csv.gz, .tsv.gz)
fn load_tensor(path: &Path, device: &Device) -> Result<Tensor> {
    let path_str = path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
    let ext = get_format_ext(path);

    let tensor = match ext.as_str() {
        "csv" => Tensor::from_csv(path_str, None)?,
        "parquet" | "pq" => Tensor::from_parquet(path_str)?.mat,
        _ => Tensor::from_tsv(path_str, None)?, // default: tsv/txt
    };

    tensor.to_device(device).map_err(Into::into)
}

#[derive(Clone, Debug, ValueEnum)]
pub enum LikelihoodType {
    Gaussian,
    Poisson,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum VariationalType {
    Gaussian,
    Susie,
}

#[derive(Args, Debug)]
pub struct RegressionArgs {
    /// Path to X matrix (supports .tsv, .csv, .parquet, and .gz variants)
    #[arg(short, long)]
    pub x: PathBuf,

    /// Path to Y matrix (supports .tsv, .csv, .parquet, and .gz variants)
    #[arg(short, long)]
    pub y: PathBuf,

    /// Likelihood model type
    #[arg(short, long, default_value = "gaussian")]
    pub model: LikelihoodType,

    /// Variational prior type
    #[arg(short, long, default_value = "gaussian")]
    pub prior: VariationalType,

    /// Number of Susie components (only for susie prior)
    #[arg(long, default_value = "5")]
    pub components: usize,

    /// Number of training iterations
    #[arg(long, default_value = "500")]
    pub iters: usize,

    /// Learning rate
    #[arg(long, default_value = "0.01")]
    pub lr: f64,

    /// Number of MC samples for gradient estimation
    #[arg(long, default_value = "50")]
    pub samples: usize,

    /// Observation noise sigma (for gaussian likelihood)
    #[arg(long, default_value = "1.0")]
    pub sigma: f64,

    /// Output path for coefficients (format detected from extension)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Use GPU if available
    #[arg(long)]
    pub gpu: bool,
}

pub fn run(args: &RegressionArgs) -> Result<()> {
    // Setup device
    let device = if args.gpu {
        #[cfg(target_os = "macos")]
        { Device::new_metal(0).unwrap_or(Device::Cpu) }
        #[cfg(target_os = "linux")]
        { Device::new_cuda(0).unwrap_or(Device::Cpu) }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        { Device::Cpu }
    } else {
        Device::Cpu
    };

    println!("Using device: {:?}", device);

    // Load data
    println!("Loading X from {:?}", args.x);
    let x_tensor = load_tensor(&args.x, &device)?;
    println!("  X shape: {:?}", x_tensor.dims());

    println!("Loading Y from {:?}", args.y);
    let y_tensor = load_tensor(&args.y, &device)?;
    println!("  Y shape: {:?}", y_tensor.dims());

    let n = x_tensor.dim(0)?;
    let p = x_tensor.dim(1)?;
    let k = y_tensor.dim(1)?;

    if y_tensor.dim(0)? != n {
        anyhow::bail!("X and Y must have same number of rows");
    }

    // Setup
    let dtype = DType::F32;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let prior = GaussianPrior::new(vb.pp("prior"), 1.0)?;
    let config = SGVBConfig::new(args.samples, true);

    let likelihood_name = match args.model {
        LikelihoodType::Gaussian => "Gaussian",
        LikelihoodType::Poisson => "Poisson",
    };

    let variational_name = match args.prior {
        VariationalType::Gaussian => "Gaussian".to_string(),
        VariationalType::Susie => format!("Susie(L={})", args.components),
    };

    println!("\nLikelihood: {}", likelihood_name);
    println!("Variational: {}", variational_name);

    // Training
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), args.lr)?;

    match (&args.model, &args.prior) {
        (LikelihoodType::Gaussian, VariationalType::Gaussian) => {
            let likelihood = GaussianLikelihood::new(y_tensor, args.sigma);
            let model = LinearRegressionSGVB::new(vb.pp("model"), x_tensor, k, prior, config.clone())?;

            println!("\nTraining...");
            for i in 0..args.iters {
                let loss = sgvb_loss(&model, &likelihood, &config)?;
                optimizer.backward_step(&loss)?;

                if i % 50 == 0 || i == args.iters - 1 {
                    let elbo = compute_elbo(&model, &likelihood, 100)?;
                    println!("  iter {:4}: loss = {:10.4}, ELBO = {:10.4}",
                             i, loss.to_scalar::<f32>()?, elbo.to_scalar::<f32>()?);
                }
            }

            let coef = model.coef_mean()?;
            println!("\nCoefficients shape: {:?}", coef.dims());

            if let Some(ref out_path) = args.output {
                let out_str = out_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;
                model.variational.to_parquet(None, None, out_str)?;
                println!("Saved variational to {:?}", out_path);
            }
        }
        (LikelihoodType::Gaussian, VariationalType::Susie) => {
            let likelihood = GaussianLikelihood::new(y_tensor, args.sigma);
            let susie = SusieVar::new(vb.pp("susie"), args.components, p, k)?;
            let model = LinearModelSGVB::from_variational(susie, x_tensor, prior, config.clone());

            println!("\nTraining...");
            for i in 0..args.iters {
                let loss = direct_elbo_loss(&model, &likelihood, config.num_samples)?;
                optimizer.backward_step(&loss)?;

                if i % 50 == 0 || i == args.iters - 1 {
                    let elbo = compute_elbo(&model, &likelihood, 100)?;
                    let pip = model.variational.pip()?;
                    let pip_vec: Vec<f32> = pip.flatten_all()?.to_vec1()?;
                    let mut indexed: Vec<(usize, f32)> = pip_vec.iter().cloned().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let top3: Vec<String> = indexed.iter().take(3)
                        .map(|(i, v)| format!("{}:{:.3}", i, v))
                        .collect();

                    println!("  iter {:4}: loss = {:10.4}, ELBO = {:10.4}, top PIPs: {}",
                             i, loss.to_scalar::<f32>()?, elbo.to_scalar::<f32>()?, top3.join(", "));
                }
            }

            let coef = model.coef_mean()?;
            let pip = model.variational.pip()?;

            println!("\nCoefficients shape: {:?}", coef.dims());

            let pip_vec: Vec<f32> = pip.flatten_all()?.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> = pip_vec.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\nTop 10 features by PIP:");
            for (i, (idx, pip_val)) in indexed.iter().take(10).enumerate() {
                println!("  {:2}. feature {:4}: PIP = {:.4}", i + 1, idx, pip_val);
            }

            if let Some(ref out_path) = args.output {
                let out_str = out_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;
                model.variational.to_parquet_sparse(None, None, out_str)?;
                println!("\nSaved variational to {:?}", out_path);
            }
        }
        (LikelihoodType::Poisson, VariationalType::Gaussian) => {
            let likelihood = PoissonLikelihood::new(y_tensor);
            let model = LinearRegressionSGVB::new(vb.pp("model"), x_tensor, k, prior, config.clone())?;

            println!("\nTraining...");
            for i in 0..args.iters {
                let loss = sgvb_loss(&model, &likelihood, &config)?;
                optimizer.backward_step(&loss)?;

                if i % 50 == 0 || i == args.iters - 1 {
                    let elbo = compute_elbo(&model, &likelihood, 100)?;
                    println!("  iter {:4}: loss = {:10.4}, ELBO = {:10.4}",
                             i, loss.to_scalar::<f32>()?, elbo.to_scalar::<f32>()?);
                }
            }

            let coef = model.coef_mean()?;
            println!("\nCoefficients shape: {:?}", coef.dims());

            if let Some(ref out_path) = args.output {
                let out_str = out_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;
                model.variational.to_parquet(None, None, out_str)?;
                println!("Saved variational to {:?}", out_path);
            }
        }
        (LikelihoodType::Poisson, VariationalType::Susie) => {
            let likelihood = PoissonLikelihood::new(y_tensor);
            let susie = SusieVar::new(vb.pp("susie"), args.components, p, k)?;
            let model = LinearModelSGVB::from_variational(susie, x_tensor, prior, config.clone());

            println!("\nTraining...");
            for i in 0..args.iters {
                let loss = direct_elbo_loss(&model, &likelihood, config.num_samples)?;
                optimizer.backward_step(&loss)?;

                if i % 50 == 0 || i == args.iters - 1 {
                    let elbo = compute_elbo(&model, &likelihood, 100)?;
                    let pip = model.variational.pip()?;
                    let pip_vec: Vec<f32> = pip.flatten_all()?.to_vec1()?;
                    let mut indexed: Vec<(usize, f32)> = pip_vec.iter().cloned().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let top3: Vec<String> = indexed.iter().take(3)
                        .map(|(i, v)| format!("{}:{:.3}", i, v))
                        .collect();

                    println!("  iter {:4}: loss = {:10.4}, ELBO = {:10.4}, top PIPs: {}",
                             i, loss.to_scalar::<f32>()?, elbo.to_scalar::<f32>()?, top3.join(", "));
                }
            }

            let coef = model.coef_mean()?;
            let pip = model.variational.pip()?;

            println!("\nCoefficients shape: {:?}", coef.dims());

            let pip_vec: Vec<f32> = pip.flatten_all()?.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> = pip_vec.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\nTop 10 features by PIP:");
            for (i, (idx, pip_val)) in indexed.iter().take(10).enumerate() {
                println!("  {:2}. feature {:4}: PIP = {:.4}", i + 1, idx, pip_val);
            }

            if let Some(ref out_path) = args.output {
                let out_str = out_path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;
                model.variational.to_parquet_sparse(None, None, out_str)?;
                println!("\nSaved variational to {:?}", out_path);
            }
        }
    }

    Ok(())
}
