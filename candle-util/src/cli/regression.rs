use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use clap::{Args, ValueEnum};
use log::info;
use matrix_util::traits::IoOps;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::sgvb::{
    composite_direct_elbo_loss, composite_elbo, compute_elbo, direct_elbo_loss,
    samples_direct_elbo_loss, samples_elbo, sgvb_loss, CompositeModel, GaussianPrior,
    LinearModelSGVB, LinearRegressionSGVB, SGVBConfig, SgvbModel, SparseVariationalOutput,
    SusieVar, VariationalOutput,
};

use super::regression_likelihood::{
    GaussianLikelihood, NegativeBinomialLikelihood, PoissonLikelihood,
};

//
// Helper functions
//

fn get_format_ext(path: &Path) -> String {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let name_lower = name.to_lowercase();

    if name_lower.ends_with(".gz") {
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

fn load_tensor(path: &Path, device: &Device) -> Result<Tensor> {
    let path_str = path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
    let tensor = match get_format_ext(path).as_str() {
        "csv" => Tensor::from_csv(path_str, None)?,
        "parquet" | "pq" => Tensor::from_parquet(path_str)?.mat,
        _ => Tensor::from_tsv(path_str, None)?,
    };
    tensor.to_device(device).map_err(Into::into)
}

fn load_names(path: &Option<PathBuf>) -> Result<Option<Vec<Box<str>>>> {
    match path {
        Some(p) => {
            let file = File::open(p)?;
            let reader = BufReader::new(file);
            let names: Vec<Box<str>> = reader
                .lines()
                .map(|l| l.map(|s| s.into_boxed_str()))
                .collect::<std::io::Result<_>>()?;
            info!("Loaded {} names from {:?}", names.len(), p);
            Ok(Some(names))
        }
        None => Ok(None),
    }
}

fn load_x_var(path: &Option<PathBuf>, n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    if let Some(ref p) = path {
        info!("Loading X_var from {:?}", p);
        let t = load_tensor(p, device)?;
        info!("  X_var shape: {:?}", t.dims());
        if t.dim(0)? != n {
            anyhow::bail!("X_var must have same number of rows as X");
        }
        Ok(t)
    } else {
        info!("Using intercept-only for variance");
        Ok(Tensor::ones((n, 1), dtype, device)?)
    }
}

fn top_k_indices(tensor: &Tensor, k: usize) -> Result<Vec<(usize, f32)>> {
    let vec: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(indexed.into_iter().take(k).collect())
}

fn format_top_pips(pip: &Tensor, k: usize) -> Result<String> {
    let top = top_k_indices(pip, k)?;
    Ok(top.iter().map(|(i, v)| format!("{}:{:.3}", i, v)).collect::<Vec<_>>().join(", "))
}

fn log_top_pips(pip: &Tensor, n: usize) -> Result<()> {
    info!("Top {} features by PIP:", n);
    for (rank, (idx, val)) in top_k_indices(pip, n)?.iter().enumerate() {
        info!("  {:2}. feature {:4}: PIP = {:.4}", rank + 1, idx, val);
    }
    Ok(())
}

fn save_output<V: VariationalOutput>(
    var: &V,
    path: &Option<PathBuf>,
    label: &str,
    row_names: Option<&[Box<str>]>,
    col_names: Option<&[Box<str>]>,
) -> Result<()> {
    if let Some(ref p) = path {
        let s = p.to_str().ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;
        var.to_parquet(row_names, col_names, s)?;
        info!("Saved {} variational to {:?}", label, p);
    }
    Ok(())
}

fn save_output_sparse<V: SparseVariationalOutput>(
    var: &V,
    path: &Option<PathBuf>,
    label: &str,
    row_names: Option<&[Box<str>]>,
    col_names: Option<&[Box<str>]>,
) -> Result<()> {
    if let Some(ref p) = path {
        let s = p.to_str().ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;
        var.to_parquet_sparse(row_names, col_names, s)?;
        info!("Saved {} variational to {:?}", label, p);
    }
    Ok(())
}

//
// Generic training loop
//

fn train<L, E, P>(
    iters: usize,
    verbose: bool,
    optimizer: &mut impl Optimizer,
    loss_fn: L,
    elbo_fn: E,
    pip_fn: Option<P>,
) -> Result<()>
where
    L: Fn() -> Result<Tensor>,
    E: Fn() -> Result<Tensor>,
    P: Fn() -> Result<String>,
{
    info!("Training for {} iterations", iters);
    for i in 0..iters {
        let loss = loss_fn()?;
        optimizer.backward_step(&loss)?;

        if i % 50 == 0 || i == iters - 1 {
            if verbose {
                let elbo = elbo_fn()?;
                match &pip_fn {
                    Some(f) => info!(
                        "iter {:4}: loss = {:10.4}, ELBO = {:10.4}, PIPs: {}",
                        i, loss.to_scalar::<f32>()?, elbo.to_scalar::<f32>()?, f()?
                    ),
                    None => info!(
                        "iter {:4}: loss = {:10.4}, ELBO = {:10.4}",
                        i, loss.to_scalar::<f32>()?, elbo.to_scalar::<f32>()?
                    ),
                }
            } else {
                info!("iter {:4}/{}", i, iters);
            }
        }
    }
    Ok(())
}

//
// CLI types
//

#[derive(Clone, Debug, ValueEnum)]
pub enum LikelihoodType {
    /// Gaussian: y ~ N(X*β, exp(X_var*γ))
    Gaussian,
    /// Poisson: y ~ Poisson(exp(X*β))
    Poisson,
    /// Negative binomial: y ~ NB(exp(X*β), exp(X_var*γ))
    Negbin,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum VariationalType {
    Gaussian,
    Susie,
}

#[derive(Args, Debug)]
pub struct RegressionArgs {
    #[arg(short, long)]
    pub x: PathBuf,

    #[arg(short, long)]
    pub y: PathBuf,

    #[arg(long)]
    pub x_var: Option<PathBuf>,

    #[arg(long, help = "File with feature names (one per line)")]
    pub feature_names: Option<PathBuf>,

    #[arg(long, help = "File with output names (one per line)")]
    pub output_names: Option<PathBuf>,

    #[arg(short, long, default_value = "gaussian")]
    pub model: LikelihoodType,

    #[arg(short, long, default_value = "gaussian")]
    pub prior: VariationalType,

    #[arg(long, default_value = "5", help = "Number of Susie layers (L)")]
    pub susie_layers: usize,

    #[arg(long, default_value = "500")]
    pub iters: usize,

    #[arg(long, default_value = "0.01")]
    pub lr: f64,

    #[arg(long, default_value = "50")]
    pub samples: usize,

    #[arg(short, long)]
    pub output: Option<PathBuf>,

    #[arg(long)]
    pub output_var: Option<PathBuf>,

    #[arg(long)]
    pub gpu: bool,

    #[arg(short, long)]
    pub verbose: bool,
}

//
// Model runners
//

fn run_gaussian_gaussian(
    args: &RegressionArgs,
    x: Tensor,
    y: Tensor,
    vb: VarBuilder,
    config: SGVBConfig,
    optimizer: &mut impl Optimizer,
    feature_names: Option<&[Box<str>]>,
    output_names: Option<&[Box<str>]>,
) -> Result<()> {
    let (n, p, k) = (x.dim(0)?, x.dim(1)?, y.dim(1)?);
    let x_var = load_x_var(&args.x_var, n, DType::F32, x.device())?;
    let p_var = x_var.dim(1)?;

    let model_mean = LinearRegressionSGVB::new(
        vb.pp("mean"), x, k,
        GaussianPrior::new(vb.pp("prior_mean"), 1.0)?,
        config.clone(),
    )?;
    let model_var = LinearRegressionSGVB::new(
        vb.pp("var"), x_var, k,
        GaussianPrior::new(vb.pp("prior_var"), 1.0)?,
        config.clone(),
    )?;
    let composite = CompositeModel::new(vec![model_mean, model_var]);
    let likelihood = GaussianLikelihood::new(y);

    info!("Mean module: {} features", p);
    info!("Variance module: {} features", p_var);

    train(
        args.iters,
        args.verbose,
        optimizer,
        || Ok(composite_direct_elbo_loss(&composite, &likelihood, config.num_samples)?),
        || Ok(composite_elbo(&composite, &likelihood, 100)?),
        None::<fn() -> Result<String>>,
    )?;

    info!("Mean coefficients shape: {:?}", composite.modules[0].coef_mean()?.dims());
    info!("Variance coefficients shape: {:?}", composite.modules[1].coef_mean()?.dims());

    save_output(&composite.modules[0].variational, &args.output, "mean", feature_names, output_names)?;
    save_output(&composite.modules[1].variational, &args.output_var, "variance", None, output_names)
}

fn run_gaussian_susie(
    args: &RegressionArgs,
    x: Tensor,
    y: Tensor,
    vb: VarBuilder,
    config: SGVBConfig,
    optimizer: &mut impl Optimizer,
    feature_names: Option<&[Box<str>]>,
    output_names: Option<&[Box<str>]>,
) -> Result<()> {
    let (n, p, k) = (x.dim(0)?, x.dim(1)?, y.dim(1)?);
    let x_var = load_x_var(&args.x_var, n, DType::F32, x.device())?;
    let p_var = x_var.dim(1)?;

    let susie = SusieVar::new(vb.pp("susie_mean"), args.susie_layers, p, k)?;
    let model_mean = LinearModelSGVB::from_variational(
        susie, x,
        GaussianPrior::new(vb.pp("prior_mean"), 1.0)?,
        config.clone(),
    );
    let model_var = LinearRegressionSGVB::new(
        vb.pp("var"), x_var, k,
        GaussianPrior::new(vb.pp("prior_var"), 1.0)?,
        config.clone(),
    )?;
    let likelihood = GaussianLikelihood::new(y);

    info!("Mean module: {} features, Susie(L={})", p, args.susie_layers);
    info!("Variance module: {} features, Gaussian", p_var);

    train(
        args.iters,
        args.verbose,
        optimizer,
        || {
            let samples = vec![
                model_mean.sample(config.num_samples)?,
                model_var.sample(config.num_samples)?,
            ];
            Ok(samples_direct_elbo_loss(&samples, &likelihood)?)
        },
        || {
            let samples = vec![model_mean.sample(100)?, model_var.sample(100)?];
            Ok(samples_elbo(&samples, &likelihood)?)
        },
        Some(|| Ok(format_top_pips(&model_mean.variational.pip()?, 3)?)),
    )?;

    info!("Mean coefficients shape: {:?}", model_mean.coef_mean()?.dims());
    info!("Variance coefficients shape: {:?}", model_var.coef_mean()?.dims());
    log_top_pips(&model_mean.variational.pip()?, 10)?;

    save_output_sparse(&model_mean.variational, &args.output, "mean", feature_names, output_names)?;
    save_output(&model_var.variational, &args.output_var, "variance", None, output_names)
}

fn run_poisson_gaussian(
    args: &RegressionArgs,
    x: Tensor,
    y: Tensor,
    vb: VarBuilder,
    config: SGVBConfig,
    optimizer: &mut impl Optimizer,
    feature_names: Option<&[Box<str>]>,
    output_names: Option<&[Box<str>]>,
) -> Result<()> {
    let k = y.dim(1)?;
    let model = LinearRegressionSGVB::new(
        vb.pp("model"), x, k,
        GaussianPrior::new(vb.pp("prior"), 1.0)?,
        config.clone(),
    )?;
    let likelihood = PoissonLikelihood::new(y);

    train(
        args.iters,
        args.verbose,
        optimizer,
        || Ok(sgvb_loss(&model, &likelihood, &config)?),
        || Ok(compute_elbo(&model, &likelihood, 100)?),
        None::<fn() -> Result<String>>,
    )?;

    info!("Coefficients shape: {:?}", model.coef_mean()?.dims());
    save_output(&model.variational, &args.output, "", feature_names, output_names)
}

fn run_poisson_susie(
    args: &RegressionArgs,
    x: Tensor,
    y: Tensor,
    vb: VarBuilder,
    config: SGVBConfig,
    optimizer: &mut impl Optimizer,
    feature_names: Option<&[Box<str>]>,
    output_names: Option<&[Box<str>]>,
) -> Result<()> {
    let (p, k) = (x.dim(1)?, y.dim(1)?);
    let susie = SusieVar::new(vb.pp("susie"), args.susie_layers, p, k)?;
    let model = LinearModelSGVB::from_variational(
        susie, x,
        GaussianPrior::new(vb.pp("prior"), 1.0)?,
        config.clone(),
    );
    let likelihood = PoissonLikelihood::new(y);

    train(
        args.iters,
        args.verbose,
        optimizer,
        || Ok(direct_elbo_loss(&model, &likelihood, config.num_samples)?),
        || Ok(compute_elbo(&model, &likelihood, 100)?),
        Some(|| Ok(format_top_pips(&model.variational.pip()?, 3)?)),
    )?;

    info!("Coefficients shape: {:?}", model.coef_mean()?.dims());
    log_top_pips(&model.variational.pip()?, 10)?;
    save_output_sparse(&model.variational, &args.output, "", feature_names, output_names)
}

fn run_negbin_gaussian(
    args: &RegressionArgs,
    x: Tensor,
    y: Tensor,
    vb: VarBuilder,
    config: SGVBConfig,
    optimizer: &mut impl Optimizer,
    feature_names: Option<&[Box<str>]>,
    output_names: Option<&[Box<str>]>,
) -> Result<()> {
    let (n, p, k) = (x.dim(0)?, x.dim(1)?, y.dim(1)?);
    let x_var = load_x_var(&args.x_var, n, DType::F32, x.device())?;
    let p_var = x_var.dim(1)?;

    let model_mean = LinearRegressionSGVB::new(
        vb.pp("mean"), x, k,
        GaussianPrior::new(vb.pp("prior_mean"), 1.0)?,
        config.clone(),
    )?;
    let model_disp = LinearRegressionSGVB::new(
        vb.pp("disp"), x_var, k,
        GaussianPrior::new(vb.pp("prior_disp"), 1.0)?,
        config.clone(),
    )?;
    let composite = CompositeModel::new(vec![model_mean, model_disp]);
    let likelihood = NegativeBinomialLikelihood::new(y);

    info!("Mean module: {} features", p);
    info!("Dispersion module: {} features", p_var);

    train(
        args.iters,
        args.verbose,
        optimizer,
        || Ok(composite_direct_elbo_loss(&composite, &likelihood, config.num_samples)?),
        || Ok(composite_elbo(&composite, &likelihood, 100)?),
        None::<fn() -> Result<String>>,
    )?;

    info!("Mean coefficients shape: {:?}", composite.modules[0].coef_mean()?.dims());
    info!("Dispersion coefficients shape: {:?}", composite.modules[1].coef_mean()?.dims());

    save_output(&composite.modules[0].variational, &args.output, "mean", feature_names, output_names)?;
    save_output(&composite.modules[1].variational, &args.output_var, "dispersion", None, output_names)
}

fn run_negbin_susie(
    args: &RegressionArgs,
    x: Tensor,
    y: Tensor,
    vb: VarBuilder,
    config: SGVBConfig,
    optimizer: &mut impl Optimizer,
    feature_names: Option<&[Box<str>]>,
    output_names: Option<&[Box<str>]>,
) -> Result<()> {
    let (n, p, k) = (x.dim(0)?, x.dim(1)?, y.dim(1)?);
    let x_var = load_x_var(&args.x_var, n, DType::F32, x.device())?;
    let p_var = x_var.dim(1)?;

    let susie = SusieVar::new(vb.pp("susie_mean"), args.susie_layers, p, k)?;
    let model_mean = LinearModelSGVB::from_variational(
        susie, x,
        GaussianPrior::new(vb.pp("prior_mean"), 1.0)?,
        config.clone(),
    );
    let model_disp = LinearRegressionSGVB::new(
        vb.pp("disp"), x_var, k,
        GaussianPrior::new(vb.pp("prior_disp"), 1.0)?,
        config.clone(),
    )?;
    let likelihood = NegativeBinomialLikelihood::new(y);

    info!("Mean module: {} features, Susie(L={})", p, args.susie_layers);
    info!("Dispersion module: {} features, Gaussian", p_var);

    train(
        args.iters,
        args.verbose,
        optimizer,
        || {
            let samples = vec![
                model_mean.sample(config.num_samples)?,
                model_disp.sample(config.num_samples)?,
            ];
            Ok(samples_direct_elbo_loss(&samples, &likelihood)?)
        },
        || {
            let samples = vec![model_mean.sample(100)?, model_disp.sample(100)?];
            Ok(samples_elbo(&samples, &likelihood)?)
        },
        Some(|| Ok(format_top_pips(&model_mean.variational.pip()?, 3)?)),
    )?;

    info!("Mean coefficients shape: {:?}", model_mean.coef_mean()?.dims());
    info!("Dispersion coefficients shape: {:?}", model_disp.coef_mean()?.dims());
    log_top_pips(&model_mean.variational.pip()?, 10)?;

    save_output_sparse(&model_mean.variational, &args.output, "mean", feature_names, output_names)?;
    save_output(&model_disp.variational, &args.output_var, "dispersion", None, output_names)
}

//
// Main entry point
//

pub fn run(args: &RegressionArgs) -> Result<()> {
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
    info!("Using device: {:?}", device);

    info!("Loading X from {:?}", args.x);
    let x = load_tensor(&args.x, &device)?;
    info!("  X shape: {:?}", x.dims());

    info!("Loading Y from {:?}", args.y);
    let y = load_tensor(&args.y, &device)?;
    info!("  Y shape: {:?}", y.dims());

    if y.dim(0)? != x.dim(0)? {
        anyhow::bail!("X and Y must have same number of rows");
    }

    let feature_names = load_names(&args.feature_names)?;
    let output_names = load_names(&args.output_names)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = SGVBConfig::new(args.samples);
    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), args.lr)?;

    info!("Likelihood: {}", match args.model {
        LikelihoodType::Gaussian => "Gaussian",
        LikelihoodType::Poisson => "Poisson",
        LikelihoodType::Negbin => "Negative Binomial",
    });
    info!("Variational: {}", match args.prior {
        VariationalType::Gaussian => "Gaussian".to_string(),
        VariationalType::Susie => format!("Susie(L={})", args.susie_layers),
    });

    let feat_ref = feature_names.as_deref();
    let out_ref = output_names.as_deref();

    match (&args.model, &args.prior) {
        (LikelihoodType::Gaussian, VariationalType::Gaussian) =>
            run_gaussian_gaussian(args, x, y, vb, config, &mut optimizer, feat_ref, out_ref),
        (LikelihoodType::Gaussian, VariationalType::Susie) =>
            run_gaussian_susie(args, x, y, vb, config, &mut optimizer, feat_ref, out_ref),
        (LikelihoodType::Poisson, VariationalType::Gaussian) =>
            run_poisson_gaussian(args, x, y, vb, config, &mut optimizer, feat_ref, out_ref),
        (LikelihoodType::Poisson, VariationalType::Susie) =>
            run_poisson_susie(args, x, y, vb, config, &mut optimizer, feat_ref, out_ref),
        (LikelihoodType::Negbin, VariationalType::Gaussian) =>
            run_negbin_gaussian(args, x, y, vb, config, &mut optimizer, feat_ref, out_ref),
        (LikelihoodType::Negbin, VariationalType::Susie) =>
            run_negbin_susie(args, x, y, vb, config, &mut optimizer, feat_ref, out_ref),
    }
}
