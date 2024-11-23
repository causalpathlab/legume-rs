use clap::{Args, Parser, Subcommand, ValueEnum};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
// use ndarray_stats;

fn scale_columns(mut X: Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let mu = X.mean_axis(Axis(0)).ok_or(anyhow::anyhow!("mean failed"))?;
    let sig = X.std_axis(Axis(0), 0.0);
    for j in 0..X.ncols() {
        X.column_mut(j).mapv_inplace(|x| (x - mu[j]) / sig[j]);
    }
    Ok(X)
}

#[derive(Args)]
pub struct SimulateArgs {
    /// number of rows
    #[arg(short, long)]
    rows: usize,
    /// number of columns
    #[arg(short, long)]
    cols: usize,
    /// number of factors
    #[arg(short, long)]
    factors: Option<usize>,
    /// number of batches
    #[arg(short, long)]
    batches: Option<usize>,
}

#[allow(dead_code)]
pub fn generate_factored_gamma_data(args: SimulateArgs) -> anyhow::Result<()> {
    use ndarray_rand::rand_distr::Gamma;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::rand_distr::Uniform;
    // use ndarray_rand::rand_distr::unifor

    let N = args.cols;
    let D = args.rows;
    let K = args.factors.unwrap_or(1);
    let B = args.batches.unwrap_or(1);

    // 1. batch membership matrix
    let batch_membership: Array1<usize> = Array1::random(N, Uniform::new(0, B));
    let mut X: Array2<f32> = Array2::zeros((B, N));

    for i in 0..N {
        let k = batch_membership[i];
        X[[k, i]] = 1.0;
    }

    // 2. batch effect matrix
    let ln_delta = scale_columns(Array2::random((D, B), StandardNormal).dot(&X))?;

    // 3. factorization model
    let beta: Array2<f32> = Array2::random((D, K), Gamma::<f32>::new((D * K) as f32, 1.0)?);
    let theta: Array2<f32> = Array2::random((K, N), Gamma::<f32>::new((K * N) as f32, 1.0)?);

    // Gamma::<f32>::new((D * K) as f32, 1.0));

    // Gamma::

    // rand::distributions::

    // let A = Array2::random((args.rows, args.cols), Uniform::new(-1.0, 1.0));
    // let B = Array2::random((args.cols, args.cols), Uniform::new(-1.0, 1.0));
    // (A, B)
    Ok(())
}
