use clap::{Args, Parser, Subcommand, ValueEnum};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
// use ndarray_stats;

#[allow(dead_code)]
/// column-wise standardization
/// * X: (D, N) matrix
pub fn scale_columns(mut X: Array2<f32>) -> anyhow::Result<Array2<f32>> {
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
    pub rows: usize,
    /// number of columns
    #[arg(short, long)]
    pub cols: usize,
    /// number of factors
    #[arg(short, long)]
    pub factors: Option<usize>,
    /// number of batches
    #[arg(short, long)]
    pub batches: Option<usize>,
}

#[allow(dead_code)]
pub fn generate_factored_gamma_data(args: SimulateArgs) -> anyhow::Result<()> {
    use ndarray_rand::rand_distr::Gamma;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::rand_distr::Uniform;
    // use ndarray_rand::rand_distr::unifor

    let nn = args.cols;
    let dd = args.rows;
    let kk = args.factors.unwrap_or(1);
    let bb = args.batches.unwrap_or(1);

    // 1. batch membership matrix
    let batch_membership: Array1<usize> = Array1::random(nn, Uniform::new(0, bb));

    // let mut X: Array2<f32> = Array2::zeros((B, N));
    // for i in 0..N {
    //     let k = batch_membership[i];
    //     X[[k, i]] = 1.0;
    // }

    // 2. batch effect matrix
    let ln_delta = Array2::random((dd, bb), StandardNormal);

    // 3. factorization model
    let beta: Array2<f32> = Array2::random((dd, kk), Gamma::<f32>::new((dd * kk) as f32, 1.0)?);
    let theta: Array2<f32> = Array2::random((kk, nn), Gamma::<f32>::new((kk * nn) as f32, 1.0)?);

    dbg!(beta);
    dbg!(theta);

    for j in 0..nn {
        // generate
        // a. beta * theta[,j]
        // b. ln_delta[,batch_membership[j]]
        // c. extract triplets
    }

    // let ln_delta = ln_delta.dot(&X);
    // let ln_delta = scale_columns(ln_delta)?;

    // Gamma::<f32>::new((D * K) as f32, 1.0));

    // Gamma::

    // rand::distributions::

    // let A = Array2::random((args.rows, args.cols), Uniform::new(-1.0, 1.0));
    // let B = Array2::random((args.cols, args.cols), Uniform::new(-1.0, 1.0));
    // (A, B)
    Ok(())
}
