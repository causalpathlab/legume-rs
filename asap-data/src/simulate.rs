use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::common_io::write_lines;
use matrix_util::dmatrix_io::*;
use matrix_util::mtx_io::write_mtx_triplets;
use matrix_util::traits::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, Uniform};
use rayon::prelude::*;

pub struct SimArgs {
    pub rows: usize,
    pub cols: usize,
    pub depth: usize,
    pub factors: usize,
    pub batches: usize,
    pub overdisp: f32,
    pub rseed: u64,
}

pub struct SimOut {
    pub ln_delta_db: DMatrix<f32>,
    pub beta_dk: DMatrix<f32>,
    pub theta_kn: DMatrix<f32>,
    pub batch_membership: Vec<usize>,
    pub triplets: Vec<(u64, u64, f32)>,
}

#[allow(dead_code)]
/// Generate a simulated dataset with a factored gamma model
/// * `args`: SimulateArgs
/// * `mtx_file`: output data mtx file (.gz recommended)
/// * `dict_file`: true dictionary file
/// * `prop_file`: true proportion file
/// * `ln_batch_file`: log batch effect file
/// * `batch_file`: true batch membership file
///
/// ```text
/// Y(i,j) ~ Poisson( delta(i, B(j)) * sum_k beta(i,k) * theta(k,j) )
/// ```
///
pub fn generate_factored_poisson_gamma_data_mtx(
    args: &SimArgs,
    mtx_file: &str,
    dict_file: &str,
    prop_file: &str,
    ln_batch_file: &str,
    batch_file: &str,
) -> anyhow::Result<()> {
    let sim = generate_factored_poisson_gamma_data(args);

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, batch_file)?;
    info!("batch membership: {:?}", &batch_file);

    sim.ln_delta_db.to_tsv(&ln_batch_file)?;
    sim.theta_kn.transpose().to_tsv(&prop_file)?;
    sim.beta_dk.to_tsv(&dict_file)?;

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        &ln_batch_file, &dict_file, &prop_file
    );

    let mut triplets = sim.triplets;

    info!(
        "sampled Poisson data with {} non-zero elements",
        triplets.len()
    );

    info!("sorting these triplets...");
    triplets.sort_by_key(|&(row, _, _)| row);
    triplets.sort_by_key(|&(_, col, _)| col);

    info!("writing them down to {}", mtx_file);

    let nn = args.cols;
    let dd = args.rows;
    write_mtx_triplets(&triplets, dd, nn, &mtx_file)?;
    Ok(())
}

#[allow(dead_code)]
/// Generate a simulated dataset with a factored gamma model
/// * `args`: SimulateArgs
/// * `mtx_file`: output data mtx file (.gz recommended)
/// * `dict_file`: true dictionary file
/// * `prop_file`: true proportion file
/// * `ln_batch_file`: log batch effect file
/// * `batch_file`: true batch membership file
///
/// ```text
/// Y(i,j) ~ Poisson( delta(i, B(j)) * sum_k beta(i,k) * theta(k,j) )
/// ```
///
pub fn generate_factored_poisson_gamma_data(args: &SimArgs) -> SimOut {
    let nn = args.cols;
    let dd = args.rows;
    let kk = args.factors;
    let bb = args.batches;
    let nnz = args.depth;
    let rseed = args.rseed;
    let overdisp = args.overdisp;

    let threshold = 0.5_f32;

    let mut rng = rand::rngs::StdRng::seed_from_u64(rseed);

    // 1. batch membership matrix
    let runif = Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    // 2. batch effect matrix
    let mut ln_delta_db = DMatrix::<f32>::rnorm(dd, bb);
    ln_delta_db.scale_columns_inplace();
    info!("simulated batch effects");

    // 3. factorization model
    let (a, b) = (1. / overdisp, (dd as f32).sqrt() * overdisp);
    let beta_dk = DMatrix::<f32>::rgamma(dd, kk, (a, b));

    let (a, b) = (1. / overdisp, (nn as f32).sqrt() * overdisp);
    let theta_kn = DMatrix::<f32>::rgamma(kk, nn, (a, b));

    // 4. putting them all together
    // let mut triplets = vec![];
    let delta_db = ln_delta_db.map(|x| x.exp());

    let triplets = theta_kn
        .column_iter()
        .enumerate()
        .par_bridge()
        .progress_count(nn as u64)
        .map(|(j, theta_j)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(rseed + j as u64);
            let b = batch_membership[j]; // batch index

            let lambda_j = if bb > 1 {
                (&beta_dk * &theta_j).component_mul(&delta_db.column(b))
            } else {
                &beta_dk * &theta_j
            };

            let tot = lambda_j.sum();
            let scale = (nnz as f32) / tot;

            lambda_j
                .iter()
                .enumerate()
                .filter_map(|(i, &l_ij)| {
                    let rpois = Poisson::new(l_ij * scale).expect("poisson sample error");
                    let y_ij = rpois.sample(&mut rng);
                    if y_ij > threshold {
                        Some((i as u64, j as u64, y_ij))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    info!(
        "sampled Poisson data with {} non-zero elements",
        triplets.len()
    );

    SimOut {
        ln_delta_db,
        beta_dk,
        theta_kn,
        batch_membership,
        triplets,
    }
}
