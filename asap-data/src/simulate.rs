use candle_core::{Device, Tensor};
use matrix_util::common_io::write_lines;
use matrix_util::mtx_io::write_mtx_triplets;
use matrix_util::tensor_io;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

pub struct SimArgs {
    pub rows: usize,
    pub cols: usize,
    pub factors: Option<usize>,
    pub batches: Option<usize>,
    pub rseed: Option<u64>,
}

#[allow(dead_code)]
/// Generate a simulated dataset with a factored gamma model
/// * `args`: SimulateArgs
/// * `mtx_file`: output data mtx file (.gz recommended)
/// * `dict_file`: true dictionary file
/// * `prop_file`: true proportion file
/// * `ln_batch_file`: log batch effect file
/// * `memb_file`: true batch membership file
///
/// ```text
/// Y(i,j) ~ Poisson( delta(i, B(j)) * sum_k beta(i,k) * theta(k,j) )
/// ```
///
pub fn generate_factored_gamma_data_mtx(
    args: &SimArgs,
    mtx_file: &str,
    dict_file: &str,
    prop_file: &str,
    ln_batch_file: &str,
    memb_file: &str,
) -> anyhow::Result<()> {
    let nn = args.cols;
    let dd = args.rows;
    let kk = args.factors.unwrap_or(1);
    let bb = args.batches.unwrap_or(1);
    let rseed = args.rseed.unwrap_or(42);

    let mut rng = rand::rngs::StdRng::seed_from_u64(rseed);

    let threshold = 0.5_f32;

    // 1. batch membership matrix
    let runif = Uniform::new(0, bb);
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    let batch_out: Vec<Box<str>> = batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, memb_file)?;

    // 2. batch effect matrix
    let ln_delta = Tensor::randn(0_f32, 1_f32, (dd, bb), &Device::Cpu)?;
    let mu = ln_delta.mean_keepdim(0)?;
    let sig = ln_delta.var_keepdim(0)?.sqrt()?;
    let ln_delta_db = ln_delta.broadcast_sub(&mu)?.broadcast_div(&sig)?;

    // 3. factorization model
    let rgamma_beta = Gamma::new(1.0, 1.0 / (dd as f32))?;

    let beta_dk = Tensor::from_vec(
        (0..(dd * kk))
            .map(|_| rgamma_beta.sample(&mut rng))
            .collect::<Vec<f32>>(),
        (dd, kk),
        &Device::Cpu,
    )?;

    let rgamma_theta = Gamma::new(1.0, 1.0 / (kk as f32))?;

    let theta_kn = Tensor::from_vec(
        (0..(kk * nn))
            .map(|_| rgamma_theta.sample(&mut rng))
            .collect::<Vec<f32>>(),
        (kk, nn),
        &Device::Cpu,
    )?;

    tensor_io::write_tsv(&ln_batch_file, &ln_delta)?;
    tensor_io::write_tsv(&dict_file, &beta_dk)?;
    tensor_io::write_tsv(&prop_file, &theta_kn)?;

    // 4. putting them all together
    let mut triplets = vec![];

    for j in 0..nn {
        let b = batch_membership[j]; // batch index

        let lambda_j = (ln_delta_db.narrow(1, b, 1)?.exp()?)
            .mul(&beta_dk.matmul(&theta_kn.narrow(1, j, 1)?)?)?
            .flatten_to(1)?;

        lambda_j
            .to_vec1::<f32>()?
            .iter()
            .map(|&x| {
                let rpois = Poisson::new(x).unwrap();
                rpois.sample(&mut rng)
            })
            .enumerate()
            .for_each(|(i, y_ij)| {
                if y_ij > threshold {
                    triplets.push((i as u64, j as u64, y_ij as f32));
                }
            });
    }

    // dbg!(&triplets);
    write_mtx_triplets(&triplets, dd, nn, &mtx_file)?;

    Ok(())
}
