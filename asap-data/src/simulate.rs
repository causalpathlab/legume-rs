use crate::common_io::{open_buf_writer, write_lines};
use crate::mtx_io::write_mtx_triplets;
use crate::ndarray_io::*;
use crate::ndarray_util::scale_columns;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::{prelude::Distribution, SeedableRng};

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
    use ndarray_rand::rand_distr::Gamma;
    use ndarray_rand::rand_distr::Poisson;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::rand_distr::Uniform;

    let nn = args.cols;
    let dd = args.rows;
    let kk = args.factors.unwrap_or(1);
    let bb = args.batches.unwrap_or(1);
    let rseed = args.rseed.unwrap_or(42);

    let mut rng = rand::rngs::StdRng::seed_from_u64(rseed);
    let threshold = 0.5_f32;

    // 1. batch membership matrix
    let batch_membership: Array1<usize> = Array1::random(nn, Uniform::new(0, bb));

    let batch_out: Vec<Box<str>> = batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, memb_file)?;

    // 2. batch effect matrix
    let ln_delta: Array2<f32> = Array2::random((dd, bb), StandardNormal);
    let ln_delta: Array2<f32> = scale_columns(ln_delta)?;

    // 3. factorization model
    let beta: Array2<f32> = Array2::random((dd, kk), Gamma::<f32>::new(1.0, 1.0 / (dd as f32))?);
    let theta: Array2<f32> = Array2::random((kk, nn), Gamma::<f32>::new(1.0, 1.0 / (kk as f32))?);

    write_tsv(&ln_batch_file, &ln_delta)?;
    write_tsv(&dict_file, &beta)?;
    write_tsv(&prop_file, &theta)?;

    // 4. putting them all together
    let mut triplets = vec![];

    for j in 0..nn {
        let b = batch_membership[j]; // batch index

        let lambda_j: Array1<f32> =
            ln_delta.column(b).mapv(|x| x.exp()) * beta.dot(&theta.column(j));

        // Sample triplets from Poisson (ignore zero values)
        let y_samples: Array1<f32> = lambda_j.mapv(|lam_ij| {
            let pois = Poisson::new(lam_ij as f32).unwrap();
            pois.sample(&mut rng)
        });

        for (i, &y_ij) in y_samples.iter().enumerate() {
            if y_ij as f32 > threshold {
                triplets.push((i as u64, j as u64, y_ij as f32));
            }
        }
    }

    write_mtx_triplets(&triplets, dd, nn, &mtx_file)?;

    Ok(())
}
