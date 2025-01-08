use asap_data::common_io::{create_temp_dir_file, read_lines};

use asap_data::simulate::*;
use asap_data::sparse_io::*;
use std::path::Path;
use std::time::Instant;

use asap_data::common_io::*;
use asap_embed::random_projection::*;
use std::sync::Arc;

fn measure_time<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    result
}

#[test]
fn random_collapse() -> anyhow::Result<()> {
    let dd = 177_usize;
    let nn = 3777_usize;

    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    let runif = Uniform::new(0_f32, 1_f32);

    let rvec: Vec<f32> = (0..(dd * nn))
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(runif))
        .collect();

    let xx = DMatrix::from_vec(dd, nn, rvec);

    let data1: Arc<Data> = Arc::from(create_sparse_dmatrix(&xx, None, None)?);

    let data_vec = vec![data1.clone()];

    let mut rp_obj = RandProjVec::new(&data_vec, None)?;

    rp_obj.step1_sample_basis_cbind(10)?;
    measure_time(|| rp_obj.step2_proj_cbind())?;

    data1.remove_backend_file()?;

    Ok(())
}
