use asap_data::common_io::{create_temp_dir_file, read_lines};

use asap_data::simulate::*;
use asap_data::sparse_io::*;
use std::borrow::{Borrow, BorrowMut};
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
    let dd = 17_usize;
    let nn = 377_usize;

    // use ndarray_rand::rand_distr::Uniform;
    // use ndarray_rand::RandomExt;

    // let mut rng = rand::thread_rng();
    // let runif = Uniform::new(0., 1.)?;
    // let xx: Array2<f32> = Array2::random((nn, dd), runif);

    // let data1: Arc<Data> = Arc::from(create_sparse_ndarray(&xx, None, None)?);
    // let data_vec = vec![data1.clone()];

    // let mut rp_obj = RandProjVec::new(&data_vec, None)?;

    // rp_obj.step1_sample_basis_cbind(10)?;
    // rp_obj.step2_proj_cbind()?;

    // data1.remove_backend_file()?;

    Ok(())
}
