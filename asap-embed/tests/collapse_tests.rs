use matrix_util::common_io::{create_temp_dir_file, read_lines};

use asap_data::simulate::*;
use asap_data::sparse_io::*;
use matrix_util::dmatrix_util::runif;
use std::path::Path;
use std::time::Instant;

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
    let dd = 500_usize;
    let nn = 1777_usize;
    let xx = runif(dd, nn);

    let data1: Arc<Data> = Arc::from(create_sparse_dmatrix(&xx, None, None)?);
    let data_vec = vec![data1.clone()];

    let mut rp_obj = RandProjVec::new(&data_vec, Some(100))?;

    rp_obj.step0_sample_basis_cbind(10)?;
    measure_time(|| rp_obj.step1_proj_cbind())?;

    data1.remove_backend_file()?;

    Ok(())
}
