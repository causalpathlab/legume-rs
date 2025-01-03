use asap_data::common_io::{create_temp_dir_file, read_lines};

use asap_data::simulate::*;
use asap_data::sparse_io::*;
use asap_data::sparse_matrix_zarr::SparseMtxData;
use ndarray_rand::RandomExt;
use std::path::Path;
use std::time::Instant;

use asap_data::common_io::*;
use asap_embed::random_projection as rp;

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
    let dd = 151_usize;
    let nn = 777_usize;

    let whole_mat = Array::random((dd, nn), rand::distributions::Uniform::new(0., 1.));

    if let Ok(mut data) = SparseMtxData::from_ndarray(&whole_mat, None, None) {
        let nrow = data.num_rows().unwrap();
        let ncol = data.num_columns().unwrap();

        let rp_result = rp::collapse_columns(&data, 3, None)?;

        let _ = rp::collapse_columns_cbind(&vec![Box::new(&data)], 3, None)?;

        data.remove_backend_file()?;
    }

    Ok(())
}
