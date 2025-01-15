use asap_data::sparse_io::*;
use matrix_util::traits::SampleOps;
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
    let nn = 1111_usize;
    let xx = DMatrix::<f32>::runif(dd, nn);

    let data1: Arc<Data> = Arc::from(create_sparse_dmatrix(&xx, None, None)?);
    let data2: Arc<Data> = Arc::from(create_sparse_dmatrix(&xx, None, None)?);
    let data_vec = vec![data1.clone(), data2.clone()];

    let mut rp_obj = RandProjVec::new(&data_vec, None)?;

    rp_obj.step0_sample_basis_cbind(5)?;

    measure_time(|| rp_obj.step1_proj_cbind())?;

    measure_time(|| rp_obj.step2_random_sorting_cbind())?;

    measure_time(|| rp_obj.build_dictionary_per_batch(None))?;

    // measure_time(|| rp_obj.step4_collapse_columns_cbind(Some(100)))?;

    data1.remove_backend_file()?;
    data2.remove_backend_file()?;

    Ok(())
}
