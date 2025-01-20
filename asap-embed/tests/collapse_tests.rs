use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::SparseIoVec;
use asap_embed::collapse_data::*;
use asap_embed::common::*;
use asap_embed::random_projection::*;
use matrix_util::traits::SampleOps;
use std::sync::Arc;
use std::time::Instant;

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
    use rayon::prelude::*;

    let dd = 50_usize;
    let nn = 1111_usize;
    let xx = DMatrix::<f32>::runif(dd, nn);

    let data1: Arc<SparseData> = Arc::from(create_sparse_dmatrix(&xx, None, None)?);
    let data2: Arc<SparseData> = Arc::from(create_sparse_dmatrix(&xx, None, None)?);

    let mut data_vec = SparseIoVec::new();
    data_vec.push(data1.clone())?;
    data_vec.push(data2.clone())?;

    let result = data_vec.project_columns(3, None)?;

    let proj_kn = result.proj;

    measure_time(|| data_vec.assign_columns_to_samples(Some(&proj_kn), None))?;

    use rand::{thread_rng, Rng};
    let nbatch = 3;
    let runif = rand_distr::Uniform::<usize>::new(0, nbatch);
    let batch_membership = (0..data_vec.num_columns().unwrap())
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(runif))
        .collect();

    measure_time(|| data_vec.register_batches(&proj_kn, &batch_membership))?;

    measure_time(|| data_vec.collapse_columns_as_assigned(None, None, None))?;

    measure_time(|| data_vec.remove_backend_file())?;
    Ok(())
}
