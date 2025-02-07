use asap_data::simulate::*;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::SparseIoVec;
use asap_embed::asap_collapse_data::*;
use asap_embed::asap_random_projection::*;
// use matrix_param::traits::Inference;
// use matrix_util::traits::*;

fn measure_time<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    use std::time::Instant;
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    result
}

#[test]
fn random_collapse() -> anyhow::Result<()> {
    let sim_args = SimArgs {
        rows: 50,
        cols: 300,
        factors: Some(5),
        batches: Some(2),
        rseed: None,
    };

    let sim_out = generate_factored_poisson_gamma_data(&sim_args);
    let batch_membership = sim_out.batch_membership;
    let triplets = sim_out.triplets;

    let mtx_shape = (sim_args.rows, sim_args.cols, triplets.len());

    let data =
        create_sparse_from_triplets(triplets, mtx_shape, None, Some(&SparseIoBackend::HDF5))?;

    let arc_data = sparse_io_box_to_arc(data);
    let mut data_vec = SparseIoVec::new();
    data_vec.push(arc_data.clone())?;

    let result = data_vec.project_columns(5, None)?;
    let proj_kn = result.proj;

    measure_time(|| data_vec.assign_columns_to_samples(&proj_kn, None))?;
    measure_time(|| data_vec.register_batches(&proj_kn, &batch_membership))?;
    let _result = measure_time(|| data_vec.collapse_columns(Some(100), None, None, None))?;

    measure_time(|| data_vec.remove_backend_file())?;

    Ok(())
}
