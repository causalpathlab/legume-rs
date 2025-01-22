use asap_data::simulate::*;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::SparseIoVec;
use asap_data::sparse_matrix_zarr::SparseMtxData;
use asap_embed::collapse_data::*;
use asap_embed::common::*;
use asap_embed::random_projection::*;
use matrix_param::traits::Inference;
use matrix_util::common_io::{create_temp_dir_file, read_lines};
use matrix_util::traits::SampleOps;
use std::sync::Arc;

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
        rows: 500,
        cols: 3000,
        factors: Some(5),
        batches: Some(3),
        rseed: None,
    };

    let mtx_file = create_temp_dir_file(".mtx.gz")?;
    let mtx_file = mtx_file.to_str().unwrap().to_string();
    let dict_file = mtx_file.replace(".mtx.gz", ".dict.gz");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.gz");
    let memb_file = mtx_file.replace(".mtx.gz", ".memb.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.gz");

    generate_factored_poisson_gamma_data_mtx(
        &sim_args,
        &mtx_file,
        &dict_file,
        &prop_file,
        &ln_batch_file,
        &memb_file,
    )?;

    let mut ln_batch_mat = Mat::from_tsv(&ln_batch_file, None)?;
    ln_batch_mat.scale_columns_inplace();

    let data = measure_time(|| SparseMtxData::from_mtx_file(&mtx_file, None, Some(true)))?;

    let arc_data = Arc::from(data);
    let mut data_vec = SparseIoVec::new();

    data_vec.push(arc_data.clone())?;

    let batch_membership = read_lines(&memb_file)?
        .iter()
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();

    // let result = data_vec.project_rows(3, None)?;
    // dbg!(result.proj);

    let result = data_vec.project_columns(5, None)?;
    let proj_kn = result.proj;

    measure_time(|| data_vec.assign_columns_to_samples(Some(&proj_kn), None))?;

    measure_time(|| data_vec.register_batches(&proj_kn, &batch_membership))?;

    let result = measure_time(|| data_vec.collapse_columns(Some(100), None, None))?;

    measure_time(|| data_vec.remove_backend_file())?;

    if let Some(delta) = result.delta {
        let mut ln_batch_hat = delta.posterior_log_mean().clone();
        ln_batch_hat.scale_columns_inplace();

        // dbg!(&ln_batch_hat);
        // dbg!(&ln_batch_mat);

        dbg!(ln_batch_mat * &ln_batch_hat);
    }

    Ok(())
}
