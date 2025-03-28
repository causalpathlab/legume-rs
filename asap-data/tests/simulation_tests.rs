use asap_data::simulate::*;
use asap_data::sparse_io::*;

#[test]
fn sparse_matrix_simulation_and_loading() -> anyhow::Result<()> {
    let args = SimArgs {
        rows: 10,
        cols: 15,
        factors: None,
        batches: None,
        rseed: None,
    };

    let _out = generate_factored_poisson_gamma_data(&args);

    let mtx_shape = (args.rows, args.cols, _out.triplets.len());

    let _data =
        create_sparse_from_triplets(_out.triplets, mtx_shape, None, None)?;

    Ok(())
}
