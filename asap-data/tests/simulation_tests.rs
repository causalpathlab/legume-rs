use asap_data::simulate::*;
use asap_data::sparse_matrix_hdf5::SparseMtxData;

#[test]
fn sparse_matrix_simulation_and_loading() -> anyhow::Result<()> {
    let args = SimulateArgs {
        rows: 10,
        cols: 15,
        factors: Some(2),
        batches: Some(2),
    };

    generate_factored_gamma_data(args)?;

    Ok(())
}
