use data_beans::simulate::*;
use data_beans::sparse_io::*;

#[test]
fn sparse_matrix_simulation_and_loading() -> anyhow::Result<()> {
    let args = SimArgs {
        rows: 10,
        cols: 15,
        depth: 100,
        factors: 1,
        batches: 1,
        overdisp: 1.,
        pve_topic: 1.,
        pve_batch: 1.,
        rseed: 42,
        hierarchical_depth: None,
    };

    let _out = generate_factored_poisson_gamma_data(&args)?;

    let mtx_shape = (args.rows, args.cols, _out.triplets.len());

    let _data = create_sparse_from_triplets(&_out.triplets, mtx_shape, None, None)?;

    _data.remove_backend_file()?;

    Ok(())
}

#[test]
fn hierarchical_simulation_and_loading() -> anyhow::Result<()> {
    let args = SimArgs {
        rows: 20,
        cols: 30,
        depth: 100,
        factors: 1,
        batches: 1,
        overdisp: 1.,
        pve_topic: 1.,
        pve_batch: 1.,
        rseed: 42,
        hierarchical_depth: Some(3), // K = 2^(3-1) = 4 leaf topics
    };

    let out = generate_factored_poisson_gamma_data(&args)?;

    // Should have 4 leaf topics
    assert_eq!(out.beta_dk.ncols(), 4);
    assert_eq!(out.beta_dk.nrows(), 20);

    // Should have hierarchy node probs
    let node_probs = out.hierarchy_node_probs.as_ref().unwrap();
    assert_eq!(node_probs.nrows(), 20); // D = 20
    assert_eq!(node_probs.ncols(), 7); // 2^3 - 1 = 7 nodes

    // Theta should have K=4 rows
    assert_eq!(out.theta_kn.nrows(), 4);
    assert_eq!(out.theta_kn.ncols(), 30);

    // Should produce non-empty triplets
    assert!(!out.triplets.is_empty());

    let mtx_shape = (args.rows, args.cols, out.triplets.len());
    let data = create_sparse_from_triplets(&out.triplets, mtx_shape, None, None)?;
    data.remove_backend_file()?;

    Ok(())
}
