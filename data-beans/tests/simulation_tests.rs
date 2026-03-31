use data_beans::simulate::*;
use data_beans::simulate_multimodal::*;
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
        n_housekeeping: 0,
        housekeeping_fold: 10.0,
        n_chromosomes: 0,
        cnv_events_per_chr: 0.5,
        cnv_block_frac: 0.15,
        cnv_gain_fold: 2.0,
        cnv_loss_fold: 0.5,
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
        n_housekeeping: 0,
        housekeeping_fold: 10.0,
        n_chromosomes: 0,
        cnv_events_per_chr: 0.5,
        cnv_block_frac: 0.15,
        cnv_gain_fold: 2.0,
        cnv_loss_fold: 0.5,
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

#[test]
fn multimodal_simulation() -> anyhow::Result<()> {
    let dd = 20;
    let nn = 30;
    let kk = 4;
    let n_delta = 5;

    let args = MultimodalSimArgs {
        rows: dd,
        cols: nn,
        depth_per_modality: vec![100, 80],
        factors: kk,
        batches: 1,
        base_scale: 1.0,
        delta_scale: 1.0,
        n_delta_features: n_delta,
        pve_topic: 1.0,
        pve_batch: 1.0,
        rseed: 42,
        shared_batch_effects: false,
        hierarchical_depth: None,
        overdisp: 1.0,
        n_housekeeping: 0,
        housekeeping_fold: 10.0,
    };

    let out = generate_multimodal_data(&args)?;

    // 2 modalities
    assert_eq!(out.triplets.len(), 2);
    assert_eq!(out.beta_dk.len(), 2);

    // Shapes
    assert_eq!(out.w_base_kd.nrows(), kk);
    assert_eq!(out.w_base_kd.ncols(), dd);
    assert_eq!(out.theta_kn.nrows(), kk);
    assert_eq!(out.theta_kn.ncols(), nn);

    for beta in &out.beta_dk {
        assert_eq!(beta.nrows(), dd);
        assert_eq!(beta.ncols(), kk);
    }

    // 1 delta matrix (M-1 = 1)
    assert_eq!(out.w_delta_kd.len(), 1);
    assert_eq!(out.spike_mask_kd.len(), 1);

    // Delta sparsity: exactly n_delta non-zero entries per topic
    let mask = &out.spike_mask_kd[0];
    for k in 0..kk {
        let nnz: usize = (0..dd).filter(|&d| mask[(k, d)] > 0.0).count();
        assert_eq!(
            nnz, n_delta,
            "topic {} has {} non-zero delta features, expected {}",
            k, nnz, n_delta
        );
    }

    // Triplets non-empty for each modality
    for (m, trips) in out.triplets.iter().enumerate() {
        assert!(!trips.is_empty(), "modality {} has no triplets", m);
    }

    // Reference vs non-reference dictionaries should differ
    let diff = (&out.beta_dk[0] - &out.beta_dk[1]).abs().sum();
    assert!(diff > 0.0, "dictionaries should differ between modalities");

    Ok(())
}

#[test]
fn cnv_simulation_ground_truth() -> anyhow::Result<()> {
    // Simulate with strong CNV effects: 500 genes, 3 chromosomes, 2000 cells, 3 topics
    let args = SimArgs {
        rows: 500,
        cols: 2000,
        depth: 200,
        factors: 3,
        batches: 2,
        overdisp: 1.0,
        pve_topic: 0.8,
        pve_batch: 0.3,
        rseed: 42,
        hierarchical_depth: None,
        n_housekeeping: 0,
        housekeeping_fold: 10.0,
        n_chromosomes: 3,
        cnv_events_per_chr: 1.0,
        cnv_block_frac: 0.2,
        cnv_gain_fold: 2.0,
        cnv_loss_fold: 0.5,
    };

    let out = generate_factored_poisson_gamma_data(&args)?;

    // Should have CNV output
    let states = out.cnv_states.as_ref().unwrap();
    let chroms = out.gene_chromosomes.as_ref().unwrap();
    let positions = out.gene_positions.as_ref().unwrap();

    assert_eq!(states.len(), 500);
    assert_eq!(chroms.len(), 500);
    assert_eq!(positions.len(), 500);

    // Should have some non-neutral genes
    let n_gain = states.iter().filter(|&&s| s == 2).count();
    let n_loss = states.iter().filter(|&&s| s == 0).count();
    let n_neutral = states.iter().filter(|&&s| s == 1).count();
    eprintln!(
        "CNV states: {} gain, {} loss, {} neutral",
        n_gain, n_loss, n_neutral
    );
    assert!(n_gain + n_loss > 0, "should have some CNV events");
    assert!(n_neutral > 0, "should have some neutral genes");

    // Genes on the same chromosome should have contiguous positions
    for chr_idx in 0..3 {
        let chr_name: Box<str> = format!("chr{}", chr_idx + 1).into();
        let chr_positions: Vec<u64> = (0..500)
            .filter(|&g| chroms[g] == chr_name)
            .map(|g| positions[g])
            .collect();
        // Positions should be monotonically increasing
        for w in chr_positions.windows(2) {
            assert!(w[0] < w[1], "positions should be sorted within chromosome");
        }
    }

    assert!(!out.triplets.is_empty());

    Ok(())
}
