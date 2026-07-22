use super::*;

fn base() -> GemEncoderArgs {
    GemEncoderArgs {
        genes_pos: vec!["a_genes.zarr.zip".into()],
        batch_files: None,
        out: "out/gme".into(),
        n_latent: 10,
        embedding_dim: 128,
        latent_noise: false,
        encoder_layers: vec![128, 128],
        context_size: 512,
        likelihood: LikelihoodArg::Nb,
        mask_fraction: 0.15,
        delta_l2: 1.0,
        feature_embedding_l2: 1.0,
        topic_smoothing: 0.01,
        num_levels: 3,
        sort_dim: 10,
        knn_pb: 10,
        num_opt_iter: 100,
        proj_dim: 64,
        n_hvg: 0,
        ignore_batch: false,
        batch_adjust: true,
        genes_sample_strip: "".into(),
        epochs: 100,
        minibatch_size: 100,
        learning_rate: 1e-2,
        weight_decay: 0.0,
        grad_clip: 1.0,
        qc: data_beans::qc_lib::QcArgs {
            no_qc: false,
            qc_mads: 5.0,
            qc_min_cell_nnz: 2,
            qc_min_counts: 0.0,
            qc_mito_pattern: None,
            qc_mito_max_frac: None,
            qc_ribo_pattern: None,
            qc_ribo_max_frac: None,
            qc_feature_min_cells: 0,
            qc_report: None,
            qc_histogram: false,
            qc_mad_on_genes: true,
            qc_mad_on_counts: true,
            qc_auto_cutoff: false,
        },
        runtime: RuntimeArgs {
            preload_data: true,
            seed: 42,
            device: ComputeDevice::Cpu,
            device_no: 0,
            threads: 16,
        },
    }
}

#[test]
fn defaults_validate() {
    base().validate().expect("default args should validate");
}

#[test]
fn missing_input_is_rejected() {
    let mut a = base();
    a.genes_pos.clear();
    assert!(a.genes().is_err());
    assert!(a.validate().is_err());
}

/// `β = softmax(α·ρᵀ)` has rank at most H, so H < K cannot represent K
/// independent factors. Catching it here beats discovering it as a silently
/// degenerate dictionary after a long fit.
#[test]
fn embedding_dim_below_n_latent_is_rejected() {
    let mut a = base();
    a.embedding_dim = 4;
    a.n_latent = 16;
    let err = a.validate().unwrap_err().to_string();
    assert!(err.contains("--embedding-dim"), "unhelpful message: {err}");
}

/// `--mask-fraction` is a probability; anything outside `[0, 1)` is not one.
#[test]
fn mask_fraction_must_be_a_proper_fraction() {
    for bad in [-0.1, 1.0, 1.5] {
        let mut a = base();
        a.mask_fraction = bad;
        assert!(a.validate().is_err(), "mask_fraction {bad} should be rejected");
    }
    let mut a = base();
    a.mask_fraction = 0.15;
    assert!(a.validate().is_ok());
}
