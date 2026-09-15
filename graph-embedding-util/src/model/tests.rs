use super::*;

fn dev() -> Device {
    Device::Cpu
}

/// The randn model init is drawn from the seed (not candle's unseedable CPU
/// device RNG), so two constructions with the same seed must produce
/// byte-identical embedding tables, and different seeds must diverge.
#[test]
fn model_init_is_seed_reproducible() {
    let dev = dev();
    // Equal feature/cell counts so the e_feat vs e_cell comparison below is
    // shape-matched and therefore a real test of salt separation.
    let args = || ModelArgs {
        n_features: 16,
        n_cells: 16,
        embedding_dim: 4,
        seed: 2026,
    };
    let init = ModelInit {
        e_feat: None,
        e_cell: None,
        b_feat: &[0f32; 16],
        b_cell: &[0f32; 16],
    };
    let build = |seed: u64| {
        let mut a = args();
        a.seed = seed;
        let vm = VarMap::new();
        let m = JointEmbedModel::new_with_init(a, &init, &vm, &dev).unwrap();
        // Init tensors must be contiguous — non-contiguous Vars break CUDA
        // matmul kernels during training.
        assert!(m.e_feat.is_contiguous(), "e_feat init must be contiguous");
        assert!(m.e_cell.is_contiguous(), "e_cell init must be contiguous");
        (
            m.e_feat.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            m.e_cell.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        )
    };

    let (ef1, ec1) = build(2026);
    let (ef2, ec2) = build(2026);
    assert_eq!(ef1, ef2, "same seed → identical e_feat");
    assert_eq!(ec1, ec2, "same seed → identical e_cell");

    let (ef3, _) = build(2027);
    assert_ne!(ef1, ef3, "different seed → different e_feat");
    // e_feat and e_cell use distinct per-tensor salts, so they must not be
    // identical to each other even under one seed.
    assert_ne!(
        ef1, ec1,
        "e_feat and e_cell must use independent sub-streams"
    );
}
