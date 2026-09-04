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

#[test]
fn pool_axis_index_add_matches_loop() {
    // 8 fine rows × H=3, grouped into 4 coarse blocks (incl. one empty).
    let dev = dev();
    let table =
        Tensor::from_vec((0..24).map(|x| x as f32).collect::<Vec<_>>(), (8, 3), &dev).unwrap();
    let bias = Tensor::from_vec(
        (0..8).map(|x| (x as f32) * 0.1).collect::<Vec<_>>(),
        8,
        &dev,
    )
    .unwrap();

    let coarse_to_fine = vec![
        vec![0, 1, 2],    // block 0
        vec![3],          // block 1
        vec![],           // block 2 (empty)
        vec![4, 5, 6, 7], // block 3
    ];
    let blocks = vec![3u32, 0, 2, 1, 0]; // mixed order, repeats allowed

    let (emb_new, bias_new) = pool_axis(&table, &bias, &blocks, &coarse_to_fine, &dev).unwrap();
    let (emb_ref, bias_ref) =
        pool_axis_loop(&table, &bias, &blocks, &coarse_to_fine, &dev).unwrap();

    let emb_n: Vec<f32> = emb_new.flatten_all().unwrap().to_vec1().unwrap();
    let emb_r: Vec<f32> = emb_ref.flatten_all().unwrap().to_vec1().unwrap();
    let bias_n: Vec<f32> = bias_new.flatten_all().unwrap().to_vec1().unwrap();
    let bias_r: Vec<f32> = bias_ref.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(emb_n.len(), emb_r.len());
    assert_eq!(bias_n.len(), bias_r.len());
    for (a, b) in emb_n.iter().zip(emb_r.iter()) {
        assert!((a - b).abs() < 1e-5, "emb mismatch: {a} vs {b}");
    }
    for (a, b) in bias_n.iter().zip(bias_r.iter()) {
        assert!((a - b).abs() < 1e-5, "bias mismatch: {a} vs {b}");
    }
}
