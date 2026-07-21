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

/// With `e_gene = 1/√H · 1` (a unit vector pointing along the all-
/// ones direction in cell-embedding space), the gated score reduces
/// to a rescaled product of axis-aligned projections. Concretely:
///
///   (e_gene · e_cell_l) = (sum_h e_cell_l[h]) / √H = m_l
///   (e_gene · e_cell_r) = m_r
///   pair_score = m_l · m_r + b_l + b_r
///
/// This test asserts the gated helper computes exactly that on a
/// small fixture, so we have a known-good closed form before
/// landing the chain integration.
#[test]
fn score_cellcell_gated_matches_closed_form() {
    let dev = dev();
    let b = 4;
    let h = 3;

    let e_cell_l = Tensor::from_vec(
        vec![
            0.1f32, 0.2, 0.3, //
            0.4, -0.1, 0.5, //
            -0.2, 0.0, 0.7, //
            0.8, 0.1, -0.3, //
        ],
        (b, h),
        &dev,
    )
    .unwrap();
    let e_cell_r = Tensor::from_vec(
        vec![
            0.0f32, 0.3, -0.2, //
            0.6, 0.4, 0.1, //
            -0.1, 0.5, 0.2, //
            -0.4, 0.2, 0.5, //
        ],
        (b, h),
        &dev,
    )
    .unwrap();
    let b_l = Tensor::from_vec(vec![0.0f32, 0.01, -0.02, 0.03], b, &dev).unwrap();
    let b_r = Tensor::from_vec(vec![-0.01f32, 0.02, 0.0, -0.03], b, &dev).unwrap();

    let unit = 1.0f32 / (h as f32).sqrt();
    let e_gene = Tensor::from_vec(vec![unit; b * h], (b, h), &dev).unwrap();

    let got = JointEmbedModel::score_cellcell_gated(&e_gene, &e_cell_l, &e_cell_r, &b_l, &b_r)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let e_l: Vec<f32> = e_cell_l.flatten_all().unwrap().to_vec1().unwrap();
    let e_r: Vec<f32> = e_cell_r.flatten_all().unwrap().to_vec1().unwrap();
    let b_l_h: Vec<f32> = b_l.to_vec1().unwrap();
    let b_r_h: Vec<f32> = b_r.to_vec1().unwrap();
    for i in 0..b {
        let m_l: f32 = (0..h).map(|j| e_l[i * h + j]).sum::<f32>() * unit;
        let m_r: f32 = (0..h).map(|j| e_r[i * h + j]).sum::<f32>() * unit;
        let expected = m_l * m_r + b_l_h[i] + b_r_h[i];
        assert!(
            (got[i] - expected).abs() < 1e-5,
            "row {i}: got={} expected={}",
            got[i],
            expected
        );
    }
}

/// Same equivalence check for the negative-side score: gated_neg
/// should produce `(unit·anchor)(unit·neg) + b_anchor + b_neg`
/// for every (B, K) entry.
#[test]
fn score_cellcell_gated_neg_matches_closed_form() {
    let dev = dev();
    let b = 3;
    let k = 2;
    let h = 4;

    let e_cell_anchor = Tensor::from_vec(
        vec![
            0.1f32, 0.2, 0.3, 0.4, //
            -0.1, 0.2, -0.3, 0.5, //
            0.6, -0.2, 0.1, 0.0, //
        ],
        (b, h),
        &dev,
    )
    .unwrap();
    let e_cell_neg = Tensor::from_vec(
        vec![
            0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, //
            -0.3, 0.4, 0.0, 0.1, 0.2, 0.1, 0.0, 0.3, //
            0.5, -0.1, 0.2, 0.0, 0.0, 0.1, 0.2, 0.4, //
        ],
        (b, k, h),
        &dev,
    )
    .unwrap();
    let b_anchor = Tensor::from_vec(vec![0.01f32, -0.02, 0.03], b, &dev).unwrap();
    let b_neg =
        Tensor::from_vec(vec![0.0f32, 0.01, -0.01, 0.02, 0.0, -0.02], (b, k), &dev).unwrap();

    let unit = 1.0f32 / (h as f32).sqrt();
    let e_gene = Tensor::from_vec(vec![unit; b * h], (b, h), &dev).unwrap();

    let got = JointEmbedModel::score_cellcell_gated_neg(
        &e_gene,
        &e_cell_anchor,
        &e_cell_neg,
        &b_anchor,
        &b_neg,
    )
    .unwrap()
    .to_vec2::<f32>()
    .unwrap();

    let a: Vec<f32> = e_cell_anchor.flatten_all().unwrap().to_vec1().unwrap();
    let n: Vec<f32> = e_cell_neg.flatten_all().unwrap().to_vec1().unwrap();
    let ba: Vec<f32> = b_anchor.to_vec1().unwrap();
    let bn: Vec<Vec<f32>> = b_neg.to_vec2().unwrap();
    for i in 0..b {
        let m_a: f32 = (0..h).map(|j| a[i * h + j]).sum::<f32>() * unit;
        for kk in 0..k {
            let m_n: f32 = (0..h).map(|j| n[i * k * h + kk * h + j]).sum::<f32>() * unit;
            let expected = m_a * m_n + ba[i] + bn[i][kk];
            assert!(
                (got[i][kk] - expected).abs() < 1e-5,
                "({i},{kk}): got={} expected={}",
                got[i][kk],
                expected
            );
        }
    }
}

////////////////////////////////////////////////////////////////////
// Softmax feature gate (SuSiE single-effect prior + selection)

/// Build a free, gated model with the given null-column init (real columns 0).
fn gated_model(n_features: usize, h: usize, vm: &VarMap) -> JointEmbedModel {
    let mut m = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features,
            n_cells: 3,
            embedding_dim: h,
            seed: 11,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &vec![0f32; n_features],
            b_cell: &[0f32; 3],
        },
        vm,
        &dev(),
    )
    .unwrap();
    // The gate is always the variational spike-and-slab single-effect. The
    // selection/materialize tests read effect MEANS (no sampling), so they are
    // unaffected by the variational log-std this allocates.
    m.enable_softmax_gate(SoftmaxGateSpec { temperature: 1.0 }, vm, &dev())
        .unwrap();
    m
}

/// An ungated model is fully inert: no gate spec, no gate Vars, no selection.
/// (The gather/materialize paths reduce to the pre-gate behaviour by construction.)
#[test]
fn ungated_model_is_inert() {
    let vm = VarMap::new();
    let m = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: 5,
            n_cells: 3,
            embedding_dim: 4,
            seed: 1,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &[0f32; 5],
            b_cell: &[0f32; 3],
        },
        &vm,
        &dev(),
    )
    .unwrap();
    assert!(m.gate.is_none());
    assert!(m.s_feat.is_none() && m.e_feat_raw.is_none());
    assert!(m.feature_selection().unwrap().is_none());
}

/// With all-zero logits (forced) a null column ⇒ softmax over `H+1` is uniform, so
/// each real-dim selection prob is `1/(H+1)` and the gate scales a base row by that.
#[test]
fn zero_logits_give_uniform_selection() {
    let (n_features, h) = (6usize, 4usize);
    let vm = VarMap::new();
    let mut m = gated_model(n_features, h, &vm);
    // Override the (prior-biased) init with all-zero logits to test the uniform case.
    m.s_feat = Some(Tensor::zeros((n_features, h + 1), DType::F32, &dev()).unwrap());

    let sel = m.feature_selection().unwrap().unwrap();
    assert_eq!(sel.dims(), &[n_features, h]);
    let expected = 1.0 / (h as f32 + 1.0);
    for x in sel.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
        assert!((x - expected).abs() < 1e-6, "sel {x} != {expected}");
    }

    // apply_softmax_gate scales an all-ones base by the same uniform weight.
    let base = Tensor::ones((n_features, h), DType::F32, &dev()).unwrap();
    let gated = m
        .apply_softmax_gate(&base, m.s_feat.as_ref().unwrap())
        .unwrap();
    for x in gated.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
        assert!((x - expected).abs() < 1e-6);
    }
}

/// The internal null init is prior-aligned (`ln(π₀·H/(1−π₀))`), so genes start ~OFF:
/// the real-dim selection mass (`rowsum`) sits near `1−π₀` (well below 0.5) — the
/// graceful-selection init that keeps junk genes from shaping `θ`.
#[test]
fn null_biased_init_starts_genes_off() {
    let (n_features, h) = (4usize, 8usize);
    let vm = VarMap::new();
    let m = gated_model(n_features, h, &vm);
    let sel = m.feature_selection().unwrap().unwrap();
    let v = sel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for r in 0..n_features {
        let rowsum: f32 = (0..h).map(|j| v[r * h + j]).sum();
        assert!(
            rowsum < 0.5,
            "row {r} real mass {rowsum} should be mostly null"
        );
    }
}

/// `materialize_e_feat` bakes the gate into `e_feat` for a free model — the frozen
/// dictionary equals `raw ⊙ softmax(S)`, so phase-2 / output readers see the gated
/// embedding while the raw Var stays reachable for the training gather.
#[test]
fn materialize_bakes_gate_free() {
    let (n_features, h) = (5usize, 4usize);
    let vm = VarMap::new();
    let mut m = gated_model(n_features, h, &vm);
    let raw = m.e_feat_raw.as_ref().unwrap().clone();
    let sel = m.feature_selection().unwrap().unwrap();
    let expect = (raw * sel)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    m.materialize_e_feat().unwrap();
    let got = m.e_feat.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in got.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-6, "materialized {a} != raw⊙sel {b}");
    }
    assert!(m.e_feat_raw.is_some(), "raw Var must survive materialize");
}

/// Gradients flow to the gate logits: a scalar loss on the gated rows produces a
/// nonzero `s_feat` gradient, so the softmax selection is learnable.
#[test]
fn backward_reaches_gate_logits() {
    let (n_features, h) = (4usize, 3usize);
    let vm = VarMap::new();
    let m = gated_model(n_features, h, &vm);
    let idx = Tensor::from_vec(
        (0..n_features as u32).collect::<Vec<_>>(),
        n_features,
        &dev(),
    )
    .unwrap();
    let base = m
        .e_feat_raw
        .as_ref()
        .unwrap()
        .index_select(&idx, 0)
        .unwrap();
    let s = m.s_feat.as_ref().unwrap().index_select(&idx, 0).unwrap();
    let gated = m.apply_softmax_gate(&base, &s).unwrap();
    let loss = gated.sqr().unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let g = grads
        .get(m.s_feat.as_ref().unwrap())
        .expect("s_feat gradient");
    let gnorm: f32 = g.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(
        gnorm > 0.0,
        "s_feat gradient must be nonzero (gate is learnable)"
    );
}

fn plain_model(n_features: usize, h: usize, seed: u64, vm: &VarMap) -> JointEmbedModel {
    JointEmbedModel::new_with_init(
        ModelArgs {
            n_features,
            n_cells: 3,
            embedding_dim: h,
            seed,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &vec![0f32; n_features],
            b_cell: &[0f32; 3],
        },
        vm,
        &dev(),
    )
    .unwrap()
}

/// The SuSiE single-effect KL loss (categorical + Gaussian) — the gate is always the
/// variational spike-and-slab, so a gated model allocates the effect log-std and its
/// `gate_kl` is finite, non-negative, and its gradient reaches the log-std. An ungated
/// model returns `None`.
#[test]
fn gate_kl_present_and_learnable() {
    let (n_features, h) = (6usize, 4usize);

    // Gated (always variational) → effect log-std allocated, KL present.
    let vm = VarMap::new();
    let m = gated_model(n_features, h, &vm);
    assert!(
        m.e_feat_logstd.is_some(),
        "variational gate allocates effect log-std"
    );
    let kl_t = m.gate_kl().unwrap().unwrap();
    let kl: f32 = kl_t.to_scalar().unwrap();
    assert!(
        kl.is_finite() && kl >= -1e-4,
        "KL finite & non-negative, got {kl}"
    );

    // Gradient reaches the effect log-std.
    let grads = kl_t.backward().unwrap();
    let g = grads
        .get(m.e_feat_logstd.as_ref().unwrap())
        .expect("logstd gradient");
    let gn: f32 = g.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(gn > 0.0, "KL gradient must reach the effect log-std");

    // Ungated model has no KL.
    let vm2 = VarMap::new();
    let ungated = plain_model(n_features, h, 1, &vm2);
    assert!(ungated.gate_kl().unwrap().is_none());
}

/// A factored model WITH velocity (`splice_delta`) gets an INDEPENDENT δ gate: enabling
/// the gate allocates `s_delta`/`delta_logstd`, `velocity_selection` is populated and
/// β-shared (a gene's spliced and unspliced rows share the same selection), and
/// `gate_kl` picks up the δ contribution (its gradient reaches `delta_logstd`).
#[test]
fn delta_gate_independent_and_beta_shared() {
    let (n_genes, h) = (3usize, 4usize);
    // 2 rows per gene: spliced (2g) + unspliced (2g+1).
    let row_to_gene: Vec<u32> = (0..n_genes as u32).flat_map(|g| [g, g]).collect();
    let unspliced: Vec<bool> = (0..2 * n_genes).map(|r| r % 2 == 1).collect();
    let n_features = row_to_gene.len();
    let vm = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev());
    let mut m = JointEmbedModel::new_factored(
        FactoredInit {
            n_features,
            n_cells: 3,
            embedding_dim: h,
            n_genes,
            row_to_gene: &row_to_gene,
            b_feat: &vec![0f32; n_features],
            b_cell: &[0f32; 3],
            seed: 5,
            unspliced_rows: Some(&unspliced),
        },
        &vm,
        vs,
        &dev(),
    )
    .unwrap();
    m.enable_softmax_gate(SoftmaxGateSpec { temperature: 1.0 }, &vm, &dev())
        .unwrap();

    // Independent velocity gate allocated (velocity present).
    let f = m.factor.as_ref().unwrap();
    assert!(
        f.s_delta.is_some() && f.delta_logstd.is_some(),
        "δ gate allocated when velocity is present"
    );

    // velocity_selection: [n_features, H], β-shared across a gene's two tracks.
    let vsel = m.velocity_selection().unwrap().unwrap();
    assert_eq!(vsel.dims(), &[n_features, h]);
    let v = vsel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for g in 0..n_genes {
        for j in 0..h {
            let spliced = v[(2 * g) * h + j];
            let unspliced_v = v[(2 * g + 1) * h + j];
            assert!(
                (spliced - unspliced_v).abs() < 1e-6,
                "gene {g} dim {j}: spliced {spliced} != unspliced {unspliced_v} (δ gate must be β-shared)"
            );
        }
    }

    // gate_kl includes the δ term: its gradient reaches delta_logstd.
    let kl = m.gate_kl().unwrap().unwrap();
    let grads = kl.backward().unwrap();
    let dl = m.factor.as_ref().unwrap().delta_logstd.as_ref().unwrap();
    let g = grads.get(dl).expect("delta_logstd gradient");
    let gn: f32 = g.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(gn > 0.0, "gate KL must reach the δ-gate log-std");
}

/// An ungated factored model with velocity gathers plainly (`β + mask·δ`) — no δ gate
/// is allocated and `velocity_selection` is `None`.
#[test]
fn ungated_factored_has_no_delta_gate() {
    let (n_genes, h) = (2usize, 3usize);
    let row_to_gene: Vec<u32> = (0..n_genes as u32).flat_map(|g| [g, g]).collect();
    let unspliced: Vec<bool> = (0..2 * n_genes).map(|r| r % 2 == 1).collect();
    let n_features = row_to_gene.len();
    let vm = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev());
    let m = JointEmbedModel::new_factored(
        FactoredInit {
            n_features,
            n_cells: 3,
            embedding_dim: h,
            n_genes,
            row_to_gene: &row_to_gene,
            b_feat: &vec![0f32; n_features],
            b_cell: &[0f32; 3],
            seed: 9,
            unspliced_rows: Some(&unspliced),
        },
        &vm,
        vs,
        &dev(),
    )
    .unwrap();
    let f = m.factor.as_ref().unwrap();
    assert!(f.s_delta.is_none() && f.delta_logstd.is_none());
    assert!(m.velocity_selection().unwrap().is_none());
    assert!(m.gate_kl().unwrap().is_none());
}
