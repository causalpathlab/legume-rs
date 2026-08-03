use super::*;

fn dev() -> Device {
    Device::Cpu
}

/// A `[rows, H]` pip table on the test device. `install_gate_pip` takes a tensor
/// because the composite fit uploads once and shares it across heads; tests have no
/// such concern, so they build one here.
fn pip_tensor(pip: &[f32], rows: usize, h: usize) -> Tensor {
    Tensor::from_slice(pip, (rows, h), &dev()).unwrap()
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
    m.enable_feature_gate(
        FeatureGateSpec {
            temperature: 1.0,
            ibp_alpha: None,
        },
        vm,
        &dev(),
    )
    .unwrap();
    m
}

/// Every DIM carries one unit of mass, whatever the logits say — asserted on a
/// deliberately LOPSIDED table, because on a uniform one it holds by symmetry and
/// the test could not fail. This is the headline invariant of the gate: no dim can
/// hoard, which is exactly what the previous per-gene axis failed to prevent
/// (measured column sums there ran 167–536, a 3.2× spread).
///
/// Also pins that there is no null column, and that the `D` rescale leaves a column
/// averaging 1 — without it every weight would be `~1/D` and the score would
/// collapse by three orders of magnitude while merely looking like a bad fit.
/// The gate is a per-(gene, dim) PROBABILITY — every weight in `(0,1)`, and each
/// entry responds only to its own logit.
///
/// This replaces `every_dim_carries_one_unit_of_mass`, which asserted the softmax's
/// conserved per-dim mass. That property is deliberately gone: it is what made the
/// gate a simplex and therefore incomparable with the sampler's PIP table. Mass is
/// now controlled softly by the learned `π_h`, so there is no exact invariant to
/// assert here — the spread is a MEASUREMENT the fit reports, not a guarantee.
#[test]
fn gate_is_a_per_entry_probability() {
    let vm = VarMap::new();
    let (n_features, h) = (7usize, 3usize);
    let mut m = gated_model(n_features, h, &vm);

    // Lopsided on purpose: dim 0 strongly favours gene 0, dim 2 the last gene.
    let mut lopsided = vec![0f32; n_features * h];
    for g in 0..n_features {
        lopsided[g * h] = if g == 0 { 4.0 } else { -2.0 };
        lopsided[g * h + 2] = if g == n_features - 1 { 3.0 } else { -1.0 };
    }
    m.s_feat = Some(Tensor::from_vec(lopsided, (n_features, h), &dev()).unwrap());

    let logits = m.s_feat.as_ref().expect("gate on");
    assert_eq!(
        logits.dims(),
        &[n_features, h],
        "width is H, no null column"
    );

    let w = m.gate_weights(GateKind::Identity, logits).unwrap();
    let flat = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (i, v) in flat.iter().enumerate() {
        assert!(
            *v > 0.0 && *v < 1.0,
            "entry {i} must be a probability, got {v}"
        );
    }
    // Independence: bumping ONE logit moves ONE weight. Under the old softmax this
    // was impossible — raising a gene necessarily lowered every other in its dim —
    // and that coupling is exactly what made the gate incomparable with a PIP.
    let mut bumped = flat.clone();
    bumped[0] = 0.0; // placeholder, replaced below
    let mut logits2 = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    logits2[0] += 3.0;
    let w2 = m
        .gate_weights(
            GateKind::Identity,
            &Tensor::from_vec(logits2, (n_features, h), &dev()).unwrap(),
        )
        .unwrap();
    let flat2 = w2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    bumped[0] = flat2[0];
    assert!(
        flat2[0] > flat[0] + 1e-4,
        "the bumped entry must rise: {} -> {}",
        flat[0],
        flat2[0]
    );
    for i in 1..flat.len() {
        assert!(
            (flat2[i] - flat[i]).abs() < 1e-6,
            "entry {i} must be untouched by another entry's logit ({} -> {})",
            flat[i],
            flat2[i]
        );
    }
}

/// The terms `single_gate_kl` sums must stay COMMENSURATE — they share one
/// `GATE_KL_WEIGHT`, so a scale gap silently deletes the smaller one. This caught a
/// real regression: when the softmax moved to the gene axis, `α` fell from
/// `≈1/(H+1)` to `≈1/D`, the forward pass compensated via a `D` rescale and the
/// KL did not, and the effect prior came out ~49,000× too small.
///
/// Measured as a RATIO, by differencing two models that differ only in the effect
/// mean `μ`. The Gaussian term is the only one that depends on `μ`, so the gap
/// between them IS that term's contribution — which is what an absolute bound on the
/// total cannot show. (The previous version leaned on the entropy being `≤ 1` by
/// construction; the Bernoulli inclusion KL has no such bound, so that inference no
/// longer holds and the test would have passed vacuously.)
#[test]
fn gate_terms_are_commensurate() {
    let vm = VarMap::new();
    let (n_features, h) = (64usize, 4usize);

    let kl_at = |mu: f32| -> f32 {
        let vm = VarMap::new();
        let mut m = gated_model(n_features, h, &vm);
        m.e_feat_raw =
            Some(Tensor::from_vec(vec![mu; n_features * h], (n_features, h), &dev()).unwrap());
        m.gate_kl()
            .unwrap()
            .expect("gate on")
            .to_vec0::<f32>()
            .unwrap()
    };
    let _ = &vm;

    let flat = kl_at(0.0); // inclusion KL + Beta prior, effect term ~minimal
    let bumped = kl_at(0.7); // + a real Gaussian effect penalty
    assert!(
        flat.is_finite() && bumped.is_finite(),
        "gate KL must be finite, got {flat} / {bumped}"
    );

    let gauss = bumped - flat;
    assert!(
        gauss > 0.0,
        "the Gaussian effect term must respond to μ at all — got {gauss}"
    );
    // Neither term may be more than ~2 orders of magnitude from the other. `flat` is
    // dominated by the inclusion KL; `gauss` is the effect penalty.
    let ratio = (gauss / flat).max(flat / gauss);
    assert!(
        ratio < 100.0,
        "inclusion KL ({flat}) and effect KL ({gauss}) must be within a couple of \
         orders of magnitude — ratio {ratio} means one has swamped or erased the other"
    );
}

/// Gather order no longer matters, because the gate is elementwise.
///
/// The INVERSE of this used to be asserted, and for good reason: a softmax down the
/// gene axis renormalized over whatever subset it was handed, so gathering first
/// trained garbage that still compiled and ran. `gathered_gate_weights` exists to
/// pin that ordering. Under `σ` the hazard is structurally impossible, and this test
/// now records that — if it ever fails, something has reintroduced a cross-gene
/// coupling into the gate and the ordering guard needs to come back.
#[test]
fn gate_order_is_invariant_under_a_sigmoid() {
    let vm = VarMap::new();
    let (n_features, h) = (8usize, 3usize);
    let m = gated_model(n_features, h, &vm);
    let logits = m.s_feat.as_ref().expect("gate on");
    // Non-uniform logits, so agreement is a real result and not a fixture artifact.
    let logits = logits
        .broadcast_add(
            &Tensor::from_vec(
                (0..n_features).map(|i| i as f32 * 0.4).collect::<Vec<_>>(),
                (n_features, 1),
                &dev(),
            )
            .unwrap(),
        )
        .unwrap();

    let idx = Tensor::from_vec(vec![0u32, 1, 2], 3, &dev()).unwrap();
    let gate_then_gather = m
        .gate_weights(GateKind::Identity, &logits)
        .unwrap()
        .index_select(&idx, 0)
        .unwrap();
    let gather_then_gate = m
        .gate_weights(GateKind::Identity, &logits.index_select(&idx, 0).unwrap())
        .unwrap();

    let a = gate_then_gather
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = gather_then_gate
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let maxdiff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        maxdiff < 1e-6,
        "an elementwise gate must give the same answer either order (max diff \
         {maxdiff}); a nonzero gap means the gate couples genes again"
    );
    // The fixture is non-degenerate: the gathered rows are not all equal.
    let spread =
        a.iter().fold(0.0f32, |m, x| m.max(*x)) - a.iter().fold(f32::MAX, |m, x| m.min(*x));
    assert!(
        spread > 1e-3,
        "fixture must vary across rows (spread {spread})"
    );
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

/// An untrained gate is inert **on dim 0 only** — and that is a deliberate change.
///
/// Before the IBP ladder every entry started at `σ(4) ≈ 0.98`, so a fresh model's
/// `Ẽ ≈ β` everywhere. The ladder tilts from the very first step, by construction:
/// that ordering IS the prior, and a prior that only switched on after some warmup
/// would just be a schedule. So the invariant weakens to a graded one.
///
/// What must NOT weaken is the reason the init is not zero: no coordinate may start
/// at a multiplier so small that `β`'s gradient through `α·β` dies, because SGD
/// cannot bring a zeroed coordinate back. The floor is the last dim's `σ(0) = 0.5`
/// — a halving, matching the old worst case, not an extinction.
#[test]
fn untrained_gate_is_inert_on_dim_zero_and_graded_after() {
    let (n_features, h) = (6usize, 4usize);
    let vm = VarMap::new();
    let m = gated_model(n_features, h, &vm);
    let sel = m.feature_selection().unwrap().unwrap();
    assert_eq!(sel.dims(), &[n_features, h], "no null column");

    for row in sel.to_vec2::<f32>().unwrap() {
        assert!(
            row[0] > 0.97 && row[0] < 1.0,
            "dim 0 must still pass the loading through nearly unchanged, got {}",
            row[0]
        );
        for w in row.windows(2) {
            assert!(
                w[1] <= w[0],
                "the ladder must be monotone across dims: {row:?}"
            );
        }
        assert!(
            row[h - 1] >= 0.5 - 1e-5,
            "no dim may start below the sigmoid's midpoint — a smaller multiplier \
             kills beta's gradient and SGD cannot recover it: {row:?}"
        );
    }

    // Applying it to an all-ones base shrinks the tail but never zeroes it.
    let base = Tensor::ones((n_features, h), DType::F32, &dev()).unwrap();
    let w = m
        .gate_weights(GateKind::Identity, m.s_feat.as_ref().unwrap())
        .unwrap();
    let gated = base.mul(&w).unwrap();
    for x in gated.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
        assert!(
            (0.5 - 1e-5..=1.0).contains(&x),
            "gated base out of range: {x}"
        );
    }
}

/// The ladder is stored per DIM and broadcast over rows, and it carries no
/// gradient — `α` is chosen, not fitted, so a `Var` here would let the fit
/// negotiate its own prior away.
#[test]
fn ibp_ladder_is_a_per_dim_constant() {
    let (n_features, h) = (6usize, 4usize);
    let vm = VarMap::new();
    let m = gated_model(n_features, h, &vm);
    let bias = m.gate_ibp_bias.as_ref().expect("gate on");
    assert_eq!(
        bias.dims(),
        &[1, h],
        "the ladder is indexed by DIM and broadcast over rows"
    );
    assert!(
        !vm.data().lock().unwrap().values().any(|v| {
            v.as_tensor().dims() == [1, h]
                && v.as_tensor().to_vec2::<f32>().ok() == bias.to_vec2::<f32>().ok()
        }),
        "the ladder must NOT be registered as a trainable Var"
    );
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
    // Normalize on the FULL table, then gather — gathering first would renormalize
    // over the subset, which is the bug `gathered_gate_weights` exists to prevent.
    let w = m
        .gathered_gate_weights(GateKind::Identity, m.s_feat.as_ref(), &idx)
        .unwrap()
        .unwrap();
    let gated = base.mul(&w).unwrap();
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
    m.enable_feature_gate(
        FeatureGateSpec {
            temperature: 1.0,
            ibp_alpha: None,
        },
        &vm,
        &dev(),
    )
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

/// A SHARING HEAD must keep the gate KL. `train_composite` evaluates `gate_kl` on
/// `ctx.axes[0]`, and with `--phase1-cells-per-pb 0` — the default in both CLIs —
/// that axis is a pb head, not the model the gate was enabled on. `π_h` lives on the
/// model rather than on the shared `s_feat`, so a head that is not handed it returns
/// `None` here and every gate regularisation silently leaves the loss with the build
/// green. That is exactly what shipped; this test is the thing that was missing.
#[test]
fn a_sharing_head_still_has_a_gate_kl() {
    let (n_features, h) = (6usize, 4usize);
    let vm = VarMap::new();
    let m = gated_model(n_features, h, &vm);
    let head = JointEmbedModel::new_sharing_features(
        ShareFeaturesArgs {
            n_cells: 5,
            embedding_dim: h,
            shared_e_feat: m.e_feat.clone(),
            shared_b_feat: m.b_feat.clone(),
            e_cell_init: None,
            b_cell_init: &[0f32; 5],
            var_prefix: "pb_l0",
            seed: 3,
            shared_s_feat: m.s_feat.clone(),
            shared_e_feat_raw: m.e_feat_raw.clone(),
            shared_e_feat_logstd: m.e_feat_logstd.clone(),
            shared_gate_ibp_bias: m.gate_ibp_bias.clone(),
            gate: m.gate,
        },
        &vm,
        &dev(),
    )
    .unwrap();
    let kl = head
        .gate_kl()
        .unwrap()
        .expect("a head that shares the gate must also share its KL");
    let v: f32 = kl.to_scalar().unwrap();
    assert!(v.is_finite(), "head gate KL must be finite, got {v}");
    let base: f32 = m.gate_kl().unwrap().unwrap().to_scalar().unwrap();
    assert!(
        (v - base).abs() < 1e-6,
        "a head shares every gate Var, so its KL must equal the primary's: {v} vs {base}"
    );
}

/// Under jitter the SELECTION is frozen input, but `β` is still trained — so the
/// Gaussian effect KL must survive even though the Bernoulli and Beta terms do not.
/// Returning `None` for the whole thing (as an earlier version did) removes the only
/// shrinkage on the loading for the entire jitter fit.
#[test]
fn jitter_and_learned_agree_when_pip_equals_alpha() {
    let (n_features, h) = (6usize, 4usize);
    let vm = VarMap::new();
    let mut m = gated_model(n_features, h, &vm);
    let learned: f32 = m.gate_kl().unwrap().unwrap().to_scalar().unwrap();

    // Both arms now compute the SAME term and differ only in the weight the
    // likelihood applies: `α` on the learned path, the frozen `pip` on jitter. So
    // handing jitter the learned gate's own `α` must reproduce its value exactly.
    // (This equality is what the selection terms used to break; their removal is
    // precisely what makes it hold.)
    let alpha = m
        .feature_selection()
        .unwrap()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    m.install_gate_pip(GateKind::Identity, &pip_tensor(&alpha, n_features, h))
        .unwrap();
    let jittered: f32 = m
        .gate_kl()
        .unwrap()
        .expect("jitter must still pay the Gaussian effect KL on β")
        .to_scalar()
        .unwrap();
    assert!(jittered.is_finite() && jittered > 0.0);
    assert!(
        (jittered - learned).abs() <= 1e-6 * learned.abs().max(1.0),
        "with pip = alpha the two arms must agree exactly: {jittered} vs {learned}"
    );

    // A zero pip switches every coordinate off, and a coordinate the likelihood never
    // reads must not be charged a prior for its loading.
    m.install_gate_pip(
        GateKind::Identity,
        &pip_tensor(&vec![0.0; n_features * h], n_features, h),
    )
    .unwrap();
    let off: f32 = m.gate_kl().unwrap().unwrap().to_scalar().unwrap();
    assert!(
        off.abs() < 1e-6,
        "a fully masked gate must cost nothing, got {off}"
    );
}

/// The default ladder must leave the LAST dim trainable.
///
/// This is the property the whole `α` choice turns on. The sampler runs the same
/// IBP at `α = 1`, which on 16 dims is an ~11-logit drop; under a gradient that
/// does not produce sparsity, it produces a frozen tail — `α ≈ 0` with
/// `dα/dS ≈ 0`, unrecoverable, because SGD cannot resurrect a zeroed multiplier
/// the way Gibbs can. So: dim 0 near the ungated init, the last dim near the
/// sigmoid's most responsive point, and the gradient there NOT vanishing.
#[test]
fn default_ibp_ladder_keeps_the_tail_trainable() {
    for h in [8usize, 16, 64] {
        let alpha = crate::model::ibp_alpha_for_drop(h, crate::model::GATE_IBP_LOGIT_DROP);
        let bias = crate::model::ibp_gate_logit_bias(alpha, h);

        assert!(bias[0].abs() < 1e-12, "h={h}: dim 0 must be unbiased");
        let drop = -bias[h - 1];
        assert!(
            (drop - crate::model::GATE_IBP_LOGIT_DROP).abs() < 1e-9,
            "h={h}: ladder must span exactly the target drop, got {drop}"
        );

        // What the gate actually sees, at the untrained logit.
        let init = f64::from(4.0f32); // GATE_LOGIT_INIT
        let a_first = 1.0 / (1.0 + (-(init + bias[0])).exp());
        let a_last = 1.0 / (1.0 + (-(init + bias[h - 1])).exp());
        assert!(
            a_first > 0.97,
            "h={h}: dim 0 starts ~ungated, got {a_first}"
        );
        assert!(
            (a_last - 0.5).abs() < 1e-6,
            "h={h}: last dim must start at the sigmoid's most responsive point, got {a_last}"
        );
        // σ'(x) = α(1−α): the tail must still have gradient to move on.
        let slope = a_last * (1.0 - a_last);
        assert!(slope > 0.2, "h={h}: tail gradient is dead ({slope})");
    }
}

/// The ladder must be monotone decreasing and strictly so — that ORDERING is the
/// entire content of the prior. A flat ladder would leave the dims exchangeable,
/// which is the condition an independent `Beta(a,b)` per dim could not escape.
#[test]
fn ibp_ladder_is_strictly_decreasing() {
    let bias = crate::model::ibp_gate_logit_bias(3.0, 12);
    for w in bias.windows(2) {
        assert!(w[1] < w[0], "ladder must decrease: {:?}", bias);
    }
    // Geometric in log-odds ⇒ EQUAL steps. Anything else is not this prior.
    let step = bias[1] - bias[0];
    for w in bias.windows(2) {
        assert!(
            ((w[1] - w[0]) - step).abs() < 1e-12,
            "ladder must be geometric (equal log-odds steps): {:?}",
            bias
        );
    }
    // Smaller α ⇒ steeper ladder ⇒ more pressure toward sparsity.
    let steep = crate::model::ibp_gate_logit_bias(0.5, 12);
    assert!(
        steep[11] < bias[11],
        "smaller alpha must give a steeper ladder"
    );
}

/// The ladder has to reach the LIKELIHOOD, not just sit in a field. If the gate's
/// multiplier ignored it, every claim above would be vacuous — and this is exactly
/// the shape of the bug where cage passed `logits: None` and the gate silently
/// never entered the graph.
#[test]
fn ibp_ladder_tilts_the_gate_weights() {
    let (n_features, h) = (8usize, 8usize);
    let vm = VarMap::new();
    let m = gated_model(n_features, h, &vm);
    let w = m
        .feature_selection()
        .unwrap()
        .expect("gated model reports a selection")
        .to_vec2::<f32>()
        .unwrap();
    for row in &w {
        assert!(
            row[h - 1] < row[0] - 0.4,
            "the gate's own weights must carry the ladder: {row:?}"
        );
    }
}
