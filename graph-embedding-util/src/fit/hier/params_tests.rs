use super::*;
use candle_util::convert::to_host;

fn as_vec(v: &Var) -> Vec<f32> {
    to_host(v.as_tensor()).unwrap()
}

#[test]
fn init_is_seeded_and_biases_are_zero() {
    let dev = Device::Cpu;
    let a = HierParams::new(3, 2, 5, 4, 7, &dev).unwrap();
    let b = HierParams::new(3, 2, 5, 4, 7, &dev).unwrap();
    let c = HierParams::new(3, 2, 5, 4, 8, &dev).unwrap();
    assert_eq!(as_vec(&a.e_u), as_vec(&b.e_u));
    assert_ne!(as_vec(&a.e_u), as_vec(&c.e_u));
    assert_eq!(a.e_u.dims(), &[3, 4]);
    assert_eq!(a.mu.dims(), &[2, 4]);
    assert_eq!(a.r.dims(), &[5, 4]);
    assert!(to_host(a.b_m.as_tensor())
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    assert!(to_host(a.b_g.as_tensor())
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    assert!(as_vec(&a.e_u).iter().all(|x| x.abs() < 1.0));
}

#[test]
fn the_non_base_tracks_start_at_zero_and_leave_the_base_tables_untouched() {
    let dev = Device::Cpu;
    let (n_u, n_m, n_g, h) = (3usize, 2usize, 5usize, 4usize);
    let plain = HierParams::new(n_u, n_m, n_g, h, 7, &dev).unwrap();
    let tracked = HierParams::new_tracked(n_u, n_m, n_g, 3, h, 2, 7, &dev).unwrap();
    assert_eq!(as_vec(&plain.e_u), as_vec(&tracked.e_u));
    assert_eq!(as_vec(&plain.mu), as_vec(&tracked.mu));
    assert_eq!(as_vec(&plain.r), as_vec(&tracked.r));
    assert!(plain.offsets.is_empty());
    assert!(plain.offset(0).is_none());
    assert!(plain.offset(1).is_none());
    assert_eq!(tracked.offsets.len(), 2);
    assert!(tracked.offset(0).is_none());
    for t in 1..3 {
        let o = tracked.offset(t).expect("a non-base track has an offset");
        assert_eq!(o.d_mu.dims(), &[n_m, h]);
        assert_eq!(o.d_b_m.dims(), &[n_m]);
        assert_eq!(o.d_r.u.dims(), &[n_g, 2], "a rank-2 row factor per gene");
        assert_eq!(o.d_r.v.dims(), &[2, h], "one shared factor");
        assert_eq!(o.d_b_g.dims(), &[n_g]);
        assert!(as_vec(&o.d_mu).iter().all(|&x| x == 0.0));
        assert!(o.delta_host().unwrap().iter().all(|&x| x == 0.0));
        assert!(o.d_r_base.is_none() && o.pinned.is_empty());
    }
    assert!(tracked.offset(3).is_none());
}

/// A preset row composes back exactly (`μ_m + r_g = row`), the module mean is
/// the mean of its given rows, and the modes set what they pin.
#[test]
fn preset_rows_compose_back_exactly_and_the_mode_sets_the_pins() {
    let dev = Device::Cpu;
    let (h, module_of) = (2usize, vec![0u32, 0, 1, 1]);
    let given = PresetGenes {
        ids: vec![0, 1, 3],
        rows: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5],
        mode: PresetMode::Freeze,
    };
    let mut p = HierParams::new(2, 2, 4, h, 1, &dev).unwrap();
    p.preset(&given, &module_of).unwrap();
    let (mu, r) = (as_vec(&p.mu), as_vec(&p.r));
    assert_eq!(
        &mu[0..2],
        &[2.0, 3.0],
        "module 0 mean of its two given rows"
    );
    assert_eq!(
        &mu[2..4],
        &[-1.0, 0.5],
        "module 1 mean of its one given row"
    );
    for (i, &g) in given.ids.iter().enumerate() {
        let m = module_of[g as usize] as usize;
        for k in 0..h {
            let composed: f32 = mu[m * h + k] + r[g as usize * h + k];
            assert!((composed - given.rows[i * h + k]).abs() < 1e-6);
        }
    }
    assert!(p.mu_frozen && p.is_frozen_gene(0) && !p.is_frozen_gene(2));
    let mask = to_host(p.r_mask.as_ref().unwrap()).unwrap();
    assert_eq!(mask, vec![0.0, 0.0, 1.0, 0.0]);
    assert!(p.lora.is_none());
    let (rho, _) = p.compose(&[0, 0, 0, 0], &[0, 1, 2, 3], &module_of).unwrap();
    for (i, &g) in given.ids.iter().enumerate() {
        for k in 0..h {
            assert_eq!(rho[(g as usize, k)], given.rows[i * h + k], "verbatim");
        }
    }

    let mut q = HierParams::new(2, 2, 4, h, 1, &dev).unwrap();
    q.preset(
        &PresetGenes {
            mode: PresetMode::Init,
            ..given.clone()
        },
        &module_of,
    )
    .unwrap();
    assert!(!q.mu_frozen && q.r_mask.is_none() && q.frozen_gene.is_empty());

    let mut l = HierParams::new(2, 2, 4, h, 1, &dev).unwrap();
    l.preset(
        &PresetGenes {
            mode: PresetMode::Lora(LoraSpec {
                rank: 1,
                lr_ratio: 4.0,
                ridge: 0.0,
            }),
            ..given.clone()
        },
        &module_of,
    )
    .unwrap();
    let lora = l.lora.as_ref().expect("factors under lora");
    assert_eq!(lora.gene.u.dims(), &[4, 1]);
    assert_eq!(lora.gene.v.dims(), &[1, h]);
    assert_eq!(lora.module.u.dims(), &[2, 1]);
    assert_eq!(lora.module.v.dims(), &[1, h]);
    assert!(as_vec(&lora.module.u).iter().all(|&x| x != 0.0));
    assert!(as_vec(&lora.module.v).iter().all(|&x| x == 0.0));
    assert_eq!(
        to_host(&lora.gene.u_mask).unwrap(),
        vec![1.0, 1.0, 0.0, 1.0]
    );
    let u = as_vec(&lora.gene.u);
    assert!(u[0] != 0.0 && u[1] != 0.0 && u[2] == 0.0 && u[3] != 0.0);
    assert!(
        as_vec(&lora.gene.v).iter().all(|&x| x == 0.0),
        "the residual starts at nothing"
    );
    assert!(l.mu_frozen && l.r_mask.is_some());
    assert!(HierParams::new(2, 2, 4, h, 1, &dev)
        .unwrap()
        .preset(
            &PresetGenes {
                mode: PresetMode::Lora(LoraSpec {
                    rank: h,
                    lr_ratio: 1.0,
                    ridge: 0.0
                }),
                ..given
            },
            &module_of
        )
        .is_err());
}

/// A given offset base `δ₀` on a non-base track: under freeze the residual
/// skips the given genes and their composed track rows are `given base row +
/// δ₀` verbatim, untouched by the module offset and the shared factor; under
/// lora and init the residual stays on every gene and `δ₀` is where it
/// starts. The refusals: an unknown track, a gene out of range, a ragged
/// table, a gene twice, and a pinned offset on a gene whose base row is free.
#[test]
fn a_given_offset_base_composes_on_its_track_and_the_mode_sets_its_pin() {
    let dev = Device::Cpu;
    let (h, module_of) = (2usize, vec![0u32, 0, 1, 1]);
    let given = PresetGenes {
        ids: vec![0, 1, 3],
        rows: vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5],
        mode: PresetMode::Freeze,
    };
    let offsets = vec![PresetOffsets {
        track: 1,
        ids: vec![0, 3],
        rows: vec![0.1, -0.1, 0.2, 0.3],
    }];
    // Genes 0..4 on track 0, genes 0 and 3 on track 1.
    let (track_of_row, gene_of_row) = (vec![0u32, 0, 0, 0, 1, 1], vec![0u32, 1, 2, 3, 0, 3]);
    let near = |a: f32, b: f32| (a - b).abs() < 1e-6;

    let mut p = HierParams::new_tracked(2, 2, 4, 2, h, 1, 1, &dev).unwrap();
    p.preset(&given, &module_of).unwrap();
    p.preset_offsets(&offsets, PresetMode::Freeze).unwrap();
    let o = &p.offsets[0];
    assert_eq!(o.pinned, vec![true, false, false, true]);
    assert_eq!(
        to_host(&o.d_r.u_mask).unwrap(),
        vec![0.0, 1.0, 1.0, 0.0],
        "the residual skips the given genes"
    );
    // Move the module offset and the shared factor: neither reaches a pinned track row.
    o.d_mu
        .set(&Tensor::from_vec(vec![0.5f32; 2 * h], (2, h), &dev).unwrap())
        .unwrap();
    o.d_r
        .v
        .set(&Tensor::from_vec(vec![0.7f32, -0.7], (1, h), &dev).unwrap())
        .unwrap();
    let (rho, _) = p.compose(&track_of_row, &gene_of_row, &module_of).unwrap();
    assert!(near(rho[(4, 0)], 1.1) && near(rho[(4, 1)], 1.9), "{rho}");
    assert!(near(rho[(5, 0)], -0.8) && near(rho[(5, 1)], 0.8), "{rho}");
    let delta = o.delta_host().unwrap();
    assert!(
        near(delta[0], 0.1) && near(delta[1], -0.1),
        "pinned: the base alone"
    );
    assert!(
        delta[2 * h..3 * h].iter().any(|&x| x != 0.0),
        "a free gene's residual moved with V"
    );

    for mode in [
        PresetMode::Lora(LoraSpec {
            rank: 1,
            lr_ratio: 4.0,
            ridge: 0.0,
        }),
        PresetMode::Init,
    ] {
        let mut q = HierParams::new_tracked(2, 2, 4, 2, h, 1, 1, &dev).unwrap();
        q.preset(
            &PresetGenes {
                mode,
                ..given.clone()
            },
            &module_of,
        )
        .unwrap();
        q.preset_offsets(&offsets, mode).unwrap();
        let o = &q.offsets[0];
        assert!(o.pinned.is_empty(), "{mode:?}: nothing pinned on the track");
        assert_eq!(to_host(&o.d_r.u_mask).unwrap(), vec![1.0; 4]);
        let (rho, _) = q.compose(&track_of_row, &gene_of_row, &module_of).unwrap();
        assert!(
            near(rho[(4, 0)], 1.1) && near(rho[(4, 1)], 1.9),
            "{mode:?}: at init the track row is the given row + δ₀: {rho}"
        );
    }

    let bad = |off: PresetOffsets, mode: PresetMode| -> bool {
        let mut p = HierParams::new_tracked(2, 2, 4, 2, h, 1, 1, &dev).unwrap();
        p.preset(&given, &module_of).unwrap();
        p.preset_offsets(&[off], mode).is_err()
    };
    let off = |track: u32, ids: Vec<u32>, n: usize| PresetOffsets {
        track,
        ids,
        rows: vec![0.0; n],
    };
    assert!(
        bad(off(0, vec![0], 2), PresetMode::Freeze),
        "the base track"
    );
    assert!(bad(off(2, vec![0], 2), PresetMode::Freeze), "no such track");
    assert!(
        bad(off(1, vec![4], 2), PresetMode::Freeze),
        "gene out of range"
    );
    assert!(bad(off(1, vec![0], 3), PresetMode::Freeze), "ragged");
    assert!(
        bad(off(1, vec![0, 0], 4), PresetMode::Freeze),
        "a gene twice"
    );
    assert!(
        bad(off(1, vec![2], 2), PresetMode::Freeze),
        "gene 2's base row is free, so its offset cannot be pinned"
    );
    assert!(
        !bad(off(1, vec![2], 2), PresetMode::Init),
        "under init a free base row takes an offset start"
    );
}
