use super::*;

fn host(v: &Var) -> Vec<f32> {
    to_host2(v.as_tensor()).unwrap()
}

#[test]
fn init_is_seeded_and_biases_are_zero() {
    let dev = Device::Cpu;
    let a = HierParams::new(3, 2, 5, 4, 7, &dev).unwrap();
    let b = HierParams::new(3, 2, 5, 4, 7, &dev).unwrap();
    let c = HierParams::new(3, 2, 5, 4, 8, &dev).unwrap();
    assert_eq!(host(&a.e_u), host(&b.e_u));
    assert_ne!(host(&a.e_u), host(&c.e_u));
    assert_eq!(a.e_u.dims(), &[3, 4]);
    assert_eq!(a.mu.dims(), &[2, 4]);
    assert_eq!(a.r.dims(), &[5, 4]);
    assert!(to_host1(a.b_m.as_tensor())
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    assert!(to_host1(a.b_g.as_tensor())
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    assert!(host(&a.e_u).iter().all(|x| x.abs() < 1.0));
}

#[test]
fn the_non_base_tracks_start_at_zero_and_leave_the_base_tables_untouched() {
    let dev = Device::Cpu;
    let (n_u, n_m, n_g, h) = (3usize, 2usize, 5usize, 4usize);
    let plain = HierParams::new(n_u, n_m, n_g, h, 7, &dev).unwrap();
    let tracked = HierParams::new_tracked(n_u, n_m, n_g, 3, h, 7, &dev).unwrap();
    assert_eq!(host(&plain.e_u), host(&tracked.e_u));
    assert_eq!(host(&plain.mu), host(&tracked.mu));
    assert_eq!(host(&plain.r), host(&tracked.r));
    assert!(plain.offsets.is_empty());
    assert!(plain.offset(0).is_none());
    assert!(plain.offset(1).is_none());
    assert_eq!(tracked.offsets.len(), 2);
    assert!(tracked.offset(0).is_none());
    for t in 1..3 {
        let o = tracked.offset(t).expect("a non-base track has an offset");
        assert_eq!(o.d_mu.dims(), &[n_m, h]);
        assert_eq!(o.d_b_m.dims(), &[n_m]);
        assert_eq!(o.d_r.dims(), &[n_g, h]);
        assert_eq!(o.d_b_g.dims(), &[n_g]);
        assert!(host(&o.d_mu).iter().all(|&x| x == 0.0));
        assert!(host(&o.d_r).iter().all(|&x| x == 0.0));
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
    let (mu, r) = (host(&p.mu), host(&p.r));
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
    assert!(p.mu_mask.is_some() && p.is_frozen_gene(0) && !p.is_frozen_gene(2));
    let mask = to_host2(p.r_mask.as_ref().unwrap()).unwrap();
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
    assert!(q.mu_mask.is_none() && q.r_mask.is_none() && q.frozen_gene.is_empty());

    let mut l = HierParams::new(2, 2, 4, h, 1, &dev).unwrap();
    l.preset(
        &PresetGenes {
            mode: PresetMode::Lora {
                rank: 1,
                lr_ratio: 4.0,
                ridge: 0.0,
            },
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
    assert!(host(&lora.module.u).iter().all(|&x| x != 0.0));
    assert!(host(&lora.module.v).iter().all(|&x| x == 0.0));
    assert_eq!(
        to_host2(&lora.gene.u_mask).unwrap(),
        vec![1.0, 1.0, 0.0, 1.0]
    );
    let u = host(&lora.gene.u);
    assert!(u[0] != 0.0 && u[1] != 0.0 && u[2] == 0.0 && u[3] != 0.0);
    assert!(
        host(&lora.gene.v).iter().all(|&x| x == 0.0),
        "the residual starts at nothing"
    );
    assert!(l.mu_mask.is_some() && l.r_mask.is_some());
    assert!(HierParams::new(2, 2, 4, h, 1, &dev)
        .unwrap()
        .preset(
            &PresetGenes {
                mode: PresetMode::Lora {
                    rank: h,
                    lr_ratio: 1.0,
                    ridge: 0.0
                },
                ..given
            },
            &module_of
        )
        .is_err());
}
