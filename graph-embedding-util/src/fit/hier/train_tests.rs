use super::*;
use crate::data::Triplet;
use crate::fit::config::{TrackInfo, TrackSpec};
use crate::fit::hier::units::UnitTable;
use crate::fit::projection::RowCollapse;
use crate::{LoraSpec, PresetMode, PresetOffsets};
use std::sync::atomic::AtomicBool;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

/// The shared baseline; each test overrides only the fields it is about.
fn cfg() -> HierConfig {
    HierConfig {
        n_modules: 2,
        epochs: 200,
        units_per_step: 8,
        modules_per_unit: 2,
        lr: 0.1,
        weight_decay: 0.0,
        unit_weight_decay: 0.0,
        seed: 3,
        offset_l2: 0.0,
        offset_rank: 2,
        device: Device::Cpu,
        module_only: Vec::new(),
        cis_gates: None,
        module_group: Vec::new(),
        background_modules: Vec::new(),
    }
}

/// `train` with the stop flag down.
fn run(
    units: &UnitTable,
    labels: &[u32],
    h: usize,
    cfg: &HierConfig,
    preset: Option<&PresetGenes>,
    offsets: &[PresetOffsets],
) -> anyhow::Result<HierOutput> {
    train(
        units,
        labels,
        h,
        cfg,
        preset,
        offsets,
        &AtomicBool::new(false),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (n(a) * n(b)).max(1e-12)
}

fn row(m: &DMatrix<f32>, r: usize) -> Vec<f32> {
    m.row(r).iter().copied().collect()
}

/// Distinct given rows for `ids`, `h` columns each.
fn given_rows(ids: &[u32], h: usize) -> Vec<f32> {
    ids.iter()
        .flat_map(|&g| (0..h).map(move |k| 0.1 * (g as f32 + 1.0) * (k as f32 - 1.5)))
        .collect()
}

fn preset(ids: &[u32], rows: &[f32], mode: PresetMode) -> PresetGenes {
    PresetGenes {
        ids: ids.to_vec(),
        rows: rows.to_vec(),
        mode,
    }
}

/// Every given gene's row came out exactly as given.
fn assert_rows_pinned(out: &HierOutput, ids: &[u32], rows: &[f32], h: usize) {
    for (i, &g) in ids.iter().enumerate() {
        for k in 0..h {
            assert_eq!(
                out.rho[(g as usize, k)],
                rows[i * h + k],
                "given gene {g} column {k} moved"
            );
        }
    }
}

/// Gene `g`'s trained row is not its random initial residual.
fn left_its_init(
    out: &HierOutput,
    units: &UnitTable,
    cfg: &HierConfig,
    h: usize,
    g: usize,
) -> bool {
    let init = HierParams::new(
        units.n_units(),
        cfg.n_modules,
        out.rho.nrows(),
        h,
        cfg.seed,
        &Device::Cpu,
    )
    .unwrap();
    let r0 = to_host(init.r.as_tensor()).unwrap();
    row(&out.rho, g)
        .iter()
        .zip(&r0[g * h..(g + 1) * h])
        .any(|(a, b)| (a - b).abs() > 1e-6)
}

/// Two planted programs: units 0..10 count genes 0..10, units 10..20 count
/// genes 10..20 (with a little of the other).
fn planted_units() -> (UnitTable, Vec<u32>) {
    let mut trip = Vec::new();
    for u in 0..20u32 {
        let own = if u < 10 { 0..10u32 } else { 10..20u32 };
        let other = if u < 10 { 10..20u32 } else { 0..10u32 };
        for g in own {
            trip.push(t(u, g, 20.0 + (g % 3) as f32));
        }
        for g in other {
            trip.push(t(u, g, 1.0));
        }
    }
    let units = UnitTable::from_pseudobulks_and_cells(&[&trip], &[20], &[], None, 20);
    let labels: Vec<u32> = (0..20u32).map(|g| u32::from(g >= 10)).collect();
    (units, labels)
}

/// The planted programs on two tracks of the same 20 genes, with the programs
/// SWAPPED on track 1, so every gene's track-1 row sits a planted shift away
/// from its track-0 row. Returns the units, the module labels and `n_g`.
fn planted_two_track_units() -> (UnitTable, Vec<u32>, u32) {
    let (n_g, n_u) = (20u32, 20u32);
    let mut trip = Vec::new();
    for u in 0..n_u {
        let own = if u < 10 { 0..10u32 } else { 10..20u32 };
        let other = if u < 10 { 10..20u32 } else { 0..10u32 };
        for g in own.clone() {
            trip.push(t(u, g, 20.0 + (g % 3) as f32));
        }
        for g in other {
            trip.push(t(u, g, 1.0));
            trip.push(t(u, g + n_g, 20.0 + (g % 3) as f32));
        }
        for g in own {
            trip.push(t(u, g + n_g, 1.0));
        }
    }
    let n_features = 2 * n_g as usize;
    let track = |name: &str| TrackInfo {
        name: name.into(),
        is_count: true,
    };
    let tracks = TrackSpec {
        track_of_row: (0..n_features)
            .map(|r| (r >= n_g as usize) as u32)
            .collect(),
        gene_of_row: (0..n_features).map(|r| (r % n_g as usize) as u32).collect(),
        tracks: vec![track("t0"), track("t1")],
    };
    let units = UnitTable::from_pseudobulks_and_cells_tracked(
        &[&trip],
        &[n_u as usize],
        &[],
        None,
        n_features,
        tracks,
    );
    let labels: Vec<u32> = (0..n_g).map(|g| u32::from(g >= 10)).collect();
    (units, labels, n_g)
}

/// Training separates the two unit groups and puts each gene's row on its
/// program's side.
#[test]
fn planted_programs_separate_units_and_genes() {
    let (units, labels) = planted_units();
    let out = run(&units, &labels, 4, &cfg(), None, &[]).unwrap();
    assert_eq!(out.rho.nrows(), 20);
    let e = |u| row(&out.e_u, u);
    assert!(cosine(&e(0), &e(1)) > cosine(&e(0), &e(15)) + 0.3);
    let score = |g: usize, u: usize| {
        row(&out.rho, g)
            .iter()
            .zip(e(u))
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + out.b_feat[g]
    };
    assert!(score(3, 0) > score(3, 15));
    assert!(out.final_loss_per_unit.is_finite());
}

#[test]
fn the_stop_flag_ends_training_early_with_finite_output() {
    let trip = vec![t(0, 0, 3.0), t(0, 1, 1.0), t(1, 1, 4.0)];
    let units = UnitTable::from_pseudobulks_and_cells(&[&trip], &[2], &[], None, 2);
    let cfg = HierConfig {
        n_modules: 1,
        epochs: 1000,
        units_per_step: 2,
        modules_per_unit: 1,
        seed: 1,
        ..cfg()
    };
    let stop = AtomicBool::new(true);
    let out = train(&units, &[0, 0], 2, &cfg, None, &[], &stop).unwrap();
    assert!(out.rho.iter().all(|v| v.is_finite()));
}

/// A unit with a non-zero composition on a track gets pair weights that sum to
/// 1 across every module it lands in ON THAT TRACK; a (unit, track) with an
/// all-zero composition is dropped from the plan entirely.
#[test]
fn draw_plan_weights_sum_to_one_per_unit() {
    let (n_m, n_t) = (3usize, 2usize);
    // (unit 0, track 0): [0.5, 0.5, 0.0]; (unit 0, track 1): [0.0, 0.0, 1.0];
    // (unit 1, track 0): all-zero (nothing counted there); (unit 1, track 1): [0.25, 0.75, 0.0]
    let um = UnitModules {
        n_tracks: n_t,
        n_modules: n_m,
        q: vec![
            0.5, 0.5, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, //
            0.25, 0.75, 0.0,
        ],
        n_um: vec![0.0; 2 * n_t * n_m],
        by_module: vec![Vec::new(), Vec::new()],
    };
    let mut rng = StdRng::seed_from_u64(7);
    let pickers = module_pickers(&um, n_m, &[]);
    assert_eq!(pickers.len(), 2 * n_t);
    let plan = draw_plan(&[0, 1], &pickers, n_m, n_t, 10, &mut rng);
    let mut sum_by_unit_track = std::collections::HashMap::new();
    for ((t, _), pairs) in &plan.pairs_by_module {
        for &(u, w) in pairs {
            *sum_by_unit_track.entry((u, *t)).or_insert(0.0f32) += w;
        }
    }
    for key in [(0u32, 0u32), (0, 1), (1, 1)] {
        let got = sum_by_unit_track.get(&key).copied().unwrap_or(0.0);
        assert!((got - 1.0).abs() < 1e-6, "{key:?} summed to {got}");
    }
    assert!(!sum_by_unit_track.contains_key(&(1, 0)));
}

/// The base track spec is what the untracked constructor builds, so the same
/// fixture trained through either one must come out identical: every table,
/// exactly, not merely close.
#[test]
fn single_track_output_is_identical_through_both_constructors() {
    let trip = vec![
        t(0, 0, 5.0),
        t(0, 1, 2.0),
        t(0, 3, 1.0),
        t(1, 1, 4.0),
        t(1, 2, 3.0),
        t(2, 0, 1.0),
        t(2, 3, 6.0),
    ];
    let cfg = HierConfig {
        epochs: 25,
        units_per_step: 2,
        seed: 17,
        ..cfg()
    };
    let labels = vec![0u32, 0, 1, 1];
    let plain = UnitTable::from_pseudobulks_and_cells(&[&trip], &[3], &[], None, 4);
    let tracked = UnitTable::from_pseudobulks_and_cells_tracked(
        &[&trip],
        &[3],
        &[],
        None,
        4,
        TrackSpec::base(4),
    );
    let a = run(&plain, &labels, 3, &cfg, None, &[]).unwrap();
    let b = run(&tracked, &labels, 3, &cfg, None, &[]).unwrap();
    assert_eq!(a.e_u, b.e_u);
    assert_eq!(a.rho, b.rho);
    assert_eq!(a.b_feat, b.b_feat);
}

/// On the swapped second track a gene's track-1 row sits on the other unit
/// group's side: the move from its track-0 row points along the planted
/// contrast between the two unit groups' embeddings.
#[test]
fn planted_two_track_programs() {
    let (units, labels, n_g) = planted_two_track_units();
    let cfg = HierConfig {
        offset_l2: 0.01,
        ..cfg()
    };
    let out = run(&units, &labels, 4, &cfg, None, &[]).unwrap();
    assert_eq!(out.rho.nrows(), 2 * n_g as usize);
    assert_eq!(out.b_feat.len(), 2 * n_g as usize);

    let group_mean = |lo: usize, hi: usize| -> Vec<f32> {
        (0..4)
            .map(|k| (lo..hi).map(|u| out.e_u[(u, k)]).sum::<f32>() / (hi - lo) as f32)
            .collect()
    };
    let (a, b) = (group_mean(0, 10), group_mean(10, 20));
    let planted: Vec<f32> = b.iter().zip(&a).map(|(x, y)| x - y).collect();
    for gene in [0usize, 3, 7] {
        let shift: Vec<f32> = (0..4)
            .map(|k| out.rho[(gene + n_g as usize, k)] - out.rho[(gene, k)])
            .collect();
        let c = cosine(&shift, &planted);
        assert!(c > 0.5, "gene {gene}: track-1 shift cosine {c}");
    }
}

/// `HierConfig::offset_l2` is a per-EPOCH weight. A step carries `1/S` of it, so
/// the ridge pulls equally hard over an epoch whatever the batch size; without
/// this, halving `units_per_step` would silently double the penalty and inflate
/// every offset row's Adagrad accumulator twice as fast.
#[test]
fn the_ridge_weight_is_spread_over_the_epochs_steps() {
    assert_eq!(per_step_offset_l2(0.8, 1), 0.8);
    assert_eq!(per_step_offset_l2(0.8, 4), 0.2);
    // A degenerate epoch (no unit, so no step) must not divide by zero.
    assert_eq!(per_step_offset_l2(0.8, 0), 0.8);
    assert_eq!(per_step_offset_l2(0.0, 7), 0.0);
}

/// Frozen genes come out of training with EXACTLY the rows they went in with;
/// the free genes still move, and every per-gene bias still trains.
#[test]
fn frozen_gene_rows_survive_training_verbatim_while_free_rows_and_biases_move() {
    let (units, labels) = planted_units();
    let h = 4;
    let ids: Vec<u32> = (0..20u32).filter(|g| g % 2 == 0).collect();
    let rows = given_rows(&ids, h);
    let cfg = HierConfig {
        epochs: 50,
        weight_decay: 0.01,
        ..cfg()
    };
    let frozen = preset(&ids, &rows, PresetMode::Freeze);
    let out = run(&units, &labels, h, &cfg, Some(&frozen), &[]).unwrap();
    assert_rows_pinned(&out, &ids, &rows, h);
    assert!(
        left_its_init(&out, &units, &cfg, h, 1),
        "a free gene trains"
    );
    assert!(ids.iter().any(|&g| out.b_feat[g as usize].abs() > 1e-6));
    assert!(out.final_loss_per_unit.is_finite());
}

/// A frozen table that covers every gene fixes the whole dictionary; the unit
/// side still separates the planted programs against it.
#[test]
fn a_fully_frozen_dictionary_still_trains_the_unit_side() {
    let (units, labels) = planted_units();
    let h = 4;
    let ids: Vec<u32> = (0..20u32).collect();
    // Program A genes along +e0, program B genes along +e1.
    let rows: Vec<f32> = ids
        .iter()
        .flat_map(|&g| {
            let mut r = vec![0.0f32; h];
            r[usize::from(g >= 10)] = 1.0;
            r[2] = 0.01 * g as f32;
            r
        })
        .collect();
    let cfg = HierConfig {
        epochs: 100,
        ..cfg()
    };
    let frozen = preset(&ids, &rows, PresetMode::Freeze);
    let out = run(&units, &labels, h, &cfg, Some(&frozen), &[]).unwrap();
    assert_rows_pinned(&out, &ids, &rows, h);
    assert!(out.e_u[(0, 0)] > out.e_u[(0, 1)]);
    assert!(out.e_u[(15, 1)] > out.e_u[(15, 0)]);
}

#[test]
fn frozen_genes_must_be_in_range_and_match_h() {
    let (units, labels) = planted_units();
    let cfg = HierConfig { epochs: 1, ..cfg() };
    let bad_gene = preset(&[20], &[0.0; 4], PresetMode::Freeze);
    assert!(run(&units, &labels, 4, &cfg, Some(&bad_gene), &[]).is_err());
    let bad_h = preset(&[0], &[0.0; 3], PresetMode::Freeze);
    assert!(run(&units, &labels, 4, &cfg, Some(&bad_h), &[]).is_err());
}

/// Under `Init` the given rows are only the starting point: training starts
/// from the given row exactly, then moves it.
#[test]
fn unfrozen_preset_rows_start_where_given_and_then_train() {
    let (units, labels) = planted_units();
    let h = 4;
    let ids: Vec<u32> = (0..20u32).collect();
    let rows: Vec<f32> = (0..20 * h).map(|i| 0.01 * i as f32 - 0.4).collect();
    let init = preset(&ids, &rows, PresetMode::Init);
    let cfg0 = HierConfig { epochs: 0, ..cfg() };
    let start = run(&units, &labels, h, &cfg0, Some(&init), &[]).unwrap();
    for g in 0..20 {
        for k in 0..h {
            assert!((start.rho[(g, k)] - rows[g * h + k]).abs() < 1e-6);
        }
    }
    let cfg = HierConfig { epochs: 50, ..cfg0 };
    let out = run(&units, &labels, h, &cfg, Some(&init), &[]).unwrap();
    let moved = (0..20)
        .filter(|&g| (0..h).any(|k| (out.rho[(g, k)] - rows[g * h + k]).abs() > 1e-4))
        .count();
    assert!(moved > 10, "only {moved} of 20 preset rows trained");
}

/// Under LoRA the anchored genes move only through two shared rank-r
/// residuals, one per module and one per gene: `out − given` on those genes
/// is not zero and has rank ≤ 2r (H = 4 leaves room at rank 1), and the free
/// genes and the biases still train. Rank ≥ H is refused.
#[test]
fn lora_preset_rows_move_only_inside_a_shared_rank_r_residual() {
    let (units, labels) = planted_units();
    let (h, rank) = (4, 1);
    let ids: Vec<u32> = (0..20u32).filter(|g| g % 2 == 0).collect();
    let rows = given_rows(&ids, h);
    let cfg = HierConfig {
        epochs: 50,
        weight_decay: 0.01,
        ..cfg()
    };
    let lora = |rank, lr_ratio| {
        preset(
            &ids,
            &rows,
            PresetMode::Lora(LoraSpec {
                rank,
                lr_ratio,
                ridge: 0.0,
            }),
        )
    };
    let out = run(&units, &labels, h, &cfg, Some(&lora(rank, 4.0)), &[]).unwrap();
    let resid = DMatrix::<f32>::from_fn(ids.len(), h, |i, k| {
        out.rho[(ids[i] as usize, k)] - rows[i * h + k]
    });
    let sv = resid.singular_values();
    assert!(sv[0] > 1e-4, "the residual never moved: {sv}");
    assert!(
        sv[2 * rank] <= 1e-4 * sv[0],
        "the residual is not rank {}: singular values {sv}",
        2 * rank
    );
    assert!(
        left_its_init(&out, &units, &cfg, h, 1),
        "a free gene trains"
    );
    assert!(ids.iter().any(|&g| out.b_feat[g as usize].abs() > 1e-6));
    assert!(out.final_loss_per_unit.is_finite());

    assert!(run(&units, &labels, h, &cfg, Some(&lora(h, 1.0)), &[]).is_err());
}

/// The track offsets' rank is its own number, never H: outside `1..=H` on a
/// tracked axis the trainer refuses and names both; at one track it is inert.
#[test]
fn the_offset_rank_is_checked_against_h_on_a_tracked_axis() {
    let (units, labels, _) = planted_two_track_units();
    let cfg = |offset_rank| HierConfig {
        epochs: 1,
        offset_l2: 0.01,
        offset_rank,
        ..cfg()
    };
    for rank in [0usize, 5] {
        let e = match run(&units, &labels, 4, &cfg(rank), None, &[]) {
            Ok(_) => panic!("rank {rank} outside 1..=H was accepted"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("rank") && e.contains("H=4"), "{e}");
    }
    assert!(
        run(&units, &labels, 4, &cfg(4), None, &[]).is_ok(),
        "rank H is legal"
    );
    let (one, labels1) = planted_units();
    assert!(
        run(&one, &labels1, 4, &cfg(9), None, &[]).is_ok(),
        "one track: no offset, no rank to check"
    );
}

/// A preset on a two-track axis: the pinned base rows come out verbatim, a
/// given offset base on track 1 comes out as `row + δ₀` verbatim under
/// freeze, the other genes' track-1 rows still move off their base rows, and
/// under lora the given offset is a start rather than a pin.
#[test]
fn a_preset_on_a_two_track_axis_pins_the_base_rows_and_a_given_offset() {
    let (units, labels, n_g) = planted_two_track_units();
    let h = 4;
    let ids: Vec<u32> = (0..n_g).filter(|g| g % 2 == 0).collect();
    let rows = given_rows(&ids, h);
    let frozen = preset(&ids, &rows, PresetMode::Freeze);
    let off_ids = vec![0u32, 4];
    let off_rows = vec![0.3f32, -0.1, 0.2, 0.0, -0.2, 0.1, 0.4, -0.3];
    let offsets = vec![PresetOffsets {
        track: 1,
        ids: off_ids.clone(),
        rows: off_rows.clone(),
    }];
    let cfg = HierConfig {
        epochs: 50,
        offset_l2: 0.01,
        ..cfg()
    };
    let out = run(&units, &labels, h, &cfg, Some(&frozen), &offsets).unwrap();
    let n_g = n_g as usize;
    assert_rows_pinned(&out, &ids, &rows, h);
    for (i, &g) in off_ids.iter().enumerate() {
        let j = ids.iter().position(|&x| x == g).unwrap();
        for k in 0..h {
            let want = rows[j * h + k] + off_rows[i * h + k];
            let got = out.rho[(g as usize + n_g, k)];
            assert!(
                (got - want).abs() < 1e-6,
                "gene {g} on track 1: {got} vs given {want}"
            );
        }
    }
    assert!(
        (0..h).any(|k| (out.rho[(1 + n_g, k)] - out.rho[(1, k)]).abs() > 1e-4),
        "a free gene's track-1 row moves off its base row"
    );
    assert!(out.final_loss_per_unit.is_finite());

    let lora = PresetGenes {
        mode: PresetMode::Lora(LoraSpec {
            rank: 1,
            lr_ratio: 4.0,
            ridge: 0.0,
        }),
        ..frozen
    };
    let out = run(&units, &labels, h, &cfg, Some(&lora), &offsets).unwrap();
    assert!(
        off_ids.iter().enumerate().any(|(i, &g)| {
            (0..h).any(|k| {
                (out.rho[(g as usize + n_g, k)] - out.rho[(g as usize, k)] - off_rows[i * h + k])
                    .abs()
                    > 1e-4
            })
        }),
        "under lora the given offset moves on from δ₀"
    );
}

/// Two planted programs over a residual block (features 0..10) and a
/// module-only block (10..30), each program in its own module per block.
fn module_only_fixture() -> (UnitTable, Vec<u32>, Vec<bool>) {
    let program = |f: u32| if f < 10 { f < 5 } else { f < 20 };
    let mut trip = Vec::new();
    for u in 0..20u32 {
        let a = u < 10;
        for f in 0..30u32 {
            let c = if program(f) == a {
                20.0 + (f % 3) as f32
            } else {
                1.0
            };
            trip.push(t(u, f, c));
        }
    }
    let units = UnitTable::from_pseudobulks_and_cells(&[&trip], &[20], &[], None, 30);
    let labels: Vec<u32> = (0..30u32)
        .map(|f| match (f < 10, program(f)) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        })
        .collect();
    let mo: Vec<bool> = (0..30).map(|f| f >= 10).collect();
    (units, labels, mo)
}

fn module_only_cfg(module_only: Vec<bool>) -> HierConfig {
    HierConfig {
        n_modules: 4,
        seed: 5,
        module_only,
        ..cfg()
    }
}

/// A module-only module's biases are log shares that sum to 1, so the
/// collapsed row bias predict rebuilds (LSE over members) is 0, whether no
/// member was observed (uniform `-ln n`) or only some were (the observed one
/// takes almost all the mass).
#[test]
fn a_module_only_modules_biases_are_log_shares_even_when_unobserved() {
    // Residual feature 0 in module 0; module-only features 1..4 in module 1.
    let labels = vec![0u32, 1, 1, 1];
    let mo = vec![false, true, true, true];
    let cfg = HierConfig {
        module_only: mo.clone(),
        ..cfg()
    };
    let collapse = RowCollapse::from_modules(&mo, &labels).unwrap();
    let biases = |trip: &[Triplet]| {
        let units = UnitTable::from_pseudobulks_and_cells(&[trip], &[2], &[], None, 4);
        let s = ModuleOnly::new(&units, &labels, &cfg).unwrap().unwrap();
        assert_eq!(s.genes, vec![1, 2, 3]);
        let mut b_feat = vec![0f32; 4];
        for (&g, &v) in s.genes.iter().zip(&s.bias) {
            b_feat[g as usize] = v;
        }
        let (_, rb) = collapse.reduce_dictionary(&[0f32; 4], &b_feat, 1);
        let lse = rb[collapse.row_of[1] as usize];
        assert!(lse.abs() < 1e-4, "LSE(bias) = {lse}, want 0; {:?}", s.bias);
        s.bias
    };

    for b in biases(&[t(0, 0, 10.0), t(1, 0, 10.0)]) {
        assert!((b + 3f32.ln()).abs() < 1e-5, "{b} vs -ln 3");
    }
    let some = biases(&[t(0, 0, 5.0), t(0, 1, 100.0), t(1, 1, 100.0)]);
    assert!(some[0] > some[1] + 5.0, "{some:?}");
}

#[test]
fn a_module_only_feature_is_its_module_row_plus_its_count_share() {
    let (units, labels, mo) = module_only_fixture();
    let out = run(&units, &labels, 4, &module_only_cfg(mo), None, &[]).unwrap();
    // Every module-only feature carries exactly its module's row.
    for f in 11..20 {
        assert_eq!(row(&out.rho, 10), row(&out.rho, f), "feature {f}");
    }
    for f in 21..30 {
        assert_eq!(row(&out.rho, 20), row(&out.rho, f), "feature {f}");
    }
    assert_ne!(row(&out.rho, 10), row(&out.rho, 20));
    // Within a module the bias differences are the log count ratios.
    let total = |f: u32| -> f32 {
        units.feats[..units.n_pb_units]
            .iter()
            .zip(&units.counts)
            .flat_map(|(fs, cs)| fs.iter().zip(cs))
            .filter(|(&g, _)| g == f)
            .map(|(_, &c)| c)
            .sum()
    };
    let got = out.b_feat[11] - out.b_feat[12];
    let want = (total(11) / total(12)).ln();
    assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    // Residual features keep a residual of their own: not all rows alike.
    assert_ne!(row(&out.rho, 0), row(&out.rho, 1));
    assert_eq!(out.labels, labels, "the membership does not move");
}

/// A pinning preset holds only the modules it has members in. The module-only
/// modules here have none, and a module-only feature has no residual to absorb
/// a held module vector, so they must train: their rows leave the random start.
#[test]
fn a_pinning_preset_leaves_modules_without_given_members_free_to_train() {
    let (units, labels, mo) = module_only_fixture();
    let h = 4;
    let ids: Vec<u32> = (0..10).collect();
    let rows = given_rows(&ids, h);
    let cfg = module_only_cfg(mo);
    let frozen = preset(&ids, &rows, PresetMode::Freeze);
    let out = run(&units, &labels, h, &cfg, Some(&frozen), &[]).unwrap();
    assert_rows_pinned(&out, &ids, &rows, h);
    let init = HierParams::new(units.n_units(), 4, 30, h, cfg.seed, &Device::Cpu).unwrap();
    let mu0 = to_host(init.mu.as_tensor()).unwrap();
    for (f, m) in [(10usize, 2usize), (20, 3)] {
        let r = row(&out.rho, f);
        assert!(
            r.iter()
                .zip(&mu0[m * h..(m + 1) * h])
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "module {m} never left its random start: {r:?}"
        );
    }
    assert_ne!(row(&out.rho, 10), row(&out.rho, 20));
}

/// A module-only feature carries no residual, so a pinning preset cannot hold
/// its own row: it holds its MODULE at the mean of the given rows, and the
/// written row is that module row (the one training used) for every member.
#[test]
fn a_pinned_module_only_feature_is_written_as_its_modules_given_mean() {
    let (units, labels, mo) = module_only_fixture();
    let h = 4;
    let ids: Vec<u32> = (0..30).collect();
    let rows = given_rows(&ids, h);
    let frozen = preset(&ids, &rows, PresetMode::Freeze);
    let out = run(&units, &labels, h, &module_only_cfg(mo), Some(&frozen), &[]).unwrap();
    assert_rows_pinned(&out, &ids[..10], &rows, h);
    for members in [10..20usize, 20..30] {
        let n = members.len() as f32;
        for k in 0..h {
            let mean: f32 = members.clone().map(|g| rows[g * h + k]).sum::<f32>() / n;
            for g in members.clone() {
                assert!(
                    (out.rho[(g, k)] - mean).abs() < 1e-5,
                    "module-only feature {g} column {k}: {} vs module mean {mean}",
                    out.rho[(g, k)]
                );
            }
        }
    }
}

#[test]
fn a_module_mixing_module_only_and_residual_features_is_refused() {
    let (units, mut labels, mo) = module_only_fixture();
    labels[12] = 0; // a module-only feature in a residual module
    let err = run(&units, &labels, 4, &module_only_cfg(mo), None, &[])
        .err()
        .expect("a mixed module must be refused");
    assert!(err.to_string().contains("module-only"), "{err}");
}

/// The threaded step is the serial step: slice seeds are drawn from the step
/// rng in slice order, so running the slices one by one with those seeds
/// gives the same losses and the same summed gradients.
#[test]
fn a_threaded_step_sums_the_slices_losses_and_gradients() {
    use crate::fit::hier::params::HierParams;
    use crate::fit::hier::partition::{Partition, TrackSupport, UnitModules};
    use crate::fit::hier::step::{step_loss, StepCtx};
    use legume_numeric::candle::convert::to_host;
    use rand::Rng;

    let (units, labels) = planted_units();
    let part = Partition::from_labels(&labels, 2);
    let um = UnitModules::new(&units, &part);
    let sup = TrackSupport::new(&units.tracks, &part);
    let pickers = module_pickers(&um, 2, &[]);
    let params = HierParams::new(units.n_units(), 2, 20, 4, 5, &Device::Cpu).unwrap();
    let skip = Vec::new();
    let ctx = StepCtx {
        units: &units,
        um: &um,
        part: &part,
        sup: &sup,
        skip_module: &skip,
    };
    let chunk: Vec<u32> = (0..12).collect();
    let loss_of = |slice: &[u32], rng: &mut StdRng| -> anyhow::Result<(StepStats, Tensor)> {
        let plan = draw_plan(slice, &pickers, 2, 1, 2, rng);
        step_loss(&params, &ctx, &plan, 0.0, 0.0, None)
    };
    let mut rng = StdRng::seed_from_u64(11);
    let (stats, grads) = step_grads(&chunk, 3, &mut rng, &loss_of).unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    let seeds: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect();
    let mut want_module = 0.0;
    let mut want_e_u: Option<Vec<f32>> = None;
    for (slice, &seed) in chunk.chunks(4).zip(&seeds) {
        let mut r = StdRng::seed_from_u64(seed);
        let (s, loss) = loss_of(slice, &mut r).unwrap();
        want_module += s.loss_module;
        let g = loss.backward().unwrap();
        let ge = to_host(g.get(&params.e_u).unwrap()).unwrap();
        want_e_u = Some(match want_e_u {
            None => ge,
            Some(acc) => acc.iter().zip(&ge).map(|(a, b)| a + b).collect(),
        });
    }
    assert!(
        (stats.loss_module - want_module).abs() < 1e-4,
        "{} vs {want_module}",
        stats.loss_module
    );
    let got = to_host(grads.get(&params.e_u).unwrap()).unwrap();
    for (a, b) in got.iter().zip(want_e_u.unwrap()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn a_step_over_fewer_units_than_threads_runs_on_one_slice() {
    use crate::fit::hier::params::HierParams;
    use crate::fit::hier::partition::{Partition, TrackSupport, UnitModules};
    use crate::fit::hier::step::{step_loss, StepCtx};

    let (units, labels) = planted_units();
    let part = Partition::from_labels(&labels, 2);
    let um = UnitModules::new(&units, &part);
    let sup = TrackSupport::new(&units.tracks, &part);
    let pickers = module_pickers(&um, 2, &[]);
    let params = HierParams::new(units.n_units(), 2, 20, 4, 5, &Device::Cpu).unwrap();
    let skip = Vec::new();
    let ctx = StepCtx {
        units: &units,
        um: &um,
        part: &part,
        sup: &sup,
        skip_module: &skip,
    };
    let loss_of = |slice: &[u32], rng: &mut StdRng| -> anyhow::Result<(StepStats, Tensor)> {
        let plan = draw_plan(slice, &pickers, 2, 1, 2, rng);
        step_loss(&params, &ctx, &plan, 0.0, 0.0, None)
    };
    let mut rng = StdRng::seed_from_u64(3);
    let (stats, grads) = step_grads(&[0, 1], 8, &mut rng, &loss_of).unwrap();
    assert!(stats.loss_module.is_finite());
    assert!(grads.get(&params.e_u).is_some());
}

#[test]
fn only_a_cpu_device_slices_a_step_across_threads() {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    assert_eq!(step_threads(&Device::Cpu), cores);
}

/// Cis gates on the planted programs: every gene of one program paired with
/// the same-index gene of the other, at alignment weight `align_weight` and
/// mixture share `mix`.
fn planted_gates(align_weight: f32, mix: f32) -> crate::fit::hier::CisCoupling {
    let n_pairs = 20usize;
    crate::fit::hier::CisCoupling {
        pairs: crate::fit::hier::CisGates {
            gene_feat: (0..20u32).collect(),
            peak_feat: (0..20u32).map(|g| (g + 10) % 20).collect(),
            abc: vec![1.0; n_pairs],
            z_log_contact: (0..n_pairs).map(|k| k as f32 / 10.0 - 1.0).collect(),
        },
        align_weight,
        mix,
    }
}

fn fit_with_gates(align_weight: f32, epochs: usize) -> HierOutput {
    fit_with_mixed_gates(align_weight, 0.0, epochs)
}

fn fit_with_mixed_gates(align_weight: f32, mix: f32, epochs: usize) -> HierOutput {
    let (units, labels) = planted_units();
    run(
        &units,
        &labels,
        4,
        &HierConfig {
            epochs,
            cis_gates: Some(planted_gates(align_weight, mix)),
            ..cfg()
        },
        None,
        &[],
    )
    .unwrap()
}

/// The mixture and the alignment together: the fit trains the gates and
/// writes a finite dictionary that differs from the unmixed one.
#[test]
fn a_fit_with_a_mixture_and_alignment_trains() {
    let mixed = fit_with_mixed_gates(1.0, 0.5, 30);
    let plain = fit_with_mixed_gates(1.0, 0.0, 30);
    let cis = mixed.cis.as_ref().expect("a cis readout");
    assert!(mixed.rho.iter().all(|v| v.is_finite()));
    assert!(mixed.final_loss_per_unit.is_finite());
    assert!((cis.theta0 - 1.0).abs() + (cis.theta1 - 0.5).abs() > 1e-3);
    let diff = (&mixed.rho - &plain.rho).abs().max();
    assert!(diff > 1e-4, "the mixture left the dictionary unchanged");
}

/// A fit moves the shared gate scalars off their start (the gates train
/// through the alignment) and returns a finite share and evidence per pair.
#[test]
fn a_fit_with_cis_gates_trains_the_gate_scalars() {
    let n_pairs = 20usize;
    let out = fit_with_gates(1.0, 30);
    let cis = out.cis.expect("a cis readout");
    assert_eq!(cis.w.len(), n_pairs);
    assert!(
        cis.w.iter().all(|w| w.is_finite() && *w >= 0.0),
        "{:?}",
        cis.w
    );
    let moved = (cis.theta0 - 1.0).abs() + (cis.theta1 - 0.5).abs();
    assert!(moved > 1e-3, "the gates never moved: {cis:?}");
    assert!(cis.align_gap.is_finite(), "gap {}", cis.align_gap);
    assert_eq!(cis.corr.len(), n_pairs);
    assert!(
        cis.corr
            .iter()
            .all(|c| c.is_finite() && c.abs() <= 1.0 + 1e-5),
        "{:?}",
        cis.corr
    );
}

/// The alignment does its job: the planted pairs join each gene to a gene of
/// the OTHER program, so without alignment its profile sits far from its
/// pooled row; a strong weight closes most of that gap.
#[test]
fn the_alignment_weight_closes_the_gene_to_atac_gap() {
    let free = fit_with_gates(0.0, 60).cis.unwrap().align_gap;
    let aligned = fit_with_gates(10.0, 60).cis.unwrap().align_gap;
    assert!(
        aligned < 0.5 * free,
        "gap without alignment {free}, with {aligned}"
    );
}

/// Alignment off, the gene likelihood is the fit without gates: the gates
/// only read the tables.
#[test]
fn at_zero_weight_the_gates_leave_the_fit_alone() {
    let (units, labels) = planted_units();
    let plain = run(
        &units,
        &labels,
        4,
        &HierConfig {
            epochs: 20,
            ..cfg()
        },
        None,
        &[],
    )
    .unwrap();
    let gated = fit_with_gates(0.0, 20);
    let diff = (&plain.rho - &gated.rho).abs().max();
    assert!(diff < 1e-5, "the dictionary moved by {diff}");
}

/// One unit on one track whose composition puts most of its counts on a
/// module-only module (module 0).
fn skewed_unit_modules() -> UnitModules {
    UnitModules {
        n_tracks: 1,
        n_modules: 3,
        q: vec![0.6, 0.3, 0.1],
        n_um: vec![0.0; 3],
        by_module: vec![Vec::new()],
    }
}

/// A module-only module has no gene level, so a draw that lands there is
/// wasted: draws go only to modules with a gene level, and a unit's pair
/// weights sum to its share of counts on those modules.
#[test]
fn module_draws_skip_module_only_modules() {
    let um = skewed_unit_modules();
    let skip = [true, false, false];
    let pickers = module_pickers(&um, 3, &skip);
    let mut rng = StdRng::seed_from_u64(5);
    let plan = draw_plan(&[0], &pickers, 3, 1, 8, &mut rng);
    let mut total = 0.0f32;
    for ((_, m), pairs) in &plan.pairs_by_module {
        assert_ne!(*m, 0, "a draw landed on the module-only module");
        total += pairs.iter().map(|p| p.1).sum::<f32>();
    }
    assert!(
        (total - 0.4).abs() < 1e-6,
        "weights sum to {total}, not the residual share"
    );
}

/// The re-weighted draw estimates the same gene-level sum as drawing over
/// every module: `E[Σ_pairs w·f(m)] = Σ_{m with a gene level} q_m·f(m)`.
#[test]
fn residual_module_draws_are_unbiased() {
    let um = skewed_unit_modules();
    let skip = [true, false, false];
    let f = [100.0f64, 2.0, 7.0];
    let want: f64 = [1usize, 2].iter().map(|&m| f64::from(um.q[m]) * f[m]).sum();
    let pickers = module_pickers(&um, 3, &skip);
    let mut rng = StdRng::seed_from_u64(9);
    let n = 20_000;
    let mut got = 0.0f64;
    for _ in 0..n {
        let plan = draw_plan(&[0], &pickers, 3, 1, 4, &mut rng);
        for ((_, m), pairs) in &plan.pairs_by_module {
            got += pairs
                .iter()
                .map(|p| f64::from(p.1) * f[*m as usize])
                .sum::<f64>();
        }
    }
    got /= f64::from(n);
    assert!((got - want).abs() < 0.02 * want, "{got} vs {want}");
}

/// A unit whose counts sit only on module-only modules draws nothing.
#[test]
fn a_unit_with_only_module_only_counts_draws_nothing() {
    let um = skewed_unit_modules();
    let pickers = module_pickers(&um, 3, &[true, true, true]);
    assert!(pickers[0].is_none());
}

/// A two-modality axis: RNA features `0..20` and ATAC features `20..40`, two
/// programs mirrored in both (units `0..20` count RNA `0..10` + ATAC `20..30`,
/// the rest the other halves), and every unit's ATAC counts scaled by a
/// planted ratio independent of its program. Modules `0, 1` are RNA, `2, 3`
/// ATAC. Returns the units, labels, per-module group, and `ln ratio` per unit.
fn planted_ratio_multiome() -> (UnitTable, Vec<u32>, Vec<u32>, Vec<f32>) {
    let n_u = 40u32;
    let log_ratio: Vec<f32> = (0..n_u).map(|u| 1.5 * (u as f32 * 1.7).sin()).collect();
    let mut trip = Vec::new();
    for u in 0..n_u {
        let a = u < n_u / 2;
        let scale = log_ratio[u as usize].exp();
        for g in 0..20u32 {
            let own = (g < 10) == a;
            let c = if own { 20.0 + (g % 3) as f32 } else { 1.0 };
            trip.push(t(u, g, c));
            trip.push(t(u, g + 20, c * scale));
        }
    }
    let units = UnitTable::from_pseudobulks_and_cells(&[&trip], &[n_u as usize], &[], None, 40);
    let labels: Vec<u32> = (0..40u32).map(|g| g / 10).collect();
    (units, labels, vec![0, 0, 1, 1], log_ratio)
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let rank = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0f32; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f32;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let m = (ra.len() as f32 - 1.0) / 2.0;
    let (mut sab, mut saa, mut sbb) = (0f32, 0f32, 0f32);
    for (x, y) in ra.iter().zip(&rb) {
        sab += (x - m) * (y - m);
        saa += (x - m) * (x - m);
        sbb += (y - m) * (y - m);
    }
    sab / (saa * sbb).sqrt()
}

/// R² of `y` regressed on the columns of `x` plus an intercept.
fn r_squared(x: &DMatrix<f32>, y: &[f32]) -> f32 {
    let n = x.nrows();
    let xi = DMatrix::<f64>::from_fn(n, x.ncols() + 1, |i, j| {
        if j == 0 {
            1.0
        } else {
            f64::from(x[(i, j - 1)])
        }
    });
    let yv = nalgebra::DVector::<f64>::from_iterator(n, y.iter().map(|&v| f64::from(v)));
    let beta = (xi.transpose() * &xi).try_inverse().expect("full rank") * xi.transpose() * &yv;
    let res = &yv - &xi * beta;
    let mean = yv.mean();
    let ss: f64 = yv.iter().map(|v| (v - mean).powi(2)).sum();
    (1.0 - res.norm_squared() / ss) as f32
}

/// With one intercept per unit and module group, a unit's ATAC:RNA count
/// ratio is absorbed by that intercept instead of being written into the unit
/// embedding: the intercept tracks the planted ratio.
#[test]
fn a_group_intercept_tracks_each_units_modality_ratio() {
    let (units, labels, group, log_ratio) = planted_ratio_multiome();
    let out = run(
        &units,
        &labels,
        4,
        &HierConfig {
            n_modules: 4,
            module_group: group,
            ..cfg()
        },
        None,
        &[],
    )
    .unwrap();
    let beta = out.group_intercepts.expect("group intercepts");
    assert_eq!((beta.nrows(), beta.ncols()), (40, 1));
    let b: Vec<f32> = beta.column(0).iter().copied().collect();
    let s = spearman(&b, &log_ratio);
    assert!(s > 0.9, "intercept vs planted ln ratio: Spearman {s}");
}

/// Without the intercept the unit embedding has to carry the ratio; with it,
/// much less of the ratio is linearly readable from `e_u`.
#[test]
fn a_group_intercept_takes_the_modality_ratio_out_of_the_unit_embedding() {
    let (units, labels, group, log_ratio) = planted_ratio_multiome();
    let fit = |group: Vec<u32>| {
        run(
            &units,
            &labels,
            4,
            &HierConfig {
                n_modules: 4,
                module_group: group,
                ..cfg()
            },
            None,
            &[],
        )
        .unwrap()
    };
    let without = r_squared(&fit(Vec::new()).e_u, &log_ratio);
    let with = r_squared(&fit(group).e_u, &log_ratio);
    assert!(
        with < 0.5 * without,
        "ln ratio R² from e_u: {with} with the intercept vs {without} without"
    );
}

/// `centre` puts every group's module rows on a zero mean and keeps the
/// differences between rows of the same group.
#[test]
fn centring_zeroes_each_groups_mean_and_keeps_within_group_differences() {
    use crate::fit::hier::params::GroupIntercepts;
    use legume_numeric::candle::candle_core::Var;
    let dev = Device::Cpu;
    let gi = GroupIntercepts::new(3, &[0, 0, 1, 1, 1], &dev)
        .unwrap()
        .unwrap();
    let raw = vec![1.0f32, 2.0, 3.0, 6.0, -1.0, 0.0, 2.0, 1.0, 5.0, 5.0];
    let mu = Var::from_vec(raw.clone(), (5, 2), &dev).unwrap();
    gi.centre(&mu).unwrap();
    let c = to_host(mu.as_tensor()).unwrap();
    for (rows, name) in [(0..2usize, "group 0"), (2..5, "group 1")] {
        for k in 0..2 {
            let mean: f32 = rows.clone().map(|m| c[m * 2 + k]).sum::<f32>() / rows.len() as f32;
            assert!(mean.abs() < 1e-6, "{name} column {k} mean {mean}");
        }
    }
    assert!(((c[0] - c[2]) - (raw[0] - raw[2])).abs() < 1e-6);
    assert!(((c[5] - c[9]) - (raw[5] - raw[9])).abs() < 1e-6);
}
