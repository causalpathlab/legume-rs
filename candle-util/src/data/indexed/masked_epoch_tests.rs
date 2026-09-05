//! Tests for the per-epoch masked loader: draws are keyed on the source row
//! (not the batch position), reproducible from the epoch seed, at the
//! requested rate, the gate is the encoder's own Anscombe value, the query set
//! is the masked context plus extras outside the context, query and context
//! targets come from the target rows and not the input, minibatches are views
//! of one shuffled block, and the probe batch has every field.

use super::{MaskSchedule, MaskedDraw, MaskedLevelData};
use crate::data::indexed::{IndexedInMemoryArgs, IndexedInMemoryData};
use crate::value_transform::anscombe_lite;
use candle_core::{Device, Tensor};
use nalgebra::DMatrix;
use std::collections::{HashMap, HashSet};

const D: usize = 8;
const P: usize = 5;
const K: usize = 4;

fn dev() -> Device {
    Device::Cpu
}

/// Five rows over eight genes. Row 4 has only two nonzero genes, so its
/// context carries two pads.
fn input() -> DMatrix<f32> {
    #[rustfmt::skip]
    let v: Vec<f32> = vec![
        5.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 4.0,
        0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 6.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 7.0, 0.0, 3.0, 0.0, 0.0, 2.0, 5.0,
        0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 8.0,
    ];
    DMatrix::from_row_slice(P, D, &v)
}

/// The target rows differ from the input everywhere (input + 10·gene + 100·row),
/// so a value packed from the wrong matrix is caught.
fn target() -> DMatrix<f32> {
    let x = input();
    DMatrix::from_fn(P, D, |p, g| x[(p, g)] + 10.0 * g as f32 + 100.0 * p as f32)
}

fn loader() -> IndexedInMemoryData {
    let x = input();
    let w = vec![1.0f32; D];
    IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
        input: &x,
        input_null: None,
        input_context_size: K,
        input_shortlist_weights: &w,
        input_mean: None,
    })
    .unwrap()
}

fn level() -> MaskedLevelData {
    loader().to_device_resident(&target(), &dev()).unwrap()
}

fn draw(extra: Option<usize>) -> MaskedDraw {
    MaskedDraw {
        schedule: MaskSchedule::Fixed,
        mask_fraction: 0.5,
        query_extra: extra,
    }
}

fn v2(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_vec2::<f32>().unwrap()
}

fn u2(t: &Tensor) -> Vec<Vec<u32>> {
    t.to_vec2::<u32>().unwrap()
}

fn u1(t: &Tensor) -> Vec<u32> {
    t.to_vec1::<u32>().unwrap()
}

/// `(source row → visible row)` over every batch of an epoch.
fn visible_by_source(ep: &super::MaskedEpoch) -> HashMap<u32, Vec<f32>> {
    let mut out = HashMap::new();
    for b in &ep.batches {
        let rows = u1(&b.base.row_ids);
        let vis = v2(&b.visible);
        for (r, &p) in rows.iter().enumerate() {
            let prev = out.insert(p, vis[r].clone());
            if let Some(prev) = prev {
                assert_eq!(
                    prev, vis[r],
                    "source row {p} drew two different masks in one epoch"
                );
            }
        }
    }
    out
}

#[test]
fn visible_draw_is_keyed_on_the_source_row_not_the_batch_position() {
    let lv = level();
    let t = target();
    let a = visible_by_source(&lv.begin_epoch(&t, 7, &draw(None), 2).unwrap());
    let b = visible_by_source(&lv.begin_epoch(&t, 7, &draw(None), 3).unwrap());
    assert_eq!(a.len(), P);
    for p in 0..P as u32 {
        assert_eq!(a[&p], b[&p], "row {p}: batch size changed its mask");
    }
    // Visible only on real slots; row 4's two pads are never visible.
    let x = input();
    for (p, vis) in &a {
        let nnz = (0..D).filter(|&g| x[(*p as usize, g)] > 0.0).count().min(K);
        for (kk, &v) in vis.iter().enumerate() {
            if kk >= nnz {
                assert_eq!(v, 0.0, "row {p} slot {kk} is a pad but visible");
            }
        }
    }
}

#[test]
fn epoch_draws_are_reproducible_and_move_across_epochs() {
    let lv = level();
    let t = target();
    let a = visible_by_source(&lv.begin_epoch(&t, 11, &draw(Some(2)), 2).unwrap());
    let b = visible_by_source(&lv.begin_epoch(&t, 11, &draw(Some(2)), 2).unwrap());
    let c = visible_by_source(&lv.begin_epoch(&t, 12, &draw(Some(2)), 2).unwrap());
    assert_eq!(a, b);
    assert_ne!(a, c, "a different epoch seed must move the masks");
}

#[test]
fn masked_fraction_matches_the_rate_and_uniform_rates_are_per_row() {
    let rows = 200;
    let k = 50;
    let x = DMatrix::from_fn(rows, 60, |p, g| 1.0 + ((p * 7 + g * 3) % 5) as f32);
    let w = vec![1.0f32; 60];
    let ld = IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
        input: &x,
        input_null: None,
        input_context_size: k,
        input_shortlist_weights: &w,
        input_mean: None,
    })
    .unwrap();
    let lv = ld.to_device_resident(&x, &dev()).unwrap();
    let fixed = MaskedDraw {
        schedule: MaskSchedule::Fixed,
        mask_fraction: 0.3,
        query_extra: None,
    };
    let ep = lv.begin_epoch(&x, 3, &fixed, 50).unwrap();
    let vis = visible_by_source(&ep);
    let visible: f32 = vis.values().map(|v| v.iter().sum::<f32>()).sum();
    let frac = 1.0 - visible / (rows * k) as f32;
    assert!(
        (frac - 0.3).abs() < 0.05,
        "masked fraction {frac} vs rate 0.3"
    );

    let uniform = MaskedDraw {
        schedule: MaskSchedule::Uniform { lo: 0.1, hi: 0.9 },
        mask_fraction: 0.3,
        query_extra: None,
    };
    let ep = lv.begin_epoch(&x, 3, &uniform, 50).unwrap();
    let vis = visible_by_source(&ep);
    let rates: Vec<f32> = vis
        .values()
        .map(|v| 1.0 - v.iter().sum::<f32>() / k as f32)
        .collect();
    let (lo, hi) = rates
        .iter()
        .fold((1.0f32, 0.0f32), |(l, h), &r| (l.min(r), h.max(r)));
    assert!(
        hi - lo > 0.3,
        "uniform schedule must vary per row: rates in [{lo}, {hi}]"
    );
}

#[test]
fn gate_equals_the_encoders_anscombe_value() {
    let lv = level();
    let ep = lv.begin_epoch(&target(), 5, &draw(None), P).unwrap();
    let b = &ep.batches[0];
    let want = anscombe_lite(&b.base.input_values, None, None).unwrap();
    assert_eq!(v2(&b.gate), v2(&want));
}

#[test]
fn query_set_holds_every_masked_real_slot_plus_extras_outside_the_context() {
    let lv = level();
    let ep = lv.begin_epoch(&target(), 9, &draw(Some(2)), P).unwrap();
    let b = &ep.batches[0];
    let q = b.query.as_ref().expect("query set requested");
    let rows = u1(&b.base.row_ids);
    let idx = u2(&b.base.input_indices);
    let vis = v2(&b.visible);
    let ids = u2(&q.ids);
    let w = v2(&q.weight);
    let vals = v2(&b.base.input_values);
    for r in 0..P {
        let p = rows[r] as usize;
        // The context is the row's top-K slots with a positive value, not the
        // whole nonzero row: a nonzero gene outside the top-K is unseen and a
        // legitimate extra.
        let context: HashSet<u32> = (0..K)
            .filter(|&kk| vals[r][kk] > 0.0)
            .map(|kk| idx[r][kk])
            .collect();
        let real: Vec<u32> = ids[r]
            .iter()
            .zip(&w[r])
            .filter(|(_, &w)| w > 0.0)
            .map(|(&g, _)| g)
            .collect();
        let masked: Vec<u32> = (0..K)
            .filter(|&kk| vals[r][kk] > 0.0 && vis[r][kk] == 0.0)
            .map(|kk| idx[r][kk])
            .collect();
        for g in &masked {
            assert!(real.contains(g), "row {p}: masked gene {g} is not a query");
        }
        let extras: Vec<u32> = real
            .iter()
            .copied()
            .filter(|g| !masked.contains(g))
            .collect();
        assert_eq!(
            extras.len(),
            2.min(D - context.len()),
            "row {p}: extras {extras:?}"
        );
        for g in &extras {
            assert!(
                !context.contains(g),
                "row {p}: extra {g} sits in the context"
            );
        }
        let mut dedup = real.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), real.len(), "row {p}: duplicate query");
        for (j, &wv) in w[r].iter().enumerate() {
            if wv == 0.0 {
                assert_eq!(ids[r][j], 0, "pad query carries a gene");
            }
        }
    }
    assert_eq!(ep.n_queries.len(), 1);
    let total: f32 = w.iter().flatten().sum();
    assert_eq!(ep.n_queries[0], total);
}

#[test]
fn query_target_and_target_at_context_come_from_the_target_rows_not_the_input() {
    let lv = level();
    let t = target();
    let ep = lv.begin_epoch(&t, 2, &draw(Some(3)), P).unwrap();
    let b = &ep.batches[0];
    let rows = u1(&b.base.row_ids);
    let idx = u2(&b.base.input_indices);
    let vals = v2(&b.base.input_values);
    let tac = v2(&b.target_at_context);
    for r in 0..P {
        let p = rows[r] as usize;
        for kk in 0..K {
            if vals[r][kk] > 0.0 {
                assert_eq!(tac[r][kk], t[(p, idx[r][kk] as usize)], "row {p} slot {kk}");
            } else {
                assert_eq!(tac[r][kk], 0.0, "pad slot must carry 0");
            }
        }
    }
    let q = b.query.as_ref().unwrap();
    let (ids, w, qt) = (u2(&q.ids), v2(&q.weight), v2(&q.target));
    for r in 0..P {
        let p = rows[r] as usize;
        for j in 0..ids[r].len() {
            if w[r][j] > 0.0 {
                assert_eq!(qt[r][j], t[(p, ids[r][j] as usize)], "row {p} query {j}");
            } else {
                assert_eq!(qt[r][j], 0.0);
            }
        }
    }
}

#[test]
fn minibatches_are_narrow_views_of_one_shuffled_block() {
    let lv = level();
    let ep = lv.begin_epoch(&target(), 4, &draw(Some(1)), 2).unwrap();
    assert_eq!(
        ep.batches.len(),
        3,
        "5 rows at batch size 2 → 3 batches, bootstrap-padded"
    );
    let mut seen: HashMap<u32, Vec<u32>> = HashMap::new();
    for b in &ep.batches {
        assert_eq!(b.base.input_indices.dims(), &[2, K]);
        assert_eq!(b.visible.dims(), &[2, K]);
        assert_eq!(b.gate.dims(), &[2, K]);
        assert_eq!(b.target_at_context.dims(), &[2, K]);
        let q = b.query.as_ref().unwrap();
        assert_eq!(q.ids.dims()[0], 2);
        let rows = u1(&b.base.row_ids);
        let idx = u2(&b.base.input_indices);
        for (r, &p) in rows.iter().enumerate() {
            if let Some(prev) = seen.insert(p, idx[r].clone()) {
                assert_eq!(prev, idx[r], "duplicated row {p} carries different content");
            }
        }
    }
    assert_eq!(seen.len(), P, "every source row appears at least once");
    assert_eq!(ep.n_visible.len(), 3);
}

#[test]
fn probe_minibatch_has_every_field_at_the_requested_size() {
    let lv = level();
    let mb = lv.probe_minibatch(&target(), 7, 1, &draw(Some(2))).unwrap();
    assert_eq!(mb.base.row_ids.dims(), &[7]);
    assert_eq!(u1(&mb.base.row_ids), vec![0, 1, 2, 3, 4, 0, 1]);
    assert_eq!(mb.visible.dims(), &[7, K]);
    assert_eq!(mb.gate.dims(), &[7, K]);
    assert_eq!(mb.target_at_context.dims(), &[7, K]);
    assert_eq!(mb.query.as_ref().unwrap().ids.dims()[0], 7);
    let none = lv.probe_minibatch(&target(), 3, 1, &draw(None)).unwrap();
    assert!(none.query.is_none());
}
