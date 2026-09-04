//! Unit tests for the velocity-drift SEM residual.

use super::{sem_penalty, PbSemTerm, TrainingParams};
use crate::fit::lineage::PbLineageLevel;
use candle_util::candle_core::{Device, Tensor, Var};

/// One edge 0→1, parent velocity v̂₀ = (1,0), step s = 1. The residual is
/// `e₁ − e₀ − v̂₀`, so the penalty vanishes exactly when `e₁ = e₀ + (1,0)` and is
/// positive otherwise.
fn one_edge_level() -> PbLineageLevel {
    PbLineageLevel {
        n_pb: 2,
        edges: vec![(0, 1, 1.0)],
        velocity: vec![1.0, 0.0, /*n0*/ 0.0, 0.0 /*n1*/],
    }
}

#[test]
fn sem_penalty_zero_at_consistency() {
    let dev = Device::Cpu;
    let h = 2;
    let term = PbSemTerm::new(&one_edge_level(), h, 1.0, 1.0, &dev)
        .unwrap()
        .unwrap();

    // e₁ = e₀ + s·v̂₀ = (0.5,0.3) + (1,0) = (1.5,0.3) → residual 0.
    let consistent = Tensor::from_vec(vec![0.5f32, 0.3, 1.5, 0.3], (2, h), &dev).unwrap();
    let pen0 = sem_penalty(&consistent, &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(pen0 < 1e-6, "penalty should vanish at consistency ({pen0})");

    // Off-consistency (e₁ not a velocity-step ahead) → strictly positive.
    let bad = Tensor::from_vec(vec![0.5f32, 0.3, 0.5, 0.3], (2, h), &dev).unwrap();
    let pen1 = sem_penalty(&bad, &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        pen1 > 1e-3,
        "penalty should be positive off-consistency ({pen1})"
    );
}

#[test]
fn sem_penalty_gradient_step_reduces() {
    let dev = Device::Cpu;
    let h = 2;
    let term = PbSemTerm::new(&one_edge_level(), h, 1.0, 1.0, &dev)
        .unwrap()
        .unwrap();

    // Start off-consistency; one gradient-descent step on e_cell must lower it.
    let var =
        Var::from_tensor(&Tensor::from_vec(vec![0.5f32, 0.3, 0.5, 0.3], (2, h), &dev).unwrap())
            .unwrap();
    let before = sem_penalty(var.as_tensor(), &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let loss = sem_penalty(var.as_tensor(), &term).unwrap();
    let grads = loss.backward().unwrap();
    let g = grads.get(var.as_tensor()).unwrap();
    let stepped = (var.as_tensor() - (g * 0.1).unwrap()).unwrap();
    let after = sem_penalty(&stepped, &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(
        after < before,
        "gradient step should reduce the SEM penalty ({before} -> {after})"
    );
}

/// A level with no oriented edges produces no term (nothing to penalize).
#[test]
fn empty_level_yields_no_term() {
    let dev = Device::Cpu;
    let level = PbLineageLevel {
        n_pb: 3,
        edges: vec![],
        velocity: vec![0.0; 6],
    };
    assert!(PbSemTerm::new(&level, 2, 1.0, 1.0, &dev).unwrap().is_none());
}

/// A level with a δ-less node still builds a term (the θ-pseudotime fallback keeps
/// it in the graph) — exercised via the fixed-KNN path in `fit`.
#[test]
fn sem_term_survives_multi_edge_level() {
    let dev = Device::Cpu;
    let level = PbLineageLevel {
        n_pb: 3,
        edges: vec![(0, 1, 1.0), (1, 2, 1.0)],
        velocity: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    assert!(PbSemTerm::new(&level, 2, 1.0, 1.0, &dev).unwrap().is_some());
}

/// `resolve_batches_per_epoch` is what turns `--batch-size` into a step budget;
/// pin the auto and explicit paths so a change there is deliberate.
#[test]
fn batches_per_epoch_resolves_auto_and_explicit() {
    use crate::training::resolve_batches_per_epoch;

    let params = |batch_size, batches_per_epoch| TrainingParams {
        module: None,
        epochs: 1,
        batches_per_epoch,
        batch_size,
        num_negatives: 4,
        seed: 42,
        objective: crate::loss::NceObjective::Softmax,
        feature_embedding_l2: 0.0,
        max_grad_norm: 0.0,
        delta_l2: 0.0,
        epoch_offset: 0,
        gpu_mem_fraction: None,
    };

    assert_eq!(resolve_batches_per_epoch(&params(1024, None), 8_192), 8);
    assert_eq!(resolve_batches_per_epoch(&params(1024, Some(7)), 8_192), 7);
    // Never zero, or the epoch would run no steps at all.
    assert!(resolve_batches_per_epoch(&params(65_536, None), 8_192) >= 1);
}

////////////////////////////////////////////
// Warm-up release across training calls  //
////////////////////////////////////////////

/// A run split into several `train_composite` calls (posterior jitter) restarts
/// its local epoch counter each call, so a warm-up keyed on the LOCAL epoch
/// releases at the wrong time or — when every call is shorter than the warm-up —
/// never at all. `epoch_offset` is what makes the rule global.
///
/// The measured consequence of getting this wrong: with 1000 epochs, a 250-epoch
/// warm-up and 4 jitter rounds of 250, `epoch == warmup` was never true in any
/// round and the membership stayed frozen for the entire fit.
#[test]
fn the_warm_up_releases_exactly_once_at_the_right_global_epoch() {
    use crate::training::{frozen_at_entry, releases_at_local_epoch};

    // 1000 epochs, warm-up 250, four rounds of 250: round 0 holds and releases
    // on its last boundary; every later round starts already released.
    let (total, warmup, rounds) = (1000usize, 250usize, 4usize);
    let per = total / rounds;
    let mut released_at: Vec<usize> = Vec::new();
    let mut frozen = true;
    for r in 0..rounds {
        let offset = r * per;
        // A call only re-freezes while the warm-up is genuinely still ahead.
        if r == 0 {
            frozen = frozen_at_entry(warmup, offset);
        } else {
            assert_eq!(
                frozen_at_entry(warmup, offset),
                offset < warmup,
                "round {r} must not re-freeze a released membership"
            );
        }
        for epoch in 0..per {
            if frozen && releases_at_local_epoch(warmup, offset, epoch) {
                released_at.push(offset + epoch);
                frozen = false;
            }
        }
    }
    assert_eq!(
        released_at,
        vec![warmup],
        "the membership must release exactly once, at global epoch {warmup}"
    );
    assert!(!frozen, "and stay released");

    // A single round is the historical path: release at the same global epoch.
    let mut hits = Vec::new();
    for epoch in 0..total {
        if releases_at_local_epoch(warmup, 0, epoch) {
            hits.push(epoch);
        }
    }
    assert_eq!(hits, vec![warmup]);

    // A warm-up that straddles a round boundary still fires inside the round
    // that contains it, not at that round's edge.
    assert!(!releases_at_local_epoch(300, 250, 0));
    assert!(releases_at_local_epoch(300, 250, 50));
    // Zero warm-up: never frozen, never a release event.
    assert!(!frozen_at_entry(0, 0));
    assert!(!releases_at_local_epoch(0, 0, 0));
}
