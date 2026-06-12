//! Two-phase training (bge-style).
//!
//! **Phase 1 — joint feature shaping.** Every step assembles
//! `L = L_cell + Σ_ℓ L_pb_ℓ` across one cell-axis minibatch + one
//! minibatch per pb level, then runs **one** AdamW backward over the full
//! VarMap. The shared feature side (β, z, δ, γ, b_agg, b_comp) accumulates
//! gradient from every axis; each cell/pb head is independent. Per-level
//! pb heads are throwaway scaffolding that exists only to provide an
//! embedding target for the pb-axis NCE loss — they're never written to
//! disk.
//!
//! **Phase 2 — dense cell re-evaluation.** The entire feature side and
//! every pb head are frozen; only `e_cell` is optimised, on the cell axis
//! alone. The auto per-epoch budget is sized by `n_cells`, so every cell
//! is swept ~once/epoch against the fixed dictionary (the joint phase-1
//! budget, sized by the largest axis, under-trains the cell side). Skipped
//! under `--no-cell-axis`, with zero cells, or `--phase2-epochs 0`.

use super::common::{candle_core, candle_nn};
use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::optim::Optimizer;
use candle_nn::{AdamW, ParamsAdamW};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::args::GemArgs;
use super::cell_solve;
use super::feature_table::FeatureTable;
use super::loss::minibatch_loss;
use super::model::{Axis, GemModel};
use super::pseudobulk::PseudobulkData;
use super::sampling::{draw_minibatch, Minibatch, SamplerState};

/// Run both training phases.  Returns the pre-L2-normalisation cell norms
/// from phase 2 (one entry per model cell), or an empty `Vec` when phase 2
/// was skipped or interrupted before completion.
pub fn train(
    args: &GemArgs,
    table: &FeatureTable,
    pb: &PseudobulkData,
    model: &mut GemModel,
    stop: &Arc<AtomicBool>,
) -> Result<Vec<f32>> {
    let sampler = SamplerState::new(table, pb, args);
    // One deterministic draw sequence threaded through both phases by
    // `&mut` borrow (each phase's producer thread is joined before the
    // borrow is released, so this is sound — see `run_phase`).
    let mut rng = StdRng::seed_from_u64(args.seed);

    let n_pb_levels = pb.n_levels();

    ////////////////////////////////////////
    // Phase 1: joint cell + pb axes
    ////////////////////////////////////////
    // Phase-1 cell axis is on iff it's not feature-only mode AND the
    // cell-axis knob isn't 0. `--phase1-cells-per-pb 0` (the default) drops
    // the cell axis from phase 1 (pure-pb) but phase 2 still projects every
    // cell; `--no-cell-axis` additionally skips phase 2 (e_cell at init).
    let use_phase1_cell_axis = !args.no_cell_axis && args.phase1_cells_per_pb != 0;
    let mut phase1_axes: Vec<Axis> = Vec::with_capacity(1 + n_pb_levels);
    if use_phase1_cell_axis {
        phase1_axes.push(Axis::Cell);
    }
    for l in 0..n_pb_levels {
        phase1_axes.push(Axis::Pb(l));
    }
    anyhow::ensure!(
        !phase1_axes.is_empty(),
        "phase 1 has no axes to train: the cell axis is off (--no-cell-axis \
         or --phase1-cells-per-pb 0) and there are zero pb levels; \
         increase --num-levels or enable the cell axis"
    );

    if use_phase1_cell_axis {
        match &sampler.phase1_cell_pools {
            Some(p) => info!(
                "phase-1 cell axis: subsampled to {} cells (≤{} per pb-sample, all levels); \
                 phase 2 still projects all {} cells",
                p.n_units, args.phase1_cells_per_pb, model.n_cells
            ),
            None => info!(
                "phase-1 cell axis: all {} cells (--phase1-cells-per-pb ≥ n_cells)",
                model.n_cells
            ),
        }
    } else if args.no_cell_axis {
        info!("phase-1 cell axis OFF (--no-cell-axis): feature-only; phase 2 skipped");
    } else {
        info!(
            "phase-1 cell axis OFF (pure-pb, --phase1-cells-per-pb 0): E_feat from pb \
             aggregates only; phase 2 still projects all {} cells",
            model.n_cells
        );
    }

    // AdamW with zero weight decay — β, z, δ, γ already carry many
    // parameters and L2-like decay would pull everything toward zero
    // faster than the NCE signal can push them apart.
    let mut optim = AdamW::new(
        model.varmap.all_vars(),
        ParamsAdamW {
            lr: args.learning_rate,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    info!("phase 1/2 (joint feature shaping): {} epochs", args.epochs);
    run_phase(
        args,
        pb,
        model,
        &sampler,
        &phase1_axes,
        &mut optim,
        args.epochs,
        /*apply_z_delta_l2=*/ true,
        &mut rng,
        stop,
    )?;
    drop(optim);

    ////////////////////////////////////////
    // Phase 2: analytical per-cell projection
    ////////////////////////////////////////
    // With the feature side frozen, each cell's embedding is independent —
    // so instead of SGD we project every cell onto the frozen dictionary in
    // parallel (Poisson MAP, ridge prior). See `cell_solve`.
    let want_phase2 =
        !args.no_cell_axis && model.n_cells > 0 && args.phase2_epochs.is_none_or(|n| n > 0);
    if want_phase2 {
        if stop.load(Ordering::Relaxed) {
            info!("phase 2 skipped (interrupted in phase 1)");
            return Ok(vec![]);
        }
        info!(
            "phase 2/2 (analytical cell projection onto frozen features, ridge λ={}): {} cells",
            args.phase2_ridge, model.n_cells
        );
        let cell_nrms = cell_solve::solve_cell_embeddings(model, &pb.cell_pools, args)
            .context("phase-2 cell projection")?;
        return Ok(cell_nrms);
    }

    Ok(vec![])
}

/// Run one composite-sum training phase over `axes` for `epochs`, stepping
/// `optim` once per minibatch bundle. A scoped producer thread draws the
/// next bundles (one minibatch per axis) on the CPU while the main thread
/// runs forward/backward on the device, so the GPU isn't stalled on the
/// single-threaded sampler. The producer borrows `rng` mutably for the
/// phase's lifetime; the scope joins before this returns, so the next
/// phase resumes the same deterministic draw sequence.
#[allow(clippy::too_many_arguments)]
fn run_phase(
    args: &GemArgs,
    pb: &PseudobulkData,
    model: &GemModel,
    sampler: &SamplerState,
    axes: &[Axis],
    optim: &mut AdamW,
    epochs: usize,
    apply_z_delta_l2: bool,
    rng: &mut StdRng,
    stop: &Arc<AtomicBool>,
) -> Result<()> {
    if epochs == 0 || axes.is_empty() {
        return Ok(());
    }

    // Resolve `--batches-per-epoch`: explicit override, or auto = one
    // weighted pass over the largest axis *in this phase* (cell-only phase
    // 2 sizes by n_cells, not max(n_cells, pb)).
    let max_axis_units = axes
        .iter()
        .map(|axis| match axis {
            // Size the cell axis by the kept-cell count when phase-1
            // subsampling is active (`--phase1-cells-per-pb k`), so the auto
            // budget actually shrinks with k; full `n_cells` otherwise.
            Axis::Cell => sampler.n_cell_units(model.n_cells),
            Axis::Pb(l) => pb.pb_pools_per_level[*l].n_units,
        })
        .max()
        .unwrap_or(0);
    let batches_per_epoch = args.batches_per_epoch.unwrap_or_else(|| {
        let bs = args.batch_size.max(1);
        max_axis_units.div_ceil(bs).max(1)
    });

    info!(
        "  {} axes, {} epochs × {} batches (auto={}, max_axis_units={})",
        axes.len(),
        epochs,
        batches_per_epoch,
        args.batches_per_epoch.is_none(),
        max_axis_units,
    );

    std::thread::scope(|scope| -> Result<()> {
        // One bundle = one minibatch per axis, tagged with the epoch it
        // belongs to so the consumer can detect epoch boundaries.
        type Bundle = (usize, Vec<(usize, Axis, Minibatch)>);
        let (tx, rx) = std::sync::mpsc::sync_channel::<Bundle>(2);

        scope.spawn(move || {
            'epochs: for epoch in 0..epochs {
                for _ in 0..batches_per_epoch {
                    if stop.load(Ordering::Relaxed) {
                        break 'epochs;
                    }
                    let mut bundle = Vec::with_capacity(axes.len());
                    for (i, &axis) in axes.iter().enumerate() {
                        bundle.push((i, axis, draw_minibatch(axis, sampler, pb, args, rng)));
                    }
                    if tx.send((epoch, bundle)).is_err() {
                        break 'epochs; // consumer gone
                    }
                }
            }
            // tx dropped here → consumer's `rx` iterator ends.
        });

        // Per-epoch loss accumulators kept **on-device** and synced to scalars
        // once per epoch (not per minibatch). `detach()` keeps the running sum
        // off the autograd graph, so each minibatch's forward graph is still
        // released immediately (no GPU-memory blow-up) while we avoid the
        // ~axes×batches GPU→CPU stalls the old per-minibatch logging incurred.
        let emit = |epoch: usize,
                    total_acc: &Option<Tensor>,
                    per_axis_acc: &[Option<Tensor>],
                    n_batches: usize| {
            if n_batches == 0 {
                return;
            }
            let n = n_batches as f32;
            let scalar = |o: &Option<Tensor>| {
                o.as_ref()
                    .and_then(|t| t.to_scalar::<f32>().ok())
                    .unwrap_or(0.0)
            };
            let per_axis_strs: Vec<String> = axes
                .iter()
                .enumerate()
                .map(|(i, axis)| {
                    let label = match axis {
                        Axis::Cell => "cell".to_string(),
                        Axis::Pb(level) => format!("pb_l{level}"),
                    };
                    format!("{}={:.3}", label, scalar(&per_axis_acc[i]) / n)
                })
                .collect();
            info!(
                "  epoch {}/{}: total={:.4} [{}] ({} batches)",
                epoch + 1,
                epochs,
                scalar(total_acc) / n,
                per_axis_strs.join(", "),
                n_batches
            );
        };

        let mut cur_epoch = 0usize;
        let mut started = false;
        let mut total_acc: Option<Tensor> = None;
        let mut per_axis_acc: Vec<Option<Tensor>> = vec![None; axes.len()];
        let mut n_batches = 0usize;

        for (epoch, bundle) in rx {
            if started && epoch != cur_epoch {
                emit(cur_epoch, &total_acc, &per_axis_acc, n_batches);
                total_acc = None;
                per_axis_acc = vec![None; axes.len()];
                n_batches = 0;
            }
            cur_epoch = epoch;
            started = true;

            let mut total: Option<Tensor> = None;
            let mut per_axis_t: Vec<Option<Tensor>> = vec![None; axes.len()];
            for (i, axis, mb) in &bundle {
                if mb.anchor.is_none() && mb.modifier.is_none() {
                    continue;
                }
                let l = minibatch_loss(model, *axis, mb)?;
                per_axis_t[*i] = Some(l.clone());
                total = Some(match total {
                    None => l,
                    Some(t) => (t + l)?,
                });
            }
            let Some(mut loss) = total else {
                continue;
            };
            // Mean-normalized L2 penalties on (z, δ) — scale-invariant across
            // G·K and K·M·H, so λ stays meaningful as model dims grow. The
            // region offset γ is left unpenalized (per-modality positional
            // baseline we *want* to fit). Skipped in phase 2: the feature
            // side is frozen, so the penalty gradient on z/δ is computed but
            // never stepped — wasted compute that also pollutes the log.
            if apply_z_delta_l2 {
                if args.z_l2 > 0.0 {
                    let z_sq = model.z.sqr()?.mean_all()?;
                    loss = (loss + (z_sq * (args.z_l2 as f64))?)?;
                }
                if args.delta_l2 > 0.0 {
                    let d_sq = model.delta.sqr()?.mean_all()?;
                    loss = (loss + (d_sq * (args.delta_l2 as f64))?)?;
                }
            }
            optim.backward_step(&loss)?;

            // Accumulate detached scalars on-device — no per-minibatch sync.
            let ld = loss.detach();
            total_acc = Some(match total_acc.take() {
                None => ld,
                Some(a) => (a + ld)?,
            });
            for (i, opt) in per_axis_t.into_iter().enumerate() {
                if let Some(t) = opt {
                    let td = t.detach();
                    per_axis_acc[i] = Some(match per_axis_acc[i].take() {
                        None => td,
                        Some(a) => (a + td)?,
                    });
                }
            }
            n_batches += 1;

            // Stop at sub-epoch granularity on Ctrl-C (epochs are ~100s);
            // the partial epoch is flushed by the final emit below.
            if stop.load(Ordering::Relaxed) {
                info!(
                    "  stopping early during epoch {} — finalizing outputs",
                    cur_epoch + 1
                );
                break;
            }
        }

        // Flush the final (or only) epoch.
        if started {
            emit(cur_epoch, &total_acc, &per_axis_acc, n_batches);
        }
        Ok(())
    })?;

    Ok(())
}
