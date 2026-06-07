//! Composite-sum training loop (bge-style).
//!
//! Every step assembles `L = L_cell + Σ_ℓ L_pb_ℓ` across one cell-axis
//! minibatch + one minibatch per pb level, then runs **one** AdamW
//! backward over the full VarMap. The shared feature side (β, z, δ, γ,
//! b_agg, b_comp) accumulates gradient from every axis; each cell/pb
//! head is independent. Per-level pb heads are throwaway scaffolding
//! that exists only to provide an embedding target for the pb-axis
//! NCE loss — they're never written to disk.

use super::common::{candle_core, candle_nn};
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::optim::Optimizer;
use candle_nn::{AdamW, ParamsAdamW};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::args::RnaModEmbedArgs;
use super::feature_table::FeatureTable;
use super::loss::minibatch_loss;
use super::model::{Axis, RnaModEmbedModel};
use super::pseudobulk::PseudobulkData;
use super::sampling::{draw_minibatch, Minibatch, SamplerState};

pub fn train(
    args: &RnaModEmbedArgs,
    table: &FeatureTable,
    pb: &PseudobulkData,
    model: &mut RnaModEmbedModel,
    stop: &Arc<AtomicBool>,
) -> Result<()> {
    let sampler = SamplerState::new(table, pb, args);
    let mut rng = StdRng::seed_from_u64(args.seed);

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

    let n_pb_levels = pb.n_levels();
    let mut axes: Vec<Axis> = Vec::with_capacity(1 + n_pb_levels);
    if !args.no_cell_axis {
        axes.push(Axis::Cell);
    }
    for l in 0..n_pb_levels {
        axes.push(Axis::Pb(l));
    }
    anyhow::ensure!(
        !axes.is_empty(),
        "--no-cell-axis with zero pb levels leaves nothing to train; \
         either increase --num-levels or drop --no-cell-axis"
    );

    // Resolve `--batches-per-epoch`: explicit override, or auto = one
    // weighted pass over the largest axis (cell / pb_per_level).
    let max_axis_units = std::cmp::max(
        model.n_cells,
        pb.pb_pools_per_level
            .iter()
            .map(|l| l.n_units)
            .max()
            .unwrap_or(0),
    );
    let batches_per_epoch = args.batches_per_epoch.unwrap_or_else(|| {
        let bs = args.batch_size.max(1);
        max_axis_units.div_ceil(bs).max(1)
    });

    info!(
        "composite-sum training: {} axes (1 cell + {} pb levels), {} epochs × {} batches \
         (auto={}, max_axis_units={})",
        axes.len(),
        n_pb_levels,
        args.epochs,
        batches_per_epoch,
        args.batches_per_epoch.is_none(),
        max_axis_units,
    );

    // Pipeline CPU sampling with GPU optimization: a scoped producer thread
    // draws the next minibatches (one per axis) while the main thread runs
    // forward/backward on the device, so the GPU isn't stalled waiting on the
    // single-threaded sampler. The producer owns `rng` (keeping the draw
    // sequence deterministic) and borrows the sampler/pools; bundles flow
    // through a small bounded channel that also throttles look-ahead.
    let axes_ref = &axes;
    let sampler_ref = &sampler;

    std::thread::scope(|scope| -> Result<()> {
        // One bundle = one minibatch per axis, tagged with the epoch it
        // belongs to so the consumer can detect epoch boundaries.
        type Bundle = (usize, Vec<(usize, Axis, Minibatch)>);
        let (tx, rx) = std::sync::mpsc::sync_channel::<Bundle>(2);

        scope.spawn(move || {
            'epochs: for epoch in 0..args.epochs {
                for _ in 0..batches_per_epoch {
                    if stop.load(Ordering::Relaxed) {
                        break 'epochs;
                    }
                    let mut bundle = Vec::with_capacity(axes_ref.len());
                    for (i, &axis) in axes_ref.iter().enumerate() {
                        bundle.push((
                            i,
                            axis,
                            draw_minibatch(axis, sampler_ref, pb, args, &mut rng),
                        ));
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
            let per_axis_strs: Vec<String> = axes_ref
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
                "epoch {}/{}: total={:.4} [{}] ({} batches)",
                epoch + 1,
                args.epochs,
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
            // baseline we *want* to fit).
            if args.z_l2 > 0.0 {
                let z_sq = model.z.sqr()?.mean_all()?;
                loss = (loss + (z_sq * (args.z_l2 as f64))?)?;
            }
            if args.delta_l2 > 0.0 {
                let d_sq = model.delta.sqr()?.mean_all()?;
                loss = (loss + (d_sq * (args.delta_l2 as f64))?)?;
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
                    "stopping early during epoch {} — finalizing outputs",
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
