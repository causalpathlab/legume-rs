//! Composite-sum training loop (bge-style).
//!
//! Every step assembles `L = L_cell + Σ_ℓ L_pb_ℓ` across one cell-axis
//! minibatch + one minibatch per pb level, then runs **one** AdamW
//! backward over the full VarMap. The shared feature side (ρ, z, Q,
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
use super::sampling::{draw_minibatch, SamplerState};

pub fn train(
    args: &RnaModEmbedArgs,
    table: &FeatureTable,
    pb: &PseudobulkData,
    model: &mut RnaModEmbedModel,
    stop: &Arc<AtomicBool>,
) -> Result<()> {
    let sampler = SamplerState::new(table, pb, args);
    let mut rng = StdRng::seed_from_u64(args.seed);

    // AdamW with zero weight decay — ρ, z, Q already carry many
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

    for epoch in 0..args.epochs {
        // Per-axis loss tensors accumulated within one minibatch so we
        // can both backward through their sum and report the breakdown.
        // We sync to scalar **once per minibatch** (after backward),
        // which releases the autograd graph immediately — accumulating
        // tensor sums across batches would pin every forward graph in
        // GPU memory until epoch end (~15 GB blow-up).
        let mut per_axis = vec![0.0_f32; axes.len()];
        let mut sum_total = 0.0_f32;
        let mut n_batches = 0_usize;
        for _ in 0..batches_per_epoch {
            let mut total: Option<Tensor> = None;
            let mut per_axis_t: Vec<Option<Tensor>> = vec![None; axes.len()];
            for (i, &axis) in axes.iter().enumerate() {
                let mb = draw_minibatch(axis, &sampler, pb, args, &mut rng);
                if mb.anchor.is_none() && mb.modifier.is_none() {
                    continue;
                }
                let l = minibatch_loss(model, axis, &mb)?;
                per_axis_t[i] = Some(l.clone());
                total = Some(match total {
                    None => l,
                    Some(t) => (t + l)?,
                });
            }
            let Some(mut loss) = total else {
                continue;
            };
            // Mean-normalized L2 penalties on (z, Q) — scale-invariant
            // across G·K and K·M·H, so λ stays meaningful as model dims
            // grow. Matches senna bge's --feature-embedding-l2 style.
            if args.z_l2 > 0.0 {
                let z_sq = model.z.sqr()?.mean_all()?;
                loss = (loss + (z_sq * (args.z_l2 as f64))?)?;
            }
            if args.q_l2 > 0.0 {
                let q_sq = model.q.sqr()?.mean_all()?;
                loss = (loss + (q_sq * (args.q_l2 as f64))?)?;
            }
            optim.backward_step(&loss)?;
            // One scalar fetch per minibatch — releases the graph too.
            sum_total += loss.to_scalar::<f32>().unwrap_or(0.0);
            for (i, opt) in per_axis_t.iter().enumerate() {
                if let Some(t) = opt {
                    per_axis[i] += t.to_scalar::<f32>().unwrap_or(0.0);
                }
            }
            n_batches += 1;

            if stop.load(Ordering::Relaxed) {
                break;
            }
        }
        if n_batches > 0 {
            let n = n_batches as f32;
            let per_axis_strs: Vec<String> = axes
                .iter()
                .enumerate()
                .map(|(i, axis)| {
                    let label = match axis {
                        Axis::Cell => "cell".to_string(),
                        Axis::Pb(level) => format!("pb_l{level}"),
                    };
                    format!("{}={:.3}", label, per_axis[i] / n)
                })
                .collect();
            info!(
                "epoch {}/{}: total={:.4} [{}] ({} batches)",
                epoch + 1,
                args.epochs,
                sum_total / n,
                per_axis_strs.join(", "),
                n_batches
            );
        }

        if stop.load(Ordering::SeqCst) {
            info!(
                "stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                args.epochs
            );
            return Ok(());
        }
    }

    Ok(())
}
