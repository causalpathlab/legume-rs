//! Composite-sum training loop (bge-style).
//!
//! Every step assembles `L = L_cell + Σ_ℓ L_pb_ℓ` across one cell-axis
//! minibatch + one minibatch per pb level, then runs **one** AdamW
//! backward over the full VarMap. The shared feature side (ρ, z, Q,
//! b_agg, b_comp) accumulates gradient from every axis; each cell/pb
//! head is independent. Per-level pb heads are throwaway scaffolding
//! that exists only to provide an embedding target for the pb-axis
//! NCE loss — they're never written to disk.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::optim::Optimizer;
use candle_nn::{AdamW, ParamsAdamW};
use log::info;
use rand::rngs::StdRng;
use rand::SeedableRng;

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
) -> Result<()> {
    let sampler = SamplerState::new(table, pb, args);
    let mut rng = StdRng::seed_from_u64(args.seed);

    // AdamW with zero weight decay — the CP factorisation already has
    // many parameters and L2-like decay would pull everything toward
    // zero faster than the NCE signal can push ρ/z/Q apart.
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
    axes.push(Axis::Cell);
    for l in 0..n_pb_levels {
        axes.push(Axis::Pb(l));
    }

    info!(
        "rmodem composite-sum training: {} axes (1 cell + {} pb levels), {} epochs × {} batches",
        axes.len(),
        n_pb_levels,
        args.epochs,
        args.batches_per_epoch
    );

    for epoch in 0..args.epochs {
        let mut sum_total = 0.0_f32;
        let mut per_axis = vec![0.0_f32; axes.len()];
        let mut n_batches = 0_usize;
        for _ in 0..args.batches_per_epoch {
            let mut total: Option<Tensor> = None;
            for (i, &axis) in axes.iter().enumerate() {
                let mb = draw_minibatch(axis, &sampler, pb, args, &mut rng);
                if mb.anchor.is_none() && mb.modifier.is_none() {
                    continue;
                }
                let l = minibatch_loss(model, axis, &mb)?;
                per_axis[i] += l.to_scalar::<f32>().unwrap_or(0.0);
                total = Some(match total {
                    None => l,
                    Some(t) => (t + l)?,
                });
            }
            let Some(loss) = total else {
                continue;
            };
            optim.backward_step(&loss)?;
            sum_total += loss.to_scalar::<f32>().unwrap_or(0.0);
            n_batches += 1;
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
                "rmodem epoch {}/{}: total={:.4} [{}] ({} batches)",
                epoch + 1,
                args.epochs,
                sum_total / n,
                per_axis_strs.join(", "),
                n_batches
            );
        }
    }

    Ok(())
}
