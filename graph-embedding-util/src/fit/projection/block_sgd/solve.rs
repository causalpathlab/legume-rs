//! The Adam loop for one block of nodes: gather the block's edges densely, hoist
//! every loop-invariant piece of the gradient, then step `Θ̃ = [Θ | c]` to
//! convergence and read back the latents plus the block's deviance. The closed-form
//! gradient the module docs justify lives here and nowhere else.

use super::pass::{adam_step_size, poisson_deviance, BlockArgs, BlockOut};
use super::{BETA1, BETA2, CHECK_EVERY, EPS, TOL};
use crate::cell_projection::SCORE_CLAMP;
use candle_util::candle_core::{DType, Tensor};
use matrix_util::traits::FusedTensorOps;

pub(super) fn solve_block(a: BlockArgs) -> anyhow::Result<BlockOut> {
    let (h, dev) = (a.input.h, a.input.dev);
    let bc = a.end - a.start;
    let d = h + 1; // [Θ | c]
    let dict = a.dict;
    let f_live = dict.b_row.dim(1)?;
    let (e_aug, e_aug_t, intercept_mask) = (&dict.e_aug, &dict.e_aug_t, &dict.intercept_mask);

    ///////////////////////////////////////
    // Gather the block's observed edges //
    ///////////////////////////////////////

    // Flat index into the block's `[Bc, F]` score matrix, so the data term is a
    // dense product against the score. See the module docs: the gather's backward
    // (`index_add`) degenerates to a single-threaded serial walk of the index list
    // on CUDA, so it is far cheaper to carry the zeros.
    let mut n_dense = vec![0f32; bc * f_live];
    // Per-cell total count, for the null-model intercept; and the folded-feature
    // count mass, which still owes `−n·c` to the loss.
    let mut n_tot = vec![0f64; bc];
    let mut n_dead = vec![0f32; bc];
    let mut n_edges = 0usize;
    for i in a.start..a.end {
        let local = i - a.start;
        let (feats, counts) = a.spec.edges.cell_slice(i);
        for (&f_pass, &n) in feats.iter().zip(counts) {
            // `edges` are indexed on the pass partition; map to the live subset.
            let g = dict.rows()[f_pass as usize];
            let l = dict.to_live[g as usize];
            n_tot[local] += f64::from(n);
            if l == u32::MAX {
                n_dead[local] += n;
            } else {
                n_dense[local * f_live + l as usize] = n;
                n_edges += 1;
            }
        }
    }
    let n_t = Tensor::from_vec(n_dense, (bc, f_live), dev)?.detach();
    let n_dead_t = Tensor::from_vec(n_dead, bc, dev)?;

    ///////////////////////////////
    // Null-model initialisation //
    ///////////////////////////////

    // Θ at its warm start (zero without one) and the exact intercept given Θ:
    //     c = ln(Σ_f n_cf) − ln(Σ_f exp(⟨e_f, θ_c⟩ + β_f) + dead_mass·1)
    // so step 1 already sits at the right depth and the optimiser only has to
    // learn the deviation. (The Newton path starts from a randn `e_cell` and a
    // zero intercept, which is why its first steps have to move so far.)
    let init_block = a
        .spec
        .init_theta
        .map(|t| Tensor::from_slice(&t[a.start * h..a.end * h], (bc, h), dev))
        .transpose()?;
    // Per-row sum of `exp(score)` over the live features, from a `[Bc, F]` score.
    let row_log_norm = |s: Tensor| -> anyhow::Result<Vec<f64>> {
        Ok(s.clamp(-SCORE_CLAMP, SCORE_CLAMP)?
            .exp()?
            .sum(1)?
            .to_vec1::<f32>()?
            .iter()
            .map(|x| (f64::from(*x) + dict.dead_mass).max(f64::MIN_POSITIVE).ln())
            .collect())
    };
    let log_norm: Vec<f64> = match &init_block {
        // Warm start: the row's own `⟨e_f, θ_c⟩` on top of the frozen bias.
        Some(t) => row_log_norm(
            t.matmul(&e_aug.narrow(0, 0, h)?.contiguous()?)?
                .broadcast_add(&dict.b_row)?,
        )?,
        // Θ = 0 ⇒ every row shares the same Σ_f exp(β_f), hoisted to the pass
        // rather than re-summed over every live feature in every block.
        None => vec![dict.null_log_norm; bc],
    };
    // Θ̃ = [Θ | c] — the latent and its intercept in ONE `[Bc, H+1]` parameter, with
    // the intercept initialised at the conditional MLE given the latent.
    let mut theta = vec![0f32; bc * d];
    if let Some(init) = a.spec.init_theta {
        for (i, row) in init[a.start * h..a.end * h].chunks_exact(h).enumerate() {
            theta[i * d..i * d + h].copy_from_slice(row);
        }
    }
    for (i, (&n, &lz)) in n_tot.iter().zip(&log_norm).enumerate() {
        theta[i * d + h] = if n > 0.0 {
            (n.ln() - lz).clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32
        } else {
            -SCORE_CLAMP as f32
        };
    }
    let mut theta = Tensor::from_vec(theta, (bc, d), dev)?;

    ////////////////////////////////////
    // Loop-invariant gradient pieces //
    ////////////////////////////////////

    // **The data term is LINEAR in the parameters**, so its gradient is a constant:
    // `Σ_cf n_cf·s_cf` with `s = Θ̃·Ẽᵀ + β` gives `∂/∂Θ̃ = −N·Ẽ`, computed once
    // here instead of rebuilt (and back-propagated through) every step. The
    // intercept column of `N·Ẽ` is `Σ_f n_cf`, so the folded rows' `−n·c` term folds
    // straight into it.
    let ne = {
        let mut ne = n_t.matmul(e_aug_t)?; // [Bc, H+1]
                                           // Folded rows still owe `−n_dead·c`, which lands on the intercept column.
        if dict.dead_mass > 0.0 {
            ne = (ne + n_dead_t.reshape((bc, 1))?.broadcast_mul(intercept_mask)?)?;
        }
        ne
    };
    // Ridge applies to the latent only — the intercept is unpenalised, so the last
    // entry of the row is zero and one broadcast multiply covers both.
    let lam_row = {
        let mut v = vec![a.input.lambda as f32; d];
        v[h] = 0.0;
        Tensor::from_vec(v, (1, d), dev)?
    };

    ///////////////////
    // The Adam loop //
    ///////////////////

    // Hand-rolled gradient, hand-rolled Adam — **not** `loss.backward()`.
    //
    // The objective's gradient is three lines (`∂L/∂s = exp(s) − n`, chain through
    // one matmul), and taking it directly instead of through autograd is worth ~7×
    // here. candle's backward for this graph emits ~29 extra `[Bc, F]`-sized kernels
    // on top of the 8 the forward needs, because every op's backward materialises a
    // full-size `zeros_like` for BOTH operands and only afterwards discards the one
    // that isn't tracked — `Op::Matmul` computes the frozen dictionary's gradient
    // (a third full matmul, 50 % of the step's FLOPs) and `Op::Binary` does the same
    // for the constant count matrix and the broadcast bias. `.detach()` does not
    // prevent any of it: it only controls whether the *node* is visited, and the
    // node is visited because the parameter is.
    //
    // `GradStore::new` is private, so candle's `AdamW` cannot be driven from a
    // hand-built gradient; Adam on a `[Bc, H+1]` parameter is a handful of
    // elementwise ops on a tensor ~30 000× smaller than the score block, so it is
    // cheaper to write than to work around.
    let mut m = Tensor::zeros((bc, d), DType::F32, dev)?;
    let mut v = Tensor::zeros((bc, d), DType::F32, dev)?;

    // Convergence is `‖ΔΘ‖/‖Θ‖` over the LATENT columns, so the intercept — which
    // starts at the null model and barely moves — cannot mask a latent still in
    // flight.
    let mut prev = theta.narrow(1, 0, h)?.contiguous()?;
    let mut steps = 0usize;
    let mut converged = false;
    let mut emitted = 0usize; // cells this block has already reported to the bar
    let loop_start = std::time::Instant::now();
    let max_steps = a.spec.max_steps;
    for step in 0..max_steps {
        // Upper bound only. `exp` overflows f32 at 88 so the ceiling is a real
        // guard; the floor is not, since `exp(−large)` underflowing to 0 is the
        // right answer for a feature the cell does not express.
        //
        // Fused rather than `broadcast_add` → `minimum` → `exp`, which is the only
        // `[Bc, F]` work between two rayon-parallel gemms and would otherwise run on
        // one core — see [`FusedTensorOps`] for why. The matmul result is fresh and
        // unaliased, which is what lets the fused path consume it in place.
        let mu = theta
            .matmul(e_aug)?
            .clamped_exp_add_inplace(&dict.b_row, SCORE_CLAMP)?;

        // ∂L/∂Θ̃ = (μ − N)·Ẽ + λΘ̃, with the constant `N·Ẽ` hoisted above. The
        // intercept column comes out of the same matmul via `Ẽ`'s ones row, and
        // picks up the gate-folded partition mass `exp(c)·Σ_dead exp(β_f)`.
        let mut g = ((mu.matmul(e_aug_t)? - &ne)? + theta.broadcast_mul(&lam_row)?)?;
        if dict.dead_mass > 0.0 {
            let dead = theta.narrow(1, h, 1)?.exp()?.affine(dict.dead_mass, 0.0)?;
            g = (g + dead.broadcast_mul(intercept_mask)?)?;
        }

        // AdamW with `weight_decay = 0` — the ridge is already in `g` above, and a
        // decoupled decay would double-count it. The decayed, bias-corrected step
        // multiplier is [`adam_step_size`], shared with the per-track loop.
        m = ((&m * BETA1)? + (&g * (1.0 - BETA1))?)?;
        v = ((&v * BETA2)? + (g.sqr()? * (1.0 - BETA2))?)?;
        let step_size = adam_step_size(dict.lr0, step, max_steps);
        theta = (&theta - (&m * step_size)?.broadcast_div(&(v.sqrt()? + EPS)?)?)?;

        steps = step + 1;
        // Converged on `‖ΔΘ‖/‖Θ‖` — a parameter criterion, immune to the
        // `partition − data` cancellation that makes the loss a poor ruler here.
        //
        // The bar is advanced on this same stride, not per step: this is where the
        // loop already pays a device sync, so the update rides along for free, and
        // ~`MAX_STEPS/CHECK_EVERY` ticks per block is plenty of motion without
        // hammering the bar's lock and reformatting its message 400 times.
        //
        // This check is re-typed, not shared, in [`super::tracks`]'s per-track
        // loop: a change here belongs there too.
        if steps.is_multiple_of(CHECK_EVERY) {
            emitted = a.progress.advance(bc, steps, emitted);
            a.progress.describe(steps);
            let cur = theta.narrow(1, 0, h)?.contiguous()?;
            // One readback, not two: each `to_scalar` is a blocking device→host copy.
            let ds = Tensor::stack(
                &[(&cur - &prev)?.sqr()?.sum_all()?, cur.sqr()?.sum_all()?],
                0,
            )?
            .to_vec1::<f32>()?;
            prev = cur;
            if ds[1] > 0.0 && f64::from(ds[0] / ds[1]).sqrt() < TOL {
                converged = true;
                break;
            }
        }
    }
    let loop_secs = loop_start.elapsed().as_secs_f64();
    a.progress.finish_block(bc, emitted);

    /////////////////////////////
    // Read back + diagnostics //
    /////////////////////////////

    let s = theta.matmul(e_aug)?.broadcast_add(&dict.b_row)?;
    // One scalar, not the `[Bc, F]` block: did the overflow guard ever bind?
    let clamped = s.max_all()?.to_scalar::<f32>()? >= SCORE_CLAMP as f32;
    // Two-sided here (unlike the training loop): the deviance takes `ln(n/μ)`, so a
    // rate that underflowed to 0 would report an infinite one.
    let s = s.clamp(-SCORE_CLAMP, SCORE_CLAMP)?;
    // Poisson deviance over this block's observed edges — see [`poisson_deviance`],
    // which the per-track loop reduces through as well.
    let deviance = if n_edges > 0 {
        poisson_deviance(&n_t, &s)?
    } else {
        0.0
    };

    // Split `Θ̃` back into the latent and its intercept column.
    Ok(BlockOut {
        latent: theta
            .narrow(1, 0, h)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?,
        intercept: theta.narrow(1, h, 1)?.flatten_all()?.to_vec1::<f32>()?,
        steps,
        converged,
        clamped,
        deviance,
        n_edges,
        loop_secs,
    })
}
