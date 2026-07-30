//! The joint θ+δ alternative to the sequential two-pass: one SGD in which `θ` is
//! pulled by the spliced **and** unspliced tracks and `δ` by the unspliced one only.
//! It needs its own design (two intercepts, so no ones-row fold) and its own block
//! solve, but returns the same [`PassOut`] pair, so the caller is none the wiser.

use super::edges::{block_cells, EdgeTable};
use super::pass::{BlockProgress, PassOut, PassStats};
use super::{
    Phase2Input, CHECK_EVERY, GATE_FOLD_EPS, LR_FLOOR_FRAC, MAX_STEPS, TARGET_DELTA_S, TOL,
};
use crate::cell_projection::SCORE_CLAMP;
use candle_util::candle_core::{DType, Tensor};
use log::info;

/////////////////////////
// Joint θ+δ solve      //
/////////////////////////

/// One track's frozen design for the joint solve. Unlike [`super::pass::run_pass`]'s
/// augmented `Ẽ` (ones row folding the intercept), there is **no** ones row — the joint
/// solve carries the spliced and unspliced intercepts as separate parameters, so each
/// track's design is just `E [F_live, H]` (+ transpose) and its bias.
struct TrackDict {
    /// Global feature ids of the live (non-gate-folded) rows of this track.
    live: Vec<u32>,
    /// `E [F_live, H]` and its transpose `Eᵀ [H, F_live]` — both needed every step.
    e: Tensor,
    e_t: Tensor,
    /// `β [1, F_live]`.
    b_row: Tensor,
    /// `ln(Σ_f exp(β_f) + dead_mass)`, the null-model normaliser for this track.
    null_log_norm: f64,
    /// Gate-folded partition mass `Σ_dead exp(β_f)` (0 at the default `GATE_FOLD_EPS`).
    dead_mass: f64,
    /// Global → live-local feature id (`u32::MAX` off this track's live set).
    to_live: Vec<u32>,
    f_live: usize,
}

fn build_track(input: &Phase2Input, rows: &[u32]) -> anyhow::Result<TrackDict> {
    let (h, dev) = (input.h, input.dev);
    let mut live: Vec<u32> = Vec::with_capacity(rows.len());
    let mut dead_mass = 0f64;
    for &g in rows {
        let e = &input.feat[g as usize * h..(g as usize + 1) * h];
        if e.iter().map(|x| x * x).sum::<f32>().sqrt() > GATE_FOLD_EPS {
            live.push(g);
        } else {
            dead_mass += f64::from(input.b_feat[g as usize]).exp();
        }
    }
    anyhow::ensure!(
        !live.is_empty(),
        "joint phase-2: a splice track has no live features to project onto"
    );
    let f_live = live.len();
    let mut e_flat = vec![0f32; f_live * h];
    let mut b_live = vec![0f32; f_live];
    for (l, &g) in live.iter().enumerate() {
        e_flat[l * h..l * h + h].copy_from_slice(&input.feat[g as usize * h..(g as usize + 1) * h]);
        b_live[l] = input.b_feat[g as usize];
    }
    let e = Tensor::from_vec(e_flat, (f_live, h), dev)?;
    let e_t = e.t()?.contiguous()?;
    let b_row = Tensor::from_vec(b_live, (1, f_live), dev)?;
    let null_log_norm = (f64::from(b_row.exp()?.sum_all()?.to_scalar::<f32>()?) + dead_mass)
        .max(f64::MIN_POSITIVE)
        .ln();
    let mut to_live = vec![u32::MAX; input.b_feat.len()];
    for (l, &g) in live.iter().enumerate() {
        to_live[g as usize] = l as u32;
    }
    Ok(TrackDict {
        live,
        e,
        e_t,
        b_row,
        null_log_norm,
        dead_mass,
        to_live,
        f_live,
    })
}

/// Joint θ+δ solve over the spliced (`s`) and unspliced (`u`) partitions in ONE SGD:
/// `θ` is pulled by **both** tracks, `δ` by the unspliced track only. Returns the same
/// `(θ, δ)` `PassOut` pair the sequential two-pass produces, so the caller's gauge-fix
/// and scatter are unchanged. `θ` carries the kept per-cell intercept `c_s` (= `b_cell`);
/// `δ`'s `c_u` is a throwaway depth for the unspliced track.
pub(super) fn run_joint_pass(
    input: &Phase2Input,
    rows_s: &[u32],
    edges_s: &EdgeTable,
    rows_u: &[u32],
    edges_u: &EdgeTable,
    cells: &[(u32, &[u32], &[f32])],
    bar: &indicatif::ProgressBar,
) -> anyhow::Result<(PassOut, PassOut)> {
    let h = input.h;
    let n_kept = cells.len();

    let s = build_track(input, rows_s)?;
    let u = build_track(input, rows_u)?;

    // Both tracks' `[Bc, F]` activations are resident at once, so size `Bc` from the
    // COMBINED live feature count (half what a single pass would take).
    let block_bc = block_cells(s.f_live + u.f_live);

    // Auto-lr from the combined dictionary rms — θ sees both tracks' scale.
    // Same L1 scale as the θ-only pass above — see the note there.
    let e_mean = {
        let sa: f64 = s
            .live
            .iter()
            .chain(u.live.iter())
            .flat_map(|&g| &input.feat[g as usize * h..(g as usize + 1) * h])
            .map(|x| f64::from(*x).abs())
            .sum();
        (sa / ((s.f_live + u.f_live) * h) as f64).max(1e-12)
    };
    let lr0 = TARGET_DELTA_S / (h as f64 * e_mean);

    info!(
        "{} [joint θ+δ] — {n_kept} cell(s) × ({}+{}) live features, blocks of {}, \
         lr {:.4} (auto: Δs≈{TARGET_DELTA_S}), ≤{MAX_STEPS} steps, ridge λ={}",
        input.label, s.f_live, u.f_live, block_bc, lr0, input.lambda,
    );

    let mut theta = vec![0f32; n_kept * h];
    let mut delta = vec![0f32; n_kept * h];
    let mut c_s = vec![0f32; n_kept];
    let mut c_u = vec![0f32; n_kept];
    let mut stats = PassStats::default();
    let n_blocks = n_kept.div_ceil(block_bc);

    for (b, start) in (0..n_kept).step_by(block_bc).enumerate() {
        let end = (start + block_bc).min(n_kept);
        let out = solve_joint_block(JointBlockArgs {
            input,
            s: &s,
            u: &u,
            rows_s,
            rows_u,
            edges_s,
            edges_u,
            lr0,
            start,
            end,
            progress: &BlockProgress {
                bar,
                stats: &stats,
                label: "joint",
                block: b + 1,
                n_blocks,
            },
        })?;
        theta[start * h..end * h].copy_from_slice(&out.theta);
        delta[start * h..end * h].copy_from_slice(&out.delta);
        c_s[start..end].copy_from_slice(&out.c_s);
        c_u[start..end].copy_from_slice(&out.c_u);
        // Fold the block's stats in (PassStats fields are module-local).
        stats.blocks += 1;
        stats.steps += out.steps;
        stats.at_cap += usize::from(!out.converged);
        stats.clamped += usize::from(out.clamped);
        stats.dev_sum += out.deviance;
        stats.dev_n += out.n_edges as f64;
        stats.secs += out.loop_secs;
    }

    info!(
        "{} [joint θ+δ] — done: ⌀{:.0} steps/block, {} of {} block(s) hit the {MAX_STEPS}-step \
         cap, mean per-edge deviance {:.4}, {:.0}s total ({:.1} ms/step)",
        input.label,
        stats.mean_steps(),
        stats.at_cap,
        stats.blocks,
        stats.mean_deviance(),
        stats.secs,
        stats.ms_per_step(),
    );

    Ok((
        PassOut {
            latent: theta,
            intercept: c_s,
        },
        PassOut {
            latent: delta,
            intercept: c_u,
        },
    ))
}

struct JointBlockArgs<'a> {
    input: &'a Phase2Input<'a>,
    s: &'a TrackDict,
    u: &'a TrackDict,
    rows_s: &'a [u32],
    rows_u: &'a [u32],
    edges_s: &'a EdgeTable,
    edges_u: &'a EdgeTable,
    lr0: f64,
    start: usize,
    end: usize,
    progress: &'a BlockProgress<'a>,
}

struct JointBlockOut {
    theta: Vec<f32>,
    delta: Vec<f32>,
    c_s: Vec<f32>,
    c_u: Vec<f32>,
    steps: usize,
    converged: bool,
    clamped: bool,
    deviance: f64,
    n_edges: usize,
    loop_secs: f64,
}

fn solve_joint_block(a: JointBlockArgs) -> anyhow::Result<JointBlockOut> {
    let (h, dev) = (a.input.h, a.input.dev);
    let bc = a.end - a.start;
    let lambda = a.input.lambda;

    // Dense counts + per-cell totals + folded-count mass, per track.
    let gather = |track: &TrackDict,
                  rows: &[u32],
                  edges: &EdgeTable|
     -> anyhow::Result<(Tensor, Vec<f64>, Tensor, usize)> {
        let f_live = track.f_live;
        let mut n_dense = vec![0f32; bc * f_live];
        let mut n_tot = vec![0f64; bc];
        let mut n_dead = vec![0f32; bc];
        let mut n_edges = 0usize;
        for i in a.start..a.end {
            let local = i - a.start;
            let (feats, counts) = edges.cell_slice(i);
            for (&f_pass, &n) in feats.iter().zip(counts) {
                let g = rows[f_pass as usize];
                let l = track.to_live[g as usize];
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
        let n_dead_t = Tensor::from_vec(n_dead, (bc, 1), dev)?;
        Ok((n_t, n_tot, n_dead_t, n_edges))
    };
    let (ns_t, ns_tot, ns_dead, n_edges_s) = gather(a.s, a.rows_s, a.edges_s)?;
    let (nu_t, nu_tot, nu_dead, n_edges_u) = gather(a.u, a.rows_u, a.edges_u)?;
    let n_edges = n_edges_s + n_edges_u;

    // Null-model intercept per track: c = ln(Σ_f n) − ln Σ_f exp(β) at θ = δ = 0.
    let init_c = |n_tot: &[f64], null_ln: f64| -> Vec<f32> {
        n_tot
            .iter()
            .map(|&n| {
                if n > 0.0 {
                    (n.ln() - null_ln).clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32
                } else {
                    -SCORE_CLAMP as f32
                }
            })
            .collect()
    };
    let mut theta = Tensor::zeros((bc, h), DType::F32, dev)?;
    let mut delta = Tensor::zeros((bc, h), DType::F32, dev)?;
    let mut c_s = Tensor::from_vec(init_c(&ns_tot, a.s.null_log_norm), (bc, 1), dev)?;
    let mut c_u = Tensor::from_vec(init_c(&nu_tot, a.u.null_log_norm), (bc, 1), dev)?;

    // Hoisted constant data-gradient pieces (the data term is linear in the params),
    // all loop-invariant: `N_u·E_u` (feeds both θ and δ), the θ-track sum
    // `N_s·E_s + N_u·E_u`, and `Σ_f n` for each intercept.
    let ne_u = nu_t.matmul(&a.u.e)?; // [Bc, H]
    let ne_sum = (ns_t.matmul(&a.s.e)? + &ne_u)?; // [Bc, H] — θ's full data gradient
    let ncs = ns_t.sum_keepdim(1)?; // [Bc, 1]
    let ncu = nu_t.sum_keepdim(1)?; // [Bc, 1]

    let (beta1, beta2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    let mut m_th = Tensor::zeros((bc, h), DType::F32, dev)?;
    let mut v_th = Tensor::zeros((bc, h), DType::F32, dev)?;
    let mut m_dl = Tensor::zeros((bc, h), DType::F32, dev)?;
    let mut v_dl = Tensor::zeros((bc, h), DType::F32, dev)?;
    let mut m_cs = Tensor::zeros((bc, 1), DType::F32, dev)?;
    let mut v_cs = Tensor::zeros((bc, 1), DType::F32, dev)?;
    let mut m_cu = Tensor::zeros((bc, 1), DType::F32, dev)?;
    let mut v_cu = Tensor::zeros((bc, 1), DType::F32, dev)?;

    let mut prev = Tensor::cat(&[&theta, &delta], 1)?;
    let mut steps = 0usize;
    let mut converged = false;
    let mut emitted = 0usize;
    let loop_start = std::time::Instant::now();
    for step in 0..MAX_STEPS {
        let frac = step as f64 / MAX_STEPS as f64;
        let lr = a.lr0 * (1.0 - frac * (1.0 - LR_FLOOR_FRAC));
        let t = (step + 1) as f64;
        let step_size = lr * (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

        // Scores. Spliced sees θ; unspliced sees the nascent state θ+δ. Upper-clamp only.
        let s_s = theta
            .matmul(&a.s.e_t)?
            .broadcast_add(&c_s)?
            .broadcast_add(&a.s.b_row)?
            .minimum(SCORE_CLAMP)?;
        let nascent = (&theta + &delta)?;
        let s_u = nascent
            .matmul(&a.u.e_t)?
            .broadcast_add(&c_u)?
            .broadcast_add(&a.u.b_row)?
            .minimum(SCORE_CLAMP)?;
        let mu_s = s_s.exp()?;
        let mu_u = s_u.exp()?;

        // ∂L/∂θ = (μ_s−N_s)·E_s + (μ_u−N_u)·E_u + λθ ; ∂L/∂δ = (μ_u−N_u)·E_u + λδ.
        // `μ_u·E_u` is shared, computed once.
        let mu_u_e = mu_u.matmul(&a.u.e)?;
        let g_theta = (((mu_s.matmul(&a.s.e)? + &mu_u_e)? - &ne_sum)? + (&theta * lambda)?)?;
        let g_delta = ((&mu_u_e - &ne_u)? + (&delta * lambda)?)?;
        // ∂L/∂c = (Σ_f μ − Σ_f n), plus the folded features' `exp(c)·dead_mass − n_dead`
        // when the gate actually folded any (dead_mass = 0 ⟺ no fold ⟺ n_dead = 0).
        let g_cs = {
            let g = (mu_s.sum_keepdim(1)? - &ncs)?;
            if a.s.dead_mass > 0.0 {
                ((g + c_s.exp()?.affine(a.s.dead_mass, 0.0)?)? - &ns_dead)?
            } else {
                g
            }
        };
        let g_cu = {
            let g = (mu_u.sum_keepdim(1)? - &ncu)?;
            if a.u.dead_mass > 0.0 {
                ((g + c_u.exp()?.affine(a.u.dead_mass, 0.0)?)? - &nu_dead)?
            } else {
                g
            }
        };

        // AdamW (decoupled decay = 0; the ridge is already in the gradients).
        let adam =
            |p: &Tensor, g: &Tensor, m: &mut Tensor, vv: &mut Tensor| -> anyhow::Result<Tensor> {
                *m = ((&*m * beta1)? + (g * (1.0 - beta1))?)?;
                *vv = ((&*vv * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                Ok((p - (&*m * step_size)?.broadcast_div(&(vv.sqrt()? + eps)?)?)?)
            };
        theta = adam(&theta, &g_theta, &mut m_th, &mut v_th)?;
        delta = adam(&delta, &g_delta, &mut m_dl, &mut v_dl)?;
        c_s = adam(&c_s, &g_cs, &mut m_cs, &mut v_cs)?;
        c_u = adam(&c_u, &g_cu, &mut m_cu, &mut v_cu)?;

        steps = step + 1;
        // Convergence on `‖Δ[θ,δ]‖ / ‖[θ,δ]‖`.
        if steps.is_multiple_of(CHECK_EVERY) {
            emitted = a.progress.advance(bc, steps, emitted);
            a.progress.describe(steps);
            let cur = Tensor::cat(&[&theta, &delta], 1)?;
            let dd = Tensor::stack(
                &[(&cur - &prev)?.sqr()?.sum_all()?, cur.sqr()?.sum_all()?],
                0,
            )?
            .to_vec1::<f32>()?;
            prev = cur;
            if dd[1] > 0.0 && f64::from(dd[0] / dd[1]).sqrt() < TOL {
                converged = true;
                break;
            }
        }
    }
    let loop_secs = loop_start.elapsed().as_secs_f64();
    a.progress.finish_block(bc, emitted);

    // Deviance over both tracks' observed edges, plus a clamp-bound check.
    let track_dev = |theta_eff: &Tensor,
                     track: &TrackDict,
                     c: &Tensor,
                     n_t: &Tensor,
                     n_edges: usize|
     -> anyhow::Result<(f64, bool)> {
        if n_edges == 0 {
            return Ok((0.0, false));
        }
        let s = theta_eff
            .matmul(&track.e_t)?
            .broadcast_add(c)?
            .broadcast_add(&track.b_row)?;
        let clamped = s.max_all()?.to_scalar::<f32>()? >= SCORE_CLAMP as f32;
        let s = s.clamp(-SCORE_CLAMP, SCORE_CLAMP)?;
        let mu = s.exp()?;
        let mask = n_t.gt(0f32)?.to_dtype(DType::F32)?;
        let log_n = n_t.clamp(1f32, f32::MAX)?.log()?;
        let term = ((n_t * (log_n - &s)?)? - ((n_t - &mu)? * &mask)?)?;
        Ok((
            f64::from(term.sum_all()?.to_scalar::<f32>()?) * 2.0,
            clamped,
        ))
    };
    let (dev_s, clamp_s) = track_dev(&theta, a.s, &c_s, &ns_t, n_edges_s)?;
    let nascent = (&theta + &delta)?;
    let (dev_u, clamp_u) = track_dev(&nascent, a.u, &c_u, &nu_t, n_edges_u)?;
    let clamped = clamp_s || clamp_u;
    let deviance = dev_s + dev_u;

    Ok(JointBlockOut {
        theta: theta.flatten_all()?.to_vec1::<f32>()?,
        delta: delta.flatten_all()?.to_vec1::<f32>()?,
        c_s: c_s.flatten_all()?.to_vec1::<f32>()?,
        c_u: c_u.flatten_all()?.to_vec1::<f32>()?,
        steps,
        converged,
        clamped,
        deviance,
        n_edges,
        loop_secs,
    })
}
