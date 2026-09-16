//! Epoch loop: every unit once per epoch in a seeded order, K modules drawn
//! per unit ∝ its share, one [`step`] per chunk of units; the composed
//! dictionary at the end.

use super::params::{HierParams, PresetGenes, RowAdagrad};
use super::partition::{Partition, TrackSupport, UnitModules};
use super::step::{apply, loss_and_grads, Grads, Optimizers, StepPlan, StepStats};
use super::units::UnitTable;
use crate::progress::new_progress_bar;
use log::info;
use matrix_util::rand_util::mix_seed;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use std::sync::atomic::{AtomicBool, Ordering};

const REPORT_EVERY: usize = 50;

pub struct HierConfig {
    pub n_modules: usize,
    pub epochs: usize,
    pub units_per_step: usize,
    pub modules_per_unit: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub seed: u64,
    /// Ridge on the non-base tracks' offset tables (see [`super::step`]), as a
    /// PER-EPOCH weight: one step carries `1 / steps_per_epoch` of it (see
    /// [`per_step_offset_l2`]), so over an epoch the penalty is exactly
    /// `offset_l2 · (mean_m ‖Δ‖² + mean_g ‖δ‖²)` whatever `units_per_step` is.
    /// Inert on a one-track axis, which has no offsets.
    pub offset_l2: f32,
}

pub struct HierOutput {
    pub e_u: DMatrix<f32>,
    /// `[n_features × H]`, one row per FEATURE ROW: the composed dictionary of
    /// that row's `(track, gene)`.
    pub rho: DMatrix<f32>,
    /// `[n_features]`, likewise per feature row.
    pub b_feat: Vec<f32>,
    /// Mean loss per unit over the last completed epoch; `NaN` when training
    /// stopped before any epoch completed (the tables are still finite).
    pub final_loss_per_unit: f64,
}

/// One module picker per `(unit, TRACK)`, indexed `u * T + t` — the layout
/// [`UnitModules::idx`] already uses, so chunking `q` by `n_m` walks the pairs
/// in that order. Built once: a unit's composition never changes during
/// training. `None` for a (unit, track) with no counts.
pub(crate) fn module_pickers(um: &UnitModules, n_m: usize) -> Vec<Option<WeightedIndex<f64>>> {
    debug_assert_eq!(um.n_modules, n_m, "picker chunk width is the module count");
    um.q.chunks_exact(n_m)
        .map(|q| {
            if q.iter().all(|&x| x == 0.0) {
                None
            } else {
                Some(
                    WeightedIndex::new(q.iter().map(|&x| f64::from(x)))
                        .expect("non-negative, not all zero"),
                )
            }
        })
        .collect()
}

/// The ridge weight ONE step carries. [`HierConfig::offset_l2`] is a per-epoch
/// weight and the ridge is exact on the full offset tables at every step, so a
/// step takes `1 / steps_per_epoch` of it: an epoch's steps then sum to exactly
/// the per-epoch figure, and `offset_l2` means the same thing at every batch
/// size. Without the division, halving `units_per_step` would double the
/// effective penalty and inflate every offset row's Adagrad accumulator twice
/// as fast.
pub(crate) fn per_step_offset_l2(offset_l2: f32, steps_per_epoch: usize) -> f32 {
    offset_l2 / steps_per_epoch.max(1) as f32
}

/// Draw `k` modules for each unit in `chunk` ∝ its composition `q_u·`, with
/// replacement, and emit one `(unit, weight)` pair per distinct module drawn,
/// weight = draw multiplicity / `k`. Weights for a unit sum to 1 across the
/// modules it lands in, so this is an unbiased estimator of the exhaustive
/// per-module sum — never dedup-and-drop the multiplicity. A unit with an
/// all-zero composition draws no modules, so it has no gene-level pairs; it
/// stays in `plan.units`, where its module-level term is exactly zero because
/// its weight (∝ total^½) is zero.
pub(crate) fn draw_plan(
    chunk: &[u32],
    pickers: &[Option<WeightedIndex<f64>>],
    n_m: usize,
    n_t: usize,
    k: usize,
    rng: &mut StdRng,
) -> StepPlan {
    // Track OUTER, then the chunk's units: at `n_t == 1` that is the plain
    // per-unit loop, so the RNG is consumed draw for draw as it was before
    // tracks existed. Buckets are indexed `t * M + m`, so the kept groups come
    // out ordered by `(track, module)` — module order at one track.
    let mut by_module: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_t * n_m];
    let inv_k = 1.0 / k.max(1) as f32;
    let mut counts: Vec<u32> = vec![0; n_m];
    for t in 0..n_t {
        for &u in chunk {
            let Some(picker) = pickers[u as usize * n_t + t].as_ref() else {
                continue;
            };
            counts.iter_mut().for_each(|c| *c = 0);
            for _ in 0..k {
                counts[picker.sample(rng)] += 1;
            }
            for (m, &c) in counts.iter().enumerate() {
                if c > 0 {
                    by_module[t * n_m + m].push((u, c as f32 * inv_k));
                }
            }
        }
    }
    StepPlan {
        units: chunk.to_vec(),
        pairs_by_module: by_module
            .into_iter()
            .enumerate()
            .filter(|(_, us)| !us.is_empty())
            .map(|(k, us)| (((k / n_m) as u32, (k % n_m) as u32), us))
            .collect(),
    }
}

pub fn train(
    units: &UnitTable,
    labels: &[u32],
    h: usize,
    cfg: &HierConfig,
    preset: Option<&PresetGenes>,
    stop: &AtomicBool,
) -> anyhow::Result<HierOutput> {
    anyhow::ensure!(
        labels.len() == units.tracks.n_genes(),
        "one module label per gene"
    );
    let part = Partition::from_labels(labels, cfg.n_modules);
    let um = UnitModules::new(units, &part);
    // Each track's support through the partition: built once here, never per
    // step. Inert on a one-track axis, where the base track covers every gene.
    let sup = TrackSupport::new(&units.tracks, &part);
    let (n_u, n_m, d) = (units.n_units(), part.n_modules(), units.tracks.n_genes());
    let n_t = units.n_tracks();
    let n_features = units.n_features;
    let mut params = HierParams::new_tracked(n_u, n_m, d, n_t, h, cfg.seed);
    if let Some(f) = preset {
        anyhow::ensure!(
            n_t == 1,
            "preset gene rows need a single-track feature axis"
        );
        params.preset(f, &part.module_of)?;
        if f.freeze {
            info!(
                "Phase 1 (hier) — {} of {d} gene rows pinned; μ pinned to their module means, \
                 biases and the other rows train",
                f.gene.len()
            );
        } else {
            info!(
                "Phase 1 (hier) — {} of {d} gene rows start from the given table and train on",
                f.gene.len()
            );
        }
    }
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(n_u, cfg.lr),
        mu: RowAdagrad::new(n_m, cfg.lr),
        r: RowAdagrad::new(d, cfg.lr),
        offsets: (1..n_t)
            .map(|_| (RowAdagrad::new(n_m, cfg.lr), RowAdagrad::new(d, cfg.lr)))
            .collect(),
    };
    let mut rng = StdRng::seed_from_u64(mix_seed(cfg.seed, 0x4849_4552));
    let pickers = module_pickers(&um, n_m);
    let mut order: Vec<u32> = (0..n_u as u32).collect();
    let steps_per_epoch = n_u.div_ceil(cfg.units_per_step.max(1));
    let offset_l2_step = per_step_offset_l2(cfg.offset_l2, steps_per_epoch);
    info!(
        "Phase 1 (hier) — {n_u} units × {d} genes on {n_t} track(s) ({n_features} feature rows) \
         in {n_m} modules, H={h}: {} epochs × {steps_per_epoch} steps of {} units, K={} \
         modules/unit, lr {}",
        cfg.epochs, cfg.units_per_step, cfg.modules_per_unit, cfg.lr
    );
    let bar = new_progress_bar(cfg.epochs as u64);
    let mut last_per_unit = f64::NAN;
    let t0 = std::time::Instant::now();
    'epochs: for epoch in 0..cfg.epochs {
        order.shuffle(&mut rng);
        let mut acc = StepStats::default();
        let mut n_units_seen = 0usize;
        for chunk in order.chunks(cfg.units_per_step.max(1)) {
            if stop.load(Ordering::Relaxed) {
                info!("Phase 1 (hier) — stop requested at epoch {epoch}");
                break 'epochs;
            }
            let plan = draw_plan(chunk, &pickers, n_m, n_t, cfg.modules_per_unit, &mut rng);
            let (stats, grads): (StepStats, Grads) =
                loss_and_grads(&params, units, &um, &part, &sup, &plan, offset_l2_step);
            apply(&mut params, &mut opt, &grads, &plan, cfg.weight_decay);
            acc.loss_module += stats.loss_module;
            acc.loss_gene += stats.loss_gene;
            acc.loss_ridge += stats.loss_ridge;
            n_units_seen += chunk.len();
        }
        let per_unit = 1.0 / n_units_seen.max(1) as f64;
        last_per_unit = (acc.loss_module + acc.loss_gene + acc.loss_ridge) * per_unit;
        bar.inc(1);
        if (epoch + 1).is_multiple_of(REPORT_EVERY) || epoch + 1 == cfg.epochs {
            let ms = t0.elapsed().as_secs_f64() * 1e3 / ((epoch + 1) * steps_per_epoch) as f64;
            info!(
                "Phase 1 (hier) — epoch {}/{}: loss/unit {:.4} (module {:.4}, gene {:.4}, \
                 ridge {:.4}), {:.1} ms/step",
                epoch + 1,
                cfg.epochs,
                last_per_unit,
                acc.loss_module * per_unit,
                acc.loss_gene * per_unit,
                acc.loss_ridge * per_unit,
                ms
            );
        }
    }
    bar.finish_and_clear();

    // Compose the flat dictionary, one row per FEATURE ROW: for row `r` naming
    // gene `g` on track `t`, ρ_r = μ_{m(g)} + r_g (+ Δ^t_{m(g)} + δ^t_g) and
    // b_r = b_{m(g)} + b_g (+ β^t_{m(g)} + γ^t_g). On the base track the offset
    // terms do not exist and the expression is the plain composed pair.
    let mut rho = DMatrix::<f32>::zeros(n_features, h);
    let mut row_buf = vec![0f32; h];
    let mut b_feat = vec![0f32; n_features];
    for row in 0..n_features {
        let t = units.tracks.track_of_row[row] as usize;
        let g = units.tracks.gene_of_row[row] as usize;
        let m = part.module_of[g] as usize;
        match params.offset(t) {
            None => {
                params.base_row(g, m, &mut row_buf);
                for k in 0..h {
                    rho[(row, k)] = row_buf[k];
                }
                b_feat[row] = params.b_m[m] + params.b_g[g];
            }
            Some(o) => {
                for k in 0..h {
                    rho[(row, k)] = params.mu[m * h + k]
                        + params.r[g * h + k]
                        + o.d_mu[m * h + k]
                        + o.d_r[g * h + k];
                }
                b_feat[row] = params.b_m[m] + params.b_g[g] + o.d_b_m[m] + o.d_b_g[g];
            }
        }
    }
    Ok(HierOutput {
        e_u: DMatrix::<f32>::from_row_slice(n_u, h, &params.e_u),
        rho,
        b_feat,
        final_loss_per_unit: last_per_unit,
    })
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod train_tests;
