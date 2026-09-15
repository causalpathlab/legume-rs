//! Epoch loop: every unit once per epoch in a seeded order, K modules drawn
//! per unit ∝ its share, one [`step`] per chunk of units; the composed
//! dictionary at the end.

use super::params::{HierParams, RowAdagrad};
use super::partition::{Partition, UnitModules};
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
}

pub struct HierOutput {
    pub e_u: DMatrix<f32>,
    pub rho: DMatrix<f32>,
    pub b_feat: Vec<f32>,
    /// Mean loss per unit over the last completed epoch; `NaN` when training
    /// stopped before any epoch completed (the tables are still finite).
    pub final_loss_per_unit: f64,
}

/// Draw `k` modules for each unit in `chunk` ∝ its composition `q_u·`, with
/// replacement, and emit one `(unit, weight)` pair per distinct module drawn,
/// weight = draw multiplicity / `k`. Weights for a unit sum to 1 across the
/// modules it lands in, so this is an unbiased estimator of the exhaustive
/// per-module sum — never dedup-and-drop the multiplicity. A unit with an
/// all-zero composition draws no modules, so it has no gene-level pairs; it
/// stays in `plan.units`, where its module-level term is exactly zero because
/// its weight (∝ total^½) is zero.
/// One module picker per unit, built once: a unit's composition never changes
/// during training. `None` for a unit with no counts.
pub(crate) fn module_pickers(um: &UnitModules, n_m: usize) -> Vec<Option<WeightedIndex<f64>>> {
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

pub(crate) fn draw_plan(
    chunk: &[u32],
    pickers: &[Option<WeightedIndex<f64>>],
    n_m: usize,
    k: usize,
    rng: &mut StdRng,
) -> StepPlan {
    let mut by_module: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_m];
    let inv_k = 1.0 / k.max(1) as f32;
    let mut counts: Vec<u32> = vec![0; n_m];
    for &u in chunk {
        let Some(picker) = pickers[u as usize].as_ref() else {
            continue;
        };
        counts.iter_mut().for_each(|c| *c = 0);
        for _ in 0..k {
            counts[picker.sample(rng)] += 1;
        }
        for (m, &c) in counts.iter().enumerate() {
            if c > 0 {
                by_module[m].push((u, c as f32 * inv_k));
            }
        }
    }
    StepPlan {
        units: chunk.to_vec(),
        pairs_by_module: by_module
            .into_iter()
            .enumerate()
            .filter(|(_, us)| !us.is_empty())
            .map(|(m, us)| (m as u32, us))
            .collect(),
    }
}

pub fn train(
    units: &UnitTable,
    labels: &[u32],
    h: usize,
    cfg: &HierConfig,
    stop: &AtomicBool,
) -> anyhow::Result<HierOutput> {
    anyhow::ensure!(
        units.tracks.is_base(),
        "multi-track training is not implemented yet"
    );
    anyhow::ensure!(
        labels.len() == units.tracks.n_genes(),
        "one module label per gene"
    );
    let part = Partition::from_labels(labels, cfg.n_modules);
    let um = UnitModules::new(units, &part);
    let (n_u, n_m, d) = (units.n_units(), part.n_modules(), units.tracks.n_genes());
    let mut params = HierParams::new(n_u, n_m, d, h, cfg.seed);
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(n_u, cfg.lr),
        mu: RowAdagrad::new(n_m, cfg.lr),
        r: RowAdagrad::new(d, cfg.lr),
    };
    let mut rng = StdRng::seed_from_u64(mix_seed(cfg.seed, 0x4849_4552));
    let pickers = module_pickers(&um, n_m);
    let mut order: Vec<u32> = (0..n_u as u32).collect();
    let steps_per_epoch = n_u.div_ceil(cfg.units_per_step.max(1));
    info!(
        "Phase 1 (hier) — {n_u} units × {d} genes in {n_m} modules, H={h}: {} epochs × \
         {steps_per_epoch} steps of {} units, K={} modules/unit, lr {}",
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
            let plan = draw_plan(chunk, &pickers, n_m, cfg.modules_per_unit, &mut rng);
            let (stats, grads): (StepStats, Grads) =
                loss_and_grads(&params, units, &um, &part, &plan);
            apply(&mut params, &mut opt, &grads, &plan, cfg.weight_decay);
            acc.loss_module += stats.loss_module;
            acc.loss_gene += stats.loss_gene;
            n_units_seen += chunk.len();
        }
        let per_unit = 1.0 / n_units_seen.max(1) as f64;
        last_per_unit = (acc.loss_module + acc.loss_gene) * per_unit;
        bar.inc(1);
        if (epoch + 1).is_multiple_of(REPORT_EVERY) || epoch + 1 == cfg.epochs {
            let ms = t0.elapsed().as_secs_f64() * 1e3 / ((epoch + 1) * steps_per_epoch) as f64;
            info!(
                "Phase 1 (hier) — epoch {}/{}: loss/unit {:.4} (module {:.4}, gene {:.4}), \
                 {:.1} ms/step",
                epoch + 1,
                cfg.epochs,
                last_per_unit,
                acc.loss_module * per_unit,
                acc.loss_gene * per_unit,
                ms
            );
        }
    }
    bar.finish_and_clear();

    // Compose the flat dictionary: ρ_g = μ_{m(g)} + r_g, b_g^flat = b_{m(g)} + b_g.
    let mut rho = DMatrix::<f32>::zeros(d, h);
    let mut b_feat = vec![0f32; d];
    for g in 0..d {
        let m = part.module_of[g] as usize;
        for k in 0..h {
            rho[(g, k)] = params.mu[m * h + k] + params.r[g * h + k];
        }
        b_feat[g] = params.b_m[m] + params.b_g[g];
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
