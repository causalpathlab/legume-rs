//! Epoch loop: every unit once per epoch in a seeded order, K modules drawn
//! per unit ∝ its share, one [`step`] per chunk of units; the composed
//! dictionary at the end.

use super::params::{HierParams, PresetGenes, PresetMode, PresetOffsets};
use super::partition::{Partition, TrackSupport, UnitModules};
use super::step::{apply, step_loss, Optimizers, StepCtx, StepPlan, StepStats};
use super::units::UnitTable;
use crate::progress::new_progress_bar;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::candle::convert::to_host;
use legume_numeric::matrix::rand_util::mix_seed;
use log::info;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use std::sync::atomic::{AtomicBool, Ordering};

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
    /// Rank of every non-base track's gene offset `u · V` (see
    /// [`super::params::TrackOffset`]): its own number, never derived from
    /// `h`; `1..=h` on a tracked axis, `h` being an unrestricted offset.
    /// Inert on a one-track axis.
    pub offset_rank: usize,
    /// Where the tables live and the steps run.
    pub device: Device,
    /// Per gene, `true` for a **module-only** feature: no residual, its row is
    /// its module's row and its bias the closed-form share of the module's
    /// counts, `ln(t_g / T_m)` over the pseudobulk units. The membership is the
    /// warm start's and does not move. Every module must hold only one kind.
    /// Empty when there are none.
    pub module_only: Vec<bool>,
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
    /// The module of every gene (the input labels).
    pub labels: Vec<u32>,
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
    preset_offsets: &[PresetOffsets],
    stop: &AtomicBool,
) -> anyhow::Result<HierOutput> {
    anyhow::ensure!(
        labels.len() == units.tracks.n_genes(),
        "one module label per gene"
    );
    if units.n_tracks() > 1 {
        crate::fit::config::validate_offset_rank(cfg.offset_rank, h)?;
    }
    let part = Partition::from_labels(labels, cfg.n_modules);
    let um = UnitModules::new(units, &part);
    // Each track's support through the partition: built once here, never per
    // step. Inert on a one-track axis, where the base track covers every gene.
    let sup = TrackSupport::new(&units.tracks, &part);
    let (n_u, n_m, d) = (units.n_units(), part.n_modules(), units.tracks.n_genes());
    let module_only = ModuleOnly::new(units, labels, cfg)?;
    let n_t = units.n_tracks();
    let n_features = units.n_features;
    let mut params =
        HierParams::new_tracked(n_u, n_m, d, n_t, h, cfg.offset_rank, cfg.seed, &cfg.device)?;
    if let Some(f) = preset {
        let mut is_module_only = vec![false; part.module_of.len()];
        for &g in module_only.iter().flat_map(|mo| &mo.genes) {
            is_module_only[g as usize] = true;
        }
        params.preset(f, &part.module_of, &is_module_only)?;
        let n_mo = f
            .ids
            .iter()
            .filter(|&&g| is_module_only[g as usize])
            .count();
        if n_mo > 0 && f.mode.pins() {
            info!(
                "Phase 1 (hier) — {n_mo} given module-only feature(s) carry no residual: \
                 each module holds the mean of its members' given rows"
            );
        }
        info!(
            "Phase 1 (hier) — {} of {d} gene rows {}{}",
            f.ids.len(),
            f.mode.describe(),
            f.mode.lora().map_or(String::new(), |l| format!(
                " (rank {}, V at {}× the rate, ridge {})",
                l.rank, l.lr_ratio, l.ridge
            ))
        );
    }
    if !preset_offsets.is_empty() {
        let mode = preset.map_or(PresetMode::Init, |f| f.mode);
        params.preset_offsets(preset_offsets, mode)?;
        info!(
            "Phase 1 (hier) — track offsets given for {} gene rows on {} track(s), {}",
            preset_offsets.iter().map(|p| p.ids.len()).sum::<usize>(),
            preset_offsets.len(),
            if matches!(mode, PresetMode::Freeze) {
                "pinned verbatim"
            } else {
                "as the start of the offset"
            }
        );
    }
    if n_t > 1 {
        info!(
            "Phase 1 (hier) — {} non-base track(s), each gene offset a rank-{} residual on \
             the base row (V at {}× the rate)",
            n_t - 1,
            cfg.offset_rank,
            params.offset_lr_ratio
        );
    }
    if let Some(mo) = &module_only {
        mo.pin(&params)?;
        info!(
            "Phase 1 (hier) — {} module-only feature(s) in {} module(s): no residual, \
             bias = share of the module's counts; {}",
            mo.genes.len(),
            mo.skip.iter().filter(|&&s| s).count(),
            mo.summary(&um, &part),
        );
    }
    let skip: Vec<bool> = module_only
        .as_ref()
        .map_or_else(Vec::new, |mo| mo.skip.clone());
    let mut opt = Optimizers::new(&params, cfg.lr)?;
    let ctx = StepCtx {
        units,
        um: &um,
        part: &part,
        sup: &sup,
        skip_module: &skip,
    };
    let mut rng = StdRng::seed_from_u64(mix_seed(cfg.seed, 0x4849_4552));
    let pickers = module_pickers(&um, n_m);
    let mut order: Vec<u32> = (0..n_u as u32).collect();
    let steps_per_epoch = n_u.div_ceil(cfg.units_per_step.max(1));
    let offset_l2_step = per_step_offset_l2(cfg.offset_l2, steps_per_epoch);
    let lora_ridge_step = params
        .lora
        .as_ref()
        .map_or(0.0, |l| per_step_offset_l2(l.ridge, steps_per_epoch));
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
            let (stats, loss): (StepStats, _) =
                step_loss(&params, &ctx, &plan, offset_l2_step, lora_ridge_step)?;
            let grads = loss.backward()?;
            apply(&mut params, &mut opt, &grads, cfg.lr, cfg.weight_decay)?;
            acc.loss_module += stats.loss_module;
            acc.loss_gene += stats.loss_gene;
            acc.loss_ridge += stats.loss_ridge;
            n_units_seen += chunk.len();
        }
        let per_unit = 1.0 / n_units_seen.max(1) as f64;
        last_per_unit = (acc.loss_module + acc.loss_gene + acc.loss_ridge) * per_unit;
        bar.inc(1);
        // Every epoch, at info: visible under `-v`, silent otherwise.
        let elapsed = t0.elapsed().as_secs_f64();
        let ms = elapsed * 1e3 / ((epoch + 1) * steps_per_epoch) as f64;
        let eta = elapsed / (epoch + 1) as f64 * (cfg.epochs - epoch - 1) as f64;
        info!(
            "Phase 1 (hier) — epoch {}/{}: loss/unit {:.4} (module {:.4}, gene {:.4}, \
             ridge {:.4}), {:.1} ms/step, eta {:.0} s",
            epoch + 1,
            cfg.epochs,
            last_per_unit,
            acc.loss_module * per_unit,
            acc.loss_gene * per_unit,
            acc.loss_ridge * per_unit,
            ms,
            eta
        );
    }
    bar.finish_and_clear();

    // The composed dictionary, one row per FEATURE ROW (see `HierParams::compose`).
    let (rho, b_feat) = params.compose(
        &units.tracks.track_of_row,
        &units.tracks.gene_of_row,
        &part.module_of,
    )?;
    let e_u_host = to_host(params.e_u.as_tensor())?;
    let e_u = DMatrix::<f32>::from_row_slice(n_u, h, &e_u_host);
    Ok(HierOutput {
        e_u,
        rho,
        b_feat,
        final_loss_per_unit: last_per_unit,
        labels: labels.to_vec(),
    })
}

/// The module-only half of the partition: which modules skip the gene level,
/// and each module-only feature's closed-form bias.
struct ModuleOnly {
    /// The module-only feature ids.
    genes: Vec<u32>,
    /// `ln(t_g / T_m)` per entry of `genes`, over the pseudobulk units.
    bias: Vec<f32>,
    /// Per module: module-only (no gene level).
    skip: Vec<bool>,
}

/// Floor on a count total before its log, so an unseen feature gets a finite
/// share instead of `ln 0`. Applied per member before the module sum, so an
/// all-zero module still has uniform shares `1/n` (LSE of biases = 0) rather
/// than every bias 0 (LSE = `ln n`).
const TOTAL_FLOOR: f32 = 1e-6;

impl ModuleOnly {
    /// `None` when [`HierConfig::module_only`] flags nothing. Refuses a module
    /// that holds both kinds of feature, and a multi-track axis.
    fn new(units: &UnitTable, labels: &[u32], cfg: &HierConfig) -> anyhow::Result<Option<Self>> {
        let mo = &cfg.module_only;
        if !mo.iter().any(|&b| b) {
            return Ok(None);
        }
        anyhow::ensure!(
            mo.len() == labels.len(),
            "module-only flags for {} features, labels for {}",
            mo.len(),
            labels.len()
        );
        anyhow::ensure!(
            units.n_tracks() == 1,
            "module-only features need a one-track feature axis"
        );
        let mut kind: Vec<Option<bool>> = vec![None; cfg.n_modules];
        for (g, (&m, &is_mo)) in labels.iter().zip(mo).enumerate() {
            match kind[m as usize] {
                None => kind[m as usize] = Some(is_mo),
                Some(k) => anyhow::ensure!(
                    k == is_mo,
                    "module {m} mixes module-only and residual features (feature {g})"
                ),
            }
        }
        // `t_g` over the pseudobulk units, then `T_m` over each module's members.
        let mut t = vec![0f32; labels.len()];
        for u in 0..units.n_pb_units {
            for (&g, &c) in units.feats[u].iter().zip(&units.counts[u]) {
                t[g as usize] += c;
            }
        }
        let mut module_total = vec![0f32; cfg.n_modules];
        for (g, &m) in labels.iter().enumerate() {
            if mo[g] {
                module_total[m as usize] += t[g].max(TOTAL_FLOOR);
            }
        }
        let genes: Vec<u32> = (0..labels.len() as u32)
            .filter(|&g| mo[g as usize])
            .collect();
        let bias = genes
            .iter()
            .map(|&g| {
                let m = labels[g as usize] as usize;
                (t[g as usize].max(TOTAL_FLOOR) / module_total[m]).ln()
            })
            .collect();
        Ok(Some(Self {
            genes,
            bias,
            skip: kind.iter().map(|k| *k == Some(true)).collect(),
        }))
    }

    /// Count mass and occupancy per kind of module, for the log: how much of
    /// the units' counts the gene level can still see, and how the members
    /// spread over the modules.
    fn summary(&self, um: &UnitModules, part: &Partition) -> String {
        let n_m = self.skip.len();
        let (mut mass_mo, mut mass_res) = (0f64, 0f64);
        for (i, &n) in um.n_um.iter().enumerate() {
            if self.skip[i % n_m] {
                mass_mo += f64::from(n);
            } else {
                mass_res += f64::from(n);
            }
        }
        let occupancy = |mo: bool| {
            let sizes: Vec<usize> = (0..n_m)
                .filter(|&m| self.skip[m] == mo)
                .map(|m| part.members[m].len())
                .collect();
            format!(
                "{} of {} occupied, largest {}",
                sizes.iter().filter(|&&s| s > 0).count(),
                sizes.len(),
                sizes.iter().max().copied().unwrap_or(0)
            )
        };
        format!(
            "count share residual {:.1}% / module-only {:.1}%; residual modules {}; \
             module-only modules {}",
            100.0 * mass_res / (mass_res + mass_mo).max(1.0),
            100.0 * mass_mo / (mass_res + mass_mo).max(1.0),
            occupancy(false),
            occupancy(true)
        )
    }

    /// Zero the module-only rows' residuals and set their biases. The gene
    /// level never scores them, so neither table takes a gradient there.
    fn pin(&self, params: &HierParams) -> anyhow::Result<()> {
        let (h, dev) = (params.h, &params.dev);
        let mut r = to_host(params.r.as_tensor())?;
        for &g in &self.genes {
            r[g as usize * h..(g as usize + 1) * h].fill(0.0);
        }
        let n_g = r.len() / h;
        params.r.set(&Tensor::from_vec(r, (n_g, h), dev)?)?;
        let mut b = to_host(params.b_g.as_tensor())?;
        for (&g, &v) in self.genes.iter().zip(&self.bias) {
            b[g as usize] = v;
        }
        let n_b = b.len();
        params.b_g.set(&Tensor::from_vec(b, n_b, dev)?)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod train_tests;
