//! Epoch loop: every unit once per epoch in a seeded order, K modules drawn
//! per unit ∝ its share, one [`step`] per chunk of units; the composed
//! dictionary at the end.

use super::cis_gates::CisMix;
use super::params::{GroupIntercepts, HierParams, PresetGenes, PresetMode, PresetOffsets};
use super::partition::{Partition, TrackSupport, UnitModules};
use super::step::{
    apply, cis_align_loss, cis_dictionary_blend, cis_mix, cis_pool, cis_readout, step_loss,
    Optimizers, StepCtx, StepPlan, StepStats,
};
use super::units::UnitTable;
use crate::progress::new_progress_bar;
use legume_numeric::candle::candle_core::backprop::GradStore;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::candle::convert::to_host;
use legume_numeric::matrix::rand_util::mix_seed;
use log::info;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct HierConfig {
    pub n_modules: usize,
    pub epochs: usize,
    pub units_per_step: usize,
    pub modules_per_unit: usize,
    pub lr: f32,
    /// Decay on the feature rows (`μ`, `r`); see [`super::step::apply`].
    pub weight_decay: f32,
    /// Decay on the unit rows `e_u` (pseudobulks and phase-1 cells).
    pub unit_weight_decay: f32,
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
    /// Optional cis pairs coupling RNA gene rows with their cis peaks'
    /// modules, and how strongly (chickpea).
    pub cis_gates: Option<super::cis_gates::CisCoupling>,
    /// Per module, its group (a modality on a multiome axis): with two or more
    /// groups every unit gets one intercept per non-reference group on its
    /// module scores (see [`super::params::GroupIntercepts`]). Empty for none.
    pub module_group: Vec<u32>,
    /// Per module, `true` for a background of near-empty or scattered
    /// features: cis pairs whose peak sits there are dropped. Empty for none.
    pub background_modules: Vec<bool>,
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
    /// Trained cis gates (shared θ, pair shares, evidence and the gap), when
    /// configured.
    pub cis: Option<super::cis_gates::CisGateReadout>,
    /// `[n_units, K − 1]` per-unit group intercepts, when
    /// [`HierConfig::module_group`] named two or more groups.
    pub group_intercepts: Option<DMatrix<f32>>,
}

/// A `(unit, track)`'s module draw: `q` restricted to the modules with a gene
/// level, and the unit's share of counts on them.
pub(crate) struct ModulePicker {
    pick: WeightedIndex<f64>,
    /// `Σ_{m with a gene level} q_um`: each draw's weight carries it, so the
    /// draws stay an unbiased estimator of `Σ_m q_um·L₂(u, m)` (a module-only
    /// module's `L₂` is zero).
    share: f32,
}

/// One module picker per `(unit, TRACK)`, indexed `u * T + t` — the layout
/// [`UnitModules::idx`] already uses, so chunking `q` by `n_m` walks the pairs
/// in that order. Built once: a unit's composition never changes during
/// training. Modules flagged in `skip` (module-only: no gene level) are never
/// drawn — a draw there would be dropped by the step. `None` for a (unit,
/// track) with no counts off the skipped modules. `skip` may be empty.
pub(crate) fn module_pickers(
    um: &UnitModules,
    n_m: usize,
    skip: &[bool],
) -> Vec<Option<ModulePicker>> {
    debug_assert_eq!(um.n_modules, n_m, "picker chunk width is the module count");
    let kept = |m: usize| !skip.get(m).copied().unwrap_or(false);
    um.q.chunks_exact(n_m)
        .map(|q| {
            let w: Vec<f64> = q
                .iter()
                .enumerate()
                .map(|(m, &x)| if kept(m) { f64::from(x) } else { 0.0 })
                .collect();
            let share: f64 = w.iter().sum();
            (share > 0.0).then(|| ModulePicker {
                pick: WeightedIndex::new(w).expect("non-negative, not all zero"),
                share: share as f32,
            })
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

/// Draw `k` modules for each unit in `chunk` ∝ its composition `q_u·` over the
/// modules with a gene level, with replacement, and emit one `(unit, weight)`
/// pair per distinct module drawn, weight = share · draw multiplicity / `k`
/// (share = the unit's count share on those modules, see [`ModulePicker`]).
/// Weights for a unit sum to that share across the modules it lands in, so
/// this is an unbiased estimator of the exhaustive per-module sum — never
/// dedup-and-drop the multiplicity. A unit with no counts on such modules
/// draws none, so it has no gene-level pairs; it stays in `plan.units` for its
/// module-level term.
pub(crate) fn draw_plan(
    chunk: &[u32],
    pickers: &[Option<ModulePicker>],
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
                counts[picker.pick.sample(rng)] += 1;
            }
            let per_draw = picker.share * inv_k;
            for (m, &c) in counts.iter().enumerate() {
                if c > 0 {
                    by_module[t * n_m + m].push((u, c as f32 * per_draw));
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

/// Threads a step is split over: the machine's parallelism **on the CPU**,
/// and one everywhere else.
///
/// Candle SGD/backprop still runs on `cfg.device` either way — CUDA/Metal
/// kernels stay on that device. Slicing only pays where slices land on
/// different *host* arithmetic units. A GPU issues kernels on one stream, so
/// multi-slice launches serialize and add contention (measured ~10× slower on
/// CUDA at 16 slices); the CPU's elementwise/reduction ops are single-threaded
/// and do want the split.
fn step_threads(dev: &Device) -> usize {
    if dev.is_cpu() {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        1
    }
}

/// Fewest units a thread's slice of a step should hold.
const MIN_UNITS_PER_SLICE: usize = 4;

/// One step's loss and gradient over `chunk`, as up to `n_threads` slices of
/// disjoint units solved on their own threads: each slice draws its own
/// modules from an rng seeded off the step's rng (seeds taken in slice
/// order), builds its loss with `loss_of`, runs its own backward, and the
/// slices' gradients are summed. Exact, since the loss is a sum over units.
pub(crate) fn step_grads<F>(
    chunk: &[u32],
    n_threads: usize,
    rng: &mut StdRng,
    loss_of: &F,
) -> anyhow::Result<(StepStats, GradStore)>
where
    F: Fn(&[u32], &mut StdRng) -> anyhow::Result<(StepStats, Tensor)> + Sync,
{
    let n_slices = (chunk.len() / MIN_UNITS_PER_SLICE).clamp(1, n_threads.max(1));
    let per_slice = chunk.len().div_ceil(n_slices).max(1);
    let slices: Vec<(&[u32], u64)> = chunk
        .chunks(per_slice)
        .map(|s| (s, rng.next_u64()))
        .collect();
    let results: Vec<anyhow::Result<(StepStats, GradStore)>> = if slices.len() == 1 {
        let (slice, seed) = slices[0];
        vec![slice_grads(slice, seed, loss_of)]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = slices
                .iter()
                .map(|&(slice, seed)| scope.spawn(move || slice_grads(slice, seed, loss_of)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a step slice panicked"))
                .collect()
        })
    };
    let mut stats = StepStats::default();
    let mut grads: Option<GradStore> = None;
    for r in results {
        let (s, g) = r?;
        stats.loss_module += s.loss_module;
        stats.loss_gene += s.loss_gene;
        stats.loss_ridge += s.loss_ridge;
        grads = Some(match grads {
            None => g,
            Some(mut acc) => {
                merge_grads(&mut acc, &g)?;
                acc
            }
        });
    }
    Ok((stats, grads.expect("at least one slice")))
}

fn slice_grads<F>(slice: &[u32], seed: u64, loss_of: &F) -> anyhow::Result<(StepStats, GradStore)>
where
    F: Fn(&[u32], &mut StdRng) -> anyhow::Result<(StepStats, Tensor)>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let (stats, loss) = loss_of(slice, &mut rng)?;
    Ok((stats, loss.backward()?))
}

/// `into += from`, id by id; an id only `from` has is copied over.
fn merge_grads(into: &mut GradStore, from: &GradStore) -> anyhow::Result<()> {
    for &id in from.get_ids() {
        let g = from.get_id(id).expect("listed id");
        let sum = match into.get_id(id) {
            Some(a) => (a + g)?,
            None => g.clone(),
        };
        into.insert_id(id, sum);
    }
    Ok(())
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
    if let Some(coupling) = &cfg.cis_gates {
        coupling.validate(n_features)?;
        anyhow::ensure!(
            n_t == 1,
            "cis gates need a one-track (multiome) feature axis"
        );
        let mut resolved = coupling.pairs.with_peak_modules(&part.module_of);
        if let Some(mo) = &module_only {
            let mut is_mo = vec![false; n_features];
            for &g in &mo.genes {
                is_mo[g as usize] = true;
            }
            let n_all = resolved.n_pairs();
            resolved = resolved.without_genes(&is_mo);
            if resolved.n_pairs() < n_all {
                info!(
                    "Phase 1 (hier) — {} cis pair(s) of module-only genes dropped",
                    n_all - resolved.n_pairs()
                );
            }
        }
        if cfg.background_modules.iter().any(|&b| b) {
            let n_all = resolved.n_pairs();
            resolved = resolved.without_peak_modules(&cfg.background_modules);
            info!(
                "Phase 1 (hier) — {} cis pair(s) to near-empty or scattered peaks dropped",
                n_all - resolved.n_pairs()
            );
        }
        params.cis = Some(
            super::cis_gates::CisGateParams::new(&resolved, &part.module_of, &cfg.device)?
                .with_mix(coupling.mix, n_features, &cfg.device)?,
        );
        info!(
            "Phase 1 (hier) — cis gates on {} pairs (shared θ); alignment weight {}, \
             mixture share {}",
            resolved.n_pairs(),
            coupling.align_weight,
            coupling.mix
        );
    }
    let align_weight = cfg.cis_gates.as_ref().map_or(0.0, |c| c.align_weight);
    let skip: Vec<bool> = module_only
        .as_ref()
        .map_or_else(Vec::new, |mo| mo.skip.clone());
    if !cfg.module_group.is_empty() {
        anyhow::ensure!(
            cfg.module_group.len() == n_m,
            "module groups for {} modules, {n_m} in the partition",
            cfg.module_group.len()
        );
        params.group = GroupIntercepts::new(n_u, &cfg.module_group, &cfg.device)?;
        if let Some(gi) = &params.group {
            anyhow::ensure!(
                preset.is_none() && preset_offsets.is_empty(),
                "per-unit group intercepts centre μ within each group, which a \
                 preset's given rows would not survive"
            );
            gi.centre(&params.mu)?;
            info!(
                "Phase 1 (hier) — per-unit intercepts on {} module group(s) (group 0 the \
                 reference): each unit's split of counts across groups is its own",
                gi.n_groups
            );
        }
    }
    let mut opt = Optimizers::new(&params, cfg.lr)?;
    let ctx = StepCtx {
        units,
        um: &um,
        part: &part,
        sup: &sup,
        skip_module: &skip,
    };
    let mut rng = StdRng::seed_from_u64(mix_seed(cfg.seed, 0x4849_4552));
    let pickers = module_pickers(&um, n_m, &skip);
    let mut order: Vec<u32> = (0..n_u as u32).collect();
    let steps_per_epoch = n_u.div_ceil(cfg.units_per_step.max(1));
    let n_threads = step_threads(&cfg.device);
    let offset_l2_step = per_step_offset_l2(cfg.offset_l2, steps_per_epoch);
    let lora_ridge_step = params
        .lora
        .as_ref()
        .map_or(0.0, |l| per_step_offset_l2(l.ridge, steps_per_epoch));
    info!(
        "Phase 1 (hier) — {n_u} units × {d} genes on {n_t} track(s) ({n_features} feature rows) \
         in {n_m} modules, H={h}: {} epochs × {steps_per_epoch} steps of {} units, K={} \
         modules/unit, lr {}, device {}, {n_threads} host slice(s) per step{}",
        cfg.epochs,
        cfg.units_per_step,
        cfg.modules_per_unit,
        cfg.lr,
        if cfg.device.is_cuda() {
            "cuda"
        } else if cfg.device.is_metal() {
            "metal"
        } else {
            "cpu"
        },
        if cfg.device.is_cpu() {
            ""
        } else {
            " (candle SGD on device; no host split)"
        },
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
            // The cis pool once per step, read by the mixture and the
            // alignment. The slices share the mixture as leaves and its
            // gradient goes back through the pool once.
            let cis_pool = cis_pool(&params)?;
            let cis_graph = match cis_pool.as_ref() {
                Some(pool) => cis_mix(&params, pool)?,
                None => None,
            };
            let cis_leaves = cis_graph.as_ref().map(CisMix::detached).transpose()?;
            let loss_of = |slice: &[u32], rng: &mut StdRng| {
                let plan = draw_plan(slice, &pickers, n_m, n_t, cfg.modules_per_unit, rng);
                step_loss(
                    &params,
                    &ctx,
                    &plan,
                    offset_l2_step,
                    lora_ridge_step,
                    cis_leaves.as_ref().map(|l| &l.mix),
                )
            };
            let (mut stats, mut grads) = step_grads(chunk, n_threads, &mut rng, &loss_of)?;
            if let (Some(graph), Some(leaves)) = (&cis_graph, &cis_leaves) {
                if let Some(through) = leaves.backprop(graph, &grads)? {
                    merge_grads(&mut grads, &through)?;
                }
            }
            // The cis alignment spans the step's units: once per step.
            if let (Some(pool), true) = (cis_pool.as_ref(), align_weight > 0.0) {
                if let Some(loss) = cis_align_loss(&params, pool, chunk, align_weight)? {
                    stats.loss_align = f64::from(loss.to_scalar::<f32>()?);
                    merge_grads(&mut grads, &loss.backward()?)?;
                }
            }
            apply(
                &mut params,
                &mut opt,
                &grads,
                cfg.lr,
                cfg.weight_decay,
                cfg.unit_weight_decay,
            )?;
            acc.loss_module += stats.loss_module;
            acc.loss_gene += stats.loss_gene;
            acc.loss_ridge += stats.loss_ridge;
            acc.loss_align += stats.loss_align;
            n_units_seen += chunk.len();
        }
        let per_unit = 1.0 / n_units_seen.max(1) as f64;
        last_per_unit =
            (acc.loss_module + acc.loss_gene + acc.loss_ridge + acc.loss_align) * per_unit;
        bar.inc(1);
        // Every epoch, at info: visible under `-v`, silent otherwise.
        let elapsed = t0.elapsed().as_secs_f64();
        let ms = elapsed * 1e3 / ((epoch + 1) * steps_per_epoch) as f64;
        let eta = elapsed / (epoch + 1) as f64 * (cfg.epochs - epoch - 1) as f64;
        info!(
            "Phase 1 (hier) — epoch {}/{}: loss/unit {:.4} (module {:.4}, gene {:.4}, \
             ridge {:.4}, align {:.4}), {:.1} ms/step, eta {:.0} s",
            epoch + 1,
            cfg.epochs,
            last_per_unit,
            acc.loss_module * per_unit,
            acc.loss_gene * per_unit,
            acc.loss_ridge * per_unit,
            acc.loss_align * per_unit,
            ms,
            eta
        );
    }
    bar.finish_and_clear();

    // The composed dictionary, one row per FEATURE ROW (see `HierParams::compose`),
    // under a cis mixture with each cis gene's row as the gene level scored it,
    // so phase 2 projects against the fitted scores.
    let (mut rho, mut b_feat) = params.compose(
        &units.tracks.track_of_row,
        &units.tracks.gene_of_row,
        &part.module_of,
    )?;
    if let Some(blend) = cis_dictionary_blend(&params)? {
        blend.apply(&mut rho, &mut b_feat);
    }
    let e_u_host = to_host(params.e_u.as_tensor())?;
    let e_u = DMatrix::<f32>::from_row_slice(n_u, h, &e_u_host);
    let cis = cis_readout(&params)?;
    let group_intercepts = match params.group.as_ref() {
        Some(gi) => Some(DMatrix::<f32>::from_row_slice(
            n_u,
            gi.n_groups - 1,
            &to_host(gi.beta.as_tensor())?,
        )),
        None => None,
    };
    Ok(HierOutput {
        e_u,
        rho,
        b_feat,
        final_loss_per_unit: last_per_unit,
        labels: labels.to_vec(),
        cis,
        group_intercepts,
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
