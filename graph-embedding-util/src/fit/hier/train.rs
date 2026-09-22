//! Epoch loop: every unit once per epoch in a seeded order, K modules drawn
//! per unit ∝ its share, one [`step`] per chunk of units; the composed
//! dictionary at the end.

use super::module_recollapse::{merge_module_rows, recollapse_modules};
use super::params::{HierParams, PresetGenes, PresetMode, PresetOffsets};
use super::partition::{Partition, TrackSupport, UnitModules};
use super::step::{apply, step_loss, step_loss_extra, Optimizers, StepCtx, StepPlan, StepStats};
use super::units::UnitTable;
use crate::fit::config::TrackSpec;
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

const REPORT_EVERY: usize = 50;

/// One feature axis's partition and everything derived from it: the units'
/// view through the partition, each track's support, and the module pickers.
/// Rebuilt as a whole when the partition re-collapses.
struct AxisState {
    axis: usize,
    part: Partition,
    tracks: TrackSpec,
    um: UnitModules,
    sup: TrackSupport,
    pickers: Vec<Option<WeightedIndex<f64>>>,
}

impl AxisState {
    /// Axis 0 carries the unit table's [`TrackSpec`]; every other axis is a
    /// single unrestricted track over its own feature ids.
    fn new(units: &UnitTable, axis: usize, part: Partition) -> Self {
        let tracks = if axis == 0 {
            units.tracks.clone()
        } else {
            TrackSpec::base(units.axes[axis].n_features)
        };
        let um = UnitModules::from_axis(units, axis, &part);
        let sup = TrackSupport::new(&tracks, &part);
        let pickers = module_pickers(&um, part.n_modules());
        Self {
            axis,
            part,
            tracks,
            um,
            sup,
            pickers,
        }
    }

    fn n_modules(&self) -> usize {
        self.part.n_modules()
    }

    fn ctx<'a>(&'a self, units: &'a UnitTable) -> StepCtx<'a> {
        StepCtx {
            units,
            um: &self.um,
            part: &self.part,
            sup: &self.sup,
            axis: self.axis,
        }
    }

    /// Merge-only re-collapse from the frozen unit profiles: the partition,
    /// its derived views, this axis's `μ` / bias rows and their optimizer
    /// state all move together. A no-op when nothing merges.
    fn recollapse(
        &mut self,
        cfg: &HierConfig,
        units: &UnitTable,
        params: &mut HierParams,
        opt: &mut Optimizers,
    ) -> anyhow::Result<()> {
        let axis = self.axis;
        let Some((new_part, map)) = recollapse_modules(units, axis, &self.part, cfg.merge_cosine)
        else {
            return Ok(());
        };
        let from_m = self.n_modules();
        if axis == 0 {
            merge_module_rows(&mut params.mu, &mut params.b_m, &map, &params.dev)?;
            for o in params.offsets.iter_mut() {
                merge_module_rows(&mut o.d_mu, &mut o.d_b_m, &map, &params.dev)?;
            }
            opt.reset_base_mu(map.n_modules, cfg.lr, &params.dev)?;
            opt.reset_offset_mu(map.n_modules, cfg.lr, &params.dev)?;
        } else {
            let extra = &mut params.extra[axis - 1];
            merge_module_rows(&mut extra.mu, &mut extra.b_m, &map, &params.dev)?;
            opt.reset_extra_mu(axis - 1, map.n_modules, cfg.lr, &params.dev)?;
        }
        self.part = new_part;
        self.um = UnitModules::from_axis(units, axis, &self.part);
        self.sup = TrackSupport::new(&self.tracks, &self.part);
        self.pickers = module_pickers(&self.um, self.n_modules());
        info!(
            "Phase 1 (hier): axis {axis} unit-profile re-collapse, M {from_m} → {}",
            self.n_modules()
        );
        Ok(())
    }
}

/// Every feature's count summed over the units of one axis.
fn feature_totals(axis: &crate::fit::hier::units::FeatureAxis) -> Vec<f32> {
    let mut total = vec![0f32; axis.n_features];
    for (feats, counts) in axis.feats.iter().zip(&axis.counts) {
        for (&f, &c) in feats.iter().zip(counts) {
            total[f as usize] += c;
        }
    }
    total
}

#[derive(Clone)]
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
    /// Re-collapse modules every this many epochs; `0` disables.
    pub merge_every: usize,
    /// Cosine threshold on frozen pb feature profiles (whole-module merge-only).
    pub merge_cosine: f32,
    /// Partitions (indices `1..`) whose features carry no residual: a feature's
    /// row is its module's row and its bias the closed-form within-module
    /// share. Ignored by the single-partition path.
    pub module_only: Vec<usize>,
}

/// Per feature partition: composed feature rows, biases, and module table.
pub struct HierAxisOut {
    pub rho: DMatrix<f32>,
    pub b_feat: Vec<f32>,
    pub mu: DMatrix<f32>,
}

pub struct HierOutput {
    pub e_u: DMatrix<f32>,
    /// `[n_features × H]`, one row per FEATURE ROW on axis 0: the composed
    /// dictionary of that row's `(track, gene)`. Mirrors `axes[0].rho`.
    pub rho: DMatrix<f32>,
    /// `[n_features]`, likewise per feature row on axis 0. Mirrors `axes[0].b_feat`.
    pub b_feat: Vec<f32>,
    /// One entry per feature partition (gene-only: length 1).
    pub axes: Vec<HierAxisOut>,
    /// Mean loss per unit over the last completed epoch; `NaN` when training
    /// stopped before any epoch completed (the tables are still finite).
    pub final_loss_per_unit: f64,
    /// `⌈n_units / units_per_step⌉` for this fit.
    pub steps_per_epoch: usize,
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

/// Threads a step is split over: the machine's parallelism.
fn step_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
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
    // A preset pins `μ` (or hangs a residual on it); a merge rewrites `μ`.
    anyhow::ensure!(
        cfg.merge_every == 0 || (preset.is_none() && preset_offsets.is_empty()),
        "module re-collapse (merge_every > 0) cannot be combined with preset rows or offsets"
    );
    // The partition and each track's support through it: built once here,
    // never per step (only a re-collapse rebuilds them). The support is inert
    // on a one-track axis, where the base track covers every gene.
    let mut ax = AxisState::new(units, 0, Partition::from_labels(labels, cfg.n_modules));
    let (n_u, n_m, d) = (units.n_units(), ax.n_modules(), units.tracks.n_genes());
    let n_t = units.n_tracks();
    let n_features = units.n_features();
    let mut params =
        HierParams::new_tracked(n_u, n_m, d, n_t, h, cfg.offset_rank, cfg.seed, &cfg.device)?;
    if let Some(f) = preset {
        params.preset(f, &ax.part.module_of)?;
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
    let mut opt = Optimizers::new(&params, cfg.lr)?;
    let mut rng = StdRng::seed_from_u64(mix_seed(cfg.seed, 0x4849_4552));
    let mut order: Vec<u32> = (0..n_u as u32).collect();
    let steps_per_epoch = n_u.div_ceil(cfg.units_per_step.max(1));
    let n_threads = step_threads();
    let offset_l2_step = per_step_offset_l2(cfg.offset_l2, steps_per_epoch);
    let lora_ridge_step = params
        .lora
        .as_ref()
        .map_or(0.0, |l| per_step_offset_l2(l.ridge, steps_per_epoch));
    info!(
        "Phase 1 (hier) — {n_u} units × {d} genes on {n_t} track(s) ({n_features} feature rows) \
         in {n_m} modules, H={h}: {} epochs × {steps_per_epoch} steps of {} units, K={} \
         modules/unit, lr {}, {n_threads} thread(s) per step",
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
            let loss_of = |slice: &[u32], rng: &mut StdRng| {
                let plan = draw_plan(
                    slice,
                    &ax.pickers,
                    ax.n_modules(),
                    n_t,
                    cfg.modules_per_unit,
                    rng,
                );
                step_loss(
                    &params,
                    &ax.ctx(units),
                    &plan,
                    offset_l2_step,
                    lora_ridge_step,
                )
            };
            let (stats, grads) = step_grads(chunk, n_threads, &mut rng, &loss_of)?;
            apply(&mut params, &mut opt, &grads, cfg.lr, cfg.weight_decay)?;
            acc.loss_module += stats.loss_module;
            acc.loss_gene += stats.loss_gene;
            acc.loss_ridge += stats.loss_ridge;
            n_units_seen += chunk.len();
        }
        let per_unit = 1.0 / n_units_seen.max(1) as f64;
        last_per_unit = (acc.loss_module + acc.loss_gene + acc.loss_ridge) * per_unit;
        bar.inc(1);
        if cfg.merge_every > 0 && (epoch + 1).is_multiple_of(cfg.merge_every) {
            ax.recollapse(cfg, units, &mut params, &mut opt)?;
        }
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

    // The composed dictionary, one row per FEATURE ROW (see `HierParams::compose`).
    let (rho, b_feat) = params.compose(
        &units.tracks.track_of_row,
        &units.tracks.gene_of_row,
        &ax.part.module_of,
    )?;
    let e_u_host = to_host(params.e_u.as_tensor())?;
    let e_u = DMatrix::<f32>::from_row_slice(n_u, h, &e_u_host);
    let mu = params.mu_host()?;
    Ok(HierOutput {
        e_u,
        rho: rho.clone(),
        b_feat: b_feat.clone(),
        axes: vec![HierAxisOut { rho, b_feat, mu }],
        final_loss_per_unit: last_per_unit,
        steps_per_epoch,
    })
}

/// Hierarchical train over one or more feature partitions.
///
/// Shared unit embedding `e_u`; each partition has its own `μ` / feature `r`.
/// Gene-only (`partitions.len() == 1`, `units.n_axes() == 1`) matches [`train`].
/// Multiome loops L₁ + K-sampled L₂ per partition each step.
pub fn train_partitions(
    units: &UnitTable,
    partitions: &[Partition],
    h: usize,
    cfg: &HierConfig,
    preset: Option<&PresetGenes>,
    preset_offsets: &[PresetOffsets],
    stop: &AtomicBool,
) -> anyhow::Result<HierOutput> {
    anyhow::ensure!(!partitions.is_empty(), "at least one feature partition");
    anyhow::ensure!(
        partitions.len() == units.n_axes(),
        "one partition per feature axis (got {} partitions for {} axes)",
        partitions.len(),
        units.n_axes()
    );
    anyhow::ensure!(
        partitions[0].module_of.len() == units.tracks.n_genes(),
        "axis-0 partition labels must cover every gene"
    );
    for (a, part) in partitions.iter().enumerate().skip(1) {
        anyhow::ensure!(
            part.module_of.len() == units.axes[a].n_features,
            "partition {a} labels must cover every feature on that axis"
        );
    }
    if partitions.len() == 1 {
        // The given partition's module count wins over `cfg.n_modules`.
        let cfg = HierConfig {
            n_modules: partitions[0].n_modules(),
            ..cfg.clone()
        };
        return train(
            units,
            &partitions[0].module_of,
            h,
            &cfg,
            preset,
            preset_offsets,
            stop,
        );
    }

    // Multi-partition path: shared e_u, per-axis tables, no gene presets on extras.
    anyhow::ensure!(
        preset.is_none() && preset_offsets.is_empty(),
        "presets / track offsets apply to the gene TrackSpec axis only; \
         use train() for a single partition with presets"
    );
    if units.n_tracks() > 1 {
        crate::fit::config::validate_offset_rank(cfg.offset_rank, h)?;
    }

    let n_u = units.n_units();
    let n_t0 = units.n_tracks();
    let n_m0 = partitions[0].n_modules();
    let d0 = units.tracks.n_genes();

    let mut axis_states: Vec<AxisState> = partitions
        .iter()
        .enumerate()
        .map(|(a, p)| AxisState::new(units, a, p.clone()))
        .collect();

    let mut params = HierParams::new_tracked(
        n_u,
        n_m0,
        d0,
        n_t0,
        h,
        cfg.offset_rank,
        cfg.seed,
        &cfg.device,
    )?;
    for &a in &cfg.module_only {
        anyhow::ensure!(
            (1..partitions.len()).contains(&a),
            "module-only partition {a}: only partitions 1..{} can drop their residual",
            partitions.len()
        );
    }
    for (a, part) in partitions.iter().enumerate().skip(1) {
        // Distinct init salt per extra axis so peaks ≠ genes.
        let module_only = cfg
            .module_only
            .contains(&a)
            .then(|| feature_totals(&units.axes[a]));
        params.push_extra_axis(
            part.n_modules(),
            units.axes[a].n_features,
            0x4158_4953 + a as u64,
            module_only,
        )?;
    }

    let mut opt = Optimizers::new(&params, cfg.lr)?;
    let mut rng = StdRng::seed_from_u64(mix_seed(cfg.seed, 0x4849_4552));
    let mut order: Vec<u32> = (0..n_u as u32).collect();
    let steps_per_epoch = n_u.div_ceil(cfg.units_per_step.max(1));
    let n_threads = step_threads();
    let offset_l2_step = per_step_offset_l2(cfg.offset_l2, steps_per_epoch);

    let m_list: Vec<String> = axis_states
        .iter()
        .map(|ax| ax.n_modules().to_string())
        .collect();
    let f_list: Vec<String> = units
        .axes
        .iter()
        .map(|ax| ax.n_features.to_string())
        .collect();
    info!(
        "Phase 1 (hier) — {n_u} units; partitions M=[{}] F=[{}]; H={h}: {} epochs × \
         {steps_per_epoch} steps of {} units, K={} modules/unit, lr {}, {n_threads} thread(s) \
         per step",
        m_list.join(","),
        f_list.join(","),
        cfg.epochs,
        cfg.units_per_step,
        cfg.modules_per_unit,
        cfg.lr
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
            let loss_of = |slice: &[u32], rng: &mut StdRng| {
                let mut stats = StepStats::default();
                let mut loss_total = None;
                for (a, ax) in axis_states.iter().enumerate() {
                    let plan = draw_plan(
                        slice,
                        &ax.pickers,
                        ax.n_modules(),
                        ax.um.n_tracks,
                        cfg.modules_per_unit,
                        rng,
                    );
                    let ctx = ax.ctx(units);
                    let (s, loss) = if a == 0 {
                        step_loss(&params, &ctx, &plan, offset_l2_step, 0.0)?
                    } else {
                        step_loss_extra(&params, &params.extra[a - 1], &ctx, &plan)?
                    };
                    stats.loss_module += s.loss_module;
                    stats.loss_gene += s.loss_gene;
                    stats.loss_ridge += s.loss_ridge;
                    legume_numeric::candle::convert::add_into(&mut loss_total, loss)?;
                }
                Ok((stats, loss_total.expect("at least one partition")))
            };
            let (stats, grads) = step_grads(chunk, n_threads, &mut rng, &loss_of)?;
            acc.loss_module += stats.loss_module;
            acc.loss_gene += stats.loss_gene;
            acc.loss_ridge += stats.loss_ridge;
            apply(&mut params, &mut opt, &grads, cfg.lr, cfg.weight_decay)?;
            n_units_seen += chunk.len();
        }
        let per_unit = 1.0 / n_units_seen.max(1) as f64;
        last_per_unit = (acc.loss_module + acc.loss_gene + acc.loss_ridge) * per_unit;
        bar.inc(1);
        if cfg.merge_every > 0 && (epoch + 1).is_multiple_of(cfg.merge_every) {
            for ax in &mut axis_states {
                ax.recollapse(cfg, units, &mut params, &mut opt)?;
            }
        }
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

    let (rho0, b0) = params.compose(
        &units.tracks.track_of_row,
        &units.tracks.gene_of_row,
        &axis_states[0].part.module_of,
    )?;
    let mut axes = vec![HierAxisOut {
        rho: rho0.clone(),
        b_feat: b0.clone(),
        mu: params.mu_host()?,
    }];
    for (i, ax) in params.extra.iter().enumerate() {
        let (rho, b_feat) = ax.compose(&axis_states[i + 1].part.module_of)?;
        axes.push(HierAxisOut {
            rho,
            b_feat,
            mu: ax.mu_host()?,
        });
    }
    let e_u_host = to_host(params.e_u.as_tensor())?;
    let e_u = DMatrix::<f32>::from_row_slice(n_u, h, &e_u_host);
    Ok(HierOutput {
        e_u,
        rho: rho0,
        b_feat: b0,
        axes,
        final_loss_per_unit: last_per_unit,
        steps_per_epoch,
    })
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod train_tests;
