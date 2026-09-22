//! Cold per-AXIS projection: one Poisson partition and one intercept per
//! feature axis (genes, peaks, ...) against one shared latent, solved from the
//! null model. The multi-modality counterpart of [`super::tracks`], which is
//! the same objective on tracks of one gene axis with a warm start.
//!
//! ```text
//! S^a    = Θ · E^aᵀ + c_a + β^a                          [Bc × F_a]
//! loss   = Σ_a [ Σ_f exp(S^a_f) − Σ_f n^a_f · S^a_f ] + (λ/2) ‖Θ‖²
//! ```
//!
//! The intercepts ride as extra columns of `Θ̃ = [Θ | c_0 … c_{A−1}]` against
//! per-axis designs whose row `H + a` is ones; one Adam state. An axis a cell
//! has no counts on is masked out of both gradients and its intercept stays at
//! the score-clamp floor. See the module docs of [`super`] for why the
//! gradient is closed-form and why `N` is dense per block.

use super::edges::{block_cells, EdgeTable};
use super::pass::{adam_step_size, poisson_deviance, BlockProgress, PassStats};
use super::{BETA1, BETA2, CHECK_EVERY, EPS, GATE_FOLD_EPS, MAX_STEPS, TARGET_DELTA_S, TOL};
use crate::cell_projection::SCORE_CLAMP;
use legume_numeric::candle::candle_core::{DType, Device, Tensor};
use legume_numeric::matrix::traits::FusedTensorOps;
use log::info;

/// One frozen feature axis: dictionary `[F × H]` row-major and bias `[F]`.
pub struct AxisDict<'a> {
    pub label: &'a str,
    pub feat: &'a [f32],
    pub b_feat: &'a [f32],
}

/// One group of cells' sparse counts on every axis. `axes[a][i]` is
/// `(feature ids ascending, counts > 0)` of `cells[i]` on axis `a`; a cell with
/// nothing on an axis has an empty pair there.
pub struct CellGroup {
    pub cells: Vec<u32>,
    pub axes: Vec<Vec<(Vec<u32>, Vec<f32>)>>,
}

/// One group's result, indexed by position in [`CellGroup::cells`]; not gauged.
pub struct GroupOut {
    /// `[n × H]` row-major.
    pub theta: Vec<f32>,
    /// `[A][n]`.
    pub intercepts: Vec<Vec<f32>>,
}

/// One axis's frozen design, built once per projector.
struct AxisDesign {
    label: String,
    n_features: usize,
    /// Every feature id of the axis, `0..F`: the partition an [`EdgeTable`]
    /// is flattened against.
    rows: Vec<u32>,
    /// Feature id → live-local id; `u32::MAX` for a gate-folded feature.
    to_live: Vec<u32>,
    f_live: usize,
    /// `Ẽᵀ [H + A, F_live]` (ones on this axis's intercept row) and `Ẽ`.
    e_aug: Tensor,
    e_aug_t: Tensor,
    /// `β [1, F_live]`.
    b_row: Tensor,
    /// `[1, H + A]` with 1.0 in this axis's intercept slot.
    intercept_mask: Tensor,
    /// `Σ_dead exp(β_f)`; 0 at the default [`GATE_FOLD_EPS`].
    dead_mass: f64,
    /// `ln(Σ_live exp(β_f) + dead_mass)`: the null partition at `Θ = 0`, so
    /// the cold intercept init is one log per cell.
    null_log_norm: f64,
}

fn build_axis(
    dict: &AxisDict,
    axis: usize,
    n_axes: usize,
    h: usize,
    dev: &Device,
) -> anyhow::Result<AxisDesign> {
    let n_features = dict.b_feat.len();
    anyhow::ensure!(
        dict.feat.len() == n_features * h,
        "axis {}: the dictionary has {} entries, expected {n_features} × {h}",
        dict.label,
        dict.feat.len()
    );
    let mut live: Vec<u32> = Vec::with_capacity(n_features);
    let mut dead_mass = 0f64;
    let mut live_mass = 0f64;
    for f in 0..n_features {
        let e = &dict.feat[f * h..(f + 1) * h];
        if e.iter().map(|x| x * x).sum::<f32>().sqrt() > GATE_FOLD_EPS {
            live.push(f as u32);
            live_mass += f64::from(dict.b_feat[f]).exp();
        } else {
            dead_mass += f64::from(dict.b_feat[f]).exp();
        }
    }
    anyhow::ensure!(
        !live.is_empty(),
        "axis {}: every feature is gate-folded, the frozen dictionary carries no signal",
        dict.label
    );
    let f_live = live.len();
    let d = h + n_axes;
    let mut e_aug = vec![0f32; d * f_live];
    let mut b_live = vec![0f32; f_live];
    let mut to_live = vec![u32::MAX; n_features];
    for (l, &f) in live.iter().enumerate() {
        let row = &dict.feat[f as usize * h..(f as usize + 1) * h];
        for (k, &v) in row.iter().enumerate() {
            e_aug[k * f_live + l] = v;
        }
        e_aug[(h + axis) * f_live + l] = 1.0;
        b_live[l] = dict.b_feat[f as usize];
        to_live[f as usize] = l as u32;
    }
    let e_aug = Tensor::from_vec(e_aug, (d, f_live), dev)?;
    let e_aug_t = e_aug.t()?.contiguous()?;
    let b_row = Tensor::from_vec(b_live, (1, f_live), dev)?;
    let intercept_mask = {
        let mut v = vec![0f32; d];
        v[h + axis] = 1.0;
        Tensor::from_vec(v, (1, d), dev)?
    };
    Ok(AxisDesign {
        label: dict.label.to_string(),
        n_features,
        rows: (0..n_features as u32).collect(),
        to_live,
        f_live,
        e_aug,
        e_aug_t,
        b_row,
        intercept_mask,
        dead_mass,
        null_log_norm: (live_mass + dead_mass).max(f64::MIN_POSITIVE).ln(),
    })
}

/// The frozen per-axis designs and everything derived from them alone: block
/// size, learning rate. Built once; every group is solved against it.
pub struct AxesProjector {
    axes: Vec<AxisDesign>,
    h: usize,
    lambda: f64,
    dev: Device,
    block_cells: usize,
    lr0: f64,
}

impl AxesProjector {
    pub fn new(dicts: &[AxisDict], h: usize, lambda: f64, dev: &Device) -> anyhow::Result<Self> {
        anyhow::ensure!(!dicts.is_empty(), "at least one feature axis");
        let n_axes = dicts.len();
        let axes: Vec<AxisDesign> = dicts
            .iter()
            .enumerate()
            .map(|(a, d)| build_axis(d, a, n_axes, h, dev))
            .collect::<anyhow::Result<_>>()?;
        // Every axis's `[Bc, F_a]` counts are resident at once: size the block
        // from the whole feature axis.
        let n_rows: usize = axes.iter().map(|a| a.n_features).sum();
        let block_cells = block_cells(n_rows);
        // One rate for one shared Θ, calibrated on every axis's live rows (the
        // arithmetic mean |e|, as the other passes do).
        let f_live_total: usize = axes.iter().map(|a| a.f_live).sum();
        let sum_abs: f64 = dicts
            .iter()
            .zip(&axes)
            .flat_map(|(d, ax)| {
                ax.to_live
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l != u32::MAX)
                    .flat_map(move |(f, _)| &d.feat[f * h..(f + 1) * h])
            })
            .map(|x| f64::from(*x).abs())
            .sum();
        let e_mean = (sum_abs / (f_live_total * h) as f64).max(1e-12);
        let lr0 = TARGET_DELTA_S / (h as f64 * e_mean);
        info!(
            "Projection [axes]: {} axes, live features [{}] (of [{}]), blocks of {block_cells}, \
             lr {lr0:.4} (auto: Δs≈{TARGET_DELTA_S}), ≤{MAX_STEPS} steps, ridge λ={lambda}",
            n_axes,
            axes.iter()
                .map(|a| a.f_live.to_string())
                .collect::<Vec<_>>()
                .join(","),
            axes.iter()
                .map(|a| a.n_features.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        Ok(Self {
            axes,
            h,
            lambda,
            dev: dev.clone(),
            block_cells,
            lr0,
        })
    }

    pub fn h(&self) -> usize {
        self.h
    }

    pub fn n_axes(&self) -> usize {
        self.axes.len()
    }

    /// Cells one group should hold: a whole number of blocks.
    pub fn group_cells(&self) -> usize {
        self.block_cells * super::GROUP_BLOCKS
    }

    /// Solve one group. `bar` advances by one tick per cell as the blocks step.
    pub fn project_group(
        &self,
        group: &CellGroup,
        bar: &indicatif::ProgressBar,
    ) -> anyhow::Result<GroupOut> {
        let (h, n) = (self.h, group.cells.len());
        anyhow::ensure!(
            group.axes.len() == self.axes.len(),
            "group carries {} axes, projector has {}",
            group.axes.len(),
            self.axes.len()
        );
        for (a, per_cell) in group.axes.iter().enumerate() {
            anyhow::ensure!(
                per_cell.len() == n,
                "axis {}: {} cell rows for {n} cells",
                self.axes[a].label,
                per_cell.len()
            );
        }
        // One edge table per axis, flattened once per group.
        let edges: Vec<EdgeTable> = self
            .axes
            .iter()
            .zip(&group.axes)
            .map(|(ax, per_cell)| {
                let cells: Vec<(u32, &[u32], &[f32])> = per_cell
                    .iter()
                    .enumerate()
                    .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
                    .collect();
                EdgeTable::build(&cells, &ax.rows, ax.n_features, None)
            })
            .collect();

        let bc = self.block_cells;
        let mut theta = vec![0f32; n * h];
        let mut intercepts = vec![vec![-(SCORE_CLAMP as f32); n]; self.axes.len()];
        let mut stats = PassStats::default();
        let n_blocks = n.div_ceil(bc);
        for (b, start) in (0..n).step_by(bc).enumerate() {
            let end = (start + bc).min(n);
            let out = self.solve_block(
                &edges,
                start,
                end,
                &BlockProgress {
                    bar,
                    stats: &stats,
                    label: "axes",
                    block: b + 1,
                    n_blocks,
                    max_steps: MAX_STEPS,
                },
            )?;
            theta[start * h..end * h].copy_from_slice(&out.latent);
            for (a, c) in out.intercepts.iter().enumerate() {
                intercepts[a][start..end].copy_from_slice(c);
            }
            stats.fold(
                out.steps,
                out.converged,
                out.clamped,
                out.deviance,
                out.n_edges,
                out.loop_secs,
            );
        }
        Ok(GroupOut { theta, intercepts })
    }
}

/// One axis's counts for one block, densely, plus what the null intercept and
/// the presence mask need.
struct AxisBlock {
    n_t: Tensor,
    n_dead: Option<Tensor>,
    n_tot: Vec<f64>,
    mask: Option<Tensor>,
    n_edges: usize,
}

struct BlockOut {
    latent: Vec<f32>,
    intercepts: Vec<Vec<f32>>,
    steps: usize,
    converged: bool,
    clamped: bool,
    deviance: f64,
    n_edges: usize,
    loop_secs: f64,
}

impl AxesProjector {
    fn gather(
        &self,
        ax: &AxisDesign,
        edges: &EdgeTable,
        start: usize,
        end: usize,
    ) -> anyhow::Result<AxisBlock> {
        let bc = end - start;
        let f_live = ax.f_live;
        let mut n_dense = vec![0f32; bc * f_live];
        let mut n_tot = vec![0f64; bc];
        let mut n_dead = vec![0f32; bc];
        let mut n_edges = 0usize;
        for i in start..end {
            let local = i - start;
            let (feats, counts) = edges.cell_slice(i);
            for (&f, &n) in feats.iter().zip(counts) {
                // `rows` is the identity on an axis, so the pass-local id IS the feature id.
                let l = ax.to_live[f as usize];
                n_tot[local] += f64::from(n);
                if l == u32::MAX {
                    n_dead[local] += n;
                } else {
                    n_dense[local * f_live + l as usize] = n;
                    n_edges += 1;
                }
            }
        }
        let present: Vec<f32> = n_tot.iter().map(|&n| f32::from(n > 0.0)).collect();
        let mask = if present.iter().all(|&p| p == 1.0) {
            None
        } else {
            Some(Tensor::from_vec(present, (bc, 1), &self.dev)?)
        };
        Ok(AxisBlock {
            n_t: Tensor::from_vec(n_dense, (bc, f_live), &self.dev)?.detach(),
            n_dead: if ax.dead_mass > 0.0 {
                Some(Tensor::from_vec(n_dead, (bc, 1), &self.dev)?)
            } else {
                None
            },
            n_tot,
            mask,
            n_edges,
        })
    }

    /// The Adam loop for one block against every axis's design, from the null
    /// model: `Θ = 0`, `c_a = ln Σ_f n_f − null_log_norm_a` (the exact
    /// conditional MLE at `Θ = 0`), floor for an absent axis.
    fn solve_block(
        &self,
        edges: &[EdgeTable],
        start: usize,
        end: usize,
        progress: &BlockProgress<'_>,
    ) -> anyhow::Result<BlockOut> {
        let (h, dev) = (self.h, &self.dev);
        let bc = end - start;
        let n_axes = self.axes.len();
        let d = h + n_axes;

        let blocks: Vec<AxisBlock> = self
            .axes
            .iter()
            .zip(edges)
            .map(|(ax, e)| self.gather(ax, e, start, end))
            .collect::<anyhow::Result<_>>()?;
        let n_edges: usize = blocks.iter().map(|b| b.n_edges).sum();

        let mut theta = vec![0f32; bc * d];
        for (a, (ax, blk)) in self.axes.iter().zip(&blocks).enumerate() {
            for (i, &n) in blk.n_tot.iter().enumerate() {
                theta[i * d + h + a] = if n > 0.0 {
                    (n.ln() - ax.null_log_norm).clamp(-SCORE_CLAMP, SCORE_CLAMP) as f32
                } else {
                    -SCORE_CLAMP as f32
                };
            }
        }
        let mut theta = Tensor::from_vec(theta, (bc, d), dev)?;

        // The data term is linear in the parameters: `−Σ_a N^a·Ẽ^a`, once.
        let mut ne = Tensor::zeros((bc, d), DType::F32, dev)?;
        for (ax, blk) in self.axes.iter().zip(&blocks) {
            ne = (ne + blk.n_t.matmul(&ax.e_aug_t)?)?;
            if let Some(n_dead) = &blk.n_dead {
                ne = (ne + n_dead.broadcast_mul(&ax.intercept_mask)?)?;
            }
        }
        let lam_row = {
            let mut v = vec![self.lambda as f32; d];
            for x in v[h..].iter_mut() {
                *x = 0.0;
            }
            Tensor::from_vec(v, (1, d), dev)?
        };

        let mut m = Tensor::zeros((bc, d), DType::F32, dev)?;
        let mut v = Tensor::zeros((bc, d), DType::F32, dev)?;
        let mut prev = theta.narrow(1, 0, h)?.contiguous()?;
        let mut steps = 0usize;
        let mut converged = false;
        let mut emitted = 0usize;
        let loop_start = std::time::Instant::now();
        for step in 0..MAX_STEPS {
            let mut g = theta.broadcast_mul(&lam_row)?;
            for (a, (ax, blk)) in self.axes.iter().zip(&blocks).enumerate() {
                let mu = theta
                    .matmul(&ax.e_aug)?
                    .clamped_exp_add_inplace(&ax.b_row, SCORE_CLAMP)?;
                let mu = match &blk.mask {
                    Some(mk) => mu.broadcast_mul(mk)?,
                    None => mu,
                };
                g = (g + mu.matmul(&ax.e_aug_t)?)?;
                if ax.dead_mass > 0.0 {
                    // A folded row's rate is `exp(β_f + c_a)`, so its partition
                    // mass lands on this axis's intercept column alone.
                    let dead = theta
                        .narrow(1, h + a, 1)?
                        .exp()?
                        .affine(ax.dead_mass, 0.0)?;
                    let dead = match &blk.mask {
                        Some(mk) => (dead * mk)?,
                        None => dead,
                    };
                    g = (g + dead.broadcast_mul(&ax.intercept_mask)?)?;
                }
            }
            let g = (g - &ne)?;
            m = ((&m * BETA1)? + (&g * (1.0 - BETA1))?)?;
            v = ((&v * BETA2)? + (g.sqr()? * (1.0 - BETA2))?)?;
            let step_size = adam_step_size(self.lr0, step, MAX_STEPS);
            theta = (&theta - (&m * step_size)?.broadcast_div(&(v.sqrt()? + EPS)?)?)?;
            steps = step + 1;
            if steps.is_multiple_of(CHECK_EVERY) {
                emitted = progress.advance(bc, steps, emitted);
                progress.describe(steps);
                let cur = theta.narrow(1, 0, h)?.contiguous()?;
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
        progress.finish_block(bc, emitted);

        let mut deviance = 0f64;
        let mut clamped = false;
        for (ax, blk) in self.axes.iter().zip(&blocks) {
            if blk.n_edges == 0 {
                continue;
            }
            let s = theta.matmul(&ax.e_aug)?.broadcast_add(&ax.b_row)?;
            clamped |= s.max_all()?.to_scalar::<f32>()? >= SCORE_CLAMP as f32;
            let s = s.clamp(-SCORE_CLAMP, SCORE_CLAMP)?;
            deviance += poisson_deviance(&blk.n_t, &s)?;
        }
        let mut intercepts = Vec::with_capacity(n_axes);
        for a in 0..n_axes {
            intercepts.push(theta.narrow(1, h + a, 1)?.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(BlockOut {
            latent: theta
                .narrow(1, 0, h)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?,
            intercepts,
            steps,
            converged,
            clamped,
            deviance,
            n_edges,
            loop_secs,
        })
    }
}

#[cfg(test)]
#[path = "axes_tests.rs"]
mod axes_tests;
