//! One optimizer step of the exact two-level softmax.
//!
//! ```text
//! p_um   = softmax_m ( ⟨e_u, μ_m⟩ + b_m )               over all M modules
//! p_ug|m = softmax_{g∈m} ( ⟨e_u, r_g⟩ + b_g )            over the genes of m
//! L₁(u)   = − Σ_m     q_um   · log p_um                  q_um   = n_um / N_u
//! L₂(u,m) = − Σ_{g∈m} q_ug|m · log p_ug|m                q_ug|m = n_ug / n_um
//! L = Σ_u w_u [ L₁(u) + (1/K) Σ_{k} L₂(u, m_k) ],        m_k ~ q_u·  (K draws)
//! ```
//!
//! With `δ¹ = p − q` at the module level and `δ² = p − q` within a module:
//!
//! ```text
//! ∂L/∂e_u = w_u [ Σ_m δ¹_um μ_m + (1/K) Σ_k Σ_{g∈m_k} δ²_ug r_g ]
//! ∂L/∂μ_m = Σ_u w_u δ¹_um e_u        ∂L/∂b_m = Σ_u w_u δ¹_um
//! ∂L/∂r_g = Σ_{u} (w_u/K) δ²_ug e_u  ∂L/∂b_g = Σ_{u} (w_u/K) δ²_ug     (u with m(g) drawn)
//! ```
//!
//! Positives are the unit's own shares; negatives are everything else in each
//! partition, in proportion to how far the prediction exceeds the share. No
//! negative is ever sampled — only the modules a unit is scored in at the gene
//! level, ∝ its share, which is an unbiased estimator of the full sum. The
//! within-module work is grouped by module: one gemm per module over the
//! units drawn into it, so a step touches a gene row at most once.

use super::params::{HierParams, RowAdagrad};
use super::partition::{Partition, UnitModules};
use super::units::UnitTable;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// The (unit, module) pairs one step evaluates at the gene level, grouped by module.
#[allow(dead_code)]
pub struct StepPlan {
    pub units: Vec<u32>,
    pub pairs_by_module: Vec<(u32, Vec<u32>)>,
}

#[allow(dead_code)]
#[derive(Default, Debug, Clone)]
pub struct StepStats {
    pub loss_module: f64,
    pub loss_gene: f64,
    pub n_units: usize,
    pub n_pairs: usize,
}

#[allow(dead_code)]
pub struct Optimizers {
    pub e_u: RowAdagrad,
    pub mu: RowAdagrad,
    pub r: RowAdagrad,
}

#[allow(dead_code)]
pub struct Grads {
    pub e_u: Vec<f32>,
    pub mu: Vec<f32>,
    pub b_m: Vec<f32>,
    pub r: Vec<(u32, Vec<f32>)>,
    pub b_g: Vec<(u32, f32)>,
}

/// Row-major `[rows × h]` gather of `table` at `idx`.
fn gather(table: &[f32], h: usize, idx: &[u32]) -> DMatrix<f32> {
    let mut m = DMatrix::<f32>::zeros(idx.len(), h);
    for (i, &r) in idx.iter().enumerate() {
        let r = r as usize;
        m.row_mut(i).copy_from_slice(&table[r * h..(r + 1) * h]);
    }
    m
}

/// In-place row softmax of `s`, returning per-row log-sum-exp.
fn softmax_rows(s: &mut DMatrix<f32>) -> Vec<f32> {
    let mut lse = Vec::with_capacity(s.nrows());
    for mut row in s.row_iter_mut() {
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|v| (v - mx).exp()).sum();
        let l = mx + z.ln();
        row.iter_mut().for_each(|v| *v = (*v - l).exp());
        lse.push(l);
    }
    lse
}

/// Pure: loss + gradients for `plan`. `modules_per_unit` is the K the plan
/// was drawn with (the 1/K importance weight).
#[allow(dead_code)]
pub fn loss_and_grads(
    params: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    plan: &StepPlan,
    modules_per_unit: usize,
) -> (StepStats, Grads) {
    let (h, n_m) = (params.h, part.n_modules());
    let b = plan.units.len();
    let inv_k = 1.0 / modules_per_unit.max(1) as f32;
    let w: Vec<f32> = plan
        .units
        .iter()
        .map(|&u| units.weight[u as usize])
        .collect();

    //////////////////
    // Module level //
    //////////////////

    let e_b = gather(&params.e_u, h, &plan.units); // [B × H]
    let mu = DMatrix::<f32>::from_row_slice(n_m, h, &params.mu); // [M × H]
    let mut s = &e_b * mu.transpose(); // [B × M]
    for mut row in s.row_iter_mut() {
        row.iter_mut().zip(&params.b_m).for_each(|(v, b)| *v += b);
    }
    let lse = softmax_rows(&mut s); // s is now p
    let mut loss_module = 0f64;
    let mut delta1 = DMatrix::<f32>::zeros(b, n_m); // w_u (p − q)
    for (i, &u) in plan.units.iter().enumerate() {
        let q = &um.q[u as usize * n_m..(u as usize + 1) * n_m];
        for m in 0..n_m {
            let p = s[(i, m)];
            if q[m] > 0.0 {
                // log p = (score − lse); recover the score from p and lse
                loss_module -= f64::from(w[i] * q[m] * (p.ln()));
            }
            delta1[(i, m)] = w[i] * (p - q[m]);
        }
    }
    let _ = lse;
    let g_e_module = &delta1 * &mu; // [B × H]
    let g_mu = delta1.transpose() * &e_b; // [M × H]
    let g_b_m: Vec<f32> = (0..n_m).map(|m| delta1.column(m).sum()).collect();

    ////////////////
    // Gene level //
    ////////////////

    // Position of each unit in the plan, for scattering e_u gradients.
    let pos_of: rustc_hash::FxHashMap<u32, usize> = plan
        .units
        .iter()
        .enumerate()
        .map(|(i, &u)| (u, i))
        .collect();
    struct ModuleOut {
        loss: f64,
        e_rows: Vec<(usize, Vec<f32>)>, // (position in plan, grad row)
        r_rows: Vec<(u32, Vec<f32>)>,
        b_rows: Vec<(u32, f32)>,
    }
    let outs: Vec<ModuleOut> = plan
        .pairs_by_module
        .par_iter()
        .map(|(m, us)| {
            let members = &part.members[*m as usize];
            let d_m = members.len();
            let r_m = gather(&params.r, h, members); // [d_m × H]
            let e_m = gather(&params.e_u, h, us); // [n × H]
            let mut s = &e_m * r_m.transpose(); // [n × d_m]
            for mut row in s.row_iter_mut() {
                for (j, &g) in members.iter().enumerate() {
                    row[j] += params.b_g[g as usize];
                }
            }
            softmax_rows(&mut s);
            let mut loss = 0f64;
            let mut delta2 = DMatrix::<f32>::zeros(us.len(), d_m); // (w_u/K) q_um (p − q_g|m)
            for (i, &u) in us.iter().enumerate() {
                let wu = units.weight[u as usize] * inv_k;
                let q_um = um.q[u as usize * n_m + *m as usize];
                let scale = wu * q_um;
                // target shares within the module
                let counts = um.by_module[u as usize]
                    .iter()
                    .find(|(k, _)| k == m)
                    .map(|(_, v)| v.as_slice())
                    .unwrap_or(&[]);
                let n_um = um.n_um[u as usize * n_m + *m as usize];
                let mut target = vec![0f32; d_m];
                for &(slot, c) in counts {
                    target[slot as usize] = c / n_um;
                }
                for j in 0..d_m {
                    let p = s[(i, j)];
                    if target[j] > 0.0 {
                        loss -= f64::from(scale * target[j] * p.ln());
                    }
                    delta2[(i, j)] = scale * (p - target[j]);
                }
            }
            let g_e = &delta2 * &r_m; // [n × H]
            let g_r = delta2.transpose() * &e_m; // [d_m × H]
            let e_rows = us
                .iter()
                .enumerate()
                .map(|(i, u)| (pos_of[u], g_e.row(i).iter().copied().collect()))
                .collect();
            let r_rows = members
                .iter()
                .enumerate()
                .map(|(j, &g)| (g, g_r.row(j).iter().copied().collect()))
                .collect();
            let b_rows = members
                .iter()
                .enumerate()
                .map(|(j, &g)| (g, delta2.column(j).sum()))
                .collect();
            ModuleOut {
                loss,
                e_rows,
                r_rows,
                b_rows,
            }
        })
        .collect();

    ////////////
    // Reduce //
    ////////////

    let mut g_e_u: Vec<f32> = g_e_module.transpose().as_slice().to_vec(); // row-major [B × H]
                                                                          // (nalgebra is column-major: transpose().as_slice() walks row-major of the original)
    let mut loss_gene = 0f64;
    let mut g_r: Vec<(u32, Vec<f32>)> = Vec::new();
    let mut g_b_g: Vec<(u32, f32)> = Vec::new();
    let mut n_pairs = 0usize;
    for o in outs {
        loss_gene += o.loss;
        for (i, row) in o.e_rows {
            n_pairs += 1;
            for k in 0..h {
                g_e_u[i * h + k] += row[k];
            }
        }
        g_r.extend(o.r_rows);
        g_b_g.extend(o.b_rows);
    }
    (
        StepStats {
            loss_module,
            loss_gene,
            n_units: b,
            n_pairs,
        },
        Grads {
            e_u: g_e_u,
            mu: g_mu.transpose().as_slice().to_vec(),
            b_m: g_b_m,
            r: g_r,
            b_g: g_b_g,
        },
    )
}

/// Apply `grads` with the row optimizers (weight decay `wd` on touched rows:
/// `row *= 1 − lr·wd` before the Adagrad step).
#[allow(dead_code)]
pub fn apply(
    params: &mut HierParams,
    opt: &mut Optimizers,
    grads: &Grads,
    plan: &StepPlan,
    wd: f32,
) {
    let h = params.h;
    for (i, &u) in plan.units.iter().enumerate() {
        let u = u as usize;
        let row = &mut params.e_u[u * h..(u + 1) * h];
        if wd > 0.0 {
            let f = 1.0 - opt.e_u.lr * wd;
            row.iter_mut().for_each(|x| *x *= f);
        }
        opt.e_u.update(u, row, &grads.e_u[i * h..(i + 1) * h]);
    }
    let n_m = params.b_m.len();
    for m in 0..n_m {
        // bias rides on the module's row: extend the gradient by one entry
        let mut g: Vec<f32> = grads.mu[m * h..(m + 1) * h].to_vec();
        g.push(grads.b_m[m]);
        let mut row: Vec<f32> = params.mu[m * h..(m + 1) * h].to_vec();
        row.push(params.b_m[m]);
        opt.mu.update(m, &mut row, &g);
        params.mu[m * h..(m + 1) * h].copy_from_slice(&row[..h]);
        params.b_m[m] = row[h];
    }
    let b_g_of: rustc_hash::FxHashMap<u32, f32> = grads.b_g.iter().copied().collect();
    for (g, gr) in &grads.r {
        let gi = *g as usize;
        let mut grow: Vec<f32> = gr.clone();
        grow.push(*b_g_of.get(g).unwrap_or(&0.0));
        let mut row: Vec<f32> = params.r[gi * h..(gi + 1) * h].to_vec();
        if wd > 0.0 {
            let f = 1.0 - opt.r.lr * wd;
            row.iter_mut().for_each(|x| *x *= f);
        }
        row.push(params.b_g[gi]);
        opt.r.update(gi, &mut row, &grow);
        params.r[gi * h..(gi + 1) * h].copy_from_slice(&row[..h]);
        params.b_g[gi] = row[h];
    }
}

#[cfg(test)]
#[path = "step_tests.rs"]
mod step_tests;
