//! Two-stage Bayesian clone gate: malignancy from burden + sketch, then
//! donor-peaky clone assignment.
//!
//! Finite mixture over background (`s=0`) and `K` clone components. Gibbs with
//! sufficient statistics; report `p_malig = E[m_i]` and MAP clone among kept
//! peaky components.

use crate::clone_call::{donor_of, CloneCallConfig, CloneRow};
use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::parquet::read_table_columns;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::multi::Dirichlet;
use rand_distr::Distribution;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::path::Path;

/// Default majority-donor purity floor for Bayes (single source of truth).
pub const DEFAULT_MIN_PURITY: f32 = 0.8;
/// Floor on residual / burden variance so σ and τ never hit zero.
const VAR_EPS: f32 = 1e-6;
/// Floor on sketch σ after empirical update.
const SIGMA_FLOOR: f32 = 1e-3;

/// Knobs for [`call_clones_bayes`].
#[derive(Debug, Clone)]
pub struct BayesCloneConfig {
    pub k_max: usize,
    pub min_cells: usize,
    /// Minimum empirical majority-donor purity to keep a clone component.
    pub min_purity: f32,
    pub p_malig_threshold: f32,
    pub n_sweeps: usize,
    pub n_burnin: usize,
    pub seed: u64,
    pub kmeans_iter: usize,
    /// Warm-start: fraction of highest-burden cells seeded into clones.
    pub warm_burden_frac: f32,
    /// Dirichlet concentration for clone donor mixes (≪ 1 ⇒ peaky).
    pub dir_alpha: f32,
    /// Dirichlet for mixture weights over {bg, clone_1..K}; bg gets `omega_bg`.
    pub omega_bg: f32,
    pub omega_clone: f32,
    /// Sketch isotropic SD for background (updated from residuals each sweep).
    pub sigma_bg: f32,
    /// Sketch isotropic SD for clone components (updated from residuals).
    pub sigma_clone: f32,
}

impl Default for BayesCloneConfig {
    fn default() -> Self {
        Self {
            k_max: 8,
            min_cells: 50,
            min_purity: DEFAULT_MIN_PURITY,
            p_malig_threshold: 0.5,
            n_sweeps: 100,
            n_burnin: 40,
            seed: 1,
            kmeans_iter: 200,
            // Top ~15% by burden; high enough to enrich true clones without
            // seeding the whole multi-donor batch tail.
            warm_burden_frac: 0.15,
            dir_alpha: 0.05,
            // Mild bg preference (plan: small malignant fraction). Large
            // omega_bg (~10) drowned real burden gaps of ~0.02 in log space.
            omega_bg: 2.0,
            omega_clone: 0.5,
            // Floors only; Gibbs updates σ from residual SSE.
            sigma_bg: 0.05,
            sigma_clone: 0.08,
        }
    }
}

/// Build [`BayesCloneConfig`] from the shared clone-call knobs.
#[must_use]
pub fn bayes_config_from_clone_call(cfg: &CloneCallConfig) -> BayesCloneConfig {
    BayesCloneConfig {
        k_max: cfg.k_max,
        min_cells: cfg.min_cells,
        min_purity: cfg.min_purity.unwrap_or(DEFAULT_MIN_PURITY),
        p_malig_threshold: cfg.p_malig_threshold,
        n_sweeps: cfg.n_sweeps,
        n_burnin: cfg.n_burnin,
        seed: cfg.seed,
        kmeans_iter: cfg.kmeans_iter,
        ..Default::default()
    }
}

/// Sibling `{prefix}.cells.parquet` for a CNV backend path.
#[must_use]
pub fn cells_table_beside(backend_path: &str) -> String {
    for suffix in [".zarr.zip", ".zarr", ".h5"] {
        if let Some(prefix) = backend_path.strip_suffix(suffix) {
            return format!("{prefix}.cells.parquet");
        }
    }
    format!("{backend_path}.cells.parquet")
}

/// Load `cnv_burden` from a `{prefix}.cells.parquet` table (`cell`, `depth`,
/// `cnv_burden`), aligned to `cell_names`. Returns `Ok(None)` if the file is
/// missing. A cell absent from the table takes its `fallback` burden (the
/// fused mean |CNV| from the sketch pass) rather than a zero, which would
/// otherwise become a `ln(1e-8)` outlier in the log-burden mixture.
pub fn load_burden_from_cells_table(
    path: &str,
    cell_names: &[Box<str>],
    fallback: &[f32],
) -> anyhow::Result<Option<Vec<f32>>> {
    if !Path::new(path).exists() {
        return Ok(None);
    }
    anyhow::ensure!(
        fallback.len() == cell_names.len(),
        "fallback burden length {} != n_cells {}",
        fallback.len(),
        cell_names.len()
    );
    let (strs, nums) = read_table_columns(path, &["cell"], &["cnv_burden"])?;
    let by_cell: FxHashMap<&str, f32> = strs[0]
        .iter()
        .zip(&nums[0])
        .map(|(c, &b)| (c.as_ref(), b as f32))
        .collect();
    let mut out = Vec::with_capacity(cell_names.len());
    let mut missing = 0usize;
    for (c, &fb) in cell_names.iter().zip(fallback) {
        match by_cell.get(c.as_ref()) {
            Some(&b) => out.push(b),
            None => {
                missing += 1;
                out.push(fb);
            }
        }
    }
    if missing > 0 {
        log::warn!(
            "{} / {} cells missing from {}; using fused mean |CNV| for those",
            missing,
            cell_names.len(),
            path
        );
    }
    Ok(Some(out))
}

/// Call clones from a genomic sketch and per-cell burdens.
///
/// `sketch` is `[C × N]`; `burden[j]` is mean |log-ratio| for cell `j`.
pub fn call_clones_bayes(
    sketch: &DMatrix<f32>,
    cell_names: &[Box<str>],
    burden: &[f32],
    cfg: &BayesCloneConfig,
) -> Vec<CloneRow> {
    let n = sketch.ncols();
    let c = sketch.nrows();
    assert_eq!(n, cell_names.len());
    assert_eq!(n, burden.len());

    let donors: Vec<Box<str>> = cell_names.iter().map(|x| donor_of(x).into()).collect();
    let mut donor_id: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut donor_of_cell = Vec::with_capacity(n);
    let mut n_donors = 0usize;
    for d in &donors {
        let id = *donor_id.entry(d.clone()).or_insert_with(|| {
            let i = n_donors;
            n_donors += 1;
            i
        });
        donor_of_cell.push(id);
    }
    let n_donors = n_donors.max(1);
    if n_donors == 1 && n > 0 {
        log::warn!(
            "single donor: majority-donor purity is trivially 1, so the purity gate \
             cannot reject a high-burden tail; treat clone calls with caution"
        );
    }
    let k_max = cfg.k_max.max(1).min(n.max(1));

    if n == 0 || c == 0 {
        return Vec::new();
    }

    let log_b: Vec<f32> = burden.iter().map(|&b| b.max(1e-8).ln()).collect();

    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut assign = warm_start(sketch, &log_b, k_max, cfg, &mut rng);

    let n_comp = k_max + 1;
    let mut mu = DMatrix::<f32>::zeros(c, n_comp);
    let mut omega = vec![1.0 / n_comp as f32; n_comp];
    let mut phi = vec![vec![1.0 / n_donors as f32; n_donors]; n_comp];
    let mut nu = [0.0f32, 0.0];
    let mut tau = [1.0f32, 1.0];
    let mut sigma_bg = cfg.sigma_bg.max(SIGMA_FLOOR);
    let mut sigma_clone = cfg.sigma_clone.max(SIGMA_FLOOR);
    // Background donor mix is uniform and data-independent.
    phi[0].fill(1.0 / n_donors as f32);
    let log_b_mean = log_b.iter().sum::<f32>() / (n as f32).max(1.0);

    let mut p_malig_acc = vec![0.0f32; n];
    let mut assign_counts = vec![vec![0u32; n_comp]; n];
    let mut n_post = 0usize;

    let total_sweeps = cfg.n_burnin + cfg.n_sweeps;
    for sweep in 0..total_sweeps {
        if sweep == 0 || (sweep + 1) % 20 == 0 || sweep + 1 == total_sweeps {
            log::info!(
                "bayes Gibbs sweep {} / {} (σ_bg={:.4}, σ_clone={:.4})",
                sweep + 1,
                total_sweeps,
                sigma_bg,
                sigma_clone
            );
        }
        update_globals(
            sketch,
            &log_b,
            log_b_mean,
            &assign,
            &donor_of_cell,
            n_donors,
            &mut mu,
            &mut omega,
            &mut phi,
            &mut nu,
            &mut tau,
            &mut sigma_bg,
            &mut sigma_clone,
            cfg,
            &mut rng,
        );
        assign = sample_assignments(
            sketch,
            &log_b,
            &donor_of_cell,
            &mu,
            &omega,
            &phi,
            &nu,
            &tau,
            sigma_bg,
            sigma_clone,
            &mut rng,
        );
        if sweep >= cfg.n_burnin {
            n_post += 1;
            for i in 0..n {
                let a = assign[i];
                if a > 0 {
                    p_malig_acc[i] += 1.0;
                }
                assign_counts[i][a] += 1;
            }
        }
    }
    let n_post = n_post.max(1) as f32;
    let p_malig: Vec<f32> = p_malig_acc.iter().map(|v| v / n_post).collect();

    let map_comp: Vec<usize> = assign_counts
        .iter()
        .map(|counts| {
            counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &cnt)| cnt)
                .map(|(k, _)| k)
                .unwrap_or(0)
        })
        .collect();

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_comp];
    for (i, &comp) in map_comp.iter().enumerate() {
        if p_malig[i] >= cfg.p_malig_threshold && comp > 0 {
            members[comp].push(i);
        }
    }

    let mut purity_of = vec![0.0f32; n_comp];
    for (k, mem) in members.iter().enumerate().skip(1) {
        if mem.len() < cfg.min_cells {
            continue;
        }
        let mut counts = vec![0usize; n_donors];
        for &i in mem {
            counts[donor_of_cell[i]] += 1;
        }
        let maj = counts.iter().copied().max().unwrap_or(0);
        purity_of[k] = maj as f32 / mem.len() as f32;
    }

    let mut kept: Vec<usize> = (1..n_comp)
        .filter(|&k| members[k].len() >= cfg.min_cells && purity_of[k] >= cfg.min_purity)
        .collect();
    kept.sort_by_key(|&k| std::cmp::Reverse(members[k].len()));
    let mut remap = vec![0usize; n_comp];
    for (new_id, &old) in kept.iter().enumerate() {
        remap[old] = new_id + 1;
    }

    log::info!(
        "bayes clone gate: {} / {} cells p_malig≥{:.2}; kept {} clone stratum(a)",
        p_malig
            .iter()
            .filter(|&&p| p >= cfg.p_malig_threshold)
            .count(),
        n,
        cfg.p_malig_threshold,
        kept.len()
    );
    for &k in &kept {
        log::info!(
            "bayes clone component {k} → stratum {}: n={}, purity={:.3}",
            remap[k],
            members[k].len(),
            purity_of[k]
        );
    }

    let spatial: Vec<f32> = (0..n_comp)
        .map(|k| {
            let col = mu.column(k);
            col.iter().map(|v| v.abs()).sum::<f32>() / (c.max(1) as f32)
        })
        .collect();

    (0..n)
        .map(|j| {
            let comp = map_comp[j];
            let stratum = if p_malig[j] >= cfg.p_malig_threshold {
                remap[comp]
            } else {
                0
            };
            let pur = if comp > 0 { purity_of[comp] } else { 0.0 };
            CloneRow {
                cell: cell_names[j].clone(),
                donor: donors[j].clone(),
                cluster: comp,
                stratum,
                purity: pur,
                spatial_score: spatial[comp.min(n_comp - 1)],
                spatial_z: 0.0,
                burden: burden[j],
                p_malig: p_malig[j],
            }
        })
        .collect()
}

fn warm_start(
    sketch: &DMatrix<f32>,
    log_b: &[f32],
    k_max: usize,
    cfg: &BayesCloneConfig,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = sketch.ncols();
    let mut assign = vec![0usize; n];
    if n == 0 || k_max == 0 {
        return assign;
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        log_b[b]
            .partial_cmp(&log_b[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_seed = ((n as f32) * cfg.warm_burden_frac.clamp(0.05, 0.5))
        .round()
        .max(k_max as f32) as usize;
    let n_seed = n_seed.min(n);
    let seeds = &order[..n_seed];
    if seeds.len() < k_max || sketch.nrows() == 0 {
        for (t, &i) in seeds.iter().enumerate() {
            assign[i] = (t % k_max) + 1;
        }
        return assign;
    }

    let c = sketch.nrows();
    let mut sub = DMatrix::<f32>::zeros(c, seeds.len());
    for (j, &i) in seeds.iter().enumerate() {
        sub.column_mut(j).copy_from(&sketch.column(i));
    }
    let labels = sub.kmeans_columns(KmeansArgs {
        num_clusters: k_max,
        max_iter: cfg.kmeans_iter,
    });
    for (j, &i) in seeds.iter().enumerate() {
        assign[i] = labels[j].min(k_max - 1) + 1;
    }
    for _ in 0..(k_max.min(n)) {
        let i = rng.random_range(0..n);
        if log_b[i] > log_b[order[n_seed / 2]] {
            assign[i] = rng.random_range(1..=k_max);
        }
    }
    assign
}

#[allow(clippy::too_many_arguments)]
fn update_globals(
    sketch: &DMatrix<f32>,
    log_b: &[f32],
    log_b_mean: f32,
    assign: &[usize],
    donor_of_cell: &[usize],
    n_donors: usize,
    mu: &mut DMatrix<f32>,
    omega: &mut [f32],
    phi: &mut [Vec<f32>],
    nu: &mut [f32; 2],
    tau: &mut [f32; 2],
    sigma_bg: &mut f32,
    sigma_clone: &mut f32,
    cfg: &BayesCloneConfig,
    rng: &mut SmallRng,
) {
    let n = assign.len();
    let c = sketch.nrows();
    let n_comp = omega.len();

    // One pass: component counts, sketch sums, donor counts, burden moments.
    let mut counts = vec![0.0f32; n_comp];
    let mut sums = DMatrix::<f32>::zeros(c, n_comp);
    let mut donor_counts = vec![vec![cfg.dir_alpha; n_donors]; n_comp];
    let mut sum_lb = [0.0f32; 2];
    let mut ss_lb = [0.0f32; 2];
    let mut n_lb = [0.0f32; 2];
    for i in 0..n {
        let a = assign[i].min(n_comp - 1);
        counts[a] += 1.0;
        for r in 0..c {
            sums[(r, a)] += sketch[(r, i)];
        }
        if a > 0 {
            donor_counts[a][donor_of_cell[i]] += 1.0;
        }
        let lb = log_b[i];
        let m = if a == 0 { 0 } else { 1 };
        sum_lb[m] += lb;
        ss_lb[m] += lb * lb;
        n_lb[m] += 1.0;
    }

    let mut alphas = vec![cfg.omega_clone; n_comp];
    alphas[0] = cfg.omega_bg;
    for k in 0..n_comp {
        alphas[k] += counts[k];
    }
    omega.copy_from_slice(&dirichlet_sample(&alphas, rng));

    // Use previous-sweep σ for the Normal–Normal μ update, then refresh σ.
    let inv_var_bg = 1.0 / (*sigma_bg * *sigma_bg + VAR_EPS);
    let inv_var_cl = 1.0 / (*sigma_clone * *sigma_clone + VAR_EPS);
    let prior_prec = 1.0f32;
    for k in 0..n_comp {
        let nk = counts[k];
        let inv_var = if k == 0 { inv_var_bg } else { inv_var_cl };
        let post_prec = prior_prec + nk * inv_var;
        for r in 0..c {
            mu[(r, k)] = if nk > 0.0 {
                (sums[(r, k)] * inv_var) / post_prec
            } else {
                0.0
            };
        }
    }

    // Empirical σ from residuals vs updated μ, shrunk toward cfg priors.
    let mut sse = [0.0f32; 2];
    let mut n_obs = [0.0f32; 2];
    for i in 0..n {
        let a = assign[i].min(n_comp - 1);
        let m = if a == 0 { 0 } else { 1 };
        for r in 0..c {
            let d = sketch[(r, i)] - mu[(r, a)];
            sse[m] += d * d;
        }
        n_obs[m] += c as f32;
    }
    let prior_obs = (c as f32) * 20.0; // ~20 cells' worth of pseudo-observations
    *sigma_bg = empir_sigma(sse[0], n_obs[0], cfg.sigma_bg, prior_obs);
    *sigma_clone = empir_sigma(sse[1], n_obs[1], cfg.sigma_clone, prior_obs);

    for k in 1..n_comp {
        phi[k] = dirichlet_sample(&donor_counts[k], rng);
    }

    // Mild log-burden separation prior (real cohorts often have Δlog b ≈ 0.2–0.4).
    let prior_n = 5.0f32;
    let prior_nu0 = log_b_mean - 0.15;
    let prior_nu1 = log_b_mean + 0.25;
    nu[0] = (sum_lb[0] + prior_n * prior_nu0) / (n_lb[0] + prior_n);
    nu[1] = (sum_lb[1] + prior_n * prior_nu1) / (n_lb[1] + prior_n);
    // Identifiability: component 1 is "higher burden" by definition, so the
    // order is enforced every sweep. On data with no burden gap this still
    // labels the high-burden tail malignant; the purity / min_cells gate is
    // what stops that tail becoming a clone, and it is vacuous with one donor.
    if nu[1] < nu[0] + 0.05 {
        let mid = 0.5 * (nu[0] + nu[1]);
        nu[0] = mid - 0.08;
        nu[1] = mid + 0.08;
    }
    for m in 0..2 {
        let var = if n_lb[m] > 1.0 {
            (ss_lb[m] - sum_lb[m] * sum_lb[m] / n_lb[m]) / (n_lb[m] - 1.0)
        } else {
            0.1
        };
        tau[m] = (var.max(0.0) + VAR_EPS).sqrt().max(0.02);
    }
}

/// Shrink residual variance toward `prior_sigma²` with `prior_obs` pseudo-counts.
fn empir_sigma(sse: f32, n_obs: f32, prior_sigma: f32, prior_obs: f32) -> f32 {
    let prior_s2 = prior_sigma.max(SIGMA_FLOOR).powi(2);
    let s2 = if n_obs > 0.0 {
        (prior_obs * prior_s2 + sse) / (prior_obs + n_obs)
    } else {
        prior_s2
    };
    (s2.max(0.0) + VAR_EPS).sqrt().clamp(SIGMA_FLOOR, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn sample_assignments(
    sketch: &DMatrix<f32>,
    log_b: &[f32],
    donor_of_cell: &[usize],
    mu: &DMatrix<f32>,
    omega: &[f32],
    phi: &[Vec<f32>],
    nu: &[f32; 2],
    tau: &[f32; 2],
    sigma_bg: f32,
    sigma_clone: f32,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let n = sketch.ncols();
    let c = sketch.nrows();
    let n_comp = omega.len();
    let sigma_bg = sigma_bg.max(SIGMA_FLOOR);
    let sigma_clone = sigma_clone.max(SIGMA_FLOOR);
    let inv_2var_bg = 0.5 / (sigma_bg * sigma_bg + VAR_EPS);
    let inv_2var_cl = 0.5 / (sigma_clone * sigma_clone + VAR_EPS);
    let log_norm_bg =
        -0.5 * (c as f32) * (2.0 * std::f32::consts::PI * (sigma_bg * sigma_bg + VAR_EPS)).ln();
    let log_norm_cl = -0.5
        * (c as f32)
        * (2.0 * std::f32::consts::PI * (sigma_clone * sigma_clone + VAR_EPS)).ln();

    // Average the C-dimensional sketch log-likelihood so it is commensurate
    // with the scalar burden term.
    let inv_c = 1.0 / (c.max(1) as f32);
    let log_omega: Vec<f32> = omega.iter().map(|&w| w.max(1e-30).ln()).collect();
    let log_phi: Vec<Vec<f32>> = phi
        .iter()
        .map(|p| p.iter().map(|&x| x.max(1e-30).ln()).collect())
        .collect();

    let mut logps = vec![0.0f32; n * n_comp];
    logps
        .par_chunks_mut(n_comp)
        .enumerate()
        .for_each(|(i, lp)| {
            let d = donor_of_cell[i];
            let lb = log_b[i];
            for k in 0..n_comp {
                let mut score = log_omega[k];
                let mut sse = 0.0f32;
                for r in 0..c {
                    let diff = sketch[(r, i)] - mu[(r, k)];
                    sse += diff * diff;
                }
                // Every component carries its donor term; φ_0 is uniform, so
                // dropping it here would bias all cells toward bg by ln(D).
                score += log_phi[k][d];
                if k == 0 {
                    score += inv_c * (log_norm_bg - inv_2var_bg * sse);
                    score += log_normal_lpdf(lb, nu[0], tau[0]);
                } else {
                    score += inv_c * (log_norm_cl - inv_2var_cl * sse);
                    score += log_normal_lpdf(lb, nu[1], tau[1]);
                }
                lp[k] = score;
            }
        });

    let mut assign = vec![0usize; n];
    let mut scratch = vec![0.0f32; n_comp];
    for i in 0..n {
        assign[i] = categorical_log(&logps[i * n_comp..(i + 1) * n_comp], &mut scratch, rng);
    }
    assign
}

fn log_normal_lpdf(x: f32, mean: f32, sd: f32) -> f32 {
    let var = sd.max(0.0) * sd.max(0.0) + VAR_EPS;
    let sd = var.sqrt();
    let z = (x - mean) / sd;
    -0.5 * z * z - sd.ln() - 0.5 * (2.0 * std::f32::consts::PI).ln()
}

fn categorical_log(logp: &[f32], scratch: &mut [f32], rng: &mut SmallRng) -> usize {
    let max = logp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n = logp.len();
    let w = &mut scratch[..n];
    let mut sum = 0.0f32;
    for i in 0..n {
        w[i] = (logp[i] - max).exp();
        sum += w[i];
    }
    if !sum.is_finite() || sum <= 0.0 {
        return 0;
    }
    for x in w.iter_mut() {
        *x /= sum;
    }
    let u: f32 = rng.random();
    let mut acc = 0.0f32;
    for (i, &wi) in w.iter().enumerate() {
        acc += wi;
        if u <= acc {
            return i;
        }
    }
    n.saturating_sub(1)
}

fn dirichlet_sample(alpha: &[f32], rng: &mut SmallRng) -> Vec<f32> {
    if alpha.len() < 2 {
        return vec![1.0];
    }
    let alpha_f64: Vec<f64> = alpha.iter().map(|&a| a.max(1e-3) as f64).collect();
    match Dirichlet::new(&alpha_f64) {
        Ok(dist) => {
            let draw: Vec<f64> = dist.sample(rng);
            draw.into_iter().map(|x| x as f32).collect()
        }
        Err(_) => {
            let u = 1.0 / alpha.len() as f32;
            vec![u; alpha.len()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_multi_donor(n_per: usize, n_donors: usize) -> (DMatrix<f32>, Vec<Box<str>>, Vec<f32>) {
        let n = n_per * n_donors;
        let mut rng = SmallRng::seed_from_u64(11);
        let mut data = Vec::with_capacity(4 * n);
        let mut names = Vec::with_capacity(n);
        let mut burden = Vec::with_capacity(n);
        for d in 0..n_donors {
            let shift = 0.02 * (d as f32 - 1.5);
            for i in 0..n_per {
                data.extend_from_slice(&[
                    shift + 0.01 * (rng.random::<f32>() - 0.5),
                    -shift + 0.01 * (rng.random::<f32>() - 0.5),
                    0.01 * (rng.random::<f32>() - 0.5),
                    0.01 * (rng.random::<f32>() - 0.5),
                ]);
                names.push(format!("c{i}@D{d}").into_boxed_str());
                burden.push(0.04 + 0.002 * (rng.random::<f32>() - 0.5));
            }
        }
        (DMatrix::from_vec(4, n, data), names, burden)
    }

    fn private_high_burden_clone(
        n_flat: usize,
        n_clone: usize,
    ) -> (DMatrix<f32>, Vec<Box<str>>, Vec<f32>) {
        let n = n_flat + n_clone;
        let mut rng = SmallRng::seed_from_u64(7);
        let mut data = Vec::with_capacity(3 * n);
        let mut names = Vec::with_capacity(n);
        let mut burden = Vec::with_capacity(n);
        for i in 0..n_flat {
            data.extend_from_slice(&[
                0.01 * (rng.random::<f32>() - 0.5),
                0.01 * (rng.random::<f32>() - 0.5),
                0.01 * (rng.random::<f32>() - 0.5),
            ]);
            names.push(format!("c{i}@Ctl").into_boxed_str());
            burden.push(0.04 + 0.002 * (rng.random::<f32>() - 0.5));
        }
        for i in 0..n_clone {
            data.extend_from_slice(&[
                -0.8 + 0.02 * (rng.random::<f32>() - 0.5),
                0.02 * (rng.random::<f32>() - 0.5),
                0.5 + 0.02 * (rng.random::<f32>() - 0.5),
            ]);
            names.push(format!("t{i}@AML").into_boxed_str());
            burden.push(0.20 + 0.01 * (rng.random::<f32>() - 0.5));
        }
        (DMatrix::from_vec(3, n, data), names, burden)
    }

    #[test]
    fn flat_burden_multi_donor_yields_no_clones() {
        let (sketch, names, burden) = flat_multi_donor(40, 4);
        let rows = call_clones_bayes(
            &sketch,
            &names,
            &burden,
            &BayesCloneConfig {
                k_max: 4,
                min_cells: 20,
                min_purity: 0.7,
                p_malig_threshold: 0.5,
                n_sweeps: 40,
                n_burnin: 20,
                seed: 3,
                kmeans_iter: 40,
                ..Default::default()
            },
        );
        let n_clone = rows.iter().filter(|r| r.stratum > 0).count();
        assert_eq!(
            n_clone, 0,
            "flat multi-donor batch must stay in stratum 0, got {n_clone}"
        );
    }

    #[test]
    fn private_high_burden_clone_is_kept() {
        let (sketch, names, burden) = private_high_burden_clone(80, 80);
        let rows = call_clones_bayes(
            &sketch,
            &names,
            &burden,
            &BayesCloneConfig {
                k_max: 3,
                min_cells: 20,
                min_purity: 0.7,
                p_malig_threshold: 0.4,
                n_sweeps: 60,
                n_burnin: 30,
                seed: 5,
                kmeans_iter: 50,
                warm_burden_frac: 0.4,
                ..Default::default()
            },
        );
        let n_clone = rows.iter().filter(|r| r.stratum > 0).count();
        let aml_cloned = rows
            .iter()
            .filter(|r| r.donor.as_ref() == "AML" && r.stratum > 0)
            .count();
        let ctl_cloned = rows
            .iter()
            .filter(|r| r.donor.as_ref() == "Ctl" && r.stratum > 0)
            .count();
        assert!(n_clone >= 40, "expected a clone, got {n_clone} cells");
        assert!(aml_cloned >= 40, "AML cells should form the clone");
        assert!(
            ctl_cloned < 15,
            "Control cells should stay in bucket 0, got {ctl_cloned}"
        );
    }

    #[test]
    fn cells_table_beside_strips_zarr_zip() {
        assert_eq!(
            cells_table_beside("/tmp/foo.zarr.zip"),
            "/tmp/foo.cells.parquet"
        );
    }
}
