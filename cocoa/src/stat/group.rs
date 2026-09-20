//! Group-level exposure model.
//!
//! ```text
//!   y1(d,i,p) ~ Poisson( tau(d, x(i)) * delta(d,i) * mu(d,p) * n(i,p) )
//!   y0(d,p)   ~ Poisson( gamma(d,p) * mu(d,p) * n(p) )
//!   delta(d,i) ~ Gamma(phi_d, phi_d)
//! ```
//!
//! `tau` is the average exposure effect, gene x exposure group. `delta` is
//! the individual effect with the exposure effect removed: a random effect
//! with mean 1 inside every group and between-individual dispersion `1/phi`.
//! `delta` is integrated over, never divided out. Uncertainty on the
//! exposure contrast is by exposure-label permutation over individuals,
//! refitting this whole model under every relabeling.
//!
//! `phi_d` follows a trend across genes: a Cox-Reid adjusted profile maximum
//! per gene, then a weighted log-linear regression of log phi on log mean
//! expression over all genes of the topic, and every gene takes the trend
//! value at its own mean. That is the robust, fast choice: the exposure
//! contrast barely depends on phi (its null comes from permutation), and a
//! two-parameter trend cannot chase per-gene noise. The fit runs in two
//! passes over gene blocks: pass 1 with phi free (yielding the per-gene
//! evidence), the global trend, and pass 2 with phi fixed at the trend.

use super::*;
use matrix_util::utils::median;
use special::Gamma as SpecialGamma;

/// Fitted group model for one topic. `mu` and `gamma` are nuisance
/// parameters of the fit and are not retained.
pub struct CocoaGroupOut {
    /// tau: average exposure effect, gene x exposure group
    pub exposure: GammaMatrix,
    /// delta: individual effect with exposure removed, gene x individual
    pub indv_delta: GammaMatrix,
    /// phi: between-individual dispersion per gene (the trend at its mean)
    pub dispersion: DVec,
    /// Per-gene dispersion evidence from the phi-free pass; `None` when phi
    /// was supplied (oracle fits).
    pub dispersion_fit: Option<DispersionFit>,
    /// The fitted across-gene trend; `None` when phi was supplied.
    pub dispersion_prior: Option<DispersionPrior>,
}

/// Per-gene dispersion evidence for one topic.
#[derive(Clone, Debug)]
pub struct DispersionFit {
    /// Cox-Reid adjusted profile maximum of log phi, unshrunk; NaN when the
    /// gene has fewer than two informative individuals in any group.
    pub log_phi_hat: DVec,
    /// Whether the maximum sits at the edge of the search bracket, in which
    /// case the gene carries no usable dispersion evidence.
    pub at_bound: Vec<bool>,
    /// Informative individuals: those with a positive expected total in a
    /// group holding at least two of them.
    pub n: Vec<usize>,
    /// log of the mean count per cell, from the data alone.
    pub log_mean: DVec,
}

/// Log-linear trend of log phi on log mean: `log phi = a + b log mean`.
#[derive(Clone, Copy, Debug)]
pub struct DispersionPrior {
    pub a: f32,
    pub b: f32,
    /// Genes that entered the trend fit.
    pub n_fit: usize,
}

const PHI_MIN: f32 = 1e-2;
const PHI_MAX: f32 = 1e4;
pub(crate) const PHI_INIT: f32 = 10.0;
/// Re-estimate phi after this many coordinate sweeps (phi-free pass).
const PHI_UPDATE_EVERY: usize = 10;
/// Golden-section steps over log phi on the full bracket (first fit) and on a
/// bracket of +/- `PHI_WARM_HALF_WIDTH` around the previous value (refits).
const PHI_SEARCH_STEPS: usize = 25;
const PHI_WARM_STEPS: usize = 18;
const PHI_WARM_HALF_WIDTH: f32 = 3.0;
/// Pseudo-individuals for the interim block-median shrinkage during pass 1.
const PHI_SHRINK_DF: f32 = 4.0;
/// A profile maximum closer than this to a bracket edge is treated as no evidence.
const BOUND_MARGIN: f32 = 0.2;
/// Fewer genes than this fall back to a constant trend.
const TREND_MIN_GENES: usize = 10;
/// Genes per independent fit; every update is row-separable, so blocks are the
/// unit of parallelism and bound memory to one block per core.
pub(crate) const GENE_BLOCK: usize = 512;
/// Stop sweeping once the largest relative change in tau and delta falls
/// below this.
const CONVERGENCE_TOL: f32 = 1e-4;

/// Whether a block fit estimates phi or holds it fixed.
enum PhiMode<'a> {
    Free,
    Fixed(&'a DVec),
}

/// One gene block of one topic after a fit.
struct BlockFit {
    exposure: GammaMatrix,
    indv_delta: GammaMatrix,
    phi: DVec,
    disp: Option<DispersionFit>,
    tau_dx: Mat,
    delta_di: Mat,
}

impl CocoaStat {
    /// Fit the group-level model for every topic, in parallel over
    /// (topic, gene block). `indv_to_group[i]` is the exposure group of
    /// individual `i`, in `0..n_groups`; individuals without an exposure get
    /// their own group.
    pub fn estimate_group_parameters(
        &self,
        indv_to_group: &[usize],
        n_groups: usize,
    ) -> anyhow::Result<Vec<CocoaGroupOut>> {
        self.estimate_group_parameters_blocked(indv_to_group, n_groups, GENE_BLOCK)
    }

    /// [`Self::estimate_group_parameters`] with an explicit gene block size.
    pub fn estimate_group_parameters_blocked(
        &self,
        indv_to_group: &[usize],
        n_groups: usize,
        block: usize,
    ) -> anyhow::Result<Vec<CocoaGroupOut>> {
        let jobs = self.block_jobs(indv_to_group, n_groups, block)?;
        let n_genes = self.y1_stat(0).nrows();
        let n_blocks = n_genes.div_ceil(block);

        // Pass 1: phi free, collect per-gene evidence and a warm start.
        let pass1: Result<Vec<BlockFit>, _> = jobs
            .par_iter()
            .map(|&(k, row0, nrows)| {
                self.optimize_group_block(
                    k,
                    row0,
                    nrows,
                    indv_to_group,
                    n_groups,
                    PhiMode::Free,
                    None,
                )
            })
            .collect();
        let pass1 = pass1?;

        // Global step per topic: the trend, then phi at every gene's mean.
        let mut priors = Vec::with_capacity(self.n_topics);
        let mut fits = Vec::with_capacity(self.n_topics);
        let mut phis = Vec::with_capacity(self.n_topics);
        for k in 0..self.n_topics {
            let disp: Vec<DispersionFit> = pass1[k * n_blocks..(k + 1) * n_blocks]
                .iter()
                .map(|b| b.disp.clone().expect("pass 1 carries dispersion evidence"))
                .collect();
            let fit = DispersionFit::vconcat(disp);
            let prior = DispersionPrior::fit(&fit);
            info!(
                "topic {}: dispersion trend log phi = {:.3} + {:.3} log mean ({} genes)",
                k, prior.a, prior.b, prior.n_fit
            );
            phis.push(prior.trend_phi(&fit));
            priors.push(prior);
            fits.push(fit);
        }

        // Pass 2: phi fixed at the trend, warm-started from pass 1.
        let pass2: Result<Vec<BlockFit>, _> = jobs
            .par_iter()
            .zip(pass1.par_iter())
            .map(|(&(k, row0, nrows), warm)| {
                self.optimize_group_block(
                    k,
                    row0,
                    nrows,
                    indv_to_group,
                    n_groups,
                    PhiMode::Fixed(&phis[k]),
                    Some((&warm.tau_dx, &warm.delta_di)),
                )
            })
            .collect();
        let mut pass2 = pass2?.into_iter();

        let mut out = Vec::with_capacity(self.n_topics);
        for (fit, prior) in fits.into_iter().zip(priors) {
            let blocks: Vec<BlockFit> = pass2.by_ref().take(n_blocks).collect();
            let mut topic = concat_blocks(blocks);
            topic.dispersion_fit = Some(fit);
            topic.dispersion_prior = Some(prior);
            out.push(topic);
        }
        info!(
            "finished group-model optimization for {} topics",
            self.n_topics
        );
        Ok(out)
    }

    /// Fit with phi supplied per topic (gene-length vectors) and never
    /// re-estimated: the oracle path used by the recovery tests.
    #[cfg(test)]
    pub fn estimate_group_parameters_with_dispersion(
        &self,
        indv_to_group: &[usize],
        n_groups: usize,
        block: usize,
        phi: &[DVec],
    ) -> anyhow::Result<Vec<CocoaGroupOut>> {
        anyhow::ensure!(
            phi.len() == self.n_topics,
            "phi has {} topics, stat has {}",
            phi.len(),
            self.n_topics
        );
        let jobs = self.block_jobs(indv_to_group, n_groups, block)?;
        let n_genes = self.y1_stat(0).nrows();
        anyhow::ensure!(
            phi.iter().all(|p| p.len() == n_genes),
            "every phi vector must have {} genes",
            n_genes
        );
        let n_blocks = n_genes.div_ceil(block);
        let fits: Result<Vec<BlockFit>, _> = jobs
            .par_iter()
            .map(|&(k, row0, nrows)| {
                self.optimize_group_block(
                    k,
                    row0,
                    nrows,
                    indv_to_group,
                    n_groups,
                    PhiMode::Fixed(&phi[k]),
                    None,
                )
            })
            .collect();
        let mut fits = fits?.into_iter();
        let out = (0..self.n_topics)
            .map(|_| concat_blocks(fits.by_ref().take(n_blocks).collect()))
            .collect();
        Ok(out)
    }

    /// (topic, first row, rows) for every block, after validating the design.
    fn block_jobs(
        &self,
        indv_to_group: &[usize],
        n_groups: usize,
        block: usize,
    ) -> anyhow::Result<Vec<(usize, usize, usize)>> {
        anyhow::ensure!(block > 0, "gene block size must be positive");
        anyhow::ensure!(self.n_topics > 0, "no topics to fit");
        let n_genes = self.y1_stat(0).nrows();
        let n_indv = self.indv_y1_stat(0).ncols();
        anyhow::ensure!(
            indv_to_group.len() == n_indv,
            "indv_to_group has {} entries, stat has {} individuals",
            indv_to_group.len(),
            n_indv
        );
        anyhow::ensure!(
            indv_to_group.iter().all(|&x| x < n_groups),
            "indv_to_group contains a group >= n_groups = {}",
            n_groups
        );
        Ok((0..self.n_topics)
            .flat_map(|k| {
                (0..n_genes)
                    .step_by(block)
                    .map(move |row0| (k, row0, block.min(n_genes - row0)))
            })
            .collect())
    }

    /// Fit genes `row0..row0 + nrows` of topic `k`.
    #[allow(clippy::too_many_arguments)]
    fn optimize_group_block(
        &self,
        k: usize,
        row0: usize,
        nrows: usize,
        indv_to_group: &[usize],
        n_groups: usize,
        phi_mode: PhiMode<'_>,
        warm: Option<(&Mat, &Mat)>,
    ) -> anyhow::Result<BlockFit> {
        let y1_dp = self.y1_stat(k).rows(row0, nrows).into_owned();
        let y0_dp = self.y0_stat(k).rows(row0, nrows).into_owned();
        let y10_dp = &y1_dp + &y0_dp;
        let y1_di = self.indv_y1_stat(k).rows(row0, nrows).into_owned();
        let size_p = self.size_stat(k);
        let size_ip = self.indv_size_stat(k);

        let n_genes = nrows;
        let n_pb = y1_dp.ncols();
        let n_indv = y1_di.ncols();

        // Indicator (individual x group) for group sums; loop-invariant.
        let g_ig = Mat::from_fn(n_indv, n_groups, |i, x| {
            (indv_to_group[i] == x) as u8 as f32
        });
        let num_dx = &y1_di * &g_ig;
        let size_ip_t = size_ip.transpose();

        // log mean count per cell, from the data alone.
        let total_cells: f32 = size_ip.iter().sum();
        let log_mean = DVec::from_fn(n_genes, |d, _| {
            ((y1_di.row(d).sum() + 0.5) / total_cells.max(f32::EPSILON)).ln()
        });

        let mut mu_param = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut gamma_param = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut tau_param = GammaMatrix::new((n_genes, n_groups), self.a0, self.b0);

        let mut phi = match phi_mode {
            PhiMode::Free => DVec::from_element(n_genes, PHI_INIT),
            PhiMode::Fixed(p) => p.rows(row0, nrows).into_owned(),
        };
        let phi_free = matches!(phi_mode, PhiMode::Free);
        // delta's prior is Gamma(phi_d, phi_d) per gene; re-set whenever phi is
        // re-estimated (takes effect at the next update).
        let mut delta_param = GammaMatrix::with_row_prior((n_genes, n_indv), &phi, &phi);

        let mut tau_dx = Mat::from_element(n_genes, n_groups, 1.0);
        let mut delta_di = Mat::from_element(n_genes, n_indv, 1.0);
        if let Some((tau, delta)) = warm {
            tau_dx.copy_from(tau);
            delta_di.copy_from(delta);
        }
        let min_sweeps = if phi_free { PHI_UPDATE_EVERY } else { 2 };

        // Work buffers, allocated once.
        let mut denom_dp = Mat::zeros(n_genes, n_pb);
        let mut m_di = Mat::zeros(n_genes, n_indv);
        let mut lambda_di = Mat::zeros(n_genes, n_indv);
        let mut prev_tau = tau_dx.clone();
        let mut prev_delta = delta_di.clone();
        let mut phi_fits = 0usize;

        // Copy `src` into `dst` with column p scaled by n(p).
        let scale_by_size = |dst: &mut Mat, src: &Mat| {
            dst.copy_from(src);
            for (mut col, &n) in dst.column_iter_mut().zip(size_p.iter()) {
                col *= n;
            }
        };

        for iter in 0..self.n_opt_iter {
            // mu(d,p): (y1 + y0) / ( sum_i tau(d,x(i)) delta(d,i) n(i,p) + gamma(d,p) n(p) )
            let tau_delta_di = tau_dx
                .select_columns(indv_to_group)
                .component_mul(&delta_di);
            scale_by_size(&mut denom_dp, gamma_param.posterior_mean());
            denom_dp.gemm(1.0, &tau_delta_di, size_ip, 1.0);
            mu_param.update_stat(&y10_dp, &denom_dp);
            mu_param.calibrate_with(CalibrateTarget::MeanOnly);
            let mu_dp = mu_param.posterior_mean();

            // gamma(d,p): y0 / ( mu(d,p) n(p) )
            scale_by_size(&mut denom_dp, mu_dp);
            gamma_param.update_stat(&y0_dp, &denom_dp);
            gamma_param.calibrate_with(CalibrateTarget::MeanOnly);

            // m(d,i) = sum_p mu(d,p) n(i,p)
            mu_dp.mul_to(&size_ip_t, &mut m_di);

            // tau(d,x): sum_{i in x} y1(d,i) / sum_{i in x} delta(d,i) m(d,i)
            let den_dx = delta_di.component_mul(&m_di) * &g_ig;
            tau_param.update_stat(&num_dx, &den_dx);
            tau_param.calibrate_with(CalibrateTarget::MeanOnly);
            tau_dx.copy_from(tau_param.posterior_mean());

            // delta(d,i) ~ Gamma(phi_d, phi_d): (y1 + phi) / (tau m + phi)
            lambda_di.copy_from(&tau_dx.select_columns(indv_to_group));
            lambda_di.component_mul_assign(&m_di);
            delta_param.update_stat(&y1_di, &lambda_di);
            delta_param.calibrate_with(CalibrateTarget::MeanOnly);
            delta_di.copy_from(delta_param.posterior_mean());

            let sweeps = iter + 1;
            let converged = sweeps >= min_sweeps
                && max_rel_change(&tau_dx, &prev_tau).max(max_rel_change(&delta_di, &prev_delta))
                    < CONVERGENCE_TOL;
            if phi_free
                && (sweeps % PHI_UPDATE_EVERY == 0 || converged || sweeps == self.n_opt_iter)
            {
                // Interim: block-median shrinkage keeps the trajectory stable
                // until the global prior exists.
                let fit = profile_dispersion(
                    &y1_di,
                    &lambda_di,
                    indv_to_group,
                    n_groups,
                    &log_mean,
                    &phi,
                    phi_fits == 0,
                    true,
                );
                phi = shrink_to_block_median(&fit, &phi);
                delta_param.set_row_prior(&phi, &phi);
                phi_fits += 1;
            }
            if converged {
                break;
            }
            prev_tau.copy_from(&tau_dx);
            prev_delta.copy_from(&delta_di);
        }

        // Pass-1 evidence: one full-bracket adjusted profile at the converged fit.
        let disp = if phi_free {
            Some(profile_dispersion(
                &y1_di,
                &lambda_di,
                indv_to_group,
                n_groups,
                &log_mean,
                &phi,
                true,
                true,
            ))
        } else {
            None
        };

        tau_param.calibrate();
        delta_param.calibrate();

        Ok(BlockFit {
            exposure: tau_param,
            indv_delta: delta_param,
            phi,
            disp,
            tau_dx,
            delta_di,
        })
    }
}

/// Largest |a - b| / (|b| + 1e-8) over all entries.
fn max_rel_change(a: &Mat, b: &Mat) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (y.abs() + 1e-8))
        .fold(0f32, f32::max)
}

/// Stack gene-block fits of one topic back into one output.
fn concat_blocks(blocks: Vec<BlockFit>) -> CocoaGroupOut {
    let n_genes: usize = blocks.iter().map(|b| b.phi.len()).sum();
    let dispersion =
        DVec::from_iterator(n_genes, blocks.iter().flat_map(|b| b.phi.iter().cloned()));
    let (exposure, indv_delta): (Vec<_>, Vec<_>) = blocks
        .into_iter()
        .map(|b| (b.exposure, b.indv_delta))
        .unzip();
    let indv_delta = GammaMatrix::vconcat(indv_delta, true);
    debug_assert!(
        indv_delta
            .row_prior()
            .is_some_and(|(a, _)| a == &dispersion),
        "stacked delta prior must equal the stacked dispersion"
    );
    CocoaGroupOut {
        exposure: GammaMatrix::vconcat(exposure, true),
        indv_delta,
        dispersion,
        dispersion_fit: None,
        dispersion_prior: None,
    }
}

impl DispersionFit {
    /// Row-stack per-block evidence in gene order.
    pub fn vconcat(blocks: Vec<DispersionFit>) -> Self {
        let n: usize = blocks.iter().map(|b| b.n.len()).sum();
        let cat = |f: &dyn Fn(&DispersionFit) -> &DVec| {
            DVec::from_iterator(n, blocks.iter().flat_map(|b| f(b).iter().cloned()))
        };
        DispersionFit {
            log_phi_hat: cat(&|b| &b.log_phi_hat),
            at_bound: blocks
                .iter()
                .flat_map(|b| b.at_bound.iter().cloned())
                .collect(),
            n: blocks.iter().flat_map(|b| b.n.iter().cloned()).collect(),
            log_mean: cat(&|b| &b.log_mean),
        }
    }

    /// Genes that carry usable evidence for the across-gene fit.
    fn informative(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n.len()).filter(move |&d| {
            self.n[d] >= 3
                && self.log_phi_hat[d].is_finite()
                && !self.at_bound[d]
                && self.log_mean[d].is_finite()
        })
    }
}

impl DispersionPrior {
    /// Weighted log-linear trend of `log_phi_hat` on `log_mean` (weights `n`)
    /// over the informative genes; a constant (their median) below
    /// `TREND_MIN_GENES` genes or without spread in `log_mean`.
    pub fn fit(fit: &DispersionFit) -> Self {
        let genes: Vec<usize> = fit.informative().collect();
        let n_fit = genes.len();
        let (a, b) = if n_fit < TREND_MIN_GENES {
            let center = if n_fit == 0 {
                PHI_INIT.ln()
            } else {
                let v: Vec<f32> = genes.iter().map(|&d| fit.log_phi_hat[d]).collect();
                median(&v)
            };
            (center, 0.0)
        } else {
            let (mut sw, mut sx, mut sy) = (0f64, 0f64, 0f64);
            for &d in &genes {
                let w = fit.n[d] as f64;
                sw += w;
                sx += w * f64::from(fit.log_mean[d]);
                sy += w * f64::from(fit.log_phi_hat[d]);
            }
            let (xbar, ybar) = (sx / sw, sy / sw);
            let (mut sxx, mut sxy) = (0f64, 0f64);
            for &d in &genes {
                let w = fit.n[d] as f64;
                let dx = f64::from(fit.log_mean[d]) - xbar;
                sxx += w * dx * dx;
                sxy += w * dx * (f64::from(fit.log_phi_hat[d]) - ybar);
            }
            if sxx > 0.0 {
                let b = sxy / sxx;
                ((ybar - b * xbar) as f32, b as f32)
            } else {
                (ybar as f32, 0.0)
            }
        };
        DispersionPrior { a, b, n_fit }
    }

    /// Trend value at a gene's log mean, kept inside the search bracket.
    pub fn log_phi_trend(&self, log_mean: f32) -> f32 {
        (self.a + self.b * log_mean).clamp(PHI_MIN.ln(), PHI_MAX.ln())
    }

    /// phi per gene: the trend at each gene's own mean.
    pub fn trend_phi(&self, fit: &DispersionFit) -> DVec {
        DVec::from_fn(fit.n.len(), |d, _| {
            self.log_phi_trend(fit.log_mean[d]).exp()
        })
    }
}

/// Interim shrinkage inside pass 1: pull each gene's log phi toward the
/// block median with `PHI_SHRINK_DF` pseudo-individuals.
fn shrink_to_block_median(fit: &DispersionFit, prev: &DVec) -> DVec {
    let fitted: Vec<f32> = fit
        .log_phi_hat
        .iter()
        .cloned()
        .filter(|x| x.is_finite())
        .collect();
    if fitted.is_empty() {
        return prev.clone();
    }
    let center = median(&fitted);
    DVec::from_fn(fit.n.len(), |d, _| {
        let lp = fit.log_phi_hat[d];
        if lp.is_finite() {
            let n = fit.n[d] as f32;
            ((n * lp + PHI_SHRINK_DF * center) / (n + PHI_SHRINK_DF)).exp()
        } else {
            prev[d]
        }
    })
}

/// Negative-binomial profile log-likelihood of one gene's per-individual
/// totals given their Poisson means, as a function of `phi`, up to terms
/// that do not involve `phi`. Only `active` individuals count. With
/// `cox_reid`, subtracts half the log information of each group's fitted
/// mean, `I_x = sum_{i in x} lambda_i phi / (lambda_i + phi)`.
fn nb_cr_profile_loglik(
    y: &[f32],
    lambda: &[f32],
    active: &[bool],
    group: &[usize],
    n_groups: usize,
    phi: f32,
    cox_reid: bool,
) -> f64 {
    let phi = f64::from(phi);
    let mut ll = 0f64;
    let mut n = 0f64;
    let mut info = vec![0f64; n_groups];
    for i in 0..y.len() {
        if !active[i] {
            continue;
        }
        let (yi, li) = (f64::from(y[i]), f64::from(lambda[i]));
        ll += SpecialGamma::ln_gamma(yi + phi).0 - (yi + phi) * (li + phi).ln();
        info[group[i]] += li * phi / (li + phi);
        n += 1.0;
    }
    ll += n * (phi * phi.ln() - SpecialGamma::ln_gamma(phi).0);
    if cox_reid {
        for ix in info.into_iter().filter(|&v| v > 0.0) {
            ll -= 0.5 * ix.ln();
        }
    }
    ll
}

/// Golden-section maximum of `f` over `[lo, hi]`.
fn golden_max(f: impl Fn(f32) -> f64, lo: f32, hi: f32, n_iter: usize) -> f32 {
    let r = 0.618_034f32;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - r * (b - a);
    let mut d = a + r * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..n_iter {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// Per-gene profile maximum of log phi. Individuals count only with a positive expected total and in a
/// group holding at least two such individuals. The first fit searches the
/// full bracket; later ones a window around `prev`. Serial over genes: the
/// caller parallelizes over gene blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn profile_dispersion(
    y1_di: &Mat,
    lambda_di: &Mat,
    group: &[usize],
    n_groups: usize,
    log_mean: &DVec,
    prev: &DVec,
    full_bracket: bool,
    cox_reid: bool,
) -> DispersionFit {
    let n_genes = y1_di.nrows();
    let n_indv = y1_di.ncols();
    let (lo, hi) = (PHI_MIN.ln(), PHI_MAX.ln());
    // Gene-contiguous copies so each gene's individuals are one slice.
    let y_id = y1_di.transpose();
    let l_id = lambda_di.transpose();

    let mut log_phi_hat = DVec::from_element(n_genes, f32::NAN);
    let mut at_bound = vec![false; n_genes];
    let mut n_out = vec![0usize; n_genes];
    let mut counts = vec![0usize; n_groups];
    let mut active = vec![false; n_indv];

    for d in 0..n_genes {
        let y = y_id.column(d);
        let l = l_id.column(d);
        counts.iter_mut().for_each(|c| *c = 0);
        for i in 0..n_indv {
            if l[i] > 0.0 {
                counts[group[i]] += 1;
            }
        }
        let mut n = 0usize;
        for i in 0..n_indv {
            active[i] = l[i] > 0.0 && counts[group[i]] >= 2;
            n += active[i] as usize;
        }
        n_out[d] = n;
        if n < 2 {
            continue;
        }
        let (y, l) = (y.as_slice(), l.as_slice());
        let f = |lp: f32| nb_cr_profile_loglik(y, l, &active, group, n_groups, lp.exp(), cox_reid);
        let lp_hat = if full_bracket {
            golden_max(f, lo, hi, PHI_SEARCH_STEPS)
        } else {
            let c = prev[d].ln();
            golden_max(
                f,
                (c - PHI_WARM_HALF_WIDTH).max(lo),
                (c + PHI_WARM_HALF_WIDTH).min(hi),
                PHI_WARM_STEPS,
            )
        };
        log_phi_hat[d] = lp_hat;
        at_bound[d] = lp_hat - BOUND_MARGIN <= lo || lp_hat + BOUND_MARGIN >= hi;
    }

    DispersionFit {
        log_phi_hat,
        at_bound,
        n: n_out,
        log_mean: log_mean.clone(),
    }
}

/// Contrast of group `g1` against group `g0`: mean over topics of
/// log tau(g1) - log tau(g0). Its null distribution comes from exposure-label
/// permutation over individuals (see `run_diff`), not from a formula.
pub fn compute_group_contrast(parameters: &[CocoaGroupOut], g1: usize, g0: usize) -> Vec<f32> {
    let n_topics = parameters.len();
    let n_genes = parameters[0].exposure.posterior_log_mean().nrows();
    let mut contrast = vec![0f32; n_genes];
    let k_f = n_topics as f32;
    for param in parameters {
        let log_tau = param.exposure.posterior_log_mean();
        for g in 0..n_genes {
            contrast[g] += (log_tau[(g, g1)] - log_tau[(g, g0)]) / k_f;
        }
    }
    contrast
}
