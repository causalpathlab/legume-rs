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
//! two-parameter trend cannot chase per-gene noise.
//!
//! Every update is row-separable, so genes are fit in independent blocks:
//! pass 1 refits each gene's own phi from its profile every few sweeps (so
//! delta is shrunk with the right strength while mu and tau settle), one
//! final profile per gene at that fixed point, the global trend, then pass 2
//! warm-started from pass 1 with phi fixed at the trend.

use super::*;
use legume_numeric::matrix::utils::median;
use special::Gamma as SpecialGamma;

#[cfg(test)]
mod tests;

/// Fitted group model for one topic. `mu` and `gamma` are nuisance
/// parameters of the fit and are not retained.
pub struct CocoaGroupOut {
    /// tau: average exposure effect, gene x exposure group
    pub exposure: GammaMatrix,
    /// delta: individual effect with exposure removed, gene x individual
    pub indv_delta: GammaMatrix,
    /// phi: between-individual dispersion per gene (the trend at its mean)
    pub dispersion: DVec,
    /// log mean count per cell per gene, the trend's covariate
    pub log_mean: DVec,
    /// The fitted across-gene trend; `None` when phi was supplied.
    pub dispersion_prior: Option<DispersionPrior>,
}

/// Dispersion evidence for one gene.
#[derive(Clone, Copy, Debug)]
pub struct GeneEvidence {
    /// Cox-Reid adjusted profile maximum of log phi; NaN when the gene has no
    /// usable evidence (too few informative individuals, or a maximum at the
    /// edge of the search bracket).
    pub log_phi_hat: f32,
    /// Informative individuals: those with cells, in a group holding at
    /// least two such individuals.
    pub n: usize,
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
/// Starting phi for pass 1, before the first per-gene refit.
const PHI_INIT: f32 = 10.0;
/// Pass 1 refits each gene's phi from its profile after this many sweeps.
const PHI_UPDATE_EVERY: usize = 10;
/// Golden-section steps over log phi: the bracket of 13.8 log units shrinks
/// to about 6e-3, finer than the trend regression can use.
const PHI_SEARCH_STEPS: usize = 16;
/// Fewer genes than this fall back to a constant trend.
const TREND_MIN_GENES: usize = 10;
/// Genes per independent fit: the unit of parallelism, bounding memory to
/// one block per core.
const GENE_BLOCK: usize = 512;
/// Stop sweeping once the largest relative change in tau and delta falls
/// below this.
const CONVERGENCE_TOL: f32 = 1e-4;

/// Which individuals inform the dispersion profile, laid out by group so the
/// objective runs over contiguous slices. Shared by every gene of a topic:
/// an individual is informative iff it has cells and its group holds at
/// least two such individuals.
struct ActiveLayout {
    /// Informative individuals, sorted by group.
    order: Vec<usize>,
    /// `order[bounds[x]..bounds[x + 1]]` are group `x`'s individuals.
    bounds: Vec<usize>,
}

impl ActiveLayout {
    fn new(has_cells: &[bool], group: &[usize], n_groups: usize) -> Self {
        let mut counts = vec![0usize; n_groups];
        for (i, &g) in group.iter().enumerate() {
            counts[g] += has_cells[i] as usize;
        }
        let mut order = Vec::with_capacity(group.len());
        let mut bounds = Vec::with_capacity(n_groups + 1);
        for (x, &count) in counts.iter().enumerate() {
            bounds.push(order.len());
            if count >= 2 {
                order.extend((0..group.len()).filter(|&i| group[i] == x && has_cells[i]));
            }
        }
        bounds.push(order.len());
        ActiveLayout { order, bounds }
    }

    fn n(&self) -> usize {
        self.order.len()
    }
}

/// One (topic, first row, rows) unit of work.
#[derive(Clone, Copy)]
struct BlockJob {
    topic: usize,
    row0: usize,
    nrows: usize,
}

/// Posterior means carried from one pass to the next as a warm start.
struct BlockState {
    tau_dx: Mat,
    delta_di: Mat,
    gamma_dp: Mat,
}

/// One gene block of one topic after a fit.
struct BlockFit {
    exposure: GammaMatrix,
    indv_delta: GammaMatrix,
    lambda_di: Mat,
    state: BlockState,
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
        let init: Vec<DVec> = (0..self.n_topics)
            .map(|_| DVec::from_element(n_genes, PHI_INIT))
            .collect();

        let layouts: Vec<ActiveLayout> = (0..self.n_topics)
            .map(|k| self.active_layout(k, indv_to_group, n_groups))
            .collect();

        // Pass 1: each gene's phi refit from its own profile as the fit settles.
        let pass1 = self.fit_pass(&jobs, indv_to_group, n_groups, &init, Some(&layouts), None)?;

        // Global step per topic: profile every gene, fit the trend, set phi.
        let mut priors = Vec::with_capacity(self.n_topics);
        let mut log_means = Vec::with_capacity(self.n_topics);
        let mut phis = Vec::with_capacity(self.n_topics);
        for (k, layout) in layouts.iter().enumerate() {
            let y1_di = self.indv_y1_stat(k);
            let mut evidence = Vec::with_capacity(n_genes);
            for (job, fit) in jobs.iter().zip(pass1.iter()).filter(|(j, _)| j.topic == k) {
                let y1 = y1_di.rows(job.row0, job.nrows);
                evidence.extend(profile_dispersion(
                    &y1.into_owned(),
                    &fit.lambda_di,
                    layout,
                    true,
                ));
            }
            let log_mean = self.log_mean_per_cell(k);
            let prior = DispersionPrior::fit(&evidence, &log_mean);
            info!(
                "topic {}: dispersion trend log phi = {:.3} + {:.3} log mean ({} genes)",
                k, prior.a, prior.b, prior.n_fit
            );
            phis.push(prior.trend_phi(&log_mean));
            priors.push(prior);
            log_means.push(log_mean);
        }

        // Pass 2: phi fixed at the trend, warm-started from pass 1.
        let pass2 = self.fit_pass(&jobs, indv_to_group, n_groups, &phis, None, Some(&pass1))?;
        let mut out = assemble(pass2, n_blocks, phis, log_means);
        for (topic, prior) in out.iter_mut().zip(priors) {
            topic.dispersion_prior = Some(prior);
        }
        info!(
            "finished group-model optimization for {} topics",
            self.n_topics
        );
        Ok(out)
    }

    /// Fit with phi supplied per topic (gene-length vectors) and never
    /// estimated: the oracle path used by the recovery tests.
    #[allow(dead_code)]
    fn fit_with_dispersion(
        &self,
        indv_to_group: &[usize],
        n_groups: usize,
        block: usize,
        phi: Vec<DVec>,
    ) -> anyhow::Result<Vec<CocoaGroupOut>> {
        let jobs = self.block_jobs(indv_to_group, n_groups, block)?;
        let n_genes = self.y1_stat(0).nrows();
        anyhow::ensure!(
            phi.len() == self.n_topics && phi.iter().all(|p| p.len() == n_genes),
            "phi must hold one vector of {} genes per topic",
            n_genes
        );
        let fits = self.fit_pass(&jobs, indv_to_group, n_groups, &phi, None, None)?;
        let log_means = (0..self.n_topics)
            .map(|k| self.log_mean_per_cell(k))
            .collect();
        Ok(assemble(fits, n_genes.div_ceil(block), phi, log_means))
    }

    /// (topic, first row, rows) for every block, after validating the design.
    fn block_jobs(
        &self,
        indv_to_group: &[usize],
        n_groups: usize,
        block: usize,
    ) -> anyhow::Result<Vec<BlockJob>> {
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
            .flat_map(|topic| {
                (0..n_genes).step_by(block).map(move |row0| BlockJob {
                    topic,
                    row0,
                    nrows: block.min(n_genes - row0),
                })
            })
            .collect())
    }

    /// One pass over all blocks in parallel, with phi per topic. With
    /// `refit` layouts, phi is refit per gene as the fit settles; `warm`
    /// seeds each block from a previous pass.
    fn fit_pass(
        &self,
        jobs: &[BlockJob],
        indv_to_group: &[usize],
        n_groups: usize,
        phi: &[DVec],
        refit: Option<&[ActiveLayout]>,
        warm: Option<&[BlockFit]>,
    ) -> anyhow::Result<Vec<BlockFit>> {
        jobs.par_iter()
            .enumerate()
            .map(|(j, &job)| {
                let phi_block = phi[job.topic].rows(job.row0, job.nrows).into_owned();
                let layout = refit.map(|l| &l[job.topic]);
                let state = warm.map(|w| &w[j].state);
                self.fit_block(job, indv_to_group, n_groups, phi_block, layout, state)
            })
            .collect()
    }

    /// Which individuals inform the profile in topic `k`.
    fn active_layout(&self, k: usize, indv_to_group: &[usize], n_groups: usize) -> ActiveLayout {
        let cells_per_indv = self.indv_size_stat(k).column_sum();
        let has_cells: Vec<bool> = cells_per_indv.iter().map(|&n| n > 0.0).collect();
        ActiveLayout::new(&has_cells, indv_to_group, n_groups)
    }

    /// log of the mean count per cell per gene in topic `k`, from the data
    /// alone (half a count is added so unexpressed genes stay finite).
    fn log_mean_per_cell(&self, k: usize) -> DVec {
        let total_cells = self.indv_size_stat(k).sum().max(f32::EPSILON);
        self.indv_y1_stat(k)
            .column_sum()
            .map(|y| ((y + 0.5) / total_cells).ln())
    }

    /// Fit one gene block. With a `refit` layout, each gene's phi is refit
    /// from its profile every `PHI_UPDATE_EVERY` sweeps; otherwise phi stays
    /// as given. `warm` seeds tau, delta, and gamma.
    fn fit_block(
        &self,
        job: BlockJob,
        indv_to_group: &[usize],
        n_groups: usize,
        mut phi: DVec,
        refit: Option<&ActiveLayout>,
        warm: Option<&BlockState>,
    ) -> anyhow::Result<BlockFit> {
        let BlockJob {
            topic: k,
            row0,
            nrows,
        } = job;
        let y1_dp = self.y1_stat(k).rows(row0, nrows).into_owned();
        let y0_dp = self.y0_stat(k).rows(row0, nrows).into_owned();
        let y10_dp = &y1_dp + &y0_dp;
        let y1_di = self.indv_y1_stat(k).rows(row0, nrows).into_owned();
        let size_p = self.size_stat(k);
        let size_ip = self.indv_size_stat(k);

        let n_genes = nrows;
        let n_pb = y1_dp.ncols();
        let n_indv = y1_di.ncols();

        // Indicator (individual x group) and its transpose: group sums and
        // the gather of tau by individual, both as small matmuls.
        let g_ig = Mat::from_fn(n_indv, n_groups, |i, x| {
            (indv_to_group[i] == x) as u8 as f32
        });
        let g_gi = g_ig.transpose();
        let num_dx = &y1_di * &g_ig;
        let size_ip_t = size_ip.transpose();

        let mut mu_param = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut gamma_param = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut tau_param = GammaMatrix::new((n_genes, n_groups), self.a0, self.b0);
        // delta's prior is Gamma(phi_d, phi_d) per gene, through the row prior.
        let mut delta_param = GammaMatrix::with_row_prior((n_genes, n_indv), &phi, &phi);

        let (mut tau_dx, mut delta_di, mut gamma_dp) = match warm {
            Some(s) => (s.tau_dx.clone(), s.delta_di.clone(), s.gamma_dp.clone()),
            None => (
                Mat::from_element(n_genes, n_groups, 1.0),
                Mat::from_element(n_genes, n_indv, 1.0),
                Mat::zeros(n_genes, n_pb),
            ),
        };

        // Work buffers, allocated once.
        let mut denom_dp = Mat::zeros(n_genes, n_pb);
        let mut m_di = Mat::zeros(n_genes, n_indv);
        let mut tau_di = Mat::zeros(n_genes, n_indv);
        let mut work_di = Mat::zeros(n_genes, n_indv);
        let mut lambda_di = Mat::zeros(n_genes, n_indv);
        let mut den_dx = Mat::zeros(n_genes, n_groups);
        let mut prev_tau = tau_dx.clone();
        let mut prev_delta = delta_di.clone();

        // Copy `src` into `dst` with column p scaled by n(p).
        let scale_by_size = |dst: &mut Mat, src: &Mat| {
            dst.copy_from(src);
            for (mut col, &n) in dst.column_iter_mut().zip(size_p.iter()) {
                col *= n;
            }
        };

        for iter in 0..self.n_opt_iter {
            // mu(d,p): (y1 + y0) / ( sum_i tau(d,x(i)) delta(d,i) n(i,p) + gamma(d,p) n(p) )
            tau_dx.mul_to(&g_gi, &mut tau_di);
            work_di.cmpy(1.0, &tau_di, &delta_di, 0.0);
            scale_by_size(&mut denom_dp, &gamma_dp);
            denom_dp.gemm(1.0, &work_di, size_ip, 1.0);
            mu_param.update_stat(&y10_dp, &denom_dp);
            mu_param.calibrate_with(CalibrateTarget::MeanOnly);
            let mu_dp = mu_param.posterior_mean();

            // gamma(d,p): y0 / ( mu(d,p) n(p) )
            scale_by_size(&mut denom_dp, mu_dp);
            gamma_param.update_stat(&y0_dp, &denom_dp);
            gamma_param.calibrate_with(CalibrateTarget::MeanOnly);
            gamma_dp.copy_from(gamma_param.posterior_mean());

            // m(d,i) = sum_p mu(d,p) n(i,p)
            mu_dp.mul_to(&size_ip_t, &mut m_di);

            // tau(d,x): sum_{i in x} y1(d,i) / sum_{i in x} delta(d,i) m(d,i)
            work_di.cmpy(1.0, &delta_di, &m_di, 0.0);
            work_di.mul_to(&g_ig, &mut den_dx);
            tau_param.update_stat(&num_dx, &den_dx);
            tau_param.calibrate_with(CalibrateTarget::MeanOnly);
            tau_dx.copy_from(tau_param.posterior_mean());

            // delta(d,i) ~ Gamma(phi_d, phi_d): (y1 + phi) / (tau m + phi)
            tau_dx.mul_to(&g_gi, &mut tau_di);
            lambda_di.cmpy(1.0, &tau_di, &m_di, 0.0);
            delta_param.update_stat(&y1_di, &lambda_di);
            delta_param.calibrate_with(CalibrateTarget::MeanOnly);
            delta_di.copy_from(delta_param.posterior_mean());

            if let Some(layout) = refit.filter(|_| (iter + 1) % PHI_UPDATE_EVERY == 0) {
                // Shrink delta with each gene's own dispersion; genes without
                // evidence keep their current phi.
                for (d, e) in profile_dispersion(&y1_di, &lambda_di, layout, true)
                    .into_iter()
                    .enumerate()
                {
                    if e.log_phi_hat.is_finite() {
                        phi[d] = e.log_phi_hat.exp();
                    }
                }
                delta_param.set_row_prior(&phi, &phi);
            }

            if iter > 0 && converged(&tau_dx, &prev_tau) && converged(&delta_di, &prev_delta) {
                break;
            }
            prev_tau.copy_from(&tau_dx);
            prev_delta.copy_from(&delta_di);
        }

        tau_param.calibrate();
        delta_param.calibrate();

        Ok(BlockFit {
            exposure: tau_param,
            indv_delta: delta_param,
            lambda_di,
            state: BlockState {
                tau_dx,
                delta_di,
                gamma_dp,
            },
        })
    }
}

/// Every entry moved by less than `CONVERGENCE_TOL` relative to `prev`.
fn converged(now: &Mat, prev: &Mat) -> bool {
    now.iter()
        .zip(prev.iter())
        .all(|(x, y)| (x - y).abs() < CONVERGENCE_TOL * (y.abs() + 1e-8))
}

/// Stack block fits into one output per topic; `phi` and `log_mean` are the
/// per-topic vectors the blocks were fit under.
fn assemble(
    fits: Vec<BlockFit>,
    n_blocks: usize,
    phi: Vec<DVec>,
    log_mean: Vec<DVec>,
) -> Vec<CocoaGroupOut> {
    let mut fits = fits.into_iter();
    phi.into_iter()
        .zip(log_mean)
        .map(|(dispersion, log_mean)| {
            let (exposure, indv_delta): (Vec<_>, Vec<_>) = fits
                .by_ref()
                .take(n_blocks)
                .map(|b| (b.exposure, b.indv_delta))
                .unzip();
            // Only the posterior planes are read downstream.
            let indv_delta = GammaMatrix::vconcat(indv_delta, false);
            debug_assert!(
                indv_delta
                    .row_prior()
                    .is_some_and(|(a, _)| a == &dispersion),
                "stacked delta prior must equal the dispersion vector"
            );
            CocoaGroupOut {
                exposure: GammaMatrix::vconcat(exposure, false),
                indv_delta,
                dispersion,
                log_mean,
                dispersion_prior: None,
            }
        })
        .collect()
}

impl DispersionPrior {
    /// Weighted log-linear trend of `log_phi_hat` on `log_mean` (weights
    /// `n`) over genes with evidence; a constant (their median) below
    /// `TREND_MIN_GENES` genes or without spread in `log_mean`.
    pub fn fit(evidence: &[GeneEvidence], log_mean: &DVec) -> Self {
        let genes: Vec<usize> = (0..evidence.len())
            .filter(|&d| {
                evidence[d].n >= 3 && evidence[d].log_phi_hat.is_finite() && log_mean[d].is_finite()
            })
            .collect();
        let n_fit = genes.len();
        if n_fit < TREND_MIN_GENES {
            let a = if n_fit == 0 {
                PHI_INIT.ln()
            } else {
                let v: Vec<f32> = genes.iter().map(|&d| evidence[d].log_phi_hat).collect();
                median(&v)
            };
            return DispersionPrior { a, b: 0.0, n_fit };
        }
        let (mut sw, mut sx, mut sy) = (0f64, 0f64, 0f64);
        for &d in &genes {
            let w = evidence[d].n as f64;
            sw += w;
            sx += w * f64::from(log_mean[d]);
            sy += w * f64::from(evidence[d].log_phi_hat);
        }
        let (xbar, ybar) = (sx / sw, sy / sw);
        let (mut sxx, mut sxy) = (0f64, 0f64);
        for &d in &genes {
            let w = evidence[d].n as f64;
            let dx = f64::from(log_mean[d]) - xbar;
            sxx += w * dx * dx;
            sxy += w * dx * (f64::from(evidence[d].log_phi_hat) - ybar);
        }
        let (a, b) = if sxx > 0.0 {
            let b = sxy / sxx;
            ((ybar - b * xbar) as f32, b as f32)
        } else {
            (ybar as f32, 0.0)
        };
        DispersionPrior { a, b, n_fit }
    }

    /// Trend value at a gene's log mean, kept inside the search bracket.
    pub fn log_phi_trend(&self, log_mean: f32) -> f32 {
        (self.a + self.b * log_mean).clamp(PHI_MIN.ln(), PHI_MAX.ln())
    }

    /// phi per gene: the trend at each gene's own mean.
    pub fn trend_phi(&self, log_mean: &DVec) -> DVec {
        log_mean.map(|m| self.log_phi_trend(m).exp())
    }
}

/// Golden-section maximum of `f` over `[lo, hi]`; the flag says whether the
/// final bracket still touches an edge, i.e. the maximum was never enclosed.
fn golden_max(f: impl Fn(f32) -> f64, lo: f32, hi: f32, n_iter: usize) -> (f32, bool) {
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
    (0.5 * (a + b), a <= lo || b >= hi)
}

/// Per-gene profile maximum of log phi over the informative individuals in
/// `layout`, given per-individual totals `y1_di` and their expected values
/// `lambda_di`. With `cox_reid`, the profile subtracts half the log
/// information of each group's fitted mean, `I_x = sum_{i in x} lambda phi /
/// (lambda + phi)`. Serial over genes: the caller parallelizes over blocks.
fn profile_dispersion(
    y1_di: &Mat,
    lambda_di: &Mat,
    layout: &ActiveLayout,
    cox_reid: bool,
) -> Vec<GeneEvidence> {
    let n_genes = y1_di.nrows();
    let n = layout.n();
    let (lo, hi) = (PHI_MIN.ln(), PHI_MAX.ln());
    let mut y = vec![0f64; n];
    let mut l = vec![0f64; n];
    let mut out = Vec::with_capacity(n_genes);
    for d in 0..n_genes {
        if n < 2 {
            out.push(GeneEvidence {
                log_phi_hat: f32::NAN,
                n,
            });
            continue;
        }
        for (j, &i) in layout.order.iter().enumerate() {
            y[j] = f64::from(y1_di[(d, i)]);
            l[j] = f64::from(lambda_di[(d, i)]);
        }
        let f =
            |lp: f32| nb_cr_profile_loglik(&y, &l, &layout.bounds, f64::from(lp.exp()), cox_reid);
        let (lp_hat, at_edge) = golden_max(f, lo, hi, PHI_SEARCH_STEPS);
        let log_phi_hat = if at_edge { f32::NAN } else { lp_hat };
        out.push(GeneEvidence { log_phi_hat, n });
    }
    out
}

/// Negative-binomial profile log-likelihood of one gene's per-individual
/// totals `y` given their Poisson means `lambda`, laid out by group, as a
/// function of `phi`, up to terms that do not involve `phi`.
fn nb_cr_profile_loglik(
    y: &[f64],
    lambda: &[f64],
    bounds: &[usize],
    phi: f64,
    cox_reid: bool,
) -> f64 {
    let mut ll = y.len() as f64 * (phi * phi.ln() - SpecialGamma::ln_gamma(phi).0);
    for w in bounds.windows(2) {
        let mut info = 0f64;
        for j in w[0]..w[1] {
            ll += SpecialGamma::ln_gamma(y[j] + phi).0 - (y[j] + phi) * (lambda[j] + phi).ln();
            info += lambda[j] * phi / (lambda[j] + phi);
        }
        if cox_reid && info > 0.0 {
            ll -= 0.5 * info.ln();
        }
    }
    ll
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
