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
//! Coordinate updates per topic: Gamma-Poisson conjugate steps for `mu`,
//! `gamma`, `tau`, and `delta`; `phi` by maximizing the negative-binomial
//! marginal of each individual's total given `tau * mu`, shrunk toward the
//! median across genes.

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
    /// phi: between-individual dispersion per gene
    pub dispersion: DVec,
}

const PHI_MIN: f32 = 1e-2;
const PHI_MAX: f32 = 1e4;
const PHI_INIT: f32 = 10.0;
/// Re-estimate phi after this many coordinate sweeps.
const PHI_UPDATE_EVERY: usize = 10;
/// Golden-section steps over log phi on the full bracket (first fit) and on a
/// bracket of +/- `PHI_WARM_HALF_WIDTH` around the previous value (refits).
const PHI_SEARCH_STEPS: usize = 25;
const PHI_WARM_STEPS: usize = 18;
const PHI_WARM_HALF_WIDTH: f32 = 3.0;
/// Pseudo-individuals pulling each gene's log phi toward the median of its block.
const PHI_SHRINK_DF: f32 = 4.0;
/// Genes per independent fit; every update is row-separable, so blocks are the
/// unit of parallelism and bound memory to one block per core. The dispersion
/// shrinkage center is the median within the block.
const GENE_BLOCK: usize = 512;
/// Stop sweeping once the largest relative change in tau and delta falls
/// below this, after at least `PHI_UPDATE_EVERY` sweeps.
const CONVERGENCE_TOL: f32 = 1e-4;

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

        let jobs: Vec<(usize, usize, usize)> = (0..self.n_topics)
            .flat_map(|k| {
                (0..n_genes)
                    .step_by(block)
                    .map(move |row0| (k, row0, block.min(n_genes - row0)))
            })
            .collect();

        let fits: Result<Vec<CocoaGroupOut>, _> = jobs
            .into_par_iter()
            .map(|(k, row0, nrows)| {
                self.optimize_group_block(k, row0, nrows, indv_to_group, n_groups)
            })
            .collect();
        let mut fits = fits?.into_iter();

        let n_blocks = n_genes.div_ceil(block);
        let mut out = Vec::with_capacity(self.n_topics);
        for _ in 0..self.n_topics {
            let blocks: Vec<CocoaGroupOut> = fits.by_ref().take(n_blocks).collect();
            out.push(concat_blocks(blocks));
        }
        info!("finished group-model optimization for {} topics", self.n_topics);
        Ok(out)
    }

    /// Fit genes `row0..row0 + nrows` of topic `k`.
    fn optimize_group_block(
        &self,
        k: usize,
        row0: usize,
        nrows: usize,
        indv_to_group: &[usize],
        n_groups: usize,
    ) -> anyhow::Result<CocoaGroupOut> {
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
        let g_ig = Mat::from_fn(n_indv, n_groups, |i, x| (indv_to_group[i] == x) as u8 as f32);
        let num_dx = &y1_di * &g_ig;
        let size_ip_t = size_ip.transpose();

        let mut mu_param = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut gamma_param = GammaMatrix::new((n_genes, n_pb), self.a0, self.b0);
        let mut tau_param = GammaMatrix::new((n_genes, n_groups), self.a0, self.b0);
        // delta's prior Gamma(phi_d, phi_d) is per gene, so it is added to the
        // statistics by hand below instead of through the scalar (a0, b0).
        let mut delta_param = GammaMatrix::new((n_genes, n_indv), 0.0, 0.0);

        let mut phi = DVec::from_element(n_genes, PHI_INIT);
        let mut phi_fits = 0usize;
        let mut tau_dx = Mat::from_element(n_genes, n_groups, 1.0);
        let mut delta_di = Mat::from_element(n_genes, n_indv, 1.0);

        // Work buffers, allocated once.
        let mut denom_dp = Mat::zeros(n_genes, n_pb);
        let mut m_di = Mat::zeros(n_genes, n_indv);
        let mut lambda_di = Mat::zeros(n_genes, n_indv);
        let mut num_di = Mat::zeros(n_genes, n_indv);
        let mut den_di = Mat::zeros(n_genes, n_indv);
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
            let tau_delta_di = tau_dx.select_columns(indv_to_group).component_mul(&delta_di);
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
            num_di.copy_from(&y1_di);
            den_di.copy_from(&lambda_di);
            for d in 0..n_genes {
                num_di.row_mut(d).add_scalar_mut(phi[d]);
                den_di.row_mut(d).add_scalar_mut(phi[d]);
            }
            delta_param.update_stat(&num_di, &den_di);
            delta_param.calibrate_with(CalibrateTarget::MeanOnly);
            delta_di.copy_from(delta_param.posterior_mean());

            let sweeps = iter + 1;
            let converged = sweeps >= PHI_UPDATE_EVERY
                && max_rel_change(&tau_dx, &prev_tau).max(max_rel_change(&delta_di, &prev_delta))
                    < CONVERGENCE_TOL;
            if sweeps % PHI_UPDATE_EVERY == 0 || converged || sweeps == self.n_opt_iter {
                phi = estimate_dispersion(&y1_di, &lambda_di, &phi, phi_fits == 0);
                phi_fits += 1;
            }
            if converged {
                break;
            }
            prev_tau.copy_from(&tau_dx);
            prev_delta.copy_from(&delta_di);
        }

        tau_param.calibrate();
        delta_param.calibrate();

        Ok(CocoaGroupOut {
            exposure: tau_param,
            indv_delta: delta_param,
            dispersion: phi,
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
fn concat_blocks(mut blocks: Vec<CocoaGroupOut>) -> CocoaGroupOut {
    if blocks.len() == 1 {
        return blocks.pop().expect("one block");
    }
    let n_genes: usize = blocks.iter().map(|b| b.dispersion.len()).sum();
    let dispersion = DVec::from_iterator(
        n_genes,
        blocks.iter().flat_map(|b| b.dispersion.iter().cloned()),
    );
    let (exposure, indv_delta): (Vec<_>, Vec<_>) = blocks
        .into_iter()
        .map(|b| (b.exposure, b.indv_delta))
        .unzip();
    CocoaGroupOut {
        exposure: GammaMatrix::vconcat(exposure, true),
        indv_delta: GammaMatrix::vconcat(indv_delta, true),
        dispersion,
    }
}

/// Negative-binomial marginal log-likelihood of one gene's per-individual
/// totals given their Poisson means, as a function of `phi`, up to terms
/// that do not involve `phi`. Pairs with a non-positive mean are skipped.
fn nb_profile_loglik(y: &[f32], lambda: &[f32], phi: f32) -> f64 {
    let phi = f64::from(phi);
    let mut ll = 0f64;
    let mut n = 0f64;
    for (&yi, &li) in y.iter().zip(lambda.iter()) {
        if li <= 0.0 {
            continue;
        }
        let (yi, li) = (f64::from(yi), f64::from(li));
        ll += SpecialGamma::ln_gamma(yi + phi).0 - (yi + phi) * (li + phi).ln();
        n += 1.0;
    }
    ll + n * (phi * phi.ln() - SpecialGamma::ln_gamma(phi).0)
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

/// Per-gene dispersion: maximize the NB marginal over log phi, then pull
/// each gene's log phi toward the median across the block's genes with
/// `PHI_SHRINK_DF` pseudo-individuals. The first fit searches the full
/// bracket; refits search around the previous value. Genes with fewer than
/// two informative individuals keep their previous value. Serial over genes:
/// the caller already parallelizes over gene blocks.
fn estimate_dispersion(y1_di: &Mat, lambda_di: &Mat, prev: &DVec, full_bracket: bool) -> DVec {
    let n_genes = y1_di.nrows();
    let (lo, hi) = (PHI_MIN.ln(), PHI_MAX.ln());
    // Gene-contiguous copies so each gene's individuals are one slice.
    let y_id = y1_di.transpose();
    let l_id = lambda_di.transpose();

    let fits: Vec<Option<(f32, usize)>> = (0..n_genes)
        .map(|d| {
            let y = y_id.column(d);
            let l = l_id.column(d);
            let n = l.iter().filter(|&&v| v > 0.0).count();
            if n < 2 {
                return None;
            }
            let (y, l) = (y.as_slice(), l.as_slice());
            let f = |lp: f32| nb_profile_loglik(y, l, lp.exp());
            let log_phi = if full_bracket {
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
            Some((log_phi, n))
        })
        .collect();

    let fitted: Vec<f32> = fits.iter().flatten().map(|(lp, _)| *lp).collect();
    if fitted.is_empty() {
        return prev.clone();
    }
    let center = median(&fitted);

    DVec::from_fn(n_genes, |d, _| match fits[d] {
        Some((lp, n)) => {
            let n = n as f32;
            ((n * lp + PHI_SHRINK_DF * center) / (n + PHI_SHRINK_DF)).exp()
        }
        None => prev[d],
    })
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
