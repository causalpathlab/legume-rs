//! Hierarchical Gibbs sampler for bulk deconvolution.
//!
//! Model, per bulk sample `s` (rayon-parallel; independent seeded chains), over
//! `R` reference components with rates `μ_{g,m}`:
//! ```text
//!   ε_{s,g} ~ Gamma(r, r)                            (NB overdispersion, mean 1)
//!   y_{s,g} ~ Poisson(ε_{s,g}·Σ_m u_{s,m} μ_{g,m})
//!   Z_{s,·,g} ~ Multinomial(y_{s,g}, p), p_m ∝ u_{s,m} μ_{g,m}   (ε cancels in the split)
//!   ε_{s,g}   ~ Gamma(r + τy_{s,g}, r + τλ_{s,g})                (conjugate)
//!   u_{s,m}   ~ Gamma(a0 + τΣ_g Z_{s,m,g}, b0 + τΣ_g ε_{s,g}μ_{g,m})
//! ```
//! Reporting maps components onto cell types through the reference readout `A`:
//! `fraction_{s,c} = Σ_m u_{s,m} A_{m,c} / Σ_m u_{s,m}`.
//!
//! `r` (`--nb-dispersion`) is the negative-binomial dispersion, held fixed
//! (freely sampling it is non-identifiable against the fractions — ε competes
//! with u through the per-component exposure). `r → ∞` recovers Poisson. `τ`
//! (`--count-scale`) tempers the likelihood (power posterior): every count
//! sufficient statistic is scaled by τ, so the posterior reflects τ·(observed
//! counts) of evidence and its variance scales as 1/τ — the CI calibration knob
//! at high read depth.
//!
//! In the low-rank reference mode the components are the cell types themselves
//! and their anchors `t_c` are resampled by one elliptical-slice step per sweep
//! under the annotate-by-projection prior `N(t̂_c, Σ_c)`, with the pooled Poisson
//! likelihood
//! `ℓ(t_c) = τ·Σ_g[A_{c,g}(ρ_g·t_c+a_g) − W_{c,g}·exp(ρ_g·t_c+a_g)]`,
//! `A_{c,g}=Σ_s Z_{s,c,g}`, `W_{c,g}=Σ_s u_{s,c}·ε_{s,g}` (ε-weighted exposure).
//! This is the coupling that lets annotation uncertainty widen (or confident
//! types pin) the fraction posterior. The archetype reference is fixed, so that
//! step is skipped entirely.

use super::anchors::AnchorPrior;
use super::args::SamplerConfig;
use super::monitor::Monitor;
use super::reference::{Reference, SCORE_CLAMP};
use super::result::{residual_stat, split_rhat_ess, Convergence, DeconvResult};
use super::source::EmbeddingSource;
use crate::embed_common::{DVec, Mat};
use log::info;
use matrix_util::running_quantile::RunningQuantiles;
use mcmc_util::engine::elliptical_slice_step;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Binomial, Distribution, Gamma, StandardNormal};
use rayon::prelude::*;

/// Offset for the anchor-chain RNG seed (separate stream from the per-sample chains).
const ANCHOR_SEED_OFFSET: u64 = 0x9E37_79B9_7F4A_7C15;
/// Floor on a sampled abundance, so a zeroed component can still recover.
const ABUNDANCE_FLOOR: f32 = 1e-12;

/// Sweep-invariant inputs shared across all samples.
struct SweepCtx<'a> {
    mu_gm: &'a [f32],
    readout: &'a Mat,
    d: usize,
    r: usize,
    c: usize,
    a0: f32,
    b0: f32,
    /// NB dispersion `r` and likelihood temperature `τ`.
    nb_r: f32,
    tau: f32,
    /// Materialise the full `R×D` allocation (the anchor ESS needs it).
    keep_contrib: bool,
    /// Accumulate the `D×C` expression contribution this sweep.
    collect_expr: bool,
}

/// Per-sample chain state and scratch, allocated once and reused every sweep.
///
/// The `R×D` allocation block is materialised only when the anchor sampler
/// needs it — that mode has `R = C`, a handful of components. The archetype
/// mode contracts counts through the readout to `C` inside the gene loop
/// instead: at `R = 500` and `D = 25 000` the block would be 50 MB per sample.
struct SampleState {
    rng: SmallRng,
    /// `R` current abundances `u`.
    w: Vec<f32>,
    /// `R` allocated count totals this sweep.
    nvec: Vec<f32>,
    /// `R` ε-weighted exposure `Σ_g ε_g μ_{g,m}`.
    m_exp: Vec<f32>,
    /// `R` per-gene component rates, and their running cumulative sum.
    rr: Vec<f32>,
    cum: Vec<f32>,
    /// `R` per-gene allocated counts, with the touched components listed so the
    /// buffer is cleared in O(touched) rather than O(R).
    counts: Vec<f32>,
    touched: Vec<u32>,
    /// `D` per-gene NB factor.
    eps: Vec<f32>,
    /// `D×C` allocation contracted through the readout, laid out `ct*D + g`.
    zc: Vec<f32>,
    /// `R×D` full allocation, laid out `m*D + g`. Empty unless `keep_contrib`.
    contrib: Vec<f32>,
    /// `C` fractions this sweep.
    frac: Vec<f32>,
}

impl SampleState {
    fn new(seed: u64, r: usize, c: usize, d: usize, keep_contrib: bool) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            w: vec![1.0; r],
            nvec: vec![0.0; r],
            m_exp: vec![0.0; r],
            rr: vec![0.0; r],
            cum: vec![0.0; r],
            counts: vec![0.0; r],
            touched: Vec::with_capacity(r.min(64)),
            eps: vec![0.0; d],
            zc: vec![0.0; c * d],
            contrib: if keep_contrib {
                vec![0.0; r * d]
            } else {
                Vec::new()
            },
            frac: vec![0.0; c],
        }
    }

    /// Split `y` counts across components with probability `rr/lam`, leaving the
    /// non-zero components in `touched` and their counts in `counts`.
    ///
    /// Two regimes. When `y < R` the multinomial is drawn as `y` categorical
    /// picks by binary search over the cumulative rates — `O(y·log R)`. When
    /// `y ≥ R` the conditional-binomial chain is cheaper at `O(R)`. The chain
    /// alone would be `O(R)` binomial draws per gene regardless of `y`, which is
    /// unaffordable once `R` is in the hundreds.
    fn allocate(&mut self, y: f32, lam: f32, r: usize) {
        self.touched.clear();
        let n = y as u64;
        if (y as usize) < r {
            for _ in 0..n {
                let u: f32 = self.rng.random::<f32>() * lam;
                // First index whose cumulative rate reaches `u`.
                let m = self.cum.partition_point(|&cv| cv < u).min(r - 1);
                if self.counts[m] == 0.0 {
                    self.touched.push(m as u32);
                }
                self.counts[m] += 1.0;
            }
            return;
        }
        let mut remaining = n;
        let mut remain_p = lam;
        for m in 0..r - 1 {
            if remaining == 0 {
                break;
            }
            let p = if remain_p > 0.0 {
                f64::from((self.rr[m] / remain_p).clamp(0.0, 1.0))
            } else {
                0.0
            };
            let zc = if p >= 1.0 {
                remaining
            } else if p <= 0.0 {
                0
            } else {
                Binomial::new(remaining, p).map_or(0, |b| b.sample(&mut self.rng))
            };
            if zc > 0 {
                self.counts[m] = zc as f32;
                self.touched.push(m as u32);
            }
            remaining -= zc;
            remain_p -= self.rr[m];
        }
        if remaining > 0 {
            self.counts[r - 1] = remaining as f32;
            self.touched.push((r - 1) as u32);
        }
    }

    /// One full pass over the genes of a single bulk sample: per-gene NB factor,
    /// gene allocation, then a tempered Gamma draw of the abundances.
    fn sweep(&mut self, ctx: &SweepCtx, y_col: &[f32]) {
        let (d, r, c) = (ctx.d, ctx.r, ctx.c);
        let (nb_r, tau) = (ctx.nb_r, ctx.tau);
        self.nvec.fill(0.0);
        self.m_exp.fill(0.0);
        if ctx.keep_contrib {
            self.contrib.fill(0.0);
        }
        if ctx.collect_expr {
            self.zc.fill(0.0);
        }

        for (g, raw) in y_col.iter().enumerate() {
            let mu_row = &ctx.mu_gm[g * r..g * r + r]; // contiguous over components
            let mut lam = 0f32;
            for (((rr, cum), &wm), &mu) in self
                .rr
                .iter_mut()
                .zip(self.cum.iter_mut())
                .zip(self.w.iter())
                .zip(mu_row)
            {
                let rc = wm * mu;
                *rr = rc;
                lam += rc;
                *cum = lam;
            }
            let y = raw.round();
            // ε_g ~ Gamma(r + τy, r + τλ): soaks per-gene misfit; ε→1 as r→∞.
            let eg = Gamma::new(f64::from(nb_r + tau * y), 1.0 / f64::from(nb_r + tau * lam))
                .map_or(1.0, |gm| gm.sample(&mut self.rng) as f32);
            self.eps[g] = eg;
            for (me, &mu) in self.m_exp.iter_mut().zip(mu_row) {
                *me += eg * mu;
            }
            if y <= 0.0 || lam <= 0.0 {
                continue;
            }

            self.allocate(y, lam, r);
            for &mu32 in &self.touched {
                let m = mu32 as usize;
                let zc = self.counts[m];
                self.nvec[m] += zc;
                if ctx.keep_contrib {
                    self.contrib[m * d + g] = zc;
                }
                if ctx.collect_expr {
                    for ct in 0..c {
                        self.zc[ct * d + g] += zc * ctx.readout[(m, ct)];
                    }
                }
                self.counts[m] = 0.0;
            }
        }

        // u_m ~ Gamma(a0 + τ·n_m, b0 + τ·M_m): tempered, ε-weighted exposure.
        for m in 0..r {
            let shape = f64::from(ctx.a0 + tau * self.nvec[m]);
            let scale = 1.0 / f64::from(ctx.b0 + tau * self.m_exp[m]);
            let draw = Gamma::new(shape, scale)
                .map_or(ABUNDANCE_FLOOR, |g| g.sample(&mut self.rng) as f32);
            self.w[m] = if draw.is_finite() && draw > ABUNDANCE_FLOOR {
                draw
            } else {
                ABUNDANCE_FLOOR
            };
        }
    }

    /// `fraction_c = Σ_m u_m A_{m,c} / Σ_m u_m`, written into `self.frac`.
    fn fractions(&mut self, reference: &Reference) {
        let total: f32 = self.w.iter().sum();
        reference.contract(&self.w, &mut self.frac);
        if total > 0.0 {
            for f in &mut self.frac {
                *f /= total;
            }
        }
    }
}

//////////////////////////////////////
// Low-rank anchors (elliptical slice) //
//////////////////////////////////////

/// Resamples the per-cell-type anchors and refreshes the reference rates.
/// Present only in the low-rank mode.
pub struct AnchorSampler<'a> {
    src: &'a EmbeddingSource,
    prior: &'a AnchorPrior,
    rng: SmallRng,
    /// `t_c = t̂_c + δ_c`.
    delta: Vec<DVec>,
    delta_sum: Vec<DVec>,
    tmat: Mat,
    /// `D×C` constant part `ρ·t̂ᵀ`, so each lnpdf only adds `ρ·δ`.
    base: Mat,
    /// `H×D`, reused every sweep to form the `C×D` log-rate in one gemm.
    rho_t: Mat,
    h: usize,
}

impl<'a> AnchorSampler<'a> {
    pub fn new(src: &'a EmbeddingSource, prior: &'a AnchorPrior, seed: u64) -> Self {
        let c = prior.mean.nrows();
        let h = src.h;
        Self {
            src,
            prior,
            rng: SmallRng::seed_from_u64(seed.wrapping_add(ANCHOR_SEED_OFFSET)),
            delta: (0..c).map(|_| DVec::zeros(h)).collect(),
            delta_sum: (0..c).map(|_| DVec::zeros(h)).collect(),
            tmat: prior.mean.clone(),
            base: &src.rho * prior.mean.transpose(),
            rho_t: src.rho.transpose(),
            h,
        }
    }

    /// Rebuild `μ_{g,c} = exp(ρ_g·t_c + a_g)` from the current anchors.
    ///
    /// `tmat·ρᵀ` is `C×D` column-major, whose buffer index `g*C+ct` already
    /// equals `ρ_g·t_c` — exactly the gene-major layout the sampler reads.
    fn refresh(&self, mu_gm: &mut [f32], c: usize) {
        let scd = &self.tmat * &self.rho_t;
        let scd_buf = scd.as_slice();
        let a = &self.src.gene_offset;
        for (i, mu) in mu_gm.iter_mut().enumerate() {
            *mu = (scd_buf[i] + a[i / c])
                .clamp(-SCORE_CLAMP, SCORE_CLAMP)
                .exp();
        }
    }

    /// One elliptical-slice step per cell type against the pooled statistics.
    fn step(&mut self, a_pool: &[f32], w_exp: &[f32], d: usize, tau: f32, collecting: bool) {
        let c = self.prior.mean.nrows();
        let a = &self.src.gene_offset;
        for ct in 0..c {
            let a_c = &a_pool[ct * d..(ct + 1) * d];
            let w_cg = &w_exp[ct * d..(ct + 1) * d];
            let base_c = self.base.column(ct);
            let lnpdf = |dl: &DVec| -> f32 {
                let rd = &self.src.rho * dl; // D
                let mut ll = 0f32;
                for g in 0..d {
                    let sg = base_c[g] + rd[g] + a[g];
                    ll += a_c[g] * sg - w_cg[g] * sg.clamp(-SCORE_CLAMP, SCORE_CLAMP).exp();
                }
                tau * ll
            };
            let prior_sample = draw_prior(self.prior, ct, self.h, &mut self.rng);
            // Recomputed every sweep: the sufficient statistics changed, so a
            // cached value from the previous sweep would be for a different target.
            let cur_ll = lnpdf(&self.delta[ct]);
            let (new_delta, _) = elliptical_slice_step(
                &self.delta[ct],
                &prior_sample,
                &lnpdf,
                cur_ll,
                &mut self.rng,
            );
            self.delta[ct] = new_delta;
            for j in 0..self.h {
                self.tmat[(ct, j)] = self.prior.mean[(ct, j)] + self.delta[ct][j];
            }
            if collecting {
                self.delta_sum[ct] += &self.delta[ct];
            }
        }
    }

    /// Mean displacement `‖δ‖` across types — the anchor-drift monitor.
    fn mean_drift(&self) -> f32 {
        if self.delta.is_empty() {
            return 0.0;
        }
        self.delta.iter().map(nalgebra::DVector::norm).sum::<f32>() / self.delta.len() as f32
    }

    fn posterior_anchors(&self, n_collect: usize) -> Mat {
        let mut out = self.prior.mean.clone();
        for ct in 0..out.nrows() {
            for j in 0..self.h {
                out[(ct, j)] += self.delta_sum[ct][j] / n_collect as f32;
            }
        }
        out
    }
}

/// Draw `ν ~ N(0, Σ_c)`: `σ_c·z` (isotropic) or `L_c·z` (full).
fn draw_prior(prior: &AnchorPrior, ct: usize, h: usize, rng: &mut SmallRng) -> DVec {
    let z = DVec::from_fn(h, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    });
    match &prior.chol {
        Some(chols) => &chols[ct] * z,
        None => z * prior.sigma[ct],
    }
}

//////////////////
// Sampler entry //
//////////////////

/// Posterior accumulators, shared by every chain in a run.
///
/// Keeping these outside the chain is what lets several chains — and, in the
/// archetype mode, several *partitions* — pool into one posterior. The
/// partition is a nuisance parameter: nothing in the problem picks a particular
/// clustering of the reference cells, and conditioning on one silently treats a
/// modelling choice as known. Pooling draws across partitions averages over it.
pub struct PosteriorAccum {
    s: usize,
    c: usize,
    d: usize,
    frac_sum: Vec<f64>,
    frac_sumsq: Vec<f64>,
    frac_quant: RunningQuantiles,
    abundance_sum: Vec<f64>,
    /// Per type a contiguous column-major D×S block: `[ct*D*S + si*D + g]`.
    zexp: Vec<f64>,
    n_collect: usize,
    n_expr: usize,
    /// Retained per-draw fractions, laid out `[draw][si*C + ct]`.
    trace: Vec<f32>,
    /// Index into `trace` at which each chain started, for between-chain R̂.
    chain_starts: Vec<usize>,
}

impl PosteriorAccum {
    #[must_use]
    pub fn new(s: usize, c: usize, d: usize) -> Self {
        Self {
            s,
            c,
            d,
            frac_sum: vec![0f64; s * c],
            frac_sumsq: vec![0f64; s * c],
            frac_quant: RunningQuantiles::new(s * c, &[0.025, 0.975]),
            abundance_sum: vec![0f64; s * c],
            zexp: vec![0f64; c * d * s],
            n_collect: 0,
            n_expr: 0,
            trace: Vec::new(),
            chain_starts: Vec::new(),
        }
    }

    fn begin_chain(&mut self) {
        self.chain_starts.push(self.n_collect);
    }

    /// Draws collected by each chain, in order.
    fn chain_lengths(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.chain_starts.len());
        for (i, &start) in self.chain_starts.iter().enumerate() {
            let end = self
                .chain_starts
                .get(i + 1)
                .copied()
                .unwrap_or(self.n_collect);
            out.push(end - start);
        }
        out
    }
}

/// Run one chain, appending its draws to `accum`.
///
/// Returns the posterior-mean anchor coordinates when the low-rank reference
/// moved them, so the caller reports where the anchors ended up rather than
/// where their prior sat.
pub fn run_chain(
    reference: &mut Reference,
    bulk: &Mat,
    init_w: &Mat,
    cfg: &SamplerConfig,
    mut anchors: Option<AnchorSampler<'_>>,
    monitor: &mut Monitor,
    accum: &mut PosteriorAccum,
) -> anyhow::Result<Option<Mat>> {
    let d = reference.n_genes;
    let r = reference.n_comp;
    let c = reference.n_types();
    let s = bulk.ncols();
    anyhow::ensure!(
        bulk.nrows() == d,
        "gibbs: bulk genes ({}) != reference genes ({d})",
        bulk.nrows()
    );
    anyhow::ensure!(
        init_w.nrows() == s && init_w.ncols() == r,
        "gibbs: init_w is {}×{}, expected {s}×{r}",
        init_w.nrows(),
        init_w.ncols()
    );
    anyhow::ensure!(
        accum.s == s && accum.c == c && accum.d == d,
        "gibbs: accumulator is {}×{}×{}, chain is {s}×{c}×{d}",
        accum.s,
        accum.c,
        accum.d
    );
    accum.begin_chain();

    let keep_contrib = anchors.is_some();
    let mut states: Vec<SampleState> = (0..s)
        .map(|si| {
            let mut st = SampleState::new(cfg.seed.wrapping_add(si as u64), r, c, d, keep_contrib);
            for m in 0..r {
                st.w[m] = init_w[(si, m)].max(ABUNDANCE_FLOOR);
            }
            st
        })
        .collect();

    let mut frac_flat = vec![0f32; s * c];

    // Only the anchor ESS needs the R×D pooled statistics.
    let mut a_pool = if keep_contrib {
        vec![0f32; r * d]
    } else {
        Vec::new()
    };
    let mut w_exp = if keep_contrib {
        vec![0f32; r * d]
    } else {
        Vec::new()
    };

    if let Some(anch) = anchors.as_ref() {
        anch.refresh(&mut reference.mu_gm, c);
    }

    let total = cfg.warmup + cfg.draws * cfg.thin.max(1);
    let mut collected_here = 0usize;
    for it in 0..total {
        let collecting = it >= cfg.warmup && (it - cfg.warmup).is_multiple_of(cfg.thin.max(1));
        let collect_expr = collecting && collected_here.is_multiple_of(cfg.expression_every.max(1));

        // 1. Per-sample NB factor + gene allocation + abundance draw (parallel).
        // Bulk is column-major, so each sample's gene column is a zero-copy slice.
        let ctx = SweepCtx {
            mu_gm: &reference.mu_gm,
            readout: &reference.readout,
            d,
            r,
            c,
            a0: cfg.a0,
            b0: cfg.b0,
            nb_r: cfg.nb_r,
            tau: cfg.tau,
            keep_contrib,
            collect_expr,
        };
        let bulk_buf = bulk.as_slice();
        states
            .par_iter_mut()
            .enumerate()
            .for_each(|(si, st)| st.sweep(&ctx, &bulk_buf[si * d..(si + 1) * d]));

        // 2. Aggregate: fractions, pooled anchor statistics, and post-warmup collection.
        if keep_contrib {
            a_pool.fill(0.0);
            w_exp.fill(0.0);
        }
        for (si, st) in states.iter_mut().enumerate() {
            st.fractions(reference);
            if keep_contrib {
                for (ap, &cn) in a_pool.iter_mut().zip(&st.contrib) {
                    *ap += cn;
                }
                for m in 0..r {
                    let wm = st.w[m];
                    for (we, &e) in w_exp[m * d..(m + 1) * d].iter_mut().zip(&st.eps) {
                        *we += wm * e;
                    }
                }
            }
            for ct in 0..c {
                frac_flat[si * c + ct] = st.frac[ct];
            }
            if collecting {
                let mut abund = vec![0f32; c];
                reference.contract(&st.w, &mut abund);
                for (ct, &ab) in abund.iter().enumerate() {
                    let idx = si * c + ct;
                    let frac = f64::from(st.frac[ct]);
                    accum.frac_sum[idx] += frac;
                    accum.frac_sumsq[idx] += frac * frac;
                    accum.abundance_sum[idx] += f64::from(ab);
                }
                if collect_expr {
                    for ct in 0..c {
                        let off = ct * d * s + si * d;
                        let dst = &mut accum.zexp[off..off + d];
                        for (z, &v) in dst.iter_mut().zip(&st.zc[ct * d..ct * d + d]) {
                            *z += f64::from(v);
                        }
                    }
                }
            }
        }

        // 3. Anchor ESS update (pooled statistics changed this sweep).
        if let Some(anch) = anchors.as_mut() {
            anch.step(&a_pool, &w_exp, d, cfg.tau, collecting);
            anch.refresh(&mut reference.mu_gm, c);
        }

        if collecting {
            accum.frac_quant.add_dense_column(&frac_flat);
            accum.trace.extend_from_slice(&frac_flat);
            accum.n_collect += 1;
            collected_here += 1;
            if collect_expr {
                accum.n_expr += 1;
            }
        }

        monitor.record(
            it,
            it >= cfg.warmup,
            &frac_flat,
            anchors.as_ref().map_or(0.0, AnchorSampler::mean_drift),
        )?;
        if accum.n_collect > 0 {
            monitor.checkpoint(it, &accum.frac_sum, &frac_flat, accum.n_collect)?;
        }
        if it.is_multiple_of(100) {
            info!(
                "deconvolve gibbs: sweep {it}/{total} (collected {} total)",
                accum.n_collect
            );
        }
    }
    Ok(anchors.map(|a| a.posterior_anchors(collected_here.max(1))))
}

/// Turn the pooled draws into the reported posterior.
///
/// `coords` and `comp_names` describe the component axis carried through to the
/// output; with several partitions pooled they are the concatenation, so the
/// archetype table shows every partition that contributed.
pub fn finalize(
    accum: &PosteriorAccum,
    bulk: &Mat,
    type_rates: &Mat,
    coords: Mat,
    anchor_names: Vec<Box<str>>,
    celltype_names: Vec<Box<str>>,
) -> anyhow::Result<DeconvResult> {
    anyhow::ensure!(accum.n_collect > 0, "gibbs: no posterior draws collected");
    anyhow::ensure!(accum.n_expr > 0, "gibbs: no expression draws collected");
    let (s, c, d) = (accum.s, accum.c, accum.d);
    let nc = accum.n_collect as f64;
    let ne = accum.n_expr as f64;
    let frac_lo = accum.frac_quant.quantile(0);
    let frac_hi = accum.frac_quant.quantile(1);

    let mut fractions_mean = Mat::zeros(s, c);
    let mut fractions_sd = Mat::zeros(s, c);
    let mut fractions_lo = Mat::zeros(s, c);
    let mut fractions_hi = Mat::zeros(s, c);
    let mut abundance_mean = Mat::zeros(s, c);
    for si in 0..s {
        for ct in 0..c {
            let idx = si * c + ct;
            let mean = (accum.frac_sum[idx] / nc) as f32;
            let var = (accum.frac_sumsq[idx] / nc - f64::from(mean) * f64::from(mean)).max(0.0);
            fractions_mean[(si, ct)] = mean;
            fractions_sd[(si, ct)] = var.sqrt() as f32;
            fractions_lo[(si, ct)] = frac_lo[idx];
            fractions_hi[(si, ct)] = frac_hi[idx];
            abundance_mean[(si, ct)] = (accum.abundance_sum[idx] / nc) as f32;
        }
    }

    let expression: Vec<Mat> = (0..c)
        .map(|ct| {
            Mat::from_fn(d, s, |g, si| {
                (accum.zexp[ct * d * s + si * d + g] / ne) as f32
            })
        })
        .collect();

    let residual = (0..s)
        .map(|si| residual_stat(bulk, type_rates, &abundance_mean, si))
        .collect();

    Ok(DeconvResult {
        fractions_mean,
        fractions_sd,
        fractions_lo,
        fractions_hi,
        abundance_mean,
        expression,
        anchors_post: coords,
        anchor_names,
        residual,
        celltype_names,
        convergence: diagnose(accum),
    })
}

/// Split-R̂ / ESS per (sample, celltype).
///
/// With one chain this is the split-half diagnostic, which catches drift but not
/// multimodality. With several chains — including chains over different
/// reference partitions — the between-chain comparison is the stronger test, and
/// disagreement there means the partition itself is moving the answer.
fn diagnose(accum: &PosteriorAccum) -> Vec<Convergence> {
    let (s, c) = (accum.s, accum.c);
    let lengths = accum.chain_lengths();
    let mut out = Vec::new();
    let mut chain = Vec::new();
    for si in 0..s {
        for ct in 0..c {
            let pull = |start: usize, len: usize, buf: &mut Vec<f32>| {
                buf.clear();
                buf.extend((0..len).map(|t| accum.trace[(start + t) * s * c + si * c + ct]));
            };
            let stat = if lengths.len() > 1 {
                // Between-chain: pool the per-chain means/variances directly.
                multi_chain_rhat(accum, &lengths, si, ct)
            } else {
                pull(0, accum.n_collect, &mut chain);
                split_rhat_ess(&chain)
            };
            if let Some((rhat, ess)) = stat {
                out.push(Convergence {
                    sample: si,
                    celltype: ct,
                    rhat,
                    ess,
                });
            }
        }
    }
    out
}

/// Gelman–Rubin R̂ across chains of possibly different lengths.
fn multi_chain_rhat(
    accum: &PosteriorAccum,
    lengths: &[usize],
    si: usize,
    ct: usize,
) -> Option<(f32, f32)> {
    let (s, c) = (accum.s, accum.c);
    let usable: Vec<(usize, usize)> = accum
        .chain_starts
        .iter()
        .zip(lengths)
        .filter(|(_, &len)| len >= 2)
        .map(|(&start, &len)| (start, len))
        .collect();
    if usable.len() < 2 {
        return None;
    }
    let mut means = Vec::with_capacity(usable.len());
    let mut vars = Vec::with_capacity(usable.len());
    let mut total = 0usize;
    for &(start, len) in &usable {
        let mut sum = 0f64;
        for t in 0..len {
            sum += f64::from(accum.trace[(start + t) * s * c + si * c + ct]);
        }
        let mean = sum / len as f64;
        let mut var = 0f64;
        for t in 0..len {
            let v = f64::from(accum.trace[(start + t) * s * c + si * c + ct]);
            var += (v - mean).powi(2);
        }
        means.push(mean);
        vars.push(var / (len as f64 - 1.0).max(1.0));
        total += len;
    }
    let n_bar = total as f64 / usable.len() as f64;
    let w = vars.iter().sum::<f64>() / vars.len() as f64;
    let grand = means.iter().sum::<f64>() / means.len() as f64;
    let b = n_bar * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>()
        / (means.len() as f64 - 1.0).max(1.0);
    if w <= 0.0 {
        return Some((1.0, total as f32));
    }
    let var_hat = (n_bar - 1.0) / n_bar * w + b / n_bar;
    let rhat = (var_hat / w).max(0.0).sqrt() as f32;
    // Effective sample size from the standard between/within ratio.
    let ess = (total as f64 * var_hat / b.max(1e-12)).clamp(1.0, total as f64) as f32;
    Some((rhat, ess))
}
