//! Hierarchical Gibbs sampler for projection-based deconvolution.
//!
//! Model, per bulk sample `s` (rayon-parallel; independent seeded chains):
//! ```text
//!   μ_{g,c} = exp(ρ_g·t_c + a_g)
//!   ε_{s,g} ~ Gamma(r, r)                            (NB overdispersion, mean 1)
//!   y_{s,g} ~ Poisson(ε_{s,g}·Σ_c w_{s,c} μ_{g,c})
//!   Z_{s,·,g} ~ Multinomial(y_{s,g}, p), p_c ∝ w_{s,c} μ_{g,c}   (ε cancels in the split)
//!   ε_{s,g}   ~ Gamma(r + τy_{s,g}, r + τλ_{s,g})                (conjugate)
//!   w_{s,c}   ~ Gamma(a0 + τΣ_g Z_{s,c,g}, b0 + τΣ_g ε_{s,g}μ_{g,c})
//! ```
//! `r` (`--nb-dispersion`) is the negative-binomial dispersion, held fixed
//! (freely sampling it is non-identifiable against the fractions — ε competes
//! with w through the per-type exposure). `r → ∞` recovers Poisson. `τ`
//! (`--count-scale`) tempers the likelihood (power posterior):
//! every count sufficient statistic is scaled by τ, so the posterior reflects
//! τ·(observed counts) of evidence and its variance scales as 1/τ — the CI
//! calibration knob at high read depth.
//! Global cell-type anchors `t_c` are resampled by one elliptical-slice step
//! per sweep under the annotate-by-projection prior `N(t̂_c, Σ_c)`, with the
//! pooled Poisson likelihood
//! `ℓ(t_c) = τ·Σ_g[A_{c,g}(ρ_g·t_c+a_g) − W_{c,g}·exp(ρ_g·t_c+a_g)]`,
//! `A_{c,g}=Σ_s Z_{s,c,g}`, `W_{c,g}=Σ_s w_{s,c}·ε_{s,g}` (ε-weighted exposure).
//! This is the coupling that lets annotation uncertainty widen (or confident
//! types pin) the fraction posterior.

use super::anchors::AnchorPrior;
use super::args::SamplerConfig;
use super::source::EmbeddingSource;
use crate::embed_common::{DVec, Mat};
use log::info;
use matrix_util::running_quantile::RunningQuantiles;
use mcmc_util::engine::elliptical_slice_step;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Gamma, StandardNormal};
use rayon::prelude::*;

/// Clamp on the Poisson log-rate before `exp`.
const SCORE_CLAMP: f32 = 30.0;
/// Offset for the anchor-chain RNG seed (separate stream from the per-sample chains).
const ANCHOR_SEED_OFFSET: u64 = 0x9E37_79B9_7F4A_7C15;

pub struct ResidualStat {
    pub total: f64,
    pub deviance: f64,
    pub pearson: f64,
}

pub struct DeconvResult {
    pub fractions_mean: Mat,
    pub fractions_sd: Mat,
    pub fractions_lo: Mat,
    pub fractions_hi: Mat,
    pub abundance_mean: Mat,
    pub expression: Vec<Mat>,
    pub anchors_post: Mat,
    pub residual: Vec<ResidualStat>,
    pub celltype_names: Vec<Box<str>>,
}

struct SampleOut {
    w_new: Vec<f32>,
    contrib: Vec<f32>, // sampled counts, laid out [c*D + g]
    eps: Vec<f32>,     // NB overdispersion factor per gene
}

/// Sweep-invariant inputs shared across all samples. `mu_gm` is gene-major
/// (`mu_gm[g*C + ct] = μ_{g,c}`) so the inner per-type loop reads a contiguous,
/// SIMD-friendly slice instead of a strided column.
struct SweepCtx<'a> {
    mu_gm: &'a [f32],
    d: usize,
    c: usize,
    a0: f32,
    b0: f32,
    /// NB dispersion `r` and likelihood temperature `τ`.
    nb_r: f32,
    tau: f32,
}

/// Sample one bulk column: per-gene NB factor `ε`, gene allocation `Z`, then a
/// tempered Gamma draw of the abundances `w`.
fn sample_one(ctx: &SweepCtx, y_col: &[f32], w_s: &[f32], rng: &mut SmallRng) -> SampleOut {
    let (d, c) = (ctx.d, ctx.c);
    let (r, tau) = (ctx.nb_r, ctx.tau);
    let mut nvec = vec![0f32; c];
    let mut contrib = vec![0f32; c * d];
    let mut eps = vec![0f32; d];
    let mut m_exp = vec![0f32; c]; // ε-weighted exposure Σ_g ε_g μ_{g,c}
    let mut rr = vec![0f32; c];
    for g in 0..d {
        let mu_row = &ctx.mu_gm[g * c..g * c + c]; // contiguous over cell types
        let mut lam = 0f32;
        for ct in 0..c {
            let rc = w_s[ct] * mu_row[ct];
            rr[ct] = rc;
            lam += rc;
        }
        let y = y_col[g].round();
        // ε_g ~ Gamma(r + τy, r + τλ): soaks per-gene misfit; ε→1 as r→∞.
        let eg = Gamma::new(f64::from(r + tau * y), 1.0 / f64::from(r + tau * lam))
            .map_or(1.0, |gm| gm.sample(rng) as f32);
        eps[g] = eg;
        for ct in 0..c {
            m_exp[ct] += eg * mu_row[ct];
        }
        if y <= 0.0 || lam <= 0.0 {
            continue;
        }
        // Multinomial(y, rr/λ) via conditional binomials — ε cancels here.
        let mut remaining = y as u64;
        let mut remain_p = lam;
        for ct in 0..c - 1 {
            if remaining == 0 {
                break;
            }
            let p = if remain_p > 0.0 {
                f64::from((rr[ct] / remain_p).clamp(0.0, 1.0))
            } else {
                0.0
            };
            let zc = if p >= 1.0 {
                remaining
            } else if p <= 0.0 {
                0
            } else {
                Binomial::new(remaining, p).map_or(0, |b| b.sample(rng))
            };
            nvec[ct] += zc as f32;
            contrib[ct * d + g] = zc as f32;
            remaining -= zc;
            remain_p -= rr[ct];
        }
        nvec[c - 1] += remaining as f32;
        contrib[(c - 1) * d + g] = remaining as f32;
    }

    // w_c ~ Gamma(a0 + τ·n_c, b0 + τ·M_c): tempered, ε-weighted exposure.
    let mut w_new = vec![0f32; c];
    for ct in 0..c {
        let shape = f64::from(ctx.a0 + tau * nvec[ct]);
        let scale = 1.0 / f64::from(ctx.b0 + tau * m_exp[ct]);
        let draw = Gamma::new(shape, scale).map_or(1e-12, |g| g.sample(rng) as f32);
        w_new[ct] = if draw.is_finite() && draw > 1e-12 {
            draw
        } else {
            1e-12
        };
    }
    SampleOut {
        w_new,
        contrib,
        eps,
    }
}

pub fn run_gibbs(
    src: &EmbeddingSource,
    bulk: &Mat,
    prior: &AnchorPrior,
    init_w: &Mat,
    cfg: &SamplerConfig,
) -> anyhow::Result<DeconvResult> {
    let d = src.rho.nrows();
    let h = src.h;
    let s = bulk.ncols();
    let c = prior.mean.nrows();
    anyhow::ensure!(
        bulk.nrows() == d,
        "gibbs: bulk genes ({}) != ρ genes ({d})",
        bulk.nrows()
    );

    let mem_floats = (c as u128) * (d as u128) * (s as u128);
    if mem_floats > 200_000_000 {
        info!(
            "deconvolve: expression tensor is large (C·D·S ≈ {} floats); consider a reduced \
             gene reference if memory is tight",
            mem_floats
        );
    }

    let a = &src.gene_offset;
    // ρ·t̂_cᵀ is constant across sweeps (anchor means are fixed); precompute the
    // base log-rate so each anchor lnpdf only adds ρ·δ.
    let base = &src.rho * prior.mean.transpose(); // D×C
                                                  // ρᵀ reused every sweep to form the C×D log-rate in one gemm.
    let rho_t = src.rho.transpose(); // H×D

    // Per-sample abundances + independent RNG streams.
    let mut w: Vec<Vec<f32>> = (0..s)
        .map(|si| (0..c).map(|ct| init_w[(si, ct)]).collect())
        .collect();
    let mut rngs: Vec<SmallRng> = (0..s)
        .map(|si| SmallRng::seed_from_u64(cfg.seed.wrapping_add(si as u64)))
        .collect();
    let mut anchor_rng = SmallRng::seed_from_u64(cfg.seed.wrapping_add(ANCHOR_SEED_OFFSET));

    // Anchor state: t_c = t̂_c + δ_c (δ starts at zero).
    let mut delta: Vec<DVec> = (0..c).map(|_| DVec::zeros(h)).collect();
    let mut tmat = prior.mean.clone(); // C×H current anchors

    // Post-warmup accumulators. Fraction mean/sd come from running moments and
    // the credible interval from streaming P² quantiles — no stored per-draw arrays.
    let mut frac_sum = vec![0f64; s * c];
    let mut frac_sumsq = vec![0f64; s * c];
    let mut frac_quant = RunningQuantiles::new(s * c, &[0.025, 0.975]);
    let mut frac_flat = vec![0f32; s * c]; // this sweep's fractions, fed to the quantiles
    let mut abundance_sum = vec![0f64; s * c];
    // Per type a column-major D×S block: [ct*D*S + si*D + g].
    let mut zexp = vec![0f64; c * d * s];
    let mut delta_sum: Vec<DVec> = (0..c).map(|_| DVec::zeros(h)).collect();
    let mut n_collect = 0usize;

    // Per-sweep scratch, allocated once and refilled each sweep.
    let mut mu_gm = vec![0f32; c * d];
    let mut a_pool = vec![0f32; c * d];
    let mut w_exp = vec![0f32; c * d];

    let total = cfg.warmup + cfg.draws * cfg.thin.max(1);
    for it in 0..total {
        // 1. Reference rates from the current anchors, gene-major. tmat·ρᵀ is
        // C×D column-major, whose buffer index g*C+ct already equals ρ_g·t_c —
        // exactly the gene-major μ layout the sampler reads contiguously.
        let scd = &tmat * &rho_t; // C×D
        let scd_buf = scd.as_slice();
        for i in 0..c * d {
            mu_gm[i] = (scd_buf[i] + a[i / c])
                .clamp(-SCORE_CLAMP, SCORE_CLAMP)
                .exp();
        }

        // 2. Per-sample NB factor + gene allocation + abundance draw (parallel).
        // Bulk is column-major, so each sample's gene column is a zero-copy slice.
        let ctx = SweepCtx {
            mu_gm: &mu_gm,
            d,
            c,
            a0: cfg.a0,
            b0: cfg.b0,
            nb_r: cfg.nb_r,
            tau: cfg.tau,
        };
        let bulk_buf = bulk.as_slice();
        let outs: Vec<SampleOut> = rngs
            .par_iter_mut()
            .enumerate()
            .map(|(si, rng)| sample_one(&ctx, &bulk_buf[si * d..(si + 1) * d], &w[si], rng))
            .collect();

        // 3. Aggregate: update w, pooled A and W, and (post-warmup) collect.
        let collecting = it >= cfg.warmup && (it - cfg.warmup).is_multiple_of(cfg.thin.max(1));
        // A_{c,g} = Σ_s Z_{s,c,g}; W_{c,g} = Σ_s w_{s,c}·ε_{s,g} (ε-weighted exposure).
        a_pool.fill(0.0);
        w_exp.fill(0.0);
        for (si, out) in outs.iter().enumerate() {
            w[si].copy_from_slice(&out.w_new);
            let wsum: f32 = out.w_new.iter().sum();
            for (ap, &cn) in a_pool.iter_mut().zip(&out.contrib) {
                *ap += cn;
            }
            for ct in 0..c {
                let wc = out.w_new[ct];
                for (we, &e) in w_exp[ct * d..(ct + 1) * d].iter_mut().zip(&out.eps) {
                    *we += wc * e;
                }
            }
            if collecting {
                for ct in 0..c {
                    let idx = si * c + ct;
                    let frac = if wsum > 0.0 {
                        out.w_new[ct] / wsum
                    } else {
                        0.0
                    };
                    frac_flat[idx] = frac;
                    frac_sum[idx] += f64::from(frac);
                    frac_sumsq[idx] += f64::from(frac) * f64::from(frac);
                    abundance_sum[idx] += f64::from(out.w_new[ct]);
                    // zexp per type is a contiguous column-major D×S block.
                    let off = ct * d * s + si * d;
                    let dst = &mut zexp[off..off + d];
                    for (z, &v) in dst.iter_mut().zip(&out.contrib[ct * d..ct * d + d]) {
                        *z += f64::from(v);
                    }
                }
            }
        }

        // 4. Anchor ESS update (one step per type; A,W changed this sweep).
        // Likelihood tempered by τ (same power posterior as the count updates).
        let tau = cfg.tau;
        for ct in 0..c {
            let a_c = &a_pool[ct * d..(ct + 1) * d];
            let w_cg = &w_exp[ct * d..(ct + 1) * d];
            let base_c = base.column(ct);
            let lnpdf = |dl: &DVec| -> f32 {
                let rd = &src.rho * dl; // D
                let mut ll = 0f32;
                for g in 0..d {
                    let sg = base_c[g] + rd[g] + a[g];
                    ll += a_c[g] * sg - w_cg[g] * sg.clamp(-SCORE_CLAMP, SCORE_CLAMP).exp();
                }
                tau * ll
            };
            let prior_sample = draw_prior(prior, ct, h, &mut anchor_rng);
            let cur_ll = lnpdf(&delta[ct]);
            let (new_delta, _) =
                elliptical_slice_step(&delta[ct], &prior_sample, &lnpdf, cur_ll, &mut anchor_rng);
            delta[ct] = new_delta;
            // t_c = t̂_c + δ_c → refresh the anchor row.
            for j in 0..h {
                tmat[(ct, j)] = prior.mean[(ct, j)] + delta[ct][j];
            }
            if collecting {
                delta_sum[ct] += &delta[ct];
            }
        }

        if collecting {
            frac_quant.add_dense_column(&frac_flat);
            n_collect += 1;
        }
        if it % 100 == 0 {
            info!("deconvolve gibbs: sweep {it}/{total} (collected {n_collect})");
        }
    }

    anyhow::ensure!(n_collect > 0, "gibbs: no posterior draws collected");
    let frac_lo = frac_quant.quantile(0);
    let frac_hi = frac_quant.quantile(1);
    let acc = Accum {
        frac_sum: &frac_sum,
        frac_sumsq: &frac_sumsq,
        frac_lo: &frac_lo,
        frac_hi: &frac_hi,
        abundance_sum: &abundance_sum,
        zexp: &zexp,
        delta_sum: &delta_sum,
        n_collect,
    };
    Ok(finalize(src, bulk, prior, &acc))
}

/// Post-warmup accumulators handed to [`finalize`].
struct Accum<'a> {
    frac_sum: &'a [f64],
    frac_sumsq: &'a [f64],
    frac_lo: &'a [f32],
    frac_hi: &'a [f32],
    abundance_sum: &'a [f64],
    zexp: &'a [f64],
    delta_sum: &'a [DVec],
    n_collect: usize,
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

fn finalize(src: &EmbeddingSource, bulk: &Mat, prior: &AnchorPrior, acc: &Accum) -> DeconvResult {
    let d = src.rho.nrows();
    let h = src.h;
    let s = bulk.ncols();
    let c = prior.mean.nrows();
    let nc = acc.n_collect as f64;
    let (zexp, delta_sum) = (acc.zexp, acc.delta_sum);

    let mut fractions_mean = Mat::zeros(s, c);
    let mut fractions_sd = Mat::zeros(s, c);
    let mut fractions_lo = Mat::zeros(s, c);
    let mut fractions_hi = Mat::zeros(s, c);
    let mut abundance_mean = Mat::zeros(s, c);
    for si in 0..s {
        for ct in 0..c {
            let idx = si * c + ct;
            let mean = (acc.frac_sum[idx] / nc) as f32;
            let var = (acc.frac_sumsq[idx] / nc - f64::from(mean) * f64::from(mean)).max(0.0);
            fractions_mean[(si, ct)] = mean;
            fractions_sd[(si, ct)] = var.sqrt() as f32;
            fractions_lo[(si, ct)] = acc.frac_lo[idx];
            fractions_hi[(si, ct)] = acc.frac_hi[idx];
            abundance_mean[(si, ct)] = (acc.abundance_sum[idx] / nc) as f32;
        }
    }

    let expression: Vec<Mat> = (0..c)
        .map(|ct| Mat::from_fn(d, s, |g, si| (zexp[ct * d * s + si * d + g] / nc) as f32))
        .collect();

    let mut anchors_post = prior.mean.clone();
    for ct in 0..c {
        for j in 0..h {
            anchors_post[(ct, j)] += delta_sum[ct][j] / acc.n_collect as f32;
        }
    }

    // Posterior-predictive residuals at the posterior-mean anchors + abundances.
    let sc = &src.rho * anchors_post.transpose();
    let mu = Mat::from_fn(d, c, |g, ct| {
        (sc[(g, ct)] + src.gene_offset[g])
            .clamp(-SCORE_CLAMP, SCORE_CLAMP)
            .exp()
    });
    let residual = (0..s)
        .map(|si| residual_stat(bulk, &mu, &abundance_mean, si))
        .collect();

    DeconvResult {
        fractions_mean,
        fractions_sd,
        fractions_lo,
        fractions_hi,
        abundance_mean,
        expression,
        anchors_post,
        residual,
        celltype_names: prior.names.clone(),
    }
}

fn residual_stat(bulk: &Mat, mu: &Mat, abundance: &Mat, si: usize) -> ResidualStat {
    let d = bulk.nrows();
    let c = mu.ncols();
    let mut total = 0f64;
    let mut deviance = 0f64;
    let (mut sy, mut sl, mut syy, mut sll, mut syl) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for g in 0..d {
        let y = f64::from(bulk[(g, si)]).max(0.0);
        let mut lam = 0f64;
        for ct in 0..c {
            lam += f64::from(abundance[(si, ct)]) * f64::from(mu[(g, ct)]);
        }
        lam = lam.max(1e-12);
        total += y;
        deviance += if y > 0.0 {
            2.0 * (y * (y / lam).ln() - (y - lam))
        } else {
            2.0 * lam
        };
        sy += y;
        sl += lam;
        syy += y * y;
        sll += lam * lam;
        syl += y * lam;
    }
    let n = d as f64;
    let cov = syl - sy * sl / n;
    let vy = (syy - sy * sy / n).max(0.0);
    let vl = (sll - sl * sl / n).max(0.0);
    let pearson = if vy > 0.0 && vl > 0.0 {
        cov / (vy.sqrt() * vl.sqrt())
    } else {
        0.0
    };
    ResidualStat {
        total,
        deviance,
        pearson,
    }
}
