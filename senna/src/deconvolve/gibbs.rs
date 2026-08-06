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

use super::args::SamplerConfig;
use super::monitor::Monitor;
use super::reference::Reference;
use super::result::{
    residual_stat, split_rhat_ess, Convergence, DeconvResult, ExpressionTensor,
    MIN_DRAWS_FOR_DIAGNOSTICS,
};
use crate::embed_common::Mat;
use log::info;
use matrix_util::running_quantile::RunningQuantiles;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Binomial, Distribution, Gamma};
use rayon::prelude::*;

/// Floor on a sampled abundance, so a zeroed component can still recover.
const ABUNDANCE_FLOOR: f32 = 1e-12;

/// Sweep-invariant inputs shared across all samples.
struct SweepCtx<'a> {
    mu_gm: &'a [f32],
    readout: &'a Mat,
    d: usize,
    r: usize,
    c: usize,
    cfg: &'a SamplerConfig,
}

/// Per-sample chain state and scratch, allocated once and reused every sweep.
///
/// Counts are contracted through the readout to `C` inside the gene loop rather
/// than staged as an `R×D` block: at `R = 500` and `D = 25 000` that block would
/// be 50 MB per sample, and every sample is live at once under rayon.
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
    /// `C` fractions this sweep, and the abundance total they were scaled by.
    frac: Vec<f32>,
    w_total: f32,
}

impl SampleState {
    fn new(seed: u64, r: usize, c: usize) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            w: vec![1.0; r],
            nvec: vec![0.0; r],
            m_exp: vec![0.0; r],
            rr: vec![0.0; r],
            cum: vec![0.0; r],
            counts: vec![0.0; r],
            touched: Vec::with_capacity(r.min(64)),
            frac: vec![0.0; c],
            w_total: 0.0,
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
    /// `lam_dst` and `zexp_dst` are this sample's slices of the run
    /// accumulators, written in place. Staging them per sample first would cost
    /// `C·D` floats for every sample alive under rayon — 292 MB at 200 samples —
    /// and buy nothing but a second copy.
    fn sweep(
        &mut self,
        ctx: &SweepCtx,
        y_col: &[f32],
        mut lam_dst: Option<&mut [f64]>,
        mut zexp_dst: Option<&mut [f32]>,
    ) {
        let (d, r, c) = (ctx.d, ctx.r, ctx.c);
        let (nb_r, tau) = (ctx.cfg.nb_r, ctx.cfg.tau);
        self.nvec.fill(0.0);
        self.m_exp.fill(0.0);
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
            if let Some(dst) = lam_dst.as_deref_mut() {
                dst[g] += f64::from(lam);
            }
            let y = raw.round();
            // ε_g ~ Gamma(r + τy, r + τλ): soaks per-gene misfit; ε→1 as r→∞.
            let eg = Gamma::new(f64::from(nb_r + tau * y), 1.0 / f64::from(nb_r + tau * lam))
                .map_or(1.0, |gm| gm.sample(&mut self.rng) as f32);
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
                if let Some(dst) = zexp_dst.as_deref_mut() {
                    for ct in 0..c {
                        dst[ct * d + g] += zc * ctx.readout[(m, ct)];
                    }
                }
                self.counts[m] = 0.0;
            }
        }

        // u_m ~ Gamma(a0 + τ·n_m, b0 + τ·M_m): tempered, ε-weighted exposure.
        for m in 0..r {
            let shape = f64::from(ctx.cfg.a0 + tau * self.nvec[m]);
            let scale = 1.0 / f64::from(ctx.cfg.b0 + tau * self.m_exp[m]);
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
        self.w_total = total;
        if total > 0.0 {
            for f in &mut self.frac {
                *f /= total;
            }
        }
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
    /// Per-component abundance sum over the POOLED component axis, `S×ΣR` —
    /// the pre-readout quantity, laid out so its columns line up with the
    /// component table's rows.
    comp_abundance_sum: Vec<f64>,
    /// Where the current chain's components start in that axis.
    comp_offset: usize,
    /// `(offset, width)` per chain, so each block can be divided by the draws
    /// that actually went into it rather than by the pooled total.
    comp_spans: Vec<(usize, usize)>,
    n_comp_total: usize,
    /// Allocated counts, laid out sample-major `[si*C*D + ct*D + g]`.
    ///
    /// Sample-major so each sample's block is contiguous and can be handed to
    /// its own rayon worker, which is what removes the per-sample staging copy.
    /// `f32` because this sums one term per retained expression draw — a few
    /// hundred — not millions, so the accumulation error stays far below the
    /// f32 resolution of the value that is finally reported.
    zexp: Vec<f32>,
    /// Posterior-mean Poisson rate, column-major `D×S`.
    ///
    /// Accumulated from the sampler's own `λ` rather than rebuilt afterwards.
    /// Rebuilding it as `Σ_c abundance_c · μ̄_{g,c}` — a per-type average rate
    /// times a per-type abundance — is a product of sums, and differs from the
    /// true `Σ_m u_m μ_{g,m}` whenever components within a type carry unequal
    /// abundance.
    lam_sum: Vec<f64>,
    n_collect: usize,
    n_expr: usize,
    /// Retained per-draw fractions, laid out `[draw][si*C + ct]`.
    trace: Vec<f32>,
    /// Index into `trace` at which each chain started, for between-chain R̂.
    chain_starts: Vec<usize>,
}

impl PosteriorAccum {
    /// Expression-tensor size (`C·D·S` f64) past which the allocation is worth
    /// warning about, in elements.
    const LARGE_TENSOR: u128 = 200_000_000;

    #[must_use]
    pub fn new(s: usize, c: usize, d: usize, n_comp_total: usize) -> Self {
        let elements = (c as u128) * (d as u128) * (s as u128);
        if elements > Self::LARGE_TENSOR {
            info!(
                "deconvolve: expression tensor is large (C·D·S ≈ {elements} f64, {} GiB); \
                 consider a reduced gene reference if memory is tight",
                elements * 8 / (1 << 30)
            );
        }
        Self {
            s,
            c,
            d,
            frac_sum: vec![0f64; s * c],
            frac_sumsq: vec![0f64; s * c],
            frac_quant: RunningQuantiles::new(s * c, &[0.025, 0.975]),
            abundance_sum: vec![0f64; s * c],
            comp_abundance_sum: vec![0f64; s * n_comp_total],
            comp_offset: 0,
            comp_spans: Vec::new(),
            n_comp_total,
            zexp: vec![0f32; c * d * s],
            lam_sum: vec![0f64; d * s],
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
pub fn run_chain(
    reference: &Reference,
    bulk: &Mat,
    init_w: &Mat,
    cfg: &SamplerConfig,
    monitor: &mut Monitor,
    accum: &mut PosteriorAccum,
) -> anyhow::Result<()> {
    let d = reference.n_genes();
    let r = reference.n_comp();
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
    anyhow::ensure!(
        accum.comp_offset + r <= accum.n_comp_total,
        "gibbs: component axis holds {} slots, chain needs {r} more from {}",
        accum.n_comp_total,
        accum.comp_offset
    );
    accum.begin_chain();
    accum.trace.reserve(cfg.draws * s * c);

    let mut states: Vec<SampleState> = (0..s)
        .map(|si| {
            let mut st = SampleState::new(cfg.seed.wrapping_add(si as u64), r, c);
            for m in 0..r {
                st.w[m] = init_w[(si, m)].max(ABUNDANCE_FLOOR);
            }
            st
        })
        .collect();

    let mut frac_flat = vec![0f32; s * c];

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
            cfg,
        };
        let bulk_buf = bulk.as_slice();
        states
            .par_iter_mut()
            .zip(accum.lam_sum.par_chunks_mut(d))
            .zip(accum.zexp.par_chunks_mut(c * d))
            .enumerate()
            .for_each(|(si, ((st, lam_dst), zexp_dst))| {
                st.sweep(
                    &ctx,
                    &bulk_buf[si * d..(si + 1) * d],
                    collecting.then_some(lam_dst),
                    collect_expr.then_some(zexp_dst),
                );
            });

        // 2. Aggregate: fractions and post-warmup collection.
        for (si, st) in states.iter_mut().enumerate() {
            st.fractions(reference);
            for ct in 0..c {
                frac_flat[si * c + ct] = st.frac[ct];
            }
            if collecting {
                let base = si * accum.n_comp_total + accum.comp_offset;
                for (acc, &w) in accum.comp_abundance_sum[base..base + r]
                    .iter_mut()
                    .zip(&st.w)
                {
                    *acc += f64::from(w);
                }
                for (ct, &f) in st.frac.iter().enumerate() {
                    let idx = si * c + ct;
                    let frac = f64::from(f);
                    accum.frac_sum[idx] += frac;
                    accum.frac_sumsq[idx] += frac * frac;
                    // The un-normalised contraction, without redoing it.
                    accum.abundance_sum[idx] += frac * f64::from(st.w_total);
                }
            }
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

        monitor.record(it, it >= cfg.warmup, &frac_flat)?;
        if accum.n_collect > 0 {
            monitor.checkpoint(it, &accum.frac_sum, accum.n_collect)?;
        }
        if it.is_multiple_of(100) {
            info!(
                "deconvolve gibbs: sweep {it}/{total} (collected {} total)",
                accum.n_collect
            );
        }
    }
    accum.comp_spans.push((accum.comp_offset, r));
    accum.comp_offset += r;
    Ok(())
}

/// The component axis carried through to the output: where each component sits
/// and what it is called. With several partitions pooled these are the
/// concatenation, so the table shows every partition that contributed.
pub struct ComponentTable {
    pub coords: Mat,
    pub names: Vec<Box<str>>,
}

/// Turn the pooled draws into the reported posterior.
///
pub fn finalize(
    mut accum: PosteriorAccum,
    bulk: &Mat,
    components: ComponentTable,
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
            // Keep the mean in f64 for the variance: squaring an f32-rounded
            // mean injects an error that swamps any sd below ~1e-4, which is
            // exactly the regime a tight posterior lives in.
            let mean = accum.frac_sum[idx] / nc;
            let var = (accum.frac_sumsq[idx] / nc - mean * mean).max(0.0);
            fractions_mean[(si, ct)] = mean as f32;
            fractions_sd[(si, ct)] = var.sqrt() as f32;
            fractions_lo[(si, ct)] = frac_lo[idx];
            fractions_hi[(si, ct)] = frac_hi[idx];
            abundance_mean[(si, ct)] = (accum.abundance_sum[idx] / nc) as f32;
        }
    }

    // Scale in place and hand the buffer over: the tensor is the largest thing
    // here, and copying it into per-type matrices would double the peak.
    let inv = 1.0 / ne as f32;
    for z in &mut accum.zexp {
        *z *= inv;
    }
    let expression = ExpressionTensor::new(std::mem::take(&mut accum.zexp), d, s, c);

    let residual = (0..s)
        .map(|si| residual_stat(bulk, &accum.lam_sum[si * d..(si + 1) * d], nc, si))
        .collect();

    Ok(DeconvResult {
        fractions_mean,
        fractions_sd,
        fractions_lo,
        fractions_hi,
        abundance_mean,
        component_abundance: component_means(&accum),
        expression,
        anchors_post: components.coords,
        anchor_names: components.names,
        residual,
        celltype_names,
        convergence: diagnose(&accum),
        n_chains: accum.chain_starts.len(),
    })
}

/// Per-component posterior-mean abundance, `S × ΣR`.
///
/// Each chain's block is divided by the draws that chain contributed, not by the
/// pooled total: a component only ever accumulates while its own chain is
/// running, so the pooled count would scale every block down by the number of
/// chains and make the values incomparable with a known per-component mass.
fn component_means(accum: &PosteriorAccum) -> Mat {
    let (s, total) = (accum.s, accum.n_comp_total);
    let lengths = accum.chain_lengths();
    let mut out = Mat::zeros(s, total);
    for (&(offset, width), &len) in accum.comp_spans.iter().zip(&lengths) {
        let n = (len as f64).max(1.0);
        for si in 0..s {
            for m in 0..width {
                out[(si, offset + m)] =
                    (accum.comp_abundance_sum[si * total + offset + m] / n) as f32;
            }
        }
    }
    out
}

/// Split-R̂ / ESS per (sample, celltype).
///
/// With one chain this is the split-half diagnostic, which catches drift but not
/// multimodality. With several chains — including chains over different
/// reference partitions — the between-chain comparison is the stronger test, and
/// disagreement there means the partition itself is moving the answer.
fn diagnose(accum: &PosteriorAccum) -> Vec<Convergence> {
    let (s, c) = (accum.s, accum.c);
    // Chains usable for the between-chain comparison, found once rather than
    // per reported scalar.
    let usable: Vec<(usize, usize)> = accum
        .chain_starts
        .iter()
        .zip(accum.chain_lengths())
        .filter(|(_, len)| *len >= MIN_DRAWS_FOR_DIAGNOSTICS)
        .map(|(&start, len)| (start, len))
        .collect();
    let n_chains = accum.chain_starts.len();
    let between_chain = usable.len() > 1;
    // With several chains but fewer than two long enough to compare, the single
    // chain path would walk the concatenated trace and read the jump between
    // chains as drift. Report nothing rather than something wrong.
    if n_chains > 1 && !between_chain {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut chain = vec![0f32; accum.n_collect];
    for si in 0..s {
        for ct in 0..c {
            let stat = if between_chain {
                multi_chain_rhat(accum, &usable, si, ct)
            } else {
                for (t, v) in chain.iter_mut().enumerate() {
                    *v = accum.trace[t * s * c + si * c + ct];
                }
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
///
/// Chains shorter than [`MIN_DRAWS_FOR_DIAGNOSTICS`] are filtered out by the
/// caller, matching the single-chain guard — otherwise a two-draw run would
/// report a confident-looking R̂ built from nothing.
fn multi_chain_rhat(
    accum: &PosteriorAccum,
    usable: &[(usize, usize)],
    si: usize,
    ct: usize,
) -> Option<(f32, f32)> {
    let (s, c) = (accum.s, accum.c);
    if usable.len() < 2 {
        return None;
    }
    let mut means = Vec::with_capacity(usable.len());
    let mut vars = Vec::with_capacity(usable.len());
    let mut total = 0usize;
    for &(start, len) in usable {
        // One pass per chain: the trace is strided by S·C, so a second pass is a
        // second sweep of cache misses over the whole buffer.
        let (mut sum, mut sumsq) = (0f64, 0f64);
        for t in 0..len {
            let v = f64::from(accum.trace[(start + t) * s * c + si * c + ct]);
            sum += v;
            sumsq += v * v;
        }
        let mean = sum / len as f64;
        let var = (sumsq - sum * mean).max(0.0);
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
