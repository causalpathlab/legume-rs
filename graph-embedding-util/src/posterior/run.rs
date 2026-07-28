//! The `--posterior` / `--mcmc` step, end to end: flags, orchestration, outputs.
//!
//! Both `senna bge` and `faba gem` run the *same* pipeline over a fitted model —
//! build the frozen index, recalibrate the intercepts, sample the Tier-1 hypers,
//! feed the learned slab variance to the gate, sample the gate, write the tables.
//! Only three things genuinely differ between them: which rows form an anchor
//! (bge: one per feature row; gem: one per gene, per splice track), whether a
//! frozen offset is carried (gem's velocity gate holds `β_g` at its MAP), and the
//! output name stem. Those are the [`TrackSpec`] fields; everything else lives
//! here so the two callers cannot drift apart on column names, JSON schema, or
//! the statistical guard below.
//!
//! The flag group ships here too, as a `clap` `Args` both CLIs flatten — the same
//! pattern `data_beans::qc_lib::QcArgs` already uses. That keeps one help text and
//! one resolution rule for a flag pair whose meaning is identical in both.

use super::dim_block::{dim_block, DimBlockConfig, DimBlockResult};
use super::frozen_diag::frozen_side_diag;
use super::gate::{gate_posterior, GateConfig, GenePosterior};
use super::hyper_ss::{hyper_ss, HyperSsConfig, HyperSsResult};
use super::index::{build_index, FrozenMap, RowGrouping};
use crate::eval::save_embedding;
use anyhow::Context;
use candle_util::candle_core::{Device, Tensor};
use log::info;
use matrix_util::traits::IoOps;

/// Default retained draws per chain when `--posterior` is given without `--mcmc`.
pub const DEFAULT_SAMPLES: usize = 200;

/// Default frozen negative-slate size — the number of other-side rows summed in
/// the Poisson rate normalizer. It is the Monte-Carlo resolution of that
/// normalizer, not a subsample of the data: every observed count still enters
/// exactly.
pub const DEFAULT_PARTITION: usize = 1024;

/// Which posterior to run after the MAP fit.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum PosteriorMode {
    /// No posterior — the run is byte-identical to one built before this existed.
    Off,
    /// Per-gene gate posterior only (PIP + credible sets + effect posterior).
    Gate,
    /// Tier-1 hyperparameters only (`σ₀²`, `π₀`, per-gene inclusion probability).
    Hyper,
    /// Both, with the learned `σ₀²` feeding the gate's slab prior.
    Both,
    /// Per-dim selection ([`super::dim_block`]): an independent Bernoulli per
    /// embedding dim with its own `(σ₀h², π₀h)`, instead of the gate's softmax
    /// over dims.
    ///
    /// Self-contained — it learns its own per-dim hyperparameters, so it neither
    /// needs nor reads the Tier-1 globals. Kept a separate mode rather than
    /// folded into [`Self::Both`] because it costs `H` likelihood passes per
    /// anchor per sweep, which is not something to add to an existing default.
    PerDim,
}

impl PosteriorMode {
    #[must_use]
    pub fn runs_gate(self) -> bool {
        matches!(self, Self::Gate | Self::Both)
    }
    #[must_use]
    pub fn runs_hyper(self) -> bool {
        matches!(self, Self::Hyper | Self::Both)
    }
    #[must_use]
    pub fn runs_per_dim(self) -> bool {
        matches!(self, Self::PerDim)
    }
}

/// The `--posterior` / `--mcmc` flag group, flattened by each CLI's args struct.
///
/// The help is written to cover both callers: on a model with more than one gate
/// (gem's identity + velocity) every gate is sampled, which is what the flag
/// means there, so one wording serves both without saying anything false.
#[derive(clap::Args, Debug, Clone)]
pub struct PosteriorArgs {
    #[arg(
        long = "mcmc",
        alias = "jitter",
        value_name = "N",
        help = "Sample the exact posterior over the fitted gate(s), N draws per chain\n\
                (implies --posterior both). Off unless given.",
        long_help = "Run MCMC over the fitted gate(s) and write the posterior tables:\n\
                     N is the number of RETAINED draws per chain (warmup is N/2 more),\n\
                     the sampler's analogue of `-i/--epochs`.\n\
                     \n\
                     Training fits the per-gene softmax gate VARIATIONALLY and reports a point estimate;\n\
                     this samples the SAME model exactly, against the frozen MAP cell side,\n\
                     so selection comes with calibrated uncertainty:\n\
                     per-dim posterior inclusion probabilities, a credible set over the dims a gene loads,\n\
                     and the effect posterior.\n\
                     \n\
                     Every gate the model has is sampled. `faba gem` has two — the identity gate on β_g\n\
                     (read from the spliced rows) and the velocity gate on δ_g (the unspliced rows, with β_g\n\
                     held at its MAP, since an unspliced row scores ⟨β_g + δ_g, e_c⟩);\n\
                     `senna bge` has only the identity gate.\n\
                     \n\
                     Passing this implies `--posterior both` (the full-Bayes answer);\n\
                     use `--posterior` to ask for less.\n\
                     Try 200 to start. Cost is roughly 0.6 core-seconds per gene per 200 draws,\n\
                     so a whole dictionary runs for minutes — Ctrl+C returns partial results.\n\
                     Writes one PIP + summary parquet pair per gate, plus {out}.posterior_hyper.json.\n\
                     `--jitter` is an accepted alias."
    )]
    pub mcmc: Option<usize>,

    #[arg(
        long = "posterior",
        value_enum,
        value_name = "MODE",
        help = "Which posterior to sample: off (default) | gate | hyper | both | perdim.\n\
                Omit and pass --mcmc N for the usual case.",
        long_help = "Which posterior to sample after the MAP fit. Chain length comes from `--mcmc N`\n\
                     (default 200 when only this flag is given).\n\
                     \n\
                       off   → no posterior; the run is byte-identical to one without these flags (default).\n\
                       gate  → the per-gene selection posterior only (PIP, credible sets, effect posterior).\n\
                       hyper → the Tier-1 hyperparameters only: the slab variance σ₀² and the sparsity π₀,\n\
                               which the gate otherwise hard-codes, sampled from the data.\n\
                       both  → both, AND the learned σ₀² is fed into the gate's slab prior.\n\
                               That coupling is the hierarchical payoff, so `both` is more than the union\n\
                               of the two — it is what `--mcmc` selects.\n\
                     \n\
                     `--posterior off` together with `--mcmc N` is an error (one turns it off, the other on)."
    )]
    pub posterior: Option<PosteriorMode>,

    #[arg(
        long = "posterior-genes",
        default_value_t = 0,
        value_name = "N",
        help = "Cap the posterior to the top-N genes by observed count (0 = all).",
        long_help = "Limit the posterior to the N genes carrying the most observed count mass\n\
                     (a gene with no counts has nothing for the likelihood to move,\n\
                     so this puts the sampling budget where the posterior is informative).\n\
                     0 (default) samples every gene.\n\
                     Use it to keep an exploratory run short; the un-sampled genes are simply absent\n\
                     from the output tables, not written as null."
    )]
    pub posterior_genes: usize,

    #[arg(
        long = "posterior-partition",
        default_value_t = DEFAULT_PARTITION,
        value_name = "N",
        help = "Cells in the frozen partition that normalizes the Poisson rate (default 1024).",
        long_help = "Size of the frozen negative slate: the number of cells summed in the Poisson\n\
                     rate normalizer, drawn ONCE per run and held fixed (a slate that moved between\n\
                     sweeps would leave the chain with no fixed target).\n\
                     \n\
                     This is the Monte-Carlo RESOLUTION of the normalizer, not a subsample of the data —\n\
                     every observed count still enters the likelihood exactly.\n\
                     Raise it on very large inputs, where the default is thin relative to the cell count;\n\
                     cost grows linearly with it. Clamped to the cell count."
    )]
    pub posterior_partition: usize,
}

/// A resolved posterior request. Reaching this type at all means the posterior
/// is ON — [`PosteriorArgs::resolve`] returns `None` for `off`, so no downstream
/// code has to re-test for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosteriorPlan {
    pub mode: PosteriorMode,
    /// Retained draws per chain; warmup is `n_samples / 2` on top.
    pub n_samples: usize,
    /// Anchors to sample: `0` = all, else the top-N by observed count.
    pub n_genes: usize,
    /// Requested slate size; clamped to the cell count when the index is built.
    pub n_partition: usize,
    /// Base seed, so a reproducible fit gives a reproducible posterior. Each
    /// sampler derives a distinct stream from it.
    pub seed: u64,
}

impl PosteriorArgs {
    /// Reconcile `--posterior` (which posterior) with `--mcmc` / `--jitter` (how
    /// long). `None` means the posterior is off.
    ///
    /// `--mcmc N` alone is the headline spelling — it means "run the sampler for
    /// `N` draws", so it turns the posterior on at [`PosteriorMode::Both`], the
    /// full-Bayes answer. `--posterior` narrows that. Asking for both `off` and a
    /// draw count is a genuine contradiction, not something to resolve silently,
    /// so it errors rather than picking a winner.
    pub fn resolve(&self, seed: u64) -> anyhow::Result<Option<PosteriorPlan>> {
        let mode = match (self.posterior, self.mcmc) {
            (Some(PosteriorMode::Off), Some(n)) => anyhow::bail!(
                "--posterior off contradicts --mcmc {n} (one turns the posterior off, \
                 the other on) — pass one or the other"
            ),
            (Some(PosteriorMode::Off) | None, None) => return Ok(None),
            (Some(mode), _) => mode,
            (None, Some(_)) => PosteriorMode::Both,
        };
        let n_samples = self.mcmc.unwrap_or(DEFAULT_SAMPLES);
        anyhow::ensure!(n_samples > 0, "--mcmc must be > 0 (got {n_samples})");
        Ok(Some(PosteriorPlan {
            mode,
            n_samples,
            n_genes: self.posterior_genes,
            n_partition: self.posterior_partition,
            seed,
        }))
    }
}

/// One gate to sample: which rows form its anchors, what it holds fixed, and what
/// its tables are called.
pub struct TrackSpec<'a> {
    /// Output name stem — `{out}.{tag}_pip.parquet` / `{tag}_posterior.parquet`,
    /// and the key under `"tracks"` in the JSON.
    pub tag: &'a str,
    /// Human name for the logs (`identity β_g`, `velocity δ_g`, …).
    pub label: &'a str,
    /// Row → anchor map; `None` means one anchor per feature row. Rows mapped to
    /// `u32::MAX` are dropped, which is how a track selects its own rows.
    pub row_to_anchor: Option<&'a [u32]>,
    /// Row names for the output tables, one per anchor.
    pub names: &'a [Box<str>],
    /// Parquet row-axis label, matched to whatever table these join against.
    pub axis: &'a str,
    /// `[n_anchors × h]` frozen loading added to the sampled one — the second
    /// effect's offset when the first is held at its MAP. `None` for a plain gate.
    pub offset: Option<&'a [f32]>,
}

/// Run every track's posterior and write the tables plus one
/// `{out}.posterior_hyper.json` covering them all.
///
/// Callers supply the tracks; everything else — calibration, the sampler order,
/// the `σ₀²` adoption guard, the table schema — is fixed here so two CLIs cannot
/// produce tables that disagree.
pub fn run_posterior(
    out_prefix: &str,
    plan: PosteriorPlan,
    unified: &crate::data::UnifiedData,
    frozen: &FrozenMap,
    tracks: &[TrackSpec],
) -> anyhow::Result<()> {
    let n_partition = plan.n_partition.min(unified.n_cells());
    let mut per_track = serde_json::Map::new();
    for track in tracks {
        let summary = run_track(out_prefix, plan, unified, frozen, track, n_partition)?;
        per_track.insert(track.tag.to_string(), summary);
    }

    // `n_partition` is the CLAMPED value, not the request: a 300-cell input asked
    // for 1024 and got 300, and the JSON should record what the sampler used.
    let json = serde_json::json!({
        "mode": format!("{:?}", plan.mode).to_lowercase(),
        "n_samples": plan.n_samples,
        "warmup": plan.n_samples / 2,
        "n_partition": n_partition,
        "seed": plan.seed,
        "interrupted": crate::stop::stop_flag().load(std::sync::atomic::Ordering::Relaxed),
        "tracks": per_track,
    });
    let path = format!("{out_prefix}.posterior_hyper.json");
    // Through `serde_json` rather than a format string: these are floats that can
    // come back non-finite from a stalled chain, and a bare `NaN` is not JSON.
    // serde emits `null`, which parses.
    std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&json)?))
        .with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");
    Ok(())
}

/// Minimum `σ₀²`-chain ESS before the learned slab variance may replace the
/// model's fixed `GATE_EFFECT_PRIOR_VAR`.
///
/// The variance and the effects it scales are strongly dependent (Neal's funnel),
/// and the centered draw mixes badly at large `σ₀²` — a measured ESS of ~3 on
/// multi-donor HCA BM and on BMMC. A funnel-degenerate scale is worse than the
/// fixed prior it would replace, so a chain below this bar is reported and
/// discarded rather than used. Lives next to the sampler that produces the
/// diagnostic, so every caller judges it the same way.
pub const SIGMA2_MIN_ESS: f32 = 10.0;

impl HyperSsResult {
    /// The learned slab variance, or `None` when its chain did not mix well
    /// enough to trust the magnitude (see [`SIGMA2_MIN_ESS`]).
    #[must_use]
    pub fn trustworthy_sigma2(&self) -> Option<f32> {
        (self.sigma_diag.min_ess >= SIGMA2_MIN_ESS
            && self.sigma2_mean.is_finite()
            && self.sigma2_mean > 0.0)
            .then_some(self.sigma2_mean as f32)
    }
}

/// One gate: build its anchored index, sample, write its two tables.
fn run_track(
    out_prefix: &str,
    plan: PosteriorPlan,
    unified: &crate::data::UnifiedData,
    frozen: &FrozenMap,
    track: &TrackSpec,
    n_partition: usize,
) -> anyhow::Result<serde_json::Value> {
    // No intercept calibration happens here, and none is needed: the samplers use
    // `multinomial_ll`, which profiles the anchor intercept out analytically. (An
    // earlier comment claimed the index was "calibrated by construction"; that has
    // been untrue since the profile likelihood landed — `build_index` never
    // calibrated, and `calibrate_anchor_bias` is now reached only from its tests.)
    let grouping = track.row_to_anchor.map(|row_to_anchor| RowGrouping {
        row_to_anchor,
        n_anchors: track.names.len(),
    });
    let idx = build_index(
        unified,
        frozen,
        n_partition,
        plan.seed,
        grouping,
        track.offset,
    )?;
    let n_empty = idx.n_empty_anchors();

    let picked = idx.top_anchors_by_count(plan.n_genes);
    let all_nodes = idx.node_terms();
    let nodes: Vec<_> = picked.iter().map(|&g| all_nodes[g]).collect();
    let side = idx.frozen_side();
    let names: Vec<Box<str>> = picked.iter().map(|&g| track.names[g].clone()).collect();
    // Geometry of the design every anchor is regressed against. Cheap, and it is
    // what says whether a PER-DIM readout is trustworthy at all — see
    // `frozen_diag`. Measured before sampling, so it is reported even for a run
    // that Ctrl+C cuts short.
    let geom = frozen_side_diag(&side);
    info!(
        "  [{}] frozen side: common-mode cos {:.3}, effective rank {:.1} raw / {:.1} centered of {}, \
         max |corr| {:.3}, max VIF {:.2}",
        track.label,
        geom.common_mode_cos,
        geom.eff_rank_raw,
        geom.eff_rank_centered,
        frozen.h,
        geom.max_abs_corr,
        geom.max_vif,
    );
    if geom.max_vif >= 5.0 {
        log::warn!(
            "  [{}] frozen dims are collinear (max VIF {:.1} ≥ 5) — per-dim inclusion \
             probabilities split mass between the correlated dims and can read \
             confidently wrong on both; prefer the gate's credible set",
            track.label,
            geom.max_vif,
        );
    }
    info!(
        "posterior [{}] ({:?}): {} of {} anchors ({n_empty} with no counts), \
         {} draws/chain (+{} warmup), {n_partition}-row partition — \
         Ctrl+C returns partial results",
        track.label,
        plan.mode,
        picked.len(),
        idx.n_anchors(),
        plan.n_samples,
        plan.n_samples / 2,
    );

    // Hypers first: the learned σ₀² feeds the gate's slab prior below, which is
    // the whole reason `both` is more than the union of its parts.
    let hyper = plan.mode.runs_hyper().then(|| {
        let t = std::time::Instant::now();
        let res = hyper_ss(
            &nodes,
            &side,
            &HyperSsConfig::new(plan.n_samples, plan.n_samples / 2, plan.seed ^ 0x1),
        );
        info!(
            "  [{}] hypers: σ₀² = {:.4} (ESS {:.0}), π₀ = {:.4} (ESS {:.0}) in {:.1}s",
            track.label,
            res.sigma2_mean,
            res.sigma_diag.min_ess,
            res.pi0_mean,
            res.pi0_diag.min_ess,
            t.elapsed().as_secs_f32()
        );
        res
    });

    // The warning is phrased by what is at stake in this mode: without a gate to
    // feed there is no prior to have refused, and saying so anyway sends the
    // reader looking for a gate that never ran.
    let learned_sigma2 = hyper.as_ref().and_then(|res| {
        let adopted = res.trustworthy_sigma2();
        if adopted.is_none() {
            let consequence = if plan.mode.runs_gate() {
                "not adopted for the gate prior — keeping the model's fixed slab variance"
            } else {
                "report it as a scale, not a value"
            };
            log::warn!(
                "  [{}] σ₀² = {:.4} is not trustworthy (ESS {:.0} < {SIGMA2_MIN_ESS}, \
                 Neal's funnel): {consequence}",
                track.label,
                res.sigma2_mean,
                res.sigma_diag.min_ess,
            );
        }
        adopted
    });

    let gate = plan.mode.runs_gate().then(|| {
        let mut cfg = GateConfig::new(plan.n_samples, plan.n_samples / 2, plan.seed ^ 0x2);
        if let Some(s2) = learned_sigma2 {
            info!(
                "  [{}] gate slab variance σ₀² ← {s2:.4} (learned)",
                track.label
            );
            cfg.effect_var = s2;
        }
        let t = std::time::Instant::now();
        let res = gate_posterior(&nodes, &side, &cfg);
        let done = res.iter().filter(|g| g.is_sampled()).count();
        info!(
            "  [{}] gate: {done} of {} anchors sampled in {:.1}s",
            track.label,
            res.len(),
            t.elapsed().as_secs_f32()
        );
        if done < res.len() {
            log::warn!(
                "  [{}] interrupted — {} anchor(s) not sampled, written as NaN",
                track.label,
                res.len() - done
            );
        }
        res
    });

    // Per-dim selection: independent inclusion per embedding dim, with its own
    // per-dim `(σ₀h², π₀h)`. Self-contained, so it neither reads `learned_sigma2`
    // nor needs the gate to have run.
    let per_dim = plan.mode.runs_per_dim().then(|| {
        let t = std::time::Instant::now();
        let res = dim_block(
            &nodes,
            &side,
            &DimBlockConfig::new(plan.n_samples, plan.n_samples / 2, plan.seed ^ 0x3),
        );
        info!(
            "  [{}] per-dim: {} anchors × {} dims over {} sweeps in {:.1}s",
            track.label,
            nodes.len(),
            res.h,
            res.n_kept,
            t.elapsed().as_secs_f32()
        );
        res
    });

    if let Some(gate) = &gate {
        write_pip(out_prefix, track, gate, &names, frozen.h)?;
        write_summary(out_prefix, track, gate, hyper.as_ref(), &names)?;
    }
    if let Some(pd) = &per_dim {
        write_dim_pip(out_prefix, track, pd, &names)?;
    }

    Ok(serde_json::json!({
        "n_anchors": picked.len(),
        "sigma2_mean": hyper.as_ref().map(|h| h.sigma2_mean),
        "sigma2_min_ess": hyper.as_ref().map(|h| h.sigma_diag.min_ess),
        "sigma2_stuck_fraction": hyper.as_ref().map(|h| h.sigma_diag.stuck_fraction),
        "pi0_mean": hyper.as_ref().map(|h| h.pi0_mean),
        "pi0_min_ess": hyper.as_ref().map(|h| h.pi0_diag.min_ess),
        "pi0_stuck_fraction": hyper.as_ref().map(|h| h.pi0_diag.stuck_fraction),
        // Did the learned σ₀² actually reach the gate, or was it discarded as
        // funnel-degenerate? Without this the gate's prior is unrecoverable.
        "adopted_sigma2": learned_sigma2,
        "sigma2_min_ess_required": SIGMA2_MIN_ESS,
        // Anchors with no counts: their likelihood is flat, so the posterior is
        // the prior. Reported because it can be over half a full gene annotation.
        "n_empty_anchors": n_empty,
        // Per-dim hypers, one entry per embedding dim. Arrays rather than a table
        // because H is small, and they are the answer to "is this dim used at all".
        "per_dim": per_dim.as_ref().map(|p| serde_json::json!({
            "sigma2": p.sigma2,
            "pi0": p.pi0,
            "sigma2_min_ess": p.sigma_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "pi0_min_ess": p.pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
        })),
        // Geometry of the frozen side (see `frozen_diag`): whether a per-dim
        // readout is trustworthy is a property of THIS, not of the chains.
        // Serialized whole so the JSON cannot drift from the struct.
        "frozen_side": geom,
    }))
}

/// `{out}.{tag}_dimpip.parquet` — `[n_sel × H]` per-dim inclusion probabilities
/// from [`dim_block`].
///
/// Same shape as [`write_pip`] and deliberately a separate file rather than extra
/// columns on it, because the two answer different questions: the gate's row is a
/// simplex over dims (they compete for one effect's worth of mass), while these are
/// independent Bernoullis and a row may sum past 1. Merging them into one table
/// would invite reading a row of one as a row of the other.
fn write_dim_pip(
    out_prefix: &str,
    track: &TrackSpec,
    res: &DimBlockResult,
    names: &[Box<str>],
) -> anyhow::Result<()> {
    write_pip_table(
        out_prefix,
        track,
        "dimpip",
        "independent per-dim inclusion probability — a row MAY sum past 1",
        res.pip.clone(),
        res.h,
        names,
    )
}

/// Write one `[n_sel × H]` selection table.
///
/// `stem` names the file and `what` is the description logged. Those descriptions
/// **must** differ between callers: a `--posterior perdim` run alongside a gate run
/// writes both tables, and reading a row of one as a row of the other is precisely
/// the mistake the separate files exist to prevent — so the log line is the reader's
/// first defence against it.
fn write_pip_table(
    out_prefix: &str,
    track: &TrackSpec,
    stem: &str,
    what: &str,
    flat: Vec<f32>,
    h: usize,
    names: &[Box<str>],
) -> anyhow::Result<()> {
    let t = Tensor::from_vec(flat, (names.len(), h), &Device::Cpu)?;
    let path = format!("{out_prefix}.{}_{stem}.parquet", track.tag);
    save_embedding(&path, &t, names, track.axis)?;
    info!("wrote {path} ({what})");
    Ok(())
}

/// `{out}.{tag}_pip.parquet` — `[n_sel × H]` posterior inclusion probabilities,
/// the calibrated analogue of the variational selection table. An anchor skipped
/// by SIGINT is a NaN row.
fn write_pip(
    out_prefix: &str,
    track: &TrackSpec,
    gate: &[GenePosterior],
    names: &[Box<str>],
    h: usize,
) -> anyhow::Result<()> {
    let mut flat = Vec::with_capacity(gate.len() * h);
    for g in gate {
        match g.is_sampled() {
            true => flat.extend_from_slice(&g.pip),
            false => flat.extend(std::iter::repeat_n(f32::NAN, h)),
        }
    }
    write_pip_table(
        out_prefix,
        track,
        "pip",
        "gate selection posterior — each row is a simplex over dims",
        flat,
        h,
        names,
    )
}

/// `{out}.{tag}_posterior.parquet` — the per-anchor summary a human reads: how
/// concentrated the selection is, how wide the credible set had to be, and how
/// large the effect is.
fn write_summary(
    out_prefix: &str,
    track: &TrackSpec,
    gate: &[GenePosterior],
    hyper: Option<&HyperSsResult>,
    names: &[Box<str>],
) -> anyhow::Result<()> {
    let cols: Vec<Box<str>> = [
        "max_pip",
        "argmax_dim",
        "cs_size",
        "cs_coverage",
        "effect_norm",
        "effect_sd",
        "inclusion_prob",
    ]
    .iter()
    .map(|s| Box::<str>::from(*s))
    .collect();
    let n_col = cols.len();

    let mut flat = Vec::with_capacity(gate.len() * n_col);
    for (i, g) in gate.iter().enumerate() {
        // `inclusion_prob` is indexed by ANCHOR position, like `gate` itself —
        // `hyper_ss` ran on the same picked subset, not on the full axis.
        let incl = hyper.map_or(f32::NAN, |hr| hr.inclusion_prob[i]);
        if !g.is_sampled() {
            flat.extend(std::iter::repeat_n(f32::NAN, n_col - 1));
            flat.push(incl); // the hypers may have finished even if the gate did not
            continue;
        }
        // `min_by` with the comparator reversed keeps the fold's first-wins tie
        // behaviour (`max_by` returns the last) without an `f32::MIN` sentinel.
        let (argmax, max_pip) = g
            .pip
            .iter()
            .copied()
            .enumerate()
            .min_by(|a, b| b.1.total_cmp(&a.1))
            .unwrap_or((0, f32::NAN));
        let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mean_sd = g.std_beta.iter().sum::<f32>() / g.std_beta.len().max(1) as f32;
        flat.extend_from_slice(&[
            max_pip,
            argmax as f32,
            g.credible_set.len() as f32,
            g.cs_coverage,
            l2(&g.mean_beta),
            mean_sd,
            incl,
        ]);
    }
    let t = Tensor::from_vec(flat, (gate.len(), n_col), &Device::Cpu)?;
    let path = format!("{out_prefix}.{}_posterior.parquet", track.tag);
    t.to_parquet_with_names(&path, (Some(names), Some(track.axis)), Some(&cols))?;
    info!("wrote {path} (per-anchor selection posterior summary)");
    Ok(())
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;
