//! The `--posterior` / `--mcmc` step, end to end: flags, orchestration, outputs.
//!
//! Both `senna bge` and `faba gem` run the *same* pipeline over a fitted model —
//! build the frozen index, sample the per-dim selection posterior, write the tables.
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
use super::index::{build_index, FrozenMap, RowGrouping};
use crate::eval::save_embedding;
use anyhow::Context;
use candle_util::candle_core::{Device, Tensor};
use log::info;

/// Default retained draws per chain when `--posterior` is given without `--mcmc`.
pub const DEFAULT_SAMPLES: usize = 200;

/// Default frozen negative-slate size — the number of other-side rows summed in
/// the Poisson rate normalizer. It is the Monte-Carlo resolution of that
/// normalizer, not a subsample of the data: every observed count still enters
/// exactly.
pub const DEFAULT_PARTITION: usize = 1024;

/// The `--posterior` / `--mcmc` flag group, flattened by each CLI's args struct.
///
/// The help is written to cover both callers: on a model with more than one gate
/// (gem's identity + velocity) every gate is sampled, which is what the flag
/// means there, so one wording serves both without saying anything false.
#[derive(clap::Args, Debug, Clone)]
pub struct PosteriorArgs {
    #[arg(
        long = "posterior",
        alias = "mcmc",
        alias = "jitter",
        value_name = "N",
        num_args = 0..=1,
        default_missing_value = "200",
        help = "Sample the posterior over feature selection, N retained draws per chain\n\
                (bare --posterior uses 200). Off unless given.",
        long_help = "Sample the exact posterior over which embedding dims each gene loads, and write\n\
                     the tables. N is the number of RETAINED draws per chain (warmup is N/2 more),\n\
                     the sampler's analogue of `-i/--epochs`. Bare `--posterior` uses 200.\n\
                     Omit it entirely and the run is byte-identical to one built before this existed.\n\
                     \n\
                     Training fits feature selection VARIATIONALLY and reports a point estimate;\n\
                     this samples it against the frozen MAP cell side, so selection comes with\n\
                     calibrated uncertainty: a per-dim posterior inclusion probability per gene,\n\
                     plus the per-dim slab variance σ₀h² and sparsity π₀h learned from the data\n\
                     rather than hard-coded.\n\
                     \n\
                     Inclusion is INDEPENDENT per dim, matching the axis the gate is trained on\n\
                     (a softmax over genes within a dim), so a gene may load several dims at once\n\
                     and its row does NOT sum to 1.\n\
                     \n\
                     Every gate the model has is sampled. `faba gem` has two — the identity gate on β_g\n\
                     (read from the spliced rows) and the velocity gate on δ_g (the unspliced rows, with β_g\n\
                     held at its MAP, since an unspliced row scores ⟨β_g + δ_g, e_c⟩);\n\
                     `senna bge` has only the identity gate.\n\
                     \n\
                     Cost scales with H likelihood passes per anchor per sweep, so a whole dictionary\n\
                     runs for minutes — Ctrl+C returns partial results. Use --posterior-genes to cap it.\n\
                     Writes one per-dim PIP parquet per gate, plus {out}.posterior_hyper.json.\n\
                     `--mcmc` and `--jitter` are accepted aliases."
    )]
    pub posterior: Option<usize>,

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
    /// Resolve `--posterior [N]` into a plan. `None` means the posterior is off,
    /// so no downstream code has to re-test for it.
    pub fn resolve(&self, seed: u64) -> anyhow::Result<Option<PosteriorPlan>> {
        let Some(n_samples) = self.posterior else {
            return Ok(None);
        };
        anyhow::ensure!(n_samples > 0, "--posterior must be > 0 (got {n_samples})");
        Ok(Some(PosteriorPlan {
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
             confidently wrong on both — read a gene's row as a profile, not a winner",
            track.label,
            geom.max_vif,
        );
    }
    info!(
        "posterior [{}]: {} of {} anchors ({n_empty} with no counts), \
         {} draws/chain (+{} warmup), {n_partition}-row partition — \
         Ctrl+C returns partial results",
        track.label,
        picked.len(),
        idx.n_anchors(),
        plan.n_samples,
        plan.n_samples / 2,
    );

    // Per-dim selection: independent inclusion per embedding dim, with its own
    // per-dim `(σ₀h², π₀h)`, learned from the data rather than hard-coded.
    let per_dim = {
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
    };
    write_dim_pip(out_prefix, track, &per_dim, &names)?;

    Ok(serde_json::json!({
        "n_anchors": picked.len(),
        // Per-dim hypers, one entry per embedding dim. Arrays rather than a table
        // because H is small, and they are the answer to "is this dim used at all".
        "per_dim": {
            "sigma2": per_dim.sigma2,
            "pi0": per_dim.pi0,
            "sigma2_min_ess": per_dim.sigma_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "pi0_min_ess": per_dim.pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
        },
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

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;
