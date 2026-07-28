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
                     Every anchor carrying at least one count is sampled; anchors with none have a\n\
                     flat likelihood, so they are skipped rather than redrawing their prior.\n\
                     Cost scales with H likelihood passes per anchor per sweep, so a whole dictionary\n\
                     runs for minutes — Ctrl+C returns partial results, and a smaller N is the way\n\
                     to shorten an exploratory run.\n\
                     Writes one PIP parquet per gate, plus {out}.posterior_hyper.json.\n\
                     `--mcmc` and `--jitter` are accepted aliases."
    )]
    pub posterior: Option<usize>,
}

/// A resolved posterior request. Reaching this type at all means the posterior
/// is ON — [`PosteriorArgs::resolve`] returns `None` for `off`, so no downstream
/// code has to re-test for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosteriorPlan {
    /// Retained draws per chain; warmup is `n_samples / 2` on top.
    pub n_samples: usize,
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
        Ok(Some(PosteriorPlan { n_samples, seed }))
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
    // Derived, not configured: the slate is the Monte-Carlo resolution of the rate
    // normalizer, and `DEFAULT_PARTITION` rows is enough at any realistic cell count
    // (cost grows linearly with it, accuracy barely does). Clamped to the cell count
    // so a small input sums its whole other side exactly.
    let n_partition = DEFAULT_PARTITION.min(unified.n_cells());
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

    // Every anchor that carries information. An anchor with no counts has a FLAT
    // likelihood (`multinomial_ll` returns 0), so sampling it just redraws the prior
    // — on real data that was over half the axis (18,666 of 34,008 on BMMC), i.e.
    // most of the budget spent learning nothing. Skipping them is not a coverage
    // choice, so it is not a flag.
    let picked = idx.informative_anchors();
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
    write_pip(out_prefix, track, &per_dim, &names)?;

    Ok(serde_json::json!({
        // Sampled anchors, and the ones skipped because they carry no counts. Both,
        // so a reader can see what fraction of the axis the posterior covers — a
        // bare count of sampled anchors reads as the whole axis when it is often
        // barely half of it.
        "n_anchors": picked.len(),
        "n_skipped_empty": n_empty,
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

/// `{out}.{tag}_pip.parquet` — `[n_sel × H]` per-dim posterior inclusion
/// probabilities from [`dim_block`], one row per sampled anchor.
///
/// Read it per ENTRY, not per row. Inclusion is independent per dim, so a row does
/// not sum to 1 and may sum past it — a gene is free to load several dims. (It was
/// briefly `_dimpip` to distinguish it from a gate table that no longer ships.)
fn write_pip(
    out_prefix: &str,
    track: &TrackSpec,
    res: &DimBlockResult,
    names: &[Box<str>],
) -> anyhow::Result<()> {
    let t = Tensor::from_vec(res.pip.clone(), (names.len(), res.h), &Device::Cpu)?;
    let path = format!("{out_prefix}.{}_pip.parquet", track.tag);
    save_embedding(&path, &t, names, track.axis)?;
    info!("wrote {path} (per-dim inclusion probability — a row MAY sum past 1)");
    Ok(())
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod run_tests;
