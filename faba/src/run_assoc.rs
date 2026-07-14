//! Entry point for `faba assoc` — modality dynamics along the lineage. Loads the
//! lineage's per-cell pseudotime + branch and a modality site matrix, then runs two
//! Bayesian tests per (site, branch): the **between-branch** contrast (the posterior of a
//! branch's pseudotime-adjusted editing excess vs the other fates — mean effect + 90%
//! credible interval + lfsr, [`crate::assoc::contrast_bayes`]) and the **within-branch**
//! trend GAM (does editing change as the branch progresses — a Bayesian ESS spline by
//! default, or a frequentist quasi-binomial/binomial variant via `--trend-method`). See
//! [`crate::assoc`].
//!
//! When the lineage was annotated (`faba lineage --markers`, which writes
//! `{from}.lineage_annot.membership.tsv`), a **second reporting level** re-runs the same
//! two tests with cells regrouped by their annotated **cell type** instead of by branch —
//! pooling the cells that share a cell type across lineages (`{out}.celltype_*.parquet`).
//! The between-cell-type contrast is clean; the within-cell-type trend is secondary,
//! because pooling divergent lineages of one cell type onto a shared pseudotime axis
//! weakens the "change along the trajectory" reading (noted in its output).
//!
//! ## Output schema
//!
//! Every table is **tidy**: a `site` key (`{gene}/{subunit}`) plus the identity broken out
//! into typed columns, rather than one `/`-joined string a reader has to take apart.
//!
//! ```text
//! branch level    site | gene | subunit | branch (i32)      | <values…>
//! cell-type level site | gene | subunit | cell_type (str)   | <values…>
//! ```
//!
//! The group column differs *because the groups differ*. A branch is an integer the lineage
//! assigned; a cell type is a name a human gave. And a cell-type aggregate pools cells
//! **across** branches, so it has no branch — and now gets no `branch` column, instead of a
//! fabricated one. (Cell-type names are also written verbatim now: the old `/`-joined key
//! forced `/` in a name to be rewritten as `-`, which quietly corrupted Cell-Ontology labels.)
//!
//! The Bayesian tables carry `ess` and `mcse_lfsr` beside `lfsr`. `lfsr` is a Monte-Carlo
//! tail proportion, so a site near `--fdr-alpha` can cross it from one `--seed` to the next;
//! `mcse_lfsr` is that error, per site, so a borderline call is visible in its own row.

use anyhow::Result;
use clap::{Args, ValueEnum};
use log::info;

use matrix_util::common_io::mkdir_parent;
use matrix_util::parquet::{write_named_table, Column};

use crate::assoc::contrast::{bin_pseudotime, site_profile};
use crate::assoc::contrast_bayes::{run_contrasts_bayes, BayesContrastConfig};
use crate::assoc::io::{load_celltypes, load_lineage, load_sites, Lineage, Site};
use crate::assoc::trend::{run_trends, TrendConfig, TrendResult};
use crate::assoc::trend_bayes::{run_trends_bayes, BayesTrendConfig};
use crate::assoc::{BayesResult, Modality};
use faba::hypothesis_tests::benjamini_hochberg;

/// Within-branch trend estimator.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum TrendMethod {
    /// Bayesian spline GAM: Gaussian smoothing prior, ESS posterior, lfsr (default).
    Bayes,
    /// Quasi-binomial spline GAM, F-test.
    Quasi,
    /// Plain binomial spline GAM, deviance LRT (χ²).
    Binomial,
}

#[derive(Args, Debug)]
pub struct AssocArgs {
    #[arg(
        long,
        short = 'f',
        help = "lineage output prefix (reads {from}.pseudotime.parquet)"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 's',
        value_delimiter = ',',
        help = "per-site modality matrices (.zarr.zip), comma-separated"
    )]
    pub sites: Vec<String>,

    #[arg(long, value_enum, help = "modality channel pair to read")]
    pub modality: Modality,

    #[arg(
        long,
        default_value_t = 10,
        help = "pseudotime bins for aligning branches"
    )]
    pub n_bins: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "min total coverage per (site, branch)"
    )]
    pub min_total_coverage: u64,

    #[arg(
        long,
        default_value_t = 10,
        help = "min cells with coverage per (site, branch)"
    )]
    pub min_cells: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed for the Bayesian (ESS) samplers"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 5,
        help = "spline knots for the within-branch trend GAM"
    )]
    pub n_knots: usize,

    #[arg(
        long,
        value_enum,
        default_value = "bayes",
        help = "within-branch trend estimator (bayes | quasi | binomial)"
    )]
    pub trend_method: TrendMethod,

    // The three sampler knobs below drive BOTH Bayesian tests — the between-group contrast
    // as much as the within-group trend — at both reporting levels. They were named
    // `--trend-*`, which reads as trend-only and hid that raising `--trend-samples` is also
    // what sharpens the contrast's `lfsr`. Renamed to say what they do; the old names stay
    // as aliases so existing pipelines keep working.
    #[arg(
        long,
        alias = "trend-prior-sd",
        default_value_t = 3.0,
        help = "bayes: prior sd on the effect coefficients (contrast AND trend)"
    )]
    pub posterior_prior_sd: f32,

    #[arg(
        long,
        alias = "trend-samples",
        default_value_t = 1000,
        help = "bayes: posterior samples per (site, group) — drives the contrast AND the trend"
    )]
    pub posterior_samples: usize,

    #[arg(
        long,
        alias = "trend-warmup",
        default_value_t = 300,
        help = "bayes: sampler warmup (contrast AND trend)"
    )]
    pub posterior_warmup: usize,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "reporting threshold: lfsr (bayes) or BH q (quasi/binomial)",
        long_help = "Reporting threshold — the count in the log, not a filter on the output.\n\n\
            On the Bayesian paths (the default) this is compared against `lfsr`, which is a\n\
            Monte-Carlo tail proportion: a site whose lfsr sits near this cutoff can cross it\n\
            from one --seed to the next. The `mcse_lfsr` column is that error, per site —\n\
            when |lfsr − alpha| is not comfortably larger than mcse_lfsr, the row is\n\
            under-sampled rather than borderline, and the answer is more --posterior-samples.\n\n\
            On the frequentist trend paths (--trend-method quasi|binomial) it is a BH q cutoff."
    )]
    pub fdr_alpha: f32,

    #[arg(
        long,
        conflicts_with = "no_celltype",
        help = "cell<TAB>cell_type TSV for the cell-type-level report \
                (default: {from}.lineage_annot.membership.tsv from `faba lineage --markers`). \
                Unassigned/unmatched cells stay in the contrast 'rest' as background but are \
                not reported as a cell type. Errors if this explicit path is missing."
    )]
    pub celltype_annot: Option<Box<str>>,

    #[arg(long, help = "skip the cell-type-level aggregation report")]
    pub no_celltype: bool,

    #[arg(
        long,
        short = 'o',
        help = "Output prefix (default: the lineage prefix)"
    )]
    pub out: Option<Box<str>>,
}

pub fn run_assoc(args: &AssocArgs) -> Result<()> {
    let out = args.out.as_deref().unwrap_or(&args.from).to_string();
    mkdir_parent(&out)?;

    let lin = load_lineage(&args.from)?;
    info!(
        "assoc: {} cells, {} branches",
        lin.cell_names.len(),
        lin.n_branches
    );
    anyhow::ensure!(
        lin.n_branches >= 2,
        "need ≥ 2 branches for a between-branch contrast"
    );

    let sites = load_sites(&args.sites, args.modality, &lin.cell_names)?;
    info!("loaded {} {} sites", sites.len(), args.modality.token());
    anyhow::ensure!(!sites.is_empty(), "no sites with both channels found");

    // Branch-level report (the primary deliverable). `strict` → error if nothing clears QC.
    run_report(
        &sites,
        &lin,
        args,
        &ReportLevel {
            names: None,
            drop_group: None,
            tag: "branch",
            unit: "branch",
            strict: true,
        },
        &out,
    )?;

    // Cell-type-level report (optional; requires `faba lineage --markers`).
    if !args.no_celltype {
        run_celltype_level(args, &lin, &sites, &out)?;
    }
    Ok(())
}

/// One reporting level's grouping + labelling knobs. The branch level groups cells by
/// principal-curve lineage (`names: None`); the cell-type level groups by annotated cell
/// type (`names: Some`, with the `unassigned` background bucket named in `drop_group`).
struct ReportLevel<'a> {
    /// Group id → display name (cell-type level), or `None` to key rows by `b{group}`.
    names: Option<&'a [Box<str>]>,
    /// A background group kept in the contrast "rest" but omitted from every report.
    drop_group: Option<usize>,
    /// Output-file infix, e.g. `"branch"` → `{out}.branch_contrast.parquet`.
    tag: &'a str,
    /// Log noun, e.g. `"branch"` or `"cell-type"`.
    unit: &'a str,
    /// Error (vs. log) when the contrast finds no QC-passing group — true for the primary level.
    strict: bool,
}

/// Run both association tests for one grouping of `lin` and write the three parquet outputs
/// (`{out}.{tag}_contrast/_profile/_trend.parquet`). Shared by the branch and cell-type
/// levels: they differ only in the grouping (`lin`), the row labels, the dropped background
/// group, and strictness. The between-group contrast is a Bayesian binomial GLM (per-bin
/// baseline conditions out pseudotime; a shrinkage prior stabilises the effect across seeds);
/// the within-group trend GAM asks whether the rate changes along the grouping's axis.
fn run_report(
    sites: &[Site],
    lin: &Lineage,
    args: &AssocArgs,
    level: &ReportLevel,
    out: &str,
) -> Result<()> {
    let keep = |g: usize| level.drop_group != Some(g);

    // Between-group contrast.
    let bcfg = BayesContrastConfig {
        n_bins: args.n_bins,
        min_total_coverage: args.min_total_coverage,
        min_cells: args.min_cells,
        prior_sd: args.posterior_prior_sd,
        n_samples: args.posterior_samples,
        warmup: args.posterior_warmup,
        seed: args.seed,
    };
    let res: Vec<_> = run_contrasts_bayes(sites, lin, &bcfg)
        .into_iter()
        .filter(|r| keep(r.branch))
        .collect();
    if res.is_empty() {
        anyhow::ensure!(!level.strict, "no (site, {}) passed QC", level.unit);
        info!(
            "no (site, {}) passed the between-{} contrast QC",
            level.unit, level.unit
        );
    } else {
        let n_conf = res.iter().filter(|r| r.lfsr < args.fdr_alpha).count();
        info!(
            "between-{} contrast: {} (site, {}); {n_conf} with lfsr < {}",
            level.unit,
            res.len(),
            level.unit,
            args.fdr_alpha
        );
        write_bayes(
            &res,
            sites,
            level,
            &format!("{out}.{}_contrast.parquet", level.tag),
        )?;
        let tested: Vec<usize> = res.iter().map(|r| r.site).collect();
        write_profile(
            &tested,
            sites,
            lin,
            level,
            args.n_bins,
            &format!("{out}.{}_profile.parquet", level.tag),
        )?;
    }

    // Within-group trend: does the rate change *along* the grouping's axis?
    let trend_path = format!("{out}.{}_trend.parquet", level.tag);
    match args.trend_method {
        TrendMethod::Bayes => {
            let tcfg = BayesTrendConfig {
                n_knots: args.n_knots,
                min_total_coverage: args.min_total_coverage,
                min_cells: args.min_cells,
                prior_sd: args.posterior_prior_sd,
                n_samples: args.posterior_samples,
                warmup: args.posterior_warmup,
                seed: args.seed,
            };
            let trends: Vec<_> = run_trends_bayes(sites, lin, &tcfg)
                .into_iter()
                .filter(|r| keep(r.branch))
                .collect();
            if trends.is_empty() {
                info!(
                    "no (site, {}) passed within-{} trend QC",
                    level.unit, level.unit
                );
            } else {
                let n_conf = trends.iter().filter(|r| r.lfsr < args.fdr_alpha).count();
                info!(
                    "{} within-{} Bayesian trends; {n_conf} with lfsr < {}",
                    trends.len(),
                    level.unit,
                    args.fdr_alpha
                );
                write_bayes(&trends, sites, level, &trend_path)?;
            }
        }
        method => {
            let tcfg = TrendConfig {
                n_knots: args.n_knots,
                min_total_coverage: args.min_total_coverage,
                min_cells: args.min_cells,
                overdispersion: method == TrendMethod::Quasi,
            };
            let trends: Vec<_> = run_trends(sites, lin, &tcfg)
                .into_iter()
                .filter(|r| keep(r.branch))
                .collect();
            if trends.is_empty() {
                info!(
                    "no (site, {}) passed within-{} trend QC",
                    level.unit, level.unit
                );
            } else {
                let tps: Vec<f32> = trends.iter().map(|r| r.p_value).collect();
                let tqs = benjamini_hochberg(&tps);
                let t_sig = tqs.iter().filter(|&&q| q < args.fdr_alpha).count();
                info!(
                    "{} within-{} trend tests; {t_sig} at q < {}",
                    trends.len(),
                    level.unit,
                    args.fdr_alpha
                );
                write_trend_freq(&trends, &tqs, sites, level, &trend_path)?;
            }
        }
    }
    Ok(())
}

/// Cell-type-level report: re-pool the lineage cells by their annotated cell type and
/// re-run the same tests via [`run_report`] — the fitting core reads only the integer group
/// id per cell, so swapping the branch grouping for a cell-type grouping needs no change to
/// the models. Requires a cell-type annotation: an explicit `--celltype-annot` must exist and
/// load (errors otherwise), while the auto-detected `{from}.lineage_annot.membership.tsv` is
/// best-effort (missing or unreadable → info/warn + skip, never failing the run whose
/// branch-level outputs are already written).
fn run_celltype_level(args: &AssocArgs, lin: &Lineage, sites: &[Site], out: &str) -> Result<()> {
    let (path, explicit) = match args.celltype_annot.as_deref() {
        Some(p) => (p.to_string(), true),
        None => (format!("{}.lineage_annot.membership.tsv", args.from), false),
    };
    if !std::path::Path::new(&path).exists() {
        anyhow::ensure!(!explicit, "--celltype-annot {path}: file not found");
        info!(
            "cell-type report skipped: no annotation at {path} \
             (run `faba lineage --markers`, or pass --celltype-annot)"
        );
        return Ok(());
    }

    // An explicit override's load errors are fatal; the auto-default's are not (the branch
    // report already succeeded, so a bad membership file should not sink the whole run).
    let ctg = match load_celltypes(&path, &lin.cell_names) {
        Ok(c) => c,
        Err(e) if !explicit => {
            log::warn!("cell-type report skipped: {e:#}");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    let n_types = ctg.names.len();
    let has_bg = ctg.unassigned_id.is_some();
    let n_reported = n_types - usize::from(has_bg);
    info!(
        "cell-type report: {} cells over {n_reported} cell type(s){} from {path}",
        ctg.ids.len(),
        if has_bg {
            " (+ unassigned background)"
        } else {
            ""
        }
    );
    // Need ≥ 2 *reported* cell types for a one-vs-rest contrast; below that there is nothing
    // to contrast (the unassigned background does not count as a fate).
    if n_reported < 2 {
        info!("cell-type report skipped: need ≥ 2 annotated cell types, found {n_reported}");
        return Ok(());
    }

    // Shares pseudotime with `lin`, so `sites` (aligned to that cell order) apply unchanged;
    // only the per-cell group id (branch) becomes the cell type. `cell_names` is unused by the
    // report path, so it is left empty rather than cloning the n barcodes.
    let ct_lin = Lineage {
        cell_names: Vec::new(),
        pseudotime: lin.pseudotime.clone(),
        branch: ctg.ids,
        n_branches: n_types,
    };
    run_report(
        sites,
        &ct_lin,
        args,
        &ReportLevel {
            names: Some(&ctg.names),
            drop_group: ctg.unassigned_id,
            tag: "celltype",
            unit: "cell-type",
            strict: false,
        },
        out,
    )
}

////////////////////
// Output writers //
////////////////////

/// One column of a report table: names are strings, statistics are `f32`, counts and ids are
/// `i32`. The owned sibling of [`Column`], which borrows — the writer has to keep the buffers
/// alive while `write_named_table` reads them.
enum Val {
    Str(Vec<Box<str>>),
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl Val {
    fn as_column(&self) -> Column<'_> {
        match self {
            Val::Str(v) => Column::Str(v),
            Val::F32(v) => Column::F32(v),
            Val::I32(v) => Column::I32(v),
        }
    }
}

/// Write one report table as a **tidy** parquet: a `site` key column, the site's identity
/// broken out into its own columns, the row's group in a column typed for what that group
/// actually *is*, then the value columns.
///
/// The identity used to be one `/`-joined string per row (`{gene}/{subunit}/b3`), which made
/// every consumer re-split it, and forced two things that were never true:
///
/// - **A branch id and a cell-type name shared one slot.** They are not the same kind of
///   thing. A branch is an integer the lineage assigned; a cell type is a name a human gave.
///   Now the branch level writes an integer `branch` column and the cell-type level writes a
///   string `cell_type` column — and a cell-type aggregate, which pools cells *across*
///   branches, gets no `branch` column at all, because it does not have one.
/// - **Cell-type names had to be mangled.** `group_label` used to rewrite `/` to `-` so the
///   name would survive being embedded in a `/`-delimited key. Cell-Ontology labels contain
///   `/` routinely, so that silently corrupted them. In its own column the name is written
///   verbatim.
fn write_report_table(
    path: &str,
    sites: &[Site],
    level: &ReportLevel,
    keys: &[(usize, usize)],
    values: Vec<(&str, Val)>,
) -> Result<()> {
    let site_key: Vec<Box<str>> = keys
        .iter()
        .map(|&(si, _)| format!("{}/{}", sites[si].gene, sites[si].subunit).into_boxed_str())
        .collect();

    // The group column, typed for the level: `branch` is an id, `cell_type` is a name.
    let group = match level.names {
        Some(nm) => (
            "cell_type",
            Val::Str(keys.iter().map(|&(_, g)| nm[g].clone()).collect()),
        ),
        None => (
            "branch",
            Val::I32(keys.iter().map(|&(_, g)| g as i32).collect()),
        ),
    };

    let owned: Vec<(&str, Val)> = [
        (
            "gene",
            Val::Str(keys.iter().map(|&(si, _)| sites[si].gene.clone()).collect()),
        ),
        (
            "subunit",
            Val::Str(
                keys.iter()
                    .map(|&(si, _)| sites[si].subunit.clone())
                    .collect(),
            ),
        ),
        group,
    ]
    .into_iter()
    .chain(values)
    .collect();

    let cols: Vec<(Box<str>, Column)> = owned
        .iter()
        .map(|(name, v)| ((*name).into(), v.as_column()))
        .collect();

    write_named_table(path, "site", &site_key, &cols)?;
    info!("Wrote {path} ({} rows)", site_key.len());
    Ok(())
}

/// Write a Bayesian test's table: the posterior of its linear contrast (mean + sd + 90%
/// credible interval), the local false sign rate, and the two numbers that say how far to
/// trust it — `ess` and `mcse_lfsr`. One row per QC-passing (site, group). No permutation
/// p-value / FWER column: `lfsr` is the report.
///
/// **Both** Bayesian tests write through here, because both report exactly [`BayesResult`] —
/// they differ in *what the contrast is* (the group's log-odds excess vs the pooled rest, for
/// the contrast; the net log-odds change start→end, for the trend), not in what they say
/// about it. One writer is what keeps the two schemas identical, as the module doc promises.
///
/// `lfsr` is a Monte-Carlo tail proportion, so a site sitting near `--fdr-alpha` can cross it
/// from one `--seed` to the next. `mcse_lfsr` is that Monte-Carlo error, per site: when
/// `|lfsr − alpha|` is not comfortably larger than it, the row is under-sampled, not
/// borderline-significant, and the answer is more `--posterior-samples`.
fn write_bayes(res: &[BayesResult], sites: &[Site], level: &ReportLevel, path: &str) -> Result<()> {
    let keys: Vec<(usize, usize)> = res.iter().map(|r| (r.site, r.branch)).collect();
    let f32_col = |f: fn(&BayesResult) -> f32| Val::F32(res.iter().map(f).collect());
    write_report_table(
        path,
        sites,
        level,
        &keys,
        vec![
            (
                "n_cells",
                Val::I32(res.iter().map(|r| r.n_cells as i32).collect()),
            ),
            (
                "total_cov",
                Val::F32(res.iter().map(|r| r.total_cov as f32).collect()),
            ),
            ("effect", f32_col(|r| r.effect)),
            ("effect_sd", f32_col(|r| r.effect_sd)),
            ("effect_lo", f32_col(|r| r.effect_lo)),
            ("effect_hi", f32_col(|r| r.effect_hi)),
            ("lfsr", f32_col(|r| r.lfsr)),
            ("ess", f32_col(|r| r.ess)),
            ("mcse_lfsr", f32_col(|r| r.mcse_lfsr)),
        ],
    )
}

/// The frequentist within-group trend GAM (`--trend-method quasi|binomial`): does the rate
/// change along the group's axis. One row per QC-passing (site, group). No `ess`/`mcse_lfsr`
/// here — this path has no sampler; its uncertainty is the p-value and its BH `q`.
fn write_trend_freq(
    res: &[TrendResult],
    qs: &[f32],
    sites: &[Site],
    level: &ReportLevel,
    path: &str,
) -> Result<()> {
    let keys: Vec<(usize, usize)> = res.iter().map(|r| (r.site, r.branch)).collect();
    write_report_table(
        path,
        sites,
        level,
        &keys,
        vec![
            (
                "n_cells",
                Val::I32(res.iter().map(|r| r.n_cells as i32).collect()),
            ),
            (
                "total_cov",
                Val::F32(res.iter().map(|r| r.total_cov as f32).collect()),
            ),
            ("stat", Val::F32(res.iter().map(|r| r.stat).collect())),
            ("effect", Val::F32(res.iter().map(|r| r.effect).collect())),
            (
                "dispersion",
                Val::F32(res.iter().map(|r| r.dispersion).collect()),
            ),
            ("p_trend", Val::F32(res.iter().map(|r| r.p_value).collect())),
            ("q", Val::F32(qs.to_vec())),
        ],
    )
}

/// The counterfactual divergence of editing along pseudotime, for the tested sites: the
/// per-(group, bin) pseudobulk rate. One row per (site, group, bin) — `bin` is its own
/// integer column rather than a `bin{b}` suffix welded onto the row key.
fn write_profile(
    tested_sites: &[usize],
    sites: &[Site],
    lin: &Lineage,
    level: &ReportLevel,
    n_bins: usize,
    path: &str,
) -> Result<()> {
    let bins = bin_pseudotime(&lin.pseudotime, n_bins);
    let mut which: Vec<usize> = tested_sites.to_vec();
    which.sort_unstable();
    which.dedup();

    let mut keys: Vec<(usize, usize)> = Vec::new();
    let (mut bin, mut kk, mut nn, mut rate) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &si in &which {
        for (l, b, k, ntot) in site_profile(&sites[si], &bins, &lin.branch, n_bins, lin.n_branches)
        {
            if level.drop_group == Some(l) {
                continue; // background bucket kept in the contrast rest but not reported
            }
            keys.push((si, l));
            bin.push(b as i32);
            kk.push(k as f32);
            nn.push(ntot as f32);
            rate.push(k as f32 / ntot as f32);
        }
    }
    write_report_table(
        path,
        sites,
        level,
        &keys,
        vec![
            ("bin", Val::I32(bin)),
            ("K", Val::F32(kk)),
            ("N", Val::F32(nn)),
            ("rate", Val::F32(rate)),
        ],
    )
}

#[cfg(test)]
mod tests;
