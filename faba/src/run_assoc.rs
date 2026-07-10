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

use anyhow::Result;
use clap::{Args, ValueEnum};
use log::info;

use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use crate::assoc::contrast::{bin_pseudotime, site_profile};
use crate::assoc::contrast_bayes::{run_contrasts_bayes, BayesContrastConfig, BayesContrastResult};
use crate::assoc::io::{load_celltypes, load_lineage, load_sites, Lineage, Site};
use crate::assoc::trend::{run_trends, TrendConfig, TrendResult};
use crate::assoc::trend_bayes::{run_trends_bayes, BayesTrendConfig, BayesTrendResult};
use crate::assoc::Modality;
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

    #[arg(
        long,
        default_value_t = 3.0,
        help = "bayes trend: prior sd on the spline coefficients"
    )]
    pub trend_prior_sd: f32,

    #[arg(
        long,
        default_value_t = 1000,
        help = "bayes trend: posterior samples per (site, branch)"
    )]
    pub trend_samples: usize,

    #[arg(long, default_value_t = 300, help = "bayes trend: ESS warmup")]
    pub trend_warmup: usize,

    #[arg(long, default_value_t = 0.1, help = "BH FDR alpha for reporting")]
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
        prior_sd: args.trend_prior_sd,
        n_samples: args.trend_samples,
        warmup: args.trend_warmup,
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
        write_branch_contrast_bayes(
            &res,
            sites,
            level,
            &format!("{out}.{}_contrast.parquet", level.tag),
        )?;
        let tested: Vec<usize> = res.iter().map(|r| r.site).collect();
        write_branch_profile(
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
                prior_sd: args.trend_prior_sd,
                n_samples: args.trend_samples,
                warmup: args.trend_warmup,
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
                write_branch_trend_bayes(&trends, sites, level, &trend_path)?;
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
                write_branch_trend(&trends, &tqs, sites, level, &trend_path)?;
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

/// Display label for group `g`: the cell-type name (with any `/` neutralised to `-`, since
/// the row key is `/`-delimited) when the level is labelled, else `b{g}` (branch level).
fn group_label(names: Option<&[Box<str>]>, g: usize) -> String {
    match names {
        Some(nm) => nm[g].replace('/', "-"),
        None => format!("b{g}"),
    }
}

/// Row key for a (site, group) record: `{gene}/{subunit}/{group_label}`.
fn site_group_key(
    sites: &[Site],
    site: usize,
    group: usize,
    names: Option<&[Box<str>]>,
) -> Box<str> {
    let s = &sites[site];
    format!("{}/{}/{}", s.gene, s.subunit, group_label(names, group)).into_boxed_str()
}

/// Assemble a group-indexed table (`rows` × `col_names`, values row-major in `vals`) under
/// the `index_name` row key and write it to Parquet. Shared by the three site-group writers.
fn write_site_branch_table(
    path: &str,
    rows: Vec<Box<str>>,
    index_name: &str,
    col_names: &[&str],
    vals: &[Vec<f32>],
) -> Result<()> {
    let mut mat = DMatrix::<f32>::zeros(rows.len(), col_names.len());
    for (i, v) in vals.iter().enumerate() {
        for (j, &x) in v.iter().enumerate() {
            mat[(i, j)] = x;
        }
    }
    let cols: Vec<Box<str>> = col_names.iter().map(|s| (*s).into()).collect();
    mat.to_parquet_with_names(path, (Some(&rows), Some(index_name)), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// `{gene}/{chr:pos}/b{branch}` × [n_cells, total_cov, effect, effect_sd, effect_lo, effect_hi,
/// lfsr] — the **Bayesian** between-branch contrast: posterior of the pseudotime-adjusted branch
/// log-odds excess (mean + 90% credible interval) and the local false sign rate, one row per
/// QC-passing (site, branch). No permutation p-value / FWER column — `lfsr` is the report.
fn write_branch_contrast_bayes(
    res: &[BayesContrastResult],
    sites: &[Site],
    level: &ReportLevel,
    path: &str,
) -> Result<()> {
    let rows = res
        .iter()
        .map(|r| site_group_key(sites, r.site, r.branch, level.names))
        .collect();
    let vals: Vec<Vec<f32>> = res
        .iter()
        .map(|r| {
            vec![
                r.n_cells as f32,
                r.total_cov as f32,
                r.effect,
                r.effect_sd,
                r.effect_lo,
                r.effect_hi,
                r.lfsr,
            ]
        })
        .collect();
    write_site_branch_table(
        path,
        rows,
        &format!("site_{}", level.tag),
        &[
            "n_cells",
            "total_cov",
            "effect",
            "effect_sd",
            "effect_lo",
            "effect_hi",
            "lfsr",
        ],
        &vals,
    )
}

/// `{gene}/{chr:pos}/b{branch}` × [n_cells, total_cov, stat, effect, dispersion,
/// p_trend, q] — the within-branch association GAM (does the rate change along the
/// branch), one row per QC-passing (site, branch).
fn write_branch_trend(
    res: &[TrendResult],
    qs: &[f32],
    sites: &[Site],
    level: &ReportLevel,
    path: &str,
) -> Result<()> {
    let rows = res
        .iter()
        .map(|r| site_group_key(sites, r.site, r.branch, level.names))
        .collect();
    let vals: Vec<Vec<f32>> = res
        .iter()
        .enumerate()
        .map(|(i, r)| {
            vec![
                r.n_cells as f32,
                r.total_cov as f32,
                r.stat,
                r.effect,
                r.dispersion,
                r.p_value,
                qs[i],
            ]
        })
        .collect();
    write_site_branch_table(
        path,
        rows,
        &format!("site_{}", level.tag),
        &[
            "n_cells",
            "total_cov",
            "stat",
            "effect",
            "dispersion",
            "p_trend",
            "q",
        ],
        &vals,
    )
}

/// `{gene}/{chr:pos}/b{branch}` × [n_cells, total_cov, effect, effect_sd, effect_lo,
/// effect_hi, lfsr] — the Bayesian within-branch association (posterior net log-odds
/// change along the branch + local false sign rate), one row per QC-passing (site,
/// branch).
fn write_branch_trend_bayes(
    res: &[BayesTrendResult],
    sites: &[Site],
    level: &ReportLevel,
    path: &str,
) -> Result<()> {
    let rows = res
        .iter()
        .map(|r| site_group_key(sites, r.site, r.branch, level.names))
        .collect();
    let vals: Vec<Vec<f32>> = res
        .iter()
        .map(|r| {
            vec![
                r.n_cells as f32,
                r.total_cov as f32,
                r.effect,
                r.effect_sd,
                r.effect_lo,
                r.effect_hi,
                r.lfsr,
            ]
        })
        .collect();
    write_site_branch_table(
        path,
        rows,
        &format!("site_{}", level.tag),
        &[
            "n_cells",
            "total_cov",
            "effect",
            "effect_sd",
            "effect_lo",
            "effect_hi",
            "lfsr",
        ],
        &vals,
    )
}

/// `{gene}/{chr:pos}/b{branch}/bin{b}` × [bin, K, N, rate] — the counterfactual
/// divergence of editing along pseudotime, for the tested sites.
fn write_branch_profile(
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

    let mut rows: Vec<Box<str>> = Vec::new();
    let mut vals: Vec<[f32; 4]> = Vec::new();
    for &si in &which {
        let s = &sites[si];
        for (l, b, k, ntot) in site_profile(s, &bins, &lin.branch, n_bins, lin.n_branches) {
            if level.drop_group == Some(l) {
                continue; // background bucket kept in the contrast rest but not reported
            }
            let group = group_label(level.names, l);
            rows.push(format!("{}/{}/{group}/bin{b}", s.gene, s.subunit).into_boxed_str());
            vals.push([b as f32, k as f32, ntot as f32, k as f32 / ntot as f32]);
        }
    }
    let mut mat = DMatrix::<f32>::zeros(rows.len(), 4);
    for (i, v) in vals.iter().enumerate() {
        for j in 0..4 {
            mat[(i, j)] = v[j];
        }
    }
    let cols: Vec<Box<str>> = ["bin", "K", "N", "rate"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    mat.to_parquet_with_names(
        path,
        (
            Some(&rows),
            Some(format!("site_{}_bin", level.tag).as_str()),
        ),
        Some(&cols),
    )?;
    info!("Wrote {path}");
    Ok(())
}
