//! Entry point for `faba assoc` — modality dynamics along the lineage. Loads the
//! lineage's per-cell pseudotime + branch and a modality site matrix, then runs two
//! tests per (site, branch): the counterfactual **between-branch** contrast (editing
//! vs the other-fate cells at matched pseudotime) and the **within-branch** trend GAM
//! (does editing change as the branch progresses — frequentist quasi-binomial/binomial
//! or a Bayesian ESS variant, via `--trend-method`). See [`crate::assoc`].

use anyhow::Result;
use clap::{Args, ValueEnum};
use log::{info, warn};

use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use crate::assoc::contrast::{
    bin_pseudotime, run_contrasts, site_profile, AssocConfig, BranchResult,
};
use crate::assoc::io::{load_lineage, load_sites, Lineage, Site};
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
        default_value_t = 500,
        help = "branch-label permutations within pseudotime bins"
    )]
    pub num_perm: usize,

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

    #[arg(long, default_value_t = 42, help = "RNG seed for permutations")]
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
        default_value_t = 800,
        help = "bayes trend: posterior samples per (site, branch)"
    )]
    pub trend_samples: usize,

    #[arg(long, default_value_t = 300, help = "bayes trend: ESS warmup")]
    pub trend_warmup: usize,

    #[arg(long, default_value_t = 0.1, help = "BH FDR alpha for reporting")]
    pub fdr_alpha: f32,

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

    let cfg = AssocConfig {
        n_bins: args.n_bins,
        num_perm: args.num_perm,
        min_total_coverage: args.min_total_coverage,
        min_cells: args.min_cells,
        seed: args.seed,
    };
    let results = run_contrasts(&sites, &lin, &cfg);
    anyhow::ensure!(!results.is_empty(), "no (site, branch) passed QC");

    let ps: Vec<f32> = results.iter().map(|r| r.p_perm).collect();
    let qs = benjamini_hochberg(&ps);
    let n_sig = qs.iter().filter(|&&q| q < args.fdr_alpha).count();
    let n_fwer = results.iter().filter(|r| r.p_fwer < args.fdr_alpha).count();
    info!(
        "{} (site,branch) tests; {n_sig} at BH q < {a}, {n_fwer} at Westfall–Young FWER < {a}",
        results.len(),
        a = args.fdr_alpha
    );
    // Westfall–Young cannot resolve a p below 1/(B+1): a test that beats every
    // permutation pins to that floor, so its FWER is a resolution limit, not a fitted
    // value. Flag it so the user can raise --num-perm for finer calibration.
    let fwer_floor = 1.0 / (1.0 + args.num_perm as f32);
    let n_floor = results
        .iter()
        .filter(|r| r.p_fwer <= fwer_floor * 1.000_1)
        .count();
    if n_floor > 0 {
        warn!(
            "{n_floor} (site,branch) hit the FWER resolution floor 1/(B+1) = {fwer_floor:.2e} \
             (observed beats all {} permutations); raise --num-perm for a finer FWER estimate",
            args.num_perm
        );
    }

    write_branch_contrast(
        &results,
        &qs,
        &sites,
        &format!("{out}.branch_contrast.parquet"),
    )?;
    write_branch_profile(
        &results,
        &sites,
        &lin,
        args.n_bins,
        &format!("{out}.branch_profile.parquet"),
    )?;

    // Within-branch association: does the rate change *along* each branch?
    let trend_path = format!("{out}.branch_trend.parquet");
    match args.trend_method {
        TrendMethod::Bayes => {
            let bcfg = BayesTrendConfig {
                n_knots: args.n_knots,
                min_total_coverage: args.min_total_coverage,
                min_cells: args.min_cells,
                prior_sd: args.trend_prior_sd,
                n_samples: args.trend_samples,
                warmup: args.trend_warmup,
                seed: args.seed,
            };
            let trends = run_trends_bayes(&sites, &lin, &bcfg);
            if trends.is_empty() {
                info!("no (site, branch) passed within-branch trend QC");
            } else {
                let n_conf = trends.iter().filter(|r| r.lfsr < args.fdr_alpha).count();
                info!(
                    "{} within-branch Bayesian trends; {n_conf} with lfsr < {}",
                    trends.len(),
                    args.fdr_alpha
                );
                write_branch_trend_bayes(&trends, &sites, &trend_path)?;
            }
        }
        method => {
            let tcfg = TrendConfig {
                n_knots: args.n_knots,
                min_total_coverage: args.min_total_coverage,
                min_cells: args.min_cells,
                overdispersion: method == TrendMethod::Quasi,
            };
            let trends = run_trends(&sites, &lin, &tcfg);
            if trends.is_empty() {
                info!("no (site, branch) passed within-branch trend QC");
            } else {
                let tps: Vec<f32> = trends.iter().map(|r| r.p_value).collect();
                let tqs = benjamini_hochberg(&tps);
                let t_sig = tqs.iter().filter(|&&q| q < args.fdr_alpha).count();
                info!(
                    "{} within-branch trend tests; {t_sig} at q < {}",
                    trends.len(),
                    args.fdr_alpha
                );
                write_branch_trend(&trends, &tqs, &sites, &trend_path)?;
            }
        }
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Output writers
////////////////////////////////////////////////////////////////////////

/// `{gene}/{subunit}/b{branch}` row key for a (site, branch) record.
fn site_branch_key(sites: &[Site], site: usize, branch: usize) -> Box<str> {
    let s = &sites[site];
    format!("{}/{}/b{}", s.gene, s.subunit, branch).into_boxed_str()
}

/// Assemble a `site_branch`-indexed table (`rows` × `col_names`, values row-major in
/// `vals`) and write it to Parquet. Shared by the three site-branch writers.
fn write_site_branch_table(
    path: &str,
    rows: Vec<Box<str>>,
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
    mat.to_parquet_with_names(path, (Some(&rows), Some("site_branch")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}

/// `{gene}/{chr:pos}/b{branch}` × [n_cells, total_cov, stat, effect, p_perm, p_fwer, q].
/// `p_fwer` is the Westfall–Young step-down min-P FWER-adjusted p; `q` is BH-FDR.
fn write_branch_contrast(
    res: &[BranchResult],
    qs: &[f32],
    sites: &[Site],
    path: &str,
) -> Result<()> {
    let rows = res
        .iter()
        .map(|r| site_branch_key(sites, r.site, r.branch))
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
                r.p_perm,
                r.p_fwer,
                qs[i],
            ]
        })
        .collect();
    write_site_branch_table(
        path,
        rows,
        &[
            "n_cells",
            "total_cov",
            "stat",
            "effect",
            "p_perm",
            "p_fwer",
            "q",
        ],
        &vals,
    )
}

/// `{gene}/{chr:pos}/b{branch}` × [n_cells, total_cov, stat, effect, dispersion,
/// p_trend, q] — the within-branch association GAM (does the rate change along the
/// branch), one row per QC-passing (site, branch).
fn write_branch_trend(res: &[TrendResult], qs: &[f32], sites: &[Site], path: &str) -> Result<()> {
    let rows = res
        .iter()
        .map(|r| site_branch_key(sites, r.site, r.branch))
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
fn write_branch_trend_bayes(res: &[BayesTrendResult], sites: &[Site], path: &str) -> Result<()> {
    let rows = res
        .iter()
        .map(|r| site_branch_key(sites, r.site, r.branch))
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
    res: &[BranchResult],
    sites: &[Site],
    lin: &Lineage,
    n_bins: usize,
    path: &str,
) -> Result<()> {
    let bins = bin_pseudotime(&lin.pseudotime, n_bins);
    let mut which: Vec<usize> = res.iter().map(|r| r.site).collect();
    which.sort_unstable();
    which.dedup();

    let mut rows: Vec<Box<str>> = Vec::new();
    let mut vals: Vec<[f32; 4]> = Vec::new();
    for &si in &which {
        let s = &sites[si];
        for (l, b, k, ntot) in site_profile(s, &bins, &lin.branch, n_bins, lin.n_branches) {
            rows.push(format!("{}/{}/b{l}/bin{b}", s.gene, s.subunit).into_boxed_str());
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
    mat.to_parquet_with_names(path, (Some(&rows), Some("site_branch_bin")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
}
