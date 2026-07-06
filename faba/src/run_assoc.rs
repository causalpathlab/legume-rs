//! Entry point for `faba assoc` — counterfactual between-branch modality contrast
//! along the lineage. Loads the lineage's per-cell pseudotime + branch, a modality
//! site matrix, and tests per (site, branch) whether editing diverges from the
//! other-fate cells at matched pseudotime (see [`crate::assoc`]).

use anyhow::Result;
use clap::Args;
use log::info;

use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use crate::assoc::contrast::{
    bin_pseudotime, run_contrasts, site_profile, AssocConfig, BranchResult,
};
use crate::assoc::io::{load_lineage, load_sites, Lineage, Site};
use crate::assoc::Modality;
use faba::hypothesis_tests::benjamini_hochberg;

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
    info!(
        "{} (site,branch) tests; {n_sig} at q < {}",
        results.len(),
        args.fdr_alpha
    );

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
    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Output writers
////////////////////////////////////////////////////////////////////////

/// `{gene}/{chr:pos}/b{branch}` × [n_cells, total_cov, stat, effect, p_perm, q].
fn write_branch_contrast(
    res: &[BranchResult],
    qs: &[f32],
    sites: &[Site],
    path: &str,
) -> Result<()> {
    let mut mat = DMatrix::<f32>::zeros(res.len(), 6);
    let mut rows: Vec<Box<str>> = Vec::with_capacity(res.len());
    for (i, r) in res.iter().enumerate() {
        let s = &sites[r.site];
        rows.push(format!("{}/{}/b{}", s.gene, s.subunit, r.branch).into_boxed_str());
        mat[(i, 0)] = r.n_cells as f32;
        mat[(i, 1)] = r.total_cov as f32;
        mat[(i, 2)] = r.stat;
        mat[(i, 3)] = r.effect;
        mat[(i, 4)] = r.p_perm;
        mat[(i, 5)] = qs[i];
    }
    let cols: Vec<Box<str>> = ["n_cells", "total_cov", "stat", "effect", "p_perm", "q"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    mat.to_parquet_with_names(path, (Some(&rows), Some("site_branch")), Some(&cols))?;
    info!("Wrote {path}");
    Ok(())
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
