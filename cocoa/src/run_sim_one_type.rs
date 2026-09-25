// #![allow(dead_code)]
use crate::common::*;

use rustc_hash::FxHashMap as HashMap;
use std::ops::Div;

use clap::Parser;
use indicatif::ParallelProgressIterator;
use legume_numeric::matrix::common_io::{mkdir_parent, write_lines, write_types};
use legume_numeric::matrix::mtx_io;
use legume_numeric::matrix::traits::{IoOps, MatOps, SampleOps};
use log::info;
use rand::SeedableRng;
use rand_distr::{weighted::WeightedIndex, Distribution, Gamma, Normal, Poisson, Uniform};

use rayon::prelude::*;

// Numerical stability constants
const MIN_VARIANCE: f32 = 1e-8;
const MIN_LAMBDA: f32 = 1e-8;
const EXPRESSION_THRESHOLD: f32 = 0.5;
const DEFAULT_EFFECT_SIZE: f32 = 10.0;

struct GlmSimulator {
    n_indv: usize,
    n_cells_per_indv: usize,
    n_covar: usize,
    n_exp_cat: usize,
    n_genes: usize,
    n_causal_genes: usize,
    pve_exposure: f32,
    pve_gene: f32,
    pve_covar: f32,
    effect_size: f32,
    rseed: u64,
    depth_gamma_hyperparam: (f32, f32),
    /// SD of the per-gene log baseline rate (0 = every gene at the same level).
    gene_mean_sd: f32,
    /// `(a, b, s0)`: log phi_g = a + b log m_g + N(0, s0^2), with the
    /// individual effect delta_gi ~ Gamma(phi_g, phi_g). `None` keeps the
    /// Gaussian residual.
    indv_dispersion: Option<(f32, f32, f32)>,
}

struct GlmOut {
    data_mn: Mat,
    sample_to_exposure: Vec<(usize, usize)>,
    confounder_nk: Mat,
    causal_m: Vec<(usize, usize)>,
    /// Per gene `(m_g, phi_g)`: baseline mean rate per cell and the planted
    /// between-individual dispersion; `None` without a dispersion trend.
    dispersion_m: Option<Vec<(f32, f32)>>,
    /// Planted parts of each gene's log rate, gene x individual: the
    /// exposure effect (zero for non-causal genes) and the confounding term.
    exposure_part_mn: Mat,
    confounding_part_mn: Mat,
}

/// One simulated gene: its log rate over individuals and, with a planted
/// dispersion trend, its baseline mean rate and phi.
struct GeneDraw {
    gene: usize,
    log_rate: Mat,
    planted: Option<(f32, f32)>,
    exposure_part: Mat,
    confounding_part: Mat,
}

struct TripletsOut {
    mtx_shape: (usize, usize, usize),
    triplets: Vec<(u64, u64, f32)>,
    samples: Vec<usize>,
}

/// `N(0, sd)` draw, or exactly zero when `sd` is zero.
/// Seed tag offset for the per-individual cell streams.
const CELL_SEED_TAG: u64 = 1 << 32;

fn normal_or_zero(rng: &mut rand::rngs::StdRng, sd: f32) -> f32 {
    if sd > 0.0 {
        Normal::new(0.0, sd).expect("positive sd").sample(rng)
    } else {
        0.0
    }
}

impl GlmSimulator {
    ///
    /// Biased assignment mechanisms
    /// `logits x(i,c) ~ sum w(i,k) α(k,c) + ε(i)`
    ///
    /// Confounded data generation for gene g and sample i
    /// `log y(i,g) ~ sum x(i,c) β(c,g) + sum w(i,k) γ(k,g) + ε(i,g)`
    ///
    fn generate_individual_glm(&self) -> anyhow::Result<GlmOut> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rseed);
        // one stream per draw, all from `rseed`
        let seed = |tag: u64| legume_numeric::matrix::rand_util::mix_seed(self.rseed, tag);

        // 1. Generate confounding factors
        let confounder_nk = Mat::rnorm_seeded(self.n_indv, self.n_covar, seed(1));

        // 2. Generate multinomial exposure assignment (sample x gene)
        // x(i,c) ~ multinomial( sum w(i,k) * effect(k,c) + eps )
        let effect_kc = Mat::rnorm_seeded(self.n_covar, self.n_exp_cat, seed(2));

        let logits_nc = Mat::rnorm_seeded(self.n_indv, self.n_exp_cat, seed(3))
            * (1. - self.pve_exposure)
            + (&confounder_nk * effect_kc).scale_columns() * self.pve_exposure;

        let assignment_n = sample_logits_each_row(logits_nc, &mut rng)?;

        let sample_to_exposure = assignment_n
            .clone()
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();

        // 3. Sample generalized linear model
        // 3.a. Pick causal genes
        let runif_gene = Uniform::new(0, self.n_genes)?;
        let runif_cat = Uniform::new(0, self.n_exp_cat)?;

        let effect_size = self.effect_size;
        let causal_genes: HashMap<usize, (usize, Mat)> = (0..self.n_causal_genes)
            .map(|_| {
                let gene = runif_gene.sample(&mut rng);
                let cat = runif_cat.sample(&mut rng);
                let ret_n = Mat::from_iterator(
                    1,
                    self.n_indv,
                    assignment_n
                        .iter()
                        .map(|&c| if c == cat { effect_size } else { -effect_size }),
                );

                let mu = ret_n.mean();
                let sig = ret_n.variance().sqrt().max(MIN_VARIANCE);

                (gene, (cat, ret_n.map(|x| (x - mu).div(sig))))
            })
            .collect();

        // 3.b. Generate individual-level data with confounding effects
        let mut data: Vec<GeneDraw> = (0..self.n_genes)
            .into_par_iter()
            .map(|g| {
                // Every draw for this gene comes from one seeded stream, so
                // the gene loop is reproducible whatever its thread order.
                let gene_seed = legume_numeric::matrix::rand_util::mix_seed(self.rseed, g as u64);
                let mut grng = rand::rngs::StdRng::seed_from_u64(gene_seed);
                let log_base = normal_or_zero(&mut grng, self.gene_mean_sd);
                // residual, irreducible errors
                // gene-specific confounding effects
                let conf_k = Mat::rnorm_seeded(1, self.n_covar, gene_seed.wrapping_add(1));
                let mut covar_n = conf_k * &confounder_nk.transpose();
                let mu_covar = covar_n.mean();
                let sig_covar = covar_n.variance().sqrt().max(MIN_VARIANCE);
                covar_n
                    .iter_mut()
                    .for_each(|x| *x = (*x - mu_covar) / sig_covar);

                // Fixed part: exposure effect (causal genes) and confounding.
                let confounding_part = &covar_n * self.pve_covar.sqrt();
                let exposure_part = match causal_genes.get(&g) {
                    Some((_cat, assign)) => assign * self.pve_gene.max(0.).sqrt(),
                    None => Mat::zeros(1, self.n_indv),
                };
                let (fixed_n, pve_fixed) = if let Some((_cat, assign)) = causal_genes.get(&g) {
                    (
                        assign * self.pve_gene.max(0.).sqrt() + covar_n * self.pve_covar.sqrt(),
                        self.pve_gene + self.pve_covar,
                    )
                } else {
                    (covar_n * self.pve_covar.sqrt(), self.pve_covar)
                };

                match self.indv_dispersion {
                    Some((a, b, s0)) => {
                        // Baseline mean rate per cell before the random effect
                        // (delta has mean 1) and before depth.
                        let m_g = fixed_n.iter().map(|u| (log_base + u).exp()).sum::<f32>()
                            / self.n_indv as f32;
                        let phi = (a + b * m_g.ln() + normal_or_zero(&mut grng, s0))
                            .exp()
                            .clamp(1e-2, 1e4);
                        let delta = Gamma::new(phi, 1.0 / phi).expect("valid gamma");
                        let log_rate = fixed_n.map(|u| {
                            let d: f32 = delta.sample(&mut grng);
                            log_base + u + d.ln()
                        });
                        GeneDraw {
                            gene: g,
                            log_rate,
                            planted: Some((m_g, phi)),
                            exposure_part,
                            confounding_part,
                        }
                    }
                    None => {
                        // residual, irreducible errors
                        let eps_n = Mat::rnorm_seeded(1, self.n_indv, gene_seed.wrapping_add(2));
                        let log_rate = (fixed_n + eps_n * (1. - pve_fixed).max(0.).sqrt())
                            .add_scalar(log_base);
                        GeneDraw {
                            gene: g,
                            log_rate,
                            planted: None,
                            exposure_part,
                            confounding_part,
                        }
                    }
                }
            })
            .collect();

        data.sort_by_key(|d| d.gene);
        let dispersion_m = self
            .indv_dispersion
            .map(|_| data.iter().map(|d| d.planted.expect("planted")).collect());
        let rows_of = |f: &dyn Fn(&GeneDraw) -> &Mat| -> Mat {
            Mat::from_rows(
                &data
                    .iter()
                    .map(|d| f(d).row(0).into_owned())
                    .collect::<Vec<_>>(),
            )
        };
        let exposure_part_mn = rows_of(&|d| &d.exposure_part);
        let confounding_part_mn = rows_of(&|d| &d.confounding_part);
        let data_mn = data
            .into_iter()
            .map(|d| d.log_rate.row(0).into_owned())
            .collect::<Vec<_>>();

        Ok(GlmOut {
            data_mn: Mat::from_rows(&data_mn),
            sample_to_exposure,
            confounder_nk,
            causal_m: causal_genes.into_iter().map(|(g, (c, _))| (g, c)).collect(),
            dispersion_m,
            exposure_part_mn,
            confounding_part_mn,
        })
    }

    ///
    /// Sample cell-level triplets from individual-level data
    /// `y(g,j) ~ Poisson( ρ(j) * μ(g,i) )` for all cell `j` in sample `i`
    ///
    fn generate_triplets(&self, ln_mu_gn: &Mat) -> anyhow::Result<TripletsOut> {
        let n_indv = ln_mu_gn.ncols();
        let n_genes = self.n_genes;
        assert_eq!(n_indv, self.n_indv);

        // Sample number of cells
        let rpois = Poisson::new(self.n_cells_per_indv as f32)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rseed);

        let num_cells = (0..n_indv)
            .map(|_i| (rpois.sample(&mut rng) as usize).max(1))
            .collect::<Vec<_>>();

        // for each individual, sample n_cells_per_indv cells
        // y(g,j) ~ Poisson(ρ(j) * μ(g,i=N(j)))
        #[allow(clippy::type_complexity)]
        let mut indv_ncells_triplets = num_cells
            .into_par_iter()
            .progress_count(n_indv as u64)
            .enumerate()
            .filter_map(
                |(indv, nn)| -> Option<(usize, usize, Vec<(u64, u64, f32)>)> {
                    // one stream per individual, so rayon's order does not matter
                    let indv_seed = legume_numeric::matrix::rand_util::mix_seed(
                        self.rseed,
                        CELL_SEED_TAG + indv as u64,
                    );
                    let mut rng = rand::rngs::StdRng::seed_from_u64(indv_seed);
                    let rho_m = Mat::rgamma_seeded(
                        1,
                        nn,
                        self.depth_gamma_hyperparam,
                        indv_seed.wrapping_add(1),
                    );
                    let mu_g = ln_mu_gn.column(indv).map(|x| x.exp()).clone();
                    let mut _triplets = Vec::with_capacity(nn * n_genes);

                    for (j, &rho_j) in rho_m.iter().enumerate() {
                        for (i, &mu_g) in mu_g.iter().enumerate() {
                            let lambda_ij = (mu_g * rho_j).max(MIN_LAMBDA);
                            if let Ok(rpois) = Poisson::new(lambda_ij) {
                                let y_ij = rpois.sample(&mut rng);
                                if y_ij > EXPRESSION_THRESHOLD {
                                    _triplets.push((i as u64, j as u64, y_ij));
                                }
                            }
                        }
                    }

                    let max_cell = _triplets.iter().map(|&(_, j, _)| j).max();
                    max_cell.map(|max_cell| (indv, max_cell as usize + 1, _triplets))
                },
            )
            .collect::<Vec<_>>();

        indv_ncells_triplets.par_sort_by_key(|&(indv, _, _)| indv);

        let mut cumsum = 0_u64;
        let mut indv_offset: HashMap<usize, u64> = Default::default();
        for &(indv, ncells, _) in &indv_ncells_triplets {
            indv_offset.insert(indv, cumsum);
            cumsum += ncells as u64;
        }

        let n_cells = cumsum as usize;
        info!("Total {} cells of {} genes", n_cells, n_genes);

        let samples: Vec<usize> = indv_ncells_triplets
            .iter()
            .flat_map(|&(indv, ncells, _)| std::iter::repeat_n(indv, ncells))
            .collect();

        // provide unified/cumulative indexes for the columns/cells across individuals
        let triplets: Vec<(u64, u64, f32)> = indv_ncells_triplets
            .into_iter()
            .par_bridge()
            .flat_map(|(indv, _, triplets)| {
                let base = *indv_offset.get(&indv).unwrap();
                triplets
                    .into_iter()
                    .map(|(i, j, y_ij)| (i, j + base, y_ij))
                    .collect::<Vec<_>>()
            })
            .collect();

        let mtx_shape = (n_genes, n_cells, triplets.len());

        Ok(TripletsOut {
            mtx_shape,
            triplets,
            samples,
        })
    }
}

#[derive(Parser, Debug, Clone)]
pub struct SimOneTypeArgs {
    #[arg(short = 'r', long, required = true, help = "Number of genes (G)")]
    n_genes: usize,

    #[arg(
        short = 'c',
        long,
        required = true,
        help = "Total number of cells (distributed across individuals via Poisson)"
    )]
    n_cells: usize,

    #[arg(
        short = 'a',
        long,
        required = true,
        help = "Number of causal genes with X → Y (each assigned a random exposure category)"
    )]
    n_causal_genes: usize,

    #[arg(
        short = 'n',
        long,
        default_value_t = 2,
        help = "Number of exposure categories for X_i"
    )]
    n_exposure: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Individuals per exposure group (total N = n_exposure × n_samples_per_exposure)"
    )]
    n_samples_per_exposure: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Dimension of W_i, the individual-level confounder (W → X and W → Y)"
    )]
    n_covariates: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "PVE on W → X: strength with which W confounds exposure assignment"
    )]
    pve_covar_exposure: f32,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "PVE on X → Y:\n\
                causal effect of exposure on gene expression (causal genes only)"
    )]
    pve_exposure_gene: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "PVE on W → Y: confounder effect on gene expression"
    )]
    pve_covar_gene: f32,

    #[arg(
        long,
        default_value_t = DEFAULT_EFFECT_SIZE,
        help = "Standardized effect size for causal genes (+/− for matching/other category)"
    )]
    effect_size: f32,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1.0,1.0",
        value_name = "SHAPE,SCALE",
        help = "Gamma(shape, scale) hyperparameters for the per-cell depth factor ρ_j"
    )]
    gamma_hyperparam: Vec<f32>,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "SD",
        help = "SD of the per-gene log baseline rate.\n\
                0 keeps every gene at the same baseline."
    )]
    gene_mean_sd: f32,

    #[arg(
        long,
        value_delimiter = ',',
        value_name = "A,B",
        help = "Dispersion-vs-mean trend for the individual effect:\n\
                log phi_g = A + B * log m_g,\n\
                with delta_gi ~ Gamma(phi_g, phi_g) replacing the Gaussian residual.\n\
                Writes {out}.dispersion.tsv.gz with the planted m_g and phi_g."
    )]
    indv_dispersion_trend: Option<Vec<f32>>,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "S0",
        requires = "indv_dispersion_trend",
        help = "Scatter of log phi_g around the trend, N(0, S0^2)"
    )]
    indv_dispersion_sd: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    rseed: u64,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend for output counts"
    )]
    backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Keep a plain `.zarr` directory instead of `.zarr.zip` archive",
        long_help = "Disable `.zarr.zip` archiving, which is the default.\n\
                     A plain `.zarr` directory is written instead.\n\
                     This matches the `--no-zip` convention faba uses."
    )]
    zip: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Also save counts in MatrixMarket (.mtx.gz) format"
    )]
    save_mtx: bool,

    #[arg(
        short,
        long,
        required = true,
        value_name = "PREFIX",
        help = "Output file name prefix"
    )]
    output: Box<str>,
}

pub fn run_sim_one_type_data(args: SimOneTypeArgs) -> anyhow::Result<()> {
    if args.gamma_hyperparam.len() != 2 {
        return Err(anyhow::anyhow!(
            "need exactly two values for `gamma-hyperparam`"
        ));
    }

    if args.gamma_hyperparam.iter().any(|&x| x <= 0.0) {
        return Err(anyhow::anyhow!(
            "need positive values for `gamma-hyperparam`"
        ));
    }

    let indv_dispersion = match args.indv_dispersion_trend.as_deref() {
        None => None,
        Some([a, b]) => {
            anyhow::ensure!(
                args.indv_dispersion_sd >= 0.0,
                "`indv-dispersion-sd` must be non-negative"
            );
            Some((*a, *b, args.indv_dispersion_sd))
        }
        Some(_) => anyhow::bail!("need exactly two values for `indv-dispersion-trend`"),
    };
    anyhow::ensure!(
        args.gene_mean_sd >= 0.0,
        "`gene-mean-sd` must be non-negative"
    );

    mkdir_parent(&args.output)?;

    let depth_gamma_hyperparam = (args.gamma_hyperparam[0], args.gamma_hyperparam[1]);

    let n_indv = args.n_exposure * args.n_samples_per_exposure;

    let sim = GlmSimulator {
        n_indv,
        n_cells_per_indv: args.n_cells.div_ceil(n_indv),
        n_covar: args.n_covariates,
        n_exp_cat: args.n_exposure,
        n_genes: args.n_genes,
        n_causal_genes: args.n_causal_genes,
        pve_exposure: args.pve_covar_exposure,
        pve_gene: args.pve_exposure_gene,
        pve_covar: args.pve_covar_gene,
        effect_size: args.effect_size,
        rseed: args.rseed,
        depth_gamma_hyperparam,
        gene_mean_sd: args.gene_mean_sd,
        indv_dispersion,
    };

    info!("Simulating underlying individual-level data...");
    let glm = sim.generate_individual_glm()?;

    info!("Populating triplets...");
    let sim_out = sim.generate_triplets(&glm.data_mn)?;
    info!("Successfully simulated");

    let output = args.output.clone();

    // Honor `--no-zip` (default = zip) → `.zarr.zip` target, matching the
    // faba / data-beans-sim conventions. Then route through
    // `resolve_backend_file` so the no-hdf5 fallback in data-beans rewrites
    // an HDF5 request to Zarr instead of failing at the factory.
    let effective_output = data_beans::zarr_io::apply_zip_flag(&output, args.zip, &args.backend);
    let (backend, backend_file) =
        data_beans::hdf5_io::resolve_backend_file(&effective_output, Some(args.backend.clone()))?;
    let backend_file = backend_file.to_string();

    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";

    let data_file = mtx_file.replace(".mtx.gz", ".data.tsv.gz");
    let conf_file = mtx_file.replace(".mtx.gz", ".conf.tsv.gz");
    let sample_file = mtx_file.replace(".mtx.gz", ".samples.gz");
    let exposure_file = mtx_file.replace(".mtx.gz", ".exposures.gz");
    let causal_file = mtx_file.replace(".mtx.gz", ".causal.gz");

    write_types(&sim_out.samples, &sample_file)?;
    write_types(
        &glm.sample_to_exposure
            .into_iter()
            .map(|(s, e)| format!("{}\t{}", s, e))
            .collect(),
        &exposure_file,
    )?;
    write_types(
        &glm.causal_m
            .into_iter()
            .map(|(g, c)| format!("{}\t{}", g, c))
            .collect(),
        &causal_file,
    )?;
    glm.confounder_nk.to_tsv(&conf_file)?;
    glm.data_mn.to_tsv(&data_file)?;
    // planted components of the log rates, for scoring without estimation
    glm.exposure_part_mn
        .to_tsv(&mtx_file.replace(".mtx.gz", ".planted_exposure.tsv.gz"))?;
    glm.confounding_part_mn
        .to_tsv(&mtx_file.replace(".mtx.gz", ".planted_confounding.tsv.gz"))?;
    if let (Some(disp), Some((a, b, s0))) = (glm.dispersion_m.as_ref(), indv_dispersion) {
        let disp_file = mtx_file.replace(".mtx.gz", ".dispersion.tsv.gz");
        let mut lines = vec!["gene\tmean\tphi".to_string()];
        lines.extend(
            disp.iter()
                .enumerate()
                .map(|(g, (m, phi))| format!("{}\t{}\t{}", g, m, phi)),
        );
        write_types(&lines, &disp_file)?;
        info!(
            "Planted dispersion trend: log phi = {} + {} log mean, scatter sd {}",
            a, b, s0
        );
    }

    info!("registering triplets ...");
    let mtx_shape = sim_out.mtx_shape;
    // Actual cell count after Poisson dispersion of n_cells_per_indv may
    // differ from `args.n_cells`; column names must match the realised count.
    let actual_n_cells = mtx_shape.1;

    let rows: Vec<Box<str>> = (0..args.n_genes)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..actual_n_cells)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    if args.save_mtx {
        let mut triplets = sim_out.triplets.clone();
        triplets.sort_by_key(|&(row, _, _)| row);
        triplets.sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, args.n_genes, actual_n_cells, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    let mut data = create_sparse_from_triplets(
        &sim_out.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);
    // Drop the data handle so the .zarr directory is fully flushed before
    // we attempt to zip it.
    drop(data);

    // No-op when `effective_output` doesn't end in `.zarr.zip`
    // (i.e. `--no-zip` or HDF5 backend).
    data_beans::zarr_io::finalize_zarr_output(&backend_file, &effective_output)?;

    info!("done");
    Ok(())
}

pub(crate) fn sample_logits_each_row(
    logits_nk: Mat,
    rng: &mut rand::rngs::StdRng,
) -> anyhow::Result<Vec<usize>> {
    let weights_nk = logits_nk.row_iter().map(|logits| {
        let maxval = logits.max();
        let expvals = logits.add_scalar(-maxval).map(|x| x.exp());
        expvals.unscale(expvals.sum())
    });

    let weights_nk = Mat::from_rows(weights_nk.collect::<Vec<_>>().as_slice());
    sample_each_row(weights_nk, rng)
}

pub(crate) fn sample_each_row(
    weights_nk: Mat,
    rng: &mut rand::rngs::StdRng,
) -> anyhow::Result<Vec<usize>> {
    let weights_vec = weights_nk.row_iter().collect::<Vec<_>>();
    Ok(weights_vec
        .iter()
        .map(|weights| {
            let disc = WeightedIndex::new(weights).expect("discrete distribution");
            disc.sample(rng)
        })
        .collect())
}
