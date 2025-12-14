// #![allow(dead_code)]
use crate::common::*;

use std::collections::HashMap;
use std::ops::Div;

use clap::Parser;
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::common_io::{mkdir, write_lines, write_types};
use matrix_util::mtx_io;
use matrix_util::traits::{IoOps, MatOps, SampleOps};
use rand::SeedableRng;
use rand_distr::{weighted::WeightedIndex, Distribution, Poisson, Uniform};

use rayon::prelude::*;

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
    rseed: u64,
    depth_gamma_hyperparam: (f32, f32),
}

struct GlmOut {
    data_mn: Mat,
    sample_to_exposure: Vec<(usize, usize)>,
    confounder_nk: Mat,
    causal_m: Vec<(usize, usize)>,
}

struct TripletsOut {
    mtx_shape: (usize, usize, usize),
    triplets: Vec<(u64, u64, f32)>,
    samples: Vec<usize>,
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

        // 1. Generate confounding factors
        let confounder_nk = Mat::rnorm(self.n_indv, self.n_covar);

        // 2. Generate multinomial exposure assignment (sample x gene)
        // x(i,c) ~ multinomial( sum w(i,k) * effect(k,c) + eps )
        let effect_kc = Mat::rnorm(self.n_covar, self.n_exp_cat);

        let logits_nc = Mat::rnorm(self.n_indv, self.n_exp_cat) * (1. - self.pve_exposure)
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
        let effect_size = 10.;

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
                let sig = ret_n.variance().sqrt().max(1e-8);

                (gene, (cat, ret_n.map(|x| (x - mu).div(sig))))
            })
            .collect();

        // 3.b. Generate individual-level data with confounding effects
        let mut data: Vec<(usize, Mat)> = (0..self.n_genes)
            .into_par_iter()
            .map(|g| {
                // residual, irreducible errors
                let eps_n = Mat::rnorm(1, self.n_indv);
                // gene-specific confounding effects
                let conf_k = Mat::rnorm(1, self.n_covar);
                let mut covar_n = conf_k * &confounder_nk.transpose();
                let mu_covar = covar_n.mean();
                let sig_covar = covar_n.variance().sqrt().max(1e-8);
                covar_n
                    .iter_mut()
                    .for_each(|x| *x = (*x - mu_covar) / sig_covar);

                if causal_genes.contains_key(&g) {
                    let (_cat, assign) = causal_genes.get(&g).expect("should have assignment");

                    let ret = assign * self.pve_gene.max(0.).sqrt()
                        + covar_n * self.pve_covar.sqrt()
                        + eps_n * (1. - self.pve_gene - self.pve_covar).max(0.).sqrt();
                    (g, ret)
                } else {
                    let ret = covar_n * self.pve_covar.sqrt()
                        + eps_n * (1. - self.pve_covar).max(0.).sqrt();
                    (g, ret)
                }
            })
            .collect();

        data.sort_by_key(|&(g, _)| g);
        let data_mn = data
            .into_iter()
            .map(|(_, x)| x.row(0).into_owned())
            .collect::<Vec<_>>();

        Ok(GlmOut {
            data_mn: Mat::from_rows(&data_mn),
            sample_to_exposure,
            confounder_nk,
            causal_m: causal_genes.into_iter().map(|(g, (c, _))| (g, c)).collect(),
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

        let threshold = 0.5;
        let eps = 1e-8;

        // Sample number of cells
        let rpois = Poisson::new(self.n_cells_per_indv as f32)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rseed);

        let num_cells = (0..n_indv)
            .into_iter()
            .map(|_i| (rpois.sample(&mut rng) as usize).max(1))
            .collect::<Vec<_>>();

        // for each individual, sample n_cells_per_indv cells
        // y(g,j) ~ Poisson(ρ(j) * μ(g,i=N(j)))
        let mut indv_ncells_triplets = num_cells
            .into_par_iter()
            .progress_count(n_indv as u64)
            .enumerate()
            .filter_map(
                |(indv, nn)| -> Option<(usize, usize, Vec<(u64, u64, f32)>)> {
                    let mut rng = rand::rng();
                    let rho_m = Mat::rgamma(1, nn, self.depth_gamma_hyperparam);
                    let mu_g = ln_mu_gn.column(indv).map(|x| x.exp()).clone();
                    let mut _triplets = Vec::with_capacity(nn * n_genes);

                    for (j, &rho_j) in rho_m.iter().enumerate() {
                        for (i, &mu_g) in mu_g.iter().enumerate() {
                            let lambda_ij = (mu_g * rho_j).max(eps);
                            if let Ok(rpois) = Poisson::new(lambda_ij) {
                                let y_ij = rpois.sample(&mut rng);
                                if y_ij > threshold {
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
        let mut indv_offset: HashMap<usize, u64> =
            HashMap::with_capacity(indv_ncells_triplets.len());
        for &(indv, ncells, _) in &indv_ncells_triplets {
            indv_offset.insert(indv, cumsum);
            cumsum += ncells as u64;
        }

        let n_cells = cumsum as usize;
        info!("Total {} cells of {} genes", n_cells, n_genes);

        let samples: Vec<usize> = indv_ncells_triplets
            .iter()
            .flat_map(|&(indv, ncells, _)| std::iter::repeat(indv).take(ncells))
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
pub struct SimArgs {
    #[arg(
        short = 'r',
        required = true,
        help = "number of genes",
        long_help = "Number of genes"
    )]
    n_genes: usize,

    #[arg(
        short = 'c',
        required = true,
        help = "number of cells",
        long_help = "Number of cells. It can have more or less number of cells."
    )]
    n_cells: usize,

    #[arg(
        short = 'a',
        required = true,
        help = "number of causal genes per cell type",
        long_help = "Number of causal genes per cell type"
    )]
    n_causal_genes: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "number of covariates",
        long_help = "Number of covariates"
    )]
    n_covariates: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "number of samples/individuals per exposure group",
        long_help = "Number of samples/individuals per exposure group"
    )]
    n_samples_per_exposure: usize,

    #[arg(
        short,
        long,
        default_value_t = 2,
        help = "number of exposure groups",
        long_help = "Number of exposure groups"
    )]
    n_exposure: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "proportion of variance explained by confounding to exposure",
        long_help = "Proportion of variance explained by confounding to exposure"
    )]
    pve_covar_exposure: f32,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "proportion of expression variance explained by causal exposure",
        long_help = "Proportion of expression variance explained by causal exposure"
    )]
    pve_exposure_gene: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "proportion of expression variance explained by covariates",
        long_help = "Proportion of expression variance explained by covariates"
    )]
    pve_covar_gene: f32,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1.0,1.0",
        help = "hyperparameter for gamma distribution",
        long_help = "Hyperparameter for gamma distribution"
    )]
    gamma_hyperparam: Vec<f32>,

    #[arg(
        long,
        default_value_t = 42,
        help = "random seed",
        long_help = "Random seed"
    )]
    rseed: u64,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "backend",
        long_help = "Backend"
    )]
    backend: SparseIoBackend,

    #[arg(
        long,
        default_value_t = false,
        help = "save mtx",
        long_help = "Save mtx"
    )]
    save_mtx: bool,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header"
    )]
    out: Box<str>,

    #[arg(long, short, help = "verbosity", long_help = "Verbosity")]
    verbose: bool,
}

pub fn run_sim_diff_data(args: SimArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

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
        rseed: args.rseed,
        depth_gamma_hyperparam,
    };

    info!("Simulating underlying individual-level data...");
    let glm = sim.generate_individual_glm()?;
    info!("Populating triplets...");
    let sim_out = sim.generate_triplets(&glm.data_mn)?;
    info!("Successfully simulated");

    let output = args.out.clone();
    mkdir(&output)?;

    let backend = args.backend.clone();
    let backend_file = match backend {
        SparseIoBackend::HDF5 => output.to_string() + ".h5",
        SparseIoBackend::Zarr => output.to_string() + ".zarr",
    };

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

    info!("registering triplets ...");
    let mtx_shape = sim_out.mtx_shape;

    let rows: Vec<Box<str>> = (0..args.n_genes)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..args.n_cells)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    if args.save_mtx {
        let mut triplets = sim_out.triplets.clone();
        triplets.sort_by_key(|&(row, _, _)| row);
        triplets.sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, args.n_genes, args.n_cells, &mtx_file)?;
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

    info!("done");
    Ok(())
}

fn sample_logits_each_row(
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

fn sample_each_row(weights_nk: Mat, rng: &mut rand::rngs::StdRng) -> anyhow::Result<Vec<usize>> {
    let weights_vec = weights_nk.row_iter().collect::<Vec<_>>();
    Ok(weights_vec
        .iter()
        .map(|weights| {
            let disc = WeightedIndex::new(weights).expect("discrete distribution");
            disc.sample(rng)
        })
        .collect())
}
