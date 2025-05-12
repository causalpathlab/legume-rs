// #![allow(dead_code)]
use crate::common::*;

pub use clap::Parser;
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::common_io::{mkdir, write_lines, write_types};
use matrix_util::mtx_io;
use matrix_util::traits::{IoOps, MatOps, SampleOps};
use rand::SeedableRng;
use rand_distr::{weighted::WeightedIndex, Distribution, Normal, Poisson, Uniform};

use rayon::prelude::*;

#[derive(Parser, Debug, Clone)]
pub struct SimArgs {
    /// number of rows
    #[arg(short = 'r', long)]
    n_rows: usize,

    /// number of columns
    #[arg(short = 'c', long)]
    n_cols: usize,

    /// number of causal rows per cell type
    #[arg(short = 'a', long)]
    n_causal_rows: usize,

    /// number of covariates
    #[arg(long, default_value_t = 1)]
    n_covariates: usize,

    /// number of factors
    #[arg(short = 'k', long, default_value_t = 1)]
    n_factors: usize,

    /// number of samples/individuals per exposure group
    #[arg(long, default_value_t = 5)]
    n_samples_per_exposure: usize,

    /// number of exposure groups
    #[arg(short, long, default_value_t = 2)]
    n_exposure: usize,

    /// depth per column
    #[arg(short, long, default_value_t = 10000)]
    depth: usize,

    /// overdispersion
    #[arg(long, default_value_t = 10.)]
    overdisp: f32,

    /// random seed
    #[arg(long, default_value_t = 42)]
    rseed: u64,

    /// backend
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// save mtx
    #[arg(long, default_value_t = false)]
    save_mtx: bool,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,
}

pub fn run_diff_data(args: SimArgs) -> anyhow::Result<()> {
    info!("Simulating triplets...");
    let sim_out = generate_diff_data(&args)?;
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

    let dict_file = mtx_file.replace(".mtx.gz", ".dict.gz");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.gz");
    let sample_file = mtx_file.replace(".mtx.gz", ".samples.gz");
    let exposure_file = mtx_file.replace(".mtx.gz", ".exposures.gz");
    let causal_file = mtx_file.replace(".mtx.gz", ".causal.gz");

    sim_out.dictionary_gene_factor.to_tsv(&dict_file)?;
    sim_out.proportions_cell_factor.to_tsv(&prop_file)?;

    write_types(
        &sim_out
            .sample_to_exposure
            .into_iter()
            .enumerate()
            .map(|(s, e)| format!("{}\t{}", s, e))
            .collect(),
        &exposure_file,
    )?;

    write_types(&sim_out.cell_to_sample, &sample_file)?;

    write_types(
        &sim_out
            .causal_genes
            .into_iter()
            .map(|x| format!("{}\t{}\t{}", x.exposure, x.gene, x.factor))
            .collect(),
        &causal_file,
    )?;

    info!("registering triplets ...");
    let mtx_shape = (args.n_rows, args.n_cols, sim_out.triplets.len());

    let rows: Vec<Box<str>> = (0..args.n_rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..args.n_cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    if args.save_mtx {
        let mut triplets = sim_out.triplets.clone();
        triplets.sort_by_key(|&(row, _, _)| row);
        triplets.sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, args.n_rows, args.n_cols, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    let mut data = create_sparse_from_triplets(
        sim_out.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    info!("done");
    Ok(())
}

pub struct CausalGene {
    pub gene: usize,
    pub factor: usize,
    pub exposure: usize,
}

pub struct SimOut {
    causal_genes: Vec<CausalGene>,
    sample_to_exposure: Vec<usize>,
    cell_to_sample: Vec<usize>,
    dictionary_gene_factor: Mat,
    proportions_cell_factor: Mat,
    triplets: Vec<(u64, u64, f32)>,
}

/// Generate
///
/// ```text
/// Y(g,j) ~ Poisson{ sum_t delta(g,t,S(j)) * beta(g,t) * theta(j,t) }
/// ```
///
/// ```text
/// ln delta(g,t,s) ~ sum_e tau(g,t,e) * A(e,s) + sum_c X(s,c) * kappa(c) + eps
/// ```
///
/// ```text
/// logit A(s) ~ sum_c X(s,c) * omega(c) + eps
/// ```
///
pub fn generate_diff_data(args: &SimArgs) -> anyhow::Result<SimOut> {
    let n_cells = args.n_cols;
    let n_genes = args.n_rows;
    let n_causal_genes = args.n_causal_rows;
    let n_factors = args.n_factors;
    let n_covariates = args.n_covariates;
    let n_samples_per_exposure = args.n_samples_per_exposure;
    let n_exposure = args.n_exposure;
    let nnz_per_cell = args.depth;
    let overdisp = args.overdisp;
    let rseed = args.rseed;

    let threshold = 0.5_f32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(rseed);

    let n_samples = n_samples_per_exposure * n_exposure;

    //////////////////////////////////////////////
    // factor/cell-type-specific causal effects //
    //////////////////////////////////////////////

    let mut tau_gf_vec = vec![Mat::zeros(n_genes, n_factors); n_exposure];

    let mut causal_genes = vec![];

    for f in 0..n_factors {
        let runif = Uniform::new(0, n_genes)?;
        let rnorm = Normal::new(0_f32, 1_f32)?;
        for e in 0..n_exposure {
            let genes: Vec<usize> = (0..n_causal_genes)
                .map(|_| runif.sample(&mut rng))
                .collect();
            for g in genes {
                tau_gf_vec[e][(g, f)] = rnorm.sample(&mut rng);
                causal_genes.push(CausalGene {
                    gene: g,
                    factor: f,
                    exposure: e,
                });
            }
        }
    }

    /////////////////////////////////////////////////
    // sample-level causal and confounding effects //
    /////////////////////////////////////////////////

    let covar_sv = Mat::rnorm(n_samples, n_covariates);

    let covar_assign_ve = Mat::rnorm(n_covariates, n_exposure);
    let covar_sample_gv = Mat::rnorm(n_genes, n_covariates);

    let logits_se = (&covar_sv * &covar_assign_ve).scale_columns()
        + Mat::rnorm(n_samples, n_exposure).scale_columns();

    let sample_to_exposure = sample_logits_each_row(logits_se, &mut rng)?;

    let mut delta_gf_vec = vec![];

    for s in 0..n_samples {
        let ln_conf_g = &covar_sample_gv * &covar_sv.row(s).transpose();

        let mut ln_delta_gf = Mat::rnorm(n_genes, n_factors);
        ln_delta_gf.column_iter_mut().for_each(|mut x| {
            x += &ln_conf_g;
        });
        ln_delta_gf.scale_columns_inplace();

        let exposure = sample_to_exposure[s];
        ln_delta_gf += &tau_gf_vec[exposure];
        ln_delta_gf.scale_columns_inplace();

        delta_gf_vec.push(ln_delta_gf.map(|x| x.exp()));
    }

    ///////////////////////////////////
    // cell type factorization model //
    ///////////////////////////////////

    let (a, b) = (1. / overdisp, (n_genes as f32).sqrt() * overdisp);
    let dictionary_gene_factor = Mat::rgamma(n_genes, n_factors, (a, b));

    let (a, b) = (1. / overdisp, (n_cells as f32).sqrt() * overdisp);
    let theta_fn = DMatrix::<f32>::rgamma(n_factors, n_cells, (a, b));
    let proportions_cell_factor = theta_fn.transpose();

    // Sample membership matrix
    let runif_sample = Uniform::new(0, n_samples)?;
    let cell_to_sample: Vec<usize> = (0..n_cells)
        .map(|_| runif_sample.sample(&mut rng))
        .collect();

    ///////////////////////////////
    // putting them all together //
    ///////////////////////////////

    let triplets: Vec<_> = theta_fn
        .column_iter()
        .enumerate()
        .par_bridge()
        .progress_count(n_cells as u64)
        .map(|(j, theta_j)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(rseed + j as u64);
            let s = cell_to_sample[j]; // sample id
            let lambda_j = dictionary_gene_factor.component_mul(&delta_gf_vec[s]) * &theta_j;

            let tot = lambda_j.sum();
            let scale = (nnz_per_cell as f32) / tot;

            lambda_j
                .iter()
                .enumerate()
                .filter_map(|(i, &l_ij)| {
                    let rpois = Poisson::new(l_ij * scale).expect("poisson sample error");
                    let y_ij = rpois.sample(&mut rng);
                    if y_ij > threshold {
                        Some((i as u64, j as u64, y_ij))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();

    info!(
        "sampled Poisson data with {} non-zero elements",
        triplets.len()
    );

    Ok(SimOut {
        causal_genes,
        sample_to_exposure,
        cell_to_sample,
        dictionary_gene_factor,
        proportions_cell_factor,
        triplets,
    })
}

fn sample_logits_each_row(
    logits_nk: Mat,
    rng: &mut rand::rngs::StdRng,
) -> anyhow::Result<Vec<usize>> {
    let weights_vec = logits_nk
        .row_iter()
        .map(|logits| {
            let maxval = logits.max();
            let expvec = &logits.add_scalar(-maxval).map(|x| x.exp());
            expvec.data.as_vec().clone()
        })
        .collect::<Vec<_>>();

    Ok(weights_vec
        .iter()
        .map(|weights| {
            let disc = WeightedIndex::new(weights).expect("discrete distribution");
            disc.sample(rng)
        })
        .collect())
}
