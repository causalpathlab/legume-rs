use crate::common::*;
use crate::run_sim_one_type::sample_logits_each_row;

use std::collections::HashMap;

use indicatif::ParallelProgressIterator;
use matrix_util::common_io::{mkdir, write_lines, write_types};
use matrix_util::mtx_io;
use matrix_util::traits::{IoOps, MatOps, SampleOps};
use rand::SeedableRng;
use rand_distr::{weighted::WeightedIndex, Distribution, Poisson, Uniform};
use rayon::prelude::*;

const MIN_VARIANCE: f32 = 1e-8;
const MIN_LAMBDA: f32 = 1e-8;
const EXPRESSION_THRESHOLD: f32 = 0.5;
const DEFAULT_EFFECT_SIZE: f32 = 10.0;
const DEFAULT_CELLTYPE_EFFECT: f32 = 5.0;

#[derive(Parser, Debug, Clone)]
pub struct SimColliderArgs {
    #[arg(short = 'r', required = true, help = "number of genes")]
    n_genes: usize,

    #[arg(short = 'c', required = true, help = "number of cells")]
    n_cells: usize,

    #[arg(
        short = 'a',
        required = true,
        help = "number of causal genes (exposure -> expression)"
    )]
    n_causal_genes: usize,

    #[arg(short = 't', required = true, help = "number of cell types")]
    n_cell_types: usize,

    #[arg(long, default_value_t = 20, help = "number of cell-type DE genes")]
    n_de_genes: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "individual-level confounder dimensions"
    )]
    n_covariates: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "cell-level confounder U_j dimensions"
    )]
    n_cell_covariates: usize,

    #[arg(long, default_value_t = 5, help = "individuals per exposure group")]
    n_samples_per_exposure: usize,

    #[arg(short, long, default_value_t = 2, help = "number of exposure groups")]
    n_exposure: usize,

    #[arg(long, default_value_t = 0.5, help = "V -> X confounding strength")]
    pve_covar_exposure: f32,

    #[arg(long, default_value_t = 0.3, help = "X -> A collider strength")]
    pve_exposure_celltype: f32,

    #[arg(long, default_value_t = 0.3, help = "U -> A strength")]
    pve_cell_covar_celltype: f32,

    #[arg(long, default_value_t = 0.3, help = "X -> Y for causal genes")]
    pve_exposure_gene: f32,

    #[arg(long, default_value_t = 0.3, help = "V -> Y individual confounding")]
    pve_covar_gene: f32,

    #[arg(long, default_value_t = 0.2, help = "U -> Y cell-level confounding")]
    pve_cell_covar_gene: f32,

    #[arg(long, default_value_t = DEFAULT_EFFECT_SIZE, help = "causal gene effect size")]
    effect_size: f32,

    #[arg(long, default_value_t = DEFAULT_CELLTYPE_EFFECT, help = "cell-type DE magnitude")]
    celltype_effect_size: f32,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1.0,1.0",
        help = "gamma distribution hyperparameters for cell depth"
    )]
    gamma_hyperparam: Vec<f32>,

    #[arg(long, default_value_t = 42, help = "random seed")]
    rseed: u64,

    #[arg(long, value_enum, default_value = "zarr", help = "backend")]
    backend: SparseIoBackend,

    #[arg(long, default_value_t = false, help = "save mtx")]
    save_mtx: bool,

    #[arg(long, short, required = true, help = "output header")]
    out: Box<str>,

    #[arg(long, short, help = "verbosity")]
    verbose: bool,
}

struct ColliderSimulator {
    n_indv: usize,
    n_cells_per_indv: usize,
    n_covar: usize,
    n_cell_covar: usize,
    n_exp_cat: usize,
    n_cell_types: usize,
    n_genes: usize,
    n_causal_genes: usize,
    n_de_genes: usize,
    pve_covar_exposure: f32,
    pve_exposure_celltype: f32,
    pve_cell_covar_celltype: f32,
    pve_exposure_gene: f32,
    pve_covar_gene: f32,
    pve_cell_covar_gene: f32,
    effect_size: f32,
    celltype_effect_size: f32,
    rseed: u64,
    depth_gamma_hyperparam: (f32, f32),
}

/// Individual-level results from Phase 1
struct IndividualOut {
    /// V_i: individual-level confounders (n_indv x n_covar)
    confounder_v: Mat,
    /// X_i: exposure assignment per individual
    exposure_assignment: Vec<usize>,
    /// P(X_i = x | V_i): propensity scores (n_indv x n_exp_cat)
    propensity_x: Mat,
}

/// Cell-level results from Phase 2+3
struct CellOut {
    mtx_shape: (usize, usize, usize),
    triplets: Vec<(u64, u64, f32)>,
    /// cell -> individual mapping
    samples: Vec<usize>,
    /// cell -> cell type assignment A_{ij}
    celltypes: Vec<usize>,
    /// U_j: cell-level confounders (n_cells x n_cell_covar)
    confounder_u: Mat,
    /// P(A_{ij} = k | X_i, U_j): propensity scores (n_cells x n_cell_types)
    propensity_a: Mat,
}

/// Gene metadata
struct GeneInfo {
    /// (gene_idx, exposure_category) for causal genes
    causal_genes: Vec<(usize, usize)>,
    /// (gene_idx, cell_type) for cell-type DE genes
    de_genes: Vec<(usize, usize)>,
    /// Causal gene effects: gene_idx -> standardized effect vector (1 x n_indv)
    causal_effects: HashMap<usize, (usize, Mat)>,
    /// Cell-type DE offsets: gene_idx -> (cell_type, offset)
    celltype_offsets: HashMap<usize, Vec<(usize, f32)>>,
}

impl ColliderSimulator {
    /// Phase 1: Generate individual-level confounders and exposure assignment
    ///
    /// V_i ~ N(0, I)
    /// X_i ~ Categorical(softmax(V_i * alpha + eps))
    fn generate_individual_level(&self) -> anyhow::Result<IndividualOut> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rseed);

        // V_i: individual-level confounders
        let confounder_v = Mat::rnorm(self.n_indv, self.n_covar);

        // Exposure assignment: logit(X_i) ~ V_i * alpha + eps
        let alpha_kc = Mat::rnorm(self.n_covar, self.n_exp_cat);

        let logits_nc = Mat::rnorm(self.n_indv, self.n_exp_cat) * (1. - self.pve_covar_exposure)
            + (&confounder_v * alpha_kc).scale_columns() * self.pve_covar_exposure;

        // Compute P(X|V) propensity scores before sampling
        let propensity_x = softmax_rows(&logits_nc);

        let exposure_assignment = sample_logits_each_row(logits_nc, &mut rng)?;

        Ok(IndividualOut {
            confounder_v,
            exposure_assignment,
            propensity_x,
        })
    }

    /// Set up gene-level effects (causal genes and cell-type DE genes)
    fn generate_gene_info(&self, exposure_assignment: &[usize]) -> anyhow::Result<GeneInfo> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rseed.wrapping_add(1));

        let runif_gene = Uniform::new(0, self.n_genes)?;
        let runif_cat = Uniform::new(0, self.n_exp_cat)?;
        let runif_ct = Uniform::new(0, self.n_cell_types)?;

        // Causal genes: exposure -> expression
        let effect_size = self.effect_size;
        let causal_effects: HashMap<usize, (usize, Mat)> = (0..self.n_causal_genes)
            .map(|_| {
                let gene = runif_gene.sample(&mut rng);
                let cat = runif_cat.sample(&mut rng);
                let ret_n = Mat::from_iterator(
                    1,
                    self.n_indv,
                    exposure_assignment.iter().map(|&c| {
                        if c == cat {
                            effect_size
                        } else {
                            -effect_size
                        }
                    }),
                );
                let mu = ret_n.mean();
                let sig = ret_n.variance().sqrt().max(MIN_VARIANCE);
                (gene, (cat, ret_n.map(|x| (x - mu) / sig)))
            })
            .collect();

        let causal_genes: Vec<(usize, usize)> =
            causal_effects.iter().map(|(&g, &(c, _))| (g, c)).collect();

        // Cell-type DE genes: different expression in specific cell types
        let ct_effect = self.celltype_effect_size;
        let mut celltype_offsets: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        let mut de_genes_list = Vec::new();

        for _ in 0..self.n_de_genes {
            let gene = runif_gene.sample(&mut rng);
            let ct = runif_ct.sample(&mut rng);
            celltype_offsets
                .entry(gene)
                .or_default()
                .push((ct, ct_effect));
            de_genes_list.push((gene, ct));
        }

        Ok(GeneInfo {
            causal_genes,
            de_genes: de_genes_list,
            causal_effects,
            celltype_offsets,
        })
    }

    /// Phase 2+3: For each individual, sample cells with collider bias
    ///
    /// For each cell j in individual i:
    ///   U_j ~ N(0, I)
    ///   A_{ij} ~ Cat(softmax(U_j * delta_k + X_i * eta_k + eps))  [collider]
    ///   log mu_{ij,g} = Δ_{g,A} + β_g * X_i + V_i * γ_g + U_j * ξ_g + ε
    ///   Y_{ij,g} ~ Poisson(ρ_j * exp(log mu))
    fn generate_cells_with_collider(
        &self,
        indv_out: &IndividualOut,
        gene_info: &GeneInfo,
    ) -> anyhow::Result<CellOut> {
        let n_indv = self.n_indv;
        let n_genes = self.n_genes;
        let n_ct = self.n_cell_types;
        let n_cell_covar = self.n_cell_covar;

        // Sample number of cells per individual
        let rpois = Poisson::new(self.n_cells_per_indv as f32)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rseed.wrapping_add(2));

        let num_cells: Vec<usize> = (0..n_indv)
            .map(|_| (rpois.sample(&mut rng) as usize).max(1))
            .collect();

        // Shared cell-type assignment coefficients (fixed across all cells)
        // delta: U -> A effect (n_cell_covar x n_cell_types)
        // eta: X -> A effect (n_exp_cat x n_cell_types)
        let delta = Mat::rnorm(n_cell_covar, n_ct);
        let eta = Mat::rnorm(self.n_exp_cat, n_ct);

        // Gene-specific confounder loadings (V -> Y and U -> Y)
        // gamma_g: V loading per gene (n_genes x n_covar)
        // xi_g: U loading per gene (n_genes x n_cell_covar)
        let gamma_gk = Mat::rnorm(n_genes, self.n_covar);
        let xi_gk = Mat::rnorm(n_genes, n_cell_covar);

        // Per-individual processing (parallelized)
        let pve_exp_ct = self.pve_exposure_celltype;
        let pve_cell_ct = self.pve_cell_covar_celltype;
        let pve_exp_gene = self.pve_exposure_gene;
        let pve_covar_gene = self.pve_covar_gene;
        let pve_cell_gene = self.pve_cell_covar_gene;
        let depth_gamma = self.depth_gamma_hyperparam;

        let results: Vec<_> = num_cells
            .into_par_iter()
            .progress_count(n_indv as u64)
            .enumerate()
            .map(|(indv, nn)| {
                let mut rng =
                    rand::rngs::StdRng::seed_from_u64(self.rseed.wrapping_add(100 + indv as u64));

                let x_i = indv_out.exposure_assignment[indv];
                let v_i = indv_out.confounder_v.row(indv); // 1 x n_covar

                // One-hot encode X_i (1 x n_exp_cat)
                let mut x_onehot = Mat::zeros(1, self.n_exp_cat);
                x_onehot[(0, x_i)] = 1.0;

                // X contribution to cell type logits (1 x n_ct)
                let x_logit = &x_onehot * &eta; // 1 x n_ct

                // V_i contribution to gene expression: V_i * gamma_g^T => (1 x n_genes)
                let v_effect_g = v_i * &gamma_gk.transpose(); // 1 x n_genes

                // Sample U_j for all cells of this individual
                let u_mat = Mat::rnorm(nn, n_cell_covar); // nn x n_cell_covar

                // Cell type logits: U_j * delta + X_i * eta + noise
                let u_logit = &u_mat * &delta; // nn x n_ct
                let noise_ct = Mat::rnorm(nn, n_ct);

                let pve_residual_ct = (1. - pve_exp_ct - pve_cell_ct).max(0.);
                let logits = u_logit.scale_columns() * pve_cell_ct.sqrt()
                    + Mat::from_fn(nn, n_ct, |_, k| x_logit[(0, k)]).scale_columns()
                        * pve_exp_ct.sqrt()
                    + noise_ct * pve_residual_ct.sqrt();

                // P(A|X,U) propensity scores
                let prop_a = softmax_rows(&logits);

                // Sample cell type assignments
                let celltypes: Vec<usize> = (0..nn)
                    .map(|j| {
                        let weights: Vec<f32> = (0..n_ct).map(|k| prop_a[(j, k)]).collect();
                        let dist = WeightedIndex::new(&weights).expect("cell type distribution");
                        dist.sample(&mut rng)
                    })
                    .collect();

                // U_j contribution to gene expression: U_j * xi_g^T => (nn x n_genes)
                let u_effect_g = &u_mat * &xi_gk.transpose(); // nn x n_genes

                // Cell depth factors
                let rho = Mat::rgamma(1, nn, depth_gamma);

                // Generate triplets: for each cell j, gene g
                let pve_residual_gene =
                    (1. - pve_exp_gene - pve_covar_gene - pve_cell_gene).max(0.);

                let mut triplets = Vec::with_capacity(nn * n_genes / 2);

                for j in 0..nn {
                    let rho_j = rho[(0, j)];
                    let ct_j = celltypes[j];

                    for g in 0..n_genes {
                        // Causal gene effect (X -> Y)
                        let causal_term =
                            if let Some((_, ref eff)) = gene_info.causal_effects.get(&g) {
                                eff[(0, indv)] * pve_exp_gene.sqrt()
                            } else {
                                0.0
                            };

                        // Cell-type DE offset
                        let ct_offset = gene_info
                            .celltype_offsets
                            .get(&g)
                            .map(|offsets| {
                                offsets
                                    .iter()
                                    .filter(|&&(ct, _)| ct == ct_j)
                                    .map(|&(_, eff)| eff)
                                    .sum::<f32>()
                            })
                            .unwrap_or(0.0);

                        // V_i confounding
                        let v_term = v_effect_g[(0, g)] * pve_covar_gene.sqrt();

                        // U_j confounding (creates collider bias)
                        let u_term = u_effect_g[(j, g)] * pve_cell_gene.sqrt();

                        // Noise
                        let eps: f32 = {
                            use rand_distr::StandardNormal;
                            let val: f32 = StandardNormal.sample(&mut rng);
                            val * pve_residual_gene.sqrt()
                        };

                        let log_mu = ct_offset + causal_term + v_term + u_term + eps;
                        let lambda = (rho_j * log_mu.exp()).max(MIN_LAMBDA);

                        if let Ok(rpois) = Poisson::new(lambda) {
                            let y: f32 = rpois.sample(&mut rng);
                            if y > EXPRESSION_THRESHOLD {
                                triplets.push((g as u64, j as u64, y));
                            }
                        }
                    }
                }

                (indv, nn, triplets, celltypes, u_mat, prop_a)
            })
            .collect();

        // Sort by individual and accumulate
        let mut sorted: Vec<_> = results;
        sorted.sort_by_key(|r| r.0);

        let mut cumsum = 0_u64;
        let mut indv_offset: HashMap<usize, u64> = HashMap::with_capacity(sorted.len());
        for &(indv, nn, _, _, _, _) in &sorted {
            indv_offset.insert(indv, cumsum);
            cumsum += nn as u64;
        }

        let n_total_cells = cumsum as usize;
        info!(
            "Total {} cells across {} individuals",
            n_total_cells, n_indv
        );

        // Collect cell metadata
        let mut samples = Vec::with_capacity(n_total_cells);
        let mut celltypes_all = Vec::with_capacity(n_total_cells);
        let mut u_rows = Vec::with_capacity(n_total_cells);
        let mut prop_a_rows = Vec::with_capacity(n_total_cells);

        for &(indv, nn, _, ref ct, ref u_mat, ref pa) in &sorted {
            for (j, &ct_j) in ct.iter().enumerate().take(nn) {
                samples.push(indv);
                celltypes_all.push(ct_j);
                u_rows.push(u_mat.row(j).clone_owned());
                prop_a_rows.push(pa.row(j).clone_owned());
            }
        }

        let confounder_u = Mat::from_rows(&u_rows);
        let propensity_a = Mat::from_rows(&prop_a_rows);

        // Offset triplet column indices
        let triplets: Vec<(u64, u64, f32)> = sorted
            .into_iter()
            .flat_map(|(indv, _, trips, _, _, _)| {
                let base = *indv_offset.get(&indv).unwrap();
                trips.into_iter().map(move |(g, j, y)| (g, j + base, y))
            })
            .collect();

        let mtx_shape = (n_genes, n_total_cells, triplets.len());

        Ok(CellOut {
            mtx_shape,
            triplets,
            samples,
            celltypes: celltypes_all,
            confounder_u,
            propensity_a,
        })
    }
}

/// Compute softmax across each row of a matrix
fn softmax_rows(logits: &Mat) -> Mat {
    let nrows = logits.nrows();
    let ncols = logits.ncols();
    let mut out = Mat::zeros(nrows, ncols);
    for i in 0..nrows {
        let row = logits.row(i);
        let max_val = row.max();
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        for (j, &e) in exp_vals.iter().enumerate() {
            out[(i, j)] = e / sum;
        }
    }
    out
}

pub fn run_sim_collider_data(args: SimColliderArgs) -> anyhow::Result<()> {
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

    let sim = ColliderSimulator {
        n_indv,
        n_cells_per_indv: args.n_cells.div_ceil(n_indv),
        n_covar: args.n_covariates,
        n_cell_covar: args.n_cell_covariates,
        n_exp_cat: args.n_exposure,
        n_cell_types: args.n_cell_types,
        n_genes: args.n_genes,
        n_causal_genes: args.n_causal_genes,
        n_de_genes: args.n_de_genes,
        pve_covar_exposure: args.pve_covar_exposure,
        pve_exposure_celltype: args.pve_exposure_celltype,
        pve_cell_covar_celltype: args.pve_cell_covar_celltype,
        pve_exposure_gene: args.pve_exposure_gene,
        pve_covar_gene: args.pve_covar_gene,
        pve_cell_covar_gene: args.pve_cell_covar_gene,
        effect_size: args.effect_size,
        celltype_effect_size: args.celltype_effect_size,
        rseed: args.rseed,
        depth_gamma_hyperparam,
    };

    info!("Phase 1: Generating individual-level data (V, X)...");
    let indv_out = sim.generate_individual_level()?;

    info!("Setting up gene effects...");
    let gene_info = sim.generate_gene_info(&indv_out.exposure_assignment)?;

    info!(
        "Phase 2+3: Generating cells with collider bias ({} types)...",
        args.n_cell_types
    );
    let cell_out = sim.generate_cells_with_collider(&indv_out, &gene_info)?;
    info!("Successfully simulated");

    // Write output files
    let output = args.out.clone();
    mkdir(&output)?;

    let backend = args.backend.clone();
    let backend_file = match backend {
        SparseIoBackend::HDF5 => output.to_string() + ".h5",
        SparseIoBackend::Zarr => output.to_string() + ".zarr",
    };

    // Cell -> individual mapping
    let sample_file = output.to_string() + ".samples.gz";
    write_types(&cell_out.samples, &sample_file)?;

    // Cell -> cell type mapping
    let celltype_file = output.to_string() + ".celltypes.gz";
    write_types(&cell_out.celltypes, &celltype_file)?;

    // Individual -> exposure mapping
    let exposure_file = output.to_string() + ".exposures.gz";
    write_types(
        &indv_out
            .exposure_assignment
            .iter()
            .enumerate()
            .map(|(s, e)| format!("{}\t{}", s, e))
            .collect(),
        &exposure_file,
    )?;

    // Causal gene labels
    let causal_file = output.to_string() + ".causal.gz";
    write_types(
        &gene_info
            .causal_genes
            .iter()
            .map(|(g, c)| format!("{}\t{}", g, c))
            .collect(),
        &causal_file,
    )?;

    // Cell-type DE gene labels
    let de_file = output.to_string() + ".de_genes.gz";
    write_types(
        &gene_info
            .de_genes
            .iter()
            .map(|(g, ct)| format!("{}\t{}", g, ct))
            .collect(),
        &de_file,
    )?;

    // V_i individual-level confounders
    let conf_v_file = output.to_string() + ".conf_V.tsv.gz";
    indv_out.confounder_v.to_tsv(&conf_v_file)?;

    // U_j cell-level confounders
    let conf_u_file = output.to_string() + ".conf_U.tsv.gz";
    cell_out.confounder_u.to_tsv(&conf_u_file)?;

    // P(X|V) propensity scores
    let prop_x_file = output.to_string() + ".propensity_X.tsv.gz";
    indv_out.propensity_x.to_tsv(&prop_x_file)?;

    // P(A|X,U) propensity scores
    let prop_a_file = output.to_string() + ".propensity_A.tsv.gz";
    cell_out.propensity_a.to_tsv(&prop_a_file)?;

    // Sparse count matrix
    info!("Registering triplets...");
    let mtx_shape = cell_out.mtx_shape;

    let rows: Vec<Box<str>> = (0..args.n_genes)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let n_total_cells = mtx_shape.1;
    let cols: Vec<Box<str>> = (0..n_total_cells)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    if args.save_mtx {
        let mtx_file = output.to_string() + ".mtx.gz";
        let row_file = output.to_string() + ".rows.gz";
        let col_file = output.to_string() + ".cols.gz";

        let mut triplets = cell_out.triplets.clone();
        triplets.sort_by_key(|&(row, _, _)| row);
        triplets.sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, args.n_genes, n_total_cells, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "Saved mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    let mut data = create_sparse_from_triplets(
        &cell_out.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    info!("Done. Output: {}", backend_file);
    Ok(())
}
