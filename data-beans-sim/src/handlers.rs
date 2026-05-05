use crate::copula::marginals::{
    nb_cdf_table, nb_inverse_cdf_from_table, nb_table_cap, phi, sample_nb, NbFit,
};
use crate::copula::reference::open_reference;
use crate::copula::{fit_global_copula, GlobalCopulaArgs};
use crate::core as simulate;
use crate::multimodal as simulate_multimodal;
use data_beans::hdf5_io::*;
use data_beans::sparse_io::*;
use data_beans::zarr_io::{apply_zip_flag, finalize_zarr_output};

use clap::Args;
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::common_io::*;
use matrix_util::mtx_io;
use matrix_util::traits::*;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Args, Debug)]
pub struct RunSimulateArgs {
    #[arg(
        short,
        long,
        help = "Number of rows, genes, or features (ignored when --reference is set)"
    )]
    pub rows: Option<usize>,

    #[arg(short, long, help = "Number of columns or cells")]
    pub cols: usize,

    /// Real single-cell reference (`.h5`, `.zarr`, `.zarr.zip`). When set,
    /// the GLM pipeline is unchanged, but the final count generation step
    /// swaps `Poisson(λ)` for a copula-coupled NB draw using per-gene
    /// dispersion `r̂_g` and a global Σ̂ fitted from this reference.
    #[arg(long)]
    pub reference: Option<Box<str>>,

    /// HVG count for the gene-gene copula. Genes outside the HVG set are
    /// sampled independently from `NB(λ_{g,j}, r̂_g)`. Used only with `--reference`.
    #[arg(long, default_value_t = 2000)]
    pub n_hvg: usize,

    /// Maximum rank of the low-rank `Σ̂` factor `F = U·diag(σ)/√N`. Effective
    /// rank is `min(rank, n_hvg, n_reference_cells)`. Used only with `--reference`.
    #[arg(long, default_value_t = 100)]
    pub copula_rank: usize,

    /// Per-gene isotropic ridge variance added at sample time on top of `Σ̂`.
    /// Used only with `--reference`.
    #[arg(long, default_value_t = 1e-3)]
    pub regularization: f32,

    /// Lower bound on the NB size parameter `r̂_g`. Tames runaway dispersion
    /// when MoM yields a near-zero `r` for noisy genes. Used only with `--reference`.
    #[arg(long, default_value_t = 1e-2)]
    pub r_floor: f32,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Depth per column (expected non-zero genes per cell)"
    )]
    pub depth: usize,

    #[arg(
        short,
        long,
        default_value_t = 1,
        help = "Number of factors (cell types, topics, states, etc.)"
    )]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Proportion of variance explained by topic membership"
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Proportion of variance explained by batch effects"
    )]
    pub pve_batch: f32,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Overdispersion parameter for Gamma dictionary"
    )]
    pub overdisp: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(long, help = "Hierarchical tree depth for binary tree dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of chromosomes for CNV simulation (0 = disabled)"
    )]
    pub n_chromosomes: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Expected CNV events per chromosome"
    )]
    pub cnv_events_per_chr: f32,

    #[arg(
        long,
        default_value_t = 0.15,
        help = "CNV block size as fraction of genes per chromosome"
    )]
    pub cnv_block_frac: f32,

    #[arg(long, default_value_t = 2.0, help = "Fold-change for CNV gain events")]
    pub cnv_gain_fold: f32,

    #[arg(long, default_value_t = 0.5, help = "Fold-change for CNV loss events")]
    pub cnv_loss_fold: f32,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output"
    )]
    pub backend: SparseIoBackend,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,
}

#[derive(Args, Debug)]
pub struct RunSimulateMultimodalArgs {
    #[arg(short, long, help = "Number of features (shared across modalities)")]
    pub rows: usize,

    #[arg(short, long, help = "Number of cells")]
    pub cols: usize,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Depth per modality (comma-separated, e.g., 1000,500)"
    )]
    pub depth: Vec<usize>,

    #[arg(short, long, default_value_t = 5, help = "Number of topics")]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches")]
    pub batches: usize,

    #[arg(long, default_value_t = 1.0, help = "Scale of base dictionary logits")]
    pub base_scale: f32,

    #[arg(long, default_value_t = 1.0, help = "Scale of non-zero delta entries")]
    pub delta_scale: f32,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of non-zero delta genes per topic"
    )]
    pub n_delta_features: usize,

    #[arg(long, default_value_t = 1.0, help = "PVE by topic membership")]
    pub pve_topic: f32,

    #[arg(long, default_value_t = 1.0, help = "PVE by batch effects")]
    pub pve_batch: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Share batch effects across modalities"
    )]
    pub shared_batch_effects: bool,

    #[arg(long, help = "Hierarchical tree depth for base dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Overdispersion for hierarchical dictionary"
    )]
    pub overdisp: f32,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(long, value_enum, default_value = "zarr", help = "Backend format")]
    pub backend: SparseIoBackend,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,
}

/// Simulate factored count data.
///
/// Without `--reference`: factored Poisson-Gamma model
/// (`y_{g,j} ~ Poisson(λ_{g,j})`, `λ = depth · β·θ · δ`) with optional CNV
/// and housekeeping injection.
///
/// With `--reference`: identical GLM through `λ_{g,j}`, but the final count
/// step swaps `Poisson(λ)` for a copula-coupled NB draw using per-gene
/// dispersion `r̂_g` and a global Σ̂ fitted from the reference. CNV is
/// disabled in this mode (passing `--n-chromosomes > 0` is a hard error).
pub fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    if cmd_args.reference.is_some() && cmd_args.n_chromosomes > 0 {
        anyhow::bail!(
            "`--reference` and `--n-chromosomes > 0` are mutually exclusive. \
             CNV synthesis is not supported in copula+NB sampling mode."
        );
    }
    if cmd_args.reference.is_some() && cmd_args.rows.is_some() {
        anyhow::bail!(
            "`--rows` and `--reference` are mutually exclusive. \
             Under `--reference` the gene count is taken from the reference."
        );
    }
    if cmd_args.reference.is_some() {
        return run_simulate_with_reference(cmd_args);
    }

    let rows = cmd_args
        .rows
        .ok_or_else(|| anyhow::anyhow!("--rows is required when --reference is not set"))?;

    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip);
    let output: Box<str> = strip_backend_suffix(&effective_output).into();

    dirname(&output).as_deref().map(mkdir).transpose()?;

    let backend = cmd_args.backend.clone();
    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;

    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";

    let dict_file = mtx_file.replace(".mtx.gz", ".dict.parquet");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.parquet");
    let batch_memb_file = mtx_file.replace(".mtx.gz", ".batch.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.parquet");

    remove_all_files(&vec![
        backend_file.clone(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let sim_args = simulate::SimArgs {
        rows,
        cols: cmd_args.cols,
        depth: cmd_args.depth,
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        overdisp: cmd_args.overdisp,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        hierarchical_depth: cmd_args.hierarchical_depth,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
        n_chromosomes: cmd_args.n_chromosomes,
        cnv_events_per_chr: cmd_args.cnv_events_per_chr,
        cnv_block_frac: cmd_args.cnv_block_frac,
        cnv_gain_fold: cmd_args.cnv_gain_fold,
        cnv_loss_fold: cmd_args.cnv_loss_fold,
    };

    let sim = simulate::generate_factored_poisson_gamma_data(&sim_args)?;
    info!("successfully generated factored Poisson-Gamma data");

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, &batch_memb_file)?;
    info!("batch membership: {:?}", &batch_memb_file);

    let mtx_shape = (sim_args.rows, sim_args.cols, sim.triplets.len());

    let rows: Vec<Box<str>> = (0..sim_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    sim.ln_delta_db
        .to_parquet_with_names(&ln_batch_file, (Some(&rows), Some("feature")), None)?;
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;
    sim.beta_dk
        .to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

    if let Some(ref node_probs) = sim.hierarchy_node_probs {
        let hierarchy_file = mtx_file.replace(".mtx.gz", ".hierarchy.parquet");
        node_probs.to_parquet_with_names(&hierarchy_file, (Some(&rows), Some("feature")), None)?;
        info!("wrote hierarchy node probabilities: {:?}", &hierarchy_file);
    }

    if let (Some(ref chromosomes), Some(ref positions), Some(ref states)) =
        (&sim.gene_chromosomes, &sim.gene_positions, &sim.cnv_states)
    {
        let cnv_file = mtx_file.replace(".mtx.gz", ".cnv_ground_truth.tsv.gz");
        let state_labels = ["loss", "neutral", "gain"];
        let cnv_lines: Vec<Box<str>> = std::iter::once("gene\tchromosome\tposition\tstate".into())
            .chain(
                rows.iter()
                    .zip(chromosomes.iter())
                    .zip(positions.iter())
                    .zip(states.iter())
                    .map(|(((g, chr), pos), &st)| {
                        format!("{}\t{}\t{}\t{}", g, chr, pos, state_labels[st as usize]).into()
                    }),
            )
            .collect();
        write_lines(&cnv_lines, &cnv_file)?;
        info!("wrote CNV ground truth (union): {:?}", &cnv_file);

        // Per-batch ground truth (for CNV detection validation)
        if let (Some(ref states_db), Some(ref clone_parent)) =
            (&sim.cnv_states_per_batch, &sim.cnv_clone_parent)
        {
            let per_batch_file = mtx_file.replace(".mtx.gz", ".cnv_per_batch_ground_truth.tsv.gz");
            let mut lines: Vec<Box<str>> = vec!["gene\tchromosome\tposition\tbatch\tstate".into()];
            for (b, batch_states) in states_db.iter().enumerate() {
                for (g, &st) in batch_states.iter().enumerate() {
                    if st != 1 {
                        // Only write non-neutral entries (sparse)
                        lines.push(
                            format!(
                                "{}\t{}\t{}\t{}\t{}",
                                g, chromosomes[g], positions[g], b, state_labels[st as usize]
                            )
                            .into(),
                        );
                    }
                }
            }
            write_lines(&lines, &per_batch_file)?;
            info!("wrote per-batch CNV ground truth: {:?}", &per_batch_file);

            // Clone tree
            let tree_file = mtx_file.replace(".mtx.gz", ".cnv_clone_tree.tsv.gz");
            let tree_lines: Vec<Box<str>> = std::iter::once("clone\tparent".into())
                .chain(
                    clone_parent
                        .iter()
                        .enumerate()
                        .map(|(b, &p)| format!("{}\t{}", b, p).into()),
                )
                .collect();
            write_lines(&tree_lines, &tree_file)?;
            info!("wrote clone tree: {:?}", &tree_file);
        }

        // Write minimal GFF for gene coordinates (so --gff works with simulated data)
        let gff_file = mtx_file.replace(".mtx.gz", ".genes.gff.gz");
        let gff_lines: Vec<Box<str>> = std::iter::once("##gff-version 3".into())
            .chain(
                rows.iter()
                    .zip(chromosomes.iter())
                    .zip(positions.iter())
                    .map(|((gene_name, chr), &pos)| {
                        // GFF3: seqname source feature start end score strand frame attributes
                        let start = pos + 1; // GFF is 1-based
                        let end = start + 1000; // dummy gene length
                        format!(
                            "{}\tsimulation\tgene\t{}\t{}\t.\t+\t.\tgene_name={}",
                            chr, start, end, gene_name,
                        )
                        .into()
                    }),
            )
            .collect();
        write_lines(&gff_lines, &gff_file)?;
        info!("wrote gene annotations: {:?}", &gff_file);
    }

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        &ln_batch_file, &dict_file, &prop_file
    );

    if cmd_args.save_mtx {
        let mut triplets = sim.triplets.clone();
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, sim_args.rows, sim_args.cols, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    info!("registering triplets ...");

    let mut data = create_sparse_from_triplets(
        &sim.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    info!("created sparse matrix: {}", backend_file);

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}

/// Reference-conditioned variant of [`run_simulate`]. The GLM pipeline
/// (`β·θ·δ` + housekeeping) mirrors the synthetic path verbatim, but
/// the final count step samples from a copula-coupled NB instead of
/// independent Poisson draws. Called from `run_simulate` when
/// `--reference` is set.
fn run_simulate_with_reference(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    let reference_path = cmd_args
        .reference
        .as_ref()
        .expect("run_simulate_with_reference called without --reference");

    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip);
    let output: Box<str> = strip_backend_suffix(&effective_output).into();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    let backend = cmd_args.backend.clone();
    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;

    info!("opening reference: {}", reference_path);
    let sc = open_reference(reference_path)?;

    let global_args = GlobalCopulaArgs {
        sc: &sc,
        n_hvg: cmd_args.n_hvg,
        copula_rank: cmd_args.copula_rank,
        regularization: cmd_args.regularization,
        r_floor: cmd_args.r_floor,
    };
    let fit = fit_global_copula(&global_args)?;

    let dd = fit.n_genes;
    let nn = cmd_args.cols;
    let bb = cmd_args.batches.max(1);
    let kk = if let Some(depth) = cmd_args.hierarchical_depth {
        1usize << (depth - 1)
    } else {
        cmd_args.factors.max(1)
    };
    let pve_topic = cmd_args.pve_topic.clamp(0.0, 1.0);
    let pve_batch = cmd_args.pve_batch.clamp(0.0, 1.0);

    let mut rng = rand::rngs::StdRng::seed_from_u64(cmd_args.rseed);

    let runif = rand_distr::Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = {
        use rand_distr::Distribution;
        (0..nn).map(|_| runif.sample(&mut rng)).collect()
    };

    // ln δ_{g,b} = √pve_batch · z₁(g,b) + √(1-pve_batch) · z₀(g),
    // both column-standardized standard normals — matches `core::generate_factored_poisson_gamma_data`.
    let mut ln_delta_db = DMatrix::<f32>::rnorm(dd, bb);
    ln_delta_db.scale_columns_inplace();
    ln_delta_db *= pve_batch.sqrt();
    let mut ln_null_d = DMatrix::<f32>::rnorm(dd, 1);
    ln_null_d.scale_columns_inplace();
    ln_null_d *= (1.0 - pve_batch).sqrt();
    for col in 0..ln_delta_db.ncols() {
        let mut ln_delta_d = ln_delta_db.column_mut(col);
        ln_delta_d += &ln_null_d.column(0);
    }
    let delta_db = ln_delta_db.map(|x| x.exp());

    let (beta_dk, hierarchy_node_probs) = if let Some(tree_depth) = cmd_args.hierarchical_depth {
        let (beta, node_probs) =
            crate::core::generate_hierarchical_dictionary(dd, tree_depth, cmd_args.overdisp, &mut rng);
        info!(
            "generated hierarchical dictionary: depth={}, K={} leaves",
            tree_depth,
            beta.ncols()
        );
        (beta, Some(node_probs))
    } else {
        let (a, b) = (1.0 / cmd_args.overdisp, (kk as f32).sqrt() * cmd_args.overdisp);
        let mut beta_dk = DMatrix::<f32>::rgamma(dd, kk, (a, b));
        if kk > 1 && pve_topic < 1.0 {
            let beta_null =
                DMatrix::<f32>::rgamma(dd, 1, (a, b)).scale(((1.0 - pve_topic) / kk as f32).sqrt());
            for k in 0..kk {
                let x = beta_dk.column(k).scale(pve_topic.sqrt()) + &beta_null;
                beta_dk.column_mut(k).copy_from(&x);
            }
        }
        (beta_dk, None)
    };

    let beta_dk = if cmd_args.n_housekeeping > 0 && cmd_args.n_housekeeping < dd {
        let mean_val = beta_dk.mean();
        let hk_mean = mean_val * cmd_args.housekeeping_fold;
        let hk_shape = 2.0_f32;
        let hk_rate = hk_shape / hk_mean;
        let hk_dist = rand_distr::Gamma::new(hk_shape, 1.0 / hk_rate).expect("housekeeping gamma");
        use rand_distr::Distribution;
        let mut beta = beta_dk;
        for g in 0..cmd_args.n_housekeeping {
            let base = hk_dist.sample(&mut rng);
            for k in 0..kk {
                beta[(g, k)] = base;
            }
        }
        info!(
            "injected {} housekeeping genes (mean={:.4}, fold={:.1}× synthetic mean {:.4})",
            cmd_args.n_housekeeping, hk_mean, cmd_args.housekeeping_fold, mean_val
        );
        beta
    } else {
        beta_dk
    };

    let theta_kn = crate::core::sample_theta_kn(kk, nn, pve_topic, &mut rng)?;

    // Copula+NB sampling: λ_{g, j} = depth · scale_j · (β θ)_{g, j} · δ_{g, b(j)}.
    let depth = cmd_args.depth;

    let triplets: Vec<(u64, u64, f32)> = (0..nn)
        .into_par_iter()
        .progress_count(nn as u64)
        .map(|j| -> Vec<(u64, u64, f32)> {
            let mut local_rng = rand::rngs::StdRng::seed_from_u64(
                cmd_args
                    .rseed
                    .wrapping_add(0x5a5a_5a5a)
                    .wrapping_add(j as u64),
            );
            let theta_j = theta_kn.column(j);
            let b = batch_membership[j];
            let mut lambda = &beta_dk * theta_j;
            for g in 0..dd {
                lambda[g] *= delta_db[(g, b)];
            }
            let total = lambda.sum().max(1e-30);
            let scale = (depth as f32) / total;
            for g in 0..dd {
                lambda[g] = (lambda[g] * scale).max(1e-8);
            }

            // Copula z* lives only in HVG space (the only dimension Σ̂ models).
            let z_star = fit.copula.sample(&mut local_rng);
            let mut counts: Vec<(u64, u64, f32)> = Vec::with_capacity(dd / 8);

            for g in 0..dd {
                let mu_g = lambda[g];
                if !mu_g.is_finite() || mu_g <= 0.0 {
                    continue;
                }
                let nb = NbFit {
                    mu: mu_g,
                    r: fit.r_hat[g],
                };
                let x = if let Some(h) = fit.hvg_pos[g] {
                    let table = nb_cdf_table(nb, nb_table_cap(nb));
                    if table.is_empty() {
                        0
                    } else {
                        let u = phi(z_star[h as usize] as f64).clamp(1e-7, 1.0 - 1e-7);
                        nb_inverse_cdf_from_table(u, &table)
                    }
                } else {
                    sample_nb(nb, &mut local_rng)
                };
                if x > 0 {
                    counts.push((g as u64, j as u64, x as f32));
                }
            }
            counts
        })
        .flatten()
        .collect();

    info!(
        "sampled {} cells producing {} nonzero triplets via copula+NB",
        nn,
        triplets.len()
    );

    // 7. Output: same companion files as the synthetic path, plus
    //    reference-only `.r.parquet` and `.hvg.gz`.
    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";
    let dict_file = format!("{}.dict.parquet", output);
    let prop_file = format!("{}.prop.parquet", output);
    let batch_memb_file = format!("{}.batch.gz", output);
    let ln_batch_file = format!("{}.ln_batch.parquet", output);
    let r_file = format!("{}.r.parquet", output);
    let hvg_file = format!("{}.hvg.gz", output);

    remove_all_files(&vec![
        backend_file.clone(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
        r_file.clone().into_boxed_str(),
        hvg_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let row_names: Vec<Box<str>> = fit.gene_names.clone();
    let col_names: Vec<Box<str>> = (0..nn)
        .map(|j| {
            let argmax_k = theta_kn.column(j).imax();
            format!("synthetic_{}_{}@{}", j, argmax_k, batch_membership[j]).into_boxed_str()
        })
        .collect();

    let batch_lines: Vec<Box<str>> = batch_membership
        .iter()
        .map(|b: &usize| b.to_string().into_boxed_str())
        .collect();
    write_lines(&batch_lines, &batch_memb_file)?;
    info!("batch membership: {}", batch_memb_file);

    ln_delta_db.to_parquet_with_names(
        &ln_batch_file,
        (Some(&row_names), Some("feature")),
        None,
    )?;
    theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&col_names), Some("cell")),
        None,
    )?;
    beta_dk.to_parquet_with_names(&dict_file, (Some(&row_names), Some("feature")), None)?;

    if let Some(node_probs) = hierarchy_node_probs.as_ref() {
        let hierarchy_file = format!("{}.hierarchy.parquet", output);
        node_probs.to_parquet_with_names(
            &hierarchy_file,
            (Some(&row_names), Some("feature")),
            None,
        )?;
        info!("hierarchy node probabilities: {}", hierarchy_file);
    }

    // Poisson collapses (`r = ∞`) are encoded as a large finite sentinel
    // so parquet readers don't choke on infinity.
    let mut r_col = DMatrix::<f32>::zeros(dd, 1);
    for g in 0..dd {
        let r = fit.r_hat[g];
        r_col[(g, 0)] = if r.is_finite() { r } else { 1e9 };
    }
    let r_label = ["r_hat".to_string().into_boxed_str()];
    r_col.to_parquet_with_names(&r_file, (Some(&row_names), Some("feature")), Some(&r_label))?;
    info!("per-gene NB dispersion r̂: {}", r_file);

    let hvg_lines: Vec<Box<str>> = fit
        .hvg_indices
        .iter()
        .map(|&g| row_names[g].clone())
        .collect();
    write_lines(&hvg_lines, &hvg_file)?;
    info!("HVGs used by copula: {}", hvg_file);

    let mut triplets = triplets;
    if cmd_args.save_mtx {
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);
        mtx_io::write_mtx_triplets(&triplets, dd, nn, &mtx_file)?;
        write_lines(&row_names, &row_file)?;
        write_lines(&col_names, &col_file)?;
        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    let mtx_shape = (dd, nn, triplets.len());
    let mut data = create_sparse_from_triplets(
        &triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;
    data.register_row_names_vec(&row_names);
    data.register_column_names_vec(&col_names);
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("wrote sparse backend: {}", backend_file);

    info!("done");
    Ok(())
}

/// Run multimodal simulation with shared base + delta dictionaries.
pub fn run_simulate_multimodal(cmd_args: &RunSimulateMultimodalArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    let sim_args = simulate_multimodal::MultimodalSimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        depth_per_modality: cmd_args.depth.clone(),
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        base_scale: cmd_args.base_scale,
        delta_scale: cmd_args.delta_scale,
        n_delta_features: cmd_args.n_delta_features,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        shared_batch_effects: cmd_args.shared_batch_effects,
        hierarchical_depth: cmd_args.hierarchical_depth,
        overdisp: cmd_args.overdisp,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
    };

    let mm = sim_args.depth_per_modality.len();
    let sim = simulate_multimodal::generate_multimodal_data(&sim_args)?;
    info!("generated multimodal data: {} modalities", mm);

    let rows: Vec<Box<str>> = (0..cmd_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    // Shared outputs
    let prop_file = format!("{}.prop.parquet", output);
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;

    let batch_file = format!("{}.batch.gz", output);
    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();
    write_lines(&batch_out, &batch_file)?;

    // W_base
    let base_file = format!("{}.w_base.parquet", output);
    sim.w_base_kd
        .to_parquet_with_names(&base_file, (None, None), None)?;

    // Per-modality outputs
    let backend = cmd_args.backend.clone();
    for m in 0..mm {
        let suffix = format!(".m{}", m);
        let modality_output = apply_zip_flag(&format!("{}{}", output, suffix), cmd_args.zip);
        let (_, backend_file) = resolve_backend_file(&modality_output, Some(backend.clone()))?;

        let mtx_shape = (cmd_args.rows, cmd_args.cols, sim.triplets[m].len());

        // Dictionary
        let dict_file = format!("{}{}.dict.parquet", output, suffix);
        sim.beta_dk[m].to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

        // Batch effects
        let ln_batch_file = format!("{}{}.ln_batch.parquet", output, suffix);
        sim.ln_delta_db[m].to_parquet_with_names(
            &ln_batch_file,
            (Some(&rows), Some("feature")),
            None,
        )?;

        // Delta (non-reference only)
        if m > 0 {
            let delta_file = format!("{}.w_delta{}.parquet", output, suffix);
            sim.w_delta_kd[m - 1].to_parquet_with_names(&delta_file, (None, None), None)?;

            let mask_file = format!("{}.spike_mask{}.parquet", output, suffix);
            sim.spike_mask_kd[m - 1].to_parquet_with_names(&mask_file, (None, None), None)?;
        }

        // MTX
        if cmd_args.save_mtx {
            let mtx_file = format!("{}{}.mtx.gz", output, suffix);
            let mut triplets = sim.triplets[m].clone();
            triplets.par_sort_by_key(|&(row, col, _)| (col, row));
            mtx_io::write_mtx_triplets(&triplets, cmd_args.rows, cmd_args.cols, &mtx_file)?;
        }

        // Sparse backend
        let mut data = create_sparse_from_triplets(
            &sim.triplets[m],
            mtx_shape,
            Some(&backend_file),
            Some(&backend),
        )?;

        data.register_row_names_vec(&rows);
        data.register_column_names_vec(&cols);

        finalize_zarr_output(&backend_file, &modality_output)?;
        info!("modality {}: {}", m, backend_file);
    }

    info!("done");
    Ok(())
}

