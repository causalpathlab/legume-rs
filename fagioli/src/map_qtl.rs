use anyhow::{bail, Result};
use clap::Args;
use log::info;
use nalgebra::DVector;
use rayon::prelude::*;

use data_beans::sparse_io::{open_sparse_matrix, SparseIoBackend};
use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::mapping::gene_mapping::{map_gene_qtl, GeneQtlResult, MappingConfig};
use fagioli::mapping::output::write_mapping_results;
use fagioli::mapping::pseudobulk::{
    aggregate_pseudobulk, individual_mask, read_cell_annotations,
};
use fagioli::simulation::load_gtf;

#[derive(Args, Debug, Clone)]
pub struct MapQtlArgs {
    /// PLINK BED file prefix (without .bed)
    #[arg(long)]
    pub bed_prefix: String,

    /// Single-cell count matrix (Zarr or HDF5 path)
    #[arg(long)]
    pub sc_counts: String,

    /// Storage backend: zarr or hdf5
    #[arg(long, default_value = "zarr")]
    pub backend: String,

    /// Cell annotations file (TSV or TSV.GZ): cell_id, individual_id, cell_type
    #[arg(long)]
    pub cell_annotations: String,

    /// GTF/GFF file for gene TSS positions
    #[arg(long)]
    pub gff_file: String,

    /// Chromosome to map
    #[arg(long)]
    pub chromosome: String,

    /// Left position bound (optional)
    #[arg(long)]
    pub left_bound: Option<u64>,

    /// Right position bound (optional)
    #[arg(long)]
    pub right_bound: Option<u64>,

    /// Cis-window size in base pairs
    #[arg(long, default_value = "1000000")]
    pub cis_window: u64,

    /// Model type: gaussian or susie
    #[arg(long, default_value = "susie")]
    pub model: String,

    /// Likelihood type: gaussian, poisson, or nb (negative binomial)
    #[arg(short, long, default_value = "gaussian")]
    pub likelihood: String,

    /// Number of Susie components (L)
    #[arg(long, default_value = "5")]
    pub num_components: usize,

    /// Number of Monte Carlo samples per ELBO estimate
    #[arg(long, default_value = "10")]
    pub num_samples: usize,

    /// Number of optimization iterations
    #[arg(long, default_value = "500")]
    pub num_iters: usize,

    /// AdamW learning rate
    #[arg(long, default_value = "0.02")]
    pub learning_rate: f64,

    /// Prior precision tau (0 = learnable)
    #[arg(long, default_value = "0.0")]
    pub prior_tau: f32,

    /// Minimum cells per individual per cell type to include
    #[arg(long, default_value = "5")]
    pub min_cells: usize,

    /// Minimum minor allele frequency (MAF) filter
    #[arg(long, default_value = "0.05")]
    pub min_maf: f32,

    /// PIP threshold for TSV output (Susie only; variants with max PIP below this are excluded)
    #[arg(long, default_value = "0.0")]
    pub pip_threshold: f32,

    /// Output prefix
    #[arg(short, long)]
    pub output: String,
}

pub fn map_qtl(args: &MapQtlArgs) -> Result<()> {
    // 1. Read genotypes
    info!("Reading genotypes from {}", args.bed_prefix);
    let mut reader = BedReader::new(&args.bed_prefix)?;
    let region = GenomicRegion::new(
        Some(args.chromosome.clone()),
        args.left_bound,
        args.right_bound,
    );
    let geno = reader.read(None, Some(region))?;
    info!(
        "Loaded genotypes: {} individuals × {} SNPs",
        geno.num_individuals(),
        geno.num_snps()
    );

    // 2. Load gene annotations
    info!("Loading gene annotations from {}", args.gff_file);
    let genes = load_gtf(
        &args.gff_file,
        Some(&args.chromosome),
        args.left_bound,
        args.right_bound,
        args.cis_window,
    )?;
    info!("Loaded {} genes", genes.genes.len());

    if genes.genes.is_empty() {
        bail!("No genes found in the specified region");
    }

    // 3. Open SC backend and read cell annotations
    info!("Opening SC backend: {}", args.sc_counts);
    let sc_backend_type = match args.backend.as_str() {
        "zarr" => SparseIoBackend::Zarr,
        "hdf5" | "h5" => SparseIoBackend::HDF5,
        other => bail!("Unknown backend: {}. Use 'zarr' or 'hdf5'", other),
    };
    let sc_backend = open_sparse_matrix(&args.sc_counts, &sc_backend_type)?;

    info!("Reading cell annotations from {}", args.cell_annotations);
    let annotations = read_cell_annotations(&args.cell_annotations)?;

    // 4. Pseudobulk aggregation
    info!("Aggregating pseudobulk expression...");
    let pseudobulk = aggregate_pseudobulk(sc_backend.as_ref(), &annotations, args.min_cells)?;

    // 5. Match individuals between genotype and pseudobulk
    let (_geno_to_pb, matched_geno_indices, matched_pb_indices) =
        match_individuals(&geno.individual_ids, &pseudobulk.individual_ids);

    info!(
        "Matched {} individuals between genotype and pseudobulk",
        matched_geno_indices.len()
    );

    if matched_geno_indices.is_empty() {
        bail!("No individuals matched between genotype and pseudobulk data");
    }

    // Subset genotype matrix to matched individuals and standardize columns
    let n_matched = matched_geno_indices.len();
    let mut matched_genotypes =
        nalgebra::DMatrix::<f32>::zeros(n_matched, geno.num_snps());
    for (new_idx, &orig_idx) in matched_geno_indices.iter().enumerate() {
        for snp in 0..geno.num_snps() {
            matched_genotypes[(new_idx, snp)] = geno.genotypes[(orig_idx, snp)];
        }
    }

    // MAF filter: remove low-MAF SNPs (computed on matched individuals)
    let maf_pass: Vec<bool> = (0..geno.num_snps())
        .map(|snp| {
            let col = matched_genotypes.column(snp);
            let mean = col.mean();
            let af = mean / 2.0; // allele frequency (dosage 0/1/2)
            let maf = af.min(1.0 - af);
            maf >= args.min_maf
        })
        .collect();

    let pass_indices: Vec<usize> = maf_pass.iter().enumerate()
        .filter(|(_, &pass)| pass)
        .map(|(i, _)| i)
        .collect();

    let n_filtered = geno.num_snps() - pass_indices.len();
    if n_filtered > 0 {
        info!(
            "MAF filter (>= {}): removed {} SNPs, {} remaining",
            args.min_maf, n_filtered, pass_indices.len()
        );
    }

    if pass_indices.is_empty() {
        bail!("No SNPs remaining after MAF filter");
    }

    // Build filtered genotype matrix and metadata
    let n_pass = pass_indices.len();
    let mut filtered_genotypes = nalgebra::DMatrix::<f32>::zeros(n_matched, n_pass);
    for (new_col, &orig_col) in pass_indices.iter().enumerate() {
        for row in 0..n_matched {
            filtered_genotypes[(row, new_col)] = matched_genotypes[(row, orig_col)];
        }
    }
    let matched_genotypes = filtered_genotypes;

    let geno = fagioli::genotype::GenotypeMatrix {
        genotypes: nalgebra::DMatrix::<f32>::zeros(0, 0), // not used after this
        individual_ids: Vec::new(),
        snp_ids: pass_indices.iter().map(|&i| geno.snp_ids[i].clone()).collect(),
        chromosomes: pass_indices.iter().map(|&i| geno.chromosomes[i].clone()).collect(),
        positions: pass_indices.iter().map(|&i| geno.positions[i]).collect(),
        allele1: pass_indices.iter().map(|&i| geno.allele1[i].clone()).collect(),
        allele2: pass_indices.iter().map(|&i| geno.allele2[i].clone()).collect(),
    };

    // Center genotype columns (subtract mean, keep original scale)
    use matrix_util::traits::MatOps;
    let mut matched_genotypes = matched_genotypes;
    matched_genotypes.centre_columns_inplace();

    // Build mapping config
    let config = MappingConfig {
        model: args.model.clone(),
        likelihood: args.likelihood.clone(),
        num_components: args.num_components,
        num_samples: args.num_samples,
        num_iters: args.num_iters,
        learning_rate: args.learning_rate,
        prior_tau: args.prior_tau,
    };

    // 6. Build gene name lookup: GFF gene_id -> GFF index
    use genomic_data::gff::GeneId;
    use std::collections::HashMap;
    let gff_gene_lookup: HashMap<&str, usize> = genes
        .genes
        .iter()
        .enumerate()
        .filter_map(|(i, g)| match &g.gene_id {
            GeneId::Ensembl(id) => Some((id.as_ref(), i)),
            _ => None,
        })
        .collect();

    // Map pseudobulk gene index -> GFF gene index (match by ENSG prefix)
    let pb_to_gff: Vec<Option<usize>> = pseudobulk
        .gene_names
        .iter()
        .map(|name| {
            // Try exact match first, then prefix before first '_'
            if let Some(&idx) = gff_gene_lookup.get(name.as_ref()) {
                return Some(idx);
            }
            if let Some(prefix) = name.split('_').next() {
                if let Some(&idx) = gff_gene_lookup.get(prefix) {
                    return Some(idx);
                }
            }
            None
        })
        .collect();

    let n_matched_genes = pb_to_gff.iter().filter(|x| x.is_some()).count();
    info!(
        "Matched {}/{} pseudobulk genes to GFF annotations",
        n_matched_genes,
        pseudobulk.gene_names.len()
    );

    // 7. For each cell type, run parallel gene mapping
    let mut all_results: Vec<GeneQtlResult> = Vec::new();

    for (ct_idx, ct_name) in pseudobulk.cell_type_names.iter().enumerate() {
        info!("Mapping cell type: {} ({}/{})", ct_name, ct_idx + 1, pseudobulk.cell_type_names.len());

        // Get pseudobulk expression for this cell type, subset to matched individuals
        let pb_expr = &pseudobulk.expression[ct_idx];
        let pb_counts = &pseudobulk.cell_counts[ct_idx];
        let num_genes = pseudobulk.gene_names.len();

        // Subset pseudobulk to matched individuals
        let mut matched_pb =
            nalgebra::DMatrix::<f32>::zeros(n_matched, num_genes);
        let mut matched_counts =
            nalgebra::DMatrix::<f32>::zeros(n_matched, num_genes);
        for (new_idx, &pb_idx) in matched_pb_indices.iter().enumerate() {
            for gene in 0..num_genes {
                matched_pb[(new_idx, gene)] = pb_expr[(pb_idx, gene)];
                matched_counts[(new_idx, gene)] = pb_counts[(pb_idx, gene)];
            }
        }

        // Build individual mask based on cell counts
        let mask = individual_mask(&matched_counts, args.min_cells as f32);

        // Parallel gene mapping
        let progress = indicatif::ProgressBar::new(num_genes as u64);
        progress.set_style(
            indicatif::ProgressStyle::default_bar()
                .template(&format!(
                    "[{{bar:40}}] {{pos}}/{{len}} genes ({})",
                    ct_name
                ))?
                .progress_chars("=> "),
        );

        let gene_results: Vec<GeneQtlResult> = (0..num_genes)
            .into_par_iter()
            .filter_map(|gene_idx| {
                // Look up the correct GFF gene for this pseudobulk gene
                let gff_idx = match pb_to_gff[gene_idx] {
                    Some(idx) => idx,
                    None => {
                        progress.inc(1);
                        return None;
                    }
                };

                let cis_snps = genes.cis_snp_indices(
                    gff_idx,
                    &geno.positions,
                    &geno.chromosomes,
                );

                if cis_snps.is_empty() {
                    progress.inc(1);
                    return None;
                }

                let y = DVector::from_fn(n_matched, |i, _| matched_pb[(i, gene_idx)]);
                let gene_id = &pseudobulk.gene_names[gene_idx];

                let result = map_gene_qtl(
                    gene_id,
                    ct_name,
                    &cis_snps,
                    &matched_genotypes,
                    &y,
                    &mask,
                    &config,
                );

                progress.inc(1);

                match result {
                    Ok(r) => Some(r),
                    Err(e) => {
                        log::warn!("Gene {} failed: {}", gene_id, e);
                        None
                    }
                }
            })
            .collect();

        progress.finish();
        info!(
            "Cell type {}: mapped {} genes with cis-SNPs",
            ct_name,
            gene_results.len()
        );
        all_results.extend(gene_results);
    }

    // 7. Write results
    info!("Writing results...");
    write_mapping_results(&args.output, &all_results, &geno, args.pip_threshold)?;

    info!(
        "QTL mapping complete: {} total gene-cell_type results",
        all_results.len()
    );

    Ok(())
}

/// Match individual IDs between genotype and pseudobulk datasets.
///
/// Returns (geno_to_pb_map, matched_geno_indices, matched_pb_indices)
/// where matched_geno_indices[i] and matched_pb_indices[i] refer to the
/// same individual in the genotype and pseudobulk datasets respectively.
fn match_individuals(
    geno_ids: &[Box<str>],
    pb_ids: &[Box<str>],
) -> (
    std::collections::HashMap<usize, usize>,
    Vec<usize>,
    Vec<usize>,
) {
    use std::collections::HashMap;

    let pb_lookup: HashMap<&str, usize> = pb_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_ref(), i))
        .collect();

    let mut geno_to_pb = HashMap::new();
    let mut matched_geno = Vec::new();
    let mut matched_pb = Vec::new();

    for (geno_idx, geno_id) in geno_ids.iter().enumerate() {
        if let Some(&pb_idx) = pb_lookup.get(geno_id.as_ref()) {
            geno_to_pb.insert(geno_idx, pb_idx);
            matched_geno.push(geno_idx);
            matched_pb.push(pb_idx);
        }
    }

    (geno_to_pb, matched_geno, matched_pb)
}
