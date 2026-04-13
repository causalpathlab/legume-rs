//! CNV detection pipeline orchestrator.
//!
//! Composes genome ordering → coarsening → factorial tree into a single
//! `detect_cnv()` call that takes `log(mu_residual)` and gene positions.
//!
//! Also provides `evaluate_cell_cnv()` for projecting cell-level residuals
//! onto learned factor profiles via linear regression.

use std::io::BufRead;

use log::info;
use nalgebra::DMatrix;

use crate::coarsening_tree::CoarseningTree;
use crate::factorial_tree::{fit_tree_factorial, FactorialTreeConfig, FactorialTreeResult};
use crate::genome_order::{GenePosition, GenomeOrder};
use crate::genomic_coarsening::GenomicCoarsening;

// ---------------------------------------------------------------------------
// Config & Result
// ---------------------------------------------------------------------------

/// Configuration for the full CNV detection pipeline.
#[derive(Debug, Clone)]
pub struct CnvDetectConfig {
    /// Decreasing correlation thresholds for multi-level coarsening (e.g. `[0.7, 0.4]`).
    pub corr_thresholds: Vec<f32>,
    /// Factorial tree model configuration.
    pub factorial: FactorialTreeConfig,
}

impl Default for CnvDetectConfig {
    fn default() -> Self {
        Self {
            corr_thresholds: vec![0.7, 0.4],
            factorial: FactorialTreeConfig::default(),
        }
    }
}

/// Result of CNV detection.
#[derive(Debug, Clone)]
pub struct CnvDetectResult {
    /// Factorial tree result (loadings, factor states, emission means, etc.).
    pub factorial_result: FactorialTreeResult,
    /// Genomic coarsening (finest level).
    pub coarsening: GenomicCoarsening,
    /// Genome ordering used.
    pub genome_order: GenomeOrder,
    /// Per-gene effective factor profiles: `[G_ordered × F]`.
    /// `profiles[g, f] = μ_f[s_f(g)]` expanded from block-level to gene-level.
    pub gene_factor_profiles: DMatrix<f32>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Run the full CNV detection pipeline.
///
/// # Arguments
/// * `log_mu_residual` — `[G × S]` matrix of log(mu_residual) in original gene order
/// * `gene_positions` — per-gene genomic positions (from GFF)
/// * `config` — pipeline configuration
///
/// # Returns
/// `CnvDetectResult` with factor profiles, loadings, and coarsening info.
pub fn detect_cnv(
    log_mu_residual: &DMatrix<f32>,
    gene_positions: &[GenePosition],
    config: &CnvDetectConfig,
) -> anyhow::Result<CnvDetectResult> {
    let n_genes = log_mu_residual.nrows();
    let n_samples = log_mu_residual.ncols();

    info!("CNV detection: {} genes × {} samples", n_genes, n_samples);

    // 1. Build genome order
    let genome_order = GenomeOrder::from_positions(gene_positions);
    let n_ordered = genome_order.ordered_indices.len();
    info!(
        "Genome order: {} / {} genes mapped to canonical chromosomes",
        n_ordered, n_genes
    );

    if n_ordered == 0 {
        anyhow::bail!("No genes mapped to canonical chromosomes");
    }

    // 2. Reorder signal to genome order
    let ordered_signal = genome_order.reorder_rows(log_mu_residual)?;

    // 3. Build coarsening tree + run tree-based factorial inference
    let tree = CoarseningTree::build(
        &ordered_signal,
        &genome_order.chr_boundaries,
        &config.corr_thresholds,
    );

    let factorial_result = fit_tree_factorial(&tree, &config.factorial);

    // The finest-level coarsening for gene expansion
    let coarsening = tree.levels[0].clone();

    // 4. Expand finest-block-level factor states to gene-level profiles
    let n_factors = factorial_result.factor_viterbi_paths.len();
    let mut gene_factor_profiles = DMatrix::<f32>::zeros(n_ordered, n_factors);
    for f in 0..n_factors {
        let means = &factorial_result.factor_emission_means[f];
        let path = &factorial_result.factor_viterbi_paths[f];
        let block_means: Vec<f32> = path.iter().map(|&s| means[s]).collect();
        let gene_vals = coarsening.expand_vec_f32_to_genes(&block_means, n_ordered);
        for g in 0..n_ordered {
            gene_factor_profiles[(g, f)] = gene_vals[g];
        }
    }

    info!(
        "CNV detection complete: {} factors, {} finest blocks, {} levels",
        n_factors,
        coarsening.num_blocks(),
        tree.n_levels(),
    );

    Ok(CnvDetectResult {
        factorial_result,
        coarsening,
        genome_order,
        gene_factor_profiles,
    })
}

/// Project cell-level residuals onto learned factor profiles via OLS.
///
/// For each cell n, solves: `L_n = argmin Σ_g (r_n(g) - Σ_f L_{n,f} · v_f(g))²`
///
/// # Arguments
/// * `cell_residuals` — `[G_ordered × N]` per-cell residual signals (in genome order)
/// * `gene_factor_profiles` — `[G_ordered × F]` from `CnvDetectResult`
///
/// # Returns
/// `[N × F]` per-cell factor loadings.
pub fn evaluate_cell_cnv(
    cell_residuals: &DMatrix<f32>,
    gene_factor_profiles: &DMatrix<f32>,
) -> DMatrix<f32> {
    let n_genes = cell_residuals.nrows();
    let n_factors = gene_factor_profiles.ncols();

    assert_eq!(
        gene_factor_profiles.nrows(),
        n_genes,
        "gene count mismatch between residuals and profiles"
    );

    // V = gene_factor_profiles [G × F]
    // Solve V^T V L^T = V^T R for L [N × F]
    // (V^T V) is [F × F] — tiny, compute once
    let vtv = gene_factor_profiles.transpose() * gene_factor_profiles; // [F × F]
                                                                       // V^T R is [F × N]
    let vtr = gene_factor_profiles.transpose() * cell_residuals; // [F × N]

    // Solve via Cholesky (V^T V is positive semi-definite)
    // Add small ridge for numerical stability
    let ridge = 1e-6 * DMatrix::<f32>::identity(n_factors, n_factors);
    let vtv_reg = vtv + ridge;

    match vtv_reg.clone().cholesky() {
        Some(chol) => {
            // Solve for all cells at once: [F × N]
            let coeff = chol.solve(&vtr);
            coeff.transpose() // [N × F]
        }
        None => {
            info!("Cholesky failed, using pseudo-inverse for cell CNV evaluation");
            let vtv_inv = vtv_reg
                .try_inverse()
                .unwrap_or_else(|| DMatrix::identity(n_factors, n_factors));
            let coeff = vtv_inv * vtr;
            coeff.transpose()
        }
    }
}

// ---------------------------------------------------------------------------
// Gene position readers
// ---------------------------------------------------------------------------

/// Read gene positions from a CNV ground truth TSV file.
///
/// Format: `gene\tchromosome\tposition\tstate` (tab-separated, gzipped or plain).
/// Returns gene positions for all rows; the `state` column is ignored.
pub fn read_gene_positions_from_cnv_tsv(path: &str) -> anyhow::Result<Vec<GenePosition>> {
    let reader = matrix_util::common_io::open_buf_reader(path)?;

    let mut positions = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // skip header
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            continue;
        }
        let gene_idx: usize = cols[0].parse()?;
        let chromosome: Box<str> = cols[1].into();
        let position: u64 = cols[2].parse()?;
        positions.push(GenePosition {
            gene_idx,
            chromosome,
            position,
        });
    }

    info!("Read {} gene positions from {}", positions.len(), path);
    Ok(positions)
}
