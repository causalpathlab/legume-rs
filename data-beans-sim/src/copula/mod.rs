//! scDesign-style copula simulator: reference-conditioned NB marginals coupled
//! by a Gaussian copula, with batch / CNV / housekeeping effects layered as
//! multiplicative shifts on the per-gene marginal mean.
//!
//! References:
//! - Li & Li 2019. *A statistical simulator scDesign for rational scRNA-seq experimental design.*
//!   Bioinformatics 35(14):i41–i50.
//! - Sun, Song, Li & Li 2021. *scDesign2: a transparent simulator that generates high-fidelity
//!   single-cell gene expression count data with gene correlations captured.* Genome Biology 22:163.
//! - Song et al. 2024. *scDesign3 generates realistic in silico data for multimodal single-cell
//!   and spatial omics.* Nature Biotechnology 42(2):247–252.
//!
//! Pipeline (per cluster):
//!  1. Fit per-gene NB marginals `(μ̂_g, r̂_g)` by method-of-moments.
//!  2. PIT each cell-gene value through `F_NB` with continuity correction, then
//!     `Φ⁻¹` to get z-scores. Build `Z ∈ ℝ^{|HVG| × |cells|}`.
//!  3. Run RSVD on `Z` to get a low-rank factor `F = U·diag(σ)/√N`, the
//!     rank-`r` analog of the Gaussian-copula Cholesky. `Σ̂ = F·Fᵀ + λI`
//!     is implicit; the dense `G × G` matrix is never formed.
//!  4. To sample new cell `j` in cluster `c` with batch `b`, clone `q`:
//!     - `z* = F · η + √λ · ε`, `η ~ N(0, I_r)`, `ε ~ N(0, I_g)`
//!     - `μ*_{g,j} = μ̂_g · exp(ln_batch[g, b]) · cnv_mult[g, q] · hk_fold(g)`
//!     - For HVG genes: `x* = F⁻¹(Φ(z*_g); μ*_g, r̂_g)` via cached per-(HVG, batch) CDF tables.
//!     - For non-HVG: independent NB sample with shifted mean (Gamma-Poisson mixture).

pub mod gaussian;
pub mod marginals;
pub mod reference;

use crate::core::{sample_cnv_blocks, CnvSimOut, CnvSimParams};
use gaussian::CopulaCovariance;
use indicatif::ParallelProgressIterator;
use log::info;
use marginals::{nb_cdf_table, nb_inverse_cdf_from_table, nb_table_cap, phi, sample_nb, NbFit};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use reference::SparseRef;

// Domain-separated seed offsets so independent random streams never collide.
const SEED_OFFSET_BATCH_EFFECT: u64 = 0xefef_efef;
const SEED_OFFSET_CNV: u64 = 0xc0c0_c0c0;
const SEED_OFFSET_SAMPLE: u64 = 0x5a5a_5a5a;

/// Per-cluster fit produced by `fit_copula`.
pub struct ClusterFit {
    pub label: usize,
    pub label_name: Box<str>,
    /// Reference-cell indices that fell into this cluster.
    pub cell_indices: Vec<usize>,
    /// Indices into `gene_names` selected as HVGs for this cluster.
    pub hvg_indices: Vec<usize>,
    /// NB marginal fits for *all* genes (length = `n_genes`). Non-HVG genes
    /// reuse this fit for independent sampling.
    pub marginals: Vec<NbFit>,
    pub copula: CopulaCovariance,
}

/// Effects sampled once during fit and applied uniformly during sampling.
pub struct CopulaEffects {
    /// `(n_genes × n_batches)` log shifts. `exp(.)` gives `δ_batch`. The
    /// batch count is `ln_batch_delta.ncols()`.
    pub ln_batch_delta: DMatrix<f32>,
    /// `Some(...)` iff `--n-chromosomes > 0`. Clone index ≡ batch index.
    pub cnv: Option<CnvSimOut>,
    /// Indices of housekeeping genes (top-k by reference mean).
    pub hk_indices: Vec<usize>,
    pub hk_fold: f32,
}

impl CopulaEffects {
    pub fn n_batches(&self) -> usize {
        self.ln_batch_delta.ncols().max(1)
    }
}

pub struct CopulaModel {
    pub n_genes: usize,
    pub gene_names: Vec<Box<str>>,
    pub clusters: Vec<ClusterFit>,
    pub effects: CopulaEffects,
}

pub enum PartitionMode {
    /// One global cluster covering all cells.
    Single,
    /// SVD + leiden inferred from the reference.
    AutoCluster {
        n_clusters: usize,
        knn: usize,
        rsvd_rank: usize,
        hvg_for_clustering: usize,
    },
    /// Pre-supplied per-cell labels (length must match `column_names`).
    Labels {
        labels: Vec<usize>,
        label_names: Vec<Box<str>>,
    },
}

pub struct CopulaFitArgs<'a> {
    pub sc: &'a SparseRef,
    pub partition: PartitionMode,
    pub n_hvg: usize,
    /// Maximum rank of the low-rank `Σ̂` factor `F = U·diag(σ)/√N`. Effective
    /// rank is `min(rank, n_hvg, n_cells_in_cluster)`. Rank ~100 typically
    /// captures most of the dependence structure for single-cell data;
    /// crank higher for richer fidelity at quadratic cost.
    pub copula_rank: usize,
    /// Per-gene isotropic ridge variance added at sample time. Acts like the
    /// previous `(1-λ)Σ̂ + λI` regularization, but without densifying Σ̂.
    pub regularization: f32,
    pub r_floor: f32,
    pub n_batches: usize,
    pub batch_effect_sigma: f32,
    pub n_chromosomes: usize,
    pub cnv_events_per_chr: f32,
    pub cnv_block_frac: f32,
    pub cnv_gain_fold: f32,
    pub cnv_loss_fold: f32,
    pub n_housekeeping: usize,
    pub housekeeping_fold: f32,
    pub rseed: u64,
}

pub struct SampleOut {
    pub triplets: Vec<(u64, u64, f32)>,
    pub cluster_assignment: Vec<usize>,
    pub batch_assignment: Vec<usize>,
    pub n_cells_total: usize,
}

/// `(cluster_label_id, label_name, reference cell indices)` per partition.
type ClusterPartition = (usize, Box<str>, Vec<usize>);
/// Per-cluster sample output: `(cluster_idx, triplets, batch_assignment_per_local_cell)`.
type ClusterSample = (usize, Vec<(u64, u64, f32)>, Vec<usize>);

/// Fit a copula model: marginals + Σ̂ per cluster, plus the shared effects.
pub fn fit_copula(args: &CopulaFitArgs) -> anyhow::Result<CopulaModel> {
    let n_genes = args
        .sc
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("reference has no num_rows"))?;
    let n_cells_total = args
        .sc
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("reference has no num_columns"))?;
    let gene_names = args.sc.row_names()?;

    info!("reference: {} genes × {} cells", n_genes, n_cells_total);

    // Global per-gene stats are needed for AutoCluster (HVG selection for the
    // embedding) and for housekeeping (top-k by mean). Compute once, share both.
    let hk_enabled = args.n_housekeeping > 0 && args.housekeeping_fold != 1.0;
    let needs_global_stats = matches!(args.partition, PartitionMode::AutoCluster { .. }) || hk_enabled;
    let global_stats = if needs_global_stats {
        let global_cells: Vec<usize> = (0..n_cells_total).collect();
        Some(reference::per_gene_stats(args.sc, &global_cells, n_genes)?)
    } else {
        None
    };

    // 1. Resolve the cluster partition.
    let (partition, label_names): (Vec<ClusterPartition>, Vec<Box<str>>) =
        match &args.partition {
            PartitionMode::Single => {
                let cells: Vec<usize> = (0..n_cells_total).collect();
                let names = vec!["all".to_string().into_boxed_str()];
                (vec![(0, names[0].clone(), cells)], names)
            }
            PartitionMode::AutoCluster {
                n_clusters,
                knn,
                rsvd_rank,
                hvg_for_clustering,
            } => {
                info!(
                    "auto-clustering: hvg_global={}, rank={}, knn={}, target_clusters={}",
                    hvg_for_clustering, rsvd_rank, knn, n_clusters
                );
                let stats = global_stats
                    .as_ref()
                    .expect("global stats computed for AutoCluster");
                let hvg_global = reference::select_hvg(stats, *hvg_for_clustering);
                let emb = reference::cell_embedding(args.sc, &hvg_global, *rsvd_rank)?;
                let labels = reference::cluster_cells(&emb, *knn, *n_clusters, args.rseed)?;
                let n_clusters_actual = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
                info!("auto-clustering produced {} clusters", n_clusters_actual);
                let groups = reference::partition_by_label(&labels);
                let names: Vec<Box<str>> = (0..n_clusters_actual)
                    .map(|c| format!("cluster_{}", c).into_boxed_str())
                    .collect();
                let parts = groups
                    .into_iter()
                    .map(|(l, cells)| (l, names[l].clone(), cells))
                    .collect();
                (parts, names)
            }
            PartitionMode::Labels {
                labels,
                label_names,
            } => {
                if labels.len() != n_cells_total {
                    anyhow::bail!("label count {} != n_cells {}", labels.len(), n_cells_total);
                }
                let groups = reference::partition_by_label(labels);
                let parts: Vec<ClusterPartition> = groups
                    .into_iter()
                    .map(|(l, cells)| (l, label_names[l].clone(), cells))
                    .collect();
                (parts, label_names.clone())
            }
        };

    info!(
        "fitting {} cluster(s): {}",
        partition.len(),
        partition
            .iter()
            .map(|(_, n, c)| format!("{}={}", n, c.len()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // 2. Per-cluster fits in parallel.
    let clusters: Vec<ClusterFit> = partition
        .into_par_iter()
        .progress_count(label_names.len() as u64)
        .map(
            |(label, label_name, cell_indices)| -> anyhow::Result<ClusterFit> {
                let n_cells = cell_indices.len();
                if n_cells < 2 {
                    anyhow::bail!(
                        "cluster '{}' has only {} cells; need ≥2 to fit a copula",
                        label_name,
                        n_cells
                    );
                }
                let (stats, marginals) = reference::per_gene_stats_and_marginals(
                    args.sc,
                    &cell_indices,
                    n_genes,
                    args.r_floor,
                )?;
                let hvg_indices = reference::select_hvg(&stats, args.n_hvg);
                let z =
                    reference::build_z_matrix(args.sc, &cell_indices, &hvg_indices, &marginals)?;
                let copula = CopulaCovariance::fit(&z, args.copula_rank, args.regularization)?;
                info!(
                    "fit cluster '{}': {} cells, {} HVGs, copula rank {} (cap {}) + ridge sd {:.3}",
                    label_name,
                    n_cells,
                    hvg_indices.len(),
                    copula.rank(),
                    args.copula_rank,
                    copula.ridge_sd,
                );
                Ok(ClusterFit {
                    label,
                    label_name,
                    cell_indices,
                    hvg_indices,
                    marginals,
                    copula,
                })
            },
        )
        .collect::<anyhow::Result<Vec<_>>>()?;

    // 3. Effects: batch δ, optional CNV, housekeeping.
    let n_batches = args.n_batches.max(1);
    let ln_batch_delta = if n_batches > 1 && args.batch_effect_sigma > 0.0 {
        let mut rng_eff =
            rand::rngs::StdRng::seed_from_u64(args.rseed.wrapping_add(SEED_OFFSET_BATCH_EFFECT));
        let dist = Normal::new(0.0_f32, args.batch_effect_sigma).unwrap();
        let mut m = DMatrix::<f32>::zeros(n_genes, n_batches);
        // Batch 0 is the reference (δ = 1, ln δ = 0).
        for b in 1..n_batches {
            for g in 0..n_genes {
                m[(g, b)] = dist.sample(&mut rng_eff);
            }
        }
        m
    } else {
        DMatrix::<f32>::zeros(n_genes, n_batches)
    };

    let cnv = if args.n_chromosomes > 0 {
        let params = CnvSimParams {
            n_genes,
            n_batches,
            n_chr: args.n_chromosomes,
            events_per_chr: args.cnv_events_per_chr,
            block_frac: args.cnv_block_frac,
            gain_fold: args.cnv_gain_fold,
            loss_fold: args.cnv_loss_fold,
        };
        let mut rng_cnv =
            rand::rngs::StdRng::seed_from_u64(args.rseed.wrapping_add(SEED_OFFSET_CNV));
        Some(sample_cnv_blocks(&params, &mut rng_cnv))
    } else {
        None
    };

    let hk_indices = if hk_enabled {
        let stats = global_stats
            .as_ref()
            .expect("global stats computed when hk_enabled");
        let mut ranked: Vec<(usize, f64)> =
            stats.iter().enumerate().map(|(g, s)| (g, s.mu)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut idx: Vec<usize> = ranked
            .into_iter()
            .take(args.n_housekeeping.min(n_genes))
            .map(|(g, _)| g)
            .collect();
        idx.sort_unstable();
        idx
    } else {
        Vec::new()
    };

    Ok(CopulaModel {
        n_genes,
        gene_names,
        clusters,
        effects: CopulaEffects {
            ln_batch_delta,
            cnv,
            hk_indices,
            hk_fold: args.housekeeping_fold,
        },
    })
}

/// Sample synthetic cells from a fitted copula model.
///
/// Produces `n_per_cluster` cells in each cluster, in the same cluster order
/// as `model.clusters`. Batch assignment within a cluster is round-robin.
pub fn sample_copula(
    model: &CopulaModel,
    n_per_cluster: usize,
    rseed: u64,
) -> anyhow::Result<SampleOut> {
    let n_clusters = model.clusters.len();
    if n_clusters == 0 {
        anyhow::bail!("no clusters in fitted model");
    }
    let n_batches = model.effects.n_batches();
    let n_total = n_per_cluster * n_clusters;
    info!(
        "sampling {} cells per cluster across {} cluster(s) with {} batch(es)",
        n_per_cluster, n_clusters, n_batches
    );

    // Effect-application invariants hoisted out of any per-gene loop.
    let has_batch_effects = model.effects.ln_batch_delta.ncols() > 1;
    let cnv = model.effects.cnv.as_ref();
    let mut hk_mask = vec![false; model.n_genes];
    for &g in &model.effects.hk_indices {
        hk_mask[g] = true;
    }
    let hk_fold = model.effects.hk_fold;

    // Per-(gene, batch) shifted μ helper, applied identically for HVG (table
    // build) and non-HVG (independent sampling) paths.
    let shift_mu = |g: usize, batch: usize, base_mu: f32| -> f32 {
        let mut mu = base_mu;
        if has_batch_effects {
            mu *= model.effects.ln_batch_delta[(g, batch)].exp();
        }
        if let Some(c) = cnv {
            mu *= c.cnv_multiplier_db[(g, batch)];
        }
        if hk_mask[g] {
            mu *= hk_fold;
        }
        mu
    };

    let per_cluster: Vec<ClusterSample> = model
        .clusters
        .par_iter()
        .enumerate()
        .map(|(c_idx, cluster)| -> anyhow::Result<_> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                rseed
                    .wrapping_add(SEED_OFFSET_SAMPLE)
                    .wrapping_add(c_idx as u64),
            );

            // Pre-build CDF tables for HVG genes per batch — eliminates the
            // dominant cost of rebuilding the table inside the per-cell loop.
            let cdf_tables: Vec<Vec<Vec<f64>>> = (0..n_batches)
                .map(|b| {
                    cluster
                        .hvg_indices
                        .iter()
                        .map(|&g| {
                            let base = cluster.marginals[g];
                            let shifted = NbFit {
                                mu: shift_mu(g, b, base.mu),
                                r: base.r,
                            };
                            if shifted.mu <= 0.0 || !shifted.mu.is_finite() {
                                Vec::new()
                            } else {
                                nb_cdf_table(shifted, nb_table_cap(shifted))
                            }
                        })
                        .collect()
                })
                .collect();

            // Non-HVG gene indices for the independent-sampling pass.
            let mut is_hvg = vec![false; model.n_genes];
            for &g in &cluster.hvg_indices {
                is_hvg[g] = true;
            }
            let non_hvg: Vec<usize> = (0..model.n_genes)
                .filter(|&g| !is_hvg[g] && cluster.marginals[g].mu > 0.0)
                .collect();

            let mut local_triplets: Vec<(u64, u64, f32)> =
                Vec::with_capacity(n_per_cluster * 4096);
            let mut batch_assn: Vec<usize> = Vec::with_capacity(n_per_cluster);

            for j_local in 0..n_per_cluster {
                let j_global = (c_idx * n_per_cluster + j_local) as u64;
                let batch = j_local % n_batches;
                let z_star = cluster.copula.sample(&mut rng);

                // Copula-coupled HVG genes: O(1) table lookup, no rebuild.
                for (h, &g) in cluster.hvg_indices.iter().enumerate() {
                    let table = &cdf_tables[batch][h];
                    if table.is_empty() {
                        continue;
                    }
                    let u = phi(z_star[h] as f64).clamp(1e-7, 1.0 - 1e-7);
                    let x = nb_inverse_cdf_from_table(u, table);
                    if x > 0 {
                        local_triplets.push((g as u64, j_global, x as f32));
                    }
                }

                // Non-HVG genes: independent NB draw with the shifted mean.
                for &g in &non_hvg {
                    let base = cluster.marginals[g];
                    let mu = shift_mu(g, batch, base.mu);
                    if !mu.is_finite() || mu <= 0.0 {
                        continue;
                    }
                    let x = sample_nb(NbFit { mu, r: base.r }, &mut rng);
                    if x > 0 {
                        local_triplets.push((g as u64, j_global, x as f32));
                    }
                }

                batch_assn.push(batch);
            }
            Ok((c_idx, local_triplets, batch_assn))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut triplets = Vec::new();
    let mut cluster_assignment = vec![0usize; n_total];
    let mut batch_assignment = vec![0usize; n_total];
    for (c_idx, mut tps, bas) in per_cluster {
        triplets.append(&mut tps);
        for (k, b) in bas.into_iter().enumerate() {
            let j = c_idx * n_per_cluster + k;
            cluster_assignment[j] = c_idx;
            batch_assignment[j] = b;
        }
    }

    Ok(SampleOut {
        triplets,
        cluster_assignment,
        batch_assignment,
        n_cells_total: n_total,
    })
}
