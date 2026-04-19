use anyhow::Result;
use log::info;
use matrix_util::dmatrix_util;
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use super::gene_annotations::GeneAnnotations;

// ── Gene role classification ────────────────────────────────────────────────

/// Role of a gene in the mediation model.
///
/// - **Mediator**: SNP → M_g → Y (β_g ≠ 0). Confounders affect M independently of Y.
/// - **Collider**: SNP → M_g ← U → Y (β_g = 0). Confounders shared with Y (γ_m ≈ ρ·γ_y).
///   Conditioning on M_g opens the spurious backdoor path.
/// - **Null**: SNP → M_g (β_g = 0). No confounder effect on expression.
///   Clean negative control — purely genetic + noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneRole {
    Mediator,
    Collider,
    Null,
}

impl std::fmt::Display for GeneRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneRole::Mediator => write!(f, "mediator"),
            GeneRole::Collider => write!(f, "collider"),
            GeneRole::Null => write!(f, "null"),
        }
    }
}

// ── Per-gene effects ────────────────────────────────────────────────────────

/// Per-gene mediation effects for the three-layer causal model:
/// SNP → Gene Expression (M) → Phenotype (Y)
#[derive(Debug, Clone)]
pub struct MediationGeneEffects {
    /// Gene index in the GeneAnnotations
    pub gene_idx: usize,
    /// Global SNP indices of cis-eQTL SNPs for this gene
    pub eqtl_snp_indices: Vec<usize>,
    /// α: eQTL effect sizes (SNP → M), length = eqtl_snp_indices.len()
    pub alpha: Vec<f32>,
    /// β: gene-to-phenotype effect (M → Y). 0.0 if non-causal.
    pub beta: f32,
    /// Role in the mediation model
    pub role: GeneRole,
    /// Whether this gene is "observed" (has eQTL data output).
    /// For causal genes: observed vs missing (unobserved mediator).
    /// For collider/null genes: always true.
    pub is_observed: bool,
    /// Total number of cis-SNPs in the gene's window (cached to avoid repeated O(M) scans)
    pub num_cis_snps: usize,
}

impl MediationGeneEffects {
    pub fn is_mediator(&self) -> bool {
        self.role == GeneRole::Mediator
    }

    pub fn is_collider(&self) -> bool {
        self.role == GeneRole::Collider
    }

    pub fn is_null(&self) -> bool {
        self.role == GeneRole::Null
    }
}

// ── Effect sampling ─────────────────────────────────────────────────────────

/// Parameters for sampling mediation effects.
pub struct MediationEffectParams<'a> {
    pub genes: &'a GeneAnnotations,
    pub snp_positions: &'a [u64],
    pub snp_chromosomes: &'a [Box<str>],
    pub n_eqtl_per_gene: usize,
    pub num_causal: usize,
    pub num_observed_causal: usize,
    pub num_collider: usize,
    pub seed: u64,
}

/// Sample mediation effects for all genes.
///
/// Assignment order: causal genes first, then collider genes from the
/// remaining pool, everything else is null.
pub fn sample_mediation_effects(
    params: &MediationEffectParams,
) -> Result<Vec<MediationGeneEffects>> {
    let genes = params.genes;
    let snp_positions = params.snp_positions;
    let snp_chromosomes = params.snp_chromosomes;
    let n_eqtl_per_gene = params.n_eqtl_per_gene;
    let num_genes = genes.genes.len();
    let num_causal = params.num_causal.min(num_genes);
    let num_observed_causal = params.num_observed_causal.min(num_causal);
    let num_collider = params
        .num_collider
        .min(num_genes.saturating_sub(num_causal));

    info!(
        "Sampling mediation effects: {} genes, {} eQTLs/gene, {} causal ({} observed), {} collider",
        num_genes, n_eqtl_per_gene, num_causal, num_observed_causal, num_collider,
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    // Shuffle gene indices, assign roles in order
    let mut gene_indices: Vec<usize> = (0..num_genes).collect();
    gene_indices.shuffle(&mut rng);

    let causal_genes: Vec<usize> = gene_indices.iter().copied().take(num_causal).collect();
    let collider_genes: Vec<usize> = gene_indices
        .iter()
        .copied()
        .skip(num_causal)
        .take(num_collider)
        .collect();

    let causal_set: rustc_hash::FxHashSet<usize> = causal_genes.iter().copied().collect();
    let observed_set: rustc_hash::FxHashSet<usize> = causal_genes[..num_observed_causal]
        .iter()
        .copied()
        .collect();
    let collider_set: rustc_hash::FxHashSet<usize> = collider_genes.iter().copied().collect();

    let mut effects = Vec::with_capacity(num_genes);

    for g in 0..num_genes {
        let cis_snps = genes.cis_snp_indices(g, snp_positions, snp_chromosomes);
        let num_cis = cis_snps.len();

        let mut cis_copy = cis_snps;
        cis_copy.shuffle(&mut rng);
        let n_eqtl = n_eqtl_per_gene.min(cis_copy.len());
        let eqtl_snp_indices: Vec<usize> = cis_copy.into_iter().take(n_eqtl).collect();

        let alpha: Vec<f32> = (0..eqtl_snp_indices.len())
            .map(|_| normal.sample(&mut rng))
            .collect();

        let role = if causal_set.contains(&g) {
            GeneRole::Mediator
        } else if collider_set.contains(&g) {
            GeneRole::Collider
        } else {
            GeneRole::Null
        };

        let beta = if role == GeneRole::Mediator {
            normal.sample(&mut rng)
        } else {
            0.0
        };

        let is_observed = !causal_set.contains(&g) || observed_set.contains(&g);

        effects.push(MediationGeneEffects {
            gene_idx: g,
            eqtl_snp_indices,
            alpha,
            beta,
            role,
            is_observed,
            num_cis_snps: num_cis,
        });
    }

    let n_with_eqtl: usize = effects
        .iter()
        .filter(|e| !e.eqtl_snp_indices.is_empty())
        .count();
    let n_no_cis: usize = effects
        .iter()
        .filter(|e| e.eqtl_snp_indices.is_empty())
        .count();
    info!(
        "Sampled effects: {} with eQTLs, {} with no cis-SNPs",
        n_with_eqtl, n_no_cis,
    );

    Ok(effects)
}

// ── Confounder mixing vector ────────────────────────────────────────────────

/// Sample the confounder mixing vector γ_y for the phenotype Y.
///
/// This must be generated BEFORE gene expressions so that collider genes
/// can correlate their confounder mixing with γ_y.
pub fn sample_confounder_mixing_y(num_confounders: usize, seed: u64) -> Vec<f32> {
    if num_confounders == 0 {
        return Vec::new();
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, (1.0 / num_confounders as f32).sqrt()).unwrap();
    (0..num_confounders)
        .map(|_| normal.sample(&mut rng))
        .collect()
}

// ── Gene expression generation ──────────────────────────────────────────────

/// Parameters for generating gene expressions.
pub struct ExpressionParams<'a> {
    pub genotypes: &'a DMatrix<f32>,
    pub effects: &'a [MediationGeneEffects],
    pub confounders: &'a DMatrix<f32>,
    pub gamma_y: &'a [f32],
    pub h2_eqtl: f32,
    pub h2_conf_m: f32,
    pub collider_correlation: f32,
    pub seed: u64,
}

/// Generate gene expression phenotypes M_g for all genes (parallel over genes).
///
/// Per gene g:
///   M_g = √h²_eqtl · std(X_{S_g} α_g) + √h²_conf_m · std(C γ_m_g) + √noise · std(ε_g)
///
/// For **collider** genes, γ_m_g = ρ·γ_y + √(1-ρ²)·γ_random,
/// so Cor(C·γ_m_g, C·γ_y) ≈ ρ.
/// For other genes, γ_m_g ~ N(0, 1/L) independently.
pub fn generate_gene_expressions(params: &ExpressionParams) -> Result<Vec<DVector<f32>>> {
    let genotypes = params.genotypes;
    let effects = params.effects;
    let confounders = params.confounders;
    let gamma_y = params.gamma_y;
    let h2_eqtl = params.h2_eqtl;
    let h2_conf_m = params.h2_conf_m;
    let seed = params.seed;
    let n = genotypes.nrows();

    if h2_eqtl + h2_conf_m > 1.0 + 1e-6 {
        anyhow::bail!(
            "h2_eqtl ({}) + h2_conf_m ({}) cannot exceed 1.0",
            h2_eqtl,
            h2_conf_m,
        );
    }
    let pve_noise = (1.0 - h2_eqtl - h2_conf_m).max(0.0);
    let rho = params.collider_correlation.clamp(0.0, 1.0);

    info!(
        "Generating gene expressions: {} genes, h2_eqtl={:.3}, h2_conf_m={:.3}, noise={:.3}, collider_rho={:.3}",
        effects.len(), h2_eqtl, h2_conf_m, pve_noise, rho,
    );

    let l = confounders.ncols();

    let expressions: Vec<DVector<f32>> = effects
        .par_iter()
        .map(|eff| {
            let g = eff.gene_idx;
            let gene_seed = seed.wrapping_add(g as u64);
            let mut rng = rand::rngs::StdRng::seed_from_u64(gene_seed);
            let normal = Normal::new(0.0f32, 1.0).unwrap();

            // Genetic component: X_{S_g} α_g
            let mut genetic = DVector::zeros(n);
            for (j, &snp_idx) in eff.eqtl_snp_indices.iter().enumerate() {
                let col = genotypes.column(snp_idx);
                for i in 0..n {
                    genetic[i] += col[i] * eff.alpha[j];
                }
            }

            let mut g_std = standardize_dvector(&genetic);
            g_std *= h2_eqtl.sqrt();

            // Confounder component: depends on gene role
            // - Mediator: independent γ_m (confounders affect M, but uncorrelated with Y)
            // - Collider: shared γ_m ≈ ρ·γ_y (opens spurious backdoor when conditioning on M)
            // - Null: no confounder effect (clean negative control)
            let has_confounder = !eff.is_null() && h2_conf_m > 0.0 && l > 0;

            let mut conf_component = DVector::zeros(n);
            if has_confounder {
                let scale = (1.0 / l as f32).sqrt();

                let gamma_m: Vec<f32> = if eff.is_collider() && !gamma_y.is_empty() {
                    // Collider: γ_m_g = ρ·γ_y + √(1-ρ²)·γ_random
                    let complement = (1.0 - rho * rho).max(0.0).sqrt();
                    gamma_y
                        .iter()
                        .map(|&gy| rho * gy + complement * normal.sample(&mut rng) * scale)
                        .collect()
                } else {
                    // Mediator: independent γ_m_g
                    (0..l).map(|_| normal.sample(&mut rng) * scale).collect()
                };

                for (j, &gm) in gamma_m.iter().enumerate() {
                    let col = confounders.column(j);
                    for i in 0..n {
                        conf_component[i] += col[i] * gm;
                    }
                }
                conf_component = standardize_dvector(&conf_component);
                conf_component *= h2_conf_m.sqrt();
            }

            // Noise component: null genes get all remaining variance (no confounder share)
            let this_pve_noise = if has_confounder {
                pve_noise
            } else {
                (1.0 - h2_eqtl).max(0.0)
            };
            let noise_raw: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            let mut noise = DVector::from_vec(noise_raw);
            noise = standardize_dvector(&noise);
            noise *= this_pve_noise.sqrt();

            g_std + conf_component + noise
        })
        .collect();

    info!("Generated {} gene expression vectors", expressions.len());
    Ok(expressions)
}

// ── Phenotype generation ────────────────────────────────────────────────────

/// Parameters for generating the mediated phenotype.
pub struct PhenotypeParams<'a> {
    pub expressions: &'a [DVector<f32>],
    pub effects: &'a [MediationGeneEffects],
    pub genotypes: &'a DMatrix<f32>,
    pub confounders: &'a DMatrix<f32>,
    pub gamma_y: &'a [f32],
    pub h2_mediated: f32,
    pub h2_direct: f32,
    pub h2_conf_y: f32,
    pub seed: u64,
}

/// Generate the complex trait Y from mediated effects.
///
/// Y = √h²_med · std(Σ_{causal g} β_g M_g) + √h²_direct · std(X·δ)
///   + √h²_conf_y · std(C·γ_y) + √noise · std(ε_y)
///
/// Takes `gamma_y` explicitly (the same vector used to correlate collider genes).
pub fn generate_mediated_phenotype(params: &PhenotypeParams) -> Result<DVector<f32>> {
    let expressions = params.expressions;
    let effects = params.effects;
    let genotypes = params.genotypes;
    let confounders = params.confounders;
    let gamma_y = params.gamma_y;
    let h2_mediated = params.h2_mediated;
    let h2_direct = params.h2_direct;
    let h2_conf_y = params.h2_conf_y;
    let seed = params.seed;
    let n = expressions.first().map_or(genotypes.nrows(), |e| e.len());

    let total_pve = h2_mediated + h2_direct + h2_conf_y;
    if total_pve > 1.0 + 1e-6 {
        anyhow::bail!(
            "h2_mediated ({}) + h2_direct ({}) + h2_conf_y ({}) cannot exceed 1.0",
            h2_mediated,
            h2_direct,
            h2_conf_y,
        );
    }
    let pve_noise = (1.0 - total_pve).max(0.0);

    info!(
        "Generating mediated phenotype: h2_med={:.3}, h2_direct={:.3}, h2_conf_y={:.3}, noise={:.3}",
        h2_mediated, h2_direct, h2_conf_y, pve_noise,
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    // Mediated component: Σ_{causal g} β_g M_g
    let mut mediated = DVector::zeros(n);
    let mut n_causal = 0usize;
    for (eff, expr) in effects.iter().zip(expressions.iter()) {
        if eff.is_mediator() {
            for i in 0..n {
                mediated[i] += eff.beta * expr[i];
            }
            n_causal += 1;
        }
    }
    info!("Mediated component sums over {} causal genes", n_causal);

    let mut med_std = standardize_dvector(&mediated);
    med_std *= h2_mediated.sqrt();

    // Direct genetic component (horizontal pleiotropy)
    let mut direct_std = DVector::zeros(n);
    if h2_direct > 0.0 {
        let m = genotypes.ncols();
        let poly_normal = Normal::new(0.0f32, (1.0 / m as f32).sqrt()).unwrap();
        let mut direct = DVector::zeros(n);
        for j in 0..m {
            let beta_j = poly_normal.sample(&mut rng);
            let col = genotypes.column(j);
            for i in 0..n {
                direct[i] += col[i] * beta_j;
            }
        }
        direct_std = standardize_dvector(&direct);
        direct_std *= h2_direct.sqrt();
    }

    // Confounder component: C · γ_y (using the SAME γ_y shared with collider genes)
    let l = confounders.ncols();
    let mut conf_component = DVector::zeros(n);
    if h2_conf_y > 0.0 && l > 0 && !gamma_y.is_empty() {
        for (j, &gy) in gamma_y.iter().enumerate() {
            let col = confounders.column(j);
            for i in 0..n {
                conf_component[i] += col[i] * gy;
            }
        }
        conf_component = standardize_dvector(&conf_component);
        conf_component *= h2_conf_y.sqrt();
    }

    // Noise
    let noise_raw: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();
    let mut noise = DVector::from_vec(noise_raw);
    noise = standardize_dvector(&noise);
    noise *= pve_noise.sqrt();

    let y = med_std + direct_std + conf_component + noise;

    info!("Generated phenotype vector: length {}", y.len());
    Ok(y)
}

// ── Winner's curse support ──────────────────────────────────────────────────

/// Split N individuals into discovery and replication subsets.
/// Returns (discovery_indices, replication_indices).
pub fn split_discovery_replication(
    n: usize,
    n_discovery: usize,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let n_disc = n_discovery.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let discovery = indices[..n_disc].to_vec();
    let replication = indices[n_disc..].to_vec();
    (discovery, replication)
}

/// Extract rows from a matrix for a subset of individuals.
pub fn subset_rows(mat: &DMatrix<f32>, indices: &[usize]) -> DMatrix<f32> {
    dmatrix_util::subset_rows(mat, indices.iter().copied())
        .expect("subset_rows: index out of bounds")
}

/// Extract elements from a DVector for a subset of individuals.
pub fn subset_dvector(v: &DVector<f32>, indices: &[usize]) -> DVector<f32> {
    DVector::from_vec(indices.iter().map(|&i| v[i]).collect())
}

// ── Utilities ───────────────────────────────────────────────────────────────

/// Standardize a DVector to mean 0, variance 1.
/// Returns zeros if variance is near-zero.
pub fn standardize_dvector(v: &DVector<f32>) -> DVector<f32> {
    let n = v.len() as f32;
    let mean = v.sum() / n;
    let centered: DVector<f32> = v.map(|x| x - mean);
    let var = centered.dot(&centered) / n;
    if var < 1e-10 {
        DVector::zeros(v.len())
    } else {
        centered / var.sqrt()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{
        generate_confounder_matrix, simulate_gene_annotations, ConfounderParams,
    };
    use matrix_util::traits::SampleOps;
    use nalgebra::DMatrix;

    fn make_test_data() -> (DMatrix<f32>, GeneAnnotations, Vec<u64>, Vec<Box<str>>) {
        let n = 500;
        let m = 200;
        let x = DMatrix::<f32>::rnorm(n, m);
        let positions: Vec<u64> = (0..m).map(|i| (i as u64) * 1000 + 1_000_000).collect();
        let chromosomes: Vec<Box<str>> = vec![Box::from("22"); m];
        let genes = simulate_gene_annotations(
            20,
            "22",
            1_000_000,
            1_000_000 + (m as u64) * 1000,
            50_000,
            42,
        );
        (x, genes, positions, chromosomes)
    }

    fn make_confounders(n: usize) -> DMatrix<f32> {
        generate_confounder_matrix(
            n,
            &ConfounderParams {
                num_confounders: 5,
                num_hidden_factors: 3,
                pve_confounders: 0.1,
            },
            100,
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_effects(
        genes: &GeneAnnotations,
        positions: &[u64],
        chromosomes: &[Box<str>],
        n_eqtl: usize,
        n_causal: usize,
        n_obs: usize,
        n_collider: usize,
        seed: u64,
    ) -> Vec<MediationGeneEffects> {
        sample_mediation_effects(&MediationEffectParams {
            genes,
            snp_positions: positions,
            snp_chromosomes: chromosomes,
            n_eqtl_per_gene: n_eqtl,
            num_causal: n_causal,
            num_observed_causal: n_obs,
            num_collider: n_collider,
            seed,
        })
        .unwrap()
    }

    #[test]
    fn test_sample_mediation_effects() {
        let (_, genes, positions, chromosomes) = make_test_data();
        let effects = sample_effects(&genes, &positions, &chromosomes, 3, 5, 3, 2, 42);

        assert_eq!(effects.len(), 20);

        let n_causal = effects.iter().filter(|e| e.is_mediator()).count();
        assert_eq!(n_causal, 5);

        let n_collider = effects.iter().filter(|e| e.is_collider()).count();
        assert_eq!(n_collider, 2);

        let n_null = effects.iter().filter(|e| e.role == GeneRole::Null).count();
        assert_eq!(n_null, 13);

        let n_observed_causal = effects
            .iter()
            .filter(|e| e.is_mediator() && e.is_observed)
            .count();
        assert_eq!(n_observed_causal, 3);

        // Non-causal genes have beta=0
        for eff in &effects {
            if !eff.is_mediator() {
                assert_eq!(eff.beta, 0.0);
            }
            assert!(eff.eqtl_snp_indices.len() <= 3);
            assert_eq!(eff.alpha.len(), eff.eqtl_snp_indices.len());
        }
    }

    #[test]
    fn test_gene_expression_variance() {
        let (x, genes, positions, chromosomes) = make_test_data();
        let effects = sample_effects(&genes, &positions, &chromosomes, 3, 5, 5, 2, 42);
        let confounders = make_confounders(x.nrows());
        let gamma_y = sample_confounder_mixing_y(confounders.ncols(), 150);

        let expressions = generate_gene_expressions(&ExpressionParams {
            genotypes: &x,
            effects: &effects,
            confounders: &confounders,
            gamma_y: &gamma_y,
            h2_eqtl: 0.4,
            h2_conf_m: 0.1,
            collider_correlation: 0.8,
            seed: 200,
        })
        .unwrap();

        assert_eq!(expressions.len(), 20);
        for expr in &expressions {
            assert_eq!(expr.len(), x.nrows());
        }
    }

    #[test]
    fn test_mediated_phenotype_variance() {
        let (x, genes, positions, chromosomes) = make_test_data();
        let effects = sample_effects(&genes, &positions, &chromosomes, 3, 5, 5, 0, 42);
        let confounders = make_confounders(x.nrows());
        let gamma_y = sample_confounder_mixing_y(confounders.ncols(), 150);

        let expressions = generate_gene_expressions(&ExpressionParams {
            genotypes: &x,
            effects: &effects,
            confounders: &confounders,
            gamma_y: &gamma_y,
            h2_eqtl: 0.3,
            h2_conf_m: 0.1,
            collider_correlation: 0.0,
            seed: 200,
        })
        .unwrap();

        let y = generate_mediated_phenotype(&PhenotypeParams {
            expressions: &expressions,
            effects: &effects,
            genotypes: &x,
            confounders: &confounders,
            gamma_y: &gamma_y,
            h2_mediated: 0.2,
            h2_direct: 0.0,
            h2_conf_y: 0.1,
            seed: 300,
        })
        .unwrap();

        assert_eq!(y.len(), x.nrows());

        let var = y.dot(&y) / y.len() as f32;
        assert!(
            (var - 1.0).abs() < 0.3,
            "Phenotype variance should be ~1.0, got {}",
            var,
        );
    }

    #[test]
    fn test_collider_gene_confounder_correlation() {
        // Collider genes should have M_g confounder component correlated with Y's
        let (x, genes, positions, chromosomes) = make_test_data();
        let n = x.nrows();

        // 5 causal, 5 collider, 10 null
        let effects = sample_effects(&genes, &positions, &chromosomes, 3, 5, 5, 5, 42);
        let confounders = make_confounders(n);
        let gamma_y = sample_confounder_mixing_y(confounders.ncols(), 150);

        let rho = 0.9;
        let expressions = generate_gene_expressions(&ExpressionParams {
            genotypes: &x,
            effects: &effects,
            confounders: &confounders,
            gamma_y: &gamma_y,
            h2_eqtl: 0.0,
            h2_conf_m: 1.0,
            collider_correlation: rho,
            seed: 200,
        })
        .unwrap();

        // With h2_eqtl=0 and h2_conf_m=1.0, expression IS the confounder component.
        // Compute C·γ_y for comparison
        let l = confounders.ncols();
        let mut c_gamma_y = DVector::zeros(n);
        #[allow(clippy::needless_range_loop)]
        for j in 0..l {
            let col = confounders.column(j);
            for i in 0..n {
                c_gamma_y[i] += col[i] * gamma_y[j];
            }
        }
        let c_gamma_y = standardize_dvector(&c_gamma_y);

        // Correlation between M_collider and C·γ_y should be high (~ρ)
        let collider_corrs: Vec<f32> = effects
            .iter()
            .filter(|e| e.is_collider())
            .map(|e| {
                let m_g = &expressions[e.gene_idx];
                let m_std = standardize_dvector(m_g);
                m_std.dot(&c_gamma_y) / n as f32
            })
            .collect();

        // Null gene correlations should be low
        let null_corrs: Vec<f32> = effects
            .iter()
            .filter(|e| e.role == GeneRole::Null)
            .map(|e| {
                let m_g = &expressions[e.gene_idx];
                let m_std = standardize_dvector(m_g);
                m_std.dot(&c_gamma_y) / n as f32
            })
            .collect();

        let mean_collider_corr =
            collider_corrs.iter().sum::<f32>() / collider_corrs.len().max(1) as f32;
        let mean_null_corr = null_corrs.iter().sum::<f32>() / null_corrs.len().max(1) as f32;

        eprintln!(
            "Mean |corr| with C·γ_y: collider={:.3}, null={:.3}",
            mean_collider_corr.abs(),
            mean_null_corr.abs(),
        );

        assert!(
            mean_collider_corr.abs() > 0.5,
            "Collider genes should have high correlation with C·γ_y, got {:.3}",
            mean_collider_corr.abs(),
        );
        assert!(
            mean_null_corr.abs() < 0.4,
            "Null genes should have low correlation with C·γ_y, got {:.3}",
            mean_null_corr.abs(),
        );
    }

    #[test]
    fn test_deterministic_seed() {
        let (_, genes, positions, chromosomes) = make_test_data();
        let e1 = sample_effects(&genes, &positions, &chromosomes, 3, 5, 3, 2, 42);
        let e2 = sample_effects(&genes, &positions, &chromosomes, 3, 5, 3, 2, 42);

        for (a, b) in e1.iter().zip(e2.iter()) {
            assert_eq!(a.eqtl_snp_indices, b.eqtl_snp_indices);
            assert_eq!(a.alpha, b.alpha);
            assert_eq!(a.beta, b.beta);
            assert_eq!(a.role, b.role);
            assert_eq!(a.is_observed, b.is_observed);
        }
    }

    #[test]
    fn test_pve_constraint_expression() {
        let (x, genes, positions, chromosomes) = make_test_data();
        let effects = sample_effects(&genes, &positions, &chromosomes, 3, 5, 5, 0, 42);
        let c = DMatrix::zeros(x.nrows(), 0);

        let result = generate_gene_expressions(&ExpressionParams {
            genotypes: &x,
            effects: &effects,
            confounders: &c,
            gamma_y: &[],
            h2_eqtl: 0.8,
            h2_conf_m: 0.5,
            collider_correlation: 0.0,
            seed: 42,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_pve_constraint_phenotype() {
        let n = 100;
        let expr = vec![DVector::zeros(n)];
        let effects = vec![MediationGeneEffects {
            gene_idx: 0,
            eqtl_snp_indices: vec![],
            alpha: vec![],
            beta: 1.0,
            role: GeneRole::Mediator,
            is_observed: true,
            num_cis_snps: 0,
        }];
        let x = DMatrix::<f32>::zeros(n, 10);
        let c = DMatrix::zeros(n, 0);

        let result = generate_mediated_phenotype(&PhenotypeParams {
            expressions: &expr,
            effects: &effects,
            genotypes: &x,
            confounders: &c,
            gamma_y: &[],
            h2_mediated: 0.5,
            h2_direct: 0.3,
            h2_conf_y: 0.5,
            seed: 42,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_split_discovery_replication() {
        let (disc, rep) = split_discovery_replication(100, 60, 42);
        assert_eq!(disc.len(), 60);
        assert_eq!(rep.len(), 40);

        // No overlap
        let disc_set: std::collections::HashSet<usize> = disc.iter().copied().collect();
        for &r in &rep {
            assert!(!disc_set.contains(&r));
        }
    }
}
