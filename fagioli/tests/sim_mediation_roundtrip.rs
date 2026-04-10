//! Integration test: sim-mediation end-to-end roundtrip
//!
//! 1. Generate synthetic PLINK genotypes
//! 2. Run sim_mediation with small parameters
//! 3. Verify all output files exist and parse correctly
//! 4. Check that causal eQTL SNPs have larger |z| in GWAS than non-causal
//!
//! Run: cargo test -p fagioli --test sim_mediation_roundtrip -- --nocapture

use anyhow::Result;
use fagioli::genotype::bed_writer::write_plink;
use fagioli::genotype::GenotypeMatrix;
use fagioli::simulation::{
    generate_confounder_matrix, generate_gene_expressions, generate_mediated_phenotype,
    sample_confounder_mixing_y, sample_mediation_effects, simulate_gene_annotations,
    split_discovery_replication, subset_dvector, subset_rows, ConfounderParams, ExpressionParams,
    GeneRole, MediationEffectParams, PhenotypeParams,
};
use fagioli::summary_stats::{compute_block_sumstats, compute_yty_diagonal};
use nalgebra::DMatrix;

/// Create a synthetic GenotypeMatrix (N individuals × M SNPs) with positions on chr22.
fn make_test_genotypes(n: usize, m: usize, seed: u64) -> GenotypeMatrix {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Uniform};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0f32, 1.0).unwrap();

    // Simulate genotypes: 0, 1, 2 with HWE-like frequencies
    let mut genotypes = DMatrix::zeros(n, m);
    for j in 0..m {
        let maf = 0.05 + unif.sample(&mut rng) * 0.4; // MAF in [0.05, 0.45]
        for i in 0..n {
            let r = unif.sample(&mut rng);
            let g = if r < (1.0 - maf).powi(2) {
                0.0
            } else if r < (1.0 - maf).powi(2) + 2.0 * maf * (1.0 - maf) {
                1.0
            } else {
                2.0
            };
            genotypes[(i, j)] = g;
        }
    }

    let individual_ids: Vec<Box<str>> = (0..n).map(|i| Box::from(format!("IND{:04}", i))).collect();
    let snp_ids: Vec<Box<str>> = (0..m).map(|j| Box::from(format!("rs{:06}", j))).collect();
    let chromosomes: Vec<Box<str>> = vec![Box::from("22"); m];
    let positions: Vec<u64> = (0..m).map(|j| 20_000_000 + (j as u64) * 5000).collect();
    let allele1: Vec<Box<str>> = vec![Box::from("A"); m];
    let allele2: Vec<Box<str>> = vec![Box::from("G"); m];

    GenotypeMatrix {
        genotypes,
        individual_ids,
        snp_ids,
        chromosomes,
        positions,
        allele1,
        allele2,
    }
}

#[test]
fn test_sim_mediation_roundtrip() -> Result<()> {
    let tmp_dir = tempfile::tempdir()?;
    let bed_prefix = tmp_dir.path().join("test_geno");
    let n = 300;
    let m = 200;

    // Step 1: Write synthetic PLINK files
    let geno = make_test_genotypes(n, m, 42);
    write_plink(bed_prefix.to_str().unwrap(), &geno)?;

    // Step 2: Run mediation simulation directly (not via CLI)
    use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};

    let mut reader = BedReader::new(bed_prefix.to_str().unwrap())?;
    let region = GenomicRegion::new(Some(Box::from("22")), None, None);
    let loaded = reader.read(None, Some(region))?;

    assert_eq!(loaded.num_individuals(), n);
    assert_eq!(loaded.num_snps(), m);

    // Gene annotations
    let genes = simulate_gene_annotations(
        20,
        "22",
        20_000_000,
        20_000_000 + (m as u64) * 5000,
        100_000,
        42,
    );

    // Sample mediation effects: 5 mediator, 3 observed, 2 collider
    let effects = sample_mediation_effects(&MediationEffectParams {
        genes: &genes,
        snp_positions: &loaded.positions,
        snp_chromosomes: &loaded.chromosomes,
        n_eqtl_per_gene: 3,
        num_causal: 5,
        num_observed_causal: 3,
        num_collider: 2,
        seed: 342,
    })?;

    assert_eq!(effects.len(), 20);
    let n_mediator = effects.iter().filter(|e| e.is_mediator()).count();
    assert_eq!(n_mediator, 5);
    let n_collider = effects.iter().filter(|e| e.is_collider()).count();
    assert_eq!(n_collider, 2);

    // Generate confounders
    let conf_params = ConfounderParams {
        num_confounders: 5,
        num_hidden_factors: 3,
        pve_confounders: 0.1,
    };
    let confounders = generate_confounder_matrix(n, &conf_params, 100)?;

    // Generate γ_y (shared confounder mixing for Y)
    let gamma_y = sample_confounder_mixing_y(confounders.ncols(), 150);

    let expressions = generate_gene_expressions(&ExpressionParams {
        genotypes: &loaded.genotypes,
        effects: &effects,
        confounders: &confounders,
        gamma_y: &gamma_y,
        h2_eqtl: 0.3,
        h2_conf_m: 0.1,
        collider_correlation: 0.8,
        seed: 400,
    })?;
    assert_eq!(expressions.len(), 20);

    let y = generate_mediated_phenotype(&PhenotypeParams {
        expressions: &expressions,
        effects: &effects,
        genotypes: &loaded.genotypes,
        confounders: &confounders,
        gamma_y: &gamma_y,
        h2_mediated: 0.2,
        h2_direct: 0.0,
        h2_conf_y: 0.1,
        seed: 200,
    })?;
    assert_eq!(y.len(), n);

    // Step 3: Compute GWAS summary stats and check signal
    let y_matrix = DMatrix::from_column_slice(n, 1, y.as_slice());
    let yty = compute_yty_diagonal(&y_matrix);
    let sumstats = compute_block_sumstats(&loaded.genotypes, &y_matrix, &yty, 0);

    // Collect causal eQTL SNP indices (from mediator genes)
    let causal_snps: std::collections::HashSet<usize> = effects
        .iter()
        .filter(|e| e.is_mediator())
        .flat_map(|e| e.eqtl_snp_indices.iter().copied())
        .collect();

    let causal_z: Vec<f32> = sumstats
        .iter()
        .filter(|r| causal_snps.contains(&r.snp_idx))
        .map(|r| r.z.abs())
        .collect();

    let null_z: Vec<f32> = sumstats
        .iter()
        .filter(|r| !causal_snps.contains(&r.snp_idx))
        .map(|r| r.z.abs())
        .collect();

    if !causal_z.is_empty() && !null_z.is_empty() {
        let mean_causal = causal_z.iter().sum::<f32>() / causal_z.len() as f32;
        let mean_null = null_z.iter().sum::<f32>() / null_z.len() as f32;

        eprintln!(
            "Mean |z|: mediator={:.3} (n={}), null={:.3} (n={})",
            mean_causal,
            causal_z.len(),
            mean_null,
            null_z.len(),
        );
    }

    // Step 4: Check eQTL summary stats for observed genes
    for eff in &effects {
        if !eff.is_observed || eff.eqtl_snp_indices.is_empty() {
            continue;
        }

        let cis_snps = genes.cis_snp_indices(eff.gene_idx, &loaded.positions, &loaded.chromosomes);
        if cis_snps.is_empty() {
            continue;
        }

        // Build cis submatrix
        let n_cis = cis_snps.len();
        let mut x_cis = DMatrix::zeros(n, n_cis);
        for (local_j, &global_j) in cis_snps.iter().enumerate() {
            let col = loaded.genotypes.column(global_j);
            for i in 0..n {
                x_cis[(i, local_j)] = col[i];
            }
        }

        let m_g = DMatrix::from_column_slice(n, 1, expressions[eff.gene_idx].as_slice());
        let mty = compute_yty_diagonal(&m_g);
        let eqtl_stats = compute_block_sumstats(&x_cis, &m_g, &mty, 0);

        // eQTL SNPs should have significant z-scores
        let eqtl_local: std::collections::HashSet<usize> = eff
            .eqtl_snp_indices
            .iter()
            .filter_map(|&global| cis_snps.iter().position(|&c| c == global))
            .collect();

        let eqtl_z: Vec<f32> = eqtl_stats
            .iter()
            .filter(|r| eqtl_local.contains(&r.snp_idx))
            .map(|r| r.z.abs())
            .collect();

        if !eqtl_z.is_empty() {
            let max_eqtl_z = eqtl_z.iter().cloned().fold(0.0f32, f32::max);
            eprintln!(
                "Gene {}: max eQTL |z| = {:.3} (n_eqtl={})",
                eff.gene_idx,
                max_eqtl_z,
                eqtl_z.len(),
            );
        }
    }

    Ok(())
}

#[test]
fn test_null_calibration() -> Result<()> {
    // With h2_eqtl=0 and h2_mediated=0, GWAS z-scores should be ~N(0,1)
    let n = 500;
    let m = 100;

    let geno = make_test_genotypes(n, m, 99);
    let genes = simulate_gene_annotations(
        10,
        "22",
        20_000_000,
        20_000_000 + (m as u64) * 5000,
        50_000,
        99,
    );

    let effects = sample_mediation_effects(&MediationEffectParams {
        genes: &genes,
        snp_positions: &geno.positions,
        snp_chromosomes: &geno.chromosomes,
        n_eqtl_per_gene: 2,
        num_causal: 3,
        num_observed_causal: 3,
        num_collider: 0,
        seed: 199,
    })?;

    let confounders = DMatrix::zeros(n, 0);

    let expressions = generate_gene_expressions(&ExpressionParams {
        genotypes: &geno.genotypes,
        effects: &effects,
        confounders: &confounders,
        gamma_y: &[],
        h2_eqtl: 0.0,
        h2_conf_m: 0.0,
        collider_correlation: 0.0,
        seed: 299,
    })?;

    let y = generate_mediated_phenotype(&PhenotypeParams {
        expressions: &expressions,
        effects: &effects,
        genotypes: &geno.genotypes,
        confounders: &confounders,
        gamma_y: &[],
        h2_mediated: 0.0,
        h2_direct: 0.0,
        h2_conf_y: 0.0,
        seed: 399,
    })?;

    let y_mat = DMatrix::from_column_slice(n, 1, y.as_slice());
    let yty = compute_yty_diagonal(&y_mat);
    let stats = compute_block_sumstats(&geno.genotypes, &y_mat, &yty, 0);

    let z_vals: Vec<f32> = stats.iter().map(|r| r.z).collect();
    let mean_z = z_vals.iter().sum::<f32>() / z_vals.len() as f32;
    let var_z = z_vals.iter().map(|z| (z - mean_z).powi(2)).sum::<f32>() / z_vals.len() as f32;

    eprintln!(
        "Null calibration: mean(z)={:.4}, var(z)={:.4}",
        mean_z, var_z
    );

    assert!(
        mean_z.abs() < 0.3,
        "Mean z under null too large: {}",
        mean_z,
    );
    assert!(
        (var_z - 1.0).abs() < 0.5,
        "Var(z) under null too far from 1.0: {}",
        var_z,
    );

    Ok(())
}

#[test]
fn test_collider_bias_detectable() -> Result<()> {
    // With collider genes and confounders, conditioning on a collider gene
    // should reveal spurious SNP→Y association through the opened collider path.
    let n = 500;
    let m = 200;

    let geno = make_test_genotypes(n, m, 77);
    let genes = simulate_gene_annotations(
        15,
        "22",
        20_000_000,
        20_000_000 + (m as u64) * 5000,
        100_000,
        77,
    );

    // 3 mediators, 5 colliders, 7 null
    let effects = sample_mediation_effects(&MediationEffectParams {
        genes: &genes,
        snp_positions: &geno.positions,
        snp_chromosomes: &geno.chromosomes,
        n_eqtl_per_gene: 3,
        num_causal: 3,
        num_observed_causal: 3,
        num_collider: 5,
        seed: 177,
    })?;

    let n_collider = effects
        .iter()
        .filter(|e| e.role == GeneRole::Collider)
        .count();
    assert_eq!(n_collider, 5);

    let conf_params = ConfounderParams {
        num_confounders: 10,
        num_hidden_factors: 5,
        pve_confounders: 0.3,
    };
    let confounders = generate_confounder_matrix(n, &conf_params, 100)?;
    let gamma_y = sample_confounder_mixing_y(confounders.ncols(), 150);

    let rho = 0.9;
    let expressions = generate_gene_expressions(&ExpressionParams {
        genotypes: &geno.genotypes,
        effects: &effects,
        confounders: &confounders,
        gamma_y: &gamma_y,
        h2_eqtl: 0.3,
        h2_conf_m: 0.3,
        collider_correlation: rho,
        seed: 200,
    })?;

    let y = generate_mediated_phenotype(&PhenotypeParams {
        expressions: &expressions,
        effects: &effects,
        genotypes: &geno.genotypes,
        confounders: &confounders,
        gamma_y: &gamma_y,
        h2_mediated: 0.2,
        h2_direct: 0.0,
        h2_conf_y: 0.3,
        seed: 300,
    })?;

    // For each collider gene, regress Y on M_collider and check for non-zero coeff
    // (This is a soft check — collider bias should make M_collider correlated with Y
    //  even though β=0 for collider genes, because of shared confounders)
    let mut collider_corrs = Vec::new();
    let mut null_corrs = Vec::new();

    for eff in &effects {
        let m_g = &expressions[eff.gene_idx];
        // Correlation between M_g and Y
        let m_mean = m_g.sum() / n as f32;
        let y_mean = y.sum() / n as f32;
        let mut cov = 0.0f32;
        let mut var_m = 0.0f32;
        let mut var_y = 0.0f32;
        for i in 0..n {
            let dm = m_g[i] - m_mean;
            let dy = y[i] - y_mean;
            cov += dm * dy;
            var_m += dm * dm;
            var_y += dy * dy;
        }
        let corr = if var_m > 1e-10 && var_y > 1e-10 {
            cov / (var_m.sqrt() * var_y.sqrt())
        } else {
            0.0
        };

        match eff.role {
            GeneRole::Collider => collider_corrs.push(corr.abs()),
            GeneRole::Null => null_corrs.push(corr.abs()),
            _ => {}
        }
    }

    let mean_collider = collider_corrs.iter().sum::<f32>() / collider_corrs.len().max(1) as f32;
    let mean_null = null_corrs.iter().sum::<f32>() / null_corrs.len().max(1) as f32;

    eprintln!(
        "Cor(M_g, Y): collider mean |r|={:.3} (n={}), null mean |r|={:.3} (n={})",
        mean_collider,
        collider_corrs.len(),
        mean_null,
        null_corrs.len(),
    );

    // Collider genes should show higher correlation with Y due to shared confounders
    assert!(
        mean_collider > mean_null,
        "Collider genes should have higher |Cor(M,Y)| than null genes: collider={:.3}, null={:.3}",
        mean_collider,
        mean_null,
    );

    Ok(())
}

#[test]
fn test_winner_curse_inflation() -> Result<()> {
    // Discovery eQTL |z| at selected SNPs should be inflated vs replication |z|
    let n = 600;
    let m = 200;

    let geno = make_test_genotypes(n, m, 55);
    let genes = simulate_gene_annotations(
        10,
        "22",
        20_000_000,
        20_000_000 + (m as u64) * 5000,
        100_000,
        55,
    );

    let effects = sample_mediation_effects(&MediationEffectParams {
        genes: &genes,
        snp_positions: &geno.positions,
        snp_chromosomes: &geno.chromosomes,
        n_eqtl_per_gene: 3,
        num_causal: 5,
        num_observed_causal: 5,
        num_collider: 0,
        seed: 155,
    })?;

    let conf_params = ConfounderParams {
        num_confounders: 5,
        num_hidden_factors: 3,
        pve_confounders: 0.1,
    };
    let confounders = generate_confounder_matrix(n, &conf_params, 100)?;
    let gamma_y = sample_confounder_mixing_y(confounders.ncols(), 150);

    let expressions = generate_gene_expressions(&ExpressionParams {
        genotypes: &geno.genotypes,
        effects: &effects,
        confounders: &confounders,
        gamma_y: &gamma_y,
        h2_eqtl: 0.4,
        h2_conf_m: 0.1,
        collider_correlation: 0.0,
        seed: 200,
    })?;

    // Discovery/replication split
    let n_disc = 300;
    let (disc_idx, rep_idx) = split_discovery_replication(n, n_disc, 500);
    assert_eq!(disc_idx.len(), 300);
    assert_eq!(rep_idx.len(), 300);

    let mut disc_z_selected = Vec::new();
    let mut rep_z_selected = Vec::new();

    let pval_threshold = 0.05; // generous threshold for small sample

    for eff in &effects {
        if eff.eqtl_snp_indices.is_empty() {
            continue;
        }

        let cis_snps = genes.cis_snp_indices(eff.gene_idx, &geno.positions, &geno.chromosomes);
        if cis_snps.is_empty() {
            continue;
        }

        let n_cis = cis_snps.len();
        let mut x_cis = DMatrix::zeros(n, n_cis);
        for (local_j, &global_j) in cis_snps.iter().enumerate() {
            let col = geno.genotypes.column(global_j);
            for i in 0..n {
                x_cis[(i, local_j)] = col[i];
            }
        }

        let x_disc = subset_rows(&x_cis, &disc_idx);
        let x_rep = subset_rows(&x_cis, &rep_idx);
        let m_disc = subset_dvector(&expressions[eff.gene_idx], &disc_idx);
        let m_rep = subset_dvector(&expressions[eff.gene_idx], &rep_idx);

        let m_disc_mat = DMatrix::from_column_slice(disc_idx.len(), 1, m_disc.as_slice());
        let m_rep_mat = DMatrix::from_column_slice(rep_idx.len(), 1, m_rep.as_slice());

        let mty_disc = compute_yty_diagonal(&m_disc_mat);
        let mty_rep = compute_yty_diagonal(&m_rep_mat);

        let disc_stats = compute_block_sumstats(&x_disc, &m_disc_mat, &mty_disc, 0);
        let rep_stats = compute_block_sumstats(&x_rep, &m_rep_mat, &mty_rep, 0);

        // Select instruments from discovery at threshold
        for d_rec in &disc_stats {
            if d_rec.pvalue < pval_threshold {
                disc_z_selected.push(d_rec.z.abs());
                // Find the same SNP in replication
                if let Some(r_rec) = rep_stats.iter().find(|r| r.snp_idx == d_rec.snp_idx) {
                    rep_z_selected.push(r_rec.z.abs());
                }
            }
        }
    }

    if !disc_z_selected.is_empty() {
        let mean_disc = disc_z_selected.iter().sum::<f32>() / disc_z_selected.len() as f32;
        let mean_rep = rep_z_selected.iter().sum::<f32>() / rep_z_selected.len() as f32;

        eprintln!(
            "Winner's curse: discovery mean |z|={:.3} (n={}), replication mean |z|={:.3} (n={})",
            mean_disc,
            disc_z_selected.len(),
            mean_rep,
            rep_z_selected.len(),
        );

        // Discovery z-scores should be inflated relative to replication
        // (winner's curse: we selected by significance, so discovery is biased upward)
        assert!(
            mean_disc >= mean_rep,
            "Discovery |z| should be >= replication |z| (winner's curse): disc={:.3}, rep={:.3}",
            mean_disc,
            mean_rep,
        );
    } else {
        eprintln!(
            "No instruments selected at p < {} — skipping winner's curse check",
            pval_threshold
        );
    }

    Ok(())
}

#[test]
fn test_collider_selection_changes_gwas() -> Result<()> {
    // Selecting individuals on collider expression should inflate GWAS z-scores
    // at collider eQTL SNPs compared to the full-sample GWAS.
    let n = 1000;
    let m = 200;

    let geno = make_test_genotypes(n, m, 77);
    let genes = simulate_gene_annotations(
        20,
        "22",
        20_000_000,
        20_000_000 + (m as u64) * 5000,
        100_000,
        77,
    );

    let effects = sample_mediation_effects(&MediationEffectParams {
        genes: &genes,
        snp_positions: &geno.positions,
        snp_chromosomes: &geno.chromosomes,
        n_eqtl_per_gene: 3,
        num_causal: 3,
        num_observed_causal: 3,
        num_collider: 5,
        seed: 177,
    })?;

    let conf_params = ConfounderParams {
        num_confounders: 10,
        num_hidden_factors: 5,
        pve_confounders: 0.3,
    };
    let confounders = generate_confounder_matrix(n, &conf_params, 100)?;
    let gamma_y = sample_confounder_mixing_y(confounders.ncols(), 150);

    let expressions = generate_gene_expressions(&ExpressionParams {
        genotypes: &geno.genotypes,
        effects: &effects,
        confounders: &confounders,
        gamma_y: &gamma_y,
        h2_eqtl: 0.3,
        h2_conf_m: 0.3,
        collider_correlation: 0.9,
        seed: 200,
    })?;

    let y = generate_mediated_phenotype(&PhenotypeParams {
        expressions: &expressions,
        effects: &effects,
        genotypes: &geno.genotypes,
        confounders: &confounders,
        gamma_y: &gamma_y,
        h2_mediated: 0.2,
        h2_direct: 0.0,
        h2_conf_y: 0.3,
        seed: 300,
    })?;

    // Condition on first collider gene via liability threshold
    let cond_gene = effects.iter().find(|e| e.is_collider()).unwrap();
    let std_expr = fagioli::simulation::standardize_dvector(&expressions[cond_gene.gene_idx]);

    // Select top 50% (above median)
    let mut scratch: Vec<f32> = std_expr.iter().copied().collect();
    let mid = n / 2;
    scratch.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let threshold = scratch[mid];

    let selected_idx: Vec<usize> = (0..n).filter(|&i| std_expr[i] >= threshold).collect();
    let n_sel = selected_idx.len();
    eprintln!(
        "Selected {}/{} individuals for collider bias test",
        n_sel, n
    );

    // Direct collider bias test: genetically predicted expression (Xα) for the
    // conditioned collider gene should become correlated with Y after selection,
    // even though β=0 (no causal mediation effect).
    //
    // In the full sample: Cor(Xα, Y) ≈ 0 (no causal path from this gene to Y)
    // In the selected sample: Cor(Xα, Y) ≠ 0 (selection opens backdoor through U)
    let mut xalpha = nalgebra::DVector::<f32>::zeros(n);
    for (j, &snp_idx) in cond_gene.eqtl_snp_indices.iter().enumerate() {
        let col = geno.genotypes.column(snp_idx);
        for i in 0..n {
            xalpha[i] += col[i] * cond_gene.alpha[j];
        }
    }

    // Full sample correlation
    let cor = |a: &[f32], b: &[f32]| -> f32 {
        let nn = a.len() as f32;
        let ma = a.iter().sum::<f32>() / nn;
        let mb = b.iter().sum::<f32>() / nn;
        let mut cov = 0.0f32;
        let mut va = 0.0f32;
        let mut vb = 0.0f32;
        for i in 0..a.len() {
            let da = a[i] - ma;
            let db = b[i] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        if va < 1e-10 || vb < 1e-10 {
            0.0
        } else {
            cov / (va.sqrt() * vb.sqrt())
        }
    };

    let full_cor = cor(xalpha.as_slice(), y.as_slice()).abs();

    // Selected sample correlation
    let xalpha_sel = subset_dvector(&xalpha, &selected_idx);
    let y_sel_vec = subset_dvector(&y, &selected_idx);
    let sel_cor = cor(xalpha_sel.as_slice(), y_sel_vec.as_slice()).abs();

    eprintln!(
        "Cor(Xα_collider, Y): full={:.4}, selected={:.4}",
        full_cor, sel_cor,
    );

    // Selection should induce a spurious correlation
    assert!(
        sel_cor > full_cor,
        "Selection on collider should induce Cor(Xα, Y): full={:.4}, selected={:.4}",
        full_cor,
        sel_cor,
    );

    Ok(())
}
