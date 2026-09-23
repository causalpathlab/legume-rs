//! Localized attention: each gene attends over its own cis peaks.
//!
//! `π_gp = softmax_{p ∈ cis(g)} [ -γ log(d_gp + c) + (ρ_g A)·(φ_p B) ]`,
//! pooled as `ψ_g = Σ_p π_gp φ_p` and trained so `ψ_g` agrees with `ρ_g`.

use chickpea::p2g::attention::{fit_attention, pool, AttentionConfig, LocalAttention};
use chickpea::p2g::cis::CisPairs;
use legume_numeric::candle::candle_core::{DType, Device, Tensor};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Pairs from explicit per-gene `(peak, distance)` lists; weights unused here.
fn pairs(genes: &[&[(u32, i64)]]) -> CisPairs {
    let mut gene_ptr = vec![0];
    let (mut peak, mut dist) = (Vec::new(), Vec::new());
    for g in genes {
        for &(p, d) in *g {
            peak.push(p);
            dist.push(d);
        }
        gene_ptr.push(peak.len());
    }
    let n = peak.len();
    CisPairs {
        gene_ptr,
        peak,
        dist,
        weight: vec![0.0; n],
        n_genes_placed: genes.len(),
        n_unparsed_peaks: 0,
        n_unreached_peaks: 0,
    }
}

fn random(rows: usize, cols: usize, rng: &mut StdRng) -> DMatrix<f32> {
    DMatrix::from_fn(rows, cols, |_, _| rng.random::<f32>() - 0.5)
}

#[test]
fn shares_sum_to_one_over_each_genes_own_peaks() {
    let mut rng = StdRng::seed_from_u64(1);
    let (rho, phi) = (random(3, 4, &mut rng), random(5, 4, &mut rng));
    // GENE3 has no peaks; peak 1 is a candidate of GENE1 and GENE2.
    let pairs = pairs(&[
        &[(0, 100), (1, 5_000)],
        &[(1, 2_000), (2, 9_000), (3, 40_000)],
        &[],
    ]);
    let cfg = AttentionConfig {
        epochs: 3,
        ..AttentionConfig::default()
    };
    let fit = fit_attention(&rho, &phi, &pairs, &cfg).unwrap();
    assert_eq!(fit.pi.len(), pairs.n_pairs());
    for g in 0..2 {
        let s: f32 = fit.pi[pairs.gene(g)].iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "gene {g}: shares sum to {s}");
        assert!(fit.pi[pairs.gene(g)].iter().all(|&v| v > 0.0));
    }
}

#[test]
fn at_the_start_the_shares_are_the_distance_prior() {
    let mut rng = StdRng::seed_from_u64(2);
    let (rho, phi) = (random(1, 4, &mut rng), random(3, 4, &mut rng));
    let pairs = pairs(&[&[(0, 0), (1, 10_000), (2, 100_000)]]);
    let cfg = AttentionConfig {
        epochs: 0,
        init_scale: 0.0,
        ..AttentionConfig::default()
    };
    let fit = fit_attention(&rho, &phi, &pairs, &cfg).unwrap();
    let k = |d: f64| (d + cfg.init_pseudocount).powf(-cfg.init_gamma);
    let z = k(0.0) + k(10_000.0) + k(100_000.0);
    for (i, d) in [0.0, 10_000.0, 100_000.0].into_iter().enumerate() {
        assert!((f64::from(fit.pi[i]) - k(d) / z).abs() < 1e-5, "pair {i}");
    }
}

#[test]
fn pooling_is_the_share_weighted_mean_of_peak_rows() {
    let mut rng = StdRng::seed_from_u64(3);
    let phi = random(3, 2, &mut rng);
    let pairs = pairs(&[&[(0, 0), (2, 0)], &[(1, 0)]]);
    let pi = [0.25f32, 0.75, 1.0];
    let psi = pool(&pairs, &pi, &phi);
    for k in 0..2 {
        let want0 = 0.25 * phi[(0, k)] + 0.75 * phi[(2, k)];
        assert!((psi[(0, k)] - want0).abs() < 1e-6);
        assert!((psi[(1, k)] - phi[(1, k)]).abs() < 1e-6);
    }
}

#[test]
fn autograd_matches_finite_differences() {
    let mut rng = StdRng::seed_from_u64(4);
    let (rho, phi) = (random(2, 3, &mut rng), random(4, 3, &mut rng));
    let pairs = pairs(&[
        &[(0, 500), (1, 8_000), (2, 30_000)],
        &[(2, 1_000), (3, 20_000)],
    ]);
    let dev = Device::Cpu;
    let att = LocalAttention::new(3, 2, 0.5, 1.0, 5_000.0, DType::F64, &dev, 9).unwrap();
    let rho_t = Tensor::from_slice(rho.transpose().as_slice(), (2, 3), &dev)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap();
    let phi_t = Tensor::from_slice(phi.transpose().as_slice(), (4, 3), &dev)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap();
    let genes = [0usize, 1];
    let loss_at = || -> f64 {
        att.batch_loss(&rho_t, &phi_t, &pairs, &genes)
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    };
    let grads = att
        .batch_loss(&rho_t, &phi_t, &pairs, &genes)
        .unwrap()
        .backward()
        .unwrap();
    for var in att.vars() {
        let g = grads
            .get(var.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let base = var
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let shape = var.as_tensor().shape().clone();
        for i in 0..base.len() {
            let eps = 1e-6;
            let at = |delta: f64| {
                let mut v = base.clone();
                v[i] += delta;
                var.set(&Tensor::from_vec(v, shape.clone(), &dev).unwrap())
                    .unwrap();
                loss_at()
            };
            let fd = (at(eps) - at(-eps)) / (2.0 * eps);
            var.set(&Tensor::from_vec(base.clone(), shape.clone(), &dev).unwrap())
                .unwrap();
            assert!(
                (fd - g[i]).abs() <= 1e-5 * (1.0 + fd.abs()),
                "entry {i}: autograd {} vs finite difference {fd}",
                g[i]
            );
        }
    }
}

/// Each gene's row is exactly one of its candidates' rows, placed FARTHEST so
/// the distance prior argues against it: the content term must find it.
#[test]
fn a_gene_built_from_one_far_peak_attends_to_it() {
    let mut rng = StdRng::seed_from_u64(5);
    let h = 6;
    let n_genes = 40;
    let per_gene = 4;
    let phi = random(n_genes * per_gene, h, &mut rng) * 2.0;
    let mut lists: Vec<Vec<(u32, i64)>> = Vec::new();
    let mut rho = DMatrix::zeros(n_genes, h);
    for g in 0..n_genes {
        let base = (g * per_gene) as u32;
        let true_peak = base + per_gene as u32 - 1;
        lists.push(
            (0..per_gene as u32)
                .map(|j| (base + j, 2_000 * i64::from(j)))
                .collect(),
        );
        rho.set_row(g, &phi.row(true_peak as usize));
    }
    let refs: Vec<&[(u32, i64)]> = lists.iter().map(Vec::as_slice).collect();
    let pairs = pairs(&refs);
    let cfg = AttentionConfig {
        rank: h,
        epochs: 300,
        ..AttentionConfig::default()
    };
    let fit = fit_attention(&rho, &phi, &pairs, &cfg).unwrap();
    let mut hits = 0;
    for g in 0..n_genes {
        let r = pairs.gene(g);
        let shares = &fit.pi[r];
        if shares[per_gene - 1] > 0.5 {
            hits += 1;
        }
    }
    assert!(
        hits >= 36,
        "only {hits} of {n_genes} genes put most of their share on the true peak"
    );
}

/// Two candidates with the SAME row: content cannot separate them, so the
/// nearer one must take the larger share.
#[test]
fn an_identical_bystander_farther_away_loses_on_distance() {
    let mut rng = StdRng::seed_from_u64(6);
    let row = random(1, 4, &mut rng);
    let phi = DMatrix::from_fn(2, 4, |_, k| row[(0, k)]);
    let rho = row.clone();
    let pairs = pairs(&[&[(0, 1_000), (1, 60_000)]]);
    let fit = fit_attention(&rho, &phi, &pairs, &AttentionConfig::default()).unwrap();
    assert!(
        fit.pi[0] > fit.pi[1],
        "near {} vs far {}",
        fit.pi[0],
        fit.pi[1]
    );
}
