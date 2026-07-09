use super::*;

const H: usize = 4;
const N_PB: usize = 60;
/// `fit_calibration` needs `MIN_CALIB_GENES_PER_DIM * H` live trained genes
/// before it will fit a map; stay above it wherever calibration is asserted.
const MIN_TRAINED: usize = MIN_CALIB_GENES_PER_DIM * H;

/// Deterministic uniform(-0.5, 0.5). No `rand` dep, and the tests stay
/// reproducible across platforms.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// A frozen pseudobulk side: `θ_pb`, `b_pb`, and the planted per-gene `β*`.
struct Fixture {
    theta: Vec<f32>,
    bias: Vec<f32>,
    beta_star: Vec<Vec<f32>>,
}

fn fixture(n_genes: usize, seed: u64) -> Fixture {
    let mut s = seed;
    let theta: Vec<f32> = (0..N_PB * H).map(|_| lcg(&mut s) * 1.6).collect();
    let bias: Vec<f32> = (0..N_PB).map(|_| lcg(&mut s) * 0.4).collect();
    let beta_star: Vec<Vec<f32>> = (0..n_genes)
        .map(|_| (0..H).map(|_| lcg(&mut s) * 1.2).collect())
        .collect();
    Fixture {
        theta,
        bias,
        beta_star,
    }
}

/// Noiseless Poisson rate `exp(⟨v, θ_p⟩ + b_p)`.
fn rate(f: &Fixture, v: &[f32], p: usize) -> f32 {
    let dot: f32 = f.theta[p * H..(p + 1) * H]
        .iter()
        .zip(v)
        .map(|(a, b)| a * b)
        .sum();
    (dot + f.bias[p]).exp()
}

/// Unit exposure (`size_p = 1`) so the planted rate IS the count and the recovery
/// tests keep their closed form. `exposure_offset_rescales_edges` covers `size ≠ 1`.
fn stacked<'a>(f: &Fixture, counts: &'a DMatrix<f32>) -> StackedPb<'a> {
    StackedPb {
        theta: f.theta.clone(),
        bias: f.bias.clone(),
        counts: vec![counts],
        sizes: vec![vec![1.0; N_PB]],
        offsets: vec![0],
    }
}

fn cfg(n_rows: usize, unspliced: Vec<bool>, row_to_gene: Vec<u32>) -> FeatureProjectionConfig {
    assert_eq!(unspliced.len(), n_rows);
    FeatureProjectionConfig {
        ridge: 1e-2,
        calib_ridge: 1e-3,
        backend_row_to_gene: row_to_gene,
        backend_unspliced_rows: unspliced,
        with_velocity: true,
        // Off by default: these check the solve + calibration algebra on planted,
        // non-degenerate genes. The gate has its own test.
        null_fdr: 0.0,
    }
}

/// Trained side: gene-level β for backend genes `0..n_trained`.
fn trained(beta: &[f32], n_trained: usize) -> TrainedBeta {
    TrainedBeta {
        beta: beta.to_vec(),
        backend_gene_id: (0..n_trained as u32).collect(),
    }
}

/// Identity gene map, all spliced: backend row `g` is gene `g`.
fn flat_setup(n_genes: usize, seed: u64) -> (Fixture, DMatrix<f32>) {
    let f = fixture(n_genes, seed);
    let counts = DMatrix::from_fn(n_genes, N_PB, |g, p| rate(&f, &f.beta_star[g], p));
    (f, counts)
}

// Trained genes carry their planted β, so the re-solve reproduces them and M ≈ I.
// Held-out genes must come back aligned to β*.
#[test]
fn held_out_beta_recovers_planted_gene() {
    let (n_genes, n_trained) = (MIN_TRAINED + 10, MIN_TRAINED + 2);
    let (f, counts) = flat_setup(n_genes, 7);
    let pb = stacked(&f, &counts);

    let beta: Vec<f32> = (0..n_trained)
        .flat_map(|g| f.beta_star[g].clone())
        .collect();
    let c = cfg(n_genes, vec![false; n_genes], (0..n_genes as u32).collect());

    let out = project_held_out_features(&pb, H, &trained(&beta, n_trained), &c);

    assert_eq!(
        out.gene_ids,
        (n_trained as u32..n_genes as u32).collect::<Vec<_>>()
    );
    assert!(
        out.calib.mean_cosine > 0.97,
        "calibration should be near-identity (cos={})",
        out.calib.mean_cosine
    );
    for (i, &g) in out.gene_ids.iter().enumerate() {
        let cos = cosine(&out.beta[i * H..(i + 1) * H], &f.beta_star[g as usize]);
        assert!(cos > 0.95, "gene {g} misaligned (cos={cos:.3})");
    }
    assert!(
        out.deviance.iter().all(|d| d.abs() < 1e-2),
        "noiseless data should fit exactly"
    );
    assert!(out.live.iter().all(|&l| l), "gate off ⇒ every gene live");
}

// Trained β lives in a rotated/scaled frame B* · A. The ridge map must learn A
// and carry held-out genes into the same frame.
#[test]
fn calibration_absorbs_linear_frame() {
    let (n_genes, n_trained) = (MIN_TRAINED + 14, MIN_TRAINED + 6);
    let (f, counts) = flat_setup(n_genes, 11);
    let pb = stacked(&f, &counts);

    // Aggressively non-orthogonal: strong off-diagonal mixing plus anisotropic
    // scaling, so an uncalibrated β̂ lands nowhere near `B* · A`.
    let a = DMatrix::<f32>::from_row_slice(
        H,
        H,
        &[
            0.6, 0.9, 0.0, -0.3, //
            -0.8, 0.5, 0.4, 0.0, //
            0.2, -0.3, 1.2, 0.7, //
            0.0, 0.4, -0.6, 0.8,
        ],
    );
    let map = |v: &[f32]| -> Vec<f32> {
        (0..H)
            .map(|j| (0..H).map(|k| v[k] * a[(k, j)]).sum())
            .collect()
    };

    let beta: Vec<f32> = (0..n_trained).flat_map(|g| map(&f.beta_star[g])).collect();
    let c = cfg(n_genes, vec![false; n_genes], (0..n_genes as u32).collect());

    let out = project_held_out_features(&pb, H, &trained(&beta, n_trained), &c);

    assert!(
        out.calib.r2 > 0.95,
        "frame not learned (R²={})",
        out.calib.r2
    );
    for (i, &g) in out.gene_ids.iter().enumerate() {
        let want = map(&f.beta_star[g as usize]);
        let cos = cosine(&out.beta[i * H..(i + 1) * H], &want);
        assert!(
            cos > 0.95,
            "gene {g} not carried into the trained frame (cos={cos:.3})"
        );
    }
}

// Two rows per gene: 2g spliced (β), 2g+1 unspliced (β + δ). The increment solve
// must recover δ* holding the freshly-solved β̂ fixed.
#[test]
fn held_out_velocity_recovers_planted_delta() {
    let (n_genes, n_trained) = (MIN_TRAINED + 8, MIN_TRAINED + 1);
    let f = fixture(n_genes * 2, 23);
    let delta_star: Vec<Vec<f32>> = (0..n_genes)
        .map(|g| f.beta_star[n_genes + g].iter().map(|x| x * 0.6).collect())
        .collect();

    let counts = DMatrix::from_fn(n_genes * 2, N_PB, |row, p| {
        let g = row / 2;
        if row % 2 == 0 {
            rate(&f, &f.beta_star[g], p)
        } else {
            let nascent: Vec<f32> = f.beta_star[g]
                .iter()
                .zip(&delta_star[g])
                .map(|(b, d)| b + d)
                .collect();
            rate(&f, &nascent, p)
        }
    });
    let pb = stacked(&f, &counts);

    let row_to_gene: Vec<u32> = (0..n_genes as u32).flat_map(|g| [g, g]).collect();
    let unspliced: Vec<bool> = (0..n_genes * 2).map(|r| r % 2 == 1).collect();
    // The trained side is the per-gene β Var — NOT the e_feat rows, whose
    // unspliced entries would carry the extra δ_g.
    let beta: Vec<f32> = (0..n_trained)
        .flat_map(|g| f.beta_star[g].clone())
        .collect();

    let c = cfg(n_genes * 2, unspliced, row_to_gene);
    let out = project_held_out_features(&pb, H, &trained(&beta, n_trained), &c);

    let delta = out.delta.as_ref().expect("velocity requested");
    for (i, &g) in out.gene_ids.iter().enumerate() {
        let cos_b = cosine(&out.beta[i * H..(i + 1) * H], &f.beta_star[g as usize]);
        assert!(cos_b > 0.95, "gene {g} β misaligned (cos={cos_b:.3})");
        let cos_d = cosine(&delta[i * H..(i + 1) * H], &delta_star[g as usize]);
        assert!(cos_d > 0.90, "gene {g} δ misaligned (cos={cos_d:.3})");
    }
}

// REGRESSION. The feature-null QC drops feature ROWS, and it keeps unspliced rows
// preferentially (‖β+δ‖² > ‖β‖²). A gene whose only surviving row is unspliced
// still has a trained β_g and must NOT be re-projected — and it must still be
// usable for calibration, re-solved from its spliced counts.
#[test]
fn gene_surviving_only_as_unspliced_is_trained_not_projected() {
    let n_genes = MIN_TRAINED + 6;
    let n_trained = MIN_TRAINED + 1;
    let f = fixture(n_genes, 31);
    let counts = DMatrix::from_fn(n_genes * 2, N_PB, |row, p| {
        rate(&f, &f.beta_star[row / 2], p)
    });
    let pb = stacked(&f, &counts);

    let row_to_gene: Vec<u32> = (0..n_genes as u32).flat_map(|g| [g, g]).collect();
    let unspliced: Vec<bool> = (0..n_genes * 2).map(|r| r % 2 == 1).collect();
    let beta: Vec<f32> = (0..n_trained)
        .flat_map(|g| f.beta_star[g].clone())
        .collect();

    let c = cfg(n_genes * 2, unspliced, row_to_gene);
    // `backend_gene_id` names the trained genes directly — how they survived (as a
    // spliced row, an unspliced row, or both) is irrelevant here.
    let out = project_held_out_features(&pb, H, &trained(&beta, n_trained), &c);

    assert_eq!(
        out.gene_ids,
        (n_trained as u32..n_genes as u32).collect::<Vec<_>>(),
        "only genes with no trained β at all may be projected"
    );
    assert_eq!(
        out.calib.n_trained, n_trained,
        "every trained gene should calibrate, even one that survived only as unspliced"
    );
    assert!(out.calib.mean_cosine > 0.97);
}

// Too few live trained genes to determine an H×H map: fall back to a SCALAR
// rescale, not to the identity. `faba annotate` compares β across genes, so
// leaving projected β on the raw Poisson-MAP scale (measured at 4–5× the trained
// norm) would let projected genes dominate every marker ranking. A scalar cannot
// fix the frame, but it fixes the part that silently corrupts downstream results.
#[test]
fn calibration_falls_back_to_scalar_when_too_few_trained_genes() {
    let (n_genes, n_trained) = (MIN_TRAINED, 3);
    let (f, counts) = flat_setup(n_genes, 37);
    let pb = stacked(&f, &counts);
    let beta: Vec<f32> = (0..n_trained)
        .flat_map(|g| f.beta_star[g].clone())
        .collect();
    let c = cfg(n_genes, vec![false; n_genes], (0..n_genes as u32).collect());

    let out = project_held_out_features(&pb, H, &trained(&beta, n_trained), &c);

    assert_eq!(out.calib.kind, CalibrationKind::Scalar);
    assert_eq!(out.calib.n_trained, n_trained);
    // Direction survives the rescale, and the scale lands near the trained one.
    let cos = cosine(&out.beta[0..H], &f.beta_star[n_trained]);
    assert!(cos > 0.95, "scalar rescale must not bend β̂ (cos={cos:.3})");
    assert!(
        (0.5..2.0).contains(&out.calib.norm_ratio),
        "scalar should put projected β on the trained scale (ratio={})",
        out.calib.norm_ratio
    );
}

// With no trained gene at all there is nothing to calibrate against, and the code
// must say so rather than invent a scale.
#[test]
fn calibration_is_identity_with_no_trained_genes() {
    let n_genes = 8;
    let (f, counts) = flat_setup(n_genes, 53);
    let pb = stacked(&f, &counts);
    let c = cfg(n_genes, vec![false; n_genes], (0..n_genes as u32).collect());

    let out = project_held_out_features(&pb, H, &trained(&[], 0), &c);

    assert_eq!(out.calib.kind, CalibrationKind::Identity);
    assert_eq!(out.gene_ids.len(), n_genes, "every gene is held out");
    assert!(out.calib.mean_cosine.is_nan());
}

// The LRT statistic is what separates a real gene from a degenerate one, and it
// must not be fooled by β̂'s norm — a near-floor gene can be solved to a LARGE β̂
// (the ridge bounds it, the evidence does not), which is exactly why the gate
// tests `D_null − D_fit` rather than `‖β̂‖²`.
#[test]
fn lrt_separates_real_genes_from_flat_ones() {
    let n_genes = 40;
    let (f, mut counts) = flat_setup(n_genes, 91);
    for g in n_genes / 2..n_genes {
        for p in 0..N_PB {
            counts[(g, p)] = 1e-4;
        }
    }
    let pb = stacked(&f, &counts);
    let beta: Vec<f32> = (0..1).flat_map(|g| f.beta_star[g].clone()).collect();
    let mut c = cfg(n_genes, vec![false; n_genes], (0..n_genes as u32).collect());
    c.null_fdr = 0.0; // gate off: inspect the raw statistic
    let out = project_held_out_features(&pb, H, &trained(&beta, 1), &c);

    let real: Vec<f32> = out
        .gene_ids
        .iter()
        .zip(&out.lrt)
        .filter(|(&g, _)| (g as usize) < n_genes / 2)
        .map(|(_, &l)| l)
        .collect();
    let flat: Vec<f32> = out
        .gene_ids
        .iter()
        .zip(&out.lrt)
        .filter(|(&g, _)| (g as usize) >= n_genes / 2)
        .map(|(_, &l)| l)
        .collect();
    let min_real = real.iter().copied().fold(f32::INFINITY, f32::min);
    let max_flat = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_real > max_flat,
        "LRT must separate real from flat genes (min real={min_real:.3e}, max flat={max_flat:.3e})"
    );
    // Raw, unclamped: at the MAP optimum `D_fit <= D_null`, so a converged solve
    // cannot produce a negative LRT. Planted genes are well-conditioned here.
    assert!(
        real.iter().all(|&l| l > 0.0),
        "a converged solve on structured data must beat the intercept-only fit"
    );
}

// With the gate on, genes flat on the pseudobulk floor are zeroed rather than
// handed a fabricated direction; genes with real signal survive.
#[test]
fn null_gate_zeroes_degenerate_genes() {
    let (n_genes, n_trained) = (60, MIN_TRAINED);
    let (f, mut counts) = flat_setup(n_genes, 43);

    // Three kinds of held-out gene, mirroring real data. The `structureless` band
    // matters: the null is ESTIMATED from the tested genes, so a fixture whose only
    // nulls are undetected (and therefore excluded from the fit) leaves the EB call
    // no null to lock onto, and it would conservatively reject the weakest real
    // genes. Real data always carries detected-but-structureless genes; so must this.
    let first_null = n_genes - 40; // 20: β = 0, detected, mild noise → small LRT
    let first_dead = n_genes - 16; // 16: flat on the floor → LRT 0, undetected
    let mut s = 7788u64;
    for g in first_null..first_dead {
        for p in 0..N_PB {
            // rate = exp(b_p) with no θ structure, jittered so the LRT is a small
            // positive draw rather than an exact 0.
            counts[(g, p)] = f.bias[p].exp() * (1.0 + 0.15 * lcg(&mut s));
        }
    }
    for g in first_dead..n_genes {
        for p in 0..N_PB {
            counts[(g, p)] = 1e-4;
        }
    }
    let pb = stacked(&f, &counts);
    let beta: Vec<f32> = (0..n_trained)
        .flat_map(|g| f.beta_star[g].clone())
        .collect();
    let mut c = cfg(n_genes, vec![false; n_genes], (0..n_genes as u32).collect());
    c.null_fdr = 0.05;

    let out = project_held_out_features(&pb, H, &trained(&beta, n_trained), &c);

    let (mut real_live, mut real_n) = (0, 0);
    let (mut null_live, mut null_n) = (0, 0);
    for (i, &g) in out.gene_ids.iter().enumerate() {
        let g = g as usize;
        let b = &out.beta[i * H..(i + 1) * H];
        if g >= first_dead {
            assert!(!out.live[i], "flat gene {g} must be called null");
            assert_eq!(out.n_detected_pb[i], 0, "flat gene {g} is detected nowhere");
            assert!(b.iter().all(|&x| x == 0.0), "null gene {g} must be zeroed");
            assert_eq!(b.iter().map(|x| x * x).sum::<f32>(), 0.0);
        } else if g >= first_null {
            null_n += 1;
            null_live += usize::from(out.live[i]);
            if !out.live[i] {
                assert!(
                    b.iter().all(|&x| x == 0.0),
                    "rejected gene {g} must be zeroed"
                );
            }
        } else {
            real_n += 1;
            real_live += usize::from(out.live[i]);
            if out.live[i] {
                let cos = cosine(b, &f.beta_star[g]);
                assert!(cos > 0.95, "live gene {g} misaligned (cos={cos:.3})");
            }
        }
    }
    assert!(
        real_live * 4 >= real_n * 3,
        "gate should keep most genes with real structure ({real_live}/{real_n})"
    );
    assert!(
        null_live * 4 <= null_n,
        "gate should reject most structureless genes ({null_live}/{null_n})"
    );
}

// Edges from several rows of one modality add on the shared pseudobulk axis, and
// stacking levels concatenates the pb index space.
#[test]
fn stacked_edges_sum_rows_and_concatenate_levels() {
    let l0 = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 10.0, 20.0]);
    let l1 = DMatrix::from_row_slice(2, 3, &[3.0, 4.0, 5.0, 30.0, 40.0, 50.0]);
    let pb = StackedPb {
        theta: vec![0.0; 5 * H],
        bias: vec![0.0; 5],
        counts: vec![&l0, &l1],
        sizes: vec![vec![1.0; 2], vec![1.0; 3]],
        offsets: vec![0, 2],
    };
    assert_eq!(
        pb.edges_for_row(0),
        vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)]
    );
    assert_eq!(
        pb.edges_for_rows(&[0, 1]),
        vec![(0, 11.0), (1, 22.0), (2, 33.0), (3, 44.0), (4, 55.0)]
    );
}

// Rates become counts: an edge carries `rate · size_p`, summed rows included. This
// is what makes the Poisson likelihood (and hence the LRT's χ² calibration) correct.
#[test]
fn exposure_offset_rescales_edges() {
    let l0 = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 10.0, 20.0]);
    let pb = StackedPb {
        theta: vec![0.0; 2 * H],
        bias: vec![0.0; 2],
        counts: vec![&l0],
        sizes: vec![vec![3.0, 5.0]],
        offsets: vec![0],
    };
    assert_eq!(pb.edges_for_row(0), vec![(0, 3.0), (1, 10.0)]);
    assert_eq!(pb.edges_for_rows(&[0, 1]), vec![(0, 33.0), (1, 110.0)]);
}

// Detection compares the SCALED edge against the SCALED column floor, so the call
// must be identical to comparing raw rates against the raw floor — for any exposure.
#[test]
fn detection_is_invariant_to_exposure() {
    // Row 0 sits exactly on the column min (undetected); row 1 is well above it.
    let l0 = DMatrix::from_row_slice(2, 3, &[0.5, 0.5, 0.5, 4.0, 0.5, 9.0]);
    for sizes in [vec![1.0, 1.0, 1.0], vec![7.0, 2.0, 13.0]] {
        let pb = StackedPb {
            theta: vec![0.0; 3 * H],
            bias: vec![0.0; 3],
            counts: vec![&l0],
            sizes: vec![sizes.clone()],
            offsets: vec![0],
        };
        let pbs = PbScalars::new(&pb);
        assert_eq!(
            count_detected(&pb.edges_for_row(0), &pbs.floor_scaled),
            0,
            "floor row detected nowhere (sizes={sizes:?})"
        );
        assert_eq!(
            count_detected(&pb.edges_for_row(1), &pbs.floor_scaled),
            2,
            "row 1 is above the floor in columns 0 and 2 (sizes={sizes:?})"
        );
    }
}
