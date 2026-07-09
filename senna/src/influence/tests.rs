use super::*;

/// `β_k = softmax_g(α_k · ρᵀ)` — the decoder's dictionary as a function of `α`.
fn beta_from(rho: &Mat, alpha: &Mat) -> Mat {
    let s = rho * alpha.transpose(); // [D, K]
    let mut b = s.clone();
    for k in 0..s.ncols() {
        let mut mx = f32::NEG_INFINITY;
        for g in 0..s.nrows() {
            mx = mx.max(s[(g, k)]);
        }
        let mut sum = 0f32;
        for g in 0..s.nrows() {
            let e = (s[(g, k)] - mx).exp();
            b[(g, k)] = e;
            sum += e;
        }
        for g in 0..s.nrows() {
            b[(g, k)] /= sum;
        }
    }
    b
}

/// `Σ_j Σ_g p_gj · log recon_gj` — the objective the gradient differentiates.
fn loglik(beta: &Mat, theta_kn: &Mat, cells: &[Vec<(usize, f32)>]) -> f64 {
    let recon = beta * theta_kn;
    let mut ll = 0f64;
    for (j, nz) in cells.iter().enumerate() {
        let total: f64 = nz.iter().map(|&(_, x)| f64::from(x)).sum();
        if total <= 0.0 {
            continue;
        }
        for &(g, x) in nz {
            let r = f64::from(recon[(g, j)]).max(1e-12);
            ll += (f64::from(x) / total) * r.ln();
        }
    }
    ll
}

#[test]
fn analytic_gradient_matches_finite_difference() {
    let (d, k, h) = (6usize, 2usize, 3usize);

    let rho = Mat::from_fn(d, h, |g, hh| ((((g * 7 + hh * 3) % 5) as f32) * 0.25) - 0.5);
    let alpha = Mat::from_fn(k, h, |kk, hh| {
        ((((kk * 3 + hh * 2) % 4) as f32) * 0.3) - 0.4
    });

    let theta_kn = {
        let mut t = Mat::from_fn(k, 3, |kk, j| 0.2 + 0.3 * (((kk + j) % 3) as f32));
        for j in 0..t.ncols() {
            let s: f32 = t.column(j).sum();
            for kk in 0..k {
                t[(kk, j)] /= s;
            }
        }
        t
    };

    let cells: Vec<Vec<(usize, f32)>> = vec![
        vec![(0, 3.0), (2, 1.0), (5, 2.0)],
        vec![(1, 4.0), (3, 2.0)],
        vec![(0, 1.0), (1, 1.0), (4, 5.0)],
    ];

    let beta = beta_from(&rho, &alpha);
    let m_kh = beta.transpose() * &rho;
    let recon = &beta * &theta_kn;
    let ctx = BlockCtx {
        theta_kn: &theta_kn,
        exp_beta: &beta,
        rho_dh: &rho,
        m_kh: &m_kh,
        recon_dn: &recon,
    };
    let (analytic, fisher, n_used) = block_gradient_sums(&cells, &ctx);
    assert_eq!(n_used, 3);
    assert_eq!(analytic.len(), k * h);
    assert!(fisher.iter().all(|&f| f >= 0.0), "Fisher diag must be >= 0");

    let eps = 1e-3f32;
    for kk in 0..k {
        for hh in 0..h {
            let mut ap = alpha.clone();
            ap[(kk, hh)] += eps;
            let mut am = alpha.clone();
            am[(kk, hh)] -= eps;

            let lp = loglik(&beta_from(&rho, &ap), &theta_kn, &cells);
            let lm = loglik(&beta_from(&rho, &am), &theta_kn, &cells);
            let numeric = (lp - lm) / (2.0 * f64::from(eps));
            let a = analytic[kk * h + hh];

            let denom = a.abs().max(numeric.abs());
            let tol = 1e-4 + 1e-2 * denom;
            assert!(
                (a - numeric).abs() < tol,
                "gradient mismatch at (k={kk}, h={hh}): analytic={a:.6e} numeric={numeric:.6e}"
            );
        }
    }
}

/// With an unregularized step the axes collapse (`τ_old = −τ_new`); the
/// EWC-regularized step must keep them distinct.
#[test]
fn regularized_step_keeps_axes_independent() {
    let p = 4;
    let new = GradStats {
        n_cells: 10,
        g_mean: vec![0.5, 0.5, 0.5, 0.5],
        // new-batch curvature uniform
        fisher: vec![1.0; p],
    };
    // old model: first two directions well-known (high Fisher), last two unexplored
    let old = GradStats {
        n_cells: 10,
        g_mean: vec![0.0; p],
        fisher: vec![100.0, 100.0, 1e-6, 1e-6],
    };

    let cf = counterfactual(&new, &old, 1.0, 2);
    assert!(cf.tau_new > 0.0, "benefit must be positive: {}", cf.tau_new);
    assert!(cf.tau_old <= 0.0, "old-data change must not be a gain here");
    // Not the degenerate collapse.
    assert!(
        (cf.tau_old + cf.tau_new).abs() > 1e-6,
        "axes collapsed: tau_new={} tau_old={}",
        cf.tau_new,
        cf.tau_old
    );
    // Movement concentrates in the unexplored topic (index 1), not the known one.
    assert!(
        cf.delta_norm_per_topic[1] > 10.0 * cf.delta_norm_per_topic[0],
        "expected movement in the low-Fisher directions: {:?}",
        cf.delta_norm_per_topic
    );
}
