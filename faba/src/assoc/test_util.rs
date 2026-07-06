//! Shared synthetic fixtures for the `assoc` unit tests.

use super::io::{Lineage, Site};

/// Deterministic Binomial counts on a logit-linear rate `logit(p) = a + b·x`, with `x`
/// on an even grid in [0, 1] and coverage `cov` per observation.
pub fn logit_line(a: f64, b: f64, cov: u32, m: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut k = Vec::with_capacity(m);
    let n = vec![cov; m];
    let mut x = Vec::with_capacity(m);
    for i in 0..m {
        let xi = i as f64 / (m as f64 - 1.0);
        let p = 1.0 / (1.0 + (-(a + b * xi)).exp());
        k.push((cov as f64 * p).round() as u32);
        x.push(xi as f32);
    }
    (k, n, x)
}

/// 2 branches × 60 cells, pseudotime rising 0→1 within each branch, coverage 30. Site A
/// rises along branch 0 (`logit p = -2.5 + 5t`) but is flat in branch 1; site B is flat
/// (0.3) everywhere. Shared by the within-branch trend tests.
pub fn two_branch_panel() -> (Lineage, Vec<Site>) {
    let (nbr, per, cov) = (2usize, 60usize, 30f32);
    let n = nbr * per;
    let (mut names, mut pt, mut branch) = (Vec::new(), Vec::new(), Vec::new());
    for b in 0..nbr {
        for j in 0..per {
            names.push(format!("c_{b}_{j}").into_boxed_str());
            branch.push(b);
            pt.push(j as f32 / (per as f32 - 1.0));
        }
    }
    let lin = Lineage {
        cell_names: names,
        pseudotime: pt.clone(),
        branch: branch.clone(),
        n_branches: nbr,
    };
    let (mut ka, mut kb) = (vec![0u32; n], vec![0u32; n]);
    let (na, nb) = (vec![cov as u32; n], vec![cov as u32; n]);
    for i in 0..n {
        let (b, t) = (branch[i], pt[i] as f64);
        let pa = if b == 0 {
            1.0 / (1.0 + (-(-2.5 + 5.0 * t)).exp())
        } else {
            0.3
        };
        ka[i] = (cov as f64 * pa).round() as u32;
        kb[i] = (cov * 0.3).round() as u32;
    }
    let sites = vec![
        Site {
            gene: "G".into(),
            subunit: "chr1:100".into(),
            k: ka,
            n: na,
        },
        Site {
            gene: "G".into(),
            subunit: "chr1:200".into(),
            k: kb,
            n: nb,
        },
    ];
    (lin, sites)
}
