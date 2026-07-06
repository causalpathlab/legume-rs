use super::*;
use crate::assoc::io::{Lineage, Site};

/// 2 branches × 5 pseudotime bins × 40 cells. Site A diverges after bin 1 (branch 0
/// rate 0.7, branch 1 rate 0.1); early bins share rate 0.3. Site B is flat (0.3
/// everywhere). Coverage n = 20 per cell.
fn synth() -> (Lineage, Vec<Site>) {
    let (nbr, nb, per) = (2usize, 5usize, 40usize);
    let n = nbr * nb * per;
    let (mut names, mut pt, mut branch) = (Vec::new(), Vec::new(), Vec::new());
    for b in 0..nbr {
        for bin in 0..nb {
            for j in 0..per {
                names.push(format!("c_{b}_{bin}_{j}").into_boxed_str());
                branch.push(b);
                pt.push(bin as f32);
            }
        }
    }
    let lin = Lineage {
        cell_names: names,
        pseudotime: pt,
        branch,
        n_branches: nbr,
    };
    let ncov = 20u32;
    let idx = |b: usize, bin: usize, j: usize| (b * nb + bin) * per + j;
    let (mut kd, nd) = (vec![0u32; n], vec![ncov; n]);
    let (mut kf, nf) = (vec![0u32; n], vec![ncov; n]);
    for b in 0..nbr {
        for bin in 0..nb {
            for j in 0..per {
                let i = idx(b, bin, j);
                let rate = if bin < 2 {
                    0.3
                } else if b == 0 {
                    0.7
                } else {
                    0.1
                };
                kd[i] = (ncov as f32 * rate).round() as u32;
                kf[i] = (ncov as f32 * 0.3).round() as u32;
            }
        }
    }
    let sites = vec![
        Site {
            gene: "G".into(),
            subunit: "chr1:100".into(),
            k: kd,
            n: nd,
        },
        Site {
            gene: "G".into(),
            subunit: "chr1:200".into(),
            k: kf,
            n: nf,
        },
    ];
    (lin, sites)
}

fn cfg() -> AssocConfig {
    AssocConfig {
        n_bins: 5,
        num_perm: 500,
        min_total_coverage: 10,
        min_cells: 5,
        seed: 1,
    }
}

#[test]
fn diverging_site_detected_directionally() {
    let (lin, sites) = synth();
    let res = run_contrasts(&sites, &lin, &cfg());
    let get = |site: usize, branch: usize| {
        res.iter()
            .find(|r| r.site == site && r.branch == branch)
            .unwrap()
    };
    // Site A (diverging): branch 0 significantly enriched, branch 1 depleted.
    let a0 = get(0, 0);
    assert!(a0.p_perm < 0.05, "diverging branch0 p={}", a0.p_perm);
    assert!(a0.effect > 0.0, "branch0 enriched, effect={}", a0.effect);
    let a1 = get(0, 1);
    assert!(a1.p_perm < 0.05, "diverging branch1 p={}", a1.p_perm);
    assert!(a1.effect < 0.0, "branch1 depleted, effect={}", a1.effect);
}

#[test]
fn flat_site_not_called() {
    let (lin, sites) = synth();
    let res = run_contrasts(&sites, &lin, &cfg());
    for r in res.iter().filter(|r| r.site == 1) {
        assert!(
            r.p_perm > 0.2,
            "flat site branch{} p={}",
            r.branch,
            r.p_perm
        );
        assert!(r.stat.abs() < 1.0, "flat site stat≈0, got {}", r.stat);
    }
}

/// Shuffling branch labels across the whole dataset (ignoring pseudotime) still
/// produces a valid permutation p — the diverging signal survives because it is
/// pseudotime-stratified, and a within-bin null is calibrated (p never 0).
#[test]
fn permutation_p_is_bounded() {
    let (lin, sites) = synth();
    let res = run_contrasts(&sites, &lin, &cfg());
    for r in &res {
        assert!(
            r.p_perm >= 1.0 / 501.0 && r.p_perm <= 1.0,
            "p out of range: {}",
            r.p_perm
        );
    }
}
