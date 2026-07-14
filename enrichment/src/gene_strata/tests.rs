use super::*;
use crate::consensus::keyed_rng;

/// A covariate with a wide, heavy-tailed spread — like real expression.
fn cov(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 / 10.0).exp()).collect()
}

#[test]
fn bins_are_equal_count_not_equal_width() {
    // Expression is heavy-tailed. Equal-WIDTH bins would put almost everything in the bottom bin
    // and leave the top bin with nothing to swap a highly-expressed marker with.
    let s = GeneStrata::by_covariate(&cov(1000));
    assert_eq!(s.n_strata(), N_STRATA);
    for bin in &s.members {
        assert_eq!(bin.len(), 100, "bins should hold equal COUNTS");
    }
}

#[test]
fn the_bin_count_is_a_ceiling_so_strata_always_have_partners() {
    // THE SILENT FAILURE this guards: with as many bins as genes, every gene is alone in its bin,
    // no swap is possible, the "random" panel is the real one, and every p-value is 1 — while the
    // null still runs and still produces numbers.
    let s = GeneStrata::by_covariate(&cov(35)); // 35 / MIN_PER_STRATUM = 3 bins, not 10
    assert_eq!(s.n_strata(), 3);
    for bin in &s.members {
        assert!(
            bin.len() >= MIN_PER_STRATUM,
            "a stratum with < {MIN_PER_STRATUM} genes cannot be shuffled"
        );
    }
    // In the limit, one bin — an unstratified draw — rather than a null that cannot move.
    assert_eq!(GeneStrata::by_covariate(&cov(5)).n_strata(), 1);
}

#[test]
fn a_null_draw_reproduces_the_panels_abundance_profile() {
    // THE WHOLE POINT. A panel of only high-abundance genes must be nulled against high-abundance
    // genes, or it beats the null on abundance alone and no biology is tested.
    let c = cov(1000);
    let s = GeneStrata::by_covariate(&c);

    // A panel drawn entirely from the top decile.
    let panel: Vec<(u32, f32)> = (900..920).map(|g| (g as u32, 1.0)).collect();
    let profile = s.profile_of(&panel);
    assert_eq!(profile[N_STRATA - 1].len(), 20, "panel is all top-decile");
    assert!(profile[..N_STRATA - 1].iter().all(Vec::is_empty));

    let mut scratch = s.scratch();
    let mut out = Vec::new();
    let mut rng = keyed_rng(1, 0, 0);
    s.draw_matched(&profile, &mut scratch, &mut out, &mut rng);

    assert_eq!(out.len(), 20);
    // Every drawn gene must come from the same (top) decile — never from the dead bottom.
    for &(g, _) in &out {
        assert!(
            c[g as usize] >= c[900],
            "gene {g} came from outside the panel's stratum"
        );
    }
    // …and it must be a genuinely random set, not the panel itself.
    let same: usize = out
        .iter()
        .filter(|(g, _)| (900..920).contains(&(*g as usize)))
        .count();
    assert!(same < 20, "the null draw was the panel itself");
}

#[test]
fn weights_stay_with_their_own_stratum() {
    // IDF weight and expression are not independent. Pooling the weights and re-dealing them would
    // hand a highly-expressed null gene a low-expressed marker's weight — matched on neither.
    let c = cov(1000);
    let s = GeneStrata::by_covariate(&c);
    // Bottom-decile genes carry weight 9.0; top-decile carry 1.0.
    let mut panel: Vec<(u32, f32)> = (0..10).map(|g| (g as u32, 9.0)).collect();
    panel.extend((990..1000).map(|g| (g as u32, 1.0)));
    let profile = s.profile_of(&panel);

    let mut scratch = s.scratch();
    let mut out = Vec::new();
    let mut rng = keyed_rng(2, 0, 0);
    s.draw_matched(&profile, &mut scratch, &mut out, &mut rng);

    assert_eq!(out.len(), 20);
    for &(g, w) in &out {
        let bin = s.stratum[g as usize];
        let expect = if bin == 0 { 9.0 } else { 1.0 };
        assert_eq!(w, expect, "gene {g} in stratum {bin} got weight {w}");
    }
}

#[test]
fn abundance_binning_separates_dead_genes_from_live_ones() {
    // The dominant nuisance on real data: ~30% of the genome is undetected, sorts to the bottom of
    // every ranking, and can never be enriched. Those genes must land in their own low strata so a
    // real panel is never nulled against them.
    let mut c = vec![0.0f32; 300]; // dead
    c.extend((0..700).map(|i| 1.0 + i as f32)); // live
    let s = GeneStrata::by_covariate(&c);

    // A live panel's null draw must never contain a dead gene.
    let panel: Vec<(u32, f32)> = (900..920).map(|g| (g as u32, 1.0)).collect();
    let profile = s.profile_of(&panel);
    let mut scratch = s.scratch();
    let mut out = Vec::new();
    let mut rng = keyed_rng(3, 0, 0);
    for _ in 0..50 {
        s.draw_matched(&profile, &mut scratch, &mut out, &mut rng);
        assert!(
            out.iter().all(|&(g, _)| c[g as usize] > 0.0),
            "a dead gene entered the null of a live panel"
        );
    }
}
