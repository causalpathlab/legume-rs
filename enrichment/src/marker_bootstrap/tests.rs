use super::*;
use crate::es::rank_descending;
use crate::specificity::{compute_specificity, SpecificityMode};

const G: usize = 1000;
const K: usize = 3;
const PROG: usize = 30; // genes [k*PROG, (k+1)*PROG) are cluster k's program
const LIVE: usize = 700; // genes [LIVE, G) are UNDETECTED — 30%, as in real data

/// A gene's baseline abundance: log-uniform over ~3 orders of magnitude, and **independent of
/// which cluster it marks** — as in real data, where how highly a gene is expressed tells you
/// nothing about which cell type it is specific to.
fn base(g: usize) -> f32 {
    let h = (g as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let u = ((h >> 40) as f32) / ((1u64 << 24) as f32); // [0, 1)
    0.05 * 1000f32.powf(u) // 0.05 .. 50
}

/// The cluster profile.
///
/// * genes `[0, K*PROG)` — cluster `g/PROG`'s program: 90% of their mass in that one cluster.
/// * genes `[K*PROG, LIVE)` — live but **non-specific**: mass spread evenly over every cluster.
/// * genes `[LIVE, G)` — **undetected**: zero everywhere, so their specificity is zero in every
///   cluster and they sort to the BOTTOM of every ranking. They can never be enriched, which is
///   exactly what makes a null full of them so easy to beat.
///
/// Abundance is `base(g)` for every live gene regardless of category, so the abundance covariate
/// carries no information about specificity — only about whether the gene is detectable at all.
/// That is the real confound, and it is the one `gene_strata` removes.
fn profile() -> Mat {
    let mut p = Mat::zeros(G, K);
    for g in 0..LIVE {
        let b = base(g);
        let prog = (g < K * PROG).then_some(g / PROG);
        for kk in 0..K {
            p[(g, kk)] = b * match prog {
                Some(pk) if pk == kk => 0.90,
                Some(_) => 0.05,
                None => 1.0 / K as f32,
            };
        }
    }
    p // genes [LIVE, G) stay zero
}

fn ranked(profile_gk: &Mat) -> Vec<Vec<u32>> {
    let spec = compute_specificity(profile_gk, SpecificityMode::Simplex);
    (0..profile_gk.ncols())
        .map(|kk| rank_descending(&(0..G).map(|gi| spec[(gi, kk)]).collect::<Vec<_>>()))
        .collect()
}

fn markers(panels: &[Vec<usize>]) -> Mat {
    let mut m = Mat::zeros(G, panels.len());
    for (cc, panel) in panels.iter().enumerate() {
        for &gi in panel {
            m[(gi, cc)] = 1.0;
        }
    }
    m
}

/// Celltype `c` is marked by 20 of cluster `c`'s program genes.
fn clean_panels() -> Vec<Vec<usize>> {
    (0..K)
        .map(|cc| (cc * PROG..cc * PROG + 20).collect())
        .collect()
}

fn cfg(n_boot: usize) -> EnrichmentBootstrapConfig {
    EnrichmentBootstrapConfig {
        n_boot,
        abstain: Abstain::Support(0.5),
        set_coverage: 0.8,
        max_set_size: 3,
        boot_num_draws: 50,
    }
}

fn run_with(
    panels: &[Vec<usize>],
    cfg: &EnrichmentBootstrapConfig,
    stratified: bool,
) -> Result<ClusterBootstrap> {
    let p = profile();
    let r = ranked(&p);
    let s = if stratified {
        GeneStrata::by_abundance(&p)
    } else {
        GeneStrata::unstratified(G)
    };
    // No permutation pool ⇒ the exact row-randomization p-value path.
    run_cluster_bootstrap(&r, &markers(panels), &s, &[], G, 0.1, 1.0, 42, cfg)
}

fn run(panels: &[Vec<usize>], cfg: &EnrichmentBootstrapConfig) -> Result<ClusterBootstrap> {
    run_with(panels, cfg, true)
}

//////////////////////////////////
// the GOseq / abundance guards //
//////////////////////////////////

#[test]
fn a_live_but_nonspecific_panel_is_called_only_by_the_unstratified_null() {
    // THE REGRESSION TEST for `gene_strata`, stated as the failure it prevents.
    //
    // Give every celltype a panel of LIVE but entirely NON-SPECIFIC genes — genes whose mass is
    // spread evenly over all three clusters. They mark nothing. Nothing should be called.
    //
    // The UNSTRATIFIED null draws 30% undetected genes, which sit pinned at the bottom of every
    // ranking and can never be enriched. So it scores far below any live panel, and a live panel
    // that is *biologically meaningless* clears it anyway — over-calling on abundance alone.
    //
    // The STRATIFIED null draws live genes to null a live panel, and the panel is exposed.
    // Spread each panel through the whole non-specific range rather than taking a contiguous block:
    // those genes all have IDENTICAL specificity, so the stable sort ranks them by gene index, and
    // a contiguous block would sit at one end of that tie and pick up a rank advantage that has
    // nothing to do with the null being tested.
    let panels: Vec<Vec<usize>> = (0..K)
        .map(|cc| (0..20).map(|j| K * PROG + cc + j * 30).collect())
        .collect();

    let uniform = run_with(&panels, &cfg(100), false).expect("runs");
    let strat = run_with(&panels, &cfg(100), true).expect("runs");

    let called = |b: &ClusterBootstrap| {
        b.consensus
            .label
            .iter()
            .filter(|&&t| t != UNASSIGNED)
            .count()
    };
    assert!(
        called(&uniform) > 0,
        "the unstratified null was supposed to be fooled by a meaningless live panel — if it is \
         not, this fixture no longer reproduces the bias and the test below proves nothing"
    );
    assert_eq!(
        called(&strat),
        0,
        "an abundance-matched null must NOT call a panel of non-specific genes; got labels {:?} \
         with support {:?}",
        strat.consensus.label,
        strat.consensus.support
    );
}

///////////////////////////
// behaviour, unchanged  //
///////////////////////////

#[test]
fn clean_panel_recovers_with_high_support() {
    let boot = run(&clean_panels(), &cfg(100)).expect("clean panel annotates");
    for kk in 0..K {
        assert_eq!(
            boot.consensus.label[kk], kk,
            "cluster {kk} should be called celltype {kk}"
        );
        assert!(
            boot.consensus.support[kk] > 0.95,
            "cluster {kk} support was {}",
            boot.consensus.support[kk]
        );
    }
    assert!(boot.usable.iter().all(|&u| u));
}

#[test]
fn scrambled_panel_abstains() {
    // Markers scattered across the whole genome carry no cluster information.
    let panels: Vec<Vec<usize>> = (0..K)
        .map(|cc| (0..20).map(|j| (cc + j * 47) % G).collect())
        .collect();
    let boot = run(&panels, &cfg(100)).expect("runs");
    let called = boot
        .consensus
        .label
        .iter()
        .filter(|&&t| t != UNASSIGNED)
        .count();
    assert_eq!(
        called, 0,
        "a panel with no cluster-specific signal must not be called: {:?} / {:?}",
        boot.consensus.label, boot.consensus.support
    );
}

#[test]
fn overlapping_panels_yield_a_two_element_set() {
    // Celltypes 0 and 1 share 18 of their 20 markers, so which of them wins cluster 0 is close to a
    // coin flip across resamples. The honest answer is the pair, not a shrug.
    let mut panels = clean_panels();
    panels[1] = (0..18).chain(PROG..PROG + 2).collect();
    let boot = run(&panels, &cfg(200)).expect("runs");
    let set = &boot.consensus.label_set[0];
    assert!(
        set.len() == 2 && set.contains(&0) && set.contains(&1),
        "cluster 0's leaders are inseparable, so the set should be {{0,1}}; got {set:?} \
         (support {:.2})",
        boot.consensus.support[0]
    );
}

#[test]
fn singleton_panel_is_not_bootstrappable() {
    // THE TRAP: a 1-marker celltype resamples to itself every draw, so it comes back at support 1.0
    // and looks like the most confident call in the run.
    let mut panels = clean_panels();
    panels[2] = vec![2 * PROG];
    let boot = run(&panels, &cfg(50)).expect("the other two types are still usable");

    assert_eq!(boot.n_live[2], 1);
    assert!(!boot.usable[2], "a 1-marker celltype must not compete");
    assert!(
        !boot.consensus.label.contains(&2),
        "the unusable celltype won a cluster anyway: {:?}",
        boot.consensus.label
    );
    assert_eq!(boot.consensus.label[0], 0);
    assert_eq!(boot.consensus.label[1], 1);
}

#[test]
fn zero_usable_types_is_an_error() {
    // The "magnet" failure: every panel a singleton.
    let panels: Vec<Vec<usize>> = (0..K).map(|cc| vec![cc * PROG]).collect();
    let err = run(&panels, &cfg(20)).expect_err("no celltype can be bootstrapped");
    assert!(
        err.to_string().contains("live markers"),
        "unexpected error: {err}"
    );
}

#[test]
fn deterministic_given_seed() {
    // Guards the keyed-RNG domain separation against rayon's scheduling.
    let a = run(&clean_panels(), &cfg(60)).expect("runs");
    let b = run(&clean_panels(), &cfg(60)).expect("runs");
    assert_eq!(a.consensus.label, b.consensus.label);
    assert_eq!(a.consensus.support, b.consensus.support);
    assert_eq!(a.decision_gap, b.decision_gap);
}

///////////////////////////////////////////
// the size / weight matching guards     //
///////////////////////////////////////////

/// Null moments for a weight multiset carried by `n` of cluster 0's program genes — so every call
/// below is stratum-matched identically and only the size / dispersion under test varies.
fn moments(weights: &[f32], seed: u64) -> (f32, f32) {
    let p = profile();
    let r = ranked(&p);
    let strata = GeneStrata::by_abundance(&p);
    let panel: Vec<(u32, f32)> = weights
        .iter()
        .enumerate()
        .map(|(j, &w)| (j as u32, w))
        .collect();
    let prof = strata.profile_of(&panel);
    let mut hit = vec![0f32; G];
    let mut scratch = strata.scratch();
    let mut drawn = Vec::new();
    let mut rng = keyed_rng(seed, 0, 0);
    let (mean, sd, _) = null_moments(
        &r,
        &strata,
        &prof,
        weights.len(),
        &[0f32; K],
        600,
        &mut hit,
        &mut scratch,
        &mut drawn,
        &mut rng,
    );
    (mean[0], sd[0])
}

#[test]
fn a_celltype_with_too_few_markers_is_dropped_not_merely_weak() {
    // A panel that barely matched the data is not a weak competitor. An enrichment walk over one or
    // two genes is noise, and the winner's curse hands the cluster to whichever noisy panel spiked.
    // `MIN_LIVE_MARKERS` is the floor the bootstrap itself cannot go below (you cannot resample a
    // single point); `AnnotateConfig::min_markers` is the user-facing bar above it.
    let mut panels = clean_panels();
    panels[2] = vec![2 * PROG, 2 * PROG + 1]; // exactly 2 — at the floor, but thin
    let boot = run(&panels, &cfg(60)).expect("the other two types are still usable");

    assert_eq!(boot.n_live[2], 2);
    assert!(
        boot.usable[2],
        "2 markers clears MIN_LIVE_MARKERS; the *drop* is the caller's --min-markers, not this floor"
    );
    // Below the floor it cannot compete at all, whatever the caller asked for.
    let mut panels = clean_panels();
    panels[2] = vec![2 * PROG];
    let boot = run(&panels, &cfg(60)).expect("runs");
    assert!(!boot.usable[2]);
    assert!(!boot.consensus.label.contains(&2));
}

#[test]
fn the_null_tracks_the_draws_size_and_weight_dispersion() {
    // A resampled panel differs from its nominal self in TWO ways, and the null must follow both or
    // it standardizes the score against the wrong thing — which inflates small panels and
    // manufactures the very winner's curse the bootstrap exists to remove.
    //
    // Size: fewer genes => a more variable KS walk => a LARGER null SD. A null cached at the
    // panel's nominal size would miss this.
    let (_, sd_8) = moments(&[1.0; 8], 1);
    let (_, sd_40) = moments(&[1.0; 40], 1);
    assert!(
        sd_8 > sd_40,
        "an 8-gene null must be noisier than a 40-gene one: sd_8={sd_8:.4}, sd_40={sd_40:.4}"
    );

    // Weights: drawing with replacement produces multiplicities, which concentrate the walk on
    // fewer genes and make it MORE variable. A null scattering binary weights would miss this.
    // Same gene count, same total mass; only the dispersion differs.
    let flat = vec![2.0f32; 40];
    let mut dispersed = vec![1.0f32; 40];
    for w in dispersed.iter_mut().take(10) {
        *w = 5.0; // 10 genes drawn 5x, 30 once — the same 80 units of mass
    }
    let (_, sd_flat) = moments(&flat, 2);
    let (_, sd_dispersed) = moments(&dispersed, 2);
    assert!(
        sd_dispersed > sd_flat,
        "a dispersed weight multiset must give a noisier null than a flat one of the same size: \
         flat={sd_flat:.4}, dispersed={sd_dispersed:.4}"
    );
}
