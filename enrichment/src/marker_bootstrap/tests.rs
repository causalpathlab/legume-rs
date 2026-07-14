use super::*;
use crate::es::rank_descending;
use crate::specificity::{compute_specificity, SpecificityMode};

const G: usize = 200;
const K: usize = 3;
const BLOCK: usize = 20; // genes [k*20, k*20+20) are cluster k's program

/// Cluster `k` is elevated 10x on genes `[k*BLOCK, (k+1)*BLOCK)`.
fn profile() -> Mat {
    let mut p = Mat::from_element(G, K, 1.0);
    for kk in 0..K {
        for gi in kk * BLOCK..(kk + 1) * BLOCK {
            p[(gi, kk)] = 10.0;
        }
    }
    p
}

fn ranked(profile_gk: &Mat) -> Vec<Vec<u32>> {
    let spec = compute_specificity(profile_gk, SpecificityMode::Simplex);
    (0..profile_gk.ncols())
        .map(|kk| {
            let scores: Vec<f32> = (0..G).map(|gi| spec[(gi, kk)]).collect();
            rank_descending(&scores)
        })
        .collect()
}

/// A marker matrix from an explicit gene list per celltype, all at weight 1.
fn markers(panels: &[Vec<usize>]) -> Mat {
    let mut m = Mat::zeros(G, panels.len());
    for (cc, panel) in panels.iter().enumerate() {
        for &gi in panel {
            m[(gi, cc)] = 1.0;
        }
    }
    m
}

/// The clean panel: celltype `c` is marked by exactly cluster `c`'s block.
fn clean_panels() -> Vec<Vec<usize>> {
    (0..K)
        .map(|cc| (cc * BLOCK..(cc + 1) * BLOCK).collect())
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

fn run(panels: &[Vec<usize>], cfg: &EnrichmentBootstrapConfig) -> Result<ClusterBootstrap> {
    let p = profile();
    let r = ranked(&p);
    // No permutation pool ⇒ the exact row-randomization p-value path.
    run_cluster_bootstrap(&r, &markers(panels), &[], G, 0.1, 1.0, 42, cfg)
}

#[test]
fn clean_panel_recovers_with_high_support() {
    let boot = run(&clean_panels(), &cfg(100)).expect("clean panel annotates");
    for kk in 0..K {
        assert_eq!(
            boot.consensus.label[kk], kk,
            "cluster {kk} should be called celltype {kk}, got {:?}",
            boot.consensus.label[kk]
        );
        assert!(
            boot.consensus.support[kk] > 0.95,
            "cluster {kk} support was {}",
            boot.consensus.support[kk]
        );
    }
    assert!(boot.usable.iter().all(|&u| u));
    assert_eq!(boot.n_live, vec![BLOCK; K]);
}

#[test]
fn scrambled_panel_abstains() {
    // Markers drawn from all over the genome carry no cluster information. Spreading each panel
    // uniformly (stride 7 from a per-celltype offset) means no celltype sits in any cluster's
    // program, so nothing should be confidently called.
    let panels: Vec<Vec<usize>> = (0..K)
        .map(|cc| (0..BLOCK).map(|j| (cc + j * 7) % G).collect())
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
        "a panel with no cluster-specific signal must not be called: labels {:?}, support {:?}",
        boot.consensus.label, boot.consensus.support
    );
}

#[test]
fn overlapping_panels_yield_a_two_element_set() {
    // Celltypes 0 and 1 share 18 of their 20 markers, so which of them wins cluster 0 is a coin
    // flip across resamples. The honest answer is the pair, not a shrug.
    let mut panels = clean_panels();
    panels[1] = panels[0][..18]
        .iter()
        .copied()
        .chain([BLOCK, BLOCK + 1]) // 2 of its own
        .collect();
    let boot = run(&panels, &cfg(200)).expect("runs");

    let set = &boot.consensus.label_set[0];
    assert!(
        set.len() == 2 && set.contains(&0) && set.contains(&1),
        "cluster 0's leaders are inseparable, so the set should be {{0,1}}; got {set:?} \
         (support {:.2}, post {:?})",
        boot.consensus.support[0],
        &boot.consensus.post[0..4.min(boot.consensus.post.len())]
    );
}

#[test]
fn singleton_panel_is_not_bootstrappable() {
    // THE TRAP: a 1-marker celltype resamples to itself every single draw. Without the
    // MIN_LIVE_MARKERS guard it comes back at support 1.0 and looks like the most confident call
    // in the run.
    let mut panels = clean_panels();
    panels[2] = vec![2 * BLOCK]; // exactly one marker
    let boot = run(&panels, &cfg(50)).expect("the other two types are still usable");

    assert_eq!(boot.n_live[2], 1);
    assert!(!boot.usable[2], "a 1-marker celltype must not compete");
    assert!(
        !boot.consensus.label.contains(&2),
        "the unusable celltype won a cluster anyway: {:?}",
        boot.consensus.label
    );
    // …and the types that ARE bootstrappable still work.
    assert_eq!(boot.consensus.label[0], 0);
    assert_eq!(boot.consensus.label[1], 1);
}

#[test]
fn zero_usable_types_is_an_error() {
    // The "magnet" failure: every panel a singleton. Better to refuse than to hand back a
    // confident-looking annotation in which the first celltype swallows the dataset.
    let panels: Vec<Vec<usize>> = (0..K).map(|cc| vec![cc * BLOCK]).collect();
    let err = run(&panels, &cfg(20)).expect_err("no celltype can be bootstrapped");
    assert!(
        err.to_string().contains("live markers"),
        "unexpected error: {err}"
    );
}

#[test]
fn deterministic_given_seed() {
    // Guards the keyed-RNG domain separation against rayon's scheduling: a replicate's draw must
    // depend on WHICH replicate it is, never on which thread got to it first.
    let a = run(&clean_panels(), &cfg(60)).expect("runs");
    let b = run(&clean_panels(), &cfg(60)).expect("runs");
    assert_eq!(a.consensus.label, b.consensus.label);
    assert_eq!(a.consensus.support, b.consensus.support);
    assert_eq!(a.decision_gap, b.decision_gap);
}

#[test]
fn resample_shrinks_the_effective_size() {
    // The assumption the moment-matching rests on: a with-replacement draw of m items covers only
    // ~0.632*m distinct ones. If this drifts, the null is standardizing against the wrong size.
    let m = 20usize;
    let mut total = 0f64;
    let reps = 400;
    for b in 0..reps {
        let mut rng = keyed_rng(7, b, 0);
        let mut seen = vec![false; m];
        for _ in 0..m {
            seen[rng.random_range(0..m)] = true;
        }
        total += seen.iter().filter(|&&s| s).count() as f64;
    }
    let frac = total / (reps as f64 * m as f64);
    assert!(
        (frac - 0.632).abs() < 0.03,
        "distinct fraction was {frac:.3}, expected ~0.632"
    );
}

/// Null moments for a given weight multiset, on this fixture's ranking.
fn moments(weights: &[f32], seed: u64) -> (f32, f32) {
    let r = ranked(&profile());
    let mut hit = vec![0f32; G];
    let mut pool: Vec<u32> = (0..G as u32).collect();
    let mut rng = keyed_rng(seed, 0, 0);
    let es_obs = vec![0f32; K];
    let (mean, sd, _) = null_moments(&r, weights, &es_obs, G, 600, &mut hit, &mut pool, &mut rng);
    (mean[0], sd[0])
}

#[test]
fn the_null_tracks_the_draws_effective_size() {
    // Half of §2d. A smaller gene set gives a more variable KS walk, so its null SD must be
    // LARGER. A null cached at the panel's nominal size |M_c| — rather than the draw's actual
    // ~0.632*|M_c| distinct genes — would miss this and under-state the noise.
    let (_, sd_8) = moments(&[1.0; 8], 1);
    let (_, sd_40) = moments(&[1.0; 40], 1);
    assert!(
        sd_8 > sd_40 * 1.5,
        "an 8-gene null must be markedly noisier than a 40-gene one: sd_8={sd_8:.4}, \
         sd_40={sd_40:.4}"
    );
}

#[test]
fn the_null_tracks_the_draws_weight_dispersion() {
    // The other half of §2d. Resampling with replacement produces multiplicities (1, 2, 3…), which
    // concentrate the walk on fewer genes and make it MORE variable. A null that scatters binary
    // weights — as the observed row-randomization null does — would miss this entirely.
    //
    // Same gene count, same total mass; only the dispersion differs.
    let uniform = vec![2.0f32; 40];
    let mut dispersed = vec![1.0f32; 40];
    for w in dispersed.iter_mut().take(10) {
        *w = 5.0; // 10 genes drawn 5x, 30 drawn once — same 80 units of mass
    }
    let (_, sd_uniform) = moments(&uniform, 2);
    let (_, sd_dispersed) = moments(&dispersed, 2);
    assert!(
        sd_dispersed > sd_uniform,
        "a dispersed weight multiset must give a noisier null than a flat one of the same size: \
         flat={sd_uniform:.4}, dispersed={sd_dispersed:.4}"
    );
}

#[test]
fn a_mismatched_null_would_inflate_small_panels_the_most() {
    // THE REGRESSION TEST for §2d, stated as the failure it prevents.
    //
    // Take what a bootstrap draw actually looks like — ~0.632*m distinct genes carrying a dispersed
    // weight multiset — and standardize it two ways:
    //
    //   matched    : the null this module computes (the draw's own size AND weights)
    //   mismatched : the shortcut (binary weights at the panel's nominal size m)
    //
    // The mismatched null under-states the draw's noise, so it INFLATES es_std. The inflation is
    // worse for small panels, because the multiset's dispersion scales as 1/sqrt(m) — and that
    // differential inflation IS the winner's curse: a small panel would out-score a large one on
    // noise alone. Assert the shortcut is biased, and that the bias is size-dependent.
    let inflation = |m: usize| -> f32 {
        // A representative draw of a size-m panel: n_hit distinct genes, dispersed weights.
        let n_hit = ((m as f32) * 0.632).round() as usize;
        let mut drawn = vec![1.0f32; n_hit];
        for w in drawn.iter_mut().take(n_hit / 3) {
            *w = 2.0; // some genes came up twice
        }
        let (_, sd_matched) = moments(&drawn, 3);
        let (_, sd_mismatched) = moments(&vec![1.0; m], 3); // the shortcut
                                                            // es_std scales as 1/sd, so the shortcut inflates the score by this factor.
        sd_matched / sd_mismatched
    };

    let small = inflation(8);
    let large = inflation(40);
    assert!(
        small > 1.0 && large > 1.0,
        "the shortcut should inflate both (it under-states the draw's noise): \
         small={small:.3}, large={large:.3}"
    );
    assert!(
        small > large,
        "the winner's curse: the shortcut must inflate the SMALL panel more than the large one \
         (small={small:.3}x, large={large:.3}x). If this ever ceases to hold, the matched null in \
         `null_moments` is no longer buying anything and the reasoning in the module doc is wrong."
    );
}
