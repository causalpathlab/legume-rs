//! Tests for the CP hyperedge fit: what enters training, what is frozen,
//! what the ridge is allowed to touch, and whether the geometry is learned.

use super::*;
use crate::eqtl::evidence::classify_states;
use crate::eqtl::select::{Pair, PairObs, Selection};

fn device() -> Device {
    Device::Cpu
}

fn config(embedding_dim: usize, num_iterations: usize) -> EqtlModelConfig {
    EqtlModelConfig {
        embedding_dim,
        num_negatives: 5,
        num_iterations,
        batch_size: 64,
        learning_rate: 0.05,
        ridge: 1e-3,
        holdout_frac: 0.2,
        shuffle_control: false,
        seed: 42,
    }
}

/// One pair as `(gene, variant, [(celltype, beta, se)])`.
type PairSpec = (u32, u32, Vec<(u32, f32, f32)>);
/// A synthetic world: the selection plus the variant, gene, and context names.
type World = (Selection, Vec<Box<str>>, Vec<Box<str>>, Vec<Box<str>>);

/// Assemble a selection from pair specifications.
fn selection(rows: Vec<PairSpec>) -> Selection {
    let pairs: Vec<Pair> = rows
        .into_iter()
        .map(|(gene, variant, obs)| Pair {
            gene,
            variant,
            obs: obs
                .into_iter()
                .map(|(celltype, beta, se)| PairObs { celltype, beta, se })
                .collect(),
        })
        .collect();
    Selection {
        n_rows: pairs.iter().map(|p| p.obs.len()).sum(),
        n_selected_variants: pairs.iter().map(|p| p.variant).max().unwrap_or(0) as usize + 1,
        n_pairs_dropped: 0,
        pairs,
    }
}

fn names(prefix: &str, n: usize) -> Vec<Box<str>> {
    (0..n).map(|i| Box::from(format!("{prefix}{i}"))).collect()
}

/// Eight contexts in two label groups, four each.
fn group_names() -> Vec<Box<str>> {
    (0..8)
        .map(|k| {
            if k < 4 {
                Box::from(format!("L1-{k}"))
            } else {
                Box::from(format!("L2-{k}"))
            }
        })
        .collect()
}

/// A planted world: every variant is an edge in one group's four
/// contexts and a certified absence in the other four, plus a decoy gene it
/// never affects (the gene-swap negative pool).
fn planted_world(n_variants: u32) -> World {
    let n_genes = n_variants; // one real gene per variant, plus a decoy
    let mut rows = Vec::new();
    for j in 0..n_variants {
        let excitatory = j % 2 == 0;
        let obs: Vec<(u32, f32, f32)> = (0..8)
            .map(|k| {
                let hot = if excitatory { k < 4 } else { k >= 4 };
                (k, if hot { 0.5 } else { 0.0 }, 0.05)
            })
            .collect();
        rows.push((j, j, obs));
        // The decoy gene: this variant does nothing there, precisely.
        let decoy = (j + 1) % n_genes;
        rows.push((
            decoy,
            j,
            (0..8).map(|k| (k, 0.0f32, 0.05f32)).collect::<Vec<_>>(),
        ));
    }
    (
        selection(rows),
        names("V", n_variants as usize),
        names("G", n_genes as usize),
        group_names(),
    )
}

#[test]
fn the_ubiquitous_gate_stays_exactly_one() {
    let (sel, variants, genes, celltypes) = planted_world(12);
    let evidence = classify_states(&sel, &celltypes, 4.0).unwrap();
    let fit = train(&evidence, &variants, &genes, &config(4, 300), &device()).unwrap();

    let k = fit.ubiquitous.expect("the ubiquitous context is fitted") as usize;
    for h in 0..fit.c.ncols() {
        assert_eq!(
            fit.c[(k, h)],
            1.0,
            "the ubiquitous gate moved at program {h}"
        );
    }
    // Real gates must have moved off their initialization band.
    let moved = (0..fit.c.nrows())
        .filter(|&i| i != k)
        .any(|i| (0..fit.c.ncols()).any(|h| !(0.3..=0.9).contains(&fit.c[(i, h)])));
    assert!(moved, "no real gate was trained at all");
}

/// The ridge must reach only the rows a batch touched. A variant that never
/// appears in a positive is never in a batch — negatives corrupt the gene
/// or context slot, never the variant — so a dense penalty would be the
/// only thing acting on it, and it would collapse toward zero.
#[test]
fn an_untrained_variant_is_not_crushed_by_the_ridge() {
    let (mut sel, mut variants, genes, celltypes) = planted_world(12);
    // A quiet variant: everywhere below the detection threshold, yet
    // everywhere powered against its own reference effect, so every cell is
    // a certified absence and the variant owns no edge — and therefore
    // never appears in a batch, since negatives corrupt the gene or the
    // context slot and never the variant.
    let quiet = variants.len() as u32;
    sel.pairs.push(Pair {
        gene: 0,
        variant: quiet,
        obs: (0..8)
            .map(|k| PairObs {
                celltype: k,
                beta: match k {
                    0 => 0.19, // z = 3.8, just under the threshold
                    1 => 0.18, // the reference effect: 2.8 * 0.05 = 0.14 < 0.18
                    _ => 0.0,
                },
                se: 0.05,
            })
            .collect(),
    });
    variants.push(Box::from("QUIET"));

    let evidence = classify_states(&sel, &celltypes, 4.0).unwrap();
    let mut cfg = config(4, 400);
    cfg.ridge = 10.0; // ruinous if it were applied to every row every step
    let fit = train(&evidence, &variants, &genes, &cfg, &device()).unwrap();

    let slot = fit.variants.slot[quiet as usize].expect("the quiet variant is fitted") as usize;
    let norm: f32 = (0..fit.u.ncols())
        .map(|h| fit.u[(slot, h)] * fit.u[(slot, h)])
        .sum::<f32>()
        .sqrt();
    assert!(
        norm > 0.05,
        "an untouched variant was regularized to {norm}, so the ridge is dense"
    );
}

/// An entity seen only in unpowered cells gets no embedding at all: unknown
/// cells are sampled in neither class, so nothing about it is ever learned.
#[test]
fn an_entity_seen_only_in_unknown_cells_is_never_fitted() {
    let (mut sel, mut variants, genes, celltypes) = planted_world(12);
    // Every cell is imprecise, so nothing is an edge and nothing is
    // certified: all eight cells land in `unknown`.
    let vague = variants.len() as u32;
    sel.pairs.push(Pair {
        gene: 0,
        variant: vague,
        obs: (0..8)
            .map(|k| PairObs {
                celltype: k,
                beta: 0.05,
                se: 10.0,
            })
            .collect(),
    });
    variants.push(Box::from("VAGUE"));

    let evidence = classify_states(&sel, &celltypes, 4.0).unwrap();
    let fit = train(&evidence, &variants, &genes, &config(4, 200), &device()).unwrap();

    assert!(
        fit.variants.slot[vague as usize].is_none(),
        "a variant known only through unpowered cells was still embedded"
    );
    assert!(fit.variants.names.iter().all(|n| n.as_ref() != "VAGUE"));
}

#[test]
fn held_out_edges_are_kept_out_of_training() {
    let (sel, variants, genes, celltypes) = planted_world(20);
    let evidence = classify_states(&sel, &celltypes, 4.0).unwrap();
    let mut cfg = config(4, 200);
    cfg.holdout_frac = 0.5;
    let fit = train(&evidence, &variants, &genes, &cfg, &device()).unwrap();

    let m = &fit.metrics;
    let expected = (0.5 * m.n_edges_real as f64).round() as usize;
    assert_eq!(m.n_heldout, expected);
    // Every remaining edge — real or ubiquitous — is either a positive or
    // was set aside for having no certified-absent negative. Held-out edges
    // are in neither group.
    assert_eq!(
        m.n_positives + m.n_positives_without_negatives,
        m.n_edges_real - m.n_heldout + m.n_edges_ubiquitous
    );
}

/// The planted structure has to come back out: held-out edges must score
/// above certified absences, and the gates of one group must resemble
/// each other more than the other group's.
#[test]
fn the_planted_specificity_is_recovered() {
    let (sel, variants, genes, celltypes) = planted_world(30);
    let evidence = classify_states(&sel, &celltypes, 4.0).unwrap();
    let fit = train(&evidence, &variants, &genes, &config(8, 1500), &device()).unwrap();

    let m = &fit.metrics;
    assert!(
        m.auc_heldout > 0.8,
        "held-out AUC {:.3} is too low to have learned the planted geometry",
        m.auc_heldout
    );
    let within = m.within_group_cosine.expect("grouped labels");
    let between = m.between_group_cosine.expect("grouped labels");
    assert!(
        within > between,
        "within-group cosine {within:.3} did not beat between {between:.3}"
    );
}

/// The control: with the edge/absent labels permuted there is no planted
/// structure left, so the held-out separation has to fall away. The
/// comparison is paired against the same world, because a synthetic this
/// small still leaks a little through the anchor — a pair keeps several
/// shuffled edges even after one is hidden — and the real collapse to
/// chance only shows on data with many contexts per pair.
#[test]
fn shuffling_the_labels_destroys_the_held_out_separation() {
    let (sel, variants, genes, celltypes) = planted_world(30);
    let evidence = classify_states(&sel, &celltypes, 4.0).unwrap();

    let real = train(&evidence, &variants, &genes, &config(8, 1500), &device()).unwrap();
    let mut cfg = config(8, 1500);
    cfg.shuffle_control = true;
    let shuffled = train(&evidence, &variants, &genes, &cfg, &device()).unwrap();

    let (a, b) = (real.metrics.auc_heldout, shuffled.metrics.auc_heldout);
    assert!(
        b < a - 0.1,
        "shuffled held-out AUC {b:.3} did not fall away from the real fit's {a:.3}"
    );
    // The gate also stops being low-rank once there is nothing to gate.
    assert!(
        shuffled.metrics.effective_rank > real.metrics.effective_rank,
        "shuffled effective rank {:.2} did not exceed the real fit's {:.2}",
        shuffled.metrics.effective_rank,
        real.metrics.effective_rank,
    );
}

// ── Diagnostic helpers ──────────────────────────────────────────────────

#[test]
fn auc_and_auprc_match_their_textbook_values() {
    // Perfect separation.
    assert!((auc(&[3.0, 4.0], &[1.0, 2.0]) - 1.0).abs() < 1e-12);
    assert!((auc(&[1.0, 2.0], &[3.0, 4.0]) - 0.0).abs() < 1e-12);
    // All ties average to one half.
    assert!((auc(&[1.0, 1.0], &[1.0, 1.0]) - 0.5).abs() < 1e-12);
    // One positive ranked first out of four: precision 1 at recall 1.
    assert!((auprc(&[5.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
    // One positive ranked last: precision 1/4.
    assert!((auprc(&[0.5], &[1.0, 2.0, 3.0]) - 0.25).abs() < 1e-12);
}

#[test]
fn effective_rank_counts_the_spread_of_the_gate() {
    // One direction of variation: rank one.
    let one = DMatrix::from_row_slice(4, 2, &[0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    assert!((effective_rank(&one) - 1.0).abs() < 1e-6);
    // Two equal directions: rank two.
    let two = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0]);
    assert!((effective_rank(&two) - 2.0).abs() < 1e-6);
}

#[test]
fn group_cosines_skip_unrecognized_names() {
    let gate = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1, 1.0]);
    assert!(group_cosines(&["L1-a", "L1-b", "L2-a", "L2-b"], &gate).is_some());
    assert!(group_cosines(&["c1", "c2", "c3", "c4"], &gate).is_none());

    let (within, between) =
        group_cosines(&["L1-a", "L1-b", "L2-a", "L2-b"], &gate).expect("grouped labels");
    assert!(within > between);
}
