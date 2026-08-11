//! Tests for the per-pair specificity readout.

use super::*;
use crate::eqtl::evidence::classify_states;
use crate::eqtl::model::{train, EqtlModelConfig};
use crate::eqtl::select::{Pair, PairObs, Selection};
use crate::eqtl::ubiquity::ubiquity_index;
use candle_util::candle_core::Device;

fn celltypes(n: usize) -> Vec<Box<str>> {
    (0..n).map(|k| Box::from(format!("C{k}"))).collect()
}

/// Two variants over two genes, each acting in a different set of cell types.
///
/// Every pair needs at least TWO strong cell types — the reference effect is
/// the runner-up by `|z|`, so a pair strong in exactly one context certifies
/// nothing and yields no negative to train against.
fn selection() -> Selection {
    let mut pairs = Vec::new();
    for gene in 0..2u32 {
        for variant in 0..2u32 {
            let obs = (0..4u32)
                .map(|k| {
                    let strong = k <= variant + 1;
                    PairObs {
                        celltype: k,
                        beta: if strong { 0.8 } else { 0.01 },
                        se: 0.1,
                    }
                })
                .collect();
            pairs.push(Pair { gene, variant, obs });
        }
    }
    let n_rows = pairs.iter().map(|p| p.obs.len()).sum();
    Selection {
        pairs,
        n_selected_variants: 2,
        n_pairs_dropped: 0,
        n_rows,
    }
}

fn fit_once() -> (crate::eqtl::EvidenceTable, Vec<UbiquityRow>, EqtlFit) {
    let sel = selection();
    let cts = celltypes(4);
    let evidence = classify_states(&sel, &cts, 4.0).unwrap();
    let ubiquity = ubiquity_index(&evidence);
    let variants: Vec<Box<str>> = (0..2).map(|v| Box::from(format!("1:{v}"))).collect();
    let genes: Vec<Box<str>> = (0..2).map(|g| Box::from(format!("G{g}"))).collect();
    let config = EqtlModelConfig {
        embedding_dim: 4,
        num_negatives: 5,
        num_iterations: 50,
        batch_size: 32,
        learning_rate: 0.05,
        ridge: 1e-3,
        holdout_frac: 0.2,
        shuffle_control: false,
        seed: 1,
    };
    let fit = train(&evidence, &variants, &genes, &config, &Device::Cpu).unwrap();
    (evidence, ubiquity, fit)
}

#[test]
fn every_row_carries_one_score_per_real_context() {
    let (evidence, ubiquity, fit) = fit_once();
    let rows = specificity_rows(&evidence, &ubiquity, &fit);
    assert!(!rows.is_empty(), "no pair was scored");
    let n_real = fit.real_contexts().len();
    for row in &rows {
        assert_eq!(row.scores.len(), n_real);
    }
}

/// `pair_product` exists to avoid recomputing `u_j * v_g` per context. It has
/// to agree with the direct scoring path, or the readout silently drifts from
/// what the model says.
#[test]
fn gated_pair_product_equals_the_direct_score() {
    let (_, _, fit) = fit_once();
    for j in 0..fit.variants.len() {
        for g in 0..fit.genes.len() {
            let uv = fit.pair_product(j, g);
            for k in 0..fit.contexts.len() {
                let shared = fit.gated(&uv, k);
                let direct = fit.score(j, g, k);
                assert!(
                    (shared - direct).abs() < 1e-5,
                    "context {k}: shared {shared} vs direct {direct}"
                );
            }
            let anchor: f32 = uv.iter().sum();
            assert!((anchor - fit.anchor(j, g)).abs() < 1e-5);
        }
    }
}

/// The reported best context must be the argmax of the reported scores —
/// they are written to adjacent columns and are read together.
#[test]
fn best_context_is_the_argmax_of_the_written_scores() {
    let (evidence, ubiquity, fit) = fit_once();
    let real = fit.real_contexts();
    for row in specificity_rows(&evidence, &ubiquity, &fit) {
        let (best_i, best) = row
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert_eq!(
            row.best_context.as_deref(),
            Some(fit.contexts.names[real[best_i]].as_ref())
        );
        assert_eq!(row.best_score, Some(*best));
    }
}

/// The ubiquitous pseudo-context is never one of the score columns: it is the
/// anchor, reported separately.
#[test]
fn the_ubiquitous_context_is_not_a_score_column() {
    let (_, _, fit) = fit_once();
    let real = fit.real_contexts();
    if let Some(ubi) = fit.ubiquitous {
        assert!(!real.contains(&(ubi as usize)));
        assert_eq!(real.len(), fit.contexts.len() - 1);
    }
}
