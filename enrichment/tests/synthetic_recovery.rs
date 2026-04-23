//! End-to-end synthetic recovery test for `enrichment::annotate`.
//!
//! Construct a small synthetic dataset where the ground truth is known,
//! then verify the full pipeline recovers it.

use enrichment::{annotate, AnnotateConfig, AnnotateOutputs, GroupInputs, Mat, SpecificityMode};
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[test]
fn synthetic_three_blocks_recovered() {
    // K=3 topics, C=3 celltypes. Larger gene universe with a modest marker
    // footprint per celltype, non-degenerate β (varying within blocks, noise
    // outside) so row randomization is well-behaved.
    let k = 3usize;
    let c = 3usize;
    let block_genes = 30usize; // markers per celltype
    let noise_genes = 200usize; // non-marker background
    let g = k * block_genes + noise_genes;
    let p = 90usize;
    let n_cells = 120usize;

    let _rng_setup = SmallRng::seed_from_u64(99);

    // β (G × K): topic k has loading 1.0 on its block; small baseline 0.05
    // outside so rows aren't all zero. (Deterministic construction — no
    // noise needed since the test has fixed topology.)
    let mut beta = Mat::zeros(g, k);
    for kk in 0..k {
        for i in 0..block_genes {
            let gi = kk * block_genes + i;
            beta[(gi, kk)] = 1.0;
            for other in 0..k {
                if other != kk {
                    beta[(gi, other)] = 0.05;
                }
            }
        }
    }
    for i in 0..noise_genes {
        let gi = k * block_genes + i;
        for kk in 0..k {
            beta[(gi, kk)] = 0.05;
        }
    }

    // Markers (G × C): celltype c's markers = its block's genes.
    let mut markers = Mat::zeros(g, c);
    for cc in 0..c {
        for i in 0..block_genes {
            markers[(cc * block_genes + i, cc)] = 1.0;
        }
    }

    // Cell membership (N × K): each cell is purely one topic.
    let mut cell_membership = Mat::zeros(n_cells, k);
    let mut true_labels: Vec<usize> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let dom = i % k;
        cell_membership[(i, dom)] = 1.0;
        true_labels.push(dom);
    }

    // PB membership (P × K): one-hot per PB.
    let mut pb_membership = Mat::zeros(p, k);
    for pi in 0..p {
        pb_membership[(pi, pi % k)] = 1.0;
    }

    // PB gene aggregates (G × P): β · pb_membership^T with block-aligned
    // signal scale so specificity on β̃ matches β's structure.
    let mut pb_gene = Mat::zeros(g, p);
    for pi in 0..p {
        let dom = pi % k;
        for gi in 0..g {
            pb_gene[(gi, pi)] = beta[(gi, dom)] * 1000.0;
        }
    }

    let gene_names: Vec<Box<str>> = (0..g)
        .map(|i| format!("gene_{i}").into_boxed_str())
        .collect();
    let cell_names: Vec<Box<str>> = (0..n_cells)
        .map(|i| format!("cell_{i}").into_boxed_str())
        .collect();
    let celltype_names: Vec<Box<str>> =
        (0..c).map(|i| format!("ct_{i}").into_boxed_str()).collect();

    let inputs = GroupInputs {
        profile_gk: beta,
        pb_gene_gp: pb_gene,
        pb_membership_pk: pb_membership,
        cell_membership_nk: cell_membership,
        gene_names,
        cell_names,
    };

    // Small K=3 synthetic: use row-randomization fallback to avoid the
    // pooled-null ~1/K floor. Real K=20 runs should use sample permutation.
    let config = AnnotateConfig {
        specificity: SpecificityMode::Simplex,
        num_row_randomization: 500,
        num_sample_perm: 0,
        batch_labels: None,
        fdr_alpha: 0.10,
        q_softmax_temperature: 1.0,
        min_confidence: 0.0,
        seed: 7,
    };

    let AnnotateOutputs {
        q_kc,
        es_kc,
        es_restandardized_kc,
        qvalue_kc,
        argmax_labels,
        ..
    } = annotate(&inputs, &markers, &celltype_names, &config).unwrap();

    // Diagonal ES should be clearly positive; off-diagonal should be smaller.
    for d in 0..k {
        assert!(
            es_kc[(d, d)] > 0.5,
            "diagonal ES [{d},{d}] = {} not strongly positive",
            es_kc[(d, d)]
        );
        for other in 0..c {
            if other != d {
                assert!(
                    es_kc[(d, d)] > es_kc[(d, other)],
                    "diag [{d},{d}]={} not greater than off [{d},{other}]={}",
                    es_kc[(d, d)],
                    es_kc[(d, other)]
                );
            }
        }
    }

    // Diagonal restandardized ES should be large and positive.
    for d in 0..k {
        assert!(
            es_restandardized_kc[(d, d)] > 2.0,
            "diagonal restandardized ES [{d},{d}] = {} not > 2 SD",
            es_restandardized_kc[(d, d)]
        );
    }

    // Diagonal q-values should be small; off-diagonal large.
    for d in 0..k {
        assert!(
            qvalue_kc[(d, d)] < 0.05,
            "diagonal q [{d},{d}] = {} not < 0.05",
            qvalue_kc[(d, d)]
        );
    }

    // Q matrix: diagonal should dominate per row.
    for d in 0..k {
        assert!(
            q_kc[(d, d)] > 0.5,
            "Q[{d},{d}] = {} should dominate row",
            q_kc[(d, d)]
        );
    }

    // Cell labels: >95% correct.
    let mut correct = 0usize;
    for (i, lab) in argmax_labels.iter().enumerate() {
        let expected = celltype_names[true_labels[i]].clone();
        if lab.label == expected {
            correct += 1;
        }
    }
    let frac = correct as f32 / n_cells as f32;
    assert!(frac > 0.95, "recovery rate {frac} < 0.95");
}
