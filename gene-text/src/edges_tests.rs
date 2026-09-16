use super::*;
use crate::vocab::{DfQcOpts, Vocabulary};

fn vocab(words: &[&str]) -> Vocabulary {
    // One document per word plus one holding all, so every word has df 2
    // and nothing is cut.
    let mut docs: Vec<Vec<Occurrence>> = words
        .iter()
        .map(|w| {
            vec![Occurrence {
                word: (*w).into(),
                start: 0,
                end: w.len(),
            }]
        })
        .collect();
    docs.push(
        words
            .iter()
            .map(|w| Occurrence {
                word: (*w).into(),
                start: 0,
                end: w.len(),
            })
            .collect(),
    );
    Vocabulary::build(
        &docs,
        &DfQcOpts {
            lower_quantile: 0.0,
            upper_quantile: 1.0,
            min_df: 0,
            max_df_frac: 1.0,
        },
    )
}

fn tok(start: usize, end: usize, v: &[f32]) -> TokenVec {
    TokenVec {
        start,
        end,
        vec: v.to_vec(),
    }
}

#[test]
fn a_word_occurrence_averages_the_tokens_overlapping_its_span_and_scores_by_cosine() {
    let v = vocab(&["kinase", "membrane"]);
    // text: "kinase membrane kinase": kinase [0,6), membrane [7,15), kinase [16,22)
    // The first kinase splits into two word pieces [0,3) and [3,6).
    let pooled = [1.0, 0.0];
    let tokens = vec![
        tok(0, 3, &[1.0, 0.0]),
        tok(3, 6, &[1.0, 0.0]),
        tok(7, 15, &[0.0, 1.0]),
        tok(16, 22, &[0.0, 1.0]),
        tok(23, 30, &[5.0, 5.0]), // a token beyond every occurrence
    ];
    let occ = vec![
        Occurrence {
            word: "kinase".into(),
            start: 0,
            end: 6,
        },
        Occurrence {
            word: "membrane".into(),
            start: 7,
            end: 15,
        },
        Occurrence {
            word: "kinase".into(),
            start: 16,
            end: 22,
        },
        Occurrence {
            word: "unknown".into(),
            start: 40,
            end: 47,
        },
    ];
    let s = score_doc(&pooled, &tokens, &occ, &v);
    assert_eq!(s.len(), 2, "the unknown word is not in the vocabulary");
    let kinase = s
        .iter()
        .find(|x| v.kept[x.word].as_ref() == "kinase")
        .unwrap();
    let membrane = s
        .iter()
        .find(|x| v.kept[x.word].as_ref() == "membrane")
        .unwrap();
    assert_eq!(kinase.count, 2);
    // occurrence 1 = mean([1,0],[1,0]) = [1,0]; occurrence 2 = [0,1]; mean = [.5,.5]
    assert_eq!(kinase.vec, vec![0.5, 0.5]);
    assert!((kinase.cos - 0.5f32.sqrt()).abs() < 1e-6);
    assert_eq!(membrane.count, 1);
    assert!(membrane.cos.abs() < 1e-6, "orthogonal to the pooled vector");
    // A truncated occurrence (no token overlaps) is skipped.
    let s2 = score_doc(&pooled, &tokens[..1], &occ[1..2], &v);
    assert!(s2.is_empty());
}

#[test]
fn weights_are_cosine_times_tf_idf_top_k_and_never_negative() {
    let v = vocab(&["a", "b", "c"]);
    let scores = vec![
        WordScore {
            word: 0,
            cos: 0.9,
            count: 1,
            vec: vec![],
        },
        WordScore {
            word: 1,
            cos: 0.5,
            count: 3,
            vec: vec![],
        },
        WordScore {
            word: 2,
            cos: -0.4,
            count: 1,
            vec: vec![],
        },
    ];
    let w = feature_word_weights(&scores, &v, 5);
    assert_eq!(w.len(), 2, "the negative cosine is dropped");
    let idf = v.idf(0) as f32;
    let of = |word: usize| w.iter().find(|(x, _)| *x == word).unwrap().1;
    assert!((of(0) - 0.9 * idf).abs() < 1e-5);
    let expect_b = 0.5 * (1.0 + (3f32).ln()) * idf;
    assert!((of(1) - expect_b).abs() < 1e-5);
    assert!(w[0].1 >= w[1].1, "sorted by weight, descending");
    assert_eq!(w[0].0, 1, "three mentions at 0.5 outweigh one at 0.9");
    assert_eq!(feature_word_weights(&scores, &v, 1).len(), 1);
}

#[test]
fn typed_edges_are_five_tab_separated_columns() {
    let mut buf: Vec<u8> = Vec::new();
    let n = write_typed_edges(
        &mut buf,
        [("gene", "TP53", "word", "apoptosis", 0.61234f32)].into_iter(),
    )
    .unwrap();
    assert_eq!(n, 1);
    assert_eq!(
        String::from_utf8(buf).unwrap(),
        "gene\tTP53\tword\tapoptosis\t0.6123\n"
    );
}

#[test]
fn top_k_cosine_and_csls_rank_the_planted_neighbour_first() {
    let dev = Device::Cpu;
    let rows = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let center = vec![0.0, 0.0, 0.0];
    let t = centred_unit(&rows, &center, &dev).unwrap();
    let knn = top_k_cosine(&t, &t, 1, true).unwrap();
    assert_eq!(knn[0][0].0, 1);
    assert_eq!(knn[1][0].0, 0);
    assert!(knn[0][0].1 > 0.99);
    let with_self = top_k_cosine(&t, &t, 1, false).unwrap();
    assert_eq!(with_self[2][0].0, 2);
    // A hub key close to everything is pushed down by CSLS.
    let keys = vec![
        vec![0.6, 0.6, 0.6], // hub
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    let kt = centred_unit(&keys, &center, &dev).unwrap();
    let plain = top_k_cosine(&t, &kt, 1, false).unwrap();
    let csls = csls_top_k(&t, &kt, 1, 2).unwrap();
    assert_eq!(csls[0][0].0, 1, "x-axis query prefers the x-axis key");
    assert_eq!(csls[2][0].0, 2);
    // For the z-axis query the hub is the plain best (nothing else is
    // close), and CSLS still returns something finite.
    assert_eq!(plain[3][0].0, 0);
    assert!(csls[3][0].1.is_finite());
    let m = mean_of(&rows, 3);
    assert!((m[0] - 0.475).abs() < 1e-6);
}
