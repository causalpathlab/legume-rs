use super::*;

const H: usize = 4;

/// `g` genes, each a distinct axis-ish vector whose *norm* is `norm[g]`.
fn emb(norms: &[f32]) -> Vec<f32> {
    let mut v = Vec::new();
    for (g, &nrm) in norms.iter().enumerate() {
        let mut row = vec![0f32; H];
        row[g % H] = nrm;
        v.extend(row);
    }
    v
}

fn markers(spec: &[&[u32]]) -> Vec<Vec<(u32, f32)>> {
    spec.iter()
        .map(|ids| ids.iter().map(|&g| (g, 1.0)).collect())
        .collect()
}

fn norm_of(feature_emb: &[f32], g: u32) -> f32 {
    let r = &feature_emb[g as usize * H..(g as usize + 1) * H];
    r.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// **The whole point of stratifying.** A free shuffle would hand type 0's long-vector panel to
/// type 1 and vice versa, so each type's *norm* would change — and since a longer centroid wins
/// cells almost regardless of direction, the null would be measuring norm rather than biology.
/// Within-stratum shuffling must leave every type's norm profile exactly as it was.
#[test]
fn the_shuffle_preserves_each_types_norm_profile() {
    // Type 0 holds the 20 longest genes, type 1 the 20 shortest. (40 genes ⇒ 4 strata of 10 —
    // the pool has to be big enough to stratify at all; see `GeneStrata::MIN_PER_STRATUM`.)
    let long: Vec<f32> = (0..20).map(|i| 8.0 - i as f32 * 0.1).collect();
    let short: Vec<f32> = (0..20).map(|i| 1.0 + i as f32 * 0.05).collect();
    let feature_emb = emb(&[long.clone(), short.clone()].concat());
    let a: Vec<u32> = (0..20).collect();
    let b: Vec<u32> = (20..40).collect();
    let tm = markers(&[&a, &b]);
    let sh = StratifiedShuffle::new(&feature_emb, &tm, H);
    let mut rng = SmallRng::seed_from_u64(1);

    let profile = |panel: &[Vec<(u32, f32)>]| -> Vec<Vec<i64>> {
        panel
            .iter()
            .map(|ms| {
                let mut v: Vec<i64> = ms
                    .iter()
                    .map(|&(g, _)| (norm_of(&feature_emb, g) * 1000.0) as i64)
                    .collect();
                v.sort_unstable();
                v
            })
            .collect()
    };
    let want = profile(&tm);

    let mut out: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 2];
    for _ in 0..20 {
        sh.draw(&mut rng, &mut out);
        assert_eq!(
            profile(&out),
            want,
            "a shuffled panel must carry the same norms as the real one — only the gene \
             identities may move"
        );
        // Sizes and weights are preserved too.
        assert_eq!(out[0].len(), 20);
        assert_eq!(out[1].len(), 20);
    }
}

/// The shuffle must actually shuffle: within a stratum, genes do move between types.
///
/// Two ways this can silently die, and the test guards both. One bin per gene leaves nothing to
/// swap (hence [`super::gene_strata`]'s minimum bin size). And if the two types occupy *disjoint*
/// norm ranges, every stratum is owned outright by one type and a within-stratum shuffle permutes
/// a type's genes among its own slots — a null that never moves a gene across a type boundary.
/// Here the types are **interleaved** in norm, which is the realistic case: real panels are not
/// sorted by norm.
#[test]
fn the_shuffle_actually_moves_genes() {
    // 40 genes of near-identical norm, types interleaved ⇒ every stratum holds both.
    let norms: Vec<f32> = (0..40).map(|i| 2.0 + i as f32 * 0.001).collect();
    let feature_emb = emb(&norms);
    let a: Vec<u32> = (0..40).filter(|i| i % 2 == 0).collect();
    let b: Vec<u32> = (0..40).filter(|i| i % 2 == 1).collect();
    let tm = markers(&[&a, &b]);
    let sh = StratifiedShuffle::new(&feature_emb, &tm, H);
    let mut rng = SmallRng::seed_from_u64(2);
    let mut out: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 2];

    let mut moved = false;
    for _ in 0..20 {
        sh.draw(&mut rng, &mut out);
        // Did any of type 1's original genes (the odd ones) land in type 0?
        if out[0].iter().any(|&(g, _)| g % 2 == 1) {
            moved = true;
            break;
        }
    }
    assert!(
        moved,
        "genes of equal norm must be free to swap types, or the null tests nothing"
    );
}

/// Every gene is accounted for: the shuffled panel is a permutation of the real one, so the gene
/// multiset as a whole never changes — only who owns what.
#[test]
fn the_shuffle_is_a_permutation() {
    let norms: Vec<f32> = (0..30).map(|i| 1.0 + i as f32 * 0.2).collect();
    let feature_emb = emb(&norms);
    let a: Vec<u32> = (0..10).collect();
    let b: Vec<u32> = (10..25).collect();
    let c: Vec<u32> = (25..30).collect();
    let tm = markers(&[&a, &b, &c]);
    let sh = StratifiedShuffle::new(&feature_emb, &tm, H);
    let mut rng = SmallRng::seed_from_u64(3);
    let mut out: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 3];

    let sorted = |panel: &[Vec<(u32, f32)>]| {
        let mut v: Vec<u32> = panel.iter().flatten().map(|&(g, _)| g).collect();
        v.sort_unstable();
        v
    };
    for _ in 0..10 {
        sh.draw(&mut rng, &mut out);
        assert_eq!(
            sorted(&out),
            sorted(&tm),
            "the gene multiset must be intact"
        );
        assert_eq!(
            out.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![10, 15, 5]
        );
    }
}
