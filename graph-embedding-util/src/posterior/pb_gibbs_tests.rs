//! Cheap structural checks on the alternating sweep. The expensive claim — that
//! a planted `[features × pb]` truth is recovered on BOTH sides — is an
//! integration test in `tests/posterior_pb_gibbs.rs`, so it can run in release.

use super::*;
use nalgebra::DMatrix;

const H: usize = 3;
const N_FEAT: usize = 6;

/// Two levels, 3 then 2 pseudobulks, over `N_FEAT` backend rows.
fn counts() -> Vec<DMatrix<f32>> {
    let l0 = DMatrix::from_fn(N_FEAT, 3, |r, c| ((r * 3 + c * 2) % 5) as f32);
    let l1 = DMatrix::from_fn(N_FEAT, 2, |r, c| ((r + c * 3) % 4) as f32);
    vec![l0, l1]
}

fn stacked(c: &[DMatrix<f32>]) -> StackedPb<'_> {
    let sizes = vec![vec![4.0, 6.0, 5.0], vec![9.0, 11.0]];
    let n_pb: usize = c.iter().map(nalgebra::Matrix::ncols).sum();
    StackedPb {
        theta: (0..n_pb * H).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect(),
        bias: sizes.iter().flatten().map(|s: &f32| s.ln()).collect(),
        counts: c.iter().collect(),
        sizes,
        offsets: vec![0, 3],
    }
}

fn feat_buffers() -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    (
        (0..N_FEAT * H).map(|i| (i % 5) as f32 * 0.15 - 0.3).collect(),
        (0..N_FEAT).map(|i| (i % 3) as f32 * 0.1 - 0.2).collect(),
        (0..N_FEAT).collect(),
    )
}

fn run(cfg: &PbGibbsConfig, anchors: Option<&AnchorMap<'_>>) -> PbGibbsResult {
    let c = counts();
    let pb = stacked(&c);
    let (e, b, map) = feat_buffers();
    let feat = FeatureSide {
        e_feat: &e,
        b_feat: &b,
        feature_to_backend_row: &map,
    };
    pb_gibbs(&pb, &feat, anchors, H, cfg).unwrap()
}




/// Under a grouping the anchor axis shrinks to the gene count, and rows that
/// share a gene must not double its warm start — they are copies of one `β_g`,
/// not two independent observations of it.
#[test]
fn grouped_anchors_pool_without_doubling_the_warm_start() {
    // Rows 0,1 → gene 0; rows 2,3 → gene 1; rows 4,5 dropped from the gene side.
    let row_to_anchor = vec![0u32, 0, 1, 1, u32::MAX, u32::MAX];
    let anchors = AnchorMap {
        row_to_anchor: &row_to_anchor,
        n_anchors: 2,
    };
    let res = run(&PbGibbsConfig::new(4, 1, 5), Some(&anchors));

    assert_eq!(res.pip.len(), 2 * H);
    assert_eq!(res.mean_beta.len(), 2 * H);
    assert_eq!(res.mean_pb.len(), 5 * H, "the pb axis is unaffected by grouping");

    // The warm start itself is the thing at risk, so check it directly.
    let (e, b, map) = feat_buffers();
    let feat = FeatureSide {
        e_feat: &e,
        b_feat: &b,
        feature_to_backend_row: &map,
    };
    let ws = warm_start_genes(&feat, Some(&anchors), 2, H);
    assert_eq!(&ws[..H], &e[..H], "gene 0 takes row 0 verbatim, not row 0 + row 1");
    assert_eq!(&ws[H..2 * H], &e[2 * H..3 * H], "gene 1 takes row 2 verbatim");
}

/// A zero-sweep request is a caller bug, not a silently empty result.
#[test]
fn zero_sweeps_is_an_error() {
    let c = counts();
    let pb = stacked(&c);
    let (e, b, map) = feat_buffers();
    let feat = FeatureSide {
        e_feat: &e,
        b_feat: &b,
        feature_to_backend_row: &map,
    };
    let cfg = PbGibbsConfig {
        n_sweeps: 0,
        burnin: 0,
        ..PbGibbsConfig::new(0, 0, 1)
    };
    assert!(pb_gibbs(&pb, &feat, None, H, &cfg).is_err());
}

////////////////////////////////////
// Two-gate (splice) variant — §9 //
////////////////////////////////////

/// Rows 0,2,4 are spliced and rows 1,3,5 unspliced, so genes 0,1,2 each own one
/// row of each track.
fn splice_maps() -> (Vec<u32>, Vec<bool>) {
    let row_to_gene = vec![0u32, 0, 1, 1, 2, 2];
    let unspliced = vec![false, true, false, true, false, true];
    (row_to_gene, unspliced)
}

fn run_splice(nested: bool, seed: u64, counts_override: Option<Vec<DMatrix<f32>>>) -> SpliceGibbsResult {
    let c = counts_override.unwrap_or_else(counts);
    let pb = stacked(&c);
    let (e, b, map) = feat_buffers();
    let feat = FeatureSide {
        e_feat: &e,
        b_feat: &b,
        feature_to_backend_row: &map,
    };
    let (row_to_gene, unspliced) = splice_maps();
    let tracks = SpliceTracks {
        row_to_gene: &row_to_gene,
        unspliced_rows: &unspliced,
        n_genes: 3,
        nested,
    };
    pb_gibbs_splice(&pb, &feat, &tracks, H, &PbGibbsConfig::new(6, 2, seed)).unwrap()
}


/// The nesting constraint: `z_δ` may only be on where `z_β` is. Asserted through
/// the posterior means, since an excluded draw contributes exactly zero — so a δ
/// coordinate can only be nonzero where β also is.
#[test]
fn nesting_keeps_delta_inside_betas_support() {
    let r = run_splice(true, 17, None);
    for g in 0..3 {
        for d in 0..H {
            let i = g * H + d;
            if r.beta_pip[i] == 0.0 {
                assert_eq!(
                    r.delta_pip[i], 0.0,
                    "gene {g} dim {d}: δ included where β never was"
                );
            }
        }
    }
}

/// A gene with counts in the unspliced track ONLY pins `β + δ` but neither
/// separately. It must be reported as unidentified rather than handed a number
/// indistinguishable from a measurement.
#[test]
fn a_gene_with_no_spliced_counts_is_flagged_unidentified() {
    // Zero out gene 1's spliced rows (row 2) in both levels; leave its unspliced
    // row (row 3) carrying counts.
    let mut c = counts();
    for m in &mut c {
        for s in 0..m.ncols() {
            m[(2, s)] = 0.0;
            m[(3, s)] = 3.0;
        }
    }
    let r = run_splice(true, 5, Some(c));
    assert!(r.delta_identified[0], "gene 0 keeps its spliced counts");
    assert!(!r.delta_identified[1], "gene 1 has no spliced counts");
    assert!(r.delta_identified[2], "gene 2 keeps its spliced counts");
}


