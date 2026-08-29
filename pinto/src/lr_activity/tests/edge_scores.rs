//! Contract of the contact-association score: instances are BOTH
//! orientations of every within-community spatial edge, grouped by batch;
//! each instance is classified by endpoint detection into a 2x2 table,
//! and the score is the Jeffreys (+1/2) posterior log odds ratio with its
//! posterior SE. Tables here are small enough to enumerate by hand.

use crate::lr_activity::edge_scores::{
    classify_contact, compute_edge_scores, jeffreys_log_odds, ContactConfig, EdgeScoresInput,
};
use crate::lr_activity::orientation::CommunityStrata;
use crate::util::common::*;

type Edge = (usize, usize, u32, Option<Box<str>>);

/// One ligand-receptor pair: (ligand name, receptor name, global ligand
/// row, global receptor row).
type LrPair = (Box<str>, Box<str>, usize, usize);

/// What [`fixture_genes`] hands back: the pairs, the global-to-local gene
/// row map, and the ligand/receptor count block.
type GeneFixture = (Vec<LrPair>, HashMap<usize, usize>, Mat);

/// Cells 0-2 sit in community 0, cells 3-5 in community 1. One bridging
/// edge (2,3) sits out by construction.
fn fixture_edges() -> Vec<Edge> {
    let b = |s: &str| -> Option<Box<str>> { Some(s.into()) };
    vec![
        (0, 1, 0, b("a")),
        (1, 2, 0, b("b")),
        (3, 4, 1, b("a")),
        (4, 5, 1, b("a")),
        (2, 3, 2, b("a")), // bridges C0 and C1: no stratum
    ]
}

/// L is global gene 10 (row 0 of x_lr), R is global gene 11 (row 1),
/// and gene 12 (row 2) is never detected anywhere.
fn fixture_genes() -> GeneFixture {
    let pairs = vec![("LIG1".into(), "REC1".into(), 10usize, 11usize)];
    let mut gene_to_local: HashMap<usize, usize> = HashMap::default();
    gene_to_local.insert(10, 0);
    gene_to_local.insert(11, 1);
    gene_to_local.insert(12, 2);
    // Counts per cell (column = cell); only detection (> 0) matters.
    let x_lr = Mat::from_row_slice(
        3,
        6,
        &[
            2.0, 5.0, 0.0, 1.0, 3.0, 4.0, // ligand:   + + - + + +
            1.0, 0.0, 7.0, 2.0, 6.0, 0.0, // receptor: + - + + + -
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // never detected
        ],
    );
    (pairs, gene_to_local, x_lr)
}

fn run_pairs(
    edges: &[Edge],
    pairs: Vec<LrPair>,
) -> (Vec<crate::lr_activity::edge_scores::EdgeScoreRow>, usize) {
    let (_, gene_to_local, x_lr) = fixture_genes();
    let strata = CommunityStrata::from_edge_modes(edges, edges, 6);
    let log_depth: Vec<f32> = (0..6).map(|i| (10 + i) as f32).collect();
    compute_edge_scores(&EdgeScoresInput {
        edges,
        strata: &strata,
        pairs: &pairs,
        gene_to_local: &gene_to_local,
        x_lr: &x_lr,
        log_depth: &log_depth,
    })
}

fn run(edges: &[Edge]) -> (Vec<crate::lr_activity::edge_scores::EdgeScoreRow>, usize) {
    let (pairs, _, _) = fixture_genes();
    run_pairs(edges, pairs)
}

/// The (batch, community) groups are exactly the ones with edges; a
/// zero-edge combination gets no row at all.
#[test]
fn a_zero_edge_batch_community_gets_no_row() {
    let (rows, n_straddling) = run(&fixture_edges());
    assert_eq!(n_straddling, 0);
    let keys: Vec<(String, u32)> = rows
        .iter()
        .map(|r| (r.batch.to_string(), r.community))
        .collect();
    // Batch "b" has no community-1 edge, and the bridge realizes nothing.
    assert_eq!(
        keys,
        vec![
            ("a".to_string(), 0),
            ("b".to_string(), 0),
            ("a".to_string(), 1)
        ]
    );
}

/// Hand-enumerated tables. The SE doubles the Woolf variance because the
/// physical contact, not the oriented instance, is the sampling unit.
///
/// (a, C0): one edge (0,1), instances (0,1) and (1,0).
///   (0,1): L+ at 0, R- at 1;  (1,0): L+ at 1, R+ at 0.
///   n=2, n11=1, nL=2, nR=1 -> cells +1/2: a=1.5 b=1.5 c=0.5 d=0.5
///   log_or = ln(1.5*0.5 / (1.5*0.5)) = 0
///   se     = sqrt(2 * (1/1.5 + 1/1.5 + 1/0.5 + 1/0.5)) = sqrt(32/3)
///
/// (a, C1): edges (3,4),(4,5), both ways.
///   L+ at 3,4,5; R+ at 3,4 only. n=4, n11=3, nL=4, nR=3
///   -> a=3.5 b=1.5 c=0.5 d=0.5
///   log_or = ln(3.5*0.5 / (1.5*0.5)) = ln(7/3)
///   se     = sqrt(2 * (1/3.5 + 1/1.5 + 2 + 2))
#[test]
fn scores_match_hand_built_tables() {
    let (rows, _) = run(&fixture_edges());

    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 0)
        .expect("(a, C0) row");
    assert_eq!(row.n_edges, 1);
    assert_eq!(row.lig_rate, 1.0, "ligand detected at both endpoints");
    assert_eq!(row.rec_rate, 0.5);
    assert!(
        row.log_or.abs() < 1e-6,
        "balanced table, got {}",
        row.log_or
    );
    let se = (32.0f32 / 3.0).sqrt();
    assert!((row.log_or_se - se).abs() < 1e-5, "got {}", row.log_or_se);
    // Unique first-endpoint cells are 0 and 1; depth fixture is 10 + cell.
    assert!((row.mean_log_depth - 10.5).abs() < 1e-6);

    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 1)
        .expect("(a, C1) row");
    assert_eq!(row.n_edges, 2);
    assert_eq!(row.lig_rate, 1.0);
    assert_eq!(row.rec_rate, 0.75);
    let lor = (7.0f32 / 3.0).ln();
    assert!((row.log_or - lor).abs() < 1e-5, "got {}", row.log_or);
    let se = (2.0 * (1.0f32 / 3.5 + 1.0 / 1.5 + 2.0 + 2.0)).sqrt();
    assert!((row.log_or_se - se).abs() < 1e-5, "got {}", row.log_or_se);
}

/// Both-orientation enumeration transposes the 2x2 under a ligand to
/// receptor swap, and the odds ratio is transpose-invariant: the score is
/// symmetric structurally, while the margins swap roles.
#[test]
fn swapping_ligand_and_receptor_transposes_but_scores_agree() {
    let (mut pairs, _, _) = fixture_genes();
    pairs.push(("REC1".into(), "LIG1".into(), 11, 10));
    let (rows, _) = run_pairs(&fixture_edges(), pairs);
    for key in [("a", 0u32), ("b", 0), ("a", 1)] {
        let group: Vec<_> = rows
            .iter()
            .filter(|r| r.batch.as_ref() == key.0 && r.community == key.1)
            .collect();
        assert_eq!(group.len(), 2, "both pair orders scored");
        let (fwd, rev) = (group[0], group[1]);
        assert!(
            (fwd.log_or - rev.log_or).abs() < 1e-6,
            "{key:?}: log_or must be symmetric"
        );
        assert!(
            (fwd.log_or_se - rev.log_or_se).abs() < 1e-6,
            "{key:?}: its SE must be symmetric"
        );
        assert_eq!(fwd.lig_rate, rev.rec_rate, "{key:?}: margins swap roles");
        assert_eq!(fwd.rec_rate, rev.lig_rate, "{key:?}: margins swap roles");
    }
}

/// A pair whose table is prior dominated (no co-detection observed, and
/// none expected under independence) is NaN in both columns: the +1/2
/// prior alone would otherwise push the estimate toward ln(2n), which
/// reads as strong POSITIVE association for a pair the stratum never
/// measured at all.
#[test]
fn a_prior_dominated_pair_is_nan_not_scored() {
    let pairs = vec![
        ("LIG1".into(), "REC1".into(), 10usize, 11usize),
        ("LIG1".into(), "REC2".into(), 10, 12),
    ];
    let (rows, _) = run_pairs(&fixture_edges(), pairs);
    let c1: Vec<_> = rows
        .iter()
        .filter(|r| r.batch.as_ref() == "a" && r.community == 1)
        .collect();
    let supported = c1.iter().find(|r| r.receptor.as_ref() == "REC1").unwrap();
    let empty = c1.iter().find(|r| r.receptor.as_ref() == "REC2").unwrap();
    assert_eq!(empty.rec_rate, 0.0);
    assert!(empty.log_or.is_nan(), "unmeasurable, not a number");
    assert!(empty.log_or_se.is_nan(), "the SE must not pretend either");
    assert!(
        supported.log_or.is_finite(),
        "the measured pair still scores"
    );
    // The margins still report: the row says WHY it is unmeasurable.
    assert_eq!(empty.lig_rate, 1.0);
}

/// The degenerate 2x2 tables, pinned at the estimator itself.
///
/// Both margins empty was the reachable untested gap: the plug-in
/// formula would return ln(2n+1) with a plateaued SE, a large positive
/// "association" for a silent pair. Tiny margins with n11 = 0 are the
/// same pathology. But a zero-count table whose margins EXPECTED
/// co-detections is informative, and must stay scored, negative:
/// jeffreys(4, 0, 2, 1) has a=0.5 b=2.5 c=1.5 d=1.5
/// -> log_or = ln(0.5*1.5 / (2.5*1.5)) = ln(1/5).
#[test]
fn degenerate_tables_are_refused_or_scored_by_information() {
    let (lor, se) = jeffreys_log_odds(100, 0, 0, 0);
    assert!(
        lor.is_nan() && se.is_nan(),
        "silent stratum: no information"
    );

    let (lor, se) = jeffreys_log_odds(1000, 0, 1, 1);
    assert!(
        lor.is_nan() && se.is_nan(),
        "prior dominated: no information"
    );

    let (lor, se) = jeffreys_log_odds(4, 0, 2, 1);
    assert!(
        (lor - 0.2f32.ln()).abs() < 1e-6,
        "expected co-detections, saw none: negative, got {lor}"
    );
    assert!(se.is_finite());
}

/// A straddling edge (labels exist, but this edge's endpoints disagree so
/// its joint label is None) belongs to no single batch: it is dropped from
/// every group and counted once.
#[test]
fn straddling_edges_are_dropped_and_counted() {
    let mut edges = fixture_edges();
    edges.push((0, 2, 0, None)); // C0-internal but straddling
    let (rows, n_straddling) = run(&edges);
    assert_eq!(n_straddling, 1, "counted once, not once per orientation");
    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 0)
        .unwrap();
    assert_eq!(row.n_edges, 1, "the straddler joined no group");
}

/// With no labels on file at all the run is single-batch: every edge lands
/// in the `all` pseudo-batch and nothing counts as straddling.
#[test]
fn unlabeled_runs_score_one_all_batch() {
    let edges: Vec<Edge> = fixture_edges()
        .into_iter()
        .map(|(i, j, k, _)| (i, j, k, None))
        .collect();
    let (rows, n_straddling) = run(&edges);
    assert_eq!(n_straddling, 0);
    assert!(rows.iter().all(|r| r.batch.as_ref() == "all"));
    let n_edges: Vec<u32> = rows.iter().map(|r| r.n_edges).collect();
    assert_eq!(n_edges, vec![2, 2], "C0 and C1, batches pooled");
}

/// The four contact configurations, from the four detection facts. The
/// ligand is annotated, so `ligand_first` is a statement about which
/// endpoint CARRIES it, and inverting it would silently flip every
/// arrow the overlay draws from roles.
#[test]
fn contact_configurations_classify_by_annotated_role() {
    assert_eq!(
        classify_contact(true, false, false, true),
        ContactConfig::OneWay { ligand_first: true },
        "ligand at the first endpoint, receptor at the second"
    );
    assert_eq!(
        classify_contact(false, true, true, false),
        ContactConfig::OneWay {
            ligand_first: false
        },
        "roles swapped ends"
    );
    assert_eq!(
        classify_contact(true, true, true, true),
        ContactConfig::Mutual
    );
    assert_eq!(
        classify_contact(true, false, true, false),
        ContactConfig::Silent,
        "two ligand carriers and no receptor is not an interaction"
    );
    assert_eq!(
        classify_contact(false, false, false, false),
        ContactConfig::Silent
    );
}

/// Configuration counts and role purity on the hand-built fixture.
///
/// (a, C0): edge (0,1). L+ at 0 and 1; R+ at 0 only. Forward (L0, R1)
/// fails, reverse (L1, R0) holds: one one-way contact, ligand at cell 1.
/// Purity: cell 1 sent once (1.0), cell 0 received once (1.0) -> 1.0.
///
/// (a, C1): edges (3,4), (4,5). L+ at 3,4,5; R+ at 3,4.
/// (3,4): both orientations hold -> mutual. (4,5): only (L5, R4) holds
/// -> one-way, ligand at 5. Identity: pooled n11 = 2*mutual + oneway
/// = 3, matching the 2x2 test above. Purity: cell 3 (1,1) -> 0,
/// cell 4 (1,2) -> 1/3, cell 5 (1,0) -> 1; mean = 4/9.
#[test]
fn configuration_counts_and_purity_match_hand_enumeration() {
    let (rows, _) = run(&fixture_edges());
    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 0)
        .unwrap();
    assert_eq!((row.n_mutual, row.n_oneway), (0, 1));
    assert!(
        (row.role_purity - 1.0).abs() < 1e-6,
        "got {}",
        row.role_purity
    );

    let row = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 1)
        .unwrap();
    assert_eq!((row.n_mutual, row.n_oneway), (1, 1));
    assert!(
        (row.role_purity - 4.0 / 9.0).abs() < 1e-6,
        "got {}",
        row.role_purity
    );
    // A pair with no co-detected contact reports NaN purity, not zero.
    let (rows, _) = run_pairs(
        &fixture_edges(),
        vec![("LIG1".into(), "REC2".into(), 10, 12)],
    );
    let empty = rows
        .iter()
        .find(|r| r.batch.as_ref() == "a" && r.community == 1)
        .unwrap();
    assert_eq!((empty.n_mutual, empty.n_oneway), (0, 0));
    assert!(empty.role_purity.is_nan());
}
