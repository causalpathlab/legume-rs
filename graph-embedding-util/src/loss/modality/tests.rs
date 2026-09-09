use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The point of the whole exercise: a protein positive draws proteins.
#[test]
fn negatives_stay_inside_the_positives_modality() {
    // 0,1,2 = genes; 3,4 = proteins.
    let of: Arc<[u32]> = Arc::from(vec![0u32, 0, 0, 1, 1]);
    let pools = ModalityPools::build(&of, &[0, 1, 2, 3, 4], &[10.0, 5.0, 1.0, 900.0, 800.0]);
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..200 {
        assert!(
            pools.draw_uniform(3, &mut rng).unwrap() >= 3,
            "protein drew a gene"
        );
        assert!(
            pools.draw_by_degree(4, &mut rng).unwrap() >= 3,
            "protein drew a gene"
        );
        assert!(
            pools.draw_uniform(0, &mut rng).unwrap() < 3,
            "gene drew a protein"
        );
        assert!(
            pools.draw_by_degree(2, &mut rng).unwrap() < 3,
            "gene drew a protein"
        );
    }
}

/// Degree weighting still applies, but only within the panel: the heavy
/// protein must dominate the protein draws without ever pulling in a gene.
#[test]
fn degree_weighting_is_relative_to_the_panel() {
    let of: Arc<[u32]> = Arc::from(vec![0u32, 0, 1, 1]);
    let pools = ModalityPools::build(&of, &[0, 1, 2, 3], &[1.0, 1.0, 999.0, 1.0]);
    let mut rng = StdRng::seed_from_u64(11);
    let heavy = (0..1000)
        .filter(|_| pools.draw_by_degree(3, &mut rng) == Some(2))
        .count();
    assert!(
        heavy > 900,
        "degree weighting lost inside the panel: {heavy}"
    );
}

/// A modality this sampler expresses only once has nothing to contrast with,
/// so the caller must be told to keep its own fallback rather than handed the
/// positive back as its own negative.
#[test]
fn a_singleton_modality_declines_the_draw() {
    let of: Arc<[u32]> = Arc::from(vec![0u32, 0, 0, 1]);
    let pools = ModalityPools::build(&of, &[0, 1, 2, 3], &[1.0, 1.0, 1.0, 1.0]);
    let mut rng = StdRng::seed_from_u64(3);
    assert!(pools.draw_uniform(3, &mut rng).is_none());
    assert!(pools.draw_by_degree(3, &mut rng).is_none());
    assert!(pools.draw_uniform(0, &mut rng).is_some());
}

/// Features the sampler does not express are absent from every panel, so a
/// negative can never name a feature with no counts on this axis.
#[test]
fn draws_never_leave_the_samplers_own_pool() {
    let of: Arc<[u32]> = Arc::from(vec![0u32, 0, 0, 1, 1, 1]);
    let pools = ModalityPools::build(&of, &[0, 2, 3, 5], &[1.0; 6]);
    let mut rng = StdRng::seed_from_u64(5);
    for _ in 0..200 {
        assert!(matches!(pools.draw_uniform(0, &mut rng), Some(0 | 2)));
        assert!(matches!(pools.draw_uniform(3, &mut rng), Some(3 | 5)));
    }
}
