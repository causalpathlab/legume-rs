use super::*;

fn spec(track_of_row: &[u32], gene_of_row: &[u32], names: &[(&str, bool)]) -> TrackSpec {
    TrackSpec {
        track_of_row: track_of_row.to_vec(),
        gene_of_row: gene_of_row.to_vec(),
        tracks: names
            .iter()
            .map(|&(name, is_count)| TrackInfo {
                name: name.into(),
                is_count,
            })
            .collect(),
    }
}

#[test]
fn base_is_one_count_track_with_gene_equal_to_row() {
    let s = TrackSpec::base(5);
    assert_eq!(s.n_tracks(), 1);
    assert_eq!(s.n_genes(), 5);
    assert!(s.is_base());
    assert_eq!(s.track_of_row, vec![0; 5]);
    assert_eq!(s.gene_of_row, vec![0, 1, 2, 3, 4]);
    assert!(s.tracks[0].is_count);
    assert_eq!(s.count_tracks(), vec![0]);
    assert_eq!(s.rows_of_track(0), vec![0, 1, 2, 3, 4]);
    s.validate(5).unwrap();
}

#[test]
fn ragged_row_tables_are_rejected() {
    let short_track = spec(&[0, 0], &[0, 1, 2], &[("t0", true)]);
    assert!(short_track.validate(3).is_err());
    let short_gene = spec(&[0, 0, 0], &[0, 1], &[("t0", true)]);
    assert!(short_gene.validate(3).is_err());
    // right lengths, wrong n_features
    assert!(TrackSpec::base(3).validate(4).is_err());
}

#[test]
fn an_out_of_range_track_id_is_rejected() {
    assert!(spec(&[0, 1], &[0, 1], &[("t0", true)]).validate(2).is_err());
}

#[test]
fn a_gap_in_the_gene_ids_is_rejected() {
    // gene 1 is never used, so the ids are not dense 0..3
    assert!(spec(&[0, 0], &[0, 2], &[("t0", true)]).validate(2).is_err());
}

#[test]
fn an_empty_base_track_is_rejected() {
    let s = spec(&[1, 1], &[0, 1], &[("t0", true), ("t1", true)]);
    assert!(s.validate(2).is_err());
}

#[test]
fn a_base_track_that_is_not_a_count_track_is_rejected() {
    assert!(spec(&[0, 0], &[0, 1], &[("t0", false)])
        .validate(2)
        .is_err());
}

#[test]
fn a_gene_repeated_within_one_track_is_rejected() {
    assert!(spec(&[0, 0], &[0, 0], &[("t0", true)]).validate(2).is_err());
}

#[test]
fn rows_of_track_is_ascending_and_count_tracks_skips_the_non_count_ones() {
    // four tracks over two genes; rows interleaved so the row order is not the track order
    let s = spec(
        &[0, 1, 2, 3, 0, 1, 2, 3],
        &[0, 0, 0, 0, 1, 1, 1, 1],
        &[
            ("count/spliced", true),
            ("count/unspliced", true),
            ("m1/c0", false),
            ("m1/c1", false),
        ],
    );
    s.validate(8).unwrap();
    assert_eq!(s.n_tracks(), 4);
    assert_eq!(s.n_genes(), 2);
    assert!(!s.is_base());
    assert_eq!(s.rows_of_track(0), vec![0, 4]);
    assert_eq!(s.rows_of_track(3), vec![3, 7]);
    assert_eq!(s.count_tracks(), vec![0, 1]);
}

#[test]
fn a_one_track_spec_whose_gene_ids_are_not_the_identity_is_rejected() {
    // One track, dense gene ids, no repeat within the track — but permuted, so
    // `r` / `b_g` would come out in gene order while the caller reads them as
    // feature rows. `is_base` must not claim this is the plain gene axis.
    let permuted = spec(&[0, 0, 0], &[2, 0, 1], &[("t0", true)]);
    assert!(!permuted.is_base());
    assert!(permuted.validate(3).is_err());
    // the identity one-track spec is still the base axis
    let identity = spec(&[0, 0, 0], &[0, 1, 2], &[("t0", true)]);
    assert!(identity.is_base());
    identity.validate(3).unwrap();
    assert!(TrackSpec::base(3).is_base());
}
