use super::{assign_tracks, contrast_channels, encoder_suffix_for};

fn names(rows: &[&str]) -> Vec<Box<str>> {
    rows.iter().map(|&s| s.into()).collect()
}

#[test]
fn ids_and_flags_on_a_mixed_axis() {
    let axis = names(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
        "GENE1/m6a/methylated",
        "GENE1/m6a/unmethylated",
        "GENE2/apa/proximal",
        "GENE2/apa/distal",
    ]);
    let plan = assign_tracks(&axis).expect("assign_tracks");

    // 6 tracks: count/spliced, count/unspliced, then every other
    // (modality, channel) pair sorted ascending — "apa" < "m6a", and within
    // apa "distal" < "proximal".
    assert_eq!(plan.tracks.len(), 6);
    assert_eq!(
        (&*plan.tracks[0].modality, &*plan.tracks[0].channel),
        ("count", "spliced")
    );
    assert!(plan.tracks[0].is_count);
    assert_eq!(
        (&*plan.tracks[1].modality, &*plan.tracks[1].channel),
        ("count", "unspliced")
    );
    assert!(plan.tracks[1].is_count);
    assert_eq!(
        (&*plan.tracks[2].modality, &*plan.tracks[2].channel),
        ("apa", "distal")
    );
    assert!(!plan.tracks[2].is_count);
    assert_eq!(
        (&*plan.tracks[3].modality, &*plan.tracks[3].channel),
        ("apa", "proximal")
    );
    assert!(!plan.tracks[3].is_count);
    assert_eq!(
        (&*plan.tracks[4].modality, &*plan.tracks[4].channel),
        ("m6a", "methylated")
    );
    assert!(!plan.tracks[4].is_count);
    assert_eq!(
        (&*plan.tracks[5].modality, &*plan.tracks[5].channel),
        ("m6a", "unmethylated")
    );
    assert!(!plan.tracks[5].is_count);

    // Row -> track / gene / base.
    assert_eq!(plan.row_track, vec![0, 1, 0, 4, 5, 3, 2]);
    // Only track 0 (count/spliced) is base; count/unspliced (track 1) is not.
    assert_eq!(
        plan.base_rows,
        vec![true, false, true, false, false, false, false]
    );
    assert_eq!(plan.gene_names, names(&["GENE1", "GENE2"]));
    assert_eq!(plan.row_gene, vec![0, 0, 1, 0, 0, 1, 1]);

    // track_of / rows_of round-trip.
    assert_eq!(plan.track_of("count", "spliced"), Some(0));
    assert_eq!(plan.track_of("count", "unspliced"), Some(1));
    assert_eq!(plan.track_of("apa", "distal"), Some(2));
    assert_eq!(plan.track_of("apa", "proximal"), Some(3));
    assert_eq!(plan.track_of("m6a", "methylated"), Some(4));
    assert_eq!(plan.track_of("m6a", "unmethylated"), Some(5));
    assert_eq!(plan.track_of("atoi", "edited"), None);
    assert_eq!(plan.rows_of(0), vec![0, 2]);
    assert_eq!(plan.rows_of(1), vec![1]);
    assert_eq!(plan.rows_of(2), vec![6]);
    assert_eq!(plan.rows_of(3), vec![5]);
    assert_eq!(plan.rows_of(4), vec![3]);
    assert_eq!(plan.rows_of(5), vec![4]);
}

#[test]
fn count_total_is_an_error_naming_the_row() {
    let axis = names(&["GENE1/count/spliced", "GENE1/count/total"]);
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("count"), "{err}");
    assert!(err.contains("GENE1/count/total"), "{err}");
    assert!(err.contains('1'), "{err}"); // total count = 1
}

#[test]
fn a_subunit_row_errors() {
    let axis = names(&["GENE1/count/spliced", "GENE1/m6a/site1/methylated"]);
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("subunit"), "{err}");
    assert!(err.contains("GENE1/m6a/site1/methylated"), "{err}");
}

#[test]
fn a_snp_row_errors() {
    let axis = names(&["GENE1/count/spliced", "GENE1/snp/something"]);
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("modality"), "{err}");
    assert!(err.contains("GENE1/snp/something"), "{err}");
}

#[test]
fn a_row_that_does_not_parse_errors() {
    let axis = names(&["GENE1/count/spliced", "not_a_feature_row"]);
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("parse"), "{err}");
    assert!(err.contains("not_a_feature_row"), "{err}");
}

#[test]
fn missing_base_track_errors() {
    let axis = names(&[
        "GENE1/count/unspliced",
        "GENE1/m6a/methylated",
        "GENE1/m6a/unmethylated",
    ]);
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("base"), "{err}");
}

#[test]
fn more_than_ten_offending_rows_are_capped_in_the_message() {
    let mut rows: Vec<String> = vec!["GENE1/count/spliced".to_string()];
    for i in 0..15 {
        rows.push(format!("bad{i}"));
    }
    let axis: Vec<Box<str>> = rows.iter().map(|s| s.as_str().into()).collect();
    let err = assign_tracks(&axis).unwrap_err().to_string();
    assert!(err.contains("15"), "{err}");
    assert!(err.contains("+5 more"), "{err}");
}

#[test]
fn encoder_suffix_names_track_zero_bare_and_others_namespaced() {
    assert_eq!(
        encoder_suffix_for(0, "count/spliced"),
        "cell_encoder.safetensors"
    );
    assert_eq!(
        encoder_suffix_for(1, "m6a/methylated"),
        "cell_encoder.m6a.methylated.safetensors"
    );
}

#[test]
fn contrast_channels_are_fixed_per_modality() {
    assert_eq!(contrast_channels("count"), Some(("unspliced", "spliced")));
    assert_eq!(
        contrast_channels("m6a"),
        Some(("methylated", "unmethylated"))
    );
    assert_eq!(contrast_channels("atoi"), Some(("edited", "unedited")));
    assert_eq!(contrast_channels("apa"), Some(("proximal", "distal")));
    assert_eq!(contrast_channels("snp"), None);
}

#[test]
fn to_ge_validates() {
    let axis = names(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
        "GENE1/m6a/methylated",
        "GENE1/m6a/unmethylated",
    ]);
    let plan = assign_tracks(&axis).expect("assign_tracks");
    let spec = plan.to_ge();
    spec.validate(axis.len()).expect("TrackSpec::validate");
    assert_eq!(spec.tracks.len(), plan.tracks.len());
    assert!(spec.tracks[0].is_count);
}
