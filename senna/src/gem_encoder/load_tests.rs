use super::*;
use crate::gem::rows::build_gene_track_map;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::<str>::from(*s)).collect()
}

/// `per_gene_rows` is what the null gather indexes through, so a gene with only
/// one track must report `None` for the other rather than silently aliasing.
#[test]
fn per_gene_rows_reports_missing_tracks() {
    let rows = names(&["A/count/spliced", "A/count/unspliced", "C/count/spliced"]);
    let (map, _) = build_gene_track_map(&rows);
    let (nascent, mature) = map.per_gene_rows();

    assert_eq!(nascent, vec![Some(1), None]);
    assert_eq!(mature, vec![Some(0), Some(2)]);
}

/// Gene weights take the MAX across a gene's tracks, not the mean: a gene whose
/// mature track is informative should still be selected when its nascent track
/// is shallow — which is the common case, nascent being the sparse side.
#[test]
fn gene_weights_take_the_max_across_tracks() {
    let rows = names(&["A/count/spliced", "A/count/unspliced", "B/count/spliced"]);
    let (map, _) = build_gene_track_map(&rows);
    let w = per_gene_weights(&[0.9, 0.01, 0.4], &map);
    assert!((w[0] - 0.9).abs() < 1e-6, "A should take its mature weight");
    assert!((w[1] - 0.4).abs() < 1e-6);
}

/// Track means must be computed from that track's rows only — mixing them would
/// hand the encoder the wrong divisor and blur the very contrast the model is
/// built to measure.
#[test]
fn per_gene_track_mean_separates_the_tracks() {
    let rows = names(&["A/count/spliced", "A/count/unspliced", "B/count/spliced"]);
    let (map, _) = build_gene_track_map(&rows);

    // rows × 2 pseudobulks
    let mu = Mat::from_row_slice(3, 2, &[10.0, 20.0, 2.0, 4.0, 6.0, 8.0]);

    let mature = per_gene_track_mean(&mu, &map, false);
    let nascent = per_gene_track_mean(&mu, &map, true);

    assert!((mature[0] - 15.0).abs() < 1e-5, "A mature = mean(10, 20)");
    assert!((nascent[0] - 3.0).abs() < 1e-5, "A nascent = mean(2, 4)");
    assert!((mature[1] - 7.0).abs() < 1e-5, "B mature = mean(6, 8)");
    assert!(
        (nascent[1] - 0.0).abs() < 1e-5,
        "B has no nascent row, so its nascent mean is 0"
    );
}
