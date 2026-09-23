//! The two-track gene axis handed to the engine: which row is which gene on
//! which track. The engine sees only gene and track ids.

use chickpea::p2g::tracks::gene_tracks;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| Box::from(s)).collect()
}

#[test]
fn rows_map_to_their_gene_and_track() {
    let rows = names(&[
        "GENE1/rna",
        "GENE2/rna",
        "GENE3/rna",
        "GENE1/atac",
        "GENE2/atac",
    ]);
    let spec = gene_tracks(&rows).unwrap();
    assert_eq!(spec.track_of_row, [0, 0, 0, 1, 1]);
    assert_eq!(spec.gene_of_row, [0, 1, 2, 0, 1]);
    assert_eq!(spec.n_tracks(), 2);
    assert!(spec.tracks.iter().all(|t| t.is_count));
    spec.validate(rows.len()).unwrap();
}

#[test]
fn a_gene_without_cis_peaks_sits_on_the_base_track_only() {
    let rows = names(&["GENE1/rna", "GENE2/rna", "GENE1/atac"]);
    let spec = gene_tracks(&rows).unwrap();
    let rows_of_gene2: Vec<u32> = (0..rows.len())
        .filter(|&r| spec.gene_of_row[r] == 1)
        .map(|r| spec.track_of_row[r])
        .collect();
    assert_eq!(rows_of_gene2, [0]);
}

#[test]
fn row_order_does_not_matter() {
    let rows = names(&["GENE2/atac", "GENE1/rna", "GENE1/atac", "GENE2/rna"]);
    let spec = gene_tracks(&rows).unwrap();
    assert_eq!(spec.track_of_row, [1, 0, 1, 0]);
    let g = &spec.gene_of_row;
    assert_eq!(g[0], g[3], "both GENE2 rows share one gene id");
    assert_eq!(g[1], g[2], "both GENE1 rows share one gene id");
    assert_ne!(g[0], g[1]);
    spec.validate(rows.len()).unwrap();
}

#[test]
fn a_row_without_a_known_tag_is_refused() {
    assert!(gene_tracks(&names(&["GENE1/rna", "GENE1"])).is_err());
    assert!(gene_tracks(&names(&["GENE1/rna", "GENE1/other"])).is_err());
}

#[test]
fn a_gene_with_only_an_atac_row_is_refused() {
    let err = gene_tracks(&names(&["GENE1/rna", "GENE2/atac"])).unwrap_err();
    assert!(err.to_string().contains("GENE2"), "{err}");
}

#[test]
fn a_gene_twice_on_one_track_is_refused() {
    assert!(gene_tracks(&names(&["GENE1/rna", "GENE1/rna"])).is_err());
}
