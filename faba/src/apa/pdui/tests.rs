use super::*;
use genomic_data::sam::CellBarcode;

fn make_annotation(site_id: &str, gene: &str, alpha: i64, weight: f32) -> ApaSiteAnnotation {
    ApaSiteAnnotation {
        site_id: site_id.into(),
        gene_name: gene.into(),
        chr: "chr1".into(),
        genomic_alpha: alpha,
        beta: 30.0_f32,
        genomic_start: alpha - 50,
        genomic_stop: alpha + 50,
        pi_weight: weight,
        expected_tail_length: 100.0_f32,
        utr_length: 1000,
    }
}

fn make_count(cell: &str, site_id: &str, count: usize) -> CellSiteCount {
    CellSiteCount {
        batch: 0,
        cell_barcode: CellBarcode::Barcode(cell.into()),
        site_id: site_id.into(),
        count,
    }
}

#[test]
fn test_pdui_forward_strand() {
    let annotations = vec![
        make_annotation("site_A", "GENE1", 500, 0.5), // proximal (lower alpha on fwd)
        make_annotation("site_B", "GENE1", 1500, 0.5), // distal (higher alpha on fwd)
    ];

    let counts = [
        make_count("CELL1", "site_A", 3), // proximal
        make_count("CELL1", "site_B", 7), // distal
        make_count("CELL2", "site_A", 5),
        make_count("CELL2", "site_B", 5),
    ];

    let result = compute_pdui(
        &counts.iter().collect::<Vec<_>>(),
        &annotations,
        Strand::Forward,
    )
    .unwrap();

    // PDUI = distal / (proximal + distal), derived from cell_counts.
    let pdui_of = |name: &str| {
        result
            .cell_counts
            .iter()
            .find(|(cb, _, _)| matches!(cb, CellBarcode::Barcode(b) if b.as_ref() == name))
            .map(|(_, p, d)| *d as f32 / (*p + *d) as f32)
            .unwrap()
    };
    let cell1_pdui = pdui_of("CELL1");
    let cell2_pdui = pdui_of("CELL2");

    assert!(
        (cell1_pdui - 0.7).abs() < 0.01,
        "CELL1 PDUI should be 0.7, got {}",
        cell1_pdui
    );
    assert!(
        (cell2_pdui - 0.5).abs() < 0.01,
        "CELL2 PDUI should be 0.5, got {}",
        cell2_pdui
    );
}

#[test]
fn test_pdui_backward_strand() {
    let annotations = vec![
        make_annotation("site_A", "GENE1", 1500, 0.5), // proximal (higher alpha on rev)
        make_annotation("site_B", "GENE1", 500, 0.5),  // distal (lower alpha on rev)
    ];

    let counts = [
        make_count("CELL1", "site_A", 2),
        make_count("CELL1", "site_B", 8),
    ];

    let result = compute_pdui(
        &counts.iter().collect::<Vec<_>>(),
        &annotations,
        Strand::Backward,
    )
    .unwrap();
    let (_, p, d) = result.cell_counts[0];
    let pdui = d as f32 / (p + d) as f32;
    assert!(
        (pdui - 0.8).abs() < 0.01,
        "PDUI should be 0.8, got {}",
        pdui
    );
}

#[test]
fn test_pdui_requires_two_sites() {
    let annotations = vec![make_annotation("site_A", "GENE1", 500, 1.0)];
    assert!(compute_pdui(&[], &annotations, Strand::Forward).is_none());
}

#[test]
fn test_pdui_cell_with_only_one_site() {
    let annotations = vec![
        make_annotation("site_A", "GENE1", 500, 0.5),
        make_annotation("site_B", "GENE1", 1500, 0.5),
    ];

    let counts = [
        make_count("CELL1", "site_B", 10), // only distal
    ];

    let result = compute_pdui(
        &counts.iter().collect::<Vec<_>>(),
        &annotations,
        Strand::Forward,
    )
    .unwrap();
    let (_, p, d) = result.cell_counts[0];
    let pdui = d as f32 / (p + d) as f32;
    assert!(
        (pdui - 1.0).abs() < 0.01,
        "PDUI should be 1.0 for distal-only, got {}",
        pdui
    );
}
