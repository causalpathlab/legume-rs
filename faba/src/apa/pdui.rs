use crate::apa::cell_assign::{ApaSiteAnnotation, CellSiteCount};
use genomic_data::sam::{CellBarcode, Strand};
use rustc_hash::FxHashMap as HashMap;

/// Result of PDUI computation for one gene/UTR
pub struct PduiResult {
    /// Gene/UTR name
    pub gene_name: Box<str>,
    /// Per-cell PDUI values
    pub cell_pdui: Vec<(CellBarcode, f32)>,
    /// Proximal site annotation
    pub proximal: ApaSiteAnnotation,
    /// Distal site annotation
    pub distal: ApaSiteAnnotation,
}

/// Compute PDUI for a single gene given its cell-site counts and site annotations.
///
/// Returns None if the gene doesn't have exactly 2 active sites.
pub fn compute_pdui(
    counts: &[CellSiteCount],
    annotations: &[ApaSiteAnnotation],
    strand: Strand,
) -> Option<PduiResult> {
    if annotations.len() != 2 {
        return None;
    }

    // Identify proximal vs distal by genomic_alpha + strand
    let (proximal_idx, distal_idx) = match strand {
        // Forward: higher alpha = more distal (further into 3'UTR)
        Strand::Forward => {
            if annotations[0].genomic_alpha < annotations[1].genomic_alpha {
                (0, 1)
            } else {
                (1, 0)
            }
        }
        // Backward: lower alpha = more distal
        Strand::Backward => {
            if annotations[0].genomic_alpha > annotations[1].genomic_alpha {
                (0, 1)
            } else {
                (1, 0)
            }
        }
    };

    let proximal = annotations[proximal_idx].clone();
    let distal = annotations[distal_idx].clone();

    // Build per-cell counts for each site
    let mut proximal_counts: HashMap<CellBarcode, usize> = HashMap::default();
    let mut distal_counts: HashMap<CellBarcode, usize> = HashMap::default();

    for c in counts {
        if c.site_id == proximal.site_id {
            *proximal_counts.entry(c.cell_barcode.clone()).or_default() += c.count;
        } else if c.site_id == distal.site_id {
            *distal_counts.entry(c.cell_barcode.clone()).or_default() += c.count;
        }
    }

    // Compute PDUI per cell (for cells with any count at either site)
    let mut all_cells: rustc_hash::FxHashSet<CellBarcode> = rustc_hash::FxHashSet::default();
    all_cells.extend(proximal_counts.keys().cloned());
    all_cells.extend(distal_counts.keys().cloned());

    let mut cell_pdui = Vec::with_capacity(all_cells.len());
    for cell in all_cells {
        let p = *proximal_counts.get(&cell).unwrap_or(&0) as f32;
        let d = *distal_counts.get(&cell).unwrap_or(&0) as f32;
        let total = p + d;
        if total > 0.0 {
            cell_pdui.push((cell, d / total));
        }
    }

    Some(PduiResult {
        gene_name: annotations[0].gene_name.clone(),
        cell_pdui,
        proximal,
        distal,
    })
}

#[cfg(test)]
mod tests {
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
        }
    }

    fn make_count(cell: &str, site_id: &str, count: usize) -> CellSiteCount {
        CellSiteCount {
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

        let counts = vec![
            make_count("CELL1", "site_A", 3), // proximal
            make_count("CELL1", "site_B", 7), // distal
            make_count("CELL2", "site_A", 5),
            make_count("CELL2", "site_B", 5),
        ];

        let result = compute_pdui(&counts, &annotations, Strand::Forward).unwrap();

        // Find CELL1 and CELL2 PDUIs
        let cell1_pdui = result
            .cell_pdui
            .iter()
            .find(|(cb, _)| matches!(cb, CellBarcode::Barcode(b) if b.as_ref() == "CELL1"))
            .map(|(_, v)| *v)
            .unwrap();
        let cell2_pdui = result
            .cell_pdui
            .iter()
            .find(|(cb, _)| matches!(cb, CellBarcode::Barcode(b) if b.as_ref() == "CELL2"))
            .map(|(_, v)| *v)
            .unwrap();

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

        let counts = vec![
            make_count("CELL1", "site_A", 2),
            make_count("CELL1", "site_B", 8),
        ];

        let result = compute_pdui(&counts, &annotations, Strand::Backward).unwrap();
        let pdui = result.cell_pdui[0].1;
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

        let counts = vec![
            make_count("CELL1", "site_B", 10), // only distal
        ];

        let result = compute_pdui(&counts, &annotations, Strand::Forward).unwrap();
        let pdui = result.cell_pdui[0].1;
        assert!(
            (pdui - 1.0).abs() < 0.01,
            "PDUI should be 1.0 for distal-only, got {}",
            pdui
        );
    }
}
