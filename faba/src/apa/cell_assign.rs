use crate::apa::em::EmResult;
use crate::apa::fragment::FragmentRecord;
use crate::apa::utr_region::UtrRegion;
use fnv::FnvHashMap as HashMap;
use genomic_data::sam::CellBarcode;

/// A cell-level count at a specific pA site.
pub struct CellSiteCount {
    pub cell_barcode: CellBarcode,
    /// Site identifier, e.g. "ENSG_SYMBOL_chr17_7590910_7590950/pA"
    pub site_id: Box<str>,
    /// UMI-deduplicated count
    pub count: usize,
}

/// Metadata for one APA site (for Parquet annotation output).
#[derive(Clone)]
pub struct ApaSiteAnnotation {
    pub site_id: Box<str>,
    pub gene_name: Box<str>,
    pub chr: Box<str>,
    pub genomic_alpha: i64,
    pub beta: f32,
    pub genomic_start: i64,
    pub genomic_stop: i64,
    pub pi_weight: f32,
    pub expected_tail_length: f32,
}

/// Assign fragments to mixture components via hard assignment (argmax gamma),
/// then deduplicate by (cell, UMI, component) and count UMIs per cell per site.
pub fn assign_fragments_to_sites(
    fragments: &[FragmentRecord],
    em_result: &EmResult,
    utr: &UtrRegion,
) -> (Vec<CellSiteCount>, Vec<ApaSiteAnnotation>) {
    // Hard-assign each fragment to the argmax component (skip noise = k=0)
    // Key: (cell_barcode, component_idx, umi) -> deduplicate
    let mut cell_component_umis: HashMap<(CellBarcode, usize), HashMap<Box<str>, ()>> =
        HashMap::default();

    for (n, frag) in fragments.iter().enumerate() {
        let gamma = &em_result.gamma[n];

        // Find argmax component (including noise at k=0)
        let (best_k, _) = gamma
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Skip noise component (k=0)
        if best_k == 0 {
            continue;
        }

        let umi_str: Box<str> = frag.umi.to_string().into();
        let key = (frag.cell_barcode.clone(), best_k);

        cell_component_umis
            .entry(key)
            .or_default()
            .insert(umi_str, ());
    }

    // Build site_id for each active component
    // Format: GENE/pA/k (0-indexed component within gene)
    let make_site_id = |component_k: usize| -> (Box<str>, i64, i64) {
        let alpha = em_result.alphas[component_k - 1];
        let beta = em_result.betas[component_k - 1];
        let (gstart, gstop) = utr.alpha_to_genomic_range(alpha.into(), beta.into());
        let site_id: Box<str> = format!("{}/pA/{}", utr.name, component_k - 1).into();
        (site_id, gstart, gstop)
    };

    // Convert to CellSiteCount
    let mut results = Vec::new();

    for ((cell_barcode, component_k), umis) in &cell_component_umis {
        let (site_id, _, _) = make_site_id(*component_k);
        results.push(CellSiteCount {
            cell_barcode: cell_barcode.clone(),
            site_id,
            count: umis.len(),
        });
    }

    // Build annotations for all non-pruned components
    let mut annotations = Vec::new();
    for (k, &alpha) in em_result.alphas.iter().enumerate() {
        let weight = em_result.weights[k + 1]; // k+1 because weights[0] is noise
        if weight <= 0.0 {
            continue;
        }
        let beta = em_result.betas[k];
        let (gstart, gstop) = utr.alpha_to_genomic_range(alpha.into(), beta.into());
        let genomic_alpha = match utr.strand {
            genomic_data::sam::Strand::Forward => utr.start + alpha as i64,
            genomic_data::sam::Strand::Backward => utr.end - alpha as i64,
        };
        let site_id: Box<str> = format!("{}/pA/{}", utr.name, k).into();
        let expected_tail_length = utr.utr_length as f32 - alpha;
        annotations.push(ApaSiteAnnotation {
            site_id,
            gene_name: utr.name.clone(),
            chr: utr.chr.clone(),
            genomic_alpha,
            beta,
            genomic_start: gstart,
            genomic_stop: gstop,
            pi_weight: weight,
            expected_tail_length,
        });
    }

    (results, annotations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apa::em::EmResult;
    use genomic_data::sam::{CellBarcode, Strand, UmiBarcode};

    #[test]
    fn test_cell_assignment_deduplication() {
        // 5 cells, 10 fragments each = 50 total
        let mut fragments = Vec::new();
        let mut gamma = Vec::new();

        for cell_idx in 0..5u32 {
            let cb = CellBarcode::Barcode(format!("CELL{:04}", cell_idx).into());
            for frag_idx in 0..10u32 {
                // Give each fragment a UMI; some will share UMIs within a cell
                let umi_id = frag_idx % 5; // 0..4, so 2 fragments per UMI
                let umi = UmiBarcode::Barcode(format!("UMI{:04}", umi_id).into());

                fragments.push(FragmentRecord {
                    x: 100.0 + frag_idx as f32 * 10.0,
                    l: 50.0,
                    r: 0.0,
                    is_junction: false,
                    pa_site: None,
                    cell_barcode: cb.clone(),
                    umi,
                });

                // Assign: first 5 frags to component 1, next 5 to component 2
                // (noise = 0, comp1 = 1, comp2 = 2)
                if frag_idx < 5 {
                    gamma.push(vec![0.05, 0.9, 0.05]); // strongly assign to comp 1
                } else {
                    gamma.push(vec![0.05, 0.05, 0.9]); // strongly assign to comp 2
                }
            }
        }

        let em_result = EmResult {
            weights: vec![0.05, 0.5, 0.45],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            gamma,
            log_lik: -1000.0,
            bic: 2000.0,
            n_iter: 10,
        };

        let utr = UtrRegion {
            chr: "chr1".into(),
            start: 10000,
            end: 13000,
            strand: Strand::Forward,
            name: "TEST_GENE".into(),
            utr_length: 3000,
        };

        let (counts, annotations) = assign_fragments_to_sites(&fragments, &em_result, &utr);

        // Should have counts for 5 cells × 2 components = 10 entries
        assert_eq!(
            counts.len(),
            10,
            "5 cells × 2 sites = 10 entries, got {}",
            counts.len()
        );

        // UMI dedup: 5 fragments per cell per component, but only 5 unique UMIs (0..4)
        // Since frag_idx 0..4 -> comp1 with UMIs 0..4, and frag_idx 5..9 -> comp2 with UMIs 0..4
        // Each cell-component pair should have 5 unique UMIs
        for count in &counts {
            assert_eq!(
                count.count, 5,
                "each cell-component pair should have 5 unique UMIs after dedup, got {}",
                count.count
            );
        }

        // Should have 2 active site annotations
        assert_eq!(
            annotations.len(),
            2,
            "should have 2 site annotations, got {}",
            annotations.len()
        );
    }

    #[test]
    fn test_noise_exclusion() {
        // All fragments assigned to noise → no counts
        let mut fragments = Vec::new();
        let mut gamma = Vec::new();

        for i in 0..20u32 {
            fragments.push(FragmentRecord {
                x: 100.0,
                l: 50.0,
                r: 0.0,
                is_junction: false,
                pa_site: None,
                cell_barcode: CellBarcode::Barcode(format!("CELL{:04}", i % 3).into()),
                umi: UmiBarcode::Barcode(format!("UMI{:04}", i).into()),
            });
            gamma.push(vec![0.95, 0.025, 0.025]); // noise dominant
        }

        let em_result = EmResult {
            weights: vec![0.9, 0.05, 0.05],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            gamma,
            log_lik: -500.0,
            bic: 1000.0,
            n_iter: 5,
        };

        let utr = UtrRegion {
            chr: "chr1".into(),
            start: 10000,
            end: 13000,
            strand: Strand::Forward,
            name: "TEST_GENE".into(),
            utr_length: 3000,
        };

        let (counts, _) = assign_fragments_to_sites(&fragments, &em_result, &utr);
        assert!(
            counts.is_empty(),
            "all-noise fragments should produce no counts, got {}",
            counts.len()
        );
    }
}
