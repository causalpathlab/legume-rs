use super::*;
use crate::apa::em::EmResult;
use genomic_data::sam::{CellBarcode, Strand, UmiBarcode};

#[test]
fn test_cell_assignment_deduplication() {
    // 5 cells, 10 fragments each = 50 total
    let mut fragments = Vec::new();
    let mut gamma: Vec<f32> = Vec::new();

    for cell_idx in 0..5u32 {
        let cb = CellBarcode::Barcode(format!("CELL{:04}", cell_idx).into());
        for frag_idx in 0..10u32 {
            // Give each fragment a UMI; some will share UMIs within a cell
            let umi_id = frag_idx % 5; // 0..4, so 2 fragments per UMI
            let umi = UmiBarcode::Hash(umi_id as u64);

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
            // (noise = 0, comp1 = 1, comp2 = 2). gamma is flat row-major
            // with stride 3.
            if frag_idx < 5 {
                gamma.extend_from_slice(&[0.05, 0.9, 0.05]);
            } else {
                gamma.extend_from_slice(&[0.05, 0.05, 0.9]);
            }
        }
    }

    let em_result = EmResult {
        weights: vec![0.05, 0.5, 0.45],
        alphas: vec![500.0, 1500.0],
        betas: vec![30.0, 30.0],
        gamma,
        n_components: 3,
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

    // Each test fragment is its own cluster (no real coarsening here).
    let cluster_idx: Vec<u32> = (0..fragments.len() as u32).collect();
    let frag_batch: Vec<u32> = vec![0; fragments.len()];
    let (counts, annotations) =
        assign_fragments_to_sites(&fragments, &cluster_idx, &frag_batch, &em_result, &utr);

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
    let mut gamma: Vec<f32> = Vec::new();

    for i in 0..20u32 {
        fragments.push(FragmentRecord {
            x: 100.0,
            l: 50.0,
            r: 0.0,
            is_junction: false,
            pa_site: None,
            cell_barcode: CellBarcode::Barcode(format!("CELL{:04}", i % 3).into()),
            umi: UmiBarcode::Hash(i as u64),
        });
        gamma.extend_from_slice(&[0.95, 0.025, 0.025]); // noise dominant
    }

    let em_result = EmResult {
        weights: vec![0.9, 0.05, 0.05],
        alphas: vec![500.0, 1500.0],
        betas: vec![30.0, 30.0],
        gamma,
        n_components: 3,
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

    let cluster_idx: Vec<u32> = (0..fragments.len() as u32).collect();
    let frag_batch: Vec<u32> = vec![0; fragments.len()];
    let (counts, _) =
        assign_fragments_to_sites(&fragments, &cluster_idx, &frag_batch, &em_result, &utr);
    assert!(
        counts.is_empty(),
        "all-noise fragments should produce no counts, got {}",
        counts.len()
    );
}
