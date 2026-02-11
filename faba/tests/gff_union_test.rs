use genomic_data::gff::*;

#[test]
fn test_union_gene_model() -> anyhow::Result<()> {
    let gff_file = "tests/chr19_small.gff";
    let gff_records = read_gff_record_vec(gff_file)?;

    println!("Total records read: {}", gff_records.len());

    // Separate protein_coding and non-coding genes
    let protein_coding_records: Vec<GffRecord> = gff_records
        .iter()
        .filter(|rec| rec.gene_type == GeneType::CodingGene)
        .cloned()
        .collect();

    let non_coding_records: Vec<GffRecord> = gff_records
        .iter()
        .filter(|rec| rec.gene_type != GeneType::CodingGene)
        .cloned()
        .collect();

    println!("Protein coding records: {}", protein_coding_records.len());
    println!("Non-coding records: {}", non_coding_records.len());

    // Count feature types for protein coding
    let mut feature_counts = std::collections::HashMap::new();
    for rec in &protein_coding_records {
        *feature_counts
            .entry(format!("{}", rec.feature_type))
            .or_insert(0) += 1;
    }
    println!("\n=== Feature Type Counts (Protein Coding) ===");
    for (ft, count) in &feature_counts {
        println!("  {}: {}", ft, count);
    }

    // Count feature types for non-coding
    let mut feature_counts_nc = std::collections::HashMap::new();
    for rec in &non_coding_records {
        *feature_counts_nc
            .entry(format!("{}", rec.feature_type))
            .or_insert(0) += 1;
    }
    if !non_coding_records.is_empty() {
        println!("\n=== Feature Type Counts (Non-Coding) ===");
        for (ft, count) in &feature_counts_nc {
            println!("  {}: {}", ft, count);
        }
    }

    // Process protein coding genes
    let UnionGeneModel {
        gene_boundaries,
        cds,
        five_prime_utr,
        three_prime_utr,
    } = build_union_gene_model(&protein_coding_records)?;

    println!("\n=== Union Gene Model Statistics (Protein Coding) ===");
    println!("Genes with boundaries: {}", gene_boundaries.len());
    println!("Genes with CDS: {}", cds.len());
    println!("Genes with 5'UTR: {}", five_prime_utr.len());
    println!("Genes with 3'UTR: {}", three_prime_utr.len());

    // Process non-coding genes
    if !non_coding_records.is_empty() {
        let UnionGeneModel {
            gene_boundaries: nc_gene_boundaries,
            cds: nc_cds,
            five_prime_utr: nc_five_prime_utr,
            three_prime_utr: nc_three_prime_utr,
        } = build_union_gene_model(&non_coding_records)?;

        println!("\n=== Union Gene Model Statistics (Non-Coding) ===");
        println!("Genes with boundaries: {}", nc_gene_boundaries.len());
        println!("Genes with CDS: {}", nc_cds.len());
        println!("Genes with 5'UTR: {}", nc_five_prime_utr.len());
        println!("Genes with 3'UTR: {}", nc_three_prime_utr.len());

        // Show example non-coding genes
        println!("\n=== Example Non-Coding Genes ===");
        let mut count = 0;
        for entry in nc_gene_boundaries.iter().take(5) {
            let gene_id = entry.key();
            let boundary_rec = entry.value();

            count += 1;
            println!(
                "\nGene #{}: {} ({:?})",
                count, gene_id, boundary_rec.gene_type
            );
            println!(
                "  Gene boundary: {}-{} ({} strand)",
                boundary_rec.start,
                boundary_rec.stop,
                if boundary_rec.strand == genomic_data::sam::Strand::Forward {
                    "+"
                } else {
                    "-"
                }
            );

            if let Some(cds_rec) = nc_cds.get(gene_id) {
                let cds_rec = cds_rec.value();
                println!(
                    "  CDS: {}-{} (length: {})",
                    cds_rec.start,
                    cds_rec.stop,
                    cds_rec.stop - cds_rec.start
                );
            } else {
                println!("  CDS: None");
            }

            if let Some(utr5_rec) = nc_five_prime_utr.get(gene_id) {
                let utr5_rec = utr5_rec.value();
                println!(
                    "  5'UTR: {}-{} (length: {})",
                    utr5_rec.start,
                    utr5_rec.stop,
                    utr5_rec.stop - utr5_rec.start
                );
            } else {
                println!("  5'UTR: None");
            }

            if let Some(utr3_rec) = nc_three_prime_utr.get(gene_id) {
                let utr3_rec = utr3_rec.value();
                println!(
                    "  3'UTR: {}-{} (length: {})",
                    utr3_rec.start,
                    utr3_rec.stop,
                    utr3_rec.stop - utr3_rec.start
                );
            } else {
                println!("  3'UTR: None");
            }
        }
    }

    // Find a gene that has all components
    println!("\n=== Example Protein-Coding Genes ===");
    let mut count = 0;
    for entry in gene_boundaries.iter().take(5) {
        let gene_id = entry.key();
        let boundary_rec = entry.value();

        count += 1;
        println!("\nGene #{}: {}", count, gene_id);
        println!(
            "  Gene boundary: {}-{} ({} strand)",
            boundary_rec.start,
            boundary_rec.stop,
            if boundary_rec.strand == genomic_data::sam::Strand::Forward {
                "+"
            } else {
                "-"
            }
        );

        if let Some(cds_rec) = cds.get(gene_id) {
            let cds_rec = cds_rec.value();
            println!(
                "  CDS: {}-{} (length: {})",
                cds_rec.start,
                cds_rec.stop,
                cds_rec.stop - cds_rec.start
            );
        } else {
            println!("  CDS: None");
        }

        if let Some(utr5_rec) = five_prime_utr.get(gene_id) {
            let utr5_rec = utr5_rec.value();
            println!(
                "  5'UTR: {}-{} (length: {})",
                utr5_rec.start,
                utr5_rec.stop,
                utr5_rec.stop - utr5_rec.start
            );
        } else {
            println!("  5'UTR: None");
        }

        if let Some(utr3_rec) = three_prime_utr.get(gene_id) {
            let utr3_rec = utr3_rec.value();
            println!(
                "  3'UTR: {}-{} (length: {})",
                utr3_rec.start,
                utr3_rec.stop,
                utr3_rec.stop - utr3_rec.start
            );
        } else {
            println!("  3'UTR: None");
        }
    }

    // Sanity checks
    println!("\n=== Sanity Checks ===");
    for entry in gene_boundaries.iter() {
        let gene_id = entry.key();
        let boundary_rec = entry.value();

        if let Some(cds_rec) = cds.get(gene_id) {
            let cds_rec = cds_rec.value();
            // CDS should be within gene boundaries
            assert!(
                cds_rec.start >= boundary_rec.start,
                "Gene {}: CDS start before gene boundary",
                gene_id
            );
            assert!(
                cds_rec.stop <= boundary_rec.stop,
                "Gene {}: CDS stop after gene boundary",
                gene_id
            );
        }

        if let Some(utr5_rec) = five_prime_utr.get(gene_id) {
            let utr5_rec = utr5_rec.value();
            // 5'UTR should overlap or be within gene boundaries
            assert!(
                utr5_rec.start >= boundary_rec.start || utr5_rec.stop <= boundary_rec.stop,
                "Gene {}: 5'UTR outside gene boundary",
                gene_id
            );
        }

        if let Some(utr3_rec) = three_prime_utr.get(gene_id) {
            let utr3_rec = utr3_rec.value();
            // 3'UTR should overlap or be within gene boundaries
            assert!(
                utr3_rec.start >= boundary_rec.start || utr3_rec.stop <= boundary_rec.stop,
                "Gene {}: 3'UTR outside gene boundary",
                gene_id
            );
        }
    }
    println!("All sanity checks passed!");

    Ok(())
}
