use genomic_data::gff::*;

#[test]
fn test_ensg00000214046() -> anyhow::Result<()> {
    let gff_file = "tests/chr19_small.gff";
    let gff_records = read_gff_record_vec(gff_file)?;

    println!("Total GFF records read: {}", gff_records.len());

    // Filter for ENSG00000214046 only
    let gene_records: Vec<GffRecord> = gff_records
        .into_iter()
        .filter(|rec| {
            let gene_id_str: Box<str> = rec.gene_id.clone().into();
            gene_id_str.starts_with("ENSG00000214046")
        })
        .collect();

    println!("Total records for ENSG00000214046: {}", gene_records.len());

    for rec in &gene_records {
        println!(
            "  Feature: {:?}, {}-{}",
            rec.feature_type, rec.start, rec.stop
        );
    }

    let UnionGeneModel {
        gene_boundaries,
        cds,
        five_prime_utr,
        three_prime_utr,
    } = build_union_gene_model(&gene_records)?;

    println!("\nGene boundaries count: {}", gene_boundaries.len());
    for entry in gene_boundaries.iter() {
        println!("  Gene ID in map: {}", entry.key());
    }

    let gene_id = GeneId::Ensembl("ENSG00000214046".into());
    println!("Looking for gene_id: {}", gene_id);

    if let Some(boundary) = gene_boundaries.get(&gene_id) {
        let boundary = boundary.value();
        println!(
            "\nGene boundary: {}-{} (strand: {:?})",
            boundary.start, boundary.stop, boundary.strand
        );
    }

    if let Some(cds_rec) = cds.get(&gene_id) {
        let cds_rec = cds_rec.value();
        println!(
            "CDS: {}-{} (length: {})",
            cds_rec.start,
            cds_rec.stop,
            cds_rec.stop - cds_rec.start
        );
    } else {
        println!("CDS: None");
    }

    if let Some(utr5) = five_prime_utr.get(&gene_id) {
        let utr5 = utr5.value();
        println!(
            "5'UTR: {}-{} (length: {})",
            utr5.start,
            utr5.stop,
            utr5.stop - utr5.start
        );
    } else {
        println!("5'UTR: None");
    }

    if let Some(utr3) = three_prime_utr.get(&gene_id) {
        let utr3 = utr3.value();
        println!(
            "3'UTR: {}-{} (length: {})",
            utr3.start,
            utr3.stop,
            utr3.stop - utr3.start
        );
    } else {
        println!("3'UTR: None");
    }

    // Let's check the start and stop codons
    println!("\nStart codons:");
    for rec in gene_records
        .iter()
        .filter(|r| r.feature_type == FeatureType::StartCodon)
    {
        println!("  {}: {}-{}", rec.gene_name, rec.start, rec.stop);
    }

    println!("\nStop codons:");
    for rec in gene_records
        .iter()
        .filter(|r| r.feature_type == FeatureType::StopCodon)
    {
        println!("  {}: {}-{}", rec.gene_name, rec.start, rec.stop);
    }

    println!("\nCDS regions:");
    for rec in gene_records
        .iter()
        .filter(|r| r.feature_type == FeatureType::CDS)
    {
        println!("  {}: {}-{}", rec.gene_name, rec.start, rec.stop);
    }

    println!("\n5' UTRs:");
    for rec in gene_records
        .iter()
        .filter(|r| r.feature_type == FeatureType::FivePrimeUTR)
    {
        println!("  {}: {}-{}", rec.gene_name, rec.start, rec.stop);
    }

    println!("\n3' UTRs:");
    for rec in gene_records
        .iter()
        .filter(|r| r.feature_type == FeatureType::ThreePrimeUTR)
    {
        println!("  {}: {}-{}", rec.gene_name, rec.start, rec.stop);
    }

    Ok(())
}
