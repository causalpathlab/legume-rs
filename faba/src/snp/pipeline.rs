use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::data::dna::Dna;
use crate::data::dna_stat_map::DnaBaseFreqMap;
use crate::data::util_htslib::{fetch_reference_base, load_fasta_index};
use crate::pipeline_util::create_gene_key_function;
use crate::snp::genotyper::{find_top_alt_allele, genotype_site, GenotypeParams, SiteInput};
use crate::snp::io::{
    build_snp_mask, load_contigs_from_fai, write_snp_sites_parquet, write_snp_sites_vcf, KnownSnps,
};
use crate::snp::{SnpGenotype, SnpSite};

use dashmap::DashMap;
use genomic_data::bed::{Bed, BedWithGene};
use genomic_data::gff::{GeneId, GffRecordMap};
use log::info;
use rustc_hash::FxHashSet;
use std::sync::{Arc, Mutex};

/// Parameters for SNP genotyping pipeline
pub struct SnpParams {
    pub bam_files: Vec<Box<str>>,
    pub genome_file: Box<str>,
    pub cell_barcode_tag: Box<str>,
    pub gene_barcode_tag: Box<str>,
    pub include_missing_barcode: bool,
    pub min_base_quality: u8,
    pub min_mapping_quality: u8,
    pub genotype_params: GenotypeParams,
    pub backend: SparseIoBackend,
    pub output: Box<str>,
    pub bulk: bool,
    /// UMI tag for deduplication (e.g., "UB"). None = no UMI dedup.
    pub umi_tag: Option<Box<str>>,
    /// Use per-base quality model (Li 2011) instead of constant error rate.
    pub use_base_quality: bool,
    /// Minimum VAF for a site to enter the SNP mask. Sites with lower VAF
    /// (likely RNA editing, not germline SNPs) are excluded. None = no filter.
    pub min_vaf: Option<f32>,
}

impl SnpParams {
    pub fn backend_file_path(&self, batch_name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, batch_name),
        }
        .into_boxed_str()
    }

    /// Create a marginal DnaBaseFreqMap configured with this pipeline's thresholds,
    /// UMI dedup, and quality accumulation settings.
    pub fn new_freq_map(&self) -> DnaBaseFreqMap<'_> {
        let mut m = DnaBaseFreqMap::new();
        m.set_quality_thresholds(self.min_base_quality, self.min_mapping_quality);
        if let Some(ref tag) = self.umi_tag {
            m.set_umi_tag(tag);
        }
        m.set_use_base_quality(self.use_base_quality);
        m
    }

    /// Create a per-cell DnaBaseFreqMap configured with this pipeline's thresholds
    /// and UMI dedup settings.
    pub fn new_freq_map_percell<'a>(
        &self,
        cell_membership: Option<&'a CellMembership>,
    ) -> DnaBaseFreqMap<'a> {
        let mut m = DnaBaseFreqMap::new_with_cell_barcode(&self.cell_barcode_tag, cell_membership);
        m.set_quality_thresholds(self.min_base_quality, self.min_mapping_quality);
        if let Some(ref tag) = self.umi_tag {
            m.set_umi_tag(tag);
        }
        m
    }
}

/// Result of gene-centric SNP genotyping: flat sites + gene-keyed map for pass 2.
pub struct GenePileupResult {
    pub sites: Vec<SnpSite>,
    pub gene_sites: DashMap<GeneId, Vec<SnpSite>>,
}

// ============================================================
// KNOWN-SITE GENOTYPING (pileup at VCF positions)
// ============================================================

/// Gene-centric genotyping at known SNP sites.
pub fn pileup_known_snps_by_gene(
    gff_map: &GffRecordMap,
    known_snps: &KnownSnps,
    params: &SnpParams,
) -> anyhow::Result<GenePileupResult> {
    let records = gff_map.records();
    let njobs = records.len() as u64;
    info!("Genotyping known SNPs across {} genes", njobs);

    let arc_sites = Arc::new(Mutex::new(Vec::<SnpSite>::new()));
    let gene_site_map = Arc::new(DashMap::<GeneId, Vec<SnpSite>>::default());

    records
        .par_iter()
        .progress_count(njobs)
        .try_for_each(|rec| -> anyhow::Result<()> {
            let chr = rec.seqname.as_ref();

            let chr_snps = match known_snps.by_chr.get(chr) {
                Some(m) => m,
                None => return Ok(()),
            };

            let gene_lb = (rec.start - 1).max(0);
            let gene_ub = rec.stop;
            let gene_positions: FxHashSet<i64> = chr_snps
                .keys()
                .filter(|&&p| p >= gene_lb && p < gene_ub)
                .copied()
                .collect();

            if gene_positions.is_empty() {
                return Ok(());
            }

            let mut freq_map = params.new_freq_map();
            freq_map.set_position_filter(gene_positions.clone());

            for bam_file in &params.bam_files {
                freq_map.update_from_gene(
                    bam_file,
                    rec,
                    &params.gene_barcode_tag,
                    params.include_missing_barcode,
                )?;
            }

            let freq = freq_map
                .marginal_frequency_map()
                .ok_or_else(|| anyhow::anyhow!("expected marginal frequency map"))?;
            let qual_map = freq_map.quality_map();

            let mut local_sites = Vec::new();
            for &pos in &gene_positions {
                if let Some(&(ref_allele, alt_allele, ref rsid)) = chr_snps.get(&pos) {
                    let counts = freq.get(&pos).cloned().unwrap_or_default();
                    let qual = qual_map.get(&pos);
                    let site = genotype_site(
                        SiteInput {
                            chr: chr.into(),
                            pos,
                            ref_allele,
                            alt_allele,
                            rsid: rsid.clone(),
                            counts,
                            qual: qual.cloned(),
                        },
                        &params.genotype_params,
                    );
                    local_sites.push(site);
                }
            }

            if !local_sites.is_empty() {
                gene_site_map
                    .entry(rec.gene_id.clone())
                    .or_default()
                    .extend(local_sites.clone());
                arc_sites.lock().expect("lock").extend(local_sites);
            }

            Ok(())
        })?;

    let mut sites = Arc::try_unwrap(arc_sites)
        .map_err(|_| anyhow::anyhow!("failed to release sites"))?
        .into_inner()?;

    sites.sort_by(|a, b| a.chr.cmp(&b.chr).then(a.pos.cmp(&b.pos)));
    sites.dedup_by(|a, b| a.chr == b.chr && a.pos == b.pos);

    let gene_sites = Arc::try_unwrap(gene_site_map)
        .map_err(|_| anyhow::anyhow!("failed to release gene_site_map"))?;

    Ok(GenePileupResult { sites, gene_sites })
}

/// Region-centric genotyping at known SNP sites (no GFF).
pub fn pileup_known_snps_by_region(
    known_snps: &KnownSnps,
    params: &SnpParams,
) -> anyhow::Result<Vec<SnpSite>> {
    let chromosomes = known_snps.chromosomes();
    let njobs = chromosomes.len() as u64;
    info!(
        "Genotyping known SNPs across {} chromosomes (region mode)",
        njobs
    );

    let arc_sites = Arc::new(Mutex::new(Vec::<SnpSite>::new()));

    chromosomes
        .par_iter()
        .progress_count(njobs)
        .try_for_each(|chr| -> anyhow::Result<()> {
            let chr_snps = match known_snps.by_chr.get(chr.as_ref()) {
                Some(m) => m,
                None => return Ok(()),
            };

            if chr_snps.is_empty() {
                return Ok(());
            }

            let positions: FxHashSet<i64> = chr_snps.keys().copied().collect();
            let min_pos = positions.iter().copied().min().unwrap_or(0);
            let max_pos = positions.iter().copied().max().unwrap_or(0);

            let bed = Bed {
                chr: chr.clone(),
                start: min_pos,
                stop: max_pos + 1,
            };

            let mut freq_map = params.new_freq_map();
            freq_map.set_position_filter(positions.clone());

            for bam_file in &params.bam_files {
                freq_map.update_from_region(bam_file, &bed)?;
            }

            let freq = freq_map
                .marginal_frequency_map()
                .ok_or_else(|| anyhow::anyhow!("expected marginal frequency map"))?;
            let qual_map = freq_map.quality_map();

            let mut local_sites = Vec::new();
            for (&pos, &(ref_allele, alt_allele, ref rsid)) in chr_snps.iter() {
                let counts = freq.get(&pos).cloned().unwrap_or_default();
                let qual = qual_map.get(&pos);
                let site = genotype_site(
                    SiteInput {
                        chr: chr.clone(),
                        pos,
                        ref_allele,
                        alt_allele,
                        rsid: rsid.clone(),
                        counts,
                        qual: qual.cloned(),
                    },
                    &params.genotype_params,
                );
                local_sites.push(site);
            }

            if !local_sites.is_empty() {
                arc_sites.lock().expect("lock").extend(local_sites);
            }

            Ok(())
        })?;

    let mut sites = Arc::try_unwrap(arc_sites)
        .map_err(|_| anyhow::anyhow!("failed to release sites"))?
        .into_inner()?;

    sites.sort_by(|a, b| a.chr.cmp(&b.chr).then(a.pos.cmp(&b.pos)));
    Ok(sites)
}

// ============================================================
// DE NOVO SNP DISCOVERY (compare reads to reference genome)
// ============================================================

/// Gene-centric de novo SNP discovery: pileup all positions, compare to reference.
pub fn discover_snps_by_gene(
    gff_map: &GffRecordMap,
    params: &SnpParams,
) -> anyhow::Result<GenePileupResult> {
    let records = gff_map.records();
    let njobs = records.len() as u64;
    info!("Discovering SNPs de novo across {} genes", njobs);

    // Validate reference genome upfront
    load_fasta_index(&params.genome_file)?;

    let arc_sites = Arc::new(Mutex::new(Vec::<SnpSite>::new()));
    let gene_site_map = Arc::new(DashMap::<GeneId, Vec<SnpSite>>::default());

    records
        .par_iter()
        .progress_count(njobs)
        .try_for_each(|rec| -> anyhow::Result<()> {
            let chr = rec.seqname.as_ref();

            // Each thread creates its own faidx reader (not thread-safe)
            let faidx = load_fasta_index(&params.genome_file)?;

            // Pileup all positions in the gene (no position filter)
            let mut freq_map = params.new_freq_map();

            for bam_file in &params.bam_files {
                freq_map.update_from_gene(
                    bam_file,
                    rec,
                    &params.gene_barcode_tag,
                    params.include_missing_barcode,
                )?;
            }

            let freq = freq_map
                .marginal_frequency_map()
                .ok_or_else(|| anyhow::anyhow!("expected marginal frequency map"))?;
            let qual_map = freq_map.quality_map();

            let gp = &params.genotype_params;
            let mut local_sites = Vec::new();

            for (&pos, counts) in freq {
                let depth = counts.total();
                if depth < gp.min_coverage {
                    continue;
                }

                let ref_dna = match fetch_reference_base(&faidx, chr, pos)? {
                    Some(b) => b,
                    None => continue,
                };
                let ref_byte = ref_dna.to_byte();

                let (alt_byte, alt_count) = match find_top_alt_allele(counts, ref_byte) {
                    Some(x) => x,
                    None => continue,
                };

                if alt_count < gp.min_alt_count {
                    continue;
                }
                if (alt_count as f64 / depth as f64) < gp.min_alt_freq {
                    continue;
                }

                let qual = qual_map.get(&pos);
                let site = genotype_site(
                    SiteInput {
                        chr: chr.into(),
                        pos,
                        ref_allele: ref_byte,
                        alt_allele: alt_byte,
                        rsid: None,
                        counts: counts.clone(),
                        qual: qual.cloned(),
                    },
                    gp,
                );
                local_sites.push(site);
            }

            if !local_sites.is_empty() {
                gene_site_map
                    .entry(rec.gene_id.clone())
                    .or_default()
                    .extend(local_sites.clone());
                arc_sites.lock().expect("lock").extend(local_sites);
            }

            Ok(())
        })?;

    let mut sites = Arc::try_unwrap(arc_sites)
        .map_err(|_| anyhow::anyhow!("failed to release sites"))?
        .into_inner()?;

    sites.sort_by(|a, b| a.chr.cmp(&b.chr).then(a.pos.cmp(&b.pos)));
    sites.dedup_by(|a, b| a.chr == b.chr && a.pos == b.pos);

    let gene_sites = Arc::try_unwrap(gene_site_map)
        .map_err(|_| anyhow::anyhow!("failed to release gene_site_map"))?;

    Ok(GenePileupResult { sites, gene_sites })
}

/// Region-centric de novo SNP discovery (no GFF): iterate BAM header contigs.
pub fn discover_snps_by_region(params: &SnpParams) -> anyhow::Result<Vec<SnpSite>> {
    use crate::data::util_htslib::create_bam_jobs;

    let jobs = create_bam_jobs(&params.bam_files[0], None, None)?;
    let njobs = jobs.len() as u64;
    info!(
        "Discovering SNPs de novo across {} regions (region mode)",
        njobs
    );

    load_fasta_index(&params.genome_file)?;

    let arc_sites = Arc::new(Mutex::new(Vec::<SnpSite>::new()));

    jobs.par_iter().progress_count(njobs).try_for_each(
        |(chr, start, stop)| -> anyhow::Result<()> {
            let faidx = load_fasta_index(&params.genome_file)?;

            let bed = Bed {
                chr: chr.clone(),
                start: *start,
                stop: *stop,
            };

            let mut freq_map = params.new_freq_map();

            for bam_file in &params.bam_files {
                freq_map.update_from_region(bam_file, &bed)?;
            }

            let freq = freq_map
                .marginal_frequency_map()
                .ok_or_else(|| anyhow::anyhow!("expected marginal frequency map"))?;
            let qual_map = freq_map.quality_map();

            let gp = &params.genotype_params;
            let mut local_sites = Vec::new();

            for (&pos, counts) in freq {
                let depth = counts.total();
                if depth < gp.min_coverage {
                    continue;
                }

                let ref_dna = match fetch_reference_base(&faidx, chr, pos)? {
                    Some(b) => b,
                    None => continue,
                };
                let ref_byte = ref_dna.to_byte();

                let (alt_byte, alt_count) = match find_top_alt_allele(counts, ref_byte) {
                    Some(x) => x,
                    None => continue,
                };

                if alt_count < gp.min_alt_count {
                    continue;
                }
                if (alt_count as f64 / depth as f64) < gp.min_alt_freq {
                    continue;
                }

                let qual = qual_map.get(&pos);
                let site = genotype_site(
                    SiteInput {
                        chr: chr.clone(),
                        pos,
                        ref_allele: ref_byte,
                        alt_allele: alt_byte,
                        rsid: None,
                        counts: counts.clone(),
                        qual: qual.cloned(),
                    },
                    gp,
                );
                local_sites.push(site);
            }

            if !local_sites.is_empty() {
                arc_sites.lock().expect("lock").extend(local_sites);
            }

            Ok(())
        },
    )?;

    let mut sites = Arc::try_unwrap(arc_sites)
        .map_err(|_| anyhow::anyhow!("failed to release sites"))?
        .into_inner()?;

    sites.sort_by(|a, b| a.chr.cmp(&b.chr).then(a.pos.cmp(&b.pos)));
    sites.dedup_by(|a, b| a.chr == b.chr && a.pos == b.pos);
    Ok(sites)
}

// ============================================================
// SECOND PASS: Per-cell allele counts (10x single-cell mode)
// ============================================================

/// Dual per-cell output: alt allele counts + total depth.
pub struct DualTriplets {
    pub alt_triplets: Vec<(CellBarcode, Box<str>, f32)>,
    pub depth_triplets: Vec<(CellBarcode, Box<str>, f32)>,
}

/// Gather per-cell allele counts and depth at called SNP sites.
pub fn gather_snp_allele_counts_by_gene(
    called_sites: &DashMap<GeneId, Vec<SnpSite>>,
    gff_map: &GffRecordMap,
    params: &SnpParams,
    bam_file: &str,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<DualTriplets> {
    let gene_key_fn = create_gene_key_function(gff_map);

    let arc_alt = Arc::new(Mutex::new(Vec::new()));
    let arc_depth = Arc::new(Mutex::new(Vec::new()));

    called_sites
        .iter()
        .par_bridge()
        .progress_count(called_sites.len() as u64)
        .try_for_each(|entry| -> anyhow::Result<()> {
            let gene_id = entry.key();
            let sites = entry.value();

            let gff = match gff_map.get(gene_id) {
                Some(g) => g,
                None => return Ok(()),
            };

            if sites.is_empty() {
                return Ok(());
            }

            let positions: FxHashSet<i64> = sites.iter().map(|s| s.pos).collect();

            let mut stat_map = params.new_freq_map_percell(cell_membership);
            stat_map.set_position_filter(positions);

            stat_map.update_from_gene(
                bam_file,
                &gff,
                &params.gene_barcode_tag,
                params.include_missing_barcode,
            )?;

            let chr = gff.seqname.as_ref();

            let bed = BedWithGene {
                chr: chr.into(),
                start: gff.start,
                stop: gff.stop,
                gene: gene_id.clone(),
                strand: gff.strand,
            };
            let gene_key = gene_key_fn(&bed);

            let mut local_alt = Vec::new();
            let mut local_depth = Vec::new();

            for site in sites.iter() {
                let feature_name: Box<str> =
                    format!("{}/SNP/{}:{}", gene_key, site.chr, site.pos).into();

                if let Some(cell_counts) = stat_map.stratified_frequency_at(site.pos) {
                    for (cb, counts) in cell_counts {
                        if !params.include_missing_barcode && cb == &CellBarcode::Missing {
                            continue;
                        }
                        let alt_count = counts.get(Dna::from_byte(site.alt_allele).as_ref());
                        let depth = counts.total();

                        if depth > 0 {
                            local_alt.push((cb.clone(), feature_name.clone(), alt_count as f32));
                            local_depth.push((cb.clone(), feature_name.clone(), depth as f32));
                        }
                    }
                }
            }

            if !local_alt.is_empty() {
                arc_alt.lock().expect("lock").extend(local_alt);
                arc_depth.lock().expect("lock").extend(local_depth);
            }

            Ok(())
        })?;

    let alt_triplets = Arc::try_unwrap(arc_alt)
        .map_err(|_| anyhow::anyhow!("failed to release alt triplets"))?
        .into_inner()?;
    let depth_triplets = Arc::try_unwrap(arc_depth)
        .map_err(|_| anyhow::anyhow!("failed to release depth triplets"))?
        .into_inner()?;

    Ok(DualTriplets {
        alt_triplets,
        depth_triplets,
    })
}

// ============================================================
// Orchestration: full SNP pipeline
// ============================================================

/// Run the full SNP genotyping pipeline.
///
/// Modes:
/// - known_snps=Some, discover=false: genotype at known VCF positions only
/// - known_snps=None, discover=true: de novo variant discovery from reference
/// - known_snps=Some, discover=true: discover + force-call at known sites, merge
pub fn run_snp_pipeline(
    known_snps: Option<&KnownSnps>,
    gff_map: Option<&GffRecordMap>,
    params: &SnpParams,
    discover: bool,
) -> anyhow::Result<FxHashSet<(Box<str>, i64)>> {
    let mut all_sites = Vec::new();
    let gene_sites = DashMap::<GeneId, Vec<SnpSite>>::default();

    // De novo discovery
    if discover {
        let result = if let Some(gff) = gff_map {
            discover_snps_by_gene(gff, params)?
        } else {
            let sites = discover_snps_by_region(params)?;
            GenePileupResult {
                sites,
                gene_sites: DashMap::default(),
            }
        };
        info!("Discovered {} candidate variant sites", result.sites.len());
        all_sites.extend(result.sites);
        for entry in result.gene_sites.into_iter() {
            gene_sites.entry(entry.0).or_default().extend(entry.1);
        }
    }

    // Known-site genotyping
    if let Some(snps) = known_snps {
        let result = if let Some(gff) = gff_map {
            let r = pileup_known_snps_by_gene(gff, snps, params)?;
            (r.sites, r.gene_sites)
        } else {
            (
                pileup_known_snps_by_region(snps, params)?,
                DashMap::default(),
            )
        };
        info!("Genotyped {} known variant sites", result.0.len());
        all_sites.extend(result.0);
        for entry in result.1.into_iter() {
            gene_sites.entry(entry.0).or_default().extend(entry.1);
        }
    }

    // Deduplicate (known sites may overlap with discovered sites)
    all_sites.sort_by(|a, b| a.chr.cmp(&b.chr).then(a.pos.cmp(&b.pos)));
    all_sites.dedup_by(|a, b| a.chr == b.chr && a.pos == b.pos);

    // Summary
    let (mut n_het, mut n_hom_alt, mut n_nocall) = (0usize, 0usize, 0usize);
    for s in &all_sites {
        match s.genotype {
            SnpGenotype::Het => n_het += 1,
            SnpGenotype::HomAlt => n_hom_alt += 1,
            SnpGenotype::NoCall => n_nocall += 1,
            SnpGenotype::HomRef => {}
        }
    }
    let n_called = all_sites.len() - n_nocall;

    info!(
        "Total {} sites: {} called ({} het, {} hom-alt), {} no-call",
        all_sites.len(),
        n_called,
        n_het,
        n_hom_alt,
        n_nocall,
    );

    // Write parquet
    let parquet_path = format!("{}/snp_sites.parquet", params.output);
    write_snp_sites_parquet(&all_sites, &parquet_path)?;
    info!("Wrote {}", parquet_path);

    // Write VCF.gz
    match load_contigs_from_fai(&params.genome_file) {
        Ok(contigs) => {
            let vcf_path = format!("{}/snp_sites.vcf.gz", params.output);
            match write_snp_sites_vcf(&all_sites, &vcf_path, &contigs) {
                Ok(()) => info!("Wrote {}", vcf_path),
                Err(e) => log::warn!("Failed to write VCF: {}", e),
            }
        }
        Err(e) => log::warn!("Cannot load .fai for VCF contigs: {}", e),
    }

    // Build mask
    let snp_mask = build_snp_mask(&all_sites, params.genotype_params.min_gq, params.min_vaf);
    info!("SNP mask: {} variant positions", snp_mask.len());

    // Pass 2: Per-cell allele counts (single-cell mode only)
    if !params.bulk {
        if let Some(gff) = gff_map {
            if !gene_sites.is_empty() {
                // Filter out NoCall sites
                for mut entry in gene_sites.iter_mut() {
                    entry
                        .value_mut()
                        .retain(|s| s.genotype != SnpGenotype::NoCall);
                }
                gene_sites.retain(|_, v| !v.is_empty());

                let batch_names = uniq_batch_names(&params.bam_files)?;

                for (bam_file, batch_name) in params.bam_files.iter().zip(batch_names.iter()) {
                    info!("Second pass (per-cell): {}", bam_file);
                    let dual =
                        gather_snp_allele_counts_by_gene(&gene_sites, gff, params, bam_file, None)?;

                    if dual.alt_triplets.is_empty() {
                        info!("no per-cell SNP data for {}", bam_file);
                        continue;
                    }

                    // Collect union names so both matrices have identical dimensions
                    let union = collect_union_names(&dual.alt_triplets, &dual.depth_triplets);

                    // Alt allele count matrix
                    let alt_data = format_data_triplets_shared(
                        dual.alt_triplets,
                        &union.feature_to_index,
                        &union.cell_to_index,
                        union.row_names.clone(),
                        union.col_names.clone(),
                    );
                    let alt_path = params.backend_file_path(&format!("{}_snp_alt", batch_name));
                    info!("Writing alt count matrix: {}", alt_path);
                    alt_data.to_backend(&alt_path)?;

                    // Total depth matrix
                    let depth_data = format_data_triplets_shared(
                        dual.depth_triplets,
                        &union.feature_to_index,
                        &union.cell_to_index,
                        union.row_names,
                        union.col_names,
                    );
                    let depth_path = params.backend_file_path(&format!("{}_snp_depth", batch_name));
                    info!("Writing depth matrix: {}", depth_path);
                    depth_data.to_backend(&depth_path)?;
                }
            }
        } else {
            info!("Skipping per-cell allele counts (no GFF provided)");
        }
    }

    Ok(snp_mask)
}
