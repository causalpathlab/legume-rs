use anyhow::Result;
use arrow::array::{ArrayRef, Float32Array, Int64Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use log::info;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::snp::SnpSite;

/// Per-position allele info: (ref_allele, alt_allele, rsid)
pub type SnpAlleles = (u8, u8, Option<Box<str>>);

/// Parsed known variant sites from a VCF file.
/// Two-level map: chr -> pos -> alleles.
/// Stores each chromosome string once, and allows O(1) lookups by (chr, pos)
/// without allocating a Box<str> for the key.
pub struct KnownSnps {
    pub by_chr: FxHashMap<Box<str>, FxHashMap<i64, SnpAlleles>>,
}

#[allow(dead_code)]
impl KnownSnps {
    pub fn num_sites(&self) -> usize {
        self.by_chr.values().map(|m| m.len()).sum()
    }

    /// Get the set of positions on a given chromosome
    pub fn positions_for_chr(&self, chr: &str) -> Option<FxHashSet<i64>> {
        self.by_chr.get(chr).map(|m| m.keys().copied().collect())
    }

    /// Look up alleles at a (chr, pos) without allocating
    pub fn get(&self, chr: &str, pos: i64) -> Option<&SnpAlleles> {
        self.by_chr.get(chr)?.get(&pos)
    }

    /// Get all unique chromosomes
    pub fn chromosomes(&self) -> Vec<Box<str>> {
        self.by_chr.keys().cloned().collect()
    }
}

/// Load known SNP sites from a VCF/BCF file.
/// Only biallelic SNPs are loaded (indels and multi-allelics are skipped).
pub fn load_known_snps(vcf_path: &str) -> Result<KnownSnps> {
    use rust_htslib::bcf::{Read, Reader};

    let mut reader = Reader::from_path(vcf_path)?;
    let header = reader.header().clone();

    let mut by_chr: FxHashMap<Box<str>, FxHashMap<i64, SnpAlleles>> = FxHashMap::default();

    for record_result in reader.records() {
        let record = record_result?;

        let rid = record.rid().ok_or_else(|| anyhow::anyhow!("missing RID"))?;
        let chr: Box<str> = std::str::from_utf8(header.rid2name(rid)?)?
            .to_string()
            .into_boxed_str();

        let pos = record.pos();

        let alleles = record.alleles();
        if alleles.len() != 2 {
            continue;
        }

        let ref_allele = alleles[0];
        let alt_allele = alleles[1];

        if ref_allele.len() != 1 || alt_allele.len() != 1 {
            continue;
        }

        let ref_byte = ref_allele[0].to_ascii_uppercase();
        let alt_byte = alt_allele[0].to_ascii_uppercase();

        if !matches!(ref_byte, b'A' | b'T' | b'G' | b'C')
            || !matches!(alt_byte, b'A' | b'T' | b'G' | b'C')
        {
            continue;
        }

        let rsid = {
            let id = record.id();
            let id_str = std::str::from_utf8(&id).unwrap_or(".");
            if id_str == "." {
                None
            } else {
                Some(id_str.to_string().into_boxed_str())
            }
        };

        by_chr
            .entry(chr)
            .or_default()
            .insert(pos, (ref_byte, alt_byte, rsid));
    }

    let n_sites: usize = by_chr.values().map(|m| m.len()).sum();
    info!("loaded {} known biallelic SNPs from {}", n_sites, vcf_path);

    Ok(KnownSnps { by_chr })
}

/// Build a SNP mask from called sites (het or hom-alt above GQ threshold).
///
/// When `min_vaf` is Some, only sites with germline-like allele fractions
/// enter the mask: het sites need VAF in [min_vaf, 1-min_vaf], hom-alt
/// sites need VAF >= 1-min_vaf. This prevents masking true RNA editing
/// sites (low/variable VAF) discovered de novo from RNA-seq data.
pub fn build_snp_mask(
    sites: &[SnpSite],
    min_gq: f32,
    min_vaf: Option<f32>,
) -> FxHashSet<(Box<str>, i64)> {
    sites
        .iter()
        .filter(|s| {
            if s.gq < min_gq {
                return false;
            }
            let depth = s.depth();
            if depth == 0 {
                return false;
            }
            let vaf = s.alt_count() as f32 / depth as f32;
            match s.genotype {
                crate::snp::SnpGenotype::Het => match min_vaf {
                    Some(v) => vaf >= v && vaf <= (1.0 - v),
                    None => true,
                },
                crate::snp::SnpGenotype::HomAlt => match min_vaf {
                    Some(v) => vaf >= (1.0 - v),
                    None => true,
                },
                _ => false,
            }
        })
        .map(|s| (s.chr.clone(), s.pos))
        .collect()
}

/// Write SNP sites to a Parquet file.
pub fn write_snp_sites_parquet<P: AsRef<Path>>(sites: &[SnpSite], path: P) -> Result<()> {
    let mut chr_vec = Vec::with_capacity(sites.len());
    let mut pos_vec = Vec::with_capacity(sites.len());
    let mut ref_vec = Vec::with_capacity(sites.len());
    let mut alt_vec = Vec::with_capacity(sites.len());
    let mut rsid_vec = Vec::with_capacity(sites.len());
    let mut gt_vec = Vec::with_capacity(sites.len());
    let mut gq_vec = Vec::with_capacity(sites.len());
    let mut count_a_vec = Vec::with_capacity(sites.len());
    let mut count_t_vec = Vec::with_capacity(sites.len());
    let mut count_g_vec = Vec::with_capacity(sites.len());
    let mut count_c_vec = Vec::with_capacity(sites.len());

    for site in sites {
        chr_vec.push(site.chr.as_ref().to_string());
        pos_vec.push(site.pos);
        ref_vec.push(String::from(site.ref_allele as char));
        alt_vec.push(String::from(site.alt_allele as char));
        rsid_vec.push(
            site.rsid
                .as_ref()
                .map(|s| s.as_ref().to_string())
                .unwrap_or_else(|| ".".to_string()),
        );
        gt_vec.push(format!("{}", site.genotype));
        gq_vec.push(site.gq);
        count_a_vec.push(site.counts.count_a() as u64);
        count_t_vec.push(site.counts.count_t() as u64);
        count_g_vec.push(site.counts.count_g() as u64);
        count_c_vec.push(site.counts.count_c() as u64);
    }

    use arrow::datatypes::{DataType, Field, Schema};

    let schema = Schema::new(vec![
        Field::new("chr", DataType::Utf8, false),
        Field::new("pos", DataType::Int64, false),
        Field::new("ref_allele", DataType::Utf8, false),
        Field::new("alt_allele", DataType::Utf8, false),
        Field::new("rsid", DataType::Utf8, false),
        Field::new("genotype", DataType::Utf8, false),
        Field::new("gq", DataType::Float32, false),
        Field::new("count_a", DataType::UInt64, false),
        Field::new("count_t", DataType::UInt64, false),
        Field::new("count_g", DataType::UInt64, false),
        Field::new("count_c", DataType::UInt64, false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(chr_vec)) as ArrayRef,
            Arc::new(Int64Array::from(pos_vec)) as ArrayRef,
            Arc::new(StringArray::from(ref_vec)) as ArrayRef,
            Arc::new(StringArray::from(alt_vec)) as ArrayRef,
            Arc::new(StringArray::from(rsid_vec)) as ArrayRef,
            Arc::new(StringArray::from(gt_vec)) as ArrayRef,
            Arc::new(Float32Array::from(gq_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(count_a_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(count_t_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(count_g_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(count_c_vec)) as ArrayRef,
        ],
    )?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Load known SNP sites from a parquet file (output of `faba snp`).
/// Reads chr, pos, ref_allele, alt_allele, rsid columns.
pub fn load_known_snps_from_parquet<P: AsRef<Path>>(path: P) -> Result<KnownSnps> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path.as_ref())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut by_chr: FxHashMap<Box<str>, FxHashMap<i64, SnpAlleles>> = FxHashMap::default();

    for batch in reader {
        let batch = batch?;

        let chr_col = batch
            .column_by_name("chr")
            .ok_or_else(|| anyhow::anyhow!("missing 'chr' column"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'chr' is not a string array"))?;

        let pos_col = batch
            .column_by_name("pos")
            .ok_or_else(|| anyhow::anyhow!("missing 'pos' column"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("'pos' is not Int64"))?;

        let ref_col = batch
            .column_by_name("ref_allele")
            .ok_or_else(|| anyhow::anyhow!("missing 'ref_allele' column"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'ref_allele' is not a string array"))?;

        let alt_col = batch
            .column_by_name("alt_allele")
            .ok_or_else(|| anyhow::anyhow!("missing 'alt_allele' column"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'alt_allele' is not a string array"))?;

        let rsid_col = batch
            .column_by_name("rsid")
            .ok_or_else(|| anyhow::anyhow!("missing 'rsid' column"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'rsid' is not a string array"))?;

        for i in 0..batch.num_rows() {
            let chr: Box<str> = chr_col.value(i).into();
            let pos = pos_col.value(i);

            let ref_str = ref_col.value(i);
            let alt_str = alt_col.value(i);
            if ref_str.len() != 1 || alt_str.len() != 1 {
                continue;
            }
            let ref_byte = ref_str.as_bytes()[0].to_ascii_uppercase();
            let alt_byte = alt_str.as_bytes()[0].to_ascii_uppercase();

            if !matches!(ref_byte, b'A' | b'T' | b'G' | b'C')
                || !matches!(alt_byte, b'A' | b'T' | b'G' | b'C')
            {
                continue;
            }

            let rsid_str = rsid_col.value(i);
            let rsid = if rsid_str == "." {
                None
            } else {
                Some(rsid_str.to_string().into_boxed_str())
            };

            by_chr
                .entry(chr)
                .or_default()
                .insert(pos, (ref_byte, alt_byte, rsid));
        }
    }

    let n_sites: usize = by_chr.values().map(|m| m.len()).sum();
    info!(
        "loaded {} known biallelic SNPs from {}",
        n_sites,
        path.as_ref().display()
    );

    Ok(KnownSnps { by_chr })
}

/// Load known SNPs from VCF/BCF or Parquet, auto-detected by file extension.
pub fn load_known_snps_auto(path: &str) -> Result<KnownSnps> {
    if path.ends_with(".parquet") {
        load_known_snps_from_parquet(path)
    } else {
        load_known_snps(path)
    }
}

/// Write SNP sites to a bgzipped VCF file (.vcf.gz).
///
/// Produces a single-sample VCF with GT, GQ, AD, and DP fields.
/// Contig info is passed as (name, length) pairs from the BAM header or FASTA index.
pub fn write_snp_sites_vcf(
    sites: &[SnpSite],
    path: &str,
    contigs: &[(Box<str>, u64)],
) -> Result<()> {
    use rust_htslib::bcf::{self, Format, Header, Writer};

    let mut header = Header::new();

    // Add contig lines
    for (name, len) in contigs {
        let line = format!("##contig=<ID={},length={}>", name, len);
        header.push_record(line.as_bytes());
    }

    // FORMAT fields
    header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
    header.push_record(
        b"##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality (Phred)\">",
    );
    header.push_record(
        b"##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths (ref, alt)\">",
    );
    header.push_record(b"##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Total read depth\">");

    header.push_sample(b"SAMPLE");

    let mut writer = Writer::from_path(path, &header, false, Format::Vcf)?;

    for site in sites {
        let mut record = writer.empty_record();

        // Set chromosome
        let rid = writer.header().name2rid(site.chr.as_bytes()).unwrap_or(0);
        record.set_rid(Some(rid));
        record.set_pos(site.pos);

        // Set alleles
        let ref_str = [site.ref_allele];
        let alt_str = [site.alt_allele];
        record
            .set_alleles(&[&ref_str, &alt_str])
            .map_err(|e| anyhow::anyhow!("set_alleles: {}", e))?;

        // Set ID (rsid)
        if let Some(ref rsid) = site.rsid {
            record.set_id(rsid.as_bytes()).ok();
        }

        // GT
        let gt = match site.genotype {
            crate::snp::SnpGenotype::HomRef => vec![
                bcf::record::GenotypeAllele::Unphased(0),
                bcf::record::GenotypeAllele::Unphased(0),
            ],
            crate::snp::SnpGenotype::Het => vec![
                bcf::record::GenotypeAllele::Unphased(0),
                bcf::record::GenotypeAllele::Unphased(1),
            ],
            crate::snp::SnpGenotype::HomAlt => vec![
                bcf::record::GenotypeAllele::Unphased(1),
                bcf::record::GenotypeAllele::Unphased(1),
            ],
            crate::snp::SnpGenotype::NoCall => vec![
                bcf::record::GenotypeAllele::UnphasedMissing,
                bcf::record::GenotypeAllele::UnphasedMissing,
            ],
        };
        record
            .push_genotypes(&gt)
            .map_err(|e| anyhow::anyhow!("push_genotypes: {}", e))?;

        // GQ
        record
            .push_format_integer(b"GQ", &[site.gq as i32])
            .map_err(|e| anyhow::anyhow!("push GQ: {}", e))?;

        // AD (ref, alt)
        let ref_count = site.ref_count() as i32;
        let alt_count = site.alt_count() as i32;
        record
            .push_format_integer(b"AD", &[ref_count, alt_count])
            .map_err(|e| anyhow::anyhow!("push AD: {}", e))?;

        // DP
        let dp = site.depth() as i32;
        record
            .push_format_integer(b"DP", &[dp])
            .map_err(|e| anyhow::anyhow!("push DP: {}", e))?;

        writer
            .write(&record)
            .map_err(|e| anyhow::anyhow!("write record: {}", e))?;
    }

    Ok(())
}

/// Load contig info from a FASTA index (.fai) file.
pub fn load_contigs_from_fai(genome_file: &str) -> Result<Vec<(Box<str>, u64)>> {
    let fai_path = format!("{}.fai", genome_file);
    let contents = std::fs::read_to_string(&fai_path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {}", fai_path, e))?;

    let mut contigs = Vec::new();
    for line in contents.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 2 {
            let name: Box<str> = fields[0].into();
            let len: u64 = fields[1].parse().unwrap_or(0);
            contigs.push((name, len));
        }
    }
    Ok(contigs)
}

/// Load a SNP mask from a parquet file (output of `faba snp`).
/// Returns (chr, pos) set for het/hom-alt sites.
pub fn load_snp_mask_from_parquet<P: AsRef<Path>>(path: P) -> Result<FxHashSet<(Box<str>, i64)>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut mask = FxHashSet::default();

    for batch in reader {
        let batch = batch?;

        let chr_col = batch
            .column_by_name("chr")
            .ok_or_else(|| anyhow::anyhow!("missing 'chr' column in SNP parquet"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'chr' column is not a string array"))?;

        let pos_col = batch
            .column_by_name("pos")
            .ok_or_else(|| anyhow::anyhow!("missing 'pos' column in SNP parquet"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("'pos' column is not an Int64 array"))?;

        let gt_col = batch
            .column_by_name("genotype")
            .ok_or_else(|| anyhow::anyhow!("missing 'genotype' column in SNP parquet"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'genotype' column is not a string array"))?;

        for i in 0..batch.num_rows() {
            let gt = gt_col.value(i);
            // Only include het and hom-alt in the mask
            if gt == "0/1" || gt == "1/1" {
                let chr: Box<str> = chr_col.value(i).into();
                let pos = pos_col.value(i);
                mask.insert((chr, pos));
            }
        }
    }

    Ok(mask)
}
