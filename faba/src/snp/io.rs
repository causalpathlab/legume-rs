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
pub fn build_snp_mask(sites: &[SnpSite], min_gq: f32) -> FxHashSet<(Box<str>, i64)> {
    sites
        .iter()
        .filter(|s| {
            s.gq >= min_gq
                && matches!(
                    s.genotype,
                    crate::snp::SnpGenotype::Het | crate::snp::SnpGenotype::HomAlt
                )
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
