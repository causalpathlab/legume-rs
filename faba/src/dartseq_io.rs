use genomic_data::gff::{GeneId, GffRecordMap};
use genomic_data::sam::Strand;
use crate::dartseq_sifter::MethylatedSite;
use anyhow::Result;
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Trait for writing DartSeq data structures to Parquet format
pub trait ToParquet {
    /// Write the data structure to a Parquet file
    fn to_parquet<P: AsRef<Path>>(&self, gff_map: &GffRecordMap, path: P) -> Result<()>;
}

/// Implementation of ToParquet for gene-to-methylated-sites map
///
/// Creates a Parquet file with flattened rows containing:
/// - chr: chromosome/sequence name
/// - m6a_pos, conversion_pos: positions
/// - gene: "{gene_id}_{gene_name}" (looked up from gff_map)
/// - strand: "+" (forward) or "-" (backward)
/// - gene_start, gene_stop: gene boundaries (1-based, from GFF)
/// - rel_pos: strand-aware relative position from gene start (0-based offset)
/// - pv: binomial test p-value
/// - wt_a, wt_t, wt_g, wt_c: wildtype base counts
/// - mut_a, mut_t, mut_g, mut_c: mutant base counts
impl ToParquet for DashMap<GeneId, Vec<MethylatedSite>> {
    fn to_parquet<P: AsRef<Path>>(&self, gff_map: &GffRecordMap, path: P) -> Result<()> {
        // Flatten the data: collect all rows
        let mut chr_vec: Vec<String> = Vec::new();
        let mut gene_ids: Vec<String> = Vec::new();
        let mut strand_vec: Vec<String> = Vec::new();
        let mut gene_start_vec: Vec<i64> = Vec::new();
        let mut gene_stop_vec: Vec<i64> = Vec::new();
        let mut rel_pos_vec: Vec<i64> = Vec::new();
        let mut m6a_pos_vec: Vec<i64> = Vec::new();
        let mut conversion_pos_vec: Vec<i64> = Vec::new();
        let mut pv_vec: Vec<f64> = Vec::new();
        let mut wt_a_vec: Vec<u64> = Vec::new();
        let mut wt_t_vec: Vec<u64> = Vec::new();
        let mut wt_g_vec: Vec<u64> = Vec::new();
        let mut wt_c_vec: Vec<u64> = Vec::new();
        let mut mut_a_vec: Vec<u64> = Vec::new();
        let mut mut_t_vec: Vec<u64> = Vec::new();
        let mut mut_g_vec: Vec<u64> = Vec::new();
        let mut mut_c_vec: Vec<u64> = Vec::new();

        // Iterate over all genes and their sites
        for entry in self.iter() {
            let (gene_id, sites) = (entry.key(), entry.value());

            // Look up gene info from gff_map
            let gff_rec = gff_map.get(gene_id);
            let (chr, gene_name, strand_str, gene_start, gene_stop, strand_obj) = gff_rec
                .map(|rec| {
                    (
                        format!("{}", rec.seqname),
                        format!("{}", rec.gene_name),
                        format!("{}", rec.strand),
                        rec.start,
                        rec.stop,
                        rec.strand.clone(),
                    )
                })
                .unwrap_or_else(|| {
                    (
                        ".".to_string(),
                        ".".to_string(),
                        ".".to_string(),
                        0,
                        0,
                        Strand::Forward,
                    )
                });

            // Create combined gene identifier
            let gene_str = format!("{}_{}", gene_id, gene_name);

            // Convert GFF coordinates (1-based, inclusive) to 0-based half-open [lb, ub)
            // to match BAM/m6a_pos coordinates
            let lb = (gene_start - 1).max(0); // GFF 1-based start -> 0-based
            let ub = gene_stop; // GFF 1-based inclusive end -> 0-based exclusive end

            // Add a row for each MethylatedSite
            for site in sites.iter() {
                // Calculate strand-aware relative position (same logic as dartseq_stat.rs)
                let rel_pos = match strand_obj {
                    Strand::Forward => site.m6a_pos - lb,
                    Strand::Backward => ub - site.m6a_pos - 1,
                };

                chr_vec.push(chr.clone());
                gene_ids.push(gene_str.clone());
                strand_vec.push(strand_str.clone());
                gene_start_vec.push(gene_start);
                gene_stop_vec.push(gene_stop);
                rel_pos_vec.push(rel_pos);
                m6a_pos_vec.push(site.m6a_pos);
                conversion_pos_vec.push(site.conversion_pos);
                pv_vec.push(site.pv);

                wt_a_vec.push(site.wt_freq.count_a() as u64);
                wt_t_vec.push(site.wt_freq.count_t() as u64);
                wt_g_vec.push(site.wt_freq.count_g() as u64);
                wt_c_vec.push(site.wt_freq.count_c() as u64);

                mut_a_vec.push(site.mut_freq.count_a() as u64);
                mut_t_vec.push(site.mut_freq.count_t() as u64);
                mut_g_vec.push(site.mut_freq.count_g() as u64);
                mut_c_vec.push(site.mut_freq.count_c() as u64);
            }
        }

        // Create Arrow arrays
        let chr_array = Arc::new(StringArray::from(chr_vec)) as ArrayRef;
        let gene_array = Arc::new(StringArray::from(gene_ids)) as ArrayRef;
        let strand_array = Arc::new(StringArray::from(strand_vec)) as ArrayRef;
        let gene_start_array = Arc::new(Int64Array::from(gene_start_vec)) as ArrayRef;
        let gene_stop_array = Arc::new(Int64Array::from(gene_stop_vec)) as ArrayRef;
        let rel_pos_array = Arc::new(Int64Array::from(rel_pos_vec)) as ArrayRef;
        let m6a_pos_array = Arc::new(Int64Array::from(m6a_pos_vec)) as ArrayRef;
        let conversion_pos_array = Arc::new(Int64Array::from(conversion_pos_vec)) as ArrayRef;
        let pv_array = Arc::new(Float64Array::from(pv_vec)) as ArrayRef;

        let wt_a_array = Arc::new(UInt64Array::from(wt_a_vec)) as ArrayRef;
        let wt_t_array = Arc::new(UInt64Array::from(wt_t_vec)) as ArrayRef;
        let wt_g_array = Arc::new(UInt64Array::from(wt_g_vec)) as ArrayRef;
        let wt_c_array = Arc::new(UInt64Array::from(wt_c_vec)) as ArrayRef;

        let mut_a_array = Arc::new(UInt64Array::from(mut_a_vec)) as ArrayRef;
        let mut_t_array = Arc::new(UInt64Array::from(mut_t_vec)) as ArrayRef;
        let mut_g_array = Arc::new(UInt64Array::from(mut_g_vec)) as ArrayRef;
        let mut_c_array = Arc::new(UInt64Array::from(mut_c_vec)) as ArrayRef;

        // Create schema
        let schema = arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("chr", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("gene", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("strand", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("gene_start", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("gene_stop", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("m6a_pos", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new(
                "conversion_pos",
                arrow::datatypes::DataType::Int64,
                false,
            ),
            arrow::datatypes::Field::new("rel_pos", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("pv", arrow::datatypes::DataType::Float64, false),
            arrow::datatypes::Field::new("wt_a", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("wt_t", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("wt_g", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("wt_c", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("mut_a", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("mut_t", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("mut_g", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("mut_c", arrow::datatypes::DataType::UInt64, false),
        ]);

        // Create RecordBatch
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                chr_array,
                gene_array,
                strand_array,
                gene_start_array,
                gene_stop_array,
                m6a_pos_array,
                conversion_pos_array,
                rel_pos_array,
                pv_array,
                wt_a_array,
                wt_t_array,
                wt_g_array,
                wt_c_array,
                mut_a_array,
                mut_t_array,
                mut_g_array,
                mut_c_array,
            ],
        )?;

        // Write to Parquet file
        let file = File::create(path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;

        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }
}
