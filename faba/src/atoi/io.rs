use crate::atoi::sifter::AtoISite;
use anyhow::Result;
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use genomic_data::gff::{GeneId, GffRecordMap};
use genomic_data::sam::Strand;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Trait for writing A-to-I data structures to Parquet format
pub trait ToParquet {
    fn to_parquet<P: AsRef<Path>>(&self, gff_map: &GffRecordMap, path: P) -> Result<()>;
}

/// ToParquet for A-to-I editing sites
impl ToParquet for DashMap<GeneId, Vec<AtoISite>> {
    fn to_parquet<P: AsRef<Path>>(&self, gff_map: &GffRecordMap, path: P) -> Result<()> {
        let mut chr_vec: Vec<String> = Vec::new();
        let mut gene_ids: Vec<String> = Vec::new();
        let mut strand_vec: Vec<String> = Vec::new();
        let mut gene_start_vec: Vec<i64> = Vec::new();
        let mut gene_stop_vec: Vec<i64> = Vec::new();
        let mut rel_pos_vec: Vec<i64> = Vec::new();
        let mut editing_pos_vec: Vec<i64> = Vec::new();
        let mut pv_vec: Vec<f64> = Vec::new();
        let mut wt_a_vec: Vec<u64> = Vec::new();
        let mut wt_t_vec: Vec<u64> = Vec::new();
        let mut wt_g_vec: Vec<u64> = Vec::new();
        let mut wt_c_vec: Vec<u64> = Vec::new();
        let mut mut_a_vec: Vec<u64> = Vec::new();
        let mut mut_t_vec: Vec<u64> = Vec::new();
        let mut mut_g_vec: Vec<u64> = Vec::new();
        let mut mut_c_vec: Vec<u64> = Vec::new();

        for entry in self.iter() {
            let (gene_id, sites) = (entry.key(), entry.value());

            let gff_rec = gff_map.get(gene_id);
            let (chr, gene_name, strand_str, gene_start, gene_stop, strand_obj) = gff_rec
                .map(|rec| {
                    (
                        format!("{}", rec.seqname),
                        format!("{}", rec.gene_name),
                        format!("{}", rec.strand),
                        rec.start,
                        rec.stop,
                        rec.strand,
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

            let gene_str = format!("{}_{}", gene_id, gene_name);
            let lb = (gene_start - 1).max(0);
            let ub = gene_stop;

            for site in sites.iter() {
                let rel_pos = match strand_obj {
                    Strand::Forward => site.editing_pos - lb,
                    Strand::Backward => ub - site.editing_pos - 1,
                };

                chr_vec.push(chr.clone());
                gene_ids.push(gene_str.clone());
                strand_vec.push(strand_str.clone());
                gene_start_vec.push(gene_start);
                gene_stop_vec.push(gene_stop);
                rel_pos_vec.push(rel_pos);
                editing_pos_vec.push(site.editing_pos);
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

        let chr_array = Arc::new(StringArray::from(chr_vec)) as ArrayRef;
        let gene_array = Arc::new(StringArray::from(gene_ids)) as ArrayRef;
        let strand_array = Arc::new(StringArray::from(strand_vec)) as ArrayRef;
        let gene_start_array = Arc::new(Int64Array::from(gene_start_vec)) as ArrayRef;
        let gene_stop_array = Arc::new(Int64Array::from(gene_stop_vec)) as ArrayRef;
        let rel_pos_array = Arc::new(Int64Array::from(rel_pos_vec)) as ArrayRef;
        let editing_pos_array = Arc::new(Int64Array::from(editing_pos_vec)) as ArrayRef;
        let pv_array = Arc::new(Float64Array::from(pv_vec)) as ArrayRef;

        let wt_a_array = Arc::new(UInt64Array::from(wt_a_vec)) as ArrayRef;
        let wt_t_array = Arc::new(UInt64Array::from(wt_t_vec)) as ArrayRef;
        let wt_g_array = Arc::new(UInt64Array::from(wt_g_vec)) as ArrayRef;
        let wt_c_array = Arc::new(UInt64Array::from(wt_c_vec)) as ArrayRef;

        let mut_a_array = Arc::new(UInt64Array::from(mut_a_vec)) as ArrayRef;
        let mut_t_array = Arc::new(UInt64Array::from(mut_t_vec)) as ArrayRef;
        let mut_g_array = Arc::new(UInt64Array::from(mut_g_vec)) as ArrayRef;
        let mut_c_array = Arc::new(UInt64Array::from(mut_c_vec)) as ArrayRef;

        let schema = arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("chr", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("gene", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("strand", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("gene_start", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("gene_stop", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("editing_pos", arrow::datatypes::DataType::Int64, false),
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

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                chr_array,
                gene_array,
                strand_array,
                gene_start_array,
                gene_stop_array,
                editing_pos_array,
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

        let file = File::create(path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;

        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }
}

/// Load an A-to-I mask from a parquet file (output of `faba atoi` or `faba dart --detect-atoi`).
/// Returns a set of (chr, editing_pos) tuples for masking.
pub fn load_atoi_mask_from_parquet<P: AsRef<Path>>(path: P) -> Result<HashSet<(Box<str>, i64)>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut mask = HashSet::new();

    for batch in reader {
        let batch = batch?;

        let chr_col = batch
            .column_by_name("chr")
            .ok_or_else(|| anyhow::anyhow!("missing 'chr' column in A-to-I parquet"))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'chr' column is not a string array"))?;

        let pos_col = batch
            .column_by_name("editing_pos")
            .ok_or_else(|| anyhow::anyhow!("missing 'editing_pos' column in A-to-I parquet"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("'editing_pos' column is not an Int64 array"))?;

        for i in 0..batch.num_rows() {
            let chr: Box<str> = chr_col.value(i).into();
            let pos = pos_col.value(i);
            mask.insert((chr, pos));
        }
    }

    Ok(mask)
}
