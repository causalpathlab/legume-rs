use crate::editing::ConversionSite;
use anyhow::Result;
use arrow::array::{ArrayRef, Float32Array, Int64Array, Int64Builder, StringArray, UInt64Array};
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

/// Trait for writing conversion site data to Parquet format
pub trait ToParquet {
    /// Write the data structure to a Parquet file
    fn to_parquet<P: AsRef<Path>>(&self, gff_map: &GffRecordMap, path: P) -> Result<()>;
}

/// Unified Parquet output for both m6A and A-to-I conversion sites.
///
/// Creates a Parquet file with flattened rows containing:
/// - chr: chromosome/sequence name
/// - gene: "{gene_id}_{gene_name}" (looked up from gff_map)
/// - strand: "+" (forward) or "-" (backward)
/// - gene_start, gene_stop: gene boundaries (1-based, from GFF)
/// - mod_type: "m6A" or "A2I"
/// - primary_pos: the main site position (m6a_pos for M6A, editing_pos for AtoI)
/// - conversion_pos: nullable Int64 (Some for M6A, None/null for AtoI)
/// - rel_pos: strand-aware relative position from gene start (0-based offset)
/// - pv: binomial test p-value
/// - wt_a, wt_t, wt_g, wt_c: wildtype base counts
/// - mut_a, mut_t, mut_g, mut_c: mutant base counts
impl ToParquet for DashMap<GeneId, Vec<ConversionSite>> {
    fn to_parquet<P: AsRef<Path>>(&self, gff_map: &GffRecordMap, path: P) -> Result<()> {
        let mut chr_vec: Vec<String> = Vec::new();
        let mut gene_ids: Vec<String> = Vec::new();
        let mut strand_vec: Vec<String> = Vec::new();
        let mut gene_start_vec: Vec<i64> = Vec::new();
        let mut gene_stop_vec: Vec<i64> = Vec::new();
        let mut mod_type_vec: Vec<String> = Vec::new();
        let mut primary_pos_vec: Vec<i64> = Vec::new();
        let mut conversion_pos_builder = Int64Builder::new();
        let mut rel_pos_vec: Vec<i64> = Vec::new();
        let mut pv_vec: Vec<f32> = Vec::new();
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

            // Convert GFF coordinates (1-based, inclusive) to 0-based half-open [lb, ub)
            // to match BAM position coordinates
            let lb = (gene_start - 1).max(0); // GFF 1-based start -> 0-based
            let ub = gene_stop; // GFF 1-based inclusive end -> 0-based exclusive end

            for site in sites.iter() {
                let primary_pos = site.primary_pos();

                // Strand-aware relative position
                let rel_pos = match strand_obj {
                    Strand::Forward => primary_pos - lb,
                    Strand::Backward => ub - primary_pos - 1,
                };

                chr_vec.push(chr.clone());
                gene_ids.push(gene_str.clone());
                strand_vec.push(strand_str.clone());
                gene_start_vec.push(gene_start);
                gene_stop_vec.push(gene_stop);
                mod_type_vec.push(site.mod_type().to_string());
                primary_pos_vec.push(primary_pos);

                // conversion_pos: nullable — Some for M6A, None (null) for AtoI
                match site {
                    ConversionSite::M6A { conversion_pos, .. } => {
                        conversion_pos_builder.append_value(*conversion_pos);
                    }
                    ConversionSite::AtoI { .. } => {
                        conversion_pos_builder.append_null();
                    }
                }

                rel_pos_vec.push(rel_pos);
                pv_vec.push(site.pv());

                wt_a_vec.push(site.wt_freq().count_a() as u64);
                wt_t_vec.push(site.wt_freq().count_t() as u64);
                wt_g_vec.push(site.wt_freq().count_g() as u64);
                wt_c_vec.push(site.wt_freq().count_c() as u64);

                mut_a_vec.push(site.mut_freq().count_a() as u64);
                mut_t_vec.push(site.mut_freq().count_t() as u64);
                mut_g_vec.push(site.mut_freq().count_g() as u64);
                mut_c_vec.push(site.mut_freq().count_c() as u64);
            }
        }

        // Create Arrow arrays
        let chr_array = Arc::new(StringArray::from(chr_vec)) as ArrayRef;
        let gene_array = Arc::new(StringArray::from(gene_ids)) as ArrayRef;
        let strand_array = Arc::new(StringArray::from(strand_vec)) as ArrayRef;
        let gene_start_array = Arc::new(Int64Array::from(gene_start_vec)) as ArrayRef;
        let gene_stop_array = Arc::new(Int64Array::from(gene_stop_vec)) as ArrayRef;
        let mod_type_array = Arc::new(StringArray::from(mod_type_vec)) as ArrayRef;
        let primary_pos_array = Arc::new(Int64Array::from(primary_pos_vec)) as ArrayRef;
        let conversion_pos_array = Arc::new(conversion_pos_builder.finish()) as ArrayRef;
        let rel_pos_array = Arc::new(Int64Array::from(rel_pos_vec)) as ArrayRef;
        let pv_array = Arc::new(Float32Array::from(pv_vec)) as ArrayRef;

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
            arrow::datatypes::Field::new("mod_type", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("primary_pos", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new(
                "conversion_pos",
                arrow::datatypes::DataType::Int64,
                true, // nullable
            ),
            arrow::datatypes::Field::new("rel_pos", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("pv", arrow::datatypes::DataType::Float32, false),
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
                mod_type_array,
                primary_pos_array,
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

        let file = File::create(path)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;

        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }
}

/// Load an A-to-I mask from a parquet file (output of `faba atoi`, `faba dart --detect-atoi`,
/// or the unified `faba editing` pipeline).
///
/// Returns a set of (chr, position) tuples for masking known A-to-I sites.
/// Tries "primary_pos" column first (new unified format), falls back to "editing_pos"
/// (legacy atoi format) for backward compatibility.
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

        // Try "primary_pos" first (unified format), fall back to "editing_pos" (legacy)
        let pos_col = batch
            .column_by_name("primary_pos")
            .or_else(|| batch.column_by_name("editing_pos"))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing position column in A-to-I parquet \
                     (expected 'primary_pos' or 'editing_pos')"
                )
            })?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("position column is not an Int64 array"))?;

        for i in 0..batch.num_rows() {
            let chr: Box<str> = chr_col.value(i).into();
            let pos = pos_col.value(i);
            mask.insert((chr, pos));
        }
    }

    Ok(mask)
}
