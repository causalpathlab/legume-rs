use crate::data::gene_model::SplicedGenes;
use crate::editing::ConversionSite;
use anyhow::Result;
use arrow::array::{
    ArrayRef, Float32Array, Float32Builder, Int64Array, Int64Builder, StringArray, UInt64Array,
};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use faba::hypothesis_tests::log_odds_ratio_woolf;
use genomic_data::gff::{GeneId, GffRecordMap};
use genomic_data::sam::Strand;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Trait for writing conversion site data to Parquet format
pub trait ToParquet {
    /// Write the data structure to a Parquet file.
    ///
    /// `spliced` supplies transcript coordinates for `rel_pos`. Not optional:
    /// an empty model already yields the same all-null column, so an `Option`
    /// here would only add a second way to say nothing -- and it would let a
    /// swallowed parse error masquerade as "every site is intronic".
    fn to_parquet<P: AsRef<Path>>(
        &self,
        gff_map: &GffRecordMap,
        spliced: &SplicedGenes,
        path: P,
    ) -> Result<()>;
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
/// - rel_pos: strand-aware position along the gene's merged EXONS (0-based),
///   i.e. a transcript coordinate. Null for an intronic site, which has none.
/// - pv: per-site p-value (Fisher exact WT-vs-MUT for m6A, beta-binomial
///   against the sequencing-error null for A-to-I). Reported for every
///   putative site; the producer applies no cutoff. `faba qc --site-max-pv`
///   thresholds it, as a MARGINAL p-value: faba does no multiplicity
///   correction, because neighbouring sites share reads and are not even
///   positively dependent (see the `qc` module for the full argument).
/// - log_odds: Haldane-corrected log odds ratio (WT vs MUT), null for A-to-I.
///   A shrunken ESTIMATE: the raw cross-product is `+inf` when the control
///   never converts. Read `log_odds_se` before reading this.
/// - log_odds_se: Woolf standard error on the same corrected cells, null for
///   A-to-I. Dominated by the SMALLEST cell rather than by either library's
///   depth, so it is large exactly where the site is thin — this is the
///   machine-readable form of "do not read an effect size off a low-abundance
///   site", which a rate difference could never express. Caveat: with zero
///   control conversions it is floored near 1.41 by the
///   pseudo-count regardless of depth, so it flags those sites without ranking
///   them; rank by `pv`. `log_odds − 1.96·log_odds_se` is a Wald lower bound if
///   one is wanted — deliberately neither a column nor a filter, because it
///   approximates the exact one-sided test `pv` already reports.
/// - coverage, converted: signal-arm reads at the site (total, converted),
///   strand-resolved via [`ConversionSite::signal_counts`]
/// - control_coverage, control_converted: the same for the control arm; both
///   0 for A-to-I, which has none
/// - wt_a, wt_t, wt_g, wt_c: base counts at the site
impl ToParquet for DashMap<GeneId, Vec<ConversionSite>> {
    fn to_parquet<P: AsRef<Path>>(
        &self,
        gff_map: &GffRecordMap,
        spliced: &SplicedGenes,
        path: P,
    ) -> Result<()> {
        let mut chr_vec: Vec<String> = Vec::new();
        let mut gene_ids: Vec<String> = Vec::new();
        let mut strand_vec: Vec<String> = Vec::new();
        let mut gene_start_vec: Vec<i64> = Vec::new();
        let mut gene_stop_vec: Vec<i64> = Vec::new();
        let mut mod_type_vec: Vec<String> = Vec::new();
        let mut primary_pos_vec: Vec<i64> = Vec::new();
        let mut conversion_pos_builder = Int64Builder::new();
        let mut rel_pos_builder = Int64Builder::new();
        let mut pv_vec: Vec<f32> = Vec::new();
        // Nullable: A-to-I is single-sample, so it has no 2×2 and no odds ratio.
        let mut log_odds_builder = Float32Builder::new();
        let mut log_odds_se_builder = Float32Builder::new();
        let mut coverage_vec: Vec<u64> = Vec::new();
        let mut converted_vec: Vec<u64> = Vec::new();
        let mut control_coverage_vec: Vec<u64> = Vec::new();
        let mut control_converted_vec: Vec<u64> = Vec::new();
        let mut wt_a_vec: Vec<u64> = Vec::new();
        let mut wt_t_vec: Vec<u64> = Vec::new();
        let mut wt_g_vec: Vec<u64> = Vec::new();
        let mut wt_c_vec: Vec<u64> = Vec::new();
        // MUT (control) base counts at the conversion position — populated for
        // m6A (the WT-vs-MUT contrast), all-zero for A-to-I (single-sample).
        let mut mut_a_vec: Vec<u64> = Vec::new();
        let mut mut_t_vec: Vec<u64> = Vec::new();
        let mut mut_g_vec: Vec<u64> = Vec::new();
        let mut mut_c_vec: Vec<u64> = Vec::new();

        // Sites within a gene are position-sorted upstream (see
        // `find_all_conversion_sites`); genes are ordered here, because a
        // DashMap has no order of its own.
        let mut ordered: Vec<_> = self.iter().collect();
        ordered.sort_unstable_by(|a, b| a.key().cmp(b.key()));

        for entry in ordered.iter() {
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

            for site in sites.iter() {
                let primary_pos = site.primary_pos();

                // Position along the gene's merged EXONS, not from its
                // genomic start: an intron is not in the transcript, so it must
                // not count toward a transcript coordinate. Null for an
                // intronic site, which has no such coordinate at all.
                let rel_pos = spliced.rel_pos(gene_id, primary_pos, strand_obj);

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

                rel_pos_builder.append_option(rel_pos);
                pv_vec.push(site.pv());

                // The reported effect size. Haldane-corrected, unlike the guard,
                // only because a Float32 column cannot hold the `+inf` the raw
                // cross-product correctly returns at a clean control. Derived
                // from the stored counts rather than cached on the site, through
                // the SAME `contrast_counts` the guard used, so the two cannot
                // drift on which base is "converted" for a given strand.
                //
                // `contrast_counts` is `None` for A-to-I, which carries both the
                // null and the reason for it: a zero would read as "OR = 1", a
                // measured absence of effect, not an absent measurement.
                let (log_odds, se) = site
                    .contrast_counts(strand_obj)
                    .map(|(a_w, u_w, a_m, u_m)| log_odds_ratio_woolf(a_w, u_w, a_m, u_m))
                    .unzip();
                log_odds_builder.append_option(log_odds.map(|v| v as f32));
                log_odds_se_builder.append_option(se.map(|v| v as f32));

                // Same strand table as the odds ratio above, so `coverage` and
                // `log_odds` cannot disagree on which base is "converted".
                let (a_w, u_w) = site.signal_counts(strand_obj);
                let (a_m, u_m) = site.control_counts(strand_obj);
                coverage_vec.push(a_w + u_w);
                converted_vec.push(a_w);
                control_coverage_vec.push(a_m + u_m);
                control_converted_vec.push(a_m);

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
        let rel_pos_array = Arc::new(rel_pos_builder.finish()) as ArrayRef;
        let pv_array = Arc::new(Float32Array::from(pv_vec)) as ArrayRef;
        let log_odds_array = Arc::new(log_odds_builder.finish()) as ArrayRef;
        let log_odds_se_array = Arc::new(log_odds_se_builder.finish()) as ArrayRef;
        let coverage_array = Arc::new(UInt64Array::from(coverage_vec)) as ArrayRef;
        let converted_array = Arc::new(UInt64Array::from(converted_vec)) as ArrayRef;
        let control_coverage_array = Arc::new(UInt64Array::from(control_coverage_vec)) as ArrayRef;
        let control_converted_array =
            Arc::new(UInt64Array::from(control_converted_vec)) as ArrayRef;

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
            arrow::datatypes::Field::new("rel_pos", arrow::datatypes::DataType::Int64, true),
            arrow::datatypes::Field::new("pv", arrow::datatypes::DataType::Float32, false),
            arrow::datatypes::Field::new("log_odds", arrow::datatypes::DataType::Float32, true),
            arrow::datatypes::Field::new("log_odds_se", arrow::datatypes::DataType::Float32, true),
            arrow::datatypes::Field::new("coverage", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("converted", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new(
                "control_coverage",
                arrow::datatypes::DataType::UInt64,
                false,
            ),
            arrow::datatypes::Field::new(
                "control_converted",
                arrow::datatypes::DataType::UInt64,
                false,
            ),
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
            Arc::new(schema),
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
                log_odds_array,
                log_odds_se_array,
                coverage_array,
                converted_array,
                control_coverage_array,
                control_converted_array,
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

        write_record_batch(&batch, path)
    }
}

/// Write one Arrow record batch as a parquet file.
pub fn write_record_batch<P: AsRef<Path>>(batch: &RecordBatch, path: P) -> Result<()> {
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}
