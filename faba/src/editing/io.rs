use crate::data::gene_model::SplicedGenes;
use crate::editing::pipeline::GeneContrastStat;
use crate::editing::ConversionSite;
use anyhow::Result;
use arrow::array::{ArrayRef, Float32Array, Int64Array, Int64Builder, StringArray, UInt64Array};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
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
/// - pv: per-site contrast p-value (site localization)
/// - gene_pv: gene-level pooled contrast p-value under `--m6a-test-level gene`
///   (NaN under per-site testing); `qvalue` is its BH adjustment
/// - qvalue: Benjamini-Hochberg q-value — per-site under site testing, the
///   shared gene q under gene testing
/// - reason: test outcome — `selected` / `low_control` / `delta` / `fdr`
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
        let mut gene_pv_vec: Vec<f32> = Vec::new();
        let mut qv_vec: Vec<f32> = Vec::new();
        let mut reason_vec: Vec<String> = Vec::new();
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

        // Row order is fixed upstream, where discovery partitions its results
        // (see `partition_by_site`), so every writer that reads a
        // `DiscoveredSites` is reproducible -- not just this one.
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
                gene_pv_vec.push(site.gene_pv());
                qv_vec.push(site.qv());
                reason_vec.push(site.reason().label().to_string());

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
        let gene_pv_array = Arc::new(Float32Array::from(gene_pv_vec)) as ArrayRef;
        let qv_array = Arc::new(Float32Array::from(qv_vec)) as ArrayRef;
        let reason_array = Arc::new(StringArray::from(reason_vec)) as ArrayRef;

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
            arrow::datatypes::Field::new("gene_pv", arrow::datatypes::DataType::Float32, false),
            arrow::datatypes::Field::new("qvalue", arrow::datatypes::DataType::Float32, false),
            arrow::datatypes::Field::new("reason", arrow::datatypes::DataType::Utf8, false),
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
                gene_pv_array,
                qv_array,
                reason_array,
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

/// Write per-gene m6A contrast records to `m6a_genes.parquet` — the gene-level
/// analogue of the per-site `m6a_sites.parquet`. One row per gene: the pooled
/// 2×2 (WT/MUT converted/unconverted) summed across the gene's candidate motif
/// C's, the pooled contrast p-value, and the Benjamini-Hochberg q-value across
/// genes. Only produced under `--m6a-test-level gene`.
///
/// Columns: chr, gene (`{gene_id}_{gene_name}`), strand, gene_start, gene_stop,
/// n_sites, wt_converted, wt_unconverted, mut_converted, mut_unconverted, pv,
/// qvalue, reason (`selected` / `low_control` / `delta` / `fdr`).
pub fn write_gene_contrast_parquet<'a, P: AsRef<Path>>(
    stats: impl IntoIterator<Item = &'a GeneContrastStat>,
    gff_map: &GffRecordMap,
    path: P,
) -> Result<()> {
    use arrow::datatypes::{DataType, Field, Schema};

    let mut chr_vec: Vec<String> = Vec::new();
    let mut gene_vec: Vec<String> = Vec::new();
    let mut strand_vec: Vec<String> = Vec::new();
    let mut gene_start_vec: Vec<i64> = Vec::new();
    let mut gene_stop_vec: Vec<i64> = Vec::new();
    let mut n_sites_vec: Vec<u64> = Vec::new();
    let mut wt_conv_vec: Vec<u64> = Vec::new();
    let mut wt_unconv_vec: Vec<u64> = Vec::new();
    let mut mut_conv_vec: Vec<u64> = Vec::new();
    let mut mut_unconv_vec: Vec<u64> = Vec::new();
    let mut pv_vec: Vec<f32> = Vec::new();
    let mut qv_vec: Vec<f32> = Vec::new();
    let mut reason_vec: Vec<String> = Vec::new();

    for s in stats {
        let (chr, gene_name, strand_str, gene_start, gene_stop) = gff_map
            .get(&s.gene_id)
            .map(|rec| {
                (
                    format!("{}", rec.seqname),
                    format!("{}", rec.gene_name),
                    format!("{}", rec.strand),
                    rec.start,
                    rec.stop,
                )
            })
            .unwrap_or_else(|| (".".into(), ".".into(), ".".into(), 0, 0));

        chr_vec.push(chr);
        gene_vec.push(format!("{}_{}", s.gene_id, gene_name));
        strand_vec.push(strand_str);
        gene_start_vec.push(gene_start);
        gene_stop_vec.push(gene_stop);
        n_sites_vec.push(s.n_sites as u64);
        wt_conv_vec.push(s.wt_converted);
        wt_unconv_vec.push(s.wt_unconverted);
        mut_conv_vec.push(s.mut_converted);
        mut_unconv_vec.push(s.mut_unconverted);
        pv_vec.push(s.pv);
        qv_vec.push(s.qv);
        reason_vec.push(s.reason.label().to_string());
    }

    let schema = Schema::new(vec![
        Field::new("chr", DataType::Utf8, false),
        Field::new("gene", DataType::Utf8, false),
        Field::new("strand", DataType::Utf8, false),
        Field::new("gene_start", DataType::Int64, false),
        Field::new("gene_stop", DataType::Int64, false),
        Field::new("n_sites", DataType::UInt64, false),
        Field::new("wt_converted", DataType::UInt64, false),
        Field::new("wt_unconverted", DataType::UInt64, false),
        Field::new("mut_converted", DataType::UInt64, false),
        Field::new("mut_unconverted", DataType::UInt64, false),
        Field::new("pv", DataType::Float32, false),
        Field::new("qvalue", DataType::Float32, false),
        Field::new("reason", DataType::Utf8, false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(chr_vec)) as ArrayRef,
            Arc::new(StringArray::from(gene_vec)) as ArrayRef,
            Arc::new(StringArray::from(strand_vec)) as ArrayRef,
            Arc::new(Int64Array::from(gene_start_vec)) as ArrayRef,
            Arc::new(Int64Array::from(gene_stop_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(n_sites_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(wt_conv_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(wt_unconv_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(mut_conv_vec)) as ArrayRef,
            Arc::new(UInt64Array::from(mut_unconv_vec)) as ArrayRef,
            Arc::new(Float32Array::from(pv_vec)) as ArrayRef,
            Arc::new(Float32Array::from(qv_vec)) as ArrayRef,
            Arc::new(StringArray::from(reason_vec)) as ArrayRef,
        ],
    )?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Emit the discovery audit outputs for one modality, shared by `faba dartseq`
/// and `faba all` so both write identical files. `prefix` is the modality file
/// stem (e.g. `"m6a"`). Writes `{prefix}_sites_unselected.parquet` (every
/// putative site that missed the cut, with its `reason`) and, under gene-level
/// testing (non-empty `gene_stats`), `{prefix}_genes.parquet` +
/// `{prefix}_genes_unselected.parquet`. The *selected* sites are written by the
/// caller after masking; this is the pre-mask audit of the test itself.
pub fn write_discovery_outputs(
    discovered: &crate::editing::pipeline::DiscoveredSites,
    gff_map: &GffRecordMap,
    spliced: &SplicedGenes,
    output_dir: &str,
    prefix: &str,
) -> Result<()> {
    let n_unselected: usize = discovered.rejected.iter().map(|e| e.value().len()).sum();
    discovered.rejected.to_parquet(
        gff_map,
        spliced,
        format!("{output_dir}/{prefix}_sites_unselected.parquet"),
    )?;
    log::info!("wrote {n_unselected} unselected {prefix} sites (with reasons)");

    if !discovered.gene_stats.is_empty() {
        let n_sel = discovered
            .gene_stats
            .iter()
            .filter(|g| g.reason.is_selected())
            .count();
        write_gene_contrast_parquet(
            discovered
                .gene_stats
                .iter()
                .filter(|g| g.reason.is_selected()),
            gff_map,
            format!("{output_dir}/{prefix}_genes.parquet"),
        )?;
        write_gene_contrast_parquet(
            discovered
                .gene_stats
                .iter()
                .filter(|g| !g.reason.is_selected()),
            gff_map,
            format!("{output_dir}/{prefix}_genes_unselected.parquet"),
        )?;
        log::info!(
            "wrote {n_sel} selected / {} unselected {prefix} genes (per-gene test)",
            discovered.gene_stats.len() - n_sel,
        );
    }
    Ok(())
}

/// Load an A-to-I mask from a parquet file (output of `faba atoi`, `faba dartseq --detect-atoi`,
/// or the unified `faba editing` pipeline).
///
/// Returns a set of (chr, position) tuples for masking known A-to-I sites.
/// Tries "primary_pos" column first (new unified format), falls back to "editing_pos"
/// (legacy atoi format) for backward compatibility.
pub fn load_atoi_mask_from_parquet<P: AsRef<Path>>(
    path: P,
) -> Result<rustc_hash::FxHashSet<(Box<str>, i64)>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut mask = rustc_hash::FxHashSet::default();

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
