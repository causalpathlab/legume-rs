//! BED output for discovered conversion sites (m6A only).
//!
//! Split out of `editing::pipeline`: writes per-cell, per-site conversion
//! stats as a bgzipped BED instead of a sparse matrix.

use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::data::conversion::*;
use crate::editing::pipeline::{gather_conversion_stats, ConversionParams};
use crate::editing::ConversionSite;

use dashmap::DashMap as HashMap;
use genomic_data::gff::{GeneId, GffRecordMap};

pub fn process_all_bam_files_to_bed(
    params: &ConversionParams,
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    gff_map: &GffRecordMap,
    output_cell_types: bool,
) -> anyhow::Result<()> {
    let membership = params.load_membership()?;

    let batch_names = uniq_batch_names(&params.wt_bam_files)?;

    for (bam_file, batch_name) in params.wt_bam_files.iter().zip(batch_names) {
        let mut stats = gather_conversion_stats(
            gene_sites,
            params,
            gff_map,
            bam_file,
            membership.as_ref(),
            None,
        )?;

        // Fix the line order HERE, where the rows are assembled, not in the
        // writer. `gather_conversion_stats` is `DashMap -> par_bridge ->
        // Mutex::extend`, so the vector arrives in whatever order the threads
        // finished. Sorting on `BedWithGene` alone did not repair that: one BED
        // row is emitted per (site, cell), so every cell at a site compares
        // equal, and `par_sort_by` is not stable — two runs on identical input
        // shuffled those blocks and no longer byte-compared. The key below is
        // total: `site_pos` separates two motif C's that share a conversion
        // position, and the barcode separates the cells at one site.
        stats.par_sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.site_pos.cmp(&b.2.site_pos))
                .then_with(|| a.0.cmp(&b.0))
        });

        let bed_path = format!("{}/{}.bed.gz", &params.output, batch_name);
        write_bed(
            &stats,
            gff_map,
            &bed_path,
            membership.as_ref(),
            output_cell_types,
        )?;
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f32 / total as f32
            } else {
                0.0
            }
        );
    }

    Ok(())
}

/// Write rows in the order given — the caller orders them (see
/// [`process_all_bam_files_to_bed`]), as discovery does for the site parquets.
/// Sorting here is what produced the bug: a writer keys on the fields it is
/// about to print, and those were not a total order over the rows it was handed.
pub fn write_bed(
    stats: &[(CellBarcode, BedWithGene, ConversionData)],
    gff_map: &GffRecordMap,
    file_path: &str,
    cell_membership: Option<&CellMembership>,
    output_cell_types: bool,
) -> anyhow::Result<()> {
    use rust_htslib::bgzf::Writer as BWriter;
    use std::io::Write;

    let lines: Vec<_> = stats
        .iter()
        .map(|(cb, bg, data)| {
            let gene_string = gff_map
                .get(&bg.gene)
                .map(|gff| match gff.gene_name {
                    GeneSymbol::Symbol(x) => format!("{}_{}", &bg.gene, x),
                    GeneSymbol::Missing => format!("{}", &bg.gene),
                })
                .unwrap_or_else(|| format!("{}", &bg.gene));

            if output_cell_types {
                if let Some(membership) = cell_membership {
                    let cell_type = membership
                        .matches_barcode(cb)
                        .unwrap_or_else(|| "unknown".into());
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.converted,
                        data.unconverted,
                        cb,
                        data.site_pos,
                        cell_type
                    )
                    .into_boxed_str()
                } else {
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tunknown",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.converted,
                        data.unconverted,
                        cb,
                        data.site_pos
                    )
                    .into_boxed_str()
                }
            } else {
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    bg.chr,
                    bg.start,
                    bg.stop,
                    bg.strand,
                    gene_string,
                    data.converted,
                    data.unconverted,
                    cb,
                    data.site_pos
                )
                .into_boxed_str()
            }
        })
        .collect();

    let header: &[u8] = if output_cell_types {
        b"#chr\tstart\tstop\tstrand\tgene\tconverted\tunconverted\tbarcode\tsite_pos\tcell_type\n"
    } else {
        b"#chr\tstart\tstop\tstrand\tgene\tconverted\tunconverted\tbarcode\tsite_pos\n"
    };

    let mut writer = BWriter::from_path(file_path)?;
    writer.write_all(header)?;
    for line in lines {
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    Ok(())
}
