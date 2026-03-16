use fnv::FnvHashSet;
use genomic_data::sam::Strand;
use log::info;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;

use std::fs::File;

pub struct GenomicSite {
    pub chr: Box<str>,
    pub position: i64,
    pub strand: Strand,
}

/// Read sites from a parquet file, auto-detecting dart vs apa vs atoi format.
pub fn read_sites(site_file: &str) -> anyhow::Result<Vec<GenomicSite>> {
    let field_names = matrix_util::parquet::peek_parquet_field_names(site_file)?;
    let has_m6a_pos = field_names.iter().any(|f| f.as_ref() == "m6a_pos");
    let has_genomic_alpha = field_names.iter().any(|f| f.as_ref() == "genomic_alpha");
    let has_primary_pos = field_names.iter().any(|f| f.as_ref() == "primary_pos");

    let file = File::open(site_file)?;
    let reader = SerializedFileReader::new(file)?;
    let row_iter = reader.get_row_iter(None)?;

    // Find column indices
    let chr_idx = field_names
        .iter()
        .position(|f| f.as_ref() == "chr")
        .ok_or_else(|| anyhow::anyhow!("missing 'chr' column in {}", site_file))?;

    let mut sites = Vec::new();

    if has_m6a_pos {
        // Dart format: chr, m6a_pos, strand
        let pos_idx = field_names
            .iter()
            .position(|f| f.as_ref() == "m6a_pos")
            .unwrap();
        let strand_idx = field_names
            .iter()
            .position(|f| f.as_ref() == "strand")
            .ok_or_else(|| anyhow::anyhow!("missing 'strand' column in {}", site_file))?;

        for record in row_iter {
            let row = record?;
            let chr: Box<str> = row.get_string(chr_idx)?.clone().into_boxed_str();
            let position = row.get_long(pos_idx)?;
            let strand_str = row.get_string(strand_idx)?;
            let strand = if strand_str == "+" {
                Strand::Forward
            } else {
                Strand::Backward
            };
            sites.push(GenomicSite {
                chr,
                position,
                strand,
            });
        }
    } else if has_genomic_alpha {
        // APA format: chr, genomic_alpha (no strand column)
        let pos_idx = field_names
            .iter()
            .position(|f| f.as_ref() == "genomic_alpha")
            .unwrap();

        for record in row_iter {
            let row = record?;
            let chr: Box<str> = row.get_string(chr_idx)?.clone().into_boxed_str();
            let position = row.get_long(pos_idx)?;
            sites.push(GenomicSite {
                chr,
                position,
                strand: Strand::Forward,
            });
        }
    } else if has_primary_pos {
        // ATOI format: chr, primary_pos, strand
        let pos_idx = field_names
            .iter()
            .position(|f| f.as_ref() == "primary_pos")
            .unwrap();
        let strand_idx = field_names
            .iter()
            .position(|f| f.as_ref() == "strand")
            .ok_or_else(|| anyhow::anyhow!("missing 'strand' column in {}", site_file))?;

        for record in row_iter {
            let row = record?;
            let chr: Box<str> = row.get_string(chr_idx)?.clone().into_boxed_str();
            let position = row.get_long(pos_idx)?;
            let strand_str = row.get_string(strand_idx)?;
            let strand = if strand_str == "+" {
                Strand::Forward
            } else {
                Strand::Backward
            };
            sites.push(GenomicSite {
                chr,
                position,
                strand,
            });
        }
    } else {
        return Err(anyhow::anyhow!(
            "unrecognized parquet format: expected 'm6a_pos' (dart), 'genomic_alpha' (apa), or 'primary_pos' (atoi) column"
        ));
    }

    // Deduplicate by (chr, position)
    let mut seen = FnvHashSet::default();
    sites.retain(|s| seen.insert((s.chr.clone(), s.position)));

    info!("loaded {} unique sites from {}", sites.len(), site_file);
    Ok(sites)
}
