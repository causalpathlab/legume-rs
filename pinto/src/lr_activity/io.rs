//! Parquet + TSV readers for `pinto lr-activity`.
//!
//! Reads:
//! - `{lc_prefix}.link_community.parquet` — per-edge (left_cell, right_cell, community).
//! - `{lc_prefix}.coord_pairs.parquet` — per-edge (left_batch, right_batch) when multi-batch.
//! - `{lr_pairs}` — two-column TSV/CSV of directional (ligand, receptor) gene names.

use crate::util::common::*;
use matrix_util::common_io::{read_lines_of_words_delim, ReadLinesOut};
use matrix_util::membership::detect_delimiter;
use matrix_util::parquet::peek_parquet_field_names;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;

/// Per-edge record joined across `link_community.parquet` and `coord_pairs.parquet`.
pub struct EdgeRecord {
    pub left_cell: Box<str>,
    pub right_cell: Box<str>,
    pub community: u32,
    /// If either `left_batch` or `right_batch` is absent the whole fit is
    /// treated as single-batch and this is `None`.
    pub batch: Option<Box<str>>,
}

/// Read `{prefix}.link_community.parquet` (columns: left_cell, right_cell, community).
pub fn read_link_community(file_path: &str) -> anyhow::Result<Vec<EdgeRecord>> {
    let file = File::open(file_path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let name_to_idx: HashMap<Box<str>, usize> = schema
        .get_fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().to_string().into_boxed_str(), i))
        .collect();

    let li = *name_to_idx
        .get("left_cell")
        .ok_or_else(|| anyhow::anyhow!("left_cell missing in {file_path}"))?;
    let ri = *name_to_idx
        .get("right_cell")
        .ok_or_else(|| anyhow::anyhow!("right_cell missing in {file_path}"))?;
    let ci = *name_to_idx
        .get("community")
        .ok_or_else(|| anyhow::anyhow!("community missing in {file_path}"))?;

    let mut out = Vec::new();
    for record in reader.get_row_iter(None)? {
        let row = record?;
        let community_f = row.get_float(ci)?;
        out.push(EdgeRecord {
            left_cell: row.get_string(li)?.clone().into_boxed_str(),
            right_cell: row.get_string(ri)?.clone().into_boxed_str(),
            community: community_f as u32,
            batch: None,
        });
    }
    Ok(out)
}

/// Attach per-edge batch labels from `{prefix}.coord_pairs.parquet` when the fit
/// was multi-batch. Matches edges by position (row order). If either
/// `left_batch` or `right_batch` is missing, leaves all `batch` as `None`.
pub fn attach_batch_from_coord_pairs(
    edges: &mut [EdgeRecord],
    coord_pairs_path: &str,
) -> anyhow::Result<()> {
    let fields = peek_parquet_field_names(coord_pairs_path)?;
    let has_batch = fields.iter().any(|f| f.as_ref() == "left_batch")
        && fields.iter().any(|f| f.as_ref() == "right_batch");
    if !has_batch {
        return Ok(());
    }

    let file = File::open(coord_pairs_path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let name_to_idx: HashMap<Box<str>, usize> = schema
        .get_fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().to_string().into_boxed_str(), i))
        .collect();

    let li = *name_to_idx.get("left_batch").unwrap();
    let ri = *name_to_idx.get("right_batch").unwrap();

    for (pos, record) in reader.get_row_iter(None)?.enumerate() {
        if pos >= edges.len() {
            return Err(anyhow::anyhow!(
                "coord_pairs.parquet has more rows than link_community.parquet"
            ));
        }
        let row = record?;
        let left_b = row.get_string(li)?.clone();
        let right_b = row.get_string(ri)?.clone();
        // Edge is single-batch iff both endpoints share a label. We only test
        // within-batch activity; cross-batch edges are dropped downstream by
        // assigning the joint label `None`.
        if left_b == right_b {
            edges[pos].batch = Some(left_b.into_boxed_str());
        } else {
            edges[pos].batch = None;
        }
    }

    Ok(())
}

/// Header keywords used to detect and skip a title row on the LR pairs file.
const LR_HEADER_KEYWORDS: &[&str] = &["ligand", "source", "receptor", "target"];

/// Parse a directional LR pairs file (two columns: ligand, receptor).
/// Delimiter is auto-detected from extension. Skips empty lines and lines with
/// fewer than two non-empty tokens. Trims whitespace.
pub fn read_lr_pairs(file_path: &str) -> anyhow::Result<Vec<(Box<str>, Box<str>)>> {
    let delim = detect_delimiter(file_path);
    let ReadLinesOut { lines, .. } = read_lines_of_words_delim(file_path, delim, -1)?;
    let mut out = Vec::with_capacity(lines.len());
    for row in lines {
        let cleaned: Vec<Box<str>> = row
            .into_iter()
            .map(|s| s.trim().to_string().into_boxed_str())
            .filter(|s| !s.is_empty())
            .collect();
        if cleaned.len() < 2 {
            continue;
        }
        let looks_header = out.is_empty()
            && [cleaned[0].as_ref(), cleaned[1].as_ref()]
                .iter()
                .any(|cell| {
                    LR_HEADER_KEYWORDS
                        .iter()
                        .any(|&kw| cell.eq_ignore_ascii_case(kw))
                });
        if looks_header {
            continue;
        }
        out.push((cleaned[0].clone(), cleaned[1].clone()));
    }
    Ok(out)
}
