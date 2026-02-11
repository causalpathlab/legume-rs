#![allow(dead_code)]

const DEFAULT_BLOCK_SIZE: usize = 100_000;

use super::dna::Dna;
use rust_htslib::bam::{self, Read};
use rust_htslib::faidx;
use std::path::Path;
use std::thread;

/// Check random access BAM index. If the index file doesn't exist, it
/// will create one with `.bai`
pub fn check_bam_index(bam_file_name: &str, idx_file_name: Option<&str>) -> anyhow::Result<()> {
    let idx_file = match idx_file_name {
        Some(x) => String::from(x),
        None => format!("{}.bai", bam_file_name),
    };

    if Path::new(&idx_file).exists() {
        return Ok(());
    }

    let ncore = thread::available_parallelism()
        .expect("failed to figure out number of cores")
        .get();

    // need to build an index for this bam file
    bam::index::build(
        bam_file_name,
        Some(&idx_file),
        bam::index::Type::Bai,
        ncore as u32,
    )?;

    Ok(())
}

///
/// * `bam_file_name` - BAM file name
/// * `block_size` - each job's size
/// * `overlap` - overlap to the left and right
///
pub fn create_bam_jobs(
    bam_file_name: &str,
    block_size: Option<usize>,
    overlap: Option<usize>,
) -> anyhow::Result<Vec<(Box<str>, i64, i64)>> {
    let br = bam::Reader::from_path(bam_file_name)
        .unwrap_or_else(|_| panic!("failed to initialize BAM file: {}", bam_file_name));

    let hdr = br.header();

    let block_size = block_size.unwrap_or(DEFAULT_BLOCK_SIZE) as i64;
    let overlap = overlap.unwrap_or(0) as i64;

    let mut ret = Vec::with_capacity(hdr.target_names().len());

    for (tid, name) in hdr.target_names().iter().enumerate() {
        let max_size = hdr.target_len(tid as u32).unwrap() as i64;
        let name_ = String::from_utf8(name.to_vec()).unwrap();
        let chr_name = name_.into_boxed_str();
        let nblock = (max_size as usize).div_ceil(block_size as usize) as i64;

        let jobs = (0..nblock)
            .map(|block| {
                let lb = if block * block_size > overlap {
                    block * block_size - overlap
                } else {
                    block * block_size
                };

                let ub = ((block + 1) * block_size + overlap).min(max_size);

                (chr_name.clone(), lb, ub)
            })
            .collect::<Vec<_>>();
        ret.extend(jobs);
    }

    Ok(ret)
}

/// Load and index a FASTA file for random access
/// If the .fai index doesn't exist, it will be created automatically
pub fn load_fasta_index(fasta_file: &str) -> anyhow::Result<faidx::Reader> {
    faidx::Reader::from_path(fasta_file)
        .map_err(|e| anyhow::anyhow!("Failed to load FASTA file {}: {}", fasta_file, e))
}

/// Fetch reference base at a specific position (0-based coordinates)
/// Returns None if position is out of bounds or chromosome not found
pub fn fetch_reference_base(
    faidx: &faidx::Reader,
    chr: &str,
    pos: i64,
) -> anyhow::Result<Option<Dna>> {
    if pos < 0 {
        return Ok(None);
    }

    let pos = pos as usize;

    match faidx.fetch_seq(chr, pos, pos) {
        Ok(seq) => Ok(seq
            .first()
            .and_then(|&b| Dna::from_byte(b.to_ascii_uppercase()))),
        Err(_) => Ok(None), // chromosome not found or position out of bounds
    }
}

/// Fetch reference sequence from start to end (0-based, inclusive)
/// Returns None if positions are invalid or chromosome not found
pub fn fetch_reference_seq(
    faidx: &faidx::Reader,
    chr: &str,
    start: i64,
    end: i64,
) -> anyhow::Result<Option<Vec<Dna>>> {
    if start < 0 || end < 0 || start > end {
        return Ok(None);
    }

    let start = start as usize;
    let end = end as usize;

    match faidx.fetch_seq(chr, start, end) {
        Ok(seq) => {
            let bases: Vec<Dna> = seq
                .iter()
                .filter_map(|&b| Dna::from_byte(b.to_ascii_uppercase()))
                .collect();
            if bases.is_empty() {
                Ok(None)
            } else {
                Ok(Some(bases))
            }
        }
        Err(_) => Ok(None),
    }
}
