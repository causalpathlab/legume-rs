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

        ret.extend(
            contig_blocks(max_size, block_size, overlap)
                .into_iter()
                .map(|(lb, ub)| (chr_name.clone(), lb, ub)),
        );
    }

    Ok(ret)
}

/// Partition one contig of length `max_size` into `[lb, ub)` blocks of
/// `block_size`, each padded by `overlap` on both sides and clamped to the contig.
///
/// Blocks always START at a multiple of `block_size` (before padding), which is
/// what lets a caller align a finer grid to the block grid: `faba depth` tiles
/// bins inside these blocks, so a block size that is a whole number of bins keeps
/// every bin edge genome-absolute. Split out of [`create_bam_jobs`] so that
/// property is testable without a BAM header.
pub fn contig_blocks(max_size: i64, block_size: i64, overlap: i64) -> Vec<(i64, i64)> {
    let nblock = (max_size as usize).div_ceil(block_size as usize) as i64;

    (0..nblock)
        .map(|block| {
            let lb = if block * block_size > overlap {
                block * block_size - overlap
            } else {
                block * block_size
            };

            let ub = ((block + 1) * block_size + overlap).min(max_size);

            (lb, ub)
        })
        .collect()
}

/// Load and index a FASTA file for random access
/// If the .fai index doesn't exist, it will be created automatically
pub fn load_fasta_index(fasta_file: &str) -> anyhow::Result<faidx::Reader> {
    faidx::Reader::from_path(fasta_file)
        .map_err(|e| anyhow::anyhow!("Failed to load FASTA file {}: {}", fasta_file, e))
}

/// Fetch reference base at a specific position (0-based coordinates, matching BAM/DnaBaseFreqMap keys)
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

/// Reference bases over `[start, end]` (0-based, inclusive), **preserving
/// length**: anything that is
/// not A/C/G/T — an assembly gap `N`, an IUPAC ambiguity code — becomes `None`
/// rather than vanishing.
///
/// [`fetch_reference_seq`] `filter_map`s those away, which silently *shortens*
/// the returned vector. That is harmless when the caller only wants base
/// composition, and wrong when it maps an index back to a genomic coordinate:
/// after the first `N`, every `pos = start + i` is off by the number of skipped
/// bases. hg38 gene spans do overlap gaps, so use this whenever the index is a
/// coordinate.
pub fn fetch_reference_bases(
    faidx: &faidx::Reader,
    chr: &str,
    start: i64,
    end: i64,
) -> anyhow::Result<Option<Vec<Option<Dna>>>> {
    if start < 0 || end < 0 || start > end {
        return Ok(None);
    }
    match faidx.fetch_seq(chr, start as usize, end as usize) {
        Ok(seq) if !seq.is_empty() => Ok(Some(
            seq.iter()
                .map(|&b| Dna::from_byte(b.to_ascii_uppercase()))
                .collect(),
        )),
        _ => Ok(None),
    }
}

/// Fetch reference sequence from start to end (0-based, inclusive), keeping
/// only A/C/G/T. Returns None if positions are invalid or chromosome not found.
///
/// Drops non-ACGT bases instead of preserving them, so the result can be
/// SHORTER than `end - start + 1` — see [`fetch_reference_bases`] when the index
/// has to stay a coordinate.
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
