const DEFAULT_BLOCK_SIZE: usize = 100_000;

use rust_htslib::bam::{self, Read};
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
        .expect(&format!("failed to initialize BAM file: {}", bam_file_name));

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
