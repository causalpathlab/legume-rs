const DEFAULT_BLOCK_SIZE: usize = 10_000;

use noodles::{bam, bgzf, sam};

/// Check random access BAM index. If the index file doesn't exist, it
/// will create one with `.bai`
pub fn check_bam_index(bam_file_name: &str, idx_file_name: Option<&str>) -> anyhow::Result<()> {
    use bam::bai;
    let idx_file = match idx_file_name {
        Some(x) => String::from(x),
        None => format!("{}.bai", bam_file_name),
    };

    let index = bam::fs::index(bam_file_name)?;

    let idx_file = std::fs::File::create(idx_file)?;
    let mut writer = bai::io::Writer::new(&idx_file);
    writer.write_index(&index)?;

    Ok(())
}

///
/// * `bam_file_name` - BAM file name
/// * `block_size` - each job's size
///
pub fn create_bam_jobs(
    bam_file_name: &str,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<(Box<str>, usize, usize)>> {
    let num_threads = std::thread::available_parallelism()?;

    let mut reader: Box<dyn sam::alignment::io::Read<_>> = Box::new(bam::io::Reader::from(
        bgzf::io::MultithreadedReader::with_worker_count(
            num_threads,
            std::fs::File::open(&bam_file_name)?,
        ),
    ));

    let block_size = block_size.unwrap_or(DEFAULT_BLOCK_SIZE);

    let header = reader.read_alignment_header()?;
    let mut ret = Vec::with_capacity(header.reference_sequences().len());

    for (name, seq) in header.reference_sequences() {
        let max_size = seq.length().get();
        let nblock = max_size.div_ceil(block_size);

        let jobs = (0..nblock)
            .map(|block| {
                let lb = block * block_size;
                let ub = ((block + 1) * block_size).min(max_size);
                (name.to_string().into(), lb, ub)
            })
            .collect::<Vec<_>>();

        ret.extend(jobs);
    }

    Ok(ret)
}
