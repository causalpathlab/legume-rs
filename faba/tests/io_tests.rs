use faba::util::gff::GffVec;

#[test]
fn gff_test() -> anyhow::Result<()> {
    use faba::util;

    let file_path = "test_data/gencode.v48.chr_patch_hapl_scaff.basic.annotation.gff3.gz";
    let genes = GffVec::from_file(file_path)?;

    for g in genes.into_iter().take(5) {
        println!("{:?}", g);
    }

    Ok(())
}

#[test]
fn read_bam_test() -> anyhow::Result<()> {
    use rust_htslib::bam::{self, Read};

    let bam_file = "test_data/WT_Soma_Ctrl1_Apc.bam";
    let index_file = "test_data/WT_Soma_Ctrl1_Apc.bam.bai";
    let br = bam::IndexedReader::from_path_and_index(bam_file, &index_file)?;

    let hdr = br.header();

    for (tid, name) in hdr.target_names().into_iter().enumerate() {
        let max_size = hdr.target_len(tid as u32).unwrap() as i64;
        let name_ = String::from_utf8(name.to_vec()).unwrap();
        let chr_name = name_.into_boxed_str();

        println!("{}:{}", chr_name, max_size);
    }

    Ok(())
}
