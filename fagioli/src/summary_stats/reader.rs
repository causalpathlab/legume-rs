use anyhow::Result;
use log::info;
use nalgebra::DMatrix;
use rust_htslib::bgzf;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

/// Read z-scores from a BGZF-compressed summary statistics file.
///
/// Returns a matrix of z-scores (M × T) aligned to the reference panel SNP ordering.
/// SNPs in the sumstat file that are not in the reference panel are skipped.
/// Reference panel SNPs missing from the sumstat file get z-score = 0.
pub fn read_sumstat_zscores(path: &str, ref_snp_ids: &[Box<str>]) -> Result<DMatrix<f32>> {
    let (zscores, _) = read_sumstat_zscores_with_n(path, ref_snp_ids)?;
    Ok(zscores)
}

/// Read z-scores and median sample size from a BGZF-compressed summary statistics file.
///
/// Returns `(zscores M × T, median_n)`. The median sample size is taken across all
/// matched records (column 5 in the sumstat file).
pub fn read_sumstat_zscores_with_n(
    path: &str,
    ref_snp_ids: &[Box<str>],
) -> Result<(DMatrix<f32>, u64)> {
    let reader = bgzf::Reader::from_path(path)?;
    let buf_reader = BufReader::new(reader);

    let snp_to_idx: HashMap<&str, usize> = ref_snp_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_ref(), i))
        .collect();

    let m = ref_snp_ids.len();
    let mut max_trait_idx: usize = 0;
    let mut records: Vec<(usize, usize, f32)> = Vec::new(); // (snp_idx, trait_idx, z)
    let mut sample_sizes: Vec<u64> = Vec::new();

    for line in buf_reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            anyhow::bail!(
                "Sumstat line has {} fields, expected at least 9: {}",
                fields.len(),
                line
            );
        }

        // columns: chr, start, end, snp_id, trait_idx, n, beta, se, z, pvalue
        let snp_id = fields[3];
        let trait_idx: usize = fields[4].parse()?;
        let n: u64 = fields[5].parse()?;
        let z: f32 = fields[8].parse()?;

        if let Some(&snp_idx) = snp_to_idx.get(snp_id) {
            max_trait_idx = max_trait_idx.max(trait_idx);
            records.push((snp_idx, trait_idx, z));
            sample_sizes.push(n);
        }
    }

    let num_traits = max_trait_idx + 1;

    sample_sizes.sort_unstable();
    let median_n = if sample_sizes.is_empty() {
        0
    } else {
        sample_sizes[sample_sizes.len() / 2]
    };

    info!(
        "Read {} z-score records for {} traits, {} matched reference SNPs, median N={}",
        records.len(),
        num_traits,
        records
            .iter()
            .map(|(s, _, _)| s)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        median_n,
    );

    let mut zscores = DMatrix::<f32>::zeros(m, num_traits);
    for (snp_idx, trait_idx, z) in records {
        zscores[(snp_idx, trait_idx)] = z;
    }

    Ok((zscores, median_n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_htslib::bgzf;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_sumstat_zscores() {
        // Create a temp BGZF file with sumstat data
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        let path_gz = format!("{}.gz", path);

        {
            let mut writer = bgzf::Writer::from_path(&path_gz).unwrap();
            writeln!(writer, "#chr\tstart\tend\tsnp_id\ttrait_idx\tn\tbeta\tse\tz\tpvalue")
                .unwrap();
            writeln!(writer, "chr1\t99\t100\trs1\t0\t100\t0.1\t0.05\t2.0\t0.05").unwrap();
            writeln!(writer, "chr1\t99\t100\trs1\t1\t100\t0.2\t0.05\t4.0\t0.001").unwrap();
            writeln!(writer, "chr1\t199\t200\trs2\t0\t100\t-0.1\t0.05\t-2.0\t0.05").unwrap();
            writeln!(writer, "chr1\t199\t200\trs2\t1\t100\t0.05\t0.05\t1.0\t0.3").unwrap();
            // rs3 not in reference panel - should be skipped
            writeln!(writer, "chr1\t299\t300\trs3\t0\t100\t0.3\t0.05\t6.0\t0.0001").unwrap();
            writer.flush().unwrap();
        }

        let ref_snp_ids: Vec<Box<str>> =
            vec![Box::from("rs1"), Box::from("rs2"), Box::from("rs_missing")];

        let zscores = read_sumstat_zscores(&path_gz, &ref_snp_ids).unwrap();

        assert_eq!(zscores.nrows(), 3); // 3 reference SNPs
        assert_eq!(zscores.ncols(), 2); // 2 traits
        assert!((zscores[(0, 0)] - 2.0).abs() < 1e-6); // rs1, trait 0
        assert!((zscores[(0, 1)] - 4.0).abs() < 1e-6); // rs1, trait 1
        assert!((zscores[(1, 0)] - (-2.0)).abs() < 1e-6); // rs2, trait 0
        assert!((zscores[(2, 0)]).abs() < 1e-6); // rs_missing -> 0

        // Cleanup
        let _ = std::fs::remove_file(&path_gz);
    }
}
