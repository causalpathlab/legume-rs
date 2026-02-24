use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};

use super::GenotypeMatrix;

const BED_MAGIC: [u8; 3] = [0x6C, 0x1B, 0x01];

/// Write a `GenotypeMatrix` to PLINK BED/BIM/FAM files.
///
/// `prefix` is the path prefix; files `{prefix}.bed`, `{prefix}.bim`, and
/// `{prefix}.fam` will be created.
///
/// Encoding (SNP-major, count_a1=true, matching genomic-data reader):
///   2.0 → 0b00, 1.0 → 0b10, 0.0 → 0b11, NaN → 0b01
pub fn write_plink(prefix: &str, geno: &GenotypeMatrix) -> Result<()> {
    write_bed(prefix, geno)?;
    write_bim(prefix, geno)?;
    write_fam(prefix, geno)?;
    Ok(())
}

fn encode_genotype(val: f32) -> u8 {
    if val.is_nan() {
        0b01 // missing
    } else {
        let v = val.round() as i32;
        match v {
            2 => 0b00, // hom secondary (count_a1: 2 copies of A1)
            1 => 0b10, // het
            0 => 0b11, // hom primary
            _ => 0b01, // treat out-of-range as missing
        }
    }
}

fn write_bed(prefix: &str, geno: &GenotypeMatrix) -> Result<()> {
    let path = format!("{}.bed", prefix);
    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&BED_MAGIC)?;

    let n = geno.num_individuals();
    let bytes_per_snp = n.div_ceil(4);

    for j in 0..geno.num_snps() {
        let mut buf = vec![0u8; bytes_per_snp];
        for i in 0..n {
            let bits = encode_genotype(geno.genotypes[(i, j)]);
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            buf[byte_idx] |= bits << bit_offset;
        }
        w.write_all(&buf)?;
    }

    w.flush()?;
    Ok(())
}

fn write_bim(prefix: &str, geno: &GenotypeMatrix) -> Result<()> {
    let path = format!("{}.bim", prefix);
    let mut w = BufWriter::new(File::create(&path)?);

    for j in 0..geno.num_snps() {
        writeln!(
            w,
            "{}\t{}\t0\t{}\t{}\t{}",
            geno.chromosomes[j],
            geno.snp_ids[j],
            geno.positions[j],
            geno.allele1[j],
            geno.allele2[j]
        )?;
    }

    w.flush()?;
    Ok(())
}

fn write_fam(prefix: &str, geno: &GenotypeMatrix) -> Result<()> {
    let path = format!("{}.fam", prefix);
    let mut w = BufWriter::new(File::create(&path)?);

    for iid in &geno.individual_ids {
        writeln!(w, "{}\t{}\t0\t0\t0\t-9", iid, iid)?;
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use genomic_data::plink::PlinkBed;
    use nalgebra::DMatrix;

    #[test]
    fn test_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("test_geno");
        let prefix_str = prefix.to_str().unwrap();

        // 4 individuals, 3 SNPs — includes padding edge case (4 fits exactly)
        let data = vec![
            2.0, 1.0, 0.0, 1.0, // SNP0
            0.0, 0.0, 2.0, 2.0, // SNP1
            1.0, 2.0, 1.0, 0.0, // SNP2
        ];
        let genotypes = DMatrix::from_vec(4, 3, data);

        let geno = GenotypeMatrix {
            genotypes,
            individual_ids: (0..4).map(|i| format!("ind{}", i).into()).collect(),
            snp_ids: (0..3).map(|j| format!("snp{}", j).into()).collect(),
            chromosomes: vec!["1".into(); 3],
            positions: vec![1000, 2000, 3000],
            allele1: vec!["A".into(); 3],
            allele2: vec!["T".into(); 3],
        };

        write_plink(prefix_str, &geno).unwrap();

        // Read back with PlinkBed
        let plink = PlinkBed::new(prefix_str).unwrap();
        assert_eq!(plink.iid_count(), 4);
        assert_eq!(plink.sid_count(), 3);

        let read_mat = plink.read_f32(None).unwrap();
        assert_eq!(read_mat.nrows(), 4);
        assert_eq!(read_mat.ncols(), 3);

        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(
                    read_mat[(i, j)],
                    geno.genotypes[(i, j)],
                    "mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_write_read_roundtrip_odd_individuals() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("test_odd");
        let prefix_str = prefix.to_str().unwrap();

        // 3 individuals (needs zero-padding in bed), 2 SNPs
        let data = vec![
            2.0, 1.0, 0.0, // SNP0
            0.0, 2.0, 1.0, // SNP1
        ];
        let genotypes = DMatrix::from_vec(3, 2, data);

        let geno = GenotypeMatrix {
            genotypes,
            individual_ids: (0..3).map(|i| format!("ind{}", i).into()).collect(),
            snp_ids: (0..2).map(|j| format!("snp{}", j).into()).collect(),
            chromosomes: vec!["1".into(); 2],
            positions: vec![100, 200],
            allele1: vec!["C".into(); 2],
            allele2: vec!["G".into(); 2],
        };

        write_plink(prefix_str, &geno).unwrap();

        let plink = PlinkBed::new(prefix_str).unwrap();
        let read_mat = plink.read_f32(None).unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(read_mat[(i, j)], geno.genotypes[(i, j)]);
            }
        }
    }
}
