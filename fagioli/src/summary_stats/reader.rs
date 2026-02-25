use anyhow::Result;
use log::{info, warn};
use nalgebra::DMatrix;
use rust_htslib::bgzf;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use crate::util::{alleles_match, chr_stripped, AlleleMatch};

/// Column indices parsed from a sumstat header.
struct SumstatColumns {
    chr: usize,
    /// "start" column (0-based BED coordinate).
    _start: Option<usize>,
    /// "end" column (1-based position).
    end: usize,
    snp_id: Option<usize>,
    a1: Option<usize>,
    a2: Option<usize>,
    trait_idx: Option<usize>,
    n: Option<usize>,
    z: usize,
}

/// Default fixed-column layout (matches our writer output):
/// chr(0) start(1) end(2) snp_id(3) a1(4) a2(5) trait_idx(6) n(7) beta(8) se(9) z(10) pvalue(11)
const DEFAULT_COLS: SumstatColumns = SumstatColumns {
    chr: 0,
    _start: Some(1),
    end: 2,
    snp_id: Some(3),
    a1: Some(4),
    a2: Some(5),
    trait_idx: Some(6),
    n: Some(7),
    z: 10,
};

/// Old fixed-column layout (no allele columns):
/// chr(0) start(1) end(2) snp_id(3) trait_idx(4) n(5) beta(6) se(7) z(8) pvalue(9)
const LEGACY_COLS: SumstatColumns = SumstatColumns {
    chr: 0,
    _start: Some(1),
    end: 2,
    snp_id: Some(3),
    a1: None,
    a2: None,
    trait_idx: Some(4),
    n: Some(5),
    z: 8,
};

/// Parse a header line (starting with '#') and return column indices.
fn parse_header(header: &str) -> Option<SumstatColumns> {
    let header = header.trim_start_matches('#');
    let cols: Vec<&str> = header.split('\t').collect();
    if cols.len() < 3 {
        return None;
    }

    let find = |names: &[&str]| -> Option<usize> {
        cols.iter().position(|c| {
            let lower = c.to_ascii_lowercase();
            names.iter().any(|n| lower == *n)
        })
    };

    let chr = find(&["chr", "chrom", "chromosome"])?;
    let end = find(&["end", "pos", "position", "bp"])?;
    let z = find(&["z", "zscore", "z_score", "z_stat"])?;

    Some(SumstatColumns {
        chr,
        _start: find(&["start"]),
        end,
        snp_id: find(&["snp_id", "snp", "rsid", "id", "variant_id"]),
        a1: find(&["a1", "ea", "effect_allele", "alt"]),
        a2: find(&["a2", "nea", "non_effect_allele", "other_allele", "ref"]),
        trait_idx: find(&["trait_idx", "trait", "phenotype_idx"]),
        n: find(&["n", "sample_size", "nobs"]),
        z,
    })
}

/// Read z-scores from a BGZF-compressed summary statistics file.
///
/// Returns a matrix of z-scores (M × T) aligned to the reference panel SNP ordering.
/// SNPs in the sumstat file that are not in the reference panel are skipped.
/// Reference panel SNPs missing from the sumstat file get z-score = 0.
pub fn read_sumstat_zscores(path: &str, ref_snp_ids: &[Box<str>]) -> Result<DMatrix<f32>> {
    let (zscores, _) = read_sumstat_zscores_with_n(
        path,
        ref_snp_ids,
        &[], // no chromosomes
        &[], // no positions
        &[], // no allele1
        &[], // no allele2
    )?;
    Ok(zscores)
}

/// Read z-scores and median sample size from a BGZF-compressed summary statistics file.
///
/// When ref_chromosomes/ref_positions are provided, primary matching uses (chr, position).
/// Falls back to snp_id matching otherwise. When ref_allele1/ref_allele2 are provided,
/// allele alignment is performed (flipping z-scores when the effect allele is swapped,
/// dropping strand-ambiguous or mismatched SNPs).
pub fn read_sumstat_zscores_with_n(
    path: &str,
    ref_snp_ids: &[Box<str>],
    ref_chromosomes: &[Box<str>],
    ref_positions: &[u64],
    ref_allele1: &[Box<str>],
    ref_allele2: &[Box<str>],
) -> Result<(DMatrix<f32>, u64)> {
    let reader = bgzf::Reader::from_path(path)?;
    let buf_reader = BufReader::new(reader);

    let m = ref_snp_ids.len();
    let has_pos = ref_chromosomes.len() == m && ref_positions.len() == m;
    let has_alleles = ref_allele1.len() == m && ref_allele2.len() == m;

    // Build chr+pos lookup: (chr_stripped, position) → ref index
    let mut pos_to_idx: HashMap<(&str, u64), usize> = HashMap::new();
    if has_pos {
        for i in 0..m {
            let key = (chr_stripped(&ref_chromosomes[i]), ref_positions[i]);
            pos_to_idx.entry(key).or_insert(i);
        }
    }

    // Build snp_id lookup
    let snp_to_idx: HashMap<&str, usize> = ref_snp_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_ref(), i))
        .collect();

    let mut max_trait_idx: usize = 0;
    let mut records: Vec<(usize, usize, f32)> = Vec::new(); // (snp_idx, trait_idx, z)
    let mut sample_sizes: Vec<u64> = Vec::new();
    let mut cols: Option<SumstatColumns> = None;
    let mut n_ambiguous: usize = 0;
    let mut n_mismatch: usize = 0;
    let mut n_flipped: usize = 0;
    let mut n_matched_pos: usize = 0;
    let mut n_matched_id: usize = 0;

    for line in buf_reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse header
        if line.starts_with('#') {
            cols = parse_header(line);
            continue;
        }

        // Auto-detect columns from first data line if no header was parsed
        let c = cols.get_or_insert_with(|| {
            let fields: Vec<&str> = line.split('\t').collect();
            // Heuristic: if field count matches new format (12 cols), use default; else legacy
            if fields.len() >= 12 {
                DEFAULT_COLS
            } else {
                LEGACY_COLS
            }
        });

        let fields: Vec<&str> = line.split('\t').collect();
        let min_fields = [c.chr, c.end, c.z]
            .into_iter()
            .chain(c.snp_id)
            .chain(c.a1)
            .chain(c.a2)
            .chain(c.trait_idx)
            .chain(c.n)
            .max()
            .unwrap_or(0)
            + 1;

        if fields.len() < min_fields {
            anyhow::bail!(
                "Sumstat line has {} fields, expected at least {}: {}",
                fields.len(),
                min_fields,
                line
            );
        }

        let chr = fields[c.chr];
        let pos: u64 = fields[c.end].parse()?;
        let snp_id = c.snp_id.map(|i| fields[i]);
        let sum_a1 = c.a1.map(|i| fields[i]);
        let sum_a2 = c.a2.map(|i| fields[i]);
        let trait_idx: usize = c
            .trait_idx
            .map(|i| fields[i].parse())
            .transpose()?
            .unwrap_or(0);
        let n: u64 = c.n.map(|i| fields[i].parse()).transpose()?.unwrap_or(0);
        let z: f32 = fields[c.z].parse()?;

        // Match to reference panel: prefer chr+pos, fall back to snp_id
        let ref_idx = if has_pos {
            let key = (chr_stripped(chr), pos);
            if let Some(&idx) = pos_to_idx.get(&key) {
                n_matched_pos += 1;
                Some(idx)
            } else if let Some(id) = snp_id {
                if let Some(&idx) = snp_to_idx.get(id) {
                    n_matched_id += 1;
                    Some(idx)
                } else {
                    None
                }
            } else {
                None
            }
        } else if let Some(id) = snp_id {
            snp_to_idx.get(id).copied().inspect(|_| {
                n_matched_id += 1;
            })
        } else {
            None
        };

        let Some(snp_idx) = ref_idx else {
            continue;
        };

        // Allele alignment
        let mut z_final = z;
        if has_alleles {
            if let (Some(sa1), Some(sa2)) = (sum_a1, sum_a2) {
                match alleles_match(sa1, sa2, &ref_allele1[snp_idx], &ref_allele2[snp_idx]) {
                    AlleleMatch::Same => {}
                    AlleleMatch::Flipped => {
                        z_final = -z;
                        n_flipped += 1;
                    }
                    AlleleMatch::Ambiguous => {
                        n_ambiguous += 1;
                        continue;
                    }
                    AlleleMatch::Mismatch => {
                        n_mismatch += 1;
                        continue;
                    }
                }
            }
            // If sumstat has no allele columns but ref does, keep z as-is (no alignment possible)
        }

        max_trait_idx = max_trait_idx.max(trait_idx);
        records.push((snp_idx, trait_idx, z_final));
        sample_sizes.push(n);
    }

    let num_traits = if records.is_empty() {
        1
    } else {
        max_trait_idx + 1
    };

    sample_sizes.sort_unstable();
    let median_n = if sample_sizes.is_empty() {
        0
    } else {
        sample_sizes[sample_sizes.len() / 2]
    };

    let n_unique_snps = records
        .iter()
        .map(|(s, _, _)| s)
        .collect::<std::collections::HashSet<_>>()
        .len();

    info!(
        "Read {} z-score records for {} traits, {} matched reference SNPs (pos: {}, id: {}), median N={}",
        records.len(),
        num_traits,
        n_unique_snps,
        n_matched_pos,
        n_matched_id,
        median_n,
    );

    if n_flipped > 0 {
        info!("Flipped z-scores for {} allele-swapped records", n_flipped);
    }
    if n_ambiguous > 0 {
        warn!(
            "Dropped {} records with strand-ambiguous alleles (A/T or C/G)",
            n_ambiguous
        );
    }
    if n_mismatch > 0 {
        warn!("Dropped {} records with mismatched alleles", n_mismatch);
    }

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

    /// Helper to write a BGZF file and return its path.
    fn write_bgzf(lines: &[&str]) -> String {
        let tmp = NamedTempFile::new().unwrap();
        let path = format!("{}.gz", tmp.path().to_str().unwrap());
        let mut writer = bgzf::Writer::from_path(&path).unwrap();
        for line in lines {
            writeln!(writer, "{}", line).unwrap();
        }
        writer.flush().unwrap();
        // Keep the tempfile alive by leaking it (tests are short-lived)
        std::mem::forget(tmp);
        path
    }

    #[test]
    fn test_read_sumstat_zscores_legacy_format() {
        // Old format without allele columns
        let path = write_bgzf(&[
            "#chr\tstart\tend\tsnp_id\ttrait_idx\tn\tbeta\tse\tz\tpvalue",
            "chr1\t99\t100\trs1\t0\t100\t0.1\t0.05\t2.0\t0.05",
            "chr1\t99\t100\trs1\t1\t100\t0.2\t0.05\t4.0\t0.001",
            "chr1\t199\t200\trs2\t0\t100\t-0.1\t0.05\t-2.0\t0.05",
            "chr1\t199\t200\trs2\t1\t100\t0.05\t0.05\t1.0\t0.3",
            "chr1\t299\t300\trs3\t0\t100\t0.3\t0.05\t6.0\t0.0001",
        ]);

        let ref_snp_ids: Vec<Box<str>> =
            vec![Box::from("rs1"), Box::from("rs2"), Box::from("rs_missing")];

        let zscores = read_sumstat_zscores(&path, &ref_snp_ids).unwrap();

        assert_eq!(zscores.nrows(), 3);
        assert_eq!(zscores.ncols(), 2);
        assert!((zscores[(0, 0)] - 2.0).abs() < 1e-6); // rs1, trait 0
        assert!((zscores[(0, 1)] - 4.0).abs() < 1e-6); // rs1, trait 1
        assert!((zscores[(1, 0)] - (-2.0)).abs() < 1e-6); // rs2, trait 0
        assert!((zscores[(2, 0)]).abs() < 1e-6); // rs_missing -> 0

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_sumstat_chr_pos_matching() {
        // New format with allele columns, match by chr+pos
        let path = write_bgzf(&[
            "#chr\tstart\tend\tsnp_id\ta1\ta2\ttrait_idx\tn\tbeta\tse\tz\tpvalue",
            // Different snp_id but same chr+pos → should match by position
            "chr1\t99\t100\trs1_different_id\tA\tC\t0\t100\t0.1\t0.05\t2.0\t0.05",
            // Mixed chr prefix ("1" vs ref "chr1") → should still match by position
            "1\t199\t200\trs2\tG\tT\t0\t100\t-0.1\t0.05\t-2.0\t0.05",
        ]);

        let ref_snp_ids: Vec<Box<str>> = vec![Box::from("rs1"), Box::from("rs2")];
        let ref_chrs: Vec<Box<str>> = vec![Box::from("chr1"), Box::from("chr1")];
        let ref_pos: Vec<u64> = vec![100, 200];
        let ref_a1: Vec<Box<str>> = vec![Box::from("A"), Box::from("G")];
        let ref_a2: Vec<Box<str>> = vec![Box::from("C"), Box::from("T")];

        let (zscores, _) =
            read_sumstat_zscores_with_n(&path, &ref_snp_ids, &ref_chrs, &ref_pos, &ref_a1, &ref_a2)
                .unwrap();

        assert_eq!(zscores.nrows(), 2);
        // rs1 matched by chr+pos despite different snp_id, alleles same
        assert!((zscores[(0, 0)] - 2.0).abs() < 1e-6);
        // rs2 matched by chr+pos with mixed chr prefix, alleles same
        assert!((zscores[(1, 0)] - (-2.0)).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_sumstat_allele_flip() {
        let path = write_bgzf(&[
            "#chr\tstart\tend\tsnp_id\ta1\ta2\ttrait_idx\tn\tbeta\tse\tz\tpvalue",
            // Same alleles as ref
            "chr1\t99\t100\trs1\tA\tC\t0\t100\t0.1\t0.05\t2.0\t0.05",
            // Flipped alleles relative to ref (a1/a2 swapped)
            "chr1\t199\t200\trs2\tG\tA\t0\t100\t-0.1\t0.05\t-3.0\t0.01",
        ]);

        let ref_snp_ids: Vec<Box<str>> = vec![Box::from("rs1"), Box::from("rs2")];
        let ref_chrs: Vec<Box<str>> = vec![Box::from("chr1"), Box::from("chr1")];
        let ref_pos: Vec<u64> = vec![100, 200];
        let ref_a1: Vec<Box<str>> = vec![Box::from("A"), Box::from("A")];
        let ref_a2: Vec<Box<str>> = vec![Box::from("C"), Box::from("G")];

        let (zscores, _) =
            read_sumstat_zscores_with_n(&path, &ref_snp_ids, &ref_chrs, &ref_pos, &ref_a1, &ref_a2)
                .unwrap();

        assert_eq!(zscores.nrows(), 2);
        // rs1: same alleles, z unchanged
        assert!((zscores[(0, 0)] - 2.0).abs() < 1e-6);
        // rs2: flipped alleles, z negated: -(-3.0) = 3.0
        assert!((zscores[(1, 0)] - 3.0).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_sumstat_strand_ambiguous_dropped() {
        let path = write_bgzf(&[
            "#chr\tstart\tend\tsnp_id\ta1\ta2\ttrait_idx\tn\tbeta\tse\tz\tpvalue",
            // A/T is strand-ambiguous
            "chr1\t99\t100\trs1\tA\tT\t0\t100\t0.1\t0.05\t2.0\t0.05",
            // C/G is strand-ambiguous
            "chr1\t199\t200\trs2\tC\tG\t0\t100\t-0.1\t0.05\t-2.0\t0.05",
            // A/G is fine
            "chr1\t299\t300\trs3\tA\tG\t0\t100\t0.3\t0.05\t5.0\t0.001",
        ]);

        let ref_snp_ids: Vec<Box<str>> = vec![Box::from("rs1"), Box::from("rs2"), Box::from("rs3")];
        let ref_chrs: Vec<Box<str>> = vec![Box::from("chr1"), Box::from("chr1"), Box::from("chr1")];
        let ref_pos: Vec<u64> = vec![100, 200, 300];
        let ref_a1: Vec<Box<str>> = vec![Box::from("A"), Box::from("C"), Box::from("A")];
        let ref_a2: Vec<Box<str>> = vec![Box::from("T"), Box::from("G"), Box::from("G")];

        let (zscores, _) =
            read_sumstat_zscores_with_n(&path, &ref_snp_ids, &ref_chrs, &ref_pos, &ref_a1, &ref_a2)
                .unwrap();

        assert_eq!(zscores.nrows(), 3);
        // rs1 and rs2 dropped (strand-ambiguous) → 0
        assert!((zscores[(0, 0)]).abs() < 1e-6);
        assert!((zscores[(1, 0)]).abs() < 1e-6);
        // rs3 kept
        assert!((zscores[(2, 0)] - 5.0).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_sumstat_mixed_chr_prefix() {
        let path = write_bgzf(&[
            "#chr\tstart\tend\tsnp_id\ta1\ta2\ttrait_idx\tn\tbeta\tse\tz\tpvalue",
            // Sumstat uses "chr1" while ref uses "1"
            "chr1\t99\t100\trs1\tA\tC\t0\t100\t0.1\t0.05\t2.0\t0.05",
            // Sumstat uses "1" while ref uses "chr1"
            "1\t199\t200\trs2\tA\tG\t0\t100\t-0.1\t0.05\t-2.0\t0.05",
        ]);

        let ref_snp_ids: Vec<Box<str>> = vec![Box::from("rs1"), Box::from("rs2")];
        let ref_chrs: Vec<Box<str>> = vec![Box::from("1"), Box::from("chr1")];
        let ref_pos: Vec<u64> = vec![100, 200];
        let ref_a1: Vec<Box<str>> = vec![Box::from("A"), Box::from("A")];
        let ref_a2: Vec<Box<str>> = vec![Box::from("C"), Box::from("G")];

        let (zscores, _) =
            read_sumstat_zscores_with_n(&path, &ref_snp_ids, &ref_chrs, &ref_pos, &ref_a1, &ref_a2)
                .unwrap();

        assert_eq!(zscores.nrows(), 2);
        assert!((zscores[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((zscores[(1, 0)] - (-2.0)).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }
}
