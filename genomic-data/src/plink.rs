//! Pure-Rust PLINK BED/BIM/FAM reader
//!
//! Core reading logic extracted from the `bed-reader` crate (v1.0.6)
//! by the FaST-LMM Team (Apache-2.0), stripped of Python/numpy/pyo3
//! dependencies. See <https://github.com/fastlmm/bed-reader>.

use anyhow::{bail, Context, Result};
use nalgebra::DMatrix;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const BED_FILE_MAGIC1: u8 = 0x6C; // 0b01101100 or 'l'
const BED_FILE_MAGIC2: u8 = 0x1B; // 0b00011011 or <esc>
const CB_HEADER_U64: u64 = 3;
const CB_HEADER_USIZE: usize = 3;

// ---------------------------------------------------------------------------
// Core helpers (from bed-reader lib.rs)
// ---------------------------------------------------------------------------

fn open_and_check(path: &Path) -> Result<(BufReader<File>, [u8; CB_HEADER_USIZE])> {
    let mut buf_reader = BufReader::new(
        File::open(path).with_context(|| format!("Cannot open BED file: {}", path.display()))?,
    );
    let mut bytes_array = [0u8; CB_HEADER_USIZE];
    buf_reader
        .read_exact(&mut bytes_array)
        .with_context(|| format!("BED file too short: {}", path.display()))?;
    if BED_FILE_MAGIC1 != bytes_array[0] || BED_FILE_MAGIC2 != bytes_array[1] {
        bail!("Invalid BED magic bytes in {}", path.display());
    }
    Ok((buf_reader, bytes_array))
}

#[inline]
fn try_div_4(in_iid_count: usize, in_sid_count: usize) -> Result<u64> {
    if in_iid_count == 0 {
        return Ok(0);
    }
    let in_iid_count_div4_u64 = in_iid_count.checked_sub(1).map_or(0, |v| v / 4 + 1) as u64;
    let in_sid_count_u64 = in_sid_count as u64;

    if in_sid_count > 0 && (u64::MAX - CB_HEADER_U64) / in_sid_count_u64 < in_iid_count_div4_u64 {
        bail!(
            "Indexes too big for BED files: {} individuals × {} SNPs",
            in_iid_count,
            in_sid_count
        );
    }

    Ok(in_iid_count_div4_u64)
}

fn set_up_two_bits_to_value(count_a1: bool, missing_value: f32) -> [f32; 4] {
    let homozygous_primary_allele = 0.0f32;
    let heterozygous_allele = 1.0f32;
    let homozygous_secondary_allele = 2.0f32;

    if count_a1 {
        [
            homozygous_secondary_allele, // look-up 0
            missing_value,               // look-up 1
            heterozygous_allele,         // look-up 2
            homozygous_primary_allele,   // look-up 3
        ]
    } else {
        [
            homozygous_primary_allele,   // look-up 0
            missing_value,               // look-up 1
            heterozygous_allele,         // look-up 2
            homozygous_secondary_allele, // look-up 3
        ]
    }
}

#[allow(clippy::type_complexity)]
fn check_and_precompute_iid_index(
    in_iid_count: usize,
    iid_index: &[isize],
) -> Result<(Vec<usize>, Vec<u8>, u64, u64)> {
    let lower_iid_count = -(in_iid_count as isize);
    let upper_iid_count: isize = (in_iid_count as isize) - 1;

    let mut i_div_4_less_start_array = vec![0usize; iid_index.len()];
    let mut i_mod_4_times_2_array = vec![0u8; iid_index.len()];

    for (idx, &in_iid_i_signed) in iid_index.iter().enumerate() {
        let in_iid_i = if (0..=upper_iid_count).contains(&in_iid_i_signed) {
            in_iid_i_signed as usize
        } else if (lower_iid_count..=-1).contains(&in_iid_i_signed) {
            in_iid_count - ((-in_iid_i_signed) as usize)
        } else {
            bail!("iid index {} out of range", in_iid_i_signed);
        };

        i_div_4_less_start_array[idx] = in_iid_i / 4;
        i_mod_4_times_2_array[idx] = (in_iid_i % 4 * 2) as u8;
    }

    let (i_div_4_start, i_div_4_len) =
        if let Some(&min_value) = i_div_4_less_start_array.iter().min() {
            let max_value = *i_div_4_less_start_array.iter().max().unwrap();
            (min_value as u64, (max_value + 1 - min_value) as u64)
        } else {
            (0, 0)
        };

    if i_div_4_start > 0 {
        for x in i_div_4_less_start_array.iter_mut() {
            *x -= i_div_4_start as usize;
        }
    }

    Ok((
        i_div_4_less_start_array,
        i_mod_4_times_2_array,
        i_div_4_start,
        i_div_4_len,
    ))
}

/// Core genotype reader — adapted from bed-reader's `internal_read_no_alloc`.
///
/// Writes into a column-major f32 slice (nalgebra DMatrix layout).
/// `out_data` has shape `out_iid_count × out_sid_count` in column-major order.
#[allow(clippy::too_many_arguments)]
fn internal_read_no_alloc(
    mut buf_reader: BufReader<File>,
    path: &Path,
    in_iid_count: usize,
    in_sid_count: usize,
    is_a1_counted: bool,
    iid_index: &[isize],
    sid_index: &[isize],
    missing_value: f32,
    out_data: &mut [f32],
    out_iid_count: usize,
) -> Result<()> {
    let in_iid_count_div4_u64 = try_div_4(in_iid_count, in_sid_count)?;
    let file_len = buf_reader.get_ref().metadata()?.len();
    let file_len2 = in_iid_count_div4_u64 * (in_sid_count as u64) + CB_HEADER_U64;
    if file_len != file_len2 {
        bail!(
            "BED file {} is ill-formed: expected {} bytes, got {}",
            path.display(),
            file_len2,
            file_len
        );
    }

    let (i_div_4_less_start_array, i_mod_4_times_2_array, i_div_4_start, i_div_4_len) =
        check_and_precompute_iid_index(in_iid_count, iid_index)?;

    let from_two_bits_to_value = set_up_two_bits_to_value(is_a1_counted, missing_value);
    let lower_sid_count = -(in_sid_count as isize);
    let upper_sid_count: isize = (in_sid_count as isize) - 1;

    // Phase 1: Sequential I/O — read packed bytes for each SNP
    let mut snp_bytes: Vec<Vec<u8>> = Vec::with_capacity(sid_index.len());
    for in_sid_i_signed in sid_index.iter() {
        let in_sid_i = if (0..=upper_sid_count).contains(in_sid_i_signed) {
            *in_sid_i_signed as u64
        } else if (lower_sid_count..=-1).contains(in_sid_i_signed) {
            (in_sid_count - ((-in_sid_i_signed) as usize)) as u64
        } else {
            bail!("sid index {} out of range", in_sid_i_signed);
        };

        let mut bytes_vector: Vec<u8> = vec![0; i_div_4_len as usize];
        let pos: u64 = in_sid_i * in_iid_count_div4_u64 + i_div_4_start + CB_HEADER_U64;
        buf_reader.seek(SeekFrom::Start(pos))?;
        buf_reader.read_exact(&mut bytes_vector)?;
        snp_bytes.push(bytes_vector);
    }

    // Phase 2: Parallel decode — each column is independent (mirrors bed-reader's par_bridge)
    let out_iid_len = iid_index.len();
    out_data
        .chunks_exact_mut(out_iid_count)
        .zip(snp_bytes.iter())
        .par_bridge()
        .for_each(|(col, bytes_vector)| {
            for out_iid_i in 0..out_iid_len {
                let i_div_4_less_start = i_div_4_less_start_array[out_iid_i];
                let i_mod_4_times_2 = i_mod_4_times_2_array[out_iid_i];
                let genotype_byte: u8 =
                    (bytes_vector[i_div_4_less_start] >> i_mod_4_times_2) & 0x03;
                col[out_iid_i] = from_two_bits_to_value[genotype_byte as usize];
            }
        });

    Ok(())
}

// ---------------------------------------------------------------------------
// FAM / BIM parsing (from bed-reader's Metadata::read_fam_or_bim)
// ---------------------------------------------------------------------------

/// Parse a whitespace-delimited metadata file (.fam or .bim), extracting the
/// specified field indices (0-based). Returns `(vec_of_columns, row_count)`.
fn read_fam_or_bim(field_indices: &[usize], path: &Path) -> Result<(Vec<Vec<String>>, usize)> {
    let mut vec_of_vec = vec![vec![]; field_indices.len()];
    let file = File::open(path).with_context(|| format!("Cannot open file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut count = 0;

    for line in reader.lines() {
        let line = line?;
        count += 1;

        let fields: Vec<&str> = line.split_whitespace().collect();

        if fields.len() != 6 {
            bail!(
                "{}:{}: expected 6 fields, got {}",
                path.display(),
                count,
                fields.len()
            );
        }

        for (out_idx, &field_idx) in field_indices.iter().enumerate() {
            vec_of_vec[out_idx].push(fields[field_idx].to_string());
        }
    }

    Ok((vec_of_vec, count))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// PLINK BED/BIM/FAM file set reader.
pub struct PlinkBed {
    bed_path: PathBuf,
    iid_count: usize,
    sid_count: usize,
    /// Individual IDs from .fam (column 1)
    pub iid: Vec<String>,
    /// SNP IDs from .bim (column 1)
    pub sid: Vec<String>,
    /// Chromosome labels from .bim (column 0)
    pub chromosome: Vec<String>,
    /// Base-pair positions from .bim (column 3)
    pub bp_position: Vec<i32>,
    /// Allele 1 (A1) from .bim (column 4)
    pub allele1: Vec<String>,
    /// Allele 2 (A2) from .bim (column 5)
    pub allele2: Vec<String>,
}

impl PlinkBed {
    /// Open a PLINK file set from prefix (e.g. "data" → data.bed, data.bim, data.fam).
    pub fn new(prefix: &str) -> Result<Self> {
        let bed_path = PathBuf::from(format!("{}.bed", prefix));
        let fam_path = PathBuf::from(format!("{}.fam", prefix));
        let bim_path = PathBuf::from(format!("{}.bim", prefix));

        // Validate BED header
        open_and_check(&bed_path)?;

        // Parse .fam — fields: FID(0) IID(1) father(2) mother(3) sex(4) pheno(5)
        let (mut fam_cols, iid_count) = read_fam_or_bim(&[1], &fam_path)?;
        let iid = fam_cols.pop().unwrap();

        // Parse .bim — fields: chr(0) sid(1) cm(2) bp(3) allele1(4) allele2(5)
        let (mut bim_cols, sid_count) = read_fam_or_bim(&[0, 1, 3, 4, 5], &bim_path)?;
        let allele2 = bim_cols.pop().unwrap();
        let allele1 = bim_cols.pop().unwrap();
        let bp_strings = bim_cols.pop().unwrap();
        let sid = bim_cols.pop().unwrap();
        let chromosome = bim_cols.pop().unwrap();

        let bp_position: Vec<i32> = bp_strings
            .iter()
            .enumerate()
            .map(|(i, s)| {
                s.parse::<i32>().with_context(|| {
                    format!(
                        "{}:{}: invalid bp_position '{}'",
                        bim_path.display(),
                        i + 1,
                        s
                    )
                })
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            bed_path,
            iid_count,
            sid_count,
            iid,
            sid,
            chromosome,
            bp_position,
            allele1,
            allele2,
        })
    }

    pub fn iid_count(&self) -> usize {
        self.iid_count
    }

    pub fn sid_count(&self) -> usize {
        self.sid_count
    }

    /// Read genotype data as a nalgebra `DMatrix<f32>` (individuals × SNPs).
    ///
    /// Optionally subset individuals with `iid_range` (e.g. `Some(0..100)`).
    /// Missing genotypes are `f32::NAN`.
    pub fn read_f32(&self, iid_range: Option<std::ops::Range<usize>>) -> Result<DMatrix<f32>> {
        let (buf_reader, _header) = open_and_check(&self.bed_path)?;

        let iid_index: Vec<isize> = match iid_range {
            Some(range) => (range.start as isize..range.end as isize).collect(),
            None => (0..self.iid_count as isize).collect(),
        };
        let sid_index: Vec<isize> = (0..self.sid_count as isize).collect();

        let out_iid_count = iid_index.len();
        let out_sid_count = sid_index.len();

        // nalgebra DMatrix is column-major; we write directly into its data buffer
        let mut out_data = vec![0.0f32; out_iid_count * out_sid_count];

        internal_read_no_alloc(
            buf_reader,
            &self.bed_path,
            self.iid_count,
            self.sid_count,
            true, // count_a1
            &iid_index,
            &sid_index,
            f32::NAN,
            &mut out_data,
            out_iid_count,
        )?;

        Ok(DMatrix::from_vec(out_iid_count, out_sid_count, out_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Create a minimal PLINK file set and verify reading.
    ///
    /// 3 individuals, 2 SNPs (SNP-major, count_a1):
    ///   SNP0: [2, 1, 0]
    ///     packed: ind0=00(2), ind1=10(1), ind2=11(0), pad=00
    ///     → 0b00_11_10_00 = 0x38
    ///   SNP1: [0, NaN, 2]
    ///     packed: ind0=11(0), ind1=01(miss), ind2=00(2), pad=00
    ///     → 0b00_00_01_11 = 0x07
    #[test]
    fn test_read_genotypes() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("test");
        let prefix_str = prefix.to_str().unwrap();

        // .fam (3 individuals)
        let mut fam = File::create(format!("{}.fam", prefix_str)).unwrap();
        writeln!(fam, "fam1 ind1 0 0 1 -9").unwrap();
        writeln!(fam, "fam1 ind2 0 0 2 -9").unwrap();
        writeln!(fam, "fam1 ind3 0 0 1 -9").unwrap();

        // .bim (2 SNPs)
        let mut bim = File::create(format!("{}.bim", prefix_str)).unwrap();
        writeln!(bim, "1 snp1 0 100 A T").unwrap();
        writeln!(bim, "1 snp2 0 200 C G").unwrap();

        // .bed
        let mut bed = File::create(format!("{}.bed", prefix_str)).unwrap();
        bed.write_all(&[BED_FILE_MAGIC1, BED_FILE_MAGIC2, 0x01])
            .unwrap();
        bed.write_all(&[0x38]).unwrap(); // SNP0
        bed.write_all(&[0x07]).unwrap(); // SNP1

        let plink = PlinkBed::new(prefix_str).unwrap();
        assert_eq!(plink.iid_count(), 3);
        assert_eq!(plink.sid_count(), 2);
        assert_eq!(plink.iid, vec!["ind1", "ind2", "ind3"]);
        assert_eq!(plink.sid, vec!["snp1", "snp2"]);
        assert_eq!(plink.chromosome, vec!["1", "1"]);
        assert_eq!(plink.bp_position, vec![100, 200]);

        let val = plink.read_f32(None).unwrap();
        assert_eq!((val.nrows(), val.ncols()), (3, 2));

        // ind0: [2.0, 0.0]
        assert_eq!(val[(0, 0)], 2.0);
        assert_eq!(val[(0, 1)], 0.0);
        // ind1: [1.0, NaN]
        assert_eq!(val[(1, 0)], 1.0);
        assert!(val[(1, 1)].is_nan());
        // ind2: [0.0, 2.0]
        assert_eq!(val[(2, 0)], 0.0);
        assert_eq!(val[(2, 1)], 2.0);

        // Test subset reading
        let val_sub = plink.read_f32(Some(0..2)).unwrap();
        assert_eq!((val_sub.nrows(), val_sub.ncols()), (2, 2));
        assert_eq!(val_sub[(0, 0)], 2.0);
        assert_eq!(val_sub[(1, 0)], 1.0);
    }
}
