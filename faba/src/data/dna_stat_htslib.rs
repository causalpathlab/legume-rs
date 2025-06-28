#![allow(dead_code)]

use crate::data::dna::*;
use crate::data::gff::*;
use crate::data::positions::*;
use crate::data::sam::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux, Read};
use std::collections::{HashMap, HashSet};

pub struct DnaBaseFreqMap<'a> {
    position_to_statsitic_with_sample: HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>,
    sample_to_position_to_statistic: HashMap<CellBarcode, HashMap<i64, DnaBaseCount>>,
    cells: HashSet<CellBarcode>,
    positions: HashSet<i64>,
    cell_barcode_tag: &'a [u8],
    gene_barcode_tag: &'a [u8],
}

impl<'a> DnaBaseFreqMap<'a> {
    pub fn new(cell_barcode_tag: &'a str, gene_barcode_tag: &'a str) -> Self {
        Self {
            position_to_statsitic_with_sample: HashMap::new(),
            sample_to_position_to_statistic: HashMap::new(),
            cells: HashSet::new(),
            positions: HashSet::new(),
            cell_barcode_tag: cell_barcode_tag.as_bytes(),
            gene_barcode_tag: gene_barcode_tag.as_bytes(),
        }
    }

    pub fn update_by_gene_in_bam(
        &mut self,
        bam_file_path: &str,
        gene_id: &Gene,
        region: (&str, i64, i64),
    ) -> anyhow::Result<()> {
        let index_file = bam_file_path.to_string() + ".bai";
        let mut bam_reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;

        bam_reader.fetch(region)?;

        let (_, lb, ub) = region;

        let bam_records: Vec<bam::Record> = bam_reader
            .records()
            .into_iter()
            .filter_map(Result::ok)
            .filter(|rec| !rec.is_duplicate())
            .collect();

        for rec in bam_records {
            let cell_barcode = match rec.aux(&self.cell_barcode_tag) {
                Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
                _ => CellBarcode::Missing,
            };

            let gene_id_found = match rec.aux(&self.gene_barcode_tag) {
                Ok(Aux::String(id)) => {
                    let id =
                        parse_ensembl_id(id).ok_or(anyhow::anyhow!("failed to parse {}", id))?;
                    Gene::Ensembl(id.into())
                }
                _ => Gene::Missing,
            };

            if gene_id_found == *gene_id {
                if !self.cells.contains(&cell_barcode) {
                    self.cells.insert(cell_barcode.clone());
                }

                let freq_map_for_this_sample = self
                    .sample_to_position_to_statistic
                    .entry(cell_barcode.clone())
                    .or_default();

                let seq = rec.seq().as_bytes();

                // Note: these are zero-based positions
                for [rpos, gpos] in rec.aligned_pairs() {
                    let (r, g, v) = (rpos as usize, gpos as usize, gpos - lb);

                    if g < (lb as usize) || g >= (ub as usize) || v < 0 {
                        continue;
                    }

                    let base = Dna::from_byte(seq[r]);
                    let genome_pos = gpos + 1; // 1-based position

                    if !self.positions.contains(&genome_pos) {
                        self.positions.insert(genome_pos.clone());
                    }

                    let freq = freq_map_for_this_sample
                        .entry(genome_pos)
                        .or_insert_with(DnaBaseCount::new);

                    freq.add(base.as_ref(), 1);

                    let freq_map = self
                        .position_to_statsitic_with_sample
                        .entry(genome_pos)
                        .or_default();

                    let freq = freq_map
                        .entry(cell_barcode.clone())
                        .or_insert_with(DnaBaseCount::new);
                    freq.add(base.as_ref(), 1);
                }
            }
        }

        Ok(())
    }

    /// update DNA count statistics using the information within this region
    /// * `bam_file_path` - file path
    /// * `region` - (chromosome, start, stop), [start, stop), zero-based
    pub fn update_by_region_in_bam(
        &mut self,
        bam_file_path: &str,
        region: (&str, i64, i64),
    ) -> anyhow::Result<()> {
        let index_file = bam_file_path.to_string() + ".bai";
        let mut bam_reader = bam::IndexedReader::from_path_and_index(bam_file_path, &index_file)?;

        bam_reader.fetch(region)?;

        let (_, lb, ub) = region;

        let bam_records: Vec<bam::Record> = bam_reader
            .records()
            .into_iter()
            .filter_map(Result::ok)
            .filter(|rec| !rec.is_duplicate())
            .collect();

        // From `https://samtools.github.io/hts-specs/SAMv1.pdf`
        //
        // All mapped segments in alignment lines are represented on the
        // forward genomic strand. For segments that have been mapped to the
        // reverse strand, the recorded SEQ is reverse complemented from the
        // original unmapped sequence and CIGAR, QUAL, and strand-sensitive
        // optional fields are reversed and thus recorded consistently with
        // the sequence bases as represented.
        for rec in bam_records {
            let cell_barcode = match rec.aux(&self.cell_barcode_tag) {
                Ok(Aux::String(barcode)) => CellBarcode::Barcode(barcode.into()),
                _ => CellBarcode::Missing,
            };

            if !self.cells.contains(&cell_barcode) {
                self.cells.insert(cell_barcode.clone());
            }

            // No need to worry about read-level strand information
            // let is_reverse = rec.is_reverse();

            let freq_map_for_this_sample = self
                .sample_to_position_to_statistic
                .entry(cell_barcode.clone())
                .or_default();

            let seq = rec.seq().as_bytes();

            //
            // Iter aligned read and reference positions on a basepair level
            // https://docs.rs/rust-htslib/latest/src/rust_htslib/bam/ext.rs.html#135
            // [read_pos, genome_pos]
            //
            // Note: these are zero-based positions
            for [rpos, gpos] in rec.aligned_pairs() {
                let (r, g, v) = (rpos as usize, gpos as usize, gpos - lb);

                if g < (lb as usize) || g >= (ub as usize) || v < 0 {
                    continue;
                }

                let base = Dna::from_byte(seq[r]);
                let genome_pos = gpos + 1; // 1-based position

                if !self.positions.contains(&genome_pos) {
                    self.positions.insert(genome_pos);
                }

                let freq = freq_map_for_this_sample
                    .entry(genome_pos)
                    .or_insert_with(DnaBaseCount::new);

                freq.add(base.as_ref(), 1);

                let freq_map = self
                    .position_to_statsitic_with_sample
                    .entry(genome_pos)
                    .or_default();

                let freq = freq_map
                    .entry(cell_barcode.clone())
                    .or_insert_with(DnaBaseCount::new);
                freq.add(base.as_ref(), 1);
            }
        }

        Ok(())
    }

    pub fn cells(&self) -> Vec<&CellBarcode> {
        self.cells.iter().collect()
    }

    pub fn frequency_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, DnaBaseCount>> {
        self.position_to_statsitic_with_sample.get(&pos)
    }

    pub fn frequency_per_sample(&self, k: &CellBarcode) -> Option<&HashMap<i64, DnaBaseCount>> {
        self.sample_to_position_to_statistic.get(k)
    }

    /// Statistics combined by position
    pub fn marginal_frequency_by_position(&self) -> HashMap<i64, DnaBaseCount> {
        let mut ret: HashMap<i64, DnaBaseCount> =
            HashMap::with_capacity(self.position_to_statsitic_with_sample.len());

        for (&pos, freq_map) in self.position_to_statsitic_with_sample.iter() {
            let accum = ret.entry(pos).or_insert_with(DnaBaseCount::new);
            for (_, freq) in freq_map.iter() {
                *accum += freq;
            }
        }
        ret
    }

    pub fn sorted_positions(&self) -> Vec<i64> {
        let mut ret = self.positions.iter().copied().collect::<Vec<_>>();
        ret.sort();
        ret.dedup();
        ret
    }

    // pub fn positions_per_gene(&self) -> HashMap<Gene, Vec<SiteInGene>> {
    //     self.positions
    //         .iter()
    //         .map(|x| x)
    //         .collect::<Vec<_>>()
    //         .stratify_by_gene()
    // }
}
