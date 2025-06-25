#![allow(dead_code)]

use crate::data::alignment::*;
use crate::data::dna::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux, Read};

pub const CELL_BARCODE_TAG: &[u8; 2] = b"CB";
pub const UMI_BARCODE_TAG: &[u8; 2] = b"UB";

use std::collections::{HashMap, HashSet};

pub struct DnaBaseFreqMap {
    position_to_statsitic_with_sample_forward: HashMap<i64, HashMap<SamSampleName, DnaBaseCount>>,
    sample_to_position_to_statistic_forward: HashMap<SamSampleName, HashMap<i64, DnaBaseCount>>,
    position_to_statsitic_with_sample_reverse: HashMap<i64, HashMap<SamSampleName, DnaBaseCount>>,
    sample_to_position_to_statistic_reverse: HashMap<SamSampleName, HashMap<i64, DnaBaseCount>>,
    samples: HashSet<SamSampleName>,
    positions: HashSet<i64>,
}

impl DnaBaseFreqMap {
    pub fn new() -> Self {
        Self {
            position_to_statsitic_with_sample_forward: HashMap::new(),
            sample_to_position_to_statistic_forward: HashMap::new(),
            position_to_statsitic_with_sample_reverse: HashMap::new(),
            sample_to_position_to_statistic_reverse: HashMap::new(),
            samples: HashSet::new(),
            positions: HashSet::new(),
        }
    }

    pub fn update_bam_region(
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

        for rec in bam_records {
            let sample_name = match rec.aux(CELL_BARCODE_TAG) {
                Ok(Aux::String(barcode)) => SamSampleName::Barcode(barcode.into()),
                _ => SamSampleName::Combined,
            };

            if !self.samples.contains(&sample_name) {
                self.samples.insert(sample_name.clone());
            }

            let is_reverse = rec.is_reverse();

            let freq_map_for_this_sample = if is_reverse {
                self.sample_to_position_to_statistic_reverse
                    .entry(sample_name.clone())
                    .or_default()
            } else {
                self.sample_to_position_to_statistic_forward
                    .entry(sample_name.clone())
                    .or_default()
            };

            let seq = rec.seq().as_bytes();

            //
            // Iter aligned read and reference positions on a basepair level
            // https://docs.rs/rust-htslib/latest/src/rust_htslib/bam/ext.rs.html#135
            // [read_pos, genome_pos]
            //
            for [rpos, gpos] in rec.aligned_pairs() {
                let (r, g, v) = (rpos as usize, gpos as usize, gpos - lb);

                if g < (lb as usize) || g >= (ub as usize) || v < 0 {
                    continue;
                }

                // let base = if is_reverse {
                //     Dna::from_byte_complement(seq[r])
                // } else {
                //     Dna::from_byte(seq[r])
                // };

                let base = Dna::from_byte(seq[r]);
		let genome_pos = gpos + 1; // 1-based position

                if !self.positions.contains(&genome_pos) {
                    self.positions.insert(genome_pos);
                }

                let freq = freq_map_for_this_sample
                    .entry(genome_pos)
                    .or_insert_with(DnaBaseCount::new);

                freq.add(base.clone(), 1);

                let freq_map = if is_reverse {
                    self.position_to_statsitic_with_sample_reverse
                        .entry(genome_pos)
                        .or_default()
                } else {
                    self.position_to_statsitic_with_sample_forward
                        .entry(genome_pos)
                        .or_default()
                };

                let freq = freq_map
                    .entry(sample_name.clone())
                    .or_insert_with(DnaBaseCount::new);
                freq.add(base.clone(), 1);
            }
        }

        Ok(())
    }

    pub fn samples(&self) -> Vec<&SamSampleName> {
        self.samples.iter().collect()
    }

    pub fn forward_frequency_at(&self, pos: &i64) -> Option<&HashMap<SamSampleName, DnaBaseCount>> {
        self.position_to_statsitic_with_sample_forward.get(pos)
    }

    pub fn reverse_frequency_at(&self, pos: &i64) -> Option<&HashMap<SamSampleName, DnaBaseCount>> {
        self.position_to_statsitic_with_sample_reverse.get(pos)
    }

    pub fn forward_frequency_per_sample(
        &self,
        k: &SamSampleName,
    ) -> Option<&HashMap<i64, DnaBaseCount>> {
        self.sample_to_position_to_statistic_forward.get(k)
    }

    pub fn reverse_frequency_per_sample(
        &self,
        k: &SamSampleName,
    ) -> Option<&HashMap<i64, DnaBaseCount>> {
        self.sample_to_position_to_statistic_reverse.get(k)
    }

    /// Statistics combined by position
    pub fn forward_marginal_frequency_by_position(&self) -> HashMap<i64, DnaBaseCount> {
        let mut ret: HashMap<i64, DnaBaseCount> =
            HashMap::with_capacity(self.position_to_statsitic_with_sample_forward.len());

        for (pos, freq_map) in self.position_to_statsitic_with_sample_forward.iter() {
            let accum = ret.entry(*pos).or_insert_with(DnaBaseCount::new);
            for (_, freq) in freq_map.iter() {
                *accum += freq;
            }
        }
        ret
    }

    pub fn reverse_marginal_frequency_by_position(&self) -> HashMap<i64, DnaBaseCount> {
        let mut ret: HashMap<i64, DnaBaseCount> =
            HashMap::with_capacity(self.position_to_statsitic_with_sample_reverse.len());

        for (pos, freq_map) in self.position_to_statsitic_with_sample_reverse.iter() {
            let accum = ret.entry(*pos).or_insert_with(DnaBaseCount::new);
            for (_, freq) in freq_map.iter() {
                *accum += freq;
            }
        }
        ret
    }

    /// sorted genomic positions
    pub fn sorted_positions(&self) -> Vec<i64> {
        let mut ret = self.positions.iter().copied().collect::<Vec<_>>();
        ret.sort();
        ret.dedup();
        ret
    }
}
