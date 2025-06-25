#![allow(dead_code)]

use std::str;

use crate::data::alignment::*;
use crate::data::dna::*;

use noodles::{bam, sam};
use noodles_core::region::Region;
use sam::alignment::record::data::field::tag::Tag as SamTag;

use std::collections::{HashMap, HashSet};

pub const CELL_BARCODE_TAG: SamTag = SamTag::new(b'C', b'B');
pub const UMI_BARCODE_TAG: SamTag = SamTag::new(b'U', b'B');

pub struct DnaBaseFreqMap {
    position_to_statsitic_with_sample_forward: HashMap<usize, HashMap<SamSampleName, DnaBaseCount>>,
    sample_to_position_to_statistic_forward: HashMap<SamSampleName, HashMap<usize, DnaBaseCount>>,
    position_to_statsitic_with_sample_reverse: HashMap<usize, HashMap<SamSampleName, DnaBaseCount>>,
    sample_to_position_to_statistic_reverse: HashMap<SamSampleName, HashMap<usize, DnaBaseCount>>,
    samples: HashSet<SamSampleName>,
    positions: HashSet<usize>,
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

    pub fn from_bam_region(bam_file_path: &str, raw_region: &str) -> anyhow::Result<Self> {
        let mut dna_freq_map = Self::new();
        dna_freq_map.update_bam_region(bam_file_path, raw_region)?;
        Ok(dna_freq_map)
    }

    pub fn update_bam_region(
        &mut self,
        bam_file_path: &str,
        raw_region: &str,
    ) -> anyhow::Result<()> {
        let region: Region = raw_region.parse()?;
        let _name = region.name().to_string();
        let mut reader =
            bam::io::indexed_reader::Builder::default().build_from_path(bam_file_path)?;

        let header = reader.read_header()?;
        let query = reader.query(&header, &region)?;

        for result in query {
            self.add_10x_record(result?)?;
        }

        Ok(())
    }

    pub fn forward_frequency_per_sample(
        &self,
        k: &SamSampleName,
    ) -> Option<&HashMap<usize, DnaBaseCount>> {
        self.sample_to_position_to_statistic_forward.get(k)
    }

    pub fn reverse_frequency_per_sample(
        &self,
        k: &SamSampleName,
    ) -> Option<&HashMap<usize, DnaBaseCount>> {
        self.sample_to_position_to_statistic_reverse.get(k)
    }

    /// Statistics combined by position
    pub fn forward_marginal_frequency_by_position(&self) -> HashMap<usize, DnaBaseCount> {
        let mut ret: HashMap<usize, DnaBaseCount> =
            HashMap::with_capacity(self.position_to_statsitic_with_sample_forward.len());

        for (pos, freq_map) in self.position_to_statsitic_with_sample_forward.iter() {
            let accum = ret.entry(*pos).or_insert_with(DnaBaseCount::new);
            for (_, freq) in freq_map.iter() {
                *accum += freq;
            }
        }
        ret
    }

    pub fn reverse_marginal_frequency_by_position(&self) -> HashMap<usize, DnaBaseCount> {
        let mut ret: HashMap<usize, DnaBaseCount> =
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
    pub fn sorted_positions(&self) -> Vec<usize> {
        let mut ret = self.positions.iter().copied().collect::<Vec<_>>();
        ret.sort();
        ret.dedup();
        ret
    }

    pub fn samples(&self) -> Vec<&SamSampleName> {
        self.samples.iter().collect()
    }

    pub fn forward_frequency_at(
        &self,
        pos: &usize,
    ) -> Option<&HashMap<SamSampleName, DnaBaseCount>> {
        self.position_to_statsitic_with_sample_forward.get(pos)
    }

    pub fn reverse_frequency_at(
        &self,
        pos: &usize,
    ) -> Option<&HashMap<SamSampleName, DnaBaseCount>> {
        self.position_to_statsitic_with_sample_reverse.get(pos)
    }

    // pub fn umis(&self) -> Vec<&SamUmiName> {
    //     self.umis.iter().collect()
    // }

    /// accumulate statistics from a short read sequence
    ///
    pub fn add_10x_record<Rec>(&mut self, record: Rec) -> anyhow::Result<()>
    where
        Rec: sam::alignment::Record,
    {
        let aux_data = record.data();

        let sample_name = match aux_data.get(&CELL_BARCODE_TAG) {
            Some(Ok(v)) => sam_sample_name(v)?,
            _ => SamSampleName::Combined,
        };

        if !self.samples.contains(&sample_name) {
            self.samples.insert(sample_name.clone());
        }

        // let umi_name = match aux_data.get(&UB) {
        //     Some(Ok(v)) => sam_umi_name(v)?,
        //     _ => SamUmiName::Combined,
        // };

        // if !self.umis.contains(&umi_name) {
        //     self.umis.insert(umi_name.clone());
        // }

        let flags = record.flags()?;

        if flags.is_duplicate() || flags.is_unmapped() || flags.is_qc_fail() {
            return Ok(());
        }

        let is_reverse = flags.is_reverse_complemented();

        let freq_map_for_this_sample = if is_reverse {
            self.sample_to_position_to_statistic_reverse
                .entry(sample_name.clone())
                .or_default()
        } else {
            self.sample_to_position_to_statistic_forward
                .entry(sample_name.clone())
                .or_default()
        };

        if let Some(Ok(pos)) = record.alignment_start() {
            let pos = pos.get();

            let sequence = record.sequence();

            for (i, b) in sequence.iter().enumerate() {
                let genome_pos = i + pos;

                let base = if is_reverse {
                    Dna::from_byte_complement(b)
                } else {
                    Dna::from_byte(b)
                };

                if !self.positions.contains(&genome_pos) {
                    self.positions.insert(genome_pos);
                }

                let freq = freq_map_for_this_sample
                    .entry(genome_pos)
                    .or_insert_with(DnaBaseCount::new);

                freq.add(base.clone(), 1);

                // let freq = self
                //     .position_to_statistic
                //     .entry(genome_pos)
                //     .or_insert_with(DnaBaseCount::new);
                // freq.add(base, 1);

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
}
