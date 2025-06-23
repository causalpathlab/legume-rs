#![allow(dead_code)]

use std::str;

use crate::data::alignment::*;
use crate::data::dna::*;

use noodles::{bam, sam};
use noodles_core::region::Region;
use sam::alignment::record::data::field::tag::Tag as SamTag;

use std::collections::{HashMap, HashSet};

pub const CB: SamTag = SamTag::new(b'C', b'B');
// pub const UB: SamTag = SamTag::new(b'U', b'B');

pub struct DnaBaseStatMap {
    position_to_statsitic_with_sample: HashMap<usize, HashMap<SamSampleName, DnaBaseCount>>,
    samples: HashSet<SamSampleName>,
    positions: HashSet<usize>,
}

impl DnaBaseStatMap {
    pub fn new() -> Self {
        Self {
            position_to_statsitic_with_sample: HashMap::new(),
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

    /// Statistics combined by position
    pub fn combine_statistic_by_position(&self) -> HashMap<usize, DnaBaseCount> {
        let mut ret: HashMap<usize, DnaBaseCount> =
            HashMap::with_capacity(self.position_to_statsitic_with_sample.len());

        for (pos, freq_map) in self.position_to_statsitic_with_sample.iter() {
            let accum = ret.entry(*pos).or_insert_with(DnaBaseCount::new);
            for (_, freq) in freq_map.iter() {
                *accum += freq;
            }
        }
        ret
    }

    /// sorted genomic positions
    pub fn sorted_positions(&self) -> Vec<usize> {
        let mut ret = self.positions.iter().map(|&x| x).collect::<Vec<_>>();
        ret.sort();
        ret
    }

    pub fn samples(&self) -> Vec<&SamSampleName> {
        self.samples.iter().collect()
    }

    pub fn statistic_at(
        &self,
        pos: &usize,
    ) -> anyhow::Result<&HashMap<SamSampleName, DnaBaseCount>> {
        self.position_to_statsitic_with_sample
            .get(pos)
            .ok_or(anyhow::anyhow!("missing"))
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

        let sample_name = match aux_data.get(&CB) {
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

        if let Some(Ok(pos)) = record.alignment_start() {
            let pos = pos.get();
            let sequence = record.sequence();
            for (i, b) in sequence.iter().enumerate() {
                let genome_pos = i + pos;

                if !self.positions.contains(&genome_pos) {
                    self.positions.insert(genome_pos);
                }

                // let freq = self
                //     .position_to_statistic
                //     .entry(genome_pos)
                //     .or_insert_with(DnaBaseCount::new);
                // freq.add(Dna::from_byte(b), 1);

                let freq_map = self
                    .position_to_statsitic_with_sample
                    .entry(genome_pos)
                    .or_insert_with(HashMap::new);

                let freq = freq_map
                    .entry(sample_name.clone())
                    .or_insert_with(DnaBaseCount::new);

                freq.add(Dna::from_byte(b), 1);
            }
        }

        Ok(())
    }
}
