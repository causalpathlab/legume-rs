#![allow(dead_code)]

use crate::data::sam::Strand;

use dashmap::DashMap as HashMap;
use matrix_util::common_io::read_lines_of_words_delim;
use rayon::prelude::*;

/// Ignore ENSEMBL version: `ENSGXXXXXXXXXXX.X_X` to `ENSGXXXXXXXXXXX`
pub fn parse_ensembl_id(ensembl_name: &str) -> Option<&str> {
    ensembl_name.split('.').next()
}

/// Gff record mapping a gene to one record
pub struct GffRecordMap {
    records: HashMap<GeneId, GffRecord>,
}

/// read gff records including multiple annotations per gene
pub fn read_gff_record_vec(file_path: &str) -> anyhow::Result<Vec<GffRecord>> {
    let lines_of_words = read_lines_of_words_delim(file_path, &['\t', ','], -1)?.lines;

    Ok(lines_of_words
        .into_iter()
        .par_bridge()
        .filter_map(|x| parse_gff(x))
        .collect::<Vec<_>>())
}

/// take specific type of records and stretch out the start and stop
/// positions
pub fn build_gene_map(
    records: &Vec<GffRecord>,
    feature_type: Option<&FeatureType>,
) -> anyhow::Result<HashMap<GeneId, GffRecord>> {
    let ret: HashMap<GeneId, GffRecord> = HashMap::new();

    for new_rec in records.iter() {
        let gene_id = &new_rec.gene_id;
        if feature_type
            .map(|x| x == &new_rec.feature_type)
            .unwrap_or(true)
        {
            if ret.contains_key(gene_id) {
                if let Some(mut rec) = ret.get_mut(gene_id) {
                    let rec = rec.value_mut();
                    rec.start = rec.start.min(new_rec.start);
                    rec.stop = rec.stop.max(new_rec.stop);
                }
            } else {
                ret.insert(gene_id.clone(), new_rec.clone());
            }
        }
    }

    Ok(ret)
}

/// Build a map of start or stop codons, selecting the appropriate one per gene based on strand
/// For start codons: picks the 5'-most (forward: lowest coords, backward: highest coords)
/// For stop codons: picks the 3'-most (forward: highest coords, backward: lowest coords)
pub fn build_codon_map(
    records: &Vec<GffRecord>,
    feature_type: &FeatureType,
) -> anyhow::Result<HashMap<GeneId, GffRecord>> {
    let ret: HashMap<GeneId, GffRecord> = HashMap::new();

    let is_start_codon = *feature_type == FeatureType::StartCodon;

    for new_rec in records.iter() {
        if &new_rec.feature_type != feature_type {
            continue;
        }

        let gene_id = &new_rec.gene_id;

        if let Some(mut rec) = ret.get_mut(gene_id) {
            let rec = rec.value_mut();

            // Choose the appropriate codon based on strand and codon type
            let should_replace = match (&new_rec.strand, is_start_codon) {
                // Forward strand, start codon: pick 5'-most (lowest coordinates)
                (Strand::Forward, true) => new_rec.start < rec.start,
                // Forward strand, stop codon: pick 3'-most (highest coordinates)
                (Strand::Forward, false) => new_rec.stop > rec.stop,
                // Backward strand, start codon: pick 5'-most (highest coordinates)
                (Strand::Backward, true) => new_rec.stop > rec.stop,
                // Backward strand, stop codon: pick 3'-most (lowest coordinates)
                (Strand::Backward, false) => new_rec.start < rec.start,
            };

            if should_replace {
                *rec = new_rec.clone();
            }
        } else {
            ret.insert(gene_id.clone(), new_rec.clone());
        }
    }

    Ok(ret)
}

pub trait QuickStat {
    fn take_average_length(&self) -> i64;
    fn take_max_length(&self) -> i64;
}

impl QuickStat for HashMap<GeneId, GffRecord> {
    fn take_average_length(&self) -> i64 {
        let len_vec = self
            .par_iter()
            .map(|x| x.value().stop - x.value().start)
            .collect::<Vec<_>>();
        let denom = len_vec.len().max(1) as i64;
        let num = len_vec.into_iter().sum::<i64>();
        num / denom
    }
    fn take_max_length(&self) -> i64 {
        let len_vec = self
            .par_iter()
            .map(|x| x.value().stop - x.value().start)
            .collect::<Vec<_>>();
        len_vec.into_iter().max().unwrap_or(1)
    }
}

pub struct UTRMap {
    pub five_prime: HashMap<GeneId, GffRecord>,
    pub three_prime: HashMap<GeneId, GffRecord>,
}

pub fn build_utr_map(records: &Vec<GffRecord>) -> anyhow::Result<UTRMap> {
    // Use build_codon_map to select the canonical start/stop codon per gene based on strand
    let start_codons = build_codon_map(records, &FeatureType::StartCodon)?;
    let stop_codons = build_codon_map(records, &FeatureType::StopCodon)?;

    let five_prime: HashMap<GeneId, GffRecord> = HashMap::new();
    let three_prime: HashMap<GeneId, GffRecord> = HashMap::new();

    for new_rec in records.iter().cloned() {
        if new_rec.feature_type == FeatureType::UTR {
            if let (Some(start), Some(stop)) = (
                start_codons
                    .get(&new_rec.gene_id)
                    .as_ref()
                    .map(|x| x.value()),
                stop_codons
                    .get(&new_rec.gene_id)
                    .as_ref()
                    .map(|x| x.value()),
            ) {
                // Calculate minimum distance between UTR and each codon
                // by finding the gap between the intervals (or overlap if negative)
                let (d_to_start, d_to_stop) = match new_rec.strand {
                    Strand::Forward => {
                        // Forward: start codon is upstream (lower coords), stop is downstream (higher coords)
                        let gap_to_start = if new_rec.stop < start.start {
                            start.start - new_rec.stop // UTR before start codon
                        } else if new_rec.start > start.stop {
                            new_rec.start - start.stop // UTR after start codon
                        } else {
                            0 // Overlapping
                        };
                        let gap_to_stop = if new_rec.stop < stop.start {
                            stop.start - new_rec.stop // UTR before stop codon
                        } else if new_rec.start > stop.stop {
                            new_rec.start - stop.stop // UTR after stop codon
                        } else {
                            0 // Overlapping
                        };
                        (gap_to_start, gap_to_stop)
                    }
                    Strand::Backward => {
                        // Backward: start codon is downstream (higher coords), stop is upstream (lower coords)
                        let gap_to_start = if new_rec.start > start.stop {
                            new_rec.start - start.stop // UTR after start codon (5' direction)
                        } else if new_rec.stop < start.start {
                            start.start - new_rec.stop // UTR before start codon
                        } else {
                            0 // Overlapping
                        };
                        let gap_to_stop = if new_rec.stop < stop.start {
                            stop.start - new_rec.stop // UTR before stop codon (3' direction)
                        } else if new_rec.start > stop.stop {
                            new_rec.start - stop.stop // UTR after stop codon
                        } else {
                            0 // Overlapping
                        };
                        (gap_to_start, gap_to_stop)
                    }
                };

                let gene_id = &new_rec.gene_id;
                if d_to_start < d_to_stop {
                    if five_prime.contains_key(gene_id) {
                        if let Some(mut rec) = five_prime.get_mut(gene_id) {
                            let rec = rec.value_mut();
                            rec.start = rec.start.min(new_rec.start);
                            rec.stop = rec.stop.max(new_rec.stop);
                        }
                    } else {
                        five_prime.insert(gene_id.clone(), new_rec.clone());
                    }
                } else {
                    if three_prime.contains_key(gene_id) {
                        if let Some(mut rec) = three_prime.get_mut(gene_id) {
                            let rec = rec.value_mut();
                            rec.start = rec.start.min(new_rec.start);
                            rec.stop = rec.stop.max(new_rec.stop);
                        }
                    } else {
                        three_prime.insert(gene_id.clone(), new_rec.clone());
                    }
                }
            }
        }
    }

    Ok(UTRMap {
        five_prime,
        three_prime,
    })
}

impl GffRecordMap {
    /// Get GFF record mapping: one gene to one record
    pub fn from(file_path: &str) -> anyhow::Result<Self> {
        let parsed_records = read_gff_record_vec(file_path)?;

        let records = build_gene_map(&parsed_records, Some(&FeatureType::Gene))?;

        Ok(Self { records })
    }

    pub fn count_gene_types(&self) -> HashMap<GeneType, usize> {
        let gene_type_counts = HashMap::new();

        self.records.par_iter().for_each(|entry| {
            let record = entry.value();
            *gene_type_counts
                .entry(record.gene_type.clone())
                .or_insert(0) += 1;
        });

        gene_type_counts
    }

    pub fn count_feature_types(&self) -> HashMap<FeatureType, usize> {
        let feature_type_counts = HashMap::new();

        self.records.par_iter().for_each(|entry| {
            let record = entry.value();
            *feature_type_counts
                .entry(record.feature_type.clone())
                .or_insert(0) += 1;
        });

        feature_type_counts
    }

    pub fn subset(&mut self, target_gene_type: GeneType) {
        self.records
            .retain(|_, rec| rec.gene_type == target_gene_type);
    }

    pub fn add_padding(&mut self, padding: i64) {
        self.records.par_iter_mut().for_each(|mut entry| {
            let rec = entry.value_mut();
            rec.start = (rec.start - padding).max(0);
            rec.stop = rec.stop + padding;
        });
    }

    /// Create a new empty `GffRecordMap`
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Get a `GffRecord` by `GeneId`
    pub fn get(&self, gene_id: &GeneId) -> Option<GffRecord> {
        self.records.get(gene_id).map(|x| x.value().clone())
    }

    /// Get all `GffRecord`s in the map
    pub fn records(&self) -> Vec<GffRecord> {
        self.records
            .iter()
            .map(|entry| entry.value().clone())
            .collect::<Vec<_>>()
    }

    /// Count the number of records in the map
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum FeatureType {
    Gene,
    Transcript,
    Exon,
    CDS,
    UTR,
    StartCodon,
    StopCodon,
    Other,
}

impl From<&str> for FeatureType {
    fn from(val: &str) -> Self {
        match val {
            "gene" | "Gene" => FeatureType::Gene,
            "transcript" | "mRNA" => FeatureType::Transcript,
            "exon" => FeatureType::Exon,
            "CDS" | "cds" => FeatureType::CDS,
            "UTR" | "utr" => FeatureType::UTR,
            "start_codon" | "start" => FeatureType::StartCodon,
            "stop_codon" | "stop" => FeatureType::StopCodon,
            _ => FeatureType::Other,
        }
    }
}

impl From<FeatureType> for Box<str> {
    fn from(feature_type: FeatureType) -> Self {
        match feature_type {
            FeatureType::Gene => Box::from("gene"),
            FeatureType::Transcript => Box::from("transcript"),
            FeatureType::Exon => Box::from("exon"),
            FeatureType::CDS => Box::from("CDS"),
            FeatureType::UTR => Box::from("UTR"),
            FeatureType::StartCodon => Box::from("start_codon"),
            FeatureType::StopCodon => Box::from("stop_codon"),
            FeatureType::Other => Box::from("."),
        }
    }
}

impl std::fmt::Display for FeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: Box<str> = self.clone().into();
        write!(f, "{}", x)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum GeneId {
    Ensembl(Box<str>),
    Missing,
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum GeneSymbol {
    Symbol(Box<str>),
    Missing,
}

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone)]
pub enum GeneType {
    CodingGene,
    PseudoGene,
    LincRNA,
    SnRNA,
    NonCoding,
}

impl From<Box<str>> for GeneType {
    fn from(val: Box<str>) -> Self {
        val.as_ref().into()
    }
}

impl From<&str> for GeneType {
    fn from(val: &str) -> Self {
        match val {
            "protein_coding" => GeneType::CodingGene,
            "pseudogene" => GeneType::PseudoGene,
            "lncRNA" => GeneType::LincRNA,
            "snRNA" => GeneType::SnRNA,
            _ if val.ends_with("pseudogene") => GeneType::PseudoGene,
            _ if val.ends_with("coding") => GeneType::CodingGene,
            _ => GeneType::NonCoding,
        }
    }
}

impl std::fmt::Display for GeneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: Box<str> = self.clone().into();
        write!(f, "{}", x)
    }
}

impl std::fmt::Display for GeneSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x: Box<str> = self.clone().into();
        write!(f, "{}", x)
    }
}

impl From<GeneId> for Box<str> {
    fn from(gene_id: GeneId) -> Self {
        match gene_id {
            GeneId::Ensembl(id) => id,
            GeneId::Missing => Box::from("."),
        }
    }
}

impl From<GeneSymbol> for Box<str> {
    fn from(gene_id: GeneSymbol) -> Self {
        match gene_id {
            GeneSymbol::Symbol(id) => id.clone(),
            GeneSymbol::Missing => Box::from("."),
        }
    }
}

/// GFF record
#[derive(Clone, Debug)]
pub struct GffRecord {
    pub seqname: Box<str>,         // sequence name
    pub feature_type: FeatureType, // may need gene only
    pub start: i64,                // 1-based
    pub stop: i64,                 // 1-based
    pub strand: Strand,
    pub gene_id: GeneId,
    pub gene_name: GeneSymbol,
    pub gene_type: GeneType,
}

/// parse a GFF line to a record
/// [gencode](https://www.gencodegenes.org/pages/data_format.html)
///
pub fn parse_gff(words: Vec<Box<str>>) -> Option<GffRecord> {
    const SPLIT_SEP: char = ';';
    const NUM_FIELDS: usize = 9;

    if words.len() != NUM_FIELDS {
        return None;
    }

    let seqname = words[0].clone();
    let feature_type: FeatureType = words[2].as_ref().into();
    let start = words[3].parse::<i64>().unwrap_or(0);
    let stop = words[4].parse::<i64>().unwrap_or(0);
    let strand = match words[6].as_ref() {
        "+" => Strand::Forward, // "+"
        _ => Strand::Backward,  // "-" other than "+"
    };

    let mut gene_id = GeneId::Missing;
    let mut gene_name = GeneSymbol::Missing;
    let mut gene_type = GeneType::NonCoding;

    fn trim(x: Option<&str>) -> Option<&str> {
        x.map(|s| {
            let s = s.trim();
            s.strip_prefix('"')
                .unwrap_or(s)
                .strip_suffix('"')
                .unwrap_or(s)
        })
    }

    for attr in words[8].split(SPLIT_SEP).map(|s| s.trim()) {
        let mut kv = attr.split(&[' ', ':', '=']);

        match (trim(kv.next()), trim(kv.next())) {
            (Some("gene_id"), Some(id)) => {
                if let Some(ensembl) = parse_ensembl_id(id) {
                    gene_id = GeneId::Ensembl(ensembl.into());
                }
            }
            (Some("gene_name"), Some(name)) => {
                gene_name = GeneSymbol::Symbol(name.into());
            }
            (Some("gene_type"), Some(gtype)) => {
                gene_type = gtype.into();
            }
            _ => {}
        }
    }

    Some(GffRecord {
        seqname,
        feature_type,
        start,
        stop,
        strand,
        gene_id,
        gene_name,
        gene_type,
    })
}
