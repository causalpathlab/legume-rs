#![allow(dead_code)]

use crate::sam::Strand;

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
    let lines_of_words = read_lines_of_words_delim(file_path, &['\t'], -1)?.lines;

    Ok(lines_of_words
        .into_iter()
        .par_bridge()
        .filter_map(parse_gff)
        .collect::<Vec<_>>())
}

/// Take specific type of records and stretch out the start and stop
/// positions
///
/// * `records`: a vector of Gff records
/// * `feature_type`: optionally select a specific type of gene feature
///
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

/// Union gene model containing gene boundaries, CDS, and UTR maps
pub struct UnionGeneModel {
    pub gene_boundaries: HashMap<GeneId, GffRecord>,
    pub cds: HashMap<GeneId, GffRecord>,
    pub five_prime_utr: HashMap<GeneId, GffRecord>,
    pub three_prime_utr: HashMap<GeneId, GffRecord>,
}

/// Helper function: Insert or extend a record in a HashMap
fn insert_or_extend_record(map: &HashMap<GeneId, GffRecord>, gene_id: &GeneId, rec: &GffRecord) {
    if let Some(mut existing) = map.get_mut(gene_id) {
        let existing = existing.value_mut();
        existing.start = existing.start.min(rec.start);
        existing.stop = existing.stop.max(rec.stop);
    } else {
        map.insert(gene_id.clone(), rec.clone());
    }
}

/// Helper function: Calculate distance between two genomic regions
fn distance_between_regions(
    rec1_start: i64,
    rec1_stop: i64,
    rec2_start: i64,
    rec2_stop: i64,
) -> i64 {
    if rec1_stop < rec2_start {
        rec2_start - rec1_stop
    } else if rec1_start > rec2_stop {
        rec1_start - rec2_stop
    } else {
        0 // Overlapping
    }
}

/// Build CDS map from existing CDS annotations and start/stop codons
fn build_cds_map(
    records: &[GffRecord],
    start_codons: &HashMap<GeneId, GffRecord>,
    stop_codons: &HashMap<GeneId, GffRecord>,
) -> HashMap<GeneId, GffRecord> {
    let cds: HashMap<GeneId, GffRecord> = HashMap::new();

    // First, collect existing CDS annotations
    for rec in records.iter() {
        if rec.feature_type == FeatureType::CDS {
            insert_or_extend_record(&cds, &rec.gene_id, rec);
        }
    }

    // Then, extend using start and stop codons
    for entry in start_codons.iter() {
        let gene_id = entry.key();
        let start_rec = entry.value();

        if let Some(stop_rec) = stop_codons.get(gene_id) {
            let stop_rec = stop_rec.value();
            let mut cds_rec = start_rec.clone();
            cds_rec.feature_type = FeatureType::CDS;
            cds_rec.start = start_rec.start.min(stop_rec.start);
            cds_rec.stop = start_rec.stop.max(stop_rec.stop);
            cds.insert(gene_id.clone(), cds_rec);
        }
    }

    cds
}

/// Build UTR maps from explicit and generic UTR features
fn build_utr_maps(
    records: &[GffRecord],
    start_codons: &HashMap<GeneId, GffRecord>,
    stop_codons: &HashMap<GeneId, GffRecord>,
) -> (HashMap<GeneId, GffRecord>, HashMap<GeneId, GffRecord>) {
    let five_prime_utr: HashMap<GeneId, GffRecord> = HashMap::new();
    let three_prime_utr: HashMap<GeneId, GffRecord> = HashMap::new();

    for rec in records.iter() {
        let gene_id = &rec.gene_id;

        match rec.feature_type {
            FeatureType::FivePrimeUTR => {
                insert_or_extend_record(&five_prime_utr, gene_id, rec);
            }
            FeatureType::ThreePrimeUTR => {
                insert_or_extend_record(&three_prime_utr, gene_id, rec);
            }
            FeatureType::UTR => {
                // Classify generic UTR by distance to start/stop codons
                if let (Some(start_codon_rec), Some(stop_codon_rec)) = (
                    start_codons.get(gene_id).as_ref().map(|x| x.value()),
                    stop_codons.get(gene_id).as_ref().map(|x| x.value()),
                ) {
                    let dist_to_start = distance_between_regions(
                        rec.start,
                        rec.stop,
                        start_codon_rec.start,
                        start_codon_rec.stop,
                    );
                    let dist_to_stop = distance_between_regions(
                        rec.start,
                        rec.stop,
                        stop_codon_rec.start,
                        stop_codon_rec.stop,
                    );

                    if dist_to_start <= dist_to_stop {
                        insert_or_extend_record(&five_prime_utr, gene_id, rec);
                    } else {
                        insert_or_extend_record(&three_prime_utr, gene_id, rec);
                    }
                }
            }
            _ => {}
        }
    }

    (five_prime_utr, three_prime_utr)
}

/// Build a union gene model for each gene
/// 1. Find the longest region using Gff/Gtf's "gene" feature
/// 2. Create CDS as the union of start codon to stop codon regions
/// 3. Create 5'UTR and 3'UTR unions based on distance to codons
pub fn build_union_gene_model(records: &Vec<GffRecord>) -> anyhow::Result<UnionGeneModel> {
    // Build gene boundaries from "gene" features
    let gene_boundaries = build_gene_map(records, Some(&FeatureType::Gene))?;

    // Get canonical start and stop codons
    let start_codons = build_codon_map(records, &FeatureType::StartCodon)?;
    let stop_codons = build_codon_map(records, &FeatureType::StopCodon)?;

    // Build CDS map
    let cds = build_cds_map(records, &start_codons, &stop_codons);

    // Build UTR maps
    let (five_prime_utr, three_prime_utr) = build_utr_maps(records, &start_codons, &stop_codons);

    Ok(UnionGeneModel {
        gene_boundaries,
        cds,
        five_prime_utr,
        three_prime_utr,
    })
}

impl Default for GffRecordMap {
    fn default() -> Self {
        Self::new()
    }
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
            rec.stop += padding;
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
    FivePrimeUTR,
    ThreePrimeUTR,
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
            "five_prime_UTR" | "5UTR" | "five_prime_utr" => FeatureType::FivePrimeUTR,
            "three_prime_UTR" | "3UTR" | "three_prime_utr" => FeatureType::ThreePrimeUTR,
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
            FeatureType::FivePrimeUTR => Box::from("five_prime_UTR"),
            FeatureType::ThreePrimeUTR => Box::from("three_prime_UTR"),
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
