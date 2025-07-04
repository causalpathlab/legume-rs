#![allow(dead_code)]

use crate::data::sam::Strand;

use matrix_util::common_io::read_lines_of_words_delim;
use rayon::prelude::*;
use std::collections::hash_map::Iter;
use std::collections::HashMap;

/// Ignore ENSEMBL version: `ENSGXXXXXXXXXXX.X_X` to `ENSGXXXXXXXXXXX`
pub fn parse_ensembl_id(ensembl_name: &str) -> Option<&str> {
    ensembl_name.split('.').next()
}

pub struct GffRecordMap {
    records: HashMap<GeneId, GffRecord>,
}

pub struct GffRecordMapIter<'a> {
    inner: Iter<'a, GeneId, GffRecord>,
}

impl<'a> IntoIterator for &'a GffRecordMap {
    type Item = (&'a GeneId, &'a GffRecord);
    type IntoIter = GffRecordMapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        GffRecordMapIter {
            inner: self.records.iter(),
        }
    }
}

impl<'a> Iterator for GffRecordMapIter<'a> {
    type Item = (&'a GeneId, &'a GffRecord);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl GffRecordMap {
    pub fn from(file_path: &str, feature_type: Option<&FeatureType>) -> anyhow::Result<Self> {
        let (lines_of_words, _) = read_lines_of_words_delim(file_path, &['\t', ','], -1)?;

        let mut records: HashMap<GeneId, GffRecord> = HashMap::new();

        let parsed_records = lines_of_words
            .into_iter()
            .par_bridge()
            .filter_map(|x| parse_gff(x, feature_type))
            .collect::<Vec<_>>();

        for new_rec in parsed_records.iter() {
            let gene_id = &new_rec.gene_id;

            if records.contains_key(gene_id) {
                if let Some(rec) = records.get_mut(&gene_id) {
                    rec.start = rec.start.min(new_rec.start);
                    rec.stop = rec.stop.max(new_rec.stop);
                }
            } else {
                records.insert(gene_id.clone(), new_rec.clone());
            }
        }

        Ok(Self { records })
    }

    pub fn count_gene_types(&self) -> HashMap<GeneType, usize> {
        let mut gene_type_counts = HashMap::new();

        for record in self.records.values() {
            *gene_type_counts
                .entry(record.gene_type.clone())
                .or_insert(0) += 1;
        }

        gene_type_counts
    }

    pub fn subset(&mut self, target_gene_type: GeneType) {
        self.records
            .retain(|_, rec| rec.gene_type == target_gene_type);
    }

    /// Create a new empty `GffRecordMap`
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Get an iterator over the records
    pub fn iter(&self) -> Iter<'_, GeneId, GffRecord> {
        self.records.iter()
    }

    /// Insert a new `GffRecord` into the map
    pub fn insert(&mut self, gene_id: GeneId, record: GffRecord) {
        self.records.insert(gene_id, record);
    }

    /// Get a reference to a `GffRecord` by `GeneId`
    pub fn get(&self, gene_id: &GeneId) -> Option<&GffRecord> {
        self.records.get(gene_id)
    }

    /// Remove a `GffRecord` by `GeneId`
    pub fn remove(&mut self, gene_id: &GeneId) -> Option<GffRecord> {
        self.records.remove(gene_id)
    }

    /// Check if a `GeneId` exists in the map
    pub fn contains(&self, gene_id: &GeneId) -> bool {
        self.records.contains_key(gene_id)
    }

    /// Get all `GeneId`s in the map
    pub fn gene_ids(&self) -> Vec<&GeneId> {
        self.records.keys().collect()
    }

    /// Get all `GffRecord`s in the map
    pub fn records(&self) -> Vec<&GffRecord> {
        self.records.values().collect()
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

impl Into<FeatureType> for &str {
    fn into(self) -> FeatureType {
        match self.as_ref() {
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
            FeatureType::StopCodon => Box::from("stop_,codon"),
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

impl Into<GeneType> for Box<str> {
    fn into(self) -> GeneType {
        self.as_ref().into()
    }
}

impl Into<GeneType> for &str {
    fn into(self) -> GeneType {
        match self {
            "protein_coding" => GeneType::CodingGene,
            "pseudogene" => GeneType::PseudoGene,
            "lncRNA" => GeneType::LincRNA,
            "snRNA" => GeneType::SnRNA,
            _ if self.ends_with("pseudogene") => GeneType::PseudoGene,
            _ if self.ends_with("coding") => GeneType::CodingGene,
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
pub fn parse_gff(
    words: Vec<Box<str>>,
    target_feature_type: Option<&FeatureType>,
) -> Option<GffRecord> {
    const SPLIT_SEP: char = ';';
    const NUM_FIELDS: usize = 9;

    if words.len() != NUM_FIELDS {
        return None;
    }

    let feature_type: FeatureType = words[2].as_ref().into();

    let is_target_feature = target_feature_type
        .map(|x| *x == feature_type)
        .unwrap_or(true);

    if !is_target_feature {
        return None;
    }

    let seqname = words[0].clone();
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
