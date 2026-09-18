//! Resolve data-beans row (gene) names to genomic loci from a GFF/GTF.
//!
//! Row names in a `data-beans` backend come in a few shapes:
//! `ENSG00000243485_MIR1302-2HG`, a bare symbol `MIR1302-2HG`, a bare
//! Ensembl id, or any of these with a `/modality/...` suffix (faba).
//!
//! Matching goes through the workspace's one canonical gene matcher,
//! [`data_beans::utilities::name_matching::GeneIndex`], built over the GFF
//! vocabulary as `{ensg}_{symbol}`: case-insensitive; exact → symbol →
//! Ensembl id → decomposed `ENSG_SYMBOL[/aux]` → HGNC alias table
//! (`HIST1H4C` ↔ `H4C3`, `MARCH2` ↔ `MARCHF2`) → flexible fallback. The
//! Ensembl version suffix is dropped on both sides first.
//!
//! Same source of truth as `faba` (`genomic_data::gff::GffRecordMap`,
//! `gene` features stretched over all their records).

use data_beans::utilities::name_matching::GeneIndex;
use genomic_data::gff::{GeneId, GeneSymbol, GffRecordMap};
use rayon::prelude::*;

/// One gene's locus, from the GFF `gene` feature. Coordinates are GFF
/// 1-based inclusive `[start, stop]`.
#[derive(Debug, Clone)]
pub struct GeneLocus {
    pub chromosome: Box<str>,
    pub start: i64,
    pub stop: i64,
    /// Transcription start: `start` on `+`, `stop` on `-`.
    pub tss: i64,
    pub gene_id: Box<str>,
    pub symbol: Box<str>,
}

impl GeneLocus {
    /// BED-style interval name `chr:start0-end` (0-based half-open) — the
    /// `chr:start-end` grammar `genomic_data::coordinates::parse_region`
    /// reads back.
    pub fn interval_name(&self) -> Box<str> {
        format!("{}:{}-{}", self.chromosome, self.start - 1, self.stop).into()
    }

    /// `{gene_id}_{symbol}`, faba's gene key.
    pub fn gene_key(&self) -> Box<str> {
        format!("{}_{}", self.gene_id, self.symbol).into()
    }
}

/// Lookup from GFF through the canonical [`GeneIndex`] matcher: a row name
/// in any of the supported shapes resolves to one [`GeneLocus`].
pub struct GeneLocusIndex {
    /// One per GFF gene, in coordinate order (first wins on duplicate keys).
    loci: Vec<GeneLocus>,
    /// Built over `{ensg}_{symbol}` keys, parallel to `loci`.
    index: GeneIndex,
}

/// Drop a faba `/modality/...` suffix and the Ensembl version on the
/// leading segment (`ENSG00000000003.15_TSPAN6` → `ENSG00000000003_TSPAN6`),
/// so the version never blocks a match.
fn normalize_query(row_name: &str) -> String {
    let core = row_name.split('/').next().unwrap_or(row_name).trim();
    let (head, rest) = match core.split_once('_') {
        Some((h, r)) => (h, Some(r)),
        None => (core, None),
    };
    let head = if head.len() >= 4 && head[..4].eq_ignore_ascii_case("ensg") {
        head.split('.').next().unwrap_or(head)
    } else {
        head
    };
    match rest {
        Some(r) => format!("{head}_{r}"),
        None => head.to_string(),
    }
}

impl GeneLocusIndex {
    /// Parse the GFF/GTF (`gene` features only) and build the index.
    pub fn from_gff(gff_file: &str) -> anyhow::Result<Self> {
        let map = GffRecordMap::from(gff_file)?;
        Ok(Self::from_record_map(&map))
    }

    pub fn from_record_map(map: &GffRecordMap) -> Self {
        let mut loci = Vec::new();
        let mut keys: Vec<Box<str>> = Vec::new();
        for rec in map.records() {
            let gene_id: Box<str> = match &rec.gene_id {
                GeneId::Ensembl(id) => id.clone(),
                GeneId::Missing => continue,
            };
            let symbol: Box<str> = match &rec.gene_name {
                GeneSymbol::Symbol(s) => s.clone(),
                GeneSymbol::Missing => gene_id.clone(),
            };
            let tss = match rec.strand {
                genomic_data::sam::Strand::Forward => rec.start,
                genomic_data::sam::Strand::Backward => rec.stop,
            };
            keys.push(format!("{gene_id}_{symbol}").into());
            loci.push(GeneLocus {
                chromosome: rec.seqname.clone(),
                start: rec.start,
                stop: rec.stop,
                tss,
                gene_id,
                symbol,
            });
        }
        let index = GeneIndex::build(&keys);
        log::info!("GFF: {} genes indexed as {{ensg}}_{{symbol}}", loci.len());
        Self { loci, index }
    }

    /// Resolve one row name through the canonical matcher.
    pub fn resolve(&self, row_name: &str) -> Option<&GeneLocus> {
        let q = normalize_query(row_name);
        if q.is_empty() {
            return None;
        }
        self.index.match_gene(&q).map(|i| &self.loci[i])
    }

    /// Resolve every row name (in parallel); `None` where a row has no GFF gene.
    pub fn resolve_all(&self, row_names: &[Box<str>]) -> Vec<Option<GeneLocus>> {
        let out: Vec<Option<GeneLocus>> = row_names
            .par_iter()
            .map(|n| self.resolve(n).cloned())
            .collect();
        let matched = out.iter().filter(|x| x.is_some()).count();
        log::info!(
            "GFF: matched {}/{} row names to gene loci",
            matched,
            row_names.len()
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genomic_data::gff::{FeatureType, GeneType, GffRecord, TranscriptId};
    use genomic_data::sam::Strand;

    fn rec(id: &str, sym: &str, chr: &str, start: i64, stop: i64, strand: Strand) -> GffRecord {
        GffRecord {
            seqname: chr.into(),
            feature_type: FeatureType::Gene,
            start,
            stop,
            strand,
            gene_id: GeneId::Ensembl(id.into()),
            gene_name: GeneSymbol::Symbol(sym.into()),
            gene_type: GeneType::CodingGene,
            transcript_id: TranscriptId::Missing,
        }
    }

    fn index() -> GeneLocusIndex {
        let records = vec![
            rec(
                "ENSG1",
                "TP53",
                "chr17",
                7_661_779,
                7_687_538,
                Strand::Backward,
            ),
            rec(
                "ENSG2",
                "MYC",
                "chr8",
                127_735_434,
                127_742_951,
                Strand::Forward,
            ),
        ];
        let map = GffRecordMap::from_map(
            genomic_data::gff::build_gene_map(&records, Some(&FeatureType::Gene)).unwrap(),
        );
        GeneLocusIndex::from_record_map(&map)
    }

    #[test]
    fn resolves_all_name_shapes() {
        let idx = index();
        assert_eq!(idx.resolve("TP53").unwrap().gene_id.as_ref(), "ENSG1");
        assert_eq!(idx.resolve("ENSG1").unwrap().symbol.as_ref(), "TP53");
        assert_eq!(idx.resolve("ENSG1.12").unwrap().symbol.as_ref(), "TP53");
        assert_eq!(idx.resolve("ENSG1_TP53").unwrap().symbol.as_ref(), "TP53");
        assert_eq!(
            idx.resolve("ENSGX_MYC/count/spliced")
                .unwrap()
                .gene_id
                .as_ref(),
            "ENSG2"
        );
        // Canonical matcher: case-insensitive, and an ENSG row whose symbol
        // moved between HGNC releases still lands on the same locus.
        assert_eq!(idx.resolve("tp53").unwrap().gene_id.as_ref(), "ENSG1");
        assert_eq!(
            idx.resolve("ENSG1.12_TP53").unwrap().symbol.as_ref(),
            "TP53"
        );
        assert!(idx.resolve("NOPE").is_none());
    }

    #[test]
    fn hgnc_alias_resolves_through_canonical_matcher() {
        let records = vec![rec(
            "ENSG9",
            "H4C3",
            "chr6",
            26_104_000,
            26_104_900,
            Strand::Forward,
        )];
        let map = GffRecordMap::from_map(
            genomic_data::gff::build_gene_map(&records, Some(&FeatureType::Gene)).unwrap(),
        );
        let idx = GeneLocusIndex::from_record_map(&map);
        // Old HGNC name in the matrix, new one in the GFF.
        assert_eq!(idx.resolve("HIST1H4C").unwrap().gene_id.as_ref(), "ENSG9");
        assert_eq!(
            idx.resolve("ENSGZ_HIST1H4C").unwrap().gene_id.as_ref(),
            "ENSG9"
        );
    }

    #[test]
    fn tss_follows_strand_and_interval_is_bed() {
        let idx = index();
        let tp53 = idx.resolve("TP53").unwrap();
        assert_eq!(tp53.tss, 7_687_538);
        assert_eq!(tp53.interval_name().as_ref(), "chr17:7661778-7687538");
        let myc = idx.resolve("MYC").unwrap();
        assert_eq!(myc.tss, 127_735_434);
        assert_eq!(myc.gene_key().as_ref(), "ENSG2_MYC");
    }
}
