use crate::gff::GeneId;
use crate::sam::Strand;
use rustc_hash::FxHashMap;

/// Compare chromosome names ignoring optional "chr" prefix.
///
/// Handles mixed conventions (e.g., "chr1" vs "1", "chrX" vs "X").
pub fn chr_eq(a: &str, b: &str) -> bool {
    chr_stripped(a) == chr_stripped(b)
}

/// Strip the "chr" prefix from a chromosome name for use as a lookup key.
pub fn chr_stripped(s: &str) -> &str {
    s.strip_prefix("chr").unwrap_or(s)
}

/// Parsed genomic coordinate for an ATAC peak.
#[derive(Debug, Clone)]
pub struct PeakCoord {
    pub chr: Box<str>,
    pub start: i64,
    pub end: i64,
}

/// Gene TSS position parsed from GFF.
#[derive(Debug, Clone)]
pub struct GeneTss {
    pub chr: Box<str>,
    pub tss: i64,
}

/// Gene TSS position **with strand** parsed from GFF. Like [`GeneTss`]
/// but retains the strand so callers (e.g. strand-resolved genomic
/// pileups) can split forward (Watson) from backward (Crick) genes.
#[derive(Debug, Clone)]
pub struct GeneLoc {
    pub chr: Box<str>,
    /// Transcription start site (start on `+`, stop on `-`).
    pub tss: i64,
    pub strand: Strand,
}

/// Gene annotation simplified for eQTL and cis-regulatory analysis.
#[derive(Debug, Clone)]
pub struct Gene {
    pub gene_id: GeneId,
    pub gene_name: Option<Box<str>>,
    pub chromosome: Box<str>,
    pub tss: u64,
    pub strand: Strand,
}

/// Collection of gene annotations with cis window utilities.
#[derive(Debug, Clone)]
pub struct GeneAnnotations {
    pub genes: Vec<Gene>,
    pub cis_window: u64,
}

impl GeneAnnotations {
    /// Get cis-regulatory window for a gene (TSS ± cis_window).
    pub fn cis_region(&self, gene_idx: usize) -> (u64, u64) {
        let gene = &self.genes[gene_idx];
        let start = gene.tss.saturating_sub(self.cis_window);
        let end = gene.tss + self.cis_window;
        (start, end)
    }

    /// Get SNP indices within cis window for a gene.
    pub fn cis_snp_indices(
        &self,
        gene_idx: usize,
        snp_positions: &[u64],
        snp_chromosomes: &[Box<str>],
    ) -> Vec<usize> {
        let gene = &self.genes[gene_idx];
        let (start, end) = self.cis_region(gene_idx);

        snp_positions
            .iter()
            .enumerate()
            .filter(|(idx, &pos)| {
                chr_eq(&snp_chromosomes[*idx], &gene.chromosome) && pos >= start && pos <= end
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter genes to a specific genomic region.
    pub fn filter_to_region(&self, chromosome: &str, start: u64, end: u64) -> GeneAnnotations {
        let filtered_genes: Vec<Gene> = self
            .genes
            .iter()
            .filter(|g| chr_eq(g.chromosome.as_ref(), chromosome) && g.tss >= start && g.tss <= end)
            .cloned()
            .collect();

        GeneAnnotations {
            genes: filtered_genes,
            cis_window: self.cis_window,
        }
    }
}

/// The one coordinate grammar: `chr:start-end` / `chr_start_end`, and
/// (when `allow_position`) a single position `chr:pos` / `chr_pos` as the
/// one-base interval `[pos, pos + 1)`. The chromosome comes back as
/// written. `None` when the name is not a coordinate.
fn parse_coordinate(name: &str, allow_position: bool) -> Option<PeakCoord> {
    let name = name.trim();
    let (chr, rest) = name.split_once(':').or_else(|| name.split_once('_'))?;
    if chr.is_empty() {
        return None;
    }
    let (start, end) = match rest.split_once('-').or_else(|| rest.split_once('_')) {
        Some((s, e)) => (s.parse::<i64>().ok()?, e.parse::<i64>().ok()?),
        None if allow_position => {
            let pos = rest.parse::<i64>().ok()?;
            (pos, pos + 1)
        }
        None => return None,
    };
    (end > start).then_some(PeakCoord {
        chr: chr.into(),
        start,
        end,
    })
}

/// Parse peak names in "chr:start-end" or "chr_start_end" format, the
/// chromosome kept verbatim.
pub fn parse_peak_coordinates(peak_names: &[Box<str>]) -> Vec<Option<PeakCoord>> {
    peak_names
        .iter()
        .map(|name| parse_coordinate(name, false))
        .collect()
}

/// Parse one genomic region name: an interval `chr:start-end` /
/// `chr_start_end`, or a single position `chr:pos` / `chr_pos` (a SNP),
/// which becomes the one-base interval `[pos, pos + 1)`. The `chr` prefix
/// is dropped so `chr1` and `1` name the same chromosome. `None` when the
/// name is not a coordinate.
pub fn parse_region(name: &str) -> Option<PeakCoord> {
    let mut r = parse_coordinate(name, true)?;
    r.chr = chr_stripped(&r.chr).into();
    Some(r)
}

/// Tile a region onto fixed windows `[i·w, (i+1)·w)`: every window the
/// region overlaps, ascending, on the region's chromosome. A window of 0 is
/// the region itself.
pub fn tile_windows(region: &PeakCoord, window: i64) -> Vec<PeakCoord> {
    if window <= 0 {
        return vec![region.clone()];
    }
    let first = region.start.div_euclid(window);
    let last = (region.end - 1).max(region.start).div_euclid(window);
    (first..=last)
        .map(|i| PeakCoord {
            chr: region.chr.clone(),
            start: i * window,
            end: (i + 1) * window,
        })
        .collect()
}

impl std::fmt::Display for PeakCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}", self.chr, self.start, self.end)
    }
}

/// Find peaks within a cis window of a gene's TSS.
pub fn find_cis_peaks(
    gene_tss: &GeneTss,
    peak_coords: &[Option<PeakCoord>],
    window: i64,
) -> Vec<usize> {
    peak_coords
        .iter()
        .enumerate()
        .filter_map(|(idx, coord)| {
            let coord = coord.as_ref()?;
            if !chr_eq(coord.chr.as_ref(), gene_tss.chr.as_ref()) {
                return None;
            }
            let mid = (coord.start + coord.end) / 2;
            if (mid - gene_tss.tss).abs() <= window {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

/// Load gene TSS positions from a GFF/GTF file, aligned to `gene_names`
/// by exact symbol match. Strand-discarding projection of
/// [`load_gene_loci`].
pub fn load_gene_tss(
    gff_file: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<Option<GeneTss>>> {
    Ok(load_gene_loci(gff_file, gene_names)?
        .into_iter()
        .map(|loc| {
            loc.map(|l| GeneTss {
                chr: l.chr,
                tss: l.tss,
            })
        })
        .collect())
}

/// Load a `gene-symbol → (chr, TSS, strand)` map from a GFF/GTF file.
///
/// Only `gene` features are kept; each gene is keyed by its symbol,
/// falling back to the Ensembl id when the symbol is missing. Unlike
/// [`load_gene_tss`], the strand is retained so callers can split
/// forward/Watson genes from backward/Crick genes. Keys are the raw GFF
/// symbols — callers that need alias-tolerant matching (e.g.
/// `ENSG…_SYMBOL` row names) should canonicalize both sides themselves.
pub fn load_gene_loci_map(gff_file: &str) -> anyhow::Result<FxHashMap<Box<str>, GeneLoc>> {
    use crate::gff::{read_gff_record_vec, FeatureType, GeneSymbol};
    use log::info;

    let records = read_gff_record_vec(gff_file)?;

    let mut loc_map: FxHashMap<Box<str>, GeneLoc> = FxHashMap::default();
    for rec in &records {
        if rec.feature_type != FeatureType::Gene {
            continue;
        }
        let tss = match rec.strand {
            Strand::Forward => rec.start,
            Strand::Backward => rec.stop,
        };
        let key: Box<str> = match &rec.gene_name {
            GeneSymbol::Symbol(s) => s.clone(),
            GeneSymbol::Missing => {
                let id: Box<str> = rec.gene_id.clone().into();
                id
            }
        };
        loc_map.insert(
            key,
            GeneLoc {
                chr: rec.seqname.clone(),
                tss,
                strand: rec.strand,
            },
        );
    }

    info!("Loaded {} gene loci from GFF", loc_map.len());
    Ok(loc_map)
}

/// Load gene TSS positions **and strand** from a GFF/GTF file, aligned
/// to `gene_names` by exact symbol match (`None` where a name has no
/// matching `gene` record). For alias-tolerant matching, prefer
/// [`load_gene_loci_map`] + a caller-side canonicalizer.
pub fn load_gene_loci(
    gff_file: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<Option<GeneLoc>>> {
    use log::info;

    let loc_map = load_gene_loci_map(gff_file)?;

    let result: Vec<Option<GeneLoc>> = gene_names
        .iter()
        .map(|name| loc_map.get(name).cloned())
        .collect();

    let matched = result.iter().filter(|x| x.is_some()).count();
    info!("Matched {}/{} genes to GFF loci", matched, gene_names.len());

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_parse_intervals_and_positions_in_both_spellings_without_the_chr_prefix() {
        let r = parse_region("chr1:1000-2000").unwrap();
        assert_eq!((r.chr.as_ref(), r.start, r.end), ("1", 1000, 2000));
        let r = parse_region("1_1000_2000").unwrap();
        assert_eq!((r.chr.as_ref(), r.start, r.end), ("1", 1000, 2000));
        let r = parse_region("chrX:5000").unwrap();
        assert_eq!((r.chr.as_ref(), r.start, r.end), ("X", 5000, 5001));
        let r = parse_region("X_5000").unwrap();
        assert_eq!((r.chr.as_ref(), r.start, r.end), ("X", 5000, 5001));
        assert!(parse_region("TP53").is_none());
        assert!(parse_region("chr1:2000-1000").is_none(), "empty interval");
        assert!(parse_region(":1-2").is_none());
        assert_eq!(parse_region("chr2:10-20").unwrap().to_string(), "2:10-20");
    }

    #[test]
    fn tiling_covers_every_overlapped_window_and_only_those() {
        let r = parse_region("chr1:4999-10001").unwrap();
        let w: Vec<String> = tile_windows(&r, 5000)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(w, vec!["1:0-5000", "1:5000-10000", "1:10000-15000"]);
        let snp = parse_region("chr1:5000").unwrap();
        let w: Vec<String> = tile_windows(&snp, 5000)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            w,
            vec!["1:5000-10000"],
            "a position at a boundary lands in one window"
        );
        let exact = parse_region("chr1:5000-10000").unwrap();
        assert_eq!(
            tile_windows(&exact, 5000).len(),
            1,
            "an exact window is one window"
        );
        assert_eq!(
            tile_windows(&exact, 0).len(),
            1,
            "window 0 keeps the region"
        );
        assert_eq!(tile_windows(&exact, 0)[0].to_string(), "1:5000-10000");
    }

    #[test]
    fn load_gene_loci_keeps_strand_and_tss() {
        // Two genes: AAA on +, BBB on -. TSS = start on +, stop on -.
        let gtf = "\
chr1\tHAVANA\tgene\t100\t200\t.\t+\t.\tgene_id \"ENSG001\"; gene_name \"AAA\"; gene_type \"protein_coding\"
chr1\tHAVANA\tgene\t400\t600\t.\t-\t.\tgene_id \"ENSG002\"; gene_name \"BBB\"; gene_type \"protein_coding\"
chr1\tHAVANA\texon\t100\t150\t.\t+\t.\tgene_id \"ENSG001\"; gene_name \"AAA\"
";
        let path = std::env::temp_dir().join(format!("genloci_{}.gtf", std::process::id()));
        std::fs::write(&path, gtf).unwrap();
        let names: Vec<Box<str>> = vec!["AAA".into(), "BBB".into(), "MISSING".into()];
        let loci = load_gene_loci(path.to_str().unwrap(), &names).unwrap();
        std::fs::remove_file(&path).ok();

        let aaa = loci[0].as_ref().expect("AAA present");
        assert!(matches!(aaa.strand, Strand::Forward));
        assert_eq!(aaa.tss, 100);
        assert_eq!(chr_stripped(&aaa.chr), "1");

        let bbb = loci[1].as_ref().expect("BBB present");
        assert!(matches!(bbb.strand, Strand::Backward));
        assert_eq!(bbb.tss, 600);

        assert!(loci[2].is_none());
    }
}
