use crate::gff::GeneId;
use crate::sam::Strand;

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

/// Parse peak names in "chr:start-end" or "chr_start_end" format.
pub fn parse_peak_coordinates(peak_names: &[Box<str>]) -> Vec<Option<PeakCoord>> {
    peak_names
        .iter()
        .map(|name| {
            // Try chr:start-end
            if let Some((chr, rest)) = name.split_once(':') {
                if let Some((s, e)) = rest.split_once('-') {
                    if let (Ok(start), Ok(end)) = (s.parse::<i64>(), e.parse::<i64>()) {
                        return Some(PeakCoord {
                            chr: chr.into(),
                            start,
                            end,
                        });
                    }
                }
            }
            // Try chr_start_end
            let parts: Vec<&str> = name.splitn(3, '_').collect();
            if parts.len() == 3 {
                if let (Ok(start), Ok(end)) = (parts[1].parse::<i64>(), parts[2].parse::<i64>()) {
                    return Some(PeakCoord {
                        chr: parts[0].into(),
                        start,
                        end,
                    });
                }
            }
            None
        })
        .collect()
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

/// Load gene TSS positions from a GFF/GTF file.
pub fn load_gene_tss(
    gff_file: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<Option<GeneTss>>> {
    use crate::gff::{read_gff_record_vec, FeatureType, GeneSymbol};
    use log::info;
    use rustc_hash::FxHashMap as HashMap;

    let records = read_gff_record_vec(gff_file)?;

    let mut tss_map: HashMap<Box<str>, GeneTss> = Default::default();
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
        tss_map.insert(
            key,
            GeneTss {
                chr: rec.seqname.clone(),
                tss,
            },
        );
    }

    info!(
        "Loaded {} gene positions from GFF, matching against {} genes",
        tss_map.len(),
        gene_names.len()
    );

    let result: Vec<Option<GeneTss>> = gene_names
        .iter()
        .map(|name| tss_map.get(name).cloned())
        .collect();

    let matched = result.iter().filter(|x| x.is_some()).count();
    info!(
        "Matched {}/{} genes to GFF positions",
        matched,
        gene_names.len()
    );

    Ok(result)
}
