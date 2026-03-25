use crate::editing::ConversionSite;
use dashmap::DashMap;
use genomic_data::gff::{GeneId, GffRecordMap};
use genomic_data::sam::Strand;
/// Build a set of (chr, pos) for all A-to-I sites, used to mask m6A candidates
pub fn build_atoi_mask(
    sites: &DashMap<GeneId, Vec<ConversionSite>>,
    gff_map: &GffRecordMap,
) -> rustc_hash::FxHashSet<(Box<str>, i64)> {
    let mut mask = rustc_hash::FxHashSet::default();
    for entry in sites.iter() {
        let gene_id = entry.key();
        let sites = entry.value();
        if let Some(gff) = gff_map.get(gene_id) {
            let chr: Box<str> = format!("{}", gff.seqname).into();
            for site in sites {
                if let ConversionSite::AtoI { editing_pos, .. } = site {
                    mask.insert((chr.clone(), *editing_pos));
                }
            }
        }
    }
    mask
}

/// Filter m6A candidate sites that overlap with A-to-I editing positions.
/// For forward strand RAC: positions [conv-2, conv-1, conv] = (R, A, C)
/// For backward strand GTY: positions [conv, conv+1, conv+2] = (G, T, Y)
pub fn filter_m6a_by_atoi_mask(
    gene_sites: &DashMap<GeneId, Vec<ConversionSite>>,
    atoi_mask: &rustc_hash::FxHashSet<(Box<str>, i64)>,
    gff_map: &GffRecordMap,
) {
    if atoi_mask.is_empty() {
        return;
    }

    let gene_ids: Vec<GeneId> = gene_sites.iter().map(|e| e.key().clone()).collect();

    for gene_id in &gene_ids {
        if let Some(gff) = gff_map.get(gene_id) {
            let chr: Box<str> = format!("{}", gff.seqname).into();
            let strand = gff.strand;

            if let Some(mut sites) = gene_sites.get_mut(gene_id) {
                sites.retain(|site| {
                    match site {
                        ConversionSite::M6A { conversion_pos, .. } => {
                            let positions_to_check: [i64; 3] = match strand {
                                Strand::Forward => {
                                    [conversion_pos - 2, conversion_pos - 1, *conversion_pos]
                                }
                                Strand::Backward => {
                                    [*conversion_pos, conversion_pos + 1, conversion_pos + 2]
                                }
                            };
                            !positions_to_check
                                .iter()
                                .any(|&pos| atoi_mask.contains(&(chr.clone(), pos)))
                        }
                        ConversionSite::AtoI { .. } => true, // keep AtoI sites
                    }
                });
            }
        }
    }

    // Remove genes with no remaining sites
    gene_sites.retain(|_, v| !v.is_empty());
}

/// Filter poly-A sites that overlap with A-to-I editing positions.
/// Each poly-A site is a single position, so just check direct (chr, pos) membership.
pub fn filter_polya_by_atoi_mask(
    gene_sites: &DashMap<GeneId, Vec<i64>>,
    atoi_mask: &rustc_hash::FxHashSet<(Box<str>, i64)>,
    gff_map: &GffRecordMap,
) {
    if atoi_mask.is_empty() {
        return;
    }

    let gene_ids: Vec<GeneId> = gene_sites.iter().map(|e| e.key().clone()).collect();

    for gene_id in &gene_ids {
        if let Some(gff) = gff_map.get(gene_id) {
            let chr: Box<str> = format!("{}", gff.seqname).into();

            if let Some(mut sites) = gene_sites.get_mut(gene_id) {
                sites.retain(|&pos| !atoi_mask.contains(&(chr.clone(), pos)));
            }
        }
    }

    gene_sites.retain(|_, v| !v.is_empty());
}
