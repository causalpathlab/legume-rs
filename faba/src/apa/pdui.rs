use crate::apa::cell_assign::{ApaSiteAnnotation, CellSiteCount};
use genomic_data::sam::{CellBarcode, Strand};
use rustc_hash::FxHashMap as HashMap;

/// Result of PDUI computation for one gene/UTR
#[allow(dead_code)]
pub struct PduiResult {
    /// Gene/UTR name
    pub gene_name: Box<str>,
    /// Per-cell `(proximal_count, distal_count)` — the channel counts the
    /// co-embedding consumes (`{gene}/apa/{proximal,distal}`); PDUI =
    /// `distal / (proximal + distal)` is derived where needed.
    pub cell_counts: Vec<(CellBarcode, usize, usize)>,
    /// Proximal site annotation
    pub proximal: ApaSiteAnnotation,
    /// Distal site annotation
    pub distal: ApaSiteAnnotation,
}

/// Compute PDUI for a single gene given its cell-site counts and site annotations.
///
/// Returns None if the gene doesn't have exactly 2 active sites.
pub fn compute_pdui(
    counts: &[&CellSiteCount],
    annotations: &[ApaSiteAnnotation],
    strand: Strand,
) -> Option<PduiResult> {
    if annotations.len() != 2 {
        return None;
    }

    // Identify proximal vs distal by genomic_alpha + strand
    let (proximal_idx, distal_idx) = match strand {
        // Forward: higher alpha = more distal (further into 3'UTR)
        Strand::Forward => {
            if annotations[0].genomic_alpha < annotations[1].genomic_alpha {
                (0, 1)
            } else {
                (1, 0)
            }
        }
        // Backward: lower alpha = more distal
        Strand::Backward => {
            if annotations[0].genomic_alpha > annotations[1].genomic_alpha {
                (0, 1)
            } else {
                (1, 0)
            }
        }
    };

    let proximal = annotations[proximal_idx].clone();
    let distal = annotations[distal_idx].clone();

    // Build per-cell counts for each site
    let mut proximal_counts: HashMap<CellBarcode, usize> = HashMap::default();
    let mut distal_counts: HashMap<CellBarcode, usize> = HashMap::default();

    for c in counts {
        if c.site_id == proximal.site_id {
            *proximal_counts.entry(c.cell_barcode.clone()).or_default() += c.count;
        } else if c.site_id == distal.site_id {
            *distal_counts.entry(c.cell_barcode.clone()).or_default() += c.count;
        }
    }

    // Compute PDUI per cell (for cells with any count at either site)
    let mut all_cells: rustc_hash::FxHashSet<CellBarcode> = rustc_hash::FxHashSet::default();
    all_cells.extend(proximal_counts.keys().cloned());
    all_cells.extend(distal_counts.keys().cloned());

    let mut cell_counts = Vec::with_capacity(all_cells.len());
    for cell in all_cells {
        let p = *proximal_counts.get(&cell).unwrap_or(&0);
        let d = *distal_counts.get(&cell).unwrap_or(&0);
        if p + d > 0 {
            cell_counts.push((cell, p, d));
        }
    }

    Some(PduiResult {
        gene_name: annotations[0].gene_name.clone(),
        cell_counts,
        proximal,
        distal,
    })
}

#[cfg(test)]
mod tests;
