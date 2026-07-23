use crate::apa::em::EmResult;
use crate::apa::fragment::FragmentRecord;
use crate::apa::utr_region::UtrRegion;
use genomic_data::sam::{CellBarcode, UmiBarcode};
use rustc_hash::FxHashMap as HashMap;

/// A cell-level count at a specific pA site.
pub struct CellSiteCount {
    /// Batch (replicate) index the contributing reads came from. The
    /// SCAPE fit is shared across batches, but counts are emitted per batch.
    pub batch: u32,
    pub cell_barcode: CellBarcode,
    /// Site identifier: `{gene}/apa/{component}` (see [`crate::apa::site_id`]).
    pub site_id: Box<str>,
    /// UMI-deduplicated count
    pub count: usize,
}

/// Metadata for one APA site (for Parquet annotation output).
#[derive(Clone)]
pub struct ApaSiteAnnotation {
    pub site_id: Box<str>,
    pub gene_name: Box<str>,
    pub chr: Box<str>,
    pub genomic_alpha: i64,
    pub beta: f32,
    pub genomic_start: i64,
    pub genomic_stop: i64,
    pub pi_weight: f32,
    pub expected_tail_length: f32,
    /// UTR length in nt. Recover normalized 5'→3' UTR position via
    /// `u = 1 - expected_tail_length / utr_length` ∈ [0, 1]
    /// (the UTR-local α is not stored directly — only `genomic_alpha`).
    pub utr_length: u32,
}

/// Assign fragments to mixture components via hard assignment (argmax γ),
/// then deduplicate by (cell, UMI, component) and count UMIs per cell
/// per site.
///
/// `cluster_idx[n]` is the cluster the n-th original fragment was
/// grouped into during pre-EM clustering — all fragments in a cluster
/// share the same posterior γ (they have identical features), so every
/// fragment in cluster m gets the same component assignment. Each
/// fragment still contributes its own (cell_barcode, umi) to the dedup
/// set, so the per-cell counts are exact.
pub fn assign_fragments_to_sites(
    fragments: &[FragmentRecord],
    cluster_idx: &[u32],
    frag_batch: &[u32],
    em_result: &EmResult,
    utr: &UtrRegion,
) -> (Vec<CellSiteCount>, Vec<ApaSiteAnnotation>) {
    debug_assert_eq!(
        fragments.len(),
        cluster_idx.len(),
        "cluster_idx must be 1:1 with fragments"
    );
    debug_assert_eq!(
        fragments.len(),
        frag_batch.len(),
        "frag_batch must be 1:1 with fragments"
    );

    // Hard-assign each fragment to the argmax component (skip noise = k=0)
    // and dedupe UMIs directly on their 64-bit hash — no per-fragment
    // `Box<str>` allocation. Dedup is keyed by (batch, cell, component) so
    // the same UMI seen in two replicates counts as two distinct molecules.
    use rustc_hash::FxHashSet;
    let mut cell_component_umis: HashMap<(u32, CellBarcode, usize), FxHashSet<u64>> =
        HashMap::default();

    for (n, frag) in fragments.iter().enumerate() {
        let m = cluster_idx[n] as usize;
        let gamma = em_result.gamma_row(m);

        // Find argmax component (including noise at k=0)
        let (best_k, _) = gamma
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Skip noise component (k=0)
        if best_k == 0 {
            continue;
        }

        // `UmiBarcode::Missing` shares the sentinel `u64::MAX`, preserving
        // the prior behaviour where all unset UMIs in a cell collapsed to
        // one observation rather than over-counting.
        let umi_h = match frag.umi {
            UmiBarcode::Hash(h) => h,
            UmiBarcode::Missing => u64::MAX,
        };
        let key = (frag_batch[n], frag.cell_barcode.clone(), best_k);

        cell_component_umis.entry(key).or_default().insert(umi_h);
    }

    // Build site_id for each active component
    // Format: GENE/apa/k (0-indexed component within gene)
    let make_site_id = |component_k: usize| -> (Box<str>, i64, i64) {
        let alpha = em_result.alphas[component_k - 1];
        let beta = em_result.betas[component_k - 1];
        let (gstart, gstop) = utr.alpha_to_genomic_range(alpha.into(), beta.into());
        let site_id = crate::apa::site_id(&utr.name, &(component_k - 1).to_string());
        (site_id, gstart, gstop)
    };

    // Convert to CellSiteCount
    let mut results = Vec::new();

    for ((batch, cell_barcode, component_k), umis) in &cell_component_umis {
        let (site_id, _, _) = make_site_id(*component_k);
        results.push(CellSiteCount {
            batch: *batch,
            cell_barcode: cell_barcode.clone(),
            site_id,
            count: umis.len(),
        });
    }

    // Build annotations for all non-pruned components
    let mut annotations = Vec::new();
    for (k, &alpha) in em_result.alphas.iter().enumerate() {
        let weight = em_result.weights[k + 1]; // k+1 because weights[0] is noise
        if weight <= 0.0 {
            continue;
        }
        let beta = em_result.betas[k];
        let (gstart, gstop) = utr.alpha_to_genomic_range(alpha.into(), beta.into());
        let genomic_alpha = match utr.strand {
            genomic_data::sam::Strand::Forward => utr.start + alpha as i64,
            genomic_data::sam::Strand::Backward => utr.end - alpha as i64,
        };
        let site_id = crate::apa::site_id(&utr.name, &k.to_string());
        // Clamp to [0, utr_length]: alpha is a UTR-local position seeded
        // by site_discovery; a tiny numerical drift past utr_length
        // would otherwise emit a meaningless negative tail length and
        // poison the `u = 1 - tail / utr_length` reconstruction.
        let expected_tail_length = (utr.utr_length as f32 - alpha).max(0.0);
        annotations.push(ApaSiteAnnotation {
            site_id,
            gene_name: utr.name.clone(),
            chr: utr.chr.clone(),
            genomic_alpha,
            beta,
            genomic_start: gstart,
            genomic_stop: gstop,
            pi_weight: weight,
            expected_tail_length,
            utr_length: utr.utr_length as u32,
        });
    }

    (results, annotations)
}

/// Fast PDUI assignment (no θ-model / EM): hard-assign each fragment to the
/// **nearest** of two given pA sites by position, dedup UMIs, and emit per-cell
/// counts + the two site annotations. Approximate versus the SCAPE soft
/// assignment, but PDUI only needs the proximal/distal split, so a nearest-site
/// rule suffices for the default (non-`--mixture`) path. `site_alphas` are the
/// two UTR-local pA positions (components 0 and 1).
pub fn assign_fragments_two_site_fast(
    fragments: &[FragmentRecord],
    frag_batch: &[u32],
    site_alphas: [f32; 2],
    beta: f32,
    utr: &UtrRegion,
) -> (Vec<CellSiteCount>, Vec<ApaSiteAnnotation>) {
    use rustc_hash::FxHashSet;
    let mut cell_site_umis: HashMap<(u32, CellBarcode, usize), FxHashSet<u64>> = HashMap::default();
    let mut site_totals = [0usize; 2];

    for (n, frag) in fragments.iter().enumerate() {
        let pos = frag.pa_site.unwrap_or(frag.x + frag.l);
        let k = usize::from((pos - site_alphas[0]).abs() > (pos - site_alphas[1]).abs());
        site_totals[k] += 1;
        let umi_h = match frag.umi {
            UmiBarcode::Hash(h) => h,
            UmiBarcode::Missing => u64::MAX,
        };
        cell_site_umis
            .entry((frag_batch[n], frag.cell_barcode.clone(), k))
            .or_default()
            .insert(umi_h);
    }

    let results: Vec<CellSiteCount> = cell_site_umis
        .iter()
        .map(|((batch, cell, k), umis)| CellSiteCount {
            batch: *batch,
            cell_barcode: cell.clone(),
            site_id: crate::apa::site_id(&utr.name, &k.to_string()),
            count: umis.len(),
        })
        .collect();

    let total = (site_totals[0] + site_totals[1]).max(1) as f32;
    let annotations: Vec<ApaSiteAnnotation> = (0..2)
        .map(|k| {
            let alpha = site_alphas[k];
            let (gstart, gstop) = utr.alpha_to_genomic_range(alpha.into(), beta.into());
            let genomic_alpha = match utr.strand {
                genomic_data::sam::Strand::Forward => utr.start + alpha as i64,
                genomic_data::sam::Strand::Backward => utr.end - alpha as i64,
            };
            ApaSiteAnnotation {
                site_id: crate::apa::site_id(&utr.name, &k.to_string()),
                gene_name: utr.name.clone(),
                chr: utr.chr.clone(),
                genomic_alpha,
                beta,
                genomic_start: gstart,
                genomic_stop: gstop,
                pi_weight: site_totals[k] as f32 / total,
                expected_tail_length: (utr.utr_length as f32 - alpha).max(0.0),
                utr_length: utr.utr_length as u32,
            }
        })
        .collect();

    (results, annotations)
}

#[cfg(test)]
mod tests;
