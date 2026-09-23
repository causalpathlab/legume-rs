//! Cis peak-to-gene candidates and fixed ABC-style contact weights.
//!
//! Every gene with a position gets the peaks whose midpoint lies within
//! `window` bp of its TSS, capped at the `max_per_gene` nearest. Each pair
//! carries the ABC contact `(d + pseudocount)^-γ`, normalised over the gene's
//! candidates, so a gene's weights are its shares of regulatory input. A peak
//! near several genes is a candidate for each of them.
//!
//! Peaks are sorted by midpoint per chromosome and each gene's window is found
//! by binary search: `O(P log P + G log P + pairs)`, never a scan over every
//! peak per gene.

use genomic_data::coordinates::{chr_stripped, GeneTss, PeakCoord};
use rustc_hash::FxHashMap;
use std::ops::Range;

/// Window, cap and contact kernel for the fixed peak-to-gene weights.
#[derive(Debug, Clone)]
pub struct AbcKernel {
    /// Max |peak midpoint − TSS| (bp) for a candidate.
    pub window: i64,
    /// Keep at most this many candidates per gene, nearest first.
    pub max_per_gene: usize,
    /// Power-law exponent `γ` of the contact `(d + pseudocount)^-γ`.
    pub gamma: f32,
    /// Distance pseudocount (bp) so a peak at the TSS has finite contact.
    pub pseudocount: f32,
}

impl Default for AbcKernel {
    fn default() -> Self {
        Self {
            window: 500_000,
            max_per_gene: 200,
            gamma: 1.0,
            pseudocount: 5_000.0,
        }
    }
}

impl AbcKernel {
    /// Unnormalised contact at distance `d` (bp).
    #[must_use]
    pub fn contact(&self, d: i64) -> f32 {
        (d.abs() as f32 + self.pseudocount).powf(-self.gamma)
    }
}

/// Candidate pairs grouped by gene (CSR over genes). Within a gene the pairs
/// are in ascending peak order.
#[derive(Debug, Clone)]
pub struct CisPairs {
    /// `gene_ptr[g]..gene_ptr[g + 1]` indexes gene `g`'s pairs.
    pub gene_ptr: Vec<usize>,
    pub peak: Vec<u32>,
    /// |peak midpoint − TSS| in bp.
    pub dist: Vec<i64>,
    /// Contact normalised over the gene's candidates (sums to 1 per gene).
    pub weight: Vec<f32>,
    /// Genes that had a position.
    pub n_genes_placed: usize,
    /// Peaks whose name did not parse as a coordinate.
    pub n_unparsed_peaks: usize,
    /// Peaks that are no gene's candidate (unparsed ones included).
    pub n_unreached_peaks: usize,
}

impl CisPairs {
    #[must_use]
    pub fn n_genes(&self) -> usize {
        self.gene_ptr.len() - 1
    }

    #[must_use]
    pub fn n_pairs(&self) -> usize {
        self.peak.len()
    }

    /// Range of gene `g`'s pairs in `peak` / `dist` / `weight`.
    #[must_use]
    pub fn gene(&self, g: usize) -> Range<usize> {
        self.gene_ptr[g]..self.gene_ptr[g + 1]
    }
}

fn midpoint(p: &PeakCoord) -> i64 {
    (p.start + p.end) / 2
}

/// Build the cis candidates of every gene (`genes[g] = None`: no position, no
/// candidates) over `peaks` (`None`: an unparseable peak name).
#[must_use]
pub fn build_cis_pairs(
    genes: &[Option<GeneTss>],
    peaks: &[Option<PeakCoord>],
    kernel: &AbcKernel,
) -> CisPairs {
    // Peaks by chromosome, sorted by midpoint.
    let mut by_chr: FxHashMap<&str, Vec<(i64, u32)>> = FxHashMap::default();
    for (i, p) in peaks.iter().enumerate() {
        if let Some(p) = p {
            by_chr
                .entry(chr_stripped(&p.chr))
                .or_default()
                .push((midpoint(p), i as u32));
        }
    }
    for v in by_chr.values_mut() {
        v.sort_unstable();
    }

    let mut gene_ptr = Vec::with_capacity(genes.len() + 1);
    gene_ptr.push(0);
    let (mut peak, mut dist, mut weight) = (Vec::new(), Vec::new(), Vec::new());
    let mut reached = vec![false; peaks.len()];
    let mut n_genes_placed = 0;
    let mut cand: Vec<(i64, u32)> = Vec::new();

    for g in genes {
        if let Some(g) = g {
            n_genes_placed += 1;
            if let Some(sorted) = by_chr.get(chr_stripped(&g.chr)) {
                let lo = sorted.partition_point(|&(m, _)| m < g.tss - kernel.window);
                let hi = sorted.partition_point(|&(m, _)| m <= g.tss + kernel.window);
                cand.clear();
                cand.extend(sorted[lo..hi].iter().map(|&(m, i)| ((m - g.tss).abs(), i)));
                if cand.len() > kernel.max_per_gene {
                    cand.sort_unstable();
                    cand.truncate(kernel.max_per_gene);
                }
                cand.sort_unstable_by_key(|&(_, i)| i);
                let total: f32 = cand.iter().map(|&(d, _)| kernel.contact(d)).sum();
                for &(d, i) in &cand {
                    peak.push(i);
                    dist.push(d);
                    weight.push(kernel.contact(d) / total);
                    reached[i as usize] = true;
                }
            }
        }
        gene_ptr.push(peak.len());
    }

    CisPairs {
        gene_ptr,
        peak,
        dist,
        weight,
        n_genes_placed,
        n_unparsed_peaks: peaks.iter().filter(|p| p.is_none()).count(),
        n_unreached_peaks: reached.iter().filter(|&&r| !r).count(),
    }
}
