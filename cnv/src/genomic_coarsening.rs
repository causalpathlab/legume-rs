//! Genome-constrained feature coarsening for CNV detection.
//!
//! Greedy bottom-up merging of adjacent genes on the same chromosome,
//! guided by Pearson correlation of their `log(mu_resid)` profiles
//! across pseudobulk samples.

use std::collections::BinaryHeap;

use nalgebra::DMatrix;

/// A contiguous block of genes on a single chromosome.
#[derive(Debug, Clone)]
pub struct GenomicBlock {
    /// Start index in the genome-ordered gene array (inclusive).
    pub start: usize,
    /// End index in the genome-ordered gene array (exclusive).
    pub end: usize,
    /// Chromosome name.
    pub chromosome: Box<str>,
}

impl GenomicBlock {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Result of genome-constrained greedy coarsening.
#[derive(Debug, Clone)]
pub struct GenomicCoarsening {
    /// Coarsened blocks in genome order.
    pub blocks: Vec<GenomicBlock>,
}

impl GenomicCoarsening {
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

impl GenomicCoarsening {
    /// Aggregate a genome-ordered `[G × S]` matrix to `[B × S]` by averaging
    /// genes within each block.
    pub fn aggregate_to_blocks(&self, log_resid: &DMatrix<f32>) -> DMatrix<f32> {
        let s = log_resid.ncols();
        let mut out = DMatrix::<f32>::zeros(self.num_blocks(), s);
        for (b, block) in self.blocks.iter().enumerate() {
            let n = block.len();
            if n > 0 {
                out.row_mut(b)
                    .copy_from(&log_resid.rows(block.start, n).row_mean());
            }
        }
        out
    }

    /// Expand block-level values `[B × C]` back to gene-level `[G × C]`.
    ///
    /// Each gene inherits the value of its parent block.
    pub fn expand_to_genes(&self, block_vals: &DMatrix<f32>, n_genes: usize) -> DMatrix<f32> {
        let c = block_vals.ncols();
        let mut out = DMatrix::<f32>::zeros(n_genes, c);
        for (b, block) in self.blocks.iter().enumerate() {
            for g in block.start..block.end {
                for j in 0..c {
                    out[(g, j)] = block_vals[(b, j)];
                }
            }
        }
        out
    }

    /// Expand a block-level vector (length B) to gene-level (length G).
    pub fn expand_vec_to_genes(&self, block_vals: &[usize], n_genes: usize) -> Vec<usize> {
        let mut out = vec![0; n_genes];
        for (b, block) in self.blocks.iter().enumerate() {
            out[block.start..block.end].fill(block_vals[b]);
        }
        out
    }

    /// Chromosome boundaries in block space.
    ///
    /// Returns `[(chr_name, block_start, block_end)]` analogous to
    /// `GenomeOrder::chr_boundaries` but in block indices.
    pub fn chr_block_boundaries(&self) -> Vec<(Box<str>, usize, usize)> {
        if self.blocks.is_empty() {
            return Vec::new();
        }
        let mut bounds = Vec::new();
        let mut cur_chr = &self.blocks[0].chromosome;
        let mut chr_start = 0usize;

        for (i, block) in self.blocks.iter().enumerate() {
            if &block.chromosome != cur_chr {
                bounds.push((cur_chr.clone(), chr_start, i));
                cur_chr = &block.chromosome;
                chr_start = i;
            }
        }
        bounds.push((cur_chr.clone(), chr_start, self.num_blocks()));
        bounds
    }
}

/// Pearson correlation between two slices.
fn pearson_corr(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Entry in the merge priority queue.
#[derive(Debug, Clone)]
struct MergeCandidate {
    /// Pearson correlation (used for ordering).
    corr: f32,
    /// Left block index.
    left: usize,
    /// Right block index (must be left + 1 in active block list).
    right: usize,
    /// Generation counter for the left block at time of creation.
    gen_left: u32,
    /// Generation counter for the right block at time of creation.
    gen_right: u32,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.corr == other.corr
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.corr
            .partial_cmp(&other.corr)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Greedy bottom-up coarsening of genome-ordered genes.
///
/// Merges adjacent genes on the same chromosome when their `log(mu_resid)`
/// profiles across samples have Pearson correlation above `corr_threshold`.
///
/// # Arguments
/// * `log_resid` — `[G_ordered × S]` matrix of log(mu_residual) in genome order
/// * `chr_bounds` — chromosome boundaries from `GenomeOrder`: `[(chr_name, start, end)]`
/// * `corr_threshold` — stop merging when max correlation falls below this (e.g., 0.5)
///
/// # Returns
/// `GenomicCoarsening` with merged blocks in genome order.
pub fn greedy_coarsen(
    log_resid: &DMatrix<f32>,
    chr_bounds: &[(Box<str>, usize, usize)],
    corr_threshold: f32,
) -> GenomicCoarsening {
    let n_genes = log_resid.nrows();
    let n_samples = log_resid.ncols();

    // Initialize: each gene is its own block
    // Track block membership: for each original gene, which block it belongs to
    // We use a union-find-like scheme: blocks[i] = (start, end, chr_idx, generation)

    // Block profiles: mean log(mu_resid) across genes in block, shape [S]
    let mut profiles: Vec<Vec<f32>> = (0..n_genes)
        .map(|g| (0..n_samples).map(|s| log_resid[(g, s)]).collect())
        .collect();

    // Block ranges: (start, end) in genome-ordered space
    let mut ranges: Vec<(usize, usize)> = (0..n_genes).map(|g| (g, g + 1)).collect();

    // Block sizes (number of original genes)
    let mut sizes: Vec<usize> = vec![1; n_genes];

    // Which chromosome each gene belongs to
    let mut gene_chr: Vec<usize> = vec![0; n_genes];
    for (chr_idx, &(_, start, end)) in chr_bounds.iter().enumerate() {
        gene_chr[start..end].fill(chr_idx);
    }

    // alive[i] = true if block i hasn't been merged into another
    let mut alive: Vec<bool> = vec![true; n_genes];

    // next/prev pointers for doubly-linked list of active blocks
    let mut next: Vec<Option<usize>> = (0..n_genes)
        .map(|i| if i + 1 < n_genes { Some(i + 1) } else { None })
        .collect();
    let mut prev: Vec<Option<usize>> = (0..n_genes)
        .map(|i| if i > 0 { Some(i - 1) } else { None })
        .collect();

    // Invalidate links across chromosome boundaries
    for &(_, _, end) in chr_bounds.iter() {
        if end > 0 && end < n_genes {
            next[end - 1] = None;
            prev[end] = None;
        }
    }

    // Generation counters for stale detection
    let mut generation: Vec<u32> = vec![0; n_genes];

    // Build initial heap of adjacent pairs (same chromosome only)
    let mut heap = BinaryHeap::new();
    for i in 0..n_genes {
        if let Some(j) = next[i] {
            if gene_chr[i] == gene_chr[j] {
                let corr = pearson_corr(&profiles[i], &profiles[j]);
                heap.push(MergeCandidate {
                    corr,
                    left: i,
                    right: j,
                    gen_left: 0,
                    gen_right: 0,
                });
            }
        }
    }

    // Greedy merge loop
    while let Some(candidate) = heap.pop() {
        if candidate.corr < corr_threshold {
            break;
        }

        let l = candidate.left;
        let r = candidate.right;

        // Check staleness
        if !alive[l]
            || !alive[r]
            || candidate.gen_left != generation[l]
            || candidate.gen_right != generation[r]
        {
            continue;
        }

        // Check adjacency is still valid
        if next[l] != Some(r) {
            continue;
        }

        // Merge r into l
        let new_size = sizes[l] + sizes[r];
        // Update profile: weighted mean
        let r_profile: Vec<f32> = profiles[r][..n_samples].to_vec();
        let wl = sizes[l] as f32;
        let wr = sizes[r] as f32;
        let wt = new_size as f32;
        for (pl, &pr) in profiles[l][..n_samples].iter_mut().zip(&r_profile) {
            *pl = (*pl * wl + pr * wr) / wt;
        }
        ranges[l].1 = ranges[r].1;
        sizes[l] = new_size;
        generation[l] += 1;

        // Kill r
        alive[r] = false;

        // Update linked list: l.next = r.next
        next[l] = next[r];
        if let Some(rn) = next[r] {
            prev[rn] = Some(l);
        }

        // Recompute correlation with new left neighbour
        if let Some(ln) = prev[l] {
            if alive[ln] && gene_chr[ln] == gene_chr[l] {
                let corr = pearson_corr(&profiles[ln], &profiles[l]);
                heap.push(MergeCandidate {
                    corr,
                    left: ln,
                    right: l,
                    gen_left: generation[ln],
                    gen_right: generation[l],
                });
            }
        }

        // Recompute correlation with new right neighbour
        if let Some(rn) = next[l] {
            if alive[rn] && gene_chr[l] == gene_chr[rn] {
                let corr = pearson_corr(&profiles[l], &profiles[rn]);
                heap.push(MergeCandidate {
                    corr,
                    left: l,
                    right: rn,
                    gen_left: generation[l],
                    gen_right: generation[rn],
                });
            }
        }
    }

    // Collect surviving blocks in genome order
    let mut blocks: Vec<GenomicBlock> = Vec::new();
    for i in 0..n_genes {
        if alive[i] {
            let chr_idx = gene_chr[i];
            let chr_name = &chr_bounds[chr_idx].0;
            blocks.push(GenomicBlock {
                start: ranges[i].0,
                end: ranges[i].1,
                chromosome: chr_name.clone(),
            });
        }
    }

    log::info!(
        "Genomic coarsening: {} genes → {} blocks (corr threshold={:.2})",
        n_genes,
        blocks.len(),
        corr_threshold,
    );

    GenomicCoarsening { blocks }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_correlated_block() {
        // 20 genes, 10 samples
        // Genes 0-4: same signal, genes 5-9: same signal (different), genes 10-19: noise
        let g = 20;
        let s = 10;
        let mut data = DMatrix::<f32>::zeros(g, s);

        // Block 1: genes 0-4 share the same profile
        let signal1: Vec<f32> = vec![1.0, -0.5, 0.3, 0.8, -1.0, 0.2, 0.7, -0.3, 0.5, -0.8];
        for i in 0..5 {
            for j in 0..s {
                data[(i, j)] = signal1[j];
            }
        }

        // Block 2: genes 5-9 share a different profile
        let signal2: Vec<f32> = vec![-1.0, 0.5, -0.3, -0.8, 1.0, -0.2, -0.7, 0.3, -0.5, 0.8];
        for i in 5..10 {
            for j in 0..s {
                data[(i, j)] = signal2[j];
            }
        }

        // Genes 10-19: each has unique noise
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        use rand::prelude::*;
        for i in 10..20 {
            for j in 0..s {
                data[(i, j)] = rng.random_range(-1.0..1.0);
            }
        }

        let chr_bounds: Vec<(Box<str>, usize, usize)> = vec![("chr1".into(), 0, g)];
        let coarsening = greedy_coarsen(&data, &chr_bounds, 0.9);

        // Block 1 (genes 0-4) should be merged into one block
        // Block 2 (genes 5-9) should be merged into one block
        // Total should be much less than 20
        assert!(
            coarsening.num_blocks() < 15,
            "expected significant merging, got {} blocks",
            coarsening.num_blocks()
        );

        // First block should span genes 0-4
        assert_eq!(coarsening.blocks[0].start, 0);
        assert_eq!(coarsening.blocks[0].end, 5);

        // Second block should span genes 5-9
        assert_eq!(coarsening.blocks[1].start, 5);
        assert_eq!(coarsening.blocks[1].end, 10);
    }

    #[test]
    fn test_no_cross_chromosome_merge() {
        // Two chromosomes, each with identical signals — should NOT merge across boundary
        let g = 10;
        let s = 5;
        let signal: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut data = DMatrix::<f32>::zeros(g, s);
        for i in 0..g {
            for j in 0..s {
                data[(i, j)] = signal[j];
            }
        }

        let chr_bounds: Vec<(Box<str>, usize, usize)> =
            vec![("chr1".into(), 0, 5), ("chr2".into(), 5, 10)];
        let coarsening = greedy_coarsen(&data, &chr_bounds, 0.5);

        // Should get exactly 2 blocks (one per chromosome), never merging across
        assert_eq!(coarsening.num_blocks(), 2);
        assert_eq!(&*coarsening.blocks[0].chromosome, "chr1");
        assert_eq!(coarsening.blocks[0].start, 0);
        assert_eq!(coarsening.blocks[0].end, 5);
        assert_eq!(&*coarsening.blocks[1].chromosome, "chr2");
        assert_eq!(coarsening.blocks[1].start, 5);
        assert_eq!(coarsening.blocks[1].end, 10);
    }

    #[test]
    fn test_aggregate_and_expand() {
        let g = 6;
        let s = 3;
        let data = DMatrix::from_row_slice(
            g,
            s,
            &[
                1.0, 2.0, 3.0, // gene 0
                3.0, 4.0, 5.0, // gene 1
                10.0, 20.0, 30.0, // gene 2
                6.0, 8.0, 10.0, // gene 3
                6.0, 8.0, 10.0, // gene 4
                6.0, 8.0, 10.0, // gene 5
            ],
        );

        let coarsening = GenomicCoarsening {
            blocks: vec![
                GenomicBlock {
                    start: 0,
                    end: 2,
                    chromosome: "chr1".into(),
                },
                GenomicBlock {
                    start: 2,
                    end: 3,
                    chromosome: "chr1".into(),
                },
                GenomicBlock {
                    start: 3,
                    end: 6,
                    chromosome: "chr2".into(),
                },
            ],
        };

        let agg = coarsening.aggregate_to_blocks(&data);
        assert_eq!(agg.nrows(), 3);
        assert_eq!(agg.ncols(), 3);

        // Block 0: mean of genes 0,1
        assert!((agg[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((agg[(0, 1)] - 3.0).abs() < 1e-6);

        // Block 1: gene 2 alone
        assert!((agg[(1, 0)] - 10.0).abs() < 1e-6);

        // Block 2: mean of genes 3,4,5
        assert!((agg[(2, 0)] - 6.0).abs() < 1e-6);
        assert!((agg[(2, 1)] - 8.0).abs() < 1e-6);

        // Expand back
        let block_vals = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expanded = coarsening.expand_to_genes(&block_vals, g);
        assert_eq!(expanded.nrows(), g);
        // Genes 0,1 get block 0 values
        assert!((expanded[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((expanded[(1, 0)] - 1.0).abs() < 1e-6);
        // Gene 2 gets block 1
        assert!((expanded[(2, 0)] - 3.0).abs() < 1e-6);
        // Genes 3,4,5 get block 2
        assert!((expanded[(3, 0)] - 5.0).abs() < 1e-6);
        assert!((expanded[(5, 0)] - 5.0).abs() < 1e-6);
    }
}
