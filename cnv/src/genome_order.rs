use rustc_hash::FxHashMap as HashMap;

/// Canonical chromosome ordering: chr1..22, chrX, chrY, chrM.
/// Returns None for unrecognized chromosomes.
fn chr_sort_key(chr: &str) -> Option<u32> {
    let stripped = chr.strip_prefix("chr").unwrap_or(chr);
    match stripped {
        "X" => Some(23),
        "Y" => Some(24),
        "M" | "MT" => Some(25),
        s => s.parse::<u32>().ok().filter(|n| (1..=22).contains(n)),
    }
}

/// A gene's genomic position for ordering along the genome.
#[derive(Debug, Clone)]
pub struct GenePosition {
    pub gene_idx: usize,
    pub chromosome: Box<str>,
    pub position: u64,
}

/// Result of ordering genes along the genome.
#[derive(Debug, Clone)]
pub struct GenomeOrder {
    /// Gene indices in genome order (length = number of genes on canonical chromosomes).
    pub ordered_indices: Vec<usize>,
    /// Chromosome boundary positions within `ordered_indices`.
    /// Each entry is (chr_name, start_offset, end_offset) where
    /// ordered_indices[start..end] are the genes on that chromosome.
    pub chr_boundaries: Vec<(Box<str>, usize, usize)>,
}

impl GenomeOrder {
    /// Build genome ordering from gene positions.
    ///
    /// Genes on non-canonical chromosomes (scaffolds, patches) are excluded.
    /// Genes are sorted by canonical chromosome order, then by position within chromosome.
    pub fn from_positions(genes: &[GenePosition]) -> Self {
        type ChrGroup = (Box<str>, Vec<(usize, u64)>);
        let mut chr_groups: HashMap<u32, ChrGroup> = HashMap::default();
        for g in genes {
            if let Some(key) = chr_sort_key(&g.chromosome) {
                chr_groups
                    .entry(key)
                    .or_insert_with(|| (g.chromosome.clone(), Vec::new()))
                    .1
                    .push((g.gene_idx, g.position));
            }
        }

        let mut chr_keys: Vec<u32> = chr_groups.keys().copied().collect();
        chr_keys.sort_unstable();

        let mut ordered_indices = Vec::with_capacity(genes.len());
        let mut chr_boundaries = Vec::with_capacity(chr_keys.len());

        for key in chr_keys {
            let (chr_name, group) = chr_groups.get_mut(&key).unwrap();
            group.sort_unstable_by_key(|&(_, pos)| pos);

            let start = ordered_indices.len();
            for &(idx, _) in group.iter() {
                ordered_indices.push(idx);
            }
            let end = ordered_indices.len();

            chr_boundaries.push((chr_name.clone(), start, end));
        }

        Self {
            ordered_indices,
            chr_boundaries,
        }
    }

    /// Number of genes in genome order (excluding non-canonical chromosomes).
    pub fn len(&self) -> usize {
        self.ordered_indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordered_indices.is_empty()
    }

    /// Reorder rows of a [genes x samples] matrix to genome order.
    ///
    /// Returns a new matrix with `self.len()` rows (non-canonical chromosomes dropped).
    pub fn reorder_rows(
        &self,
        mat: &nalgebra::DMatrix<f32>,
    ) -> anyhow::Result<nalgebra::DMatrix<f32>> {
        matrix_util::dmatrix_util::subset_rows(mat, self.ordered_indices.iter().copied())
    }
}

/// Build genome ordering from parallel slices of chromosome names and positions.
///
/// This is a convenience function for when gene annotations come as separate vectors
/// rather than a Vec<GenePosition>.
pub fn build_genome_order(chromosomes: &[Box<str>], positions: &[u64]) -> GenomeOrder {
    assert_eq!(chromosomes.len(), positions.len());
    let genes: Vec<GenePosition> = chromosomes
        .iter()
        .zip(positions.iter())
        .enumerate()
        .map(|(idx, (chr, &pos))| GenePosition {
            gene_idx: idx,
            chromosome: chr.clone(),
            position: pos,
        })
        .collect();
    GenomeOrder::from_positions(&genes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_order_basic() {
        let genes = vec![
            GenePosition {
                gene_idx: 0,
                chromosome: "chr2".into(),
                position: 500,
            },
            GenePosition {
                gene_idx: 1,
                chromosome: "chr1".into(),
                position: 1000,
            },
            GenePosition {
                gene_idx: 2,
                chromosome: "chr1".into(),
                position: 200,
            },
            GenePosition {
                gene_idx: 3,
                chromosome: "chrX".into(),
                position: 100,
            },
            GenePosition {
                gene_idx: 4,
                chromosome: "chr2".into(),
                position: 100,
            },
        ];

        let order = GenomeOrder::from_positions(&genes);
        // chr1: gene2(200), gene1(1000)
        // chr2: gene4(100), gene0(500)
        // chrX: gene3(100)
        assert_eq!(order.ordered_indices, vec![2, 1, 4, 0, 3]);
        assert_eq!(order.chr_boundaries.len(), 3);
        assert_eq!(order.chr_boundaries[0], ("chr1".into(), 0, 2));
        assert_eq!(order.chr_boundaries[1], ("chr2".into(), 2, 4));
        assert_eq!(order.chr_boundaries[2], ("chrX".into(), 4, 5));
    }

    #[test]
    fn test_excludes_noncanonical() {
        let genes = vec![
            GenePosition {
                gene_idx: 0,
                chromosome: "chr1".into(),
                position: 100,
            },
            GenePosition {
                gene_idx: 1,
                chromosome: "chrUn_gl000220".into(),
                position: 50,
            },
            GenePosition {
                gene_idx: 2,
                chromosome: "chr1_random".into(),
                position: 50,
            },
        ];

        let order = GenomeOrder::from_positions(&genes);
        assert_eq!(order.len(), 1);
        assert_eq!(order.ordered_indices, vec![0]);
    }

    #[test]
    fn test_reorder_rows() {
        let genes = vec![
            GenePosition {
                gene_idx: 0,
                chromosome: "chr2".into(),
                position: 100,
            },
            GenePosition {
                gene_idx: 1,
                chromosome: "chr1".into(),
                position: 200,
            },
        ];
        let order = GenomeOrder::from_positions(&genes);

        // 2x3 matrix: gene0=[1,2,3], gene1=[4,5,6]
        let mat = nalgebra::DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reordered = order.reorder_rows(&mat).unwrap();

        // After reorder: gene1 (chr1) first, then gene0 (chr2)
        assert_eq!(reordered[(0, 0)], 4.0);
        assert_eq!(reordered[(1, 0)], 1.0);
    }
}
