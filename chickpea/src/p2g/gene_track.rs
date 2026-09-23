//! Peak counts aggregated onto genes: the gene axis's second track.
//!
//! A cell's count for gene `g` on this track is `a_gc = Σ_p w_gp x_pc` over
//! `g`'s cis peaks, with `w` the fixed weights of [`super::cis::CisPairs`].
//! Aggregation runs cell by cell over the cell's nonzero peaks through the
//! peak → genes inverse, so its cost is one pass over the ATAC counts.

use super::cis::CisPairs;
use crate::common::*;
use data_beans::sparse_io::create_sparse_streaming_empty;
use rayon::prelude::*;

/// The pairs inverted to CSR over peaks: every gene each peak feeds, ascending.
#[derive(Debug, Clone)]
pub struct PeakToGenes {
    ptr: Vec<usize>,
    gene: Vec<u32>,
    weight: Vec<f32>,
}

impl PeakToGenes {
    #[must_use]
    pub fn new(pairs: &CisPairs, n_peaks: usize) -> Self {
        let mut ptr = vec![0usize; n_peaks + 1];
        for &p in &pairs.peak {
            ptr[p as usize + 1] += 1;
        }
        for i in 0..n_peaks {
            ptr[i + 1] += ptr[i];
        }
        let mut next = ptr.clone();
        let mut gene = vec![0u32; pairs.n_pairs()];
        let mut weight = vec![0f32; pairs.n_pairs()];
        // Genes are visited in ascending order, so each peak's list is sorted.
        for g in 0..pairs.n_genes() {
            for k in pairs.gene(g) {
                let slot = &mut next[pairs.peak[k] as usize];
                gene[*slot] = g as u32;
                weight[*slot] = pairs.weight[k];
                *slot += 1;
            }
        }
        Self { ptr, gene, weight }
    }

    /// Genes fed by peak `p`, ascending.
    #[must_use]
    pub fn genes_of(&self, p: usize) -> &[u32] {
        &self.gene[self.ptr[p]..self.ptr[p + 1]]
    }

    /// One cell's gene counts from its `(peak, count)` entries: ascending gene
    /// order, zeros dropped.
    #[must_use]
    pub fn aggregate(&self, cell: &[(u32, f32)]) -> Vec<(u32, f32)> {
        let mut acc: Vec<(u32, f32)> = Vec::new();
        for &(p, x) in cell {
            if x == 0.0 {
                continue;
            }
            let r = self.ptr[p as usize]..self.ptr[p as usize + 1];
            acc.extend(
                self.gene[r.clone()]
                    .iter()
                    .zip(&self.weight[r])
                    .map(|(&g, &w)| (g, w * x)),
            );
        }
        acc.sort_unstable_by_key(|&(g, _)| g);
        let mut out: Vec<(u32, f32)> = Vec::with_capacity(acc.len());
        for (g, v) in acc {
            match out.last_mut() {
                Some(last) if last.0 == g => last.1 += v,
                _ => out.push((g, v)),
            }
        }
        out
    }
}

/// Shape of a written gene track.
#[derive(Debug, Clone, Copy)]
pub struct GeneTrackSummary {
    pub n_genes: usize,
    pub n_cells: usize,
    pub nnz: usize,
}

/// Aggregate every cell of `atac` onto genes and write the result as a sparse
/// genes × cells matrix at `out_file`. Rows are the genes with at least one cis
/// peak, in gene order, named from `gene_names`; columns are `atac`'s cells,
/// unchanged. Streams `block` cells at a time in two passes (count, then write),
/// so memory stays one block of cells.
pub fn write_gene_track(
    atac: &SparseData,
    map: &PeakToGenes,
    pairs: &CisPairs,
    gene_names: &[Box<str>],
    out_file: &str,
    block: usize,
) -> anyhow::Result<GeneTrackSummary> {
    anyhow::ensure!(block > 0, "write_gene_track: block must be positive");
    anyhow::ensure!(
        gene_names.len() == pairs.n_genes(),
        "{} gene names for {} genes",
        gene_names.len(),
        pairs.n_genes()
    );
    let n_cells = atac
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("the ATAC matrix has no column count"))?;

    // Genes with cis peaks, compacted to track rows in gene order.
    let mut row_of = vec![u32::MAX; pairs.n_genes()];
    let mut row_names: Vec<Box<str>> = Vec::new();
    for g in 0..pairs.n_genes() {
        if !pairs.gene(g).is_empty() {
            row_of[g] = row_names.len() as u32;
            row_names.push(gene_names[g].clone());
        }
    }

    // One block's aggregated columns, rows compacted (still ascending).
    let aggregate_block = |cols: std::ops::Range<usize>| -> anyhow::Result<Vec<Vec<(u32, f32)>>> {
        let csc = atac.read_columns_csc(cols.collect())?;
        Ok(csc
            .col_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|col| {
                let cell: Vec<(u32, f32)> = col
                    .row_indices()
                    .iter()
                    .zip(col.values())
                    .map(|(&p, &x)| (p as u32, x))
                    .collect();
                map.aggregate(&cell)
                    .into_iter()
                    .map(|(g, v)| (row_of[g as usize], v))
                    .collect()
            })
            .collect())
    };
    let blocks: Vec<std::ops::Range<usize>> = (0..n_cells)
        .step_by(block)
        .map(|lb| lb..(lb + block).min(n_cells))
        .collect();

    let mut nnz = 0usize;
    for cols in &blocks {
        nnz += aggregate_block(cols.clone())?
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    }

    let mut out = create_sparse_streaming_empty(Some(out_file), Some(&atac.backend_type()))?;
    out.begin_streaming_csc((row_names.len(), n_cells, nnz))?;
    let mut nnz_offset = 0u64;
    for cols in &blocks {
        let agg = aggregate_block(cols.clone())?;
        let mut colptr = Vec::with_capacity(agg.len());
        let (mut rows, mut vals) = (Vec::new(), Vec::new());
        for col in &agg {
            colptr.push(rows.len() as u64);
            for &(r, v) in col {
                rows.push(u64::from(r));
                vals.push(v);
            }
        }
        out.append_csc_slab(cols.start as u64, nnz_offset, &colptr, &rows, &vals)?;
        nnz_offset += vals.len() as u64;
    }
    out.finalize_streaming_csc()?;
    out.build_csr_from_csc_streaming()?;
    out.register_row_names_vec(&row_names);
    out.register_column_names_vec(&atac.column_names()?);

    Ok(GeneTrackSummary {
        n_genes: row_names.len(),
        n_cells,
        nnz,
    })
}
