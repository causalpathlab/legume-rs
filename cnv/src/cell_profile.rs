//! Per-cell inferCNV profiles, streamed out of `data-beans` backends.
//!
//! Recipe (inferCNV, Broad):
//!
//! 1. **Reference pass** over the normal cells: per-gene mean raw count
//!    (for the expression filter) and per-gene mean of
//!    `ln(1 + scale · x / depth)` — the diploid baseline.
//! 2. **Gene axis**: rows with a GFF locus on a canonical chromosome and
//!    mean reference count ≥ `min_mean_expr`, put in genome order
//!    ([`GenomeOrder`]).
//! 3. **Query pass**, one block of cells at a time: log-normalise, subtract
//!    the reference mean, clip at `±clip`, chromosome-bounded window
//!    average ([`crate::infercnv::smooth_ordered`], `O(G)` per cell), then
//!    median-centre each cell. Optionally average genes into fixed genomic
//!    tiles (`bin_size` bp, [`genomic_data::coordinates::tile_windows`]
//!    grammar).
//! 4. Rows go out as a fresh sparse backend whose row names are genomic
//!    intervals `chr:start-end` and whose columns are the query cells,
//!    written as CSC slabs so no `[G × N]` matrix is ever resident.
//!
//! Reference cells never see the query: with explicit normals the zero of
//! the log-ratio is diploid. Without normals the cohort mean is the baseline
//! and any CNV shared by every cell is invisible — a warning says so.

use crate::gene_loci::GeneLocus;
use crate::genome_order::{GenePosition, GenomeOrder};
use crate::infercnv::{smooth_ordered, InferCnvConfig};

use data_beans::sparse_io::{create_sparse_streaming_empty, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans::zarr_io::{finalize_output, prepare_output};
use genomic_data::coordinates::PeakCoord;
use matrix_util::common_io::write_lines;
use matrix_util::dmatrix_util::build_columns_par;
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;

/// Knobs for [`run_cell_profiles`].
#[derive(Debug, Clone)]
pub struct CellProfileConfig {
    /// Smoothing window in genes (inferCNV default 101). `≤ 1` disables.
    pub window: usize,
    /// Clip the per-gene log-ratio to `±clip` before smoothing. `≤ 0` disables.
    pub clip: f32,
    /// Depth-normalisation target: `ln(1 + scale · x / depth)`.
    pub scale: f32,
    /// Drop genes whose mean raw count across reference cells is below this.
    pub min_mean_expr: f32,
    /// Average smoothed genes into `bin_size`-bp tiles keyed by TSS. `0` keeps
    /// one row per gene (classic inferCNV). Prefer `1_000_000` (1 Mb) when
    /// gene-level rows are too many for large cohorts.
    pub bin_size: i64,
    /// Cells per streamed block.
    pub block_size: usize,
    /// Median-centre each cell after smoothing.
    pub center: bool,
    /// Chromosomes dropped from the gene axis (compared without the `chr`
    /// prefix). inferCNV's default is `X, Y, M`: sex and mitochondrial
    /// content shift whole chromosomes for reasons that are not copy number.
    pub exclude_chr: Vec<Box<str>>,
}

impl Default for CellProfileConfig {
    fn default() -> Self {
        Self {
            window: 101,
            clip: 3.0,
            scale: 1e4,
            min_mean_expr: 0.1,
            bin_size: 0,
            block_size: 1000,
            center: true,
            exclude_chr: ["chrX", "chrY", "chrM"].map(Into::into).to_vec(),
        }
    }
}

impl CellProfileConfig {
    fn is_excluded_chr(&self, chr: &str) -> bool {
        use genomic_data::coordinates::chr_eq;
        self.exclude_chr.iter().any(|x| chr_eq(x, chr))
    }
}

/// Per-gene sums over the reference cells (length = `data.num_rows()`).
pub struct ReferenceStats {
    pub n_cells: usize,
    pub raw_sum: Vec<f32>,
    pub log_sum: Vec<f32>,
}

impl ReferenceStats {
    pub fn raw_mean(&self, g: usize) -> f32 {
        self.raw_sum[g] / (self.n_cells.max(1) as f32)
    }
    pub fn log_mean(&self, g: usize) -> f32 {
        self.log_sum[g] / (self.n_cells.max(1) as f32)
    }
}

/// `ln(1 + scale · x / depth)`; `0` when the cell has no counts.
#[inline]
fn log_norm(x: f32, depth: f32, scale: f32) -> f32 {
    if depth > 0.0 {
        (x / depth * scale).ln_1p()
    } else {
        0.0
    }
}

fn column_depths(csc: &CscMatrix<f32>) -> Vec<f32> {
    (0..csc.ncols())
        .into_par_iter()
        .map(|j| csc.col(j).values().iter().sum::<f32>())
        .collect()
}

/// One pass over `ref_cols`: per-gene raw-count sum and log-normalised sum.
pub fn reference_stats(
    data: &SparseIoVec,
    ref_cols: &[usize],
    cfg: &CellProfileConfig,
) -> anyhow::Result<ReferenceStats> {
    let g = data.num_rows();
    let mut raw_sum = vec![0f32; g];
    let mut log_sum = vec![0f32; g];

    let blocks = matrix_util::utils::generate_minibatch_intervals(
        ref_cols.len(),
        0,
        Some(cfg.block_size.max(1)),
    );
    for (lb, ub) in blocks {
        let csc = data.read_columns_csc(ref_cols[lb..ub].iter().copied())?;
        let depths = column_depths(&csc);
        let (r, l) = (0..csc.ncols())
            .into_par_iter()
            .fold(
                || (vec![0f32; g], vec![0f32; g]),
                |(mut r, mut l), j| {
                    let col = csc.col(j);
                    let d = depths[j];
                    for (&i, &x) in col.row_indices().iter().zip(col.values()) {
                        r[i] += x;
                        l[i] += log_norm(x, d, cfg.scale);
                    }
                    (r, l)
                },
            )
            .reduce(
                || (vec![0f32; g], vec![0f32; g]),
                |(mut ra, mut la), (rb, lb_)| {
                    ra.iter_mut().zip(&rb).for_each(|(a, b)| *a += b);
                    la.iter_mut().zip(&lb_).for_each(|(a, b)| *a += b);
                    (ra, la)
                },
            );
        raw_sum.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
        log_sum.iter_mut().zip(&l).for_each(|(a, b)| *a += b);
    }
    Ok(ReferenceStats {
        n_cells: ref_cols.len(),
        raw_sum,
        log_sum,
    })
}

/// The genome-ordered gene axis and its mapping onto output rows.
pub struct GenomeFeatures {
    /// Genome order over the kept genes (`ordered_indices` index `kept_rows`).
    pub order: GenomeOrder,
    /// Data row index of each kept gene.
    pub kept_rows: Vec<usize>,
    /// Locus of each kept gene.
    pub kept_loci: Vec<GeneLocus>,
    /// Reference log-mean per **ordered** position.
    pub ref_log_mean: Vec<f32>,
    /// Output row names (`chr:start-end`).
    pub row_names: Vec<Box<str>>,
    /// Output row of each ordered position.
    pub row_of_ordered: Vec<usize>,
    /// Genes per output row.
    pub genes_per_row: Vec<usize>,
    /// Ordered position of each **data** row (`usize::MAX` = not used).
    /// Built once here so every streamed block skips the rebuild.
    pub ord_of_data_row: Vec<usize>,
}

impl GenomeFeatures {
    pub fn n_ordered(&self) -> usize {
        self.order.len()
    }
    pub fn n_rows(&self) -> usize {
        self.row_names.len()
    }

    /// Select, order, and (optionally) bin the gene axis.
    pub fn build(
        loci: &[Option<GeneLocus>],
        stats: &ReferenceStats,
        cfg: &CellProfileConfig,
    ) -> anyhow::Result<Self> {
        let mut kept_rows = Vec::new();
        let mut kept_loci = Vec::new();
        let mut n_no_locus = 0usize;
        let mut n_low_expr = 0usize;
        let mut n_excl_chr = 0usize;
        for (g, locus) in loci.iter().enumerate() {
            let Some(l) = locus else {
                n_no_locus += 1;
                continue;
            };
            if cfg.is_excluded_chr(&l.chromosome) {
                n_excl_chr += 1;
                continue;
            }
            if stats.raw_mean(g) < cfg.min_mean_expr {
                n_low_expr += 1;
                continue;
            }
            kept_rows.push(g);
            kept_loci.push(l.clone());
        }
        log::info!(
            "gene filter: {} kept, {} without GFF locus, {} on excluded chromosomes {:?}, {} below mean count {}",
            kept_rows.len(),
            n_no_locus,
            n_excl_chr,
            cfg.exclude_chr,
            n_low_expr,
            cfg.min_mean_expr
        );

        let positions: Vec<GenePosition> = kept_loci
            .iter()
            .enumerate()
            .map(|(k, l)| GenePosition {
                gene_idx: k,
                chromosome: l.chromosome.clone(),
                position: l.tss.max(0) as u64,
            })
            .collect();
        let order = GenomeOrder::from_positions(&positions);
        anyhow::ensure!(
            !order.is_empty(),
            "no kept gene maps to a canonical chromosome (chr1..22, X, Y, M)"
        );
        let n_dropped = kept_rows.len() - order.len();
        if n_dropped > 0 {
            log::info!("{} genes on non-canonical contigs dropped", n_dropped);
        }

        let ref_log_mean: Vec<f32> = order
            .ordered_indices
            .iter()
            .map(|&k| stats.log_mean(kept_rows[k]))
            .collect();

        // Output rows: one per gene, or one per `bin_size` tile of the TSS.
        let mut row_names: Vec<Box<str>> = Vec::new();
        let mut row_of_ordered = Vec::with_capacity(order.len());
        let mut genes_per_row: Vec<usize> = Vec::new();
        let mut last_tile: Option<(Box<str>, i64)> = None;
        for &k in &order.ordered_indices {
            let l = &kept_loci[k];
            if cfg.bin_size > 0 {
                let tile_start = (l.tss - 1).max(0).div_euclid(cfg.bin_size) * cfg.bin_size;
                let same = last_tile
                    .as_ref()
                    .is_some_and(|(c, s)| c.as_ref() == l.chromosome.as_ref() && *s == tile_start);
                if !same {
                    row_names.push(
                        PeakCoord {
                            chr: l.chromosome.clone(),
                            start: tile_start,
                            end: tile_start + cfg.bin_size,
                        }
                        .to_string()
                        .into(),
                    );
                    genes_per_row.push(0);
                    last_tile = Some((l.chromosome.clone(), tile_start));
                }
            } else {
                row_names.push(l.interval_name());
                genes_per_row.push(0);
            }
            let r = row_names.len() - 1;
            row_of_ordered.push(r);
            genes_per_row[r] += 1;
        }
        log::info!(
            "output axis: {} rows over {} ordered genes, {} chromosomes",
            row_names.len(),
            order.len(),
            order.chr_boundaries.len()
        );

        let mut ord_of_data_row = vec![usize::MAX; loci.len()];
        for (o, &k) in order.ordered_indices.iter().enumerate() {
            ord_of_data_row[kept_rows[k]] = o;
        }

        Ok(Self {
            order,
            kept_rows,
            kept_loci,
            ref_log_mean,
            row_names,
            row_of_ordered,
            genes_per_row,
            ord_of_data_row,
        })
    }

    /// Lines for `{out}.features.tsv.gz`.
    pub fn feature_table(&self) -> Vec<Box<str>> {
        let mut genes_by_row: Vec<Vec<&str>> = vec![Vec::new(); self.n_rows()];
        let mut keys: Vec<Box<str>> = Vec::with_capacity(self.n_ordered());
        for &k in &self.order.ordered_indices {
            keys.push(self.kept_loci[k].gene_key());
        }
        for (o, &r) in self.row_of_ordered.iter().enumerate() {
            genes_by_row[r].push(&keys[o]);
        }
        let mut lines: Vec<Box<str>> = vec!["feature\tchr\tstart\tend\tn_genes\tgenes".into()];
        for (r, name) in self.row_names.iter().enumerate() {
            let coord =
                genomic_data::coordinates::parse_peak_coordinates(std::slice::from_ref(name))
                    .pop()
                    .flatten();
            let (chr, s, e) = coord
                .map(|c| (c.chr.to_string(), c.start, c.end))
                .unwrap_or_else(|| (String::new(), 0, 0));
            lines.push(
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    name,
                    chr,
                    s,
                    e,
                    self.genes_per_row[r],
                    genes_by_row[r].join(",")
                )
                .into(),
            );
        }
        lines
    }
}

/// Smoothed, centred CNV profile of one block of cells:
/// `[n_rows × n_cells]` plus each cell's depth.
pub fn profile_block(
    csc: &CscMatrix<f32>,
    feats: &GenomeFeatures,
    cfg: &CellProfileConfig,
) -> (DMatrix<f32>, Vec<f32>) {
    let n = csc.ncols();
    let g_ord = feats.n_ordered();
    let depths = column_depths(csc);
    let ord_of_row = &feats.ord_of_data_row;
    debug_assert_eq!(ord_of_row.len(), csc.nrows());

    let clip = cfg.clip;
    let log_ratio = build_columns_par(g_ord, n, |j, col| {
        // Absent gene ⇒ log-normalised 0 ⇒ ratio = −ref_mean.
        col.iter_mut()
            .zip(&feats.ref_log_mean)
            .for_each(|(v, m)| *v = -m);
        let c = csc.col(j);
        let d = depths[j];
        for (&i, &x) in c.row_indices().iter().zip(c.values()) {
            let o = ord_of_row[i];
            if o != usize::MAX {
                col[o] += log_norm(x, d, cfg.scale);
            }
        }
        if clip > 0.0 {
            col.iter_mut().for_each(|v| *v = v.clamp(-clip, clip));
        }
    });

    let smooth_cfg = InferCnvConfig {
        window: cfg.window,
        clip: None,
        center_cells: false,
    };
    let mut smoothed = smooth_ordered(&log_ratio, &feats.order, &smooth_cfg);

    if cfg.center && g_ord > 0 {
        smoothed
            .as_mut_slice()
            .par_chunks_mut(g_ord)
            .for_each(|col| {
                let mut buf: Vec<f32> = col.iter().copied().filter(|v| v.is_finite()).collect();
                if buf.is_empty() {
                    return;
                }
                let m = buf.len() / 2;
                let med = *buf
                    .select_nth_unstable_by(m, |a, b| a.total_cmp(b))
                    .1;
                for v in col.iter_mut() {
                    *v -= med;
                }
            });
    }

    let n_rows = feats.n_rows();
    if n_rows == g_ord {
        return (smoothed, depths);
    }
    let inv_genes_per_row: Vec<f32> = feats
        .genes_per_row
        .iter()
        .map(|&k| 1.0 / (k.max(1) as f32))
        .collect();
    let binned = build_columns_par(n_rows, n, |j, col| {
        col.fill(0.0);
        let src = smoothed.column(j);
        for (&r, &v) in feats.row_of_ordered.iter().zip(src.iter()) {
            col[r] += v;
        }
        col.iter_mut()
            .zip(&inv_genes_per_row)
            .for_each(|(v, w)| *v *= w);
    });
    (binned, depths)
}

/// Where the outputs went.
pub struct CellProfileOutputs {
    pub backend: Box<str>,
    pub features: Box<str>,
    pub cells: Box<str>,
    pub n_rows: usize,
    pub n_cells: usize,
}

/// Stream per-cell CNV profiles for `query_cols` into `{out}.zarr.zip`
/// (rows = genomic intervals, columns = query cells), with
/// `{out}.features.tsv.gz` and `{out}.cells.tsv.gz` alongside.
pub fn run_cell_profiles(
    data: &SparseIoVec,
    ref_cols: &[usize],
    query_cols: &[usize],
    loci: &[Option<GeneLocus>],
    cfg: &CellProfileConfig,
    out_prefix: &str,
) -> anyhow::Result<CellProfileOutputs> {
    anyhow::ensure!(!query_cols.is_empty(), "no query cells");
    anyhow::ensure!(!ref_cols.is_empty(), "no reference cells");

    log::info!(
        "reference pass: {} cells, {} genes",
        ref_cols.len(),
        data.num_rows()
    );
    let stats = reference_stats(data, ref_cols, cfg)?;
    let feats = GenomeFeatures::build(loci, &stats, cfg)?;

    let cell_names = data.column_names()?;
    let n_rows = feats.n_rows();
    let n_cells = query_cols.len();
    let nnz = n_rows * n_cells;

    let (effective_out, backend, working_file) =
        prepare_output(out_prefix, SparseIoBackend::Zarr, true)?;
    let mut cells_tsv: Vec<Box<str>> = vec!["cell\tdepth\tcnv_burden".into()];

    {
        let mut out = create_sparse_streaming_empty(Some(&working_file), Some(&backend))?;
        out.begin_streaming_csc((n_rows, n_cells, nnz))?;

        let blocks = matrix_util::utils::generate_minibatch_intervals(
            n_cells,
            0,
            Some(cfg.block_size.max(1)),
        );
        let n_blocks = blocks.len();
        let max_block = blocks.iter().map(|(lb, ub)| ub - lb).max().unwrap_or(0);
        // Dense output: every block tiles the same row indices and the same
        // column pointers. Build once at the largest block size; slice per
        // block (only the last one is shorter).
        let rows_tiled: Vec<u64> = (0..(n_rows * max_block) as u64)
            .map(|k| k % n_rows as u64)
            .collect();
        let colptr_full: Vec<u64> = (0..max_block as u64).map(|j| j * n_rows as u64).collect();
        let inv_n_rows = 1.0 / (n_rows.max(1) as f32);

        let mut nnz_offset = 0u64;
        for (b, (lb, ub)) in blocks.into_iter().enumerate() {
            let cols = &query_cols[lb..ub];
            let csc = data.read_columns_csc(cols.iter().copied())?;
            let (block, depths) = profile_block(&csc, &feats, cfg);
            let n_block = ub - lb;

            let burden: Vec<f32> = block
                .as_slice()
                .par_chunks(n_rows.max(1))
                .map(|col| col.iter().map(|v| v.abs()).sum::<f32>() * inv_n_rows)
                .collect();
            cells_tsv.extend(cols.iter().enumerate().map(|(j, &c)| {
                format!("{}\t{}\t{:.6}", cell_names[c], depths[j], burden[j]).into_boxed_str()
            }));

            // nalgebra is column-major: the slice is already CSC value order.
            out.append_csc_slab(
                lb as u64,
                nnz_offset,
                &colptr_full[..n_block],
                &rows_tiled[..n_rows * n_block],
                block.as_slice(),
            )?;
            nnz_offset += (n_rows * n_block) as u64;
            log::info!("block {}/{}: cells {}..{}", b + 1, n_blocks, lb, ub);
        }

        out.finalize_streaming_csc()?;
        out.build_csr_from_csc_streaming()?;
        out.register_row_names_vec(&feats.row_names);
        let query_names: Vec<Box<str>> = query_cols.iter().map(|&c| cell_names[c].clone()).collect();
        out.register_column_names_vec(&query_names);
    }
    let backend_path = finalize_output(&working_file, &effective_out)?.to_string();

    let features_path = format!("{}.features.tsv.gz", out_prefix);
    write_lines(&feats.feature_table(), &features_path)?;
    let cells_path = format!("{}.cells.tsv.gz", out_prefix);
    write_lines(&cells_tsv, &cells_path)?;

    log::info!(
        "wrote {} ({} intervals × {} cells), {}, {}",
        backend_path,
        n_rows,
        n_cells,
        features_path,
        cells_path
    );
    Ok(CellProfileOutputs {
        backend: backend_path.into(),
        features: features_path.into(),
        cells: cells_path.into(),
        n_rows,
        n_cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn locus(chr: &str, tss: i64, id: &str) -> GeneLocus {
        GeneLocus {
            chromosome: chr.into(),
            start: tss,
            stop: tss + 1000,
            tss,
            gene_id: id.into(),
            symbol: id.into(),
        }
    }

    fn stats(g: usize, n: usize) -> ReferenceStats {
        ReferenceStats {
            n_cells: n,
            raw_sum: vec![n as f32; g],
            log_sum: vec![0.0; g],
        }
    }

    #[test]
    fn features_bin_by_tss_tile() {
        // 4 genes: chr1 at 100, 1500 (same 1kb tile? no: 0-999 vs 1000-1999),
        // chr1 at 1200 (tile 1000-1999 with 1500), chr2 at 50.
        let loci = vec![
            Some(locus("chr1", 1500, "b")),
            Some(locus("chr1", 100, "a")),
            None,
            Some(locus("chr1", 1200, "c")),
            Some(locus("chr2", 50, "d")),
        ];
        let cfg = CellProfileConfig {
            bin_size: 1000,
            min_mean_expr: 0.0,
            ..Default::default()
        };
        let f = GenomeFeatures::build(&loci, &stats(5, 3), &cfg).unwrap();
        assert_eq!(f.n_ordered(), 4);
        assert_eq!(
            f.row_names.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
            vec!["chr1:0-1000", "chr1:1000-2000", "chr2:0-1000"]
        );
        assert_eq!(f.genes_per_row, vec![1, 2, 1]);
        let table = f.feature_table();
        assert!(table[2].ends_with("\t2\tc_c,b_b"), "{}", table[2]);
    }

    #[test]
    fn gene_rows_use_bed_interval_and_expression_filter() {
        let loci = vec![Some(locus("chr3", 10, "x")), Some(locus("chr3", 5000, "y"))];
        let mut st = stats(2, 10);
        st.raw_sum[1] = 0.5; // mean 0.05 < 0.1 ⇒ dropped
        let f = GenomeFeatures::build(&loci, &st, &CellProfileConfig::default()).unwrap();
        assert_eq!(f.n_rows(), 1);
        assert_eq!(f.row_names[0].as_ref(), "chr3:9-1010");
    }

    #[test]
    fn profile_block_flat_cell_is_zero_after_centering() {
        // 3 genes on chr1; reference log-mean = log-normalised value of the
        // same counts ⇒ ratio 0 everywhere ⇒ profile 0.
        let loci: Vec<Option<GeneLocus>> = (0..3)
            .map(|i| Some(locus("chr1", 100 * (i as i64 + 1), &format!("g{i}"))))
            .collect();
        let cfg = CellProfileConfig {
            window: 3,
            min_mean_expr: 0.0,
            block_size: 10,
            ..Default::default()
        };
        let counts = [2.0f32, 4.0, 6.0];
        let depth: f32 = counts.iter().sum();
        let st = ReferenceStats {
            n_cells: 1,
            raw_sum: counts.to_vec(),
            log_sum: counts.iter().map(|&x| log_norm(x, depth, cfg.scale)).collect(),
        };
        let f = GenomeFeatures::build(&loci, &st, &cfg).unwrap();
        let dense = DMatrix::from_column_slice(3, 1, &counts);
        let csc = CscMatrix::from(&dense);
        let (prof, depths) = profile_block(&csc, &f, &cfg);
        assert_eq!(depths, vec![depth]);
        for v in prof.iter() {
            assert!(v.abs() < 1e-6, "{v}");
        }
    }
}
