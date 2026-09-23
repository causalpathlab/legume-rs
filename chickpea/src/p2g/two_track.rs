//! The gene-centric embedding: RNA gene rows and peak-aggregated gene rows as
//! two tracks of one gene axis, fit by the shared two-phase engine.
//!
//! Peaks never enter training. They are aggregated onto genes once, through
//! the fixed cis weights ([`super::cis`], [`super::gene_track`]), and written
//! as `{work}.atac_gene.zarr`. The engine then sees each gene on two tracks:
//! the RNA row is the base, the peak-aggregated row is the base plus a
//! ridge-shrunk low-rank offset, and both are scored against the same
//! pseudobulk embeddings.

use super::cis::{build_cis_pairs, AbcKernel, CisPairs};
use super::gene_track::{write_gene_track, PeakToGenes};
use super::tracks::{gene_tracks, ATAC_TAG, RNA_TAG};
use crate::common::*;
use data_beans::sparse_io::open_sparse_matrix_by_path;
use genomic_data::coordinates::{parse_peak_coordinates, GeneTss};
use graph_embedding_util as ge;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::traits::ConvertMatOps;

/// Knobs of the two-track fit. Defaults follow `senna bge` / `senna gem`.
#[derive(Debug, Clone)]
pub struct TwoTrackConfig {
    pub embedding_dim: usize,
    pub epochs: usize,
    pub num_levels: usize,
    pub sort_dim: usize,
    pub proj_dim: usize,
    /// Hard gene partition size for the exact two-level softmax.
    pub feature_modules: usize,
    pub phase1_cells_per_pb: usize,
    /// Rank of the peak-aggregated track's gene offset (capped at the dimension).
    pub offset_rank: usize,
    /// Ridge pulling a gene's peak-aggregated row toward its RNA row.
    pub offset_l2: f32,
    pub seed: u64,
    /// Cells per block when writing the peak-aggregated track.
    pub block_size: usize,
}

impl Default for TwoTrackConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            epochs: 1000,
            num_levels: 3,
            sort_dim: 10,
            proj_dim: 50,
            feature_modules: 128,
            phase1_cells_per_pb: 16,
            offset_rank: ge::LoraSpec::default().rank,
            offset_l2: 1.0,
            seed: 42,
            block_size: 1000,
        }
    }
}

/// Inputs; `gene_positions` is aligned to the RNA file's rows.
pub struct TwoTrackInput<'a> {
    pub rna_file: &'a str,
    pub atac_file: &'a str,
    /// One batch label per barcode-aligned cell, if any.
    pub batch_file: Option<&'a str>,
    pub gene_positions: &'a [Option<GeneTss>],
    pub kernel: &'a AbcKernel,
    /// Prefix for intermediate files (`{work}.atac_gene.zarr`).
    pub work_prefix: &'a str,
}

/// The fitted tables, keyed by name.
pub struct TwoTrackEmbedding {
    /// Genes in RNA-row order.
    pub genes: Vec<Box<str>>,
    /// `[genes × H]`, the base (RNA) rows.
    pub rna_rows: nalgebra::DMatrix<f32>,
    /// `[genes with cis peaks × H]`, the peak-aggregated rows.
    pub atac_rows: nalgebra::DMatrix<f32>,
    /// Per gene, its row in `atac_rows` (`None`: no cis peaks).
    pub atac_gene: Vec<Option<usize>>,
    /// `[cells × H]`.
    pub cell_rows: nalgebra::DMatrix<f32>,
    pub barcodes: Vec<Box<str>>,
    /// `[finest pseudobulks × H]`, in the cells' frame.
    pub pb_rows: nalgebra::DMatrix<f32>,
    /// Finest pseudobulk of every cell, aligned with `barcodes`.
    pub cell_to_pb: Vec<usize>,
    /// The cis pairs over the RNA file's genes and the ATAC file's peaks.
    pub pairs: CisPairs,
    /// Gene names in RNA-file order (the order of `pairs`' genes).
    pub rna_genes: Vec<Box<str>>,
    /// Peak names in ATAC-file order (the ids in `pairs.peak`).
    pub peak_names: Vec<Box<str>>,
}

pub fn embed_two_track(
    inp: &TwoTrackInput,
    cfg: &TwoTrackConfig,
) -> anyhow::Result<TwoTrackEmbedding> {
    ////////////////////////////////////////
    // Peaks onto genes, once, then on disk //
    ////////////////////////////////////////
    let rna = open_sparse_matrix_by_path(inp.rna_file)?;
    let gene_names = rna.row_names()?;
    anyhow::ensure!(
        inp.gene_positions.len() == gene_names.len(),
        "{} gene positions for {} RNA rows",
        inp.gene_positions.len(),
        gene_names.len()
    );
    let atac = open_sparse_matrix_by_path(inp.atac_file)?;
    let peak_names = atac.row_names()?;
    let peaks = parse_peak_coordinates(&peak_names);
    let pairs = build_cis_pairs(inp.gene_positions, &peaks, inp.kernel);
    info!(
        "Cis pairs: {} genes placed of {}, {} pairs; {} of {} peaks reach no gene \
         ({} unparsed)",
        pairs.n_genes_placed,
        gene_names.len(),
        pairs.n_pairs(),
        pairs.n_unreached_peaks,
        peaks.len(),
        pairs.n_unparsed_peaks
    );
    let track_file = format!("{}.atac_gene.zarr", inp.work_prefix);
    let map = PeakToGenes::new(&pairs, peaks.len());
    let summary = write_gene_track(
        atac.as_ref(),
        &map,
        &pairs,
        &gene_names,
        &track_file,
        cfg.block_size,
    )?;
    info!(
        "Peak-aggregated track: {} genes × {} cells, {} nonzeros → {track_file}",
        summary.n_genes, summary.n_cells, summary.nnz
    );
    drop((rna, atac));

    /////////////////////////
    // Two tracks, one fit //
    /////////////////////////
    let mut unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: vec![inp.rna_file.into(), track_file.into()],
        batch_files: inp.batch_file.map(|b| vec![b.into()]),
        feature_kind: Some(ge::FeatureNameKind::Gene { delim: '_' }),
        column_alignment: ColumnAlignment::Union,
        per_file_feature_suffix: Some(vec![RNA_TAG.into(), ATAC_TAG.into()]),
        ..Default::default()
    })?;
    let tracks = gene_tracks(&unified.feature_names)?;
    let h = cfg.embedding_dim;
    let config = ge::FitConfig {
        embedding_dim: h,
        anchor_batches: None,
        bulk_batches: None,
        emit_finest_collapse: true,
        num_levels: cfg.num_levels,
        sort_dim: cfg.sort_dim,
        knn_pb_samples: 10,
        num_opt_iter: 30,
        proj_dim: cfg.proj_dim,
        epochs: cfg.epochs,
        batches_per_epoch: None,
        batch_size: 1024,
        learning_rate: 0.01,
        seed: cfg.seed,
        device: Device::Cpu,
        block_size: None,
        hvg_weights: None,
        refine: ge::RefineParams::default(),
        weight_decay: 0.0,
        phase1_cells_per_pb: cfg.phase1_cells_per_pb,
        hier_units_per_step: 256,
        hier_modules_per_unit: 8,
        module_only_min_rows: 0,
        feature_modules: ge::FeatureModuleArgs {
            feature_modules: Some(cfg.feature_modules),
        }
        .resolve(None)?,
        tracks: Some(tracks),
        offset_l2: cfg.offset_l2,
        offset_rank: cfg.offset_rank.min(h),
        preset_features: None,
        preset_offsets: Vec::new(),
        strata: None,
    };
    let out = ge::fit(&mut unified, config)?;

    ////////////////////////////////
    // Rows back to genes & cells //
    ////////////////////////////////
    let feat = nalgebra::DMatrix::<f32>::from_tensor(&out.model.e_feat)?;
    let spec = gene_tracks(&unified.feature_names)?;
    anyhow::ensure!(
        feat.nrows() == spec.track_of_row.len(),
        "{} feature rows for {} named rows",
        feat.nrows(),
        spec.track_of_row.len()
    );
    let n_genes = spec.n_genes();
    let mut rna_row_of_gene = vec![usize::MAX; n_genes];
    let mut atac_row_of_gene = vec![usize::MAX; n_genes];
    for (r, (&t, &g)) in spec.track_of_row.iter().zip(&spec.gene_of_row).enumerate() {
        if t == 0 {
            rna_row_of_gene[g as usize] = r;
        } else {
            atac_row_of_gene[g as usize] = r;
        }
    }
    let base = |r: usize| {
        unified.feature_names[r]
            .rsplit_once('/')
            .map_or("", |(n, _)| n)
    };
    let genes: Vec<Box<str>> = rna_row_of_gene.iter().map(|&r| base(r).into()).collect();
    let rna_rows = feat.select_rows(rna_row_of_gene.iter());
    let with_atac: Vec<usize> = (0..n_genes)
        .filter(|&g| atac_row_of_gene[g] != usize::MAX)
        .collect();
    let atac_rows = feat.select_rows(with_atac.iter().map(|&g| &atac_row_of_gene[g]));
    let mut atac_gene = vec![None; n_genes];
    for (k, &g) in with_atac.iter().enumerate() {
        atac_gene[g] = Some(k);
    }
    let cell_rows = nalgebra::DMatrix::<f32>::from_tensor(&out.model.e_cell)?;
    let pb_rows = out
        .pb_embeddings
        .last()
        .ok_or_else(|| anyhow::anyhow!("the fit returned no pseudobulk embeddings"))?
        .e_pb
        .clone();
    let (_, cell_to_pb) = out
        .finest_collapse
        .ok_or_else(|| anyhow::anyhow!("the fit returned no finest collapse"))?;
    anyhow::ensure!(
        cell_to_pb.len() == cell_rows.nrows(),
        "{} pseudobulk memberships for {} cells",
        cell_to_pb.len(),
        cell_rows.nrows()
    );

    Ok(TwoTrackEmbedding {
        genes,
        rna_rows,
        atac_rows,
        atac_gene,
        cell_rows,
        barcodes: unified.barcodes.clone(),
        pb_rows,
        cell_to_pb,
        pairs,
        rna_genes: gene_names,
        peak_names,
    })
}
