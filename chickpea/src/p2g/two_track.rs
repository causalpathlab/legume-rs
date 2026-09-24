//! Gene + peak multiome embedding (bge multiome recipe).
//!
//! Loads RNA genes and ATAC peaks onto one feature axis with modality suffixes.
//! Wide ATAC is `module_only` (peak row = its module row `μ_m`). Phase 2 is a
//! single encoder over that axis. The cis pairs go into phase 1 as gates on
//! the RNA gene scores; their trained weights are the links.

use super::cis::{build_cis_pairs, AbcKernel, CisPairs};
use crate::common::*;
use data_beans::sparse_io::open_sparse_matrix_by_path;
use genomic_data::coordinates::{parse_peak_coordinates, GeneTss};
use graph_embedding_util as ge;
use legume_numeric::matrix::traits::ConvertMatOps;
use rustc_hash::FxHashMap;

/// Suffix of the RNA rows on the joint axis.
pub const RNA_TAG: &str = "rna";
/// Suffix of the ATAC peak rows on the joint axis.
pub const ATAC_TAG: &str = "atac";

/// ATAC modules under this many peaks hold scattered peaks: they join the
/// near-empty background, which is never a link candidate.
const ATAC_MIN_MODULE_SIZE: usize = 10;
/// RNA modules under this many genes (singletons) hold outlier genes: they
/// join the flat background and go module-only.
const RNA_MIN_MODULE_SIZE: usize = 2;
/// ATAC modules never cross a genomic window of this many bp on one
/// chromosome, so no module sits in every gene's cis window.
const ATAC_MODULE_WINDOW: i64 = 10_000_000;

/// Knobs of the multiome fit. Defaults follow `senna bge --multiome`.
#[derive(Debug, Clone)]
pub struct TwoTrackConfig {
    pub embedding_dim: usize,
    pub epochs: usize,
    pub num_levels: usize,
    pub sort_dim: usize,
    pub proj_dim: usize,
    /// Modules per modality (RNA and ATAC each get this many).
    pub feature_modules: usize,
    pub phase1_cells_per_pb: usize,
    /// On the ATAC modality, module-only when it has at least this many peaks.
    /// `0` disables. Bge default is 100_000; tests use a small threshold.
    pub module_only_min_rows: usize,
    pub seed: u64,
    pub device: crate::common::ComputeDevice,
    pub device_no: usize,
}

impl Default for TwoTrackConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            epochs: 1000,
            num_levels: 3,
            sort_dim: 10,
            proj_dim: 50,
            feature_modules: 1024,
            phase1_cells_per_pb: 16,
            module_only_min_rows: 100_000,
            seed: 42,
            device: crate::common::ComputeDevice::Cpu,
            device_no: 0,
        }
    }
}

/// Inputs; `gene_positions` is aligned to the RNA file's rows.
pub struct TwoTrackInput<'a> {
    pub rna_file: &'a str,
    pub atac_file: &'a str,
    pub batch_file: Option<&'a str>,
    pub gene_positions: &'a [Option<GeneTss>],
    pub kernel: &'a AbcKernel,
    pub work_prefix: &'a str,
}

/// Fitted tables.
pub struct TwoTrackEmbedding {
    /// Genes in RNA-file order.
    pub genes: Vec<Box<str>>,
    /// `[genes × H]`, RNA-track composed rows.
    pub rna_rows: nalgebra::DMatrix<f32>,
    /// `[peaks × H]`, each peak's module embedding `μ_{m(p)}`.
    pub peak_rows: nalgebra::DMatrix<f32>,
    /// Module id of each peak (ATAC-file order).
    pub module_of_peak: Vec<u32>,
    /// `[cells × H]`.
    pub cell_rows: nalgebra::DMatrix<f32>,
    pub barcodes: Vec<Box<str>>,
    pub pairs: CisPairs,
    pub rna_genes: Vec<Box<str>>,
    pub peak_names: Vec<Box<str>>,
    /// Phase-1 cis gate weights `w` in [`CisPairs`] order (`0` if a pair was
    /// dropped from the engine table).
    pub gate_w: Vec<f32>,
    pub gate_theta0: f32,
    pub gate_theta1: f32,
    pub gate_theta3: f32,
    pub gate_gamma1: f32,
    pub gate_gamma2: f32,
}

pub fn embed_two_track(
    input: &TwoTrackInput,
    cfg: &TwoTrackConfig,
) -> anyhow::Result<TwoTrackEmbedding> {
    let rna = open_sparse_matrix_by_path(input.rna_file)?;
    let gene_names = rna.row_names()?;
    anyhow::ensure!(
        input.gene_positions.len() == gene_names.len(),
        "{} gene positions for {} RNA rows",
        input.gene_positions.len(),
        gene_names.len()
    );
    let atac = open_sparse_matrix_by_path(input.atac_file)?;
    let peak_names = atac.row_names()?;
    let peaks = parse_peak_coordinates(&peak_names);
    let pairs = build_cis_pairs(input.gene_positions, &peaks, input.kernel);
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
    drop((rna, atac));

    // One axis: RNA genes ∪ ATAC peaks (bge multiome).
    let mut unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: vec![input.rna_file.into(), input.atac_file.into()],
        batch_files: input.batch_file.map(|b| vec![b.into()]),
        // Exact: RNA symbols and peak loci are disjoint names. Mixed would
        // rewrite `GENE_1` → `1` via the gene-symbol rule and break lookup.
        feature_kind: Some(ge::FeatureNameKind::Exact),
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::Union,
        per_file_feature_suffix: Some(vec![RNA_TAG.into(), ATAC_TAG.into()]),
        ..Default::default()
    })?;

    let feature_names = unified.feature_names.clone();
    let mut rna_feat: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut atac_feat: FxHashMap<Box<str>, u32> = FxHashMap::default();
    for (r, name) in feature_names.iter().enumerate() {
        let Some((base, tag)) = name.rsplit_once('/') else {
            anyhow::bail!("feature `{name}` carries no modality tag");
        };
        match tag {
            RNA_TAG => {
                rna_feat.insert(base.into(), r as u32);
            }
            ATAC_TAG => {
                atac_feat.insert(base.into(), r as u32);
            }
            _ => anyhow::bail!("feature `{name}`: unknown modality `{tag}`"),
        }
    }

    // ATAC modules stay inside one genomic block; RNA rows have none.
    let feature_block = {
        let blocks = super::cis::peak_blocks(&peaks, ATAC_MODULE_WINDOW);
        let mut per_row = vec![u32::MAX; feature_names.len()];
        for (p, name) in peak_names.iter().enumerate() {
            if let Some(&r) = atac_feat.get(name.as_ref()) {
                per_row[r as usize] = blocks[p];
            }
        }
        Some(per_row)
    };

    // Cis gates on the unified axis (peak→module resolved inside the fit).
    let (mut gene_feat, mut peak_feat, mut abc, mut z_log, mut pair_idx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (g, gene) in gene_names.iter().enumerate() {
        let Some(&gf) = rna_feat.get(gene.as_ref()) else {
            continue;
        };
        for k in pairs.gene(g) {
            let p = pairs.peak[k] as usize;
            let Some(&pf) = atac_feat.get(peak_names[p].as_ref()) else {
                continue;
            };
            gene_feat.push(gf);
            peak_feat.push(pf);
            abc.push(pairs.weight[k]);
            z_log.push(input.kernel.contact(pairs.dist[k]).ln());
            pair_idx.push(k as u32);
        }
    }
    let mut z_log = nalgebra::DMatrix::from_vec(z_log.len(), 1, z_log);
    z_log.scale_columns_inplace();
    let cis_gates = (!gene_feat.is_empty()).then_some(ge::CisGates {
        gene_feat,
        peak_feat,
        abc,
        z_log_contact: z_log.as_slice().to_vec(),
    });

    let h = cfg.embedding_dim;
    let device = cfg.device.to_device(cfg.device_no)?;
    info!("Embedding device: {} (#{})", cfg.device, cfg.device_no);
    let config = ge::FitConfig {
        embedding_dim: h,
        anchor_batches: None,
        bulk_batches: None,
        emit_finest_collapse: false,
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
        device,
        block_size: None,
        hvg_weights: None,
        refine: ge::RefineParams::default(),
        weight_decay: 0.0,
        phase1_cells_per_pb: cfg.phase1_cells_per_pb,
        hier_units_per_step: 256,
        hier_modules_per_unit: 8,
        module_only_min_rows: cfg.module_only_min_rows,
        feature_modules: ge::FeatureModuleArgs {
            feature_modules: Some(cfg.feature_modules),
        }
        .resolve(None)?,
        tracks: None,
        offset_l2: 0.0,
        offset_rank: 1,
        preset_features: None,
        preset_offsets: Vec::new(),
        strata: None,
        cis_gates,
        flat_module_only: true,
        multiome: Some(ge::MultiomeOptions {
            modality_intercepts: true,
            module_only_min_size: ATAC_MIN_MODULE_SIZE,
            residual_min_size: RNA_MIN_MODULE_SIZE,
            feature_block,
        }),
    };
    let out = ge::fit(&mut unified, config)?;

    let feat = nalgebra::DMatrix::<f32>::from_tensor(&out.model.e_feat)?;
    anyhow::ensure!(
        feat.nrows() == unified.feature_names.len(),
        "{} feature rows for {} names",
        feat.nrows(),
        unified.feature_names.len()
    );
    anyhow::ensure!(
        out.module_labels.len() == feat.nrows(),
        "{} module labels for {} features",
        out.module_labels.len(),
        feat.nrows()
    );

    let mut rna_row_idx = Vec::with_capacity(gene_names.len());
    for g in &gene_names {
        let r = *rna_feat
            .get(g.as_ref())
            .ok_or_else(|| anyhow::anyhow!("RNA gene `{g}` missing from the loaded axis"))?
            as usize;
        rna_row_idx.push(r);
    }
    let mut peak_row_idx = Vec::with_capacity(peak_names.len());
    let mut module_of_peak = Vec::with_capacity(peak_names.len());
    for p in &peak_names {
        let r = *atac_feat
            .get(p.as_ref())
            .ok_or_else(|| anyhow::anyhow!("peak `{p}` missing from the loaded axis"))?
            as usize;
        peak_row_idx.push(r);
        module_of_peak.push(out.module_labels[r]);
    }

    let rna_rows = feat.select_rows(rna_row_idx.iter());
    let peak_rows = feat.select_rows(peak_row_idx.iter());
    let cell_rows = nalgebra::DMatrix::<f32>::from_tensor(&out.model.e_cell)?;

    let (pairs, gate_w, gate_theta0, gate_theta1, gate_theta3, gate_gamma1, gate_gamma2) =
        match out.cis_gates.as_ref() {
            Some(cis) => {
                anyhow::ensure!(
                    cis.w.len() == cis.source_idx.len(),
                    "cis readout has {} weights for {} source indices",
                    cis.w.len(),
                    cis.source_idx.len()
                );
                // `source_idx` indexes the wired `pair_idx` → original CisPairs.
                let mut keep = Vec::with_capacity(cis.source_idx.len());
                for &src in &cis.source_idx {
                    let k = *pair_idx.get(src as usize).ok_or_else(|| {
                        anyhow::anyhow!(
                            "cis source_idx {src} out of range ({} wired pairs)",
                            pair_idx.len()
                        )
                    })?;
                    keep.push(k);
                }
                debug_assert!(keep.windows(2).all(|w| w[0] <= w[1]));
                let pairs = pairs.keep_indices(&keep);
                anyhow::ensure!(
                    pairs.n_pairs() == cis.w.len(),
                    "pruned pairs {} != gate weights {}",
                    pairs.n_pairs(),
                    cis.w.len()
                );
                (
                    pairs,
                    cis.w.clone(),
                    cis.theta0,
                    cis.theta1,
                    cis.theta3,
                    cis.gamma1,
                    cis.gamma2,
                )
            }
            None => {
                let w = pairs.weight.clone();
                (pairs, w, 0.0, 0.0, 0.0, 0.0, 1.0)
            }
        };
    Ok(TwoTrackEmbedding {
        genes: gene_names.clone(),
        rna_rows,
        peak_rows,
        module_of_peak,
        cell_rows,
        barcodes: unified.barcodes.clone(),
        pairs,
        rna_genes: gene_names,
        peak_names,
        gate_w,
        gate_theta0,
        gate_theta1,
        gate_theta3,
        gate_gamma1,
        gate_gamma2,
    })
}
