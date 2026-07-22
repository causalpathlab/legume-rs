//! Per-gene 1D Gaussian mixture orchestration for m6A / A-to-I sites.
//!
//! Split out of `editing::pipeline` (which handles site discovery, stats
//! gathering, and backend output). Components are fit on the pooled
//! replicates but counts are written per batch.

use crate::common::*;
use crate::data::conversion::*;
use crate::editing::pipeline::{gather_conversion_stats, ConversionParams, MixtureWeightMode};
use crate::editing::sifter::*;
use crate::editing::ConversionSite;

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use genomic_data::gff::{GeneId, GffRecordMap};
use std::sync::{Arc, Mutex};

/// Run per-gene 1D Gaussian mixture model on discovered sites and output results.
///
/// For each gene, collects cell-level observations from the second-pass stats,
/// fits a GMM via BIC model selection, and outputs:
/// - A sparse (cells x mixture_components) count matrix
///   with feature IDs like `GENE/m6A/0`, `GENE/A2I/1`
/// - A `{m6a,atoi}_components.parquet` file
pub fn run_mixture_model(
    params: &ConversionParams,
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    gff_map: &GffRecordMap,
    mixture_params: &crate::editing::mixture::MixtureParams,
    valid_cells: Option<&rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>>>,
) -> anyhow::Result<()> {
    use crate::editing::mixture::{
        fit_gene_mixture, MixtureComponentAnnotation, WeightedObservation,
    };

    let membership = params.load_membership()?;

    // Collect cell-level stats from each replicate, tagging each observation
    // with its batch index. Components are fit on the POOLED observations
    // (shared across replicates), but the per-cell counts are written out PER
    // BATCH, so the batch tag rides along to the output split.
    // Signal (wt) AND control (mut) batches: the mut mixture matrices are the
    // background sanity check and feed gem (see `quant_bam_files`).
    let quant_bam_files = params.quant_bam_files();
    let batch_names = uniq_batch_names(&quant_bam_files)?;

    let mut fit_stats: Vec<(usize, CellBarcode, BedWithGene, ConversionData)> = Vec::new();
    for (batch_idx, bam_file) in quant_bam_files.iter().enumerate() {
        // Filter each batch by its own per-library called cell set, looked up by
        // BAM file path (stable across the QC and quant passes).
        let batch_valid_cells = valid_cells.and_then(|m| m.get(bam_file));
        let stats = gather_conversion_stats(
            gene_sites,
            params,
            gff_map,
            bam_file,
            membership.as_ref(),
            batch_valid_cells,
        )?;
        fit_stats.extend(
            stats
                .into_iter()
                .map(|(cb, bed, cd)| (batch_idx, cb, bed, cd)),
        );
    }

    info!(
        "Mixture model: collected {} observations across {} batches",
        fit_stats.len(),
        batch_names.len(),
    );

    if fit_stats.is_empty() {
        info!("No observations to fit mixture model");
        return Ok(());
    }

    // Build a global cell index keyed by (batch, barcode) so identical
    // barcodes in different replicates stay distinct cells — and the
    // per-cell component counts can be split back out per batch downstream.
    let mut unique_cells: Vec<(usize, CellBarcode)> = fit_stats
        .iter()
        .map(|(b, cb, _, _)| (*b, cb.clone()))
        .collect();
    unique_cells.sort();
    unique_cells.dedup();
    let cell_to_idx: rustc_hash::FxHashMap<(usize, CellBarcode), usize> = unique_cells
        .iter()
        .enumerate()
        .map(|(i, key)| (key.clone(), i))
        .collect();

    // Group observations by gene, converting to strand-aware relative position.
    // The third tuple element is the per-observation weight (raw count under
    // Converted mode, regularized effective count under Posterior mode).
    let make_obs = |batch_idx: usize,
                    cb: &CellBarcode,
                    bed: &BedWithGene,
                    meth: &ConversionData|
     -> (GeneId, (usize, f32, f32)) {
        let cell_idx = cell_to_idx[&(batch_idx, cb.clone())];
        let c = meth.converted as f32;
        let u = meth.unconverted as f32;
        let n = c + u;
        let w = match params.mixture_weight_mode {
            MixtureWeightMode::Converted => c,
            MixtureWeightMode::Posterior => {
                let a = params.mixture_prior_alpha;
                let b = params.mixture_prior_beta;
                if n + a + b > 0.0 {
                    let r_hat = (c + a) / (n + a + b);
                    n * r_hat
                } else {
                    0.0
                }
            }
        };
        // Convert absolute site_pos to strand-aware relative position
        let rel_pos = if let Some(gff) = gff_map.get(&bed.gene) {
            let lb = (gff.start - 1).max(0); // GFF 1-based -> 0-based
            let ub = gff.stop;
            match gff.strand {
                Strand::Forward => (meth.site_pos - lb) as f32,
                Strand::Backward => (ub - meth.site_pos - 1) as f32,
            }
        } else {
            meth.site_pos as f32
        };
        (bed.gene.clone(), (cell_idx, rel_pos, w))
    };

    let mut gene_obs: rustc_hash::FxHashMap<GeneId, Vec<(usize, f32, f32)>> =
        rustc_hash::FxHashMap::default();
    for (batch_idx, cb, bed, meth) in &fit_stats {
        let (gene, obs) = make_obs(*batch_idx, cb, bed, meth);
        gene_obs.entry(gene).or_default().push(obs);
    }

    // Resolve the component-calling bandwidth once for this modality. An
    // explicit --mixture-bandwidth (> 0) wins; otherwise estimate a global value
    // from the empirical within-gene site spacing (cluster-aware for A-to-I).
    let resolved_params = {
        let mut p = mixture_params.clone();
        if p.bandwidth <= 0.0 {
            use crate::editing::bandwidth::{estimate_bandwidth, BandwidthParams};
            let cluster_aware = matches!(params.mod_type, ModificationType::AtoI);
            let bw_params = if cluster_aware {
                BandwidthParams::atoi()
            } else {
                BandwidthParams::m6a()
            };
            let per_gene: Vec<Vec<(f32, f32)>> = gene_obs
                .values()
                .map(|obs| obs.iter().map(|&(_, pos, w)| (pos, w)).collect())
                .collect();
            let est = estimate_bandwidth(&per_gene, &bw_params);
            info!(
                "Mixture: auto bandwidth = {:.1} nt ({} gaps pooled, {})",
                est.bandwidth,
                est.n_gaps,
                if cluster_aware {
                    "A-to-I cluster-aware"
                } else {
                    "m6A"
                }
            );
            p.bandwidth = est.bandwidth;
        } else {
            info!(
                "Mixture: bandwidth = {:.1} nt (user-specified)",
                p.bandwidth
            );
        }
        p
    };
    let mixture_params = &resolved_params;

    // Fit mixture per gene in parallel
    let gene_entries: Vec<_> = gene_obs.into_iter().collect();
    // Triplets carry the batch index so the single pooled fit can be
    // demultiplexed into one matrix per replicate at write time.
    type Triplet = (usize, CellBarcode, Box<str>, f32);
    let arc_triplets: Arc<Mutex<Vec<Triplet>>> = Arc::new(Mutex::new(Vec::new()));
    let arc_annotations: Arc<Mutex<Vec<MixtureComponentAnnotation>>> =
        Arc::new(Mutex::new(Vec::new()));

    gene_entries.par_iter().for_each(|(gene_id, obs_list)| {
        let gene_name: Box<str> = gff_map
            .get(gene_id)
            .map(|gff| match &gff.gene_name {
                GeneSymbol::Symbol(s) => format!("{}_{}", gene_id, s),
                GeneSymbol::Missing => format!("{}", gene_id),
            })
            .unwrap_or_else(|| format!("{}", gene_id))
            .into();

        // `gene_length` feeds the EM's uniform-noise normalizer (where a
        // 1000-nt fallback is fine) AND the parquet sidecar (where it
        // should not smuggle a sentinel out as if it were a measurement).
        // Diverge: EM gets the fallback, sidecar gets NaN for missing.
        // GFF coords are 1-based inclusive (genomic-data/src/gff.rs:508),
        // so the nucleotide span is `stop - start + 1`, not `stop - start`.
        let gene_span = gff_map
            .get(gene_id)
            .map(|gff| (gff.stop - gff.start + 1) as f32);
        let gene_length = gene_span.unwrap_or(1000.0);
        let gene_length_emit = gene_span.unwrap_or(f32::NAN);

        let cell_observations: Vec<WeightedObservation> = obs_list
            .iter()
            .map(|&(cell_idx, position, count)| WeightedObservation {
                cell_idx,
                position,
                count,
            })
            .collect();

        if let Some(result) = fit_gene_mixture(&cell_observations, gene_length, mixture_params) {
            // Build component annotations (filter pi=0) and a forward old→new map
            // for the 1-based GMM component index used in result.cell_component_counts.
            let mut local_annotations = Vec::new();
            let mut old_to_new: rustc_hash::FxHashMap<usize, usize> =
                rustc_hash::FxHashMap::default();
            for (j, (&mu, &sigma)) in result
                .gmm
                .mus
                .iter()
                .zip(result.gmm.sigmas.iter())
                .enumerate()
            {
                let pi = result.gmm.weights[j + 1]; // skip noise
                if pi > 0.0 {
                    let new_idx = local_annotations.len();
                    old_to_new.insert(j, new_idx);
                    local_annotations.push(MixtureComponentAnnotation {
                        gene_name: gene_name.clone(),
                        component_idx: new_idx,
                        mu,
                        sigma,
                        pi,
                        gene_length: gene_length_emit,
                    });
                }
            }

            // Optionally drop genes that resolved to a single active
            // component: their lone per-cell count is just the gene total,
            // so the row carries no relative/differential signal.
            if mixture_params.drop_single_component && local_annotations.len() < 2 {
                return;
            }

            // Build triplets: (batch, cell_barcode, feature_id, count).
            // Feature IDs: GENE/m6A/0, GENE/A2I/1, etc. using renumbered indices.
            // Each count references the shared (batch, barcode) cell index, so
            // the per-batch output split routes it to its own matrix.
            let mod_suffix = match params.mod_type {
                ModificationType::M6A { .. } => "m6A",
                ModificationType::AtoI => "A2I",
            };
            let mut local_triplets = Vec::new();
            for (cell_idx, component, count) in result.cell_component_counts.iter() {
                if *component == 0 {
                    continue; // skip noise
                }
                // Map old GMM component (1-based, skip noise) to new consecutive index
                if let Some(&new_idx) = old_to_new.get(&(component - 1)) {
                    let feature_id: Box<str> =
                        format!("{}/{}/{}", gene_name, mod_suffix, new_idx).into();
                    let (batch_idx, cb) = &unique_cells[*cell_idx];
                    local_triplets.push((*batch_idx, cb.clone(), feature_id, *count));
                }
            }

            arc_triplets.lock().expect("lock").extend(local_triplets);
            arc_annotations
                .lock()
                .expect("lock")
                .extend(local_annotations);
        }
    });

    let triplets_data = Arc::try_unwrap(arc_triplets)
        .map_err(|_| anyhow::anyhow!("failed to unwrap triplets"))?
        .into_inner()?;

    let annotations = Arc::try_unwrap(arc_annotations)
        .map_err(|_| anyhow::anyhow!("failed to unwrap annotations"))?
        .into_inner()?;

    info!(
        "Mixture model: {} triplets, {} component annotations",
        triplets_data.len(),
        annotations.len()
    );

    if triplets_data.is_empty() {
        info!("No mixture results to output");
        return Ok(());
    }

    // The component fit is shared (pooled), but each replicate gets its own
    // matrix, named `{batch}_{m6a,atoi}_mixture`. Because every batch is
    // scored against the same components, the rows (GENE/mod/component) are
    // a common vocabulary — reorder them to a shared sorted union so the
    // per-batch matrices stack/join directly.
    let mod_tag = match params.mod_type {
        ModificationType::M6A { .. } => "m6a_mixture",
        ModificationType::AtoI => "atoi_mixture",
    };

    let mut by_batch: rustc_hash::FxHashMap<usize, Vec<(CellBarcode, Box<str>, f32)>> =
        rustc_hash::FxHashMap::default();
    for (batch_idx, cb, feat, count) in triplets_data {
        by_batch
            .entry(batch_idx)
            .or_default()
            .push((cb, feat, count));
    }

    let mut all_rows = HashSet::<Box<str>>::default();
    let mut out_files: Vec<crate::quant::BackendOutputPath> = Vec::new();
    for (batch_idx, batch_name) in batch_names.iter().enumerate() {
        let Some(trip) = by_batch.remove(&batch_idx) else {
            continue;
        };
        if trip.is_empty() {
            continue;
        }
        let out = params.backend_output_path(&format!("{}_{}", batch_name, mod_tag));
        let data = format_data_triplets(trip).to_backend(&out.write_path)?;
        all_rows.extend(data.row_names()?);
        info!("Mixture model: created {}", &out.target_path);
        drop(data);
        // Defer finalize() until rows are reordered on the staging .zarr.
        out_files.push(out);
    }

    let mut rows_sorted: Vec<_> = all_rows.into_iter().collect();
    rows_sorted.sort();
    for out in &out_files {
        open_sparse_matrix(&out.write_path, &params.backend)?.reorder_rows(&rows_sorted)?;
    }
    for out in out_files {
        out.finalize()?;
    }

    // Write annotations parquet (shared component definitions, single file)
    if !annotations.is_empty() {
        let components_name = match params.mod_type {
            ModificationType::M6A { .. } => "m6a_components",
            ModificationType::AtoI => "atoi_components",
        };
        write_mixture_annotations(
            &annotations,
            &format!("{}/{}.parquet", &params.output, components_name),
        )?;
    }

    Ok(())
}

fn write_mixture_annotations(
    annotations: &[crate::editing::mixture::MixtureComponentAnnotation],
    path: &str,
) -> anyhow::Result<()> {
    use arrow::array::{ArrayRef, Float32Array, StringArray, UInt64Array};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let gene_names: Vec<&str> = annotations.iter().map(|a| a.gene_name.as_ref()).collect();
    let component_idxs: Vec<u64> = annotations.iter().map(|a| a.component_idx as u64).collect();
    let mus: Vec<f32> = annotations.iter().map(|a| a.mu).collect();
    let sigmas: Vec<f32> = annotations.iter().map(|a| a.sigma).collect();
    let pis: Vec<f32> = annotations.iter().map(|a| a.pi).collect();
    let gene_lengths: Vec<f32> = annotations.iter().map(|a| a.gene_length).collect();

    let schema = arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("gene_name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("component_idx", arrow::datatypes::DataType::UInt64, false),
        arrow::datatypes::Field::new("mu", arrow::datatypes::DataType::Float32, false),
        arrow::datatypes::Field::new("sigma", arrow::datatypes::DataType::Float32, false),
        arrow::datatypes::Field::new("pi", arrow::datatypes::DataType::Float32, false),
        arrow::datatypes::Field::new("gene_length", arrow::datatypes::DataType::Float32, false),
    ]);

    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![
            std::sync::Arc::new(StringArray::from(gene_names)) as ArrayRef,
            std::sync::Arc::new(UInt64Array::from(component_idxs)) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(mus)) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(sigmas)) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(pis)) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(gene_lengths)) as ArrayRef,
        ],
    )?;

    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    info!(
        "Wrote {} mixture annotations to {}",
        annotations.len(),
        path
    );
    Ok(())
}
