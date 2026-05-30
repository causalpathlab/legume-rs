//! Transcript-position region binning for modifier components.
//!
//! Per locked decision #1, a satellite feature row `{gene}/{m}/{k}` is
//! assigned a **region** from the normalized 5'-relative position of its
//! GMM component: `u = μ / gene_length ∈ [0, 1)`, binned into `R` fixed
//! fractional bins. The region keys the additive offset `γ_{m,r,:}` in
//! the model's log-deviation gate, so two same-gene/same-modality
//! components sitting in different regions get distinguishable
//! embeddings (multi-site resolution).
//!
//! True CDS/UTR regions are a later refinement; APA can use a UTR-local
//! `α` once that lands.
//!
//! The region lookup is keyed by `(gene_name, modality_name,
//! component_idx)` — the same three coordinates that name a modifier
//! row. `μ` and `gene_length` come from the per-modality
//! `*_components.parquet` annotation emitted by
//! `editing::mixture_pipeline` (`MixtureComponentAnnotation`).

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;

/// Read a `*_components.parquet` sidecar (columns `gene_name`,
/// `component_idx`, `mu`, `gene_length`) and tag every record with the
/// given modality name. The modality label must match the one in the
/// modifier row names (`m6A`, `A2I`, `pA`) so the `(gene, modality,
/// component)` lookup lines up with `FeatureTable`'s parsed rows.
pub fn load_component_annotations(path: &str, modality: &str) -> Result<Vec<ComponentAnnotation>> {
    use arrow::array::{Float32Array, StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("parquet reader for {path}"))?
        .build()?;

    let modality: Box<str> = modality.into();
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        let gene_col = batch
            .column_by_name("gene_name")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow::anyhow!("{path}: missing/!string 'gene_name'"))?;
        let comp_col = batch
            .column_by_name("component_idx")
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| anyhow::anyhow!("{path}: missing/!u64 'component_idx'"))?;
        let mu_col = batch
            .column_by_name("mu")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
            .ok_or_else(|| anyhow::anyhow!("{path}: missing/!f32 'mu'"))?;
        let len_col = batch
            .column_by_name("gene_length")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
            .ok_or_else(|| anyhow::anyhow!("{path}: missing/!f32 'gene_length'"))?;
        for i in 0..batch.num_rows() {
            out.push(ComponentAnnotation {
                gene_name: gene_col.value(i).into(),
                modality: modality.clone(),
                component_idx: comp_col.value(i) as u32,
                mu: mu_col.value(i),
                gene_length: len_col.value(i),
            });
        }
    }
    Ok(out)
}

/// One annotation record, modality-tagged. Mirrors the columns of a
/// `*_components.parquet` (`gene_name`, `component_idx`, `mu`,
/// `gene_length`) plus the modality the file belongs to.
#[derive(Clone, Debug)]
pub struct ComponentAnnotation {
    pub gene_name: Box<str>,
    pub modality: Box<str>,
    pub component_idx: u32,
    pub mu: f32,
    pub gene_length: f32,
}

/// `(gene_name, modality_name, component_idx) → region_id`. Built once
/// from the annotation records, consulted by `FeatureTable::build` while
/// classifying modifier rows.
pub struct RegionMap {
    /// Number of fractional bins R. `region_id ∈ 0..n_regions`.
    pub n_regions: usize,
    by_key: FxHashMap<(Box<str>, Box<str>, u32), u32>,
}

impl RegionMap {
    /// Build the region lookup from annotation records. A record with a
    /// non-finite or non-positive `gene_length`, or a non-finite `mu`,
    /// is skipped (the row falls back to region 0 at lookup time). The
    /// normalized position `u = μ / gene_length` is clamped to
    /// `[0, 1)` before binning so off-end components still land in a
    /// valid bin rather than overflowing `R`.
    pub fn from_records(records: &[ComponentAnnotation], n_regions: usize) -> Self {
        let n_regions = n_regions.max(1);
        let mut by_key: FxHashMap<(Box<str>, Box<str>, u32), u32> = FxHashMap::default();
        for rec in records {
            let region = match region_for(rec.mu, rec.gene_length, n_regions) {
                Some(r) => r,
                None => continue,
            };
            by_key.insert(
                (rec.gene_name.clone(), rec.modality.clone(), rec.component_idx),
                region,
            );
        }
        Self { n_regions, by_key }
    }

    /// Empty map (no annotations): every lookup falls back to region 0.
    /// Used when no `*_components.parquet` is supplied, collapsing
    /// `γ_{m,r,:}` to a single per-modality offset.
    #[allow(dead_code)] // used by tests; also a stable public helper
    pub fn empty(n_regions: usize) -> Self {
        Self {
            n_regions: n_regions.max(1),
            by_key: FxHashMap::default(),
        }
    }

    /// Region for `(gene, modality, component)`. Falls back to region 0
    /// when the triple was not annotated (count-comp details, missing
    /// files, or components dropped from the sidecar).
    pub fn lookup(&self, gene: &str, modality: &str, component: u32) -> u32 {
        self.by_key
            .get(&(gene.into(), modality.into(), component))
            .copied()
            .unwrap_or(0)
    }
}

/// Bin `u = μ / gene_length` into one of `n_regions` fixed fractional
/// bins. Returns `None` for un-binnable inputs (non-finite μ, non-finite
/// or non-positive gene_length).
fn region_for(mu: f32, gene_length: f32, n_regions: usize) -> Option<u32> {
    if !mu.is_finite() || !gene_length.is_finite() || gene_length <= 0.0 {
        return None;
    }
    let u = (mu / gene_length).clamp(0.0, 1.0 - f32::EPSILON);
    let r = (u * n_regions as f32).floor() as i64;
    let r = r.clamp(0, n_regions as i64 - 1);
    Some(r as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_normalized_position() {
        // R=5 fractional bins over u ∈ [0,1).
        assert_eq!(region_for(0.0, 100.0, 5), Some(0)); // u=0
        assert_eq!(region_for(10.0, 100.0, 5), Some(0)); // u=0.1 → bin 0
        assert_eq!(region_for(25.0, 100.0, 5), Some(1)); // u=0.25 → bin 1
        assert_eq!(region_for(99.0, 100.0, 5), Some(4)); // u≈0.99 → bin 4
        assert_eq!(region_for(150.0, 100.0, 5), Some(4)); // off-end clamps
    }

    #[test]
    fn rejects_bad_inputs() {
        assert_eq!(region_for(f32::NAN, 100.0, 5), None);
        assert_eq!(region_for(10.0, 0.0, 5), None);
        assert_eq!(region_for(10.0, f32::NAN, 5), None);
    }

    #[test]
    fn lookup_falls_back_to_zero() {
        let recs = vec![ComponentAnnotation {
            gene_name: "geneA".into(),
            modality: "m6A".into(),
            component_idx: 0,
            mu: 80.0,
            gene_length: 100.0,
        }];
        let map = RegionMap::from_records(&recs, 5);
        assert_eq!(map.lookup("geneA", "m6A", 0), 4); // u=0.8 → bin 4
        assert_eq!(map.lookup("geneA", "m6A", 1), 0); // unannotated → 0
        assert_eq!(map.lookup("geneB", "m6A", 0), 0); // unknown gene → 0
    }
}
