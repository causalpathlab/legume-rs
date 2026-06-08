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

/// Read a `*_components.parquet` sidecar and tag every record with the
/// given modality name. The modality label must match the one in the
/// modifier row names (`m6A`, `A2I`, `pA`) so the `(gene, modality,
/// component)` lookup lines up with `FeatureTable`'s parsed rows.
///
/// Two sidecar schemas are accepted — they share the **same** matrix row
/// convention `{gene}/{modality}/{k}`, so the component index `k` always
/// lines up; only the sidecar layout differs:
///
/// * **GMM** (`faba m6a` / `atoi`, via `editing::mixture_pipeline`):
///   `gene_name`, `component_idx` (any integer type), `mu`, `gene_length`.
///   Position `u = μ / gene_length`.
/// * **APA** (`faba apa`, via `apa::pipeline`): `site_id`
///   (`{gene}/pA/{k}`), `expected_tail_length`, `utr_length`. The gene and
///   component `k` are parsed from `site_id` (matching the matrix row),
///   and the 5'→3' UTR position is recovered as
///   `u = 1 − expected_tail_length / utr_length` — encoded here as
///   `mu = utr_length − expected_tail_length`, `gene_length = utr_length`
///   so the shared `u = μ / gene_length` binning needs no special case.
pub fn load_component_annotations(path: &str, modality: &str) -> Result<Vec<ComponentAnnotation>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("parquet reader for {path}"))?
        .build()?;

    let modality: Box<str> = modality.into();
    let mut out = Vec::new();
    for batch in reader {
        let batch = batch?;
        // Sniff the schema by a distinguishing column, then dispatch.
        if batch.column_by_name("component_idx").is_some() {
            load_gmm_batch(&batch, &modality, path, &mut out)?;
        } else if batch.column_by_name("site_id").is_some() {
            load_apa_batch(&batch, &modality, path, &mut out)?;
        } else {
            anyhow::bail!(
                "{path}: unrecognized component sidecar schema — \
                 need 'component_idx' (GMM) or 'site_id' (apa)"
            );
        }
    }
    Ok(out)
}

/// GMM sidecar batch (m6A / A2I): `gene_name` + `component_idx` + `mu` +
/// `gene_length`, position `u = μ / gene_length`.
fn load_gmm_batch(
    batch: &arrow::record_batch::RecordBatch,
    modality: &str,
    path: &str,
    out: &mut Vec<ComponentAnnotation>,
) -> Result<()> {
    use arrow::array::StringArray;
    let gene_col = batch
        .column_by_name("gene_name")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| anyhow::anyhow!("{path}: missing/!string 'gene_name'"))?;
    let comp = read_u32_col(batch, "component_idx", path)?;
    let mu_col = f32_col(batch, "mu", path)?;
    let len_col = f32_col(batch, "gene_length", path)?;
    for (i, &component_idx) in comp.iter().enumerate() {
        out.push(ComponentAnnotation {
            gene_name: gene_col.value(i).into(),
            modality: modality.into(),
            component_idx,
            mu: mu_col.value(i),
            gene_length: len_col.value(i),
        });
    }
    Ok(())
}

/// APA sidecar batch: `site_id` (`{gene}/pA/{k}`), `expected_tail_length`,
/// `utr_length`. Gene and component `k` are parsed from `site_id`
/// (matching the matrix row), and the 5'→3' UTR position is recovered as
/// `u = 1 − etl/utr` — encoded as `mu = utr − etl`, `gene_length = utr` so
/// the shared `u = μ / gene_length` binning needs no special case.
fn load_apa_batch(
    batch: &arrow::record_batch::RecordBatch,
    modality: &str,
    path: &str,
    out: &mut Vec<ComponentAnnotation>,
) -> Result<()> {
    use arrow::array::StringArray;
    let site_col = batch
        .column_by_name("site_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| anyhow::anyhow!("{path}: missing/!string 'site_id'"))?;
    let etl_col = f32_col(batch, "expected_tail_length", path)?;
    let utr = read_u32_col(batch, "utr_length", path)?;
    for (i, &utr_raw) in utr.iter().enumerate() {
        let Some((gene, comp)) = parse_site_id(site_col.value(i)) else {
            continue; // unparseable site_id → no matching matrix row
        };
        let utr_len = utr_raw as f32;
        out.push(ComponentAnnotation {
            gene_name: gene,
            modality: modality.into(),
            component_idx: comp,
            mu: utr_len - etl_col.value(i),
            gene_length: utr_len,
        });
    }
    Ok(())
}

/// Downcast a named column to `Float32Array` (cheap Arc clone of the
/// backing buffer).
fn f32_col(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    path: &str,
) -> Result<arrow::array::Float32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<arrow::array::Float32Array>())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{path}: missing/!f32 '{name}'"))
}

/// Read an integer arrow column as `Vec<u32>`, accepting any of the common
/// integer widths (`u64`/`u32`/`i64`/`i32`). Relaxing the type lets the
/// GMM sidecar's `u64 component_idx` and the apa sidecar's `u32
/// utr_length` flow through one path. Negative values clamp to 0.
fn read_u32_col(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    path: &str,
) -> Result<Vec<u32>> {
    use arrow::array::{Int32Array, Int64Array, UInt32Array, UInt64Array};
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("{path}: missing integer column '{name}'"))?;
    let any = col.as_any();
    if let Some(a) = any.downcast_ref::<UInt64Array>() {
        Ok((0..a.len()).map(|i| a.value(i) as u32).collect())
    } else if let Some(a) = any.downcast_ref::<UInt32Array>() {
        Ok((0..a.len()).map(|i| a.value(i)).collect())
    } else if let Some(a) = any.downcast_ref::<Int64Array>() {
        Ok((0..a.len()).map(|i| a.value(i).max(0) as u32).collect())
    } else if let Some(a) = any.downcast_ref::<Int32Array>() {
        Ok((0..a.len()).map(|i| a.value(i).max(0) as u32).collect())
    } else {
        anyhow::bail!("{path}: column '{name}' is not an integer type")
    }
}

/// Parse an apa `site_id` (`{gene}/pA/{k}`) into `(gene, component_idx)` by
/// reusing the matrix row-name grammar (`feature_table::parse_feature_name`
/// then `parse_component_idx`), so the region key can never drift from how
/// the matrix row is parsed. Returns `None` if the shape or the integer `k`
/// doesn't parse.
fn parse_site_id(site_id: &str) -> Option<(Box<str>, u32)> {
    let key = super::feature_table::parse_feature_name(site_id)?;
    let k = super::feature_table::parse_component_idx(&key.detail)?;
    Some((key.gene, k))
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
                (
                    rec.gene_name.clone(),
                    rec.modality.clone(),
                    rec.component_idx,
                ),
                region,
            );
        }
        Self { n_regions, by_key }
    }

    /// Empty map (no annotations): every lookup falls back to region 0.
    /// Equivalent to `from_records(&[], n_regions)`; the production path
    /// uses that form, so this is a test-only convenience.
    #[cfg(test)]
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

    #[test]
    fn parses_site_id() {
        assert_eq!(parse_site_id("ENSG_X/pA/0"), Some(("ENSG_X".into(), 0)));
        assert_eq!(parse_site_id("G/pA/11"), Some(("G".into(), 11))); // multi-digit k
        assert_eq!(parse_site_id("G/pA/x"), None); // non-integer k
        assert_eq!(parse_site_id("/pA/0"), None); // empty gene
        assert_eq!(parse_site_id("nope"), None); // wrong shape
    }

    fn write_parquet(path: &std::path::Path, batch: &arrow::record_batch::RecordBatch) {
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        let file = std::fs::File::create(path).unwrap();
        let props = WriterProperties::builder().build();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        w.write(batch).unwrap();
        w.close().unwrap();
    }

    #[test]
    fn loads_apa_schema() {
        use arrow::array::{ArrayRef, Float32Array, StringArray, UInt32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        // APA sidecar: site_id (`{gene}/pA/{k}`) + expected_tail_length +
        // utr_length. Position u = 1 − etl/utr: comp0 → 0.8, comp2 → 0.1.
        let schema = Arc::new(Schema::new(vec![
            Field::new("site_id", DataType::Utf8, false),
            Field::new("gene_name", DataType::Utf8, false),
            Field::new("expected_tail_length", DataType::Float32, false),
            Field::new("utr_length", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["GENEA/pA/0", "GENEA/pA/2"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["GENEA", "GENEA"])) as ArrayRef,
                Arc::new(Float32Array::from(vec![200.0_f32, 900.0])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![1000_u32, 1000])) as ArrayRef,
            ],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("faba_apa_{}.parquet", std::process::id()));
        write_parquet(&path, &batch);
        let recs = load_component_annotations(path.to_str().unwrap(), "pA").unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(recs.len(), 2);
        // gene + component parsed from site_id (not the gene_name column).
        assert!(recs
            .iter()
            .any(|r| &*r.gene_name == "GENEA" && r.component_idx == 0));
        assert!(recs
            .iter()
            .any(|r| &*r.gene_name == "GENEA" && r.component_idx == 2));
        // u = 1 − etl/utr → region bins via the shared μ/gene_length path.
        let map = RegionMap::from_records(&recs, 5);
        assert_eq!(map.lookup("GENEA", "pA", 0), 4); // u=0.8 → bin 4
        assert_eq!(map.lookup("GENEA", "pA", 2), 0); // u=0.1 → bin 0
    }

    #[test]
    fn loads_gmm_relaxed_int_type() {
        use arrow::array::{ArrayRef, Float32Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        // GMM sidecar with component_idx as i64 (not the canonical u64) —
        // the relaxed integer reader must still accept it.
        let schema = Arc::new(Schema::new(vec![
            Field::new("gene_name", DataType::Utf8, false),
            Field::new("component_idx", DataType::Int64, false),
            Field::new("mu", DataType::Float32, false),
            Field::new("gene_length", DataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["GENEB"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![3_i64])) as ArrayRef,
                Arc::new(Float32Array::from(vec![80.0_f32])) as ArrayRef,
                Arc::new(Float32Array::from(vec![100.0_f32])) as ArrayRef,
            ],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("faba_gmm_{}.parquet", std::process::id()));
        write_parquet(&path, &batch);
        let recs = load_component_annotations(path.to_str().unwrap(), "m6A").unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].component_idx, 3); // i64 read through
        assert_eq!(recs[0].gene_name.as_ref(), "GENEB");
        let map = RegionMap::from_records(&recs, 5);
        assert_eq!(map.lookup("GENEB", "m6A", 3), 4); // u=0.8 → bin 4
    }
}
