//! General-purpose zarr I/O for reading numeric arrays and attributes from
//! zarr directories or `.zarr.zip` archives.
//!
//! Also provides a convenience function for reading coordinate-style data
//! (row names + numeric matrix) that returns `MatWithNames<DMatrix<f32>>`,
//! matching the parquet reader interface in matrix-util.

use log::info;
use matrix_util::traits::MatWithNames;
use nalgebra::DMatrix;
use rand_distr::num_traits::FromPrimitive;
use std::sync::Arc;
use zarrs::array::{data_type, Array as ZArray};
use zarrs::config::MetadataRetrieveVersion;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorageTraits as ZReadStorageTraits;
use zarrs::storage::ReadableWritableListableStorageTraits as ZStorageTraits;
use zarrs_zip::ZipStorageAdapter;

// ── V2 → V3 migration ──────────────────────────────────────────────────

/// Upgrade a zarr v2 array to v3 format in-place (no-op if already v3).
pub fn update_zarr_to_v3(store: Arc<dyn ZStorageTraits>, key_name: &str) -> anyhow::Result<()> {
    use anyhow::Context;
    use zarrs::config::MetadataEraseVersion;
    use zarrs::metadata::ArrayMetadata;

    let arr = ZArray::open_opt(store.clone(), key_name, &MetadataRetrieveVersion::Default)?;

    if let ArrayMetadata::V2(_v2) = arr.metadata() {
        let arr = arr.to_v3().with_context(|| "unable to convert to v3")?;
        arr.store_metadata()
            .with_context(|| "failed to store meta data")?;
        arr.erase_metadata_opt(MetadataEraseVersion::V2)
            .with_context(|| "failed to erase the old one")?;
    };

    Ok(())
}

// ── Store management ────────────────────────────────────────────────────

/// A read store paired with an optional write store (present for directories, `None` for zips).
pub type ZarrStoreRw = (Arc<dyn ZReadStorageTraits>, Option<Arc<FilesystemStore>>);

/// Detect the path prefix inside a zarr zip (empty if entries are at root).
fn detect_zip_zarr_prefix(zip_path: &std::path::Path) -> anyhow::Result<Box<str>> {
    let filename = zip_path
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid zip path: {:?}", zip_path))?;
    let expected_prefix = format!("{}/", filename.strip_suffix(".zip").unwrap_or(""));

    let file = std::fs::File::open(zip_path)?;
    let archive = zip::ZipArchive::new(std::io::BufReader::new(file))?;
    let has_prefix = archive
        .file_names()
        .any(|name| name.starts_with(&expected_prefix));

    Ok(if has_prefix {
        expected_prefix.into()
    } else {
        "".into()
    })
}

/// Open a zarr store, returning both a read store and an optional write store.
///
/// Supports both `.zarr` directories and `.zarr.zip` archives.
/// For zip archives, uses [`ZipStorageAdapter`] for direct random-access
/// reads without extracting to a temp directory; the write store is `None`.
pub fn open_zarr_store_rw(path: &str) -> anyhow::Result<ZarrStoreRw> {
    let p = std::path::Path::new(path);
    let is_zip = p.extension().is_some_and(|e| e == "zip");

    if is_zip {
        let parent = p
            .parent()
            .ok_or_else(|| anyhow::anyhow!("no parent directory for {}", path))?;
        let filename = p
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("no filename for {}", path))?
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 filename in {}", path))?;
        let zarr_prefix = detect_zip_zarr_prefix(p)?;
        let fs = Arc::new(FilesystemStore::new(parent)?);
        let key = zarrs::storage::StoreKey::new(filename)?;
        info!(
            "Opening zarr zip store: {} (prefix: {:?})",
            path, zarr_prefix
        );
        Ok((
            Arc::new(ZipStorageAdapter::new_with_path(fs, key, &*zarr_prefix)?),
            None,
        ))
    } else {
        let fs = Arc::new(FilesystemStore::new(path)?);
        Ok((fs.clone(), Some(fs)))
    }
}

/// Open a zarr store for reading (convenience wrapper around [`open_zarr_store_rw`]).
pub fn open_zarr_store(path: &str) -> anyhow::Result<Arc<dyn ZReadStorageTraits>> {
    open_zarr_store_rw(path).map(|(r, _)| r)
}

/// If `zip` is true, ensure the output path ends with `.zarr.zip`.
/// Otherwise pass through unchanged.
pub fn apply_zip_flag(output: &str, zip: bool) -> Box<str> {
    if zip && !output.ends_with(".zarr.zip") {
        let base = crate::hdf5_io::strip_backend_suffix(output);
        format!("{}.zarr.zip", base).into()
    } else {
        output.into()
    }
}

/// Extract a `.zarr.zip` archive into `target_dir`, transparently stripping
/// any internal prefix produced by [`matrix_util::common_io::zip_dir`] so
/// the result is a flat `.zarr` directory regardless of the zip's filename.
pub fn extract_zarr_zip(zip_path: &str, target_dir: &str) -> anyhow::Result<()> {
    let zip_file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(zip_file))?;

    // Detect prefix directly from the already-opened archive (avoids a
    // second open + central-directory parse).
    let filename = std::path::Path::new(zip_path)
        .file_name()
        .and_then(|f| f.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid zip path: {}", zip_path))?;
    let expected_prefix = format!("{}/", filename.strip_suffix(".zip").unwrap_or(""));
    let prefix: &str = if archive
        .file_names()
        .any(|n| n.starts_with(&expected_prefix))
    {
        &expected_prefix
    } else {
        ""
    };

    std::fs::create_dir_all(target_dir)?;
    let target = std::path::Path::new(target_dir);

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let rel = file.name().strip_prefix(prefix).unwrap_or(file.name());
        if rel.is_empty() {
            continue;
        }
        let out_path = target.join(rel);
        if file.is_dir() {
            std::fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut outfile = std::fs::File::create(&out_path)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

/// Copy input backend to a writable output location. Transparently extracts
/// `.zarr.zip` archives into a `.zarr` directory so the output can be opened
/// read/write.
pub fn materialize_writable_backend(src: &str, dst: &str) -> anyhow::Result<()> {
    if src.ends_with(".zarr.zip") {
        info!("extracting {} → {}", src, dst);
        extract_zarr_zip(src, dst)
    } else {
        matrix_util::common_io::recursive_copy(src, dst)
    }
}

/// If `target_path` ends with `.zarr.zip`, zip the `zarr_dir` into it and
/// remove the directory. Otherwise this is a no-op (the directory IS the target).
pub fn finalize_zarr_output(zarr_dir: &str, target_path: &str) -> anyhow::Result<()> {
    if target_path.ends_with(".zarr.zip") {
        info!("Zipping zarr output: {} → {}", zarr_dir, target_path);
        matrix_util::common_io::zip_dir(zarr_dir, target_path)?;
        std::fs::remove_dir_all(zarr_dir)?;
    }
    Ok(())
}

// ── Attribute reading ───────────────────────────────────────────────────

/// Read an attribute from a zarr array node.
///
/// ```text
/// read_zarr_array_attr::<Vec<String>>(store, "/cell_summary", "column_names")
/// ```
pub fn read_zarr_array_attr<V: serde::de::DeserializeOwned>(
    store: Arc<dyn ZReadStorageTraits>,
    array_path: &str,
    attr_name: &str,
) -> anyhow::Result<V> {
    let arr = ZArray::open_opt(store, array_path, &MetadataRetrieveVersion::Default)?;
    let attr = arr.attributes().get(attr_name).ok_or_else(|| {
        anyhow::anyhow!(
            "attribute '{}' not found on array '{}'",
            attr_name,
            array_path
        )
    })?;
    Ok(serde_json::from_value(attr.clone())?)
}

// ── Flat array readers ──────────────────────────────────────────────────

/// Retrieve a zarr array as a flat `Vec<f32>` (row-major) and its shape.
///
/// Handles f32, f64, u32, u64 source types with automatic conversion.
pub fn read_zarr_flat_f32(
    store: Arc<dyn ZReadStorageTraits>,
    key: &str,
) -> anyhow::Result<(Vec<f32>, Vec<u64>)> {
    let arr = ZArray::open_opt(store, key, &MetadataRetrieveVersion::Default)?;
    let shape = arr.shape().to_vec();
    let subset = arr.subset_all();

    let dt = arr.data_type();
    let data = if *dt == data_type::float32() {
        arr.retrieve_array_subset::<Vec<f32>>(&subset)?
    } else if *dt == data_type::float64() {
        arr.retrieve_array_subset::<Vec<f64>>(&subset)?
            .into_iter()
            .map(|x| x as f32)
            .collect()
    } else if *dt == data_type::uint32() {
        arr.retrieve_array_subset::<Vec<u32>>(&subset)?
            .into_iter()
            .map(|x| x as f32)
            .collect()
    } else if *dt == data_type::uint64() {
        arr.retrieve_array_subset::<Vec<u64>>(&subset)?
            .into_iter()
            .map(|x| x as f32)
            .collect()
    } else {
        anyhow::bail!("unsupported zarr data type: {:?}", dt)
    };

    Ok((data, shape))
}

/// Retrieve a zarr array as a flat `Vec<u32>` and its shape.
pub fn read_zarr_flat_u32(
    store: Arc<dyn ZReadStorageTraits>,
    key: &str,
) -> anyhow::Result<(Vec<u32>, Vec<u64>)> {
    let arr = ZArray::open_opt(store, key, &MetadataRetrieveVersion::Default)?;
    let shape = arr.shape().to_vec();
    let subset = arr.subset_all();

    let dt = arr.data_type();
    let data = if *dt == data_type::uint32() {
        arr.retrieve_array_subset::<Vec<u32>>(&subset)?
    } else if *dt == data_type::uint64() {
        arr.retrieve_array_subset::<Vec<u64>>(&subset)?
            .into_iter()
            .map(|x| x as u32)
            .collect()
    } else {
        anyhow::bail!("unsupported zarr data type for u32: {:?}", dt)
    };

    Ok((data, shape))
}

// ── Generic array/attribute readers (moved from misc.rs) ────────────────

/// Read a full ndarray from zarr storage.
pub fn read_zarr_ndarray<T>(
    store: Arc<dyn ZReadStorageTraits>,
    key_name: &str,
) -> anyhow::Result<ndarray::ArrayD<T>>
where
    T: zarrs::array::ElementOwned + FromPrimitive,
{
    let arr = ZArray::open_opt(store, key_name, &MetadataRetrieveVersion::Default)?;

    let dt = arr.data_type();
    if *dt == data_type::float32() {
        let array: ndarray::ArrayD<f32> =
            arr.retrieve_array_subset::<ndarray::ArrayD<f32>>(&arr.subset_all())?;
        Ok(array.mapv(|x| T::from_f32(x).unwrap()))
    } else if *dt == data_type::float64() {
        let array: ndarray::ArrayD<f64> =
            arr.retrieve_array_subset::<ndarray::ArrayD<f64>>(&arr.subset_all())?;
        Ok(array.mapv(|x| T::from_f64(x).unwrap()))
    } else if *dt == data_type::uint32() {
        let array: ndarray::ArrayD<u32> =
            arr.retrieve_array_subset::<ndarray::ArrayD<u32>>(&arr.subset_all())?;
        Ok(array.mapv(|x| T::from_u32(x).unwrap()))
    } else if *dt == data_type::uint64() {
        let array: ndarray::ArrayD<u64> =
            arr.retrieve_array_subset::<ndarray::ArrayD<u64>>(&arr.subset_all())?;
        Ok(array.mapv(|x| T::from_u64(x).unwrap()))
    } else {
        anyhow::bail!("unsupported zarr data type: {:?}", dt)
    }
}

/// Read a numeric vector from zarr storage.
pub fn read_zarr_numerics<T>(
    store: Arc<dyn ZReadStorageTraits>,
    key_name: &str,
) -> anyhow::Result<Vec<T>>
where
    T: zarrs::array::ElementOwned + FromPrimitive,
{
    let arr = ZArray::open_opt(store, key_name, &MetadataRetrieveVersion::Default)?;

    let dt = arr.data_type();
    let ret = if *dt == data_type::float32() {
        arr.retrieve_array_subset::<Vec<f32>>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_f32(x).unwrap())
            .collect()
    } else if *dt == data_type::float64() {
        arr.retrieve_array_subset::<Vec<f64>>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_f64(x).unwrap())
            .collect()
    } else if *dt == data_type::uint32() {
        arr.retrieve_array_subset::<Vec<u32>>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_u32(x).unwrap())
            .collect()
    } else if *dt == data_type::uint64() {
        arr.retrieve_array_subset::<Vec<u64>>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_u64(x).unwrap())
            .collect()
    } else {
        anyhow::bail!("unsupported zarr data type: {:?}", dt);
    };

    Ok(ret)
}

/// Extract an attribute from a zarr group node.
///
/// `key_name` is parsed as `"group_path/attr_name"`, e.g.
/// `"/cell_features/features/id"` → group `"/cell_features/features"`, attr `"id"`.
pub fn read_zarr_group_attr<V>(
    store: Arc<dyn ZReadStorageTraits>,
    key_name: &str,
) -> anyhow::Result<V>
where
    V: serde::de::DeserializeOwned,
{
    use anyhow::Context;

    fn parse_key_name(key_name: &str) -> (Box<str>, Box<str>) {
        let trimmed = key_name.strip_prefix('/').unwrap_or(key_name);
        match trimmed.rsplit_once('/') {
            Some((left, right)) => (
                format!("/{}", left).into_boxed_str(),
                right.to_string().into_boxed_str(),
            ),
            None => (
                "/".to_string().into_boxed_str(),
                trimmed.to_string().into_boxed_str(),
            ),
        }
    }

    let (group_name, attr_name) = parse_key_name(key_name);

    let group = zarrs::group::Group::open_opt(
        store,
        group_name.as_ref(),
        &MetadataRetrieveVersion::Default,
    )
    .with_context(|| format!("Failed to open group '{}'", group_name))?;

    let attr_value = group
        .attributes()
        .get(attr_name.as_ref())
        .with_context(|| {
            format!(
                "Attribute '{}' not found in group '{}'",
                attr_name, group_name
            )
        })?;

    Ok(serde_json::from_value(attr_value.clone())?)
}

/// Read a string array from zarr storage.
pub fn read_zarr_strings(
    store: Arc<dyn ZReadStorageTraits>,
    key_name: &str,
) -> anyhow::Result<Vec<Box<str>>> {
    let arr = ZArray::open_opt(store, key_name, &MetadataRetrieveVersion::Default)?;

    Ok(arr
        .retrieve_array_subset::<Vec<String>>(&arr.subset_all())?
        .into_iter()
        .map(|x| x.into_boxed_str())
        .collect())
}

// ── 10x cell ID encoding ────────────────────────────────────────────────

/// Hex-digit → shifted-alpha lookup table for 10x cell ID encoding.
fn hex_to_shifted_lookup() -> [Option<char>; 256] {
    let mut lookup = [None; 256];
    for (i, ch) in "0123456789abcdef".chars().enumerate() {
        lookup[ch as usize] = Some(
            [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            ][i],
        );
    }
    lookup
}

/// Encode a single `(barcode_u32, suffix_u32)` pair into a 10x cell-ID string.
fn encode_10x_cell_id(
    lookup: &[Option<char>; 256],
    barcode: u32,
    suffix: u32,
) -> anyhow::Result<Box<str>> {
    let barcode: String = format!("{:08x}", barcode)
        .chars()
        .map(|ch| lookup[ch as usize].ok_or_else(|| anyhow::anyhow!("invalid hex char: {}", ch)))
        .collect::<anyhow::Result<String>>()?;
    Ok(format!("{}-{}", barcode, suffix).into_boxed_str())
}

/// Parse a 10x Xenium `[N, 2]` u32 cell_id ndarray into string barcodes.
///
/// See [10x docs](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/3.4/advanced/xoa-output-zarr#cellID).
pub fn parse_10x_cell_id(
    input: ndarray::ArrayView<u32, ndarray::IxDyn>,
) -> anyhow::Result<Vec<Box<str>>> {
    anyhow::ensure!(
        input.ndim() == 2 && input.shape()[1] == 2,
        "Must be 2D with shape [N, 2]"
    );
    let lookup = hex_to_shifted_lookup();
    input
        .outer_iter()
        .map(|row| encode_10x_cell_id(&lookup, row[0], row[1]))
        .collect()
}

/// Parse a flat row-major `[N, 2]` u32 buffer into 10x Xenium cell-ID strings.
///
/// Same encoding as [`parse_10x_cell_id`] but works on a flat slice.
pub fn parse_10x_cell_id_flat(data: &[u32], nrows: usize) -> anyhow::Result<Vec<Box<str>>> {
    anyhow::ensure!(data.len() == nrows * 2, "cell_id buffer size mismatch");
    let lookup = hex_to_shifted_lookup();
    (0..nrows)
        .map(|i| encode_10x_cell_id(&lookup, data[i * 2], data[i * 2 + 1]))
        .collect()
}

// ── Matrix reading (generic) ────────────────────────────────────────────

/// Build a column-major `DMatrix<f32>` by selecting columns from a flat
/// row-major `[nrows, ncols_total]` buffer.
fn select_columns_to_dmatrix(
    data: &[f32],
    nrows: usize,
    ncols_total: usize,
    selected: &[usize],
) -> anyhow::Result<DMatrix<f32>> {
    let k = selected.len();
    let mut col_major = vec![0.0f32; nrows * k];
    for (col_out, &col_in) in selected.iter().enumerate() {
        anyhow::ensure!(col_in < ncols_total, "column index {} out of range", col_in);
        for row in 0..nrows {
            col_major[row + col_out * nrows] = data[row * ncols_total + col_in];
        }
    }
    Ok(DMatrix::from_vec(nrows, k, col_major))
}

/// Resolve column selection: explicit indices, name lookup, or default.
fn resolve_columns(
    column_indices: &[usize],
    column_names: &[Box<str>],
    all_col_names: &[Box<str>],
    default_cols: &[usize],
) -> anyhow::Result<Vec<usize>> {
    if !column_indices.is_empty() {
        Ok(column_indices.to_vec())
    } else if !column_names.is_empty() {
        // Keep only names that exist — allows generous defaults
        // covering multiple platforms (e.g. Visium + Xenium).
        let matched: Vec<usize> = column_names
            .iter()
            .filter_map(|name| all_col_names.iter().position(|c| c == name))
            .collect();
        if matched.is_empty() {
            anyhow::bail!(
                "none of the requested columns {:?} found (available: {:?})",
                column_names,
                all_col_names
            );
        }
        Ok(matched)
    } else {
        Ok(default_cols.to_vec())
    }
}

/// Read a numeric matrix from a zarr array, selecting columns by index or
/// name (looked up via an attribute on the array).
///
/// * `file_path` — zarr directory or `.zarr.zip`
/// * `data_array` — path to the numeric `[N, C]` array (e.g. `/cell_summary`)
/// * `col_names_attr` — attribute name on `data_array` that holds column
///   names (e.g. `"column_names"`); pass `None` to skip name lookup
/// * `row_names_array` — path to the row-name array (e.g. `/cell_id`);
///   `None` generates numeric row names `"0", "1", …`
/// * `row_names_10x` — if `true`, parse `row_names_array` as 10x u32 `[N,2]`
///   cell IDs; if `false`, read as string array
/// * `column_indices` / `column_names` — which columns to select
pub fn read_zarr_matrix(
    file_path: &str,
    data_array: &str,
    col_names_attr: Option<&str>,
    row_names_array: Option<&str>,
    row_names_10x: bool,
    column_indices: &[usize],
    column_names: &[Box<str>],
) -> anyhow::Result<MatWithNames<DMatrix<f32>>> {
    let store = open_zarr_store(file_path)?;

    // Read the data array
    let (flat, shape) = read_zarr_flat_f32(store.clone(), data_array)?;
    let nrows = shape[0] as usize;
    let ncols_total = if shape.len() > 1 {
        shape[1] as usize
    } else {
        1
    };

    // Read column names from attribute (if available)
    let all_col_names: Vec<Box<str>> = if let Some(attr) = col_names_attr {
        read_zarr_array_attr(store.clone(), data_array, attr).unwrap_or_else(|_| {
            (0..ncols_total)
                .map(|i| i.to_string().into_boxed_str())
                .collect()
        })
    } else {
        (0..ncols_total)
            .map(|i| i.to_string().into_boxed_str())
            .collect()
    };

    // Resolve column selection
    let selected = resolve_columns(column_indices, column_names, &all_col_names, &[0, 1])?;
    let sel_names: Vec<Box<str>> = selected
        .iter()
        .map(|&i| {
            all_col_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| i.to_string().into_boxed_str())
        })
        .collect();

    // Build the matrix
    let mat = select_columns_to_dmatrix(&flat, nrows, ncols_total, &selected)?;

    // Read row names
    let row_names = if let Some(rn_path) = row_names_array {
        if row_names_10x {
            let (id_data, id_shape) = read_zarr_flat_u32(store.clone(), rn_path)?;
            anyhow::ensure!(
                id_shape[0] as usize == nrows,
                "row names array rows ({}) != data rows ({})",
                id_shape[0],
                nrows
            );
            parse_10x_cell_id_flat(&id_data, nrows)?
        } else {
            read_zarr_strings(store.clone(), rn_path)?
        }
    } else {
        (0..nrows).map(|i| i.to_string().into_boxed_str()).collect()
    };

    info!(
        "Read {} x {} from zarr {}{}: {:?}",
        nrows,
        selected.len(),
        file_path,
        data_array,
        sel_names
    );

    Ok(MatWithNames {
        rows: row_names,
        cols: sel_names,
        mat,
    })
}

// ── Xenium convenience ──────────────────────────────────────────────────

/// Read cell coordinates from a Xenium-style zarr file.
///
/// Shorthand for [`read_zarr_matrix`] with Xenium defaults:
///   - data array: `/cell_summary`
///   - column names attribute: `column_names`
///   - row names: `/cell_id` (10x encoded)
pub fn read_zarr_coordinates(
    file_path: &str,
    column_indices: &[usize],
    column_names: &[Box<str>],
) -> anyhow::Result<MatWithNames<DMatrix<f32>>> {
    read_zarr_matrix(
        file_path,
        "/cell_summary",
        Some("column_names"),
        Some("/cell_id"),
        true,
        column_indices,
        column_names,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xenium_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("docs/temp/cells.zarr.zip");
        p.exists().then_some(p)
    }

    #[test]
    fn test_parse_10x_cell_id_flat() {
        let data = vec![16844u32, 1, 22527, 1];
        let ids = parse_10x_cell_id_flat(&data, 2).unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids[0].ends_with("-1"), "got: {}", ids[0]);
        assert!(ids[1].ends_with("-1"), "got: {}", ids[1]);
        assert_eq!(ids[0].len(), 10);
    }

    #[test]
    fn test_parse_10x_cell_id_flat_mismatch() {
        assert!(parse_10x_cell_id_flat(&[1, 2, 3], 2).is_err());
    }

    #[test]
    fn test_read_zarr_coordinates_by_name() {
        let Some(p) = xenium_path() else { return };
        let result = read_zarr_coordinates(
            p.to_str().unwrap(),
            &[],
            &["cell_centroid_x".into(), "cell_centroid_y".into()],
        )
        .unwrap();

        assert_eq!(result.mat.ncols(), 2);
        assert!(result.mat.nrows() > 0);
        assert_eq!(result.rows.len(), result.mat.nrows());
        assert_eq!(result.cols[0].as_ref(), "cell_centroid_x");
        assert_eq!(result.cols[1].as_ref(), "cell_centroid_y");
        assert!(result.mat.min() >= 0.0);
    }

    #[test]
    fn test_read_zarr_coordinates_by_index() {
        let Some(p) = xenium_path() else { return };
        let result = read_zarr_coordinates(p.to_str().unwrap(), &[0, 1], &[]).unwrap();
        assert_eq!(result.mat.ncols(), 2);
        assert_eq!(result.cols[0].as_ref(), "cell_centroid_x");
    }

    #[test]
    fn test_read_zarr_coordinates_default() {
        let Some(p) = xenium_path() else { return };
        let result = read_zarr_coordinates(p.to_str().unwrap(), &[], &[]).unwrap();
        assert_eq!(result.mat.ncols(), 2);
    }

    #[test]
    fn test_read_zarr_coordinates_bad_column() {
        let Some(p) = xenium_path() else { return };
        let result = read_zarr_coordinates(
            p.to_str().unwrap(),
            &[],
            &["nonexistent".to_string().into_boxed_str()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_read_zarr_coordinates_mixed_defaults() {
        // Simulates pinto's generous defaults: Visium names that won't match
        // plus Xenium names that will — only matching names should be kept.
        let Some(p) = xenium_path() else { return };
        let result = read_zarr_coordinates(
            p.to_str().unwrap(),
            &[],
            &[
                "pxl_row_in_fullres".into(),
                "pxl_col_in_fullres".into(),
                "cell_centroid_x".into(),
                "cell_centroid_y".into(),
            ],
        )
        .unwrap();
        assert_eq!(result.mat.ncols(), 2);
        assert_eq!(result.cols[0].as_ref(), "cell_centroid_x");
        assert_eq!(result.cols[1].as_ref(), "cell_centroid_y");
        assert!(result.mat.nrows() > 0);
    }

    #[test]
    fn test_read_zarr_array_attr() {
        let Some(p) = xenium_path() else { return };
        let store = open_zarr_store(p.to_str().unwrap()).unwrap();
        let names: Vec<Box<str>> =
            read_zarr_array_attr(store, "/cell_summary", "column_names").unwrap();
        assert!(!names.is_empty());
        assert!(names.contains(&"cell_centroid_x".to_string().into_boxed_str()));
    }

    #[test]
    fn test_read_zarr_matrix_generic() {
        let Some(p) = xenium_path() else { return };
        // Read cell_summary selecting columns 2,3 (cell_area, nucleus_centroid_x)
        let result = read_zarr_matrix(
            p.to_str().unwrap(),
            "/cell_summary",
            Some("column_names"),
            Some("/cell_id"),
            true,
            &[2, 3],
            &[],
        )
        .unwrap();
        assert_eq!(result.mat.ncols(), 2);
        assert_eq!(result.cols[0].as_ref(), "cell_area");
        assert_eq!(result.cols[1].as_ref(), "nucleus_centroid_x");
    }
}
