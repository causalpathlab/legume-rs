use hdf5::types::FixedAscii;
use hdf5::types::FixedUnicode;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenUnicode;

use rand_distr::num_traits;
use zarrs::storage::ReadableWritableListableStorageTraits as ZStorageTraits;

use crate::sparse_io::*;

/// Column names and their string data from an H5AD obs/var dataframe
pub struct H5adDataFrame {
    pub col_names: Vec<Box<str>>,
    pub col_data: Vec<Vec<Box<str>>>,
}

/// Resolve the backend type and the corresponding file path
///
/// If you want to decide the backend by the file name:
/// `resolve_backend_file(&data_file_path, None)`
/// If you want to check/revise output backend file name:
/// `resolve_backend_file(&output_header, Some(backend))`
///
/// If there were an extension string in the output_header,
/// we will change the backend to accommodate the backend type
/// implicated by the file name.
///
pub fn resolve_backend_file(
    file_path: &str,
    backend: Option<SparseIoBackend>,
) -> anyhow::Result<(SparseIoBackend, Box<str>)> {
    use matrix_util::common_io::file_ext;
    let ext = file_ext(file_path).unwrap_or(Box::<str>::from(""));

    if let Some(backend) = backend {
        let mut resolved_backend = backend;
        let mut backend_file = file_path.to_string();

        // if needed, change the backend to match with the file extension
        match ext.as_ref() {
            "zarr" => {
                resolved_backend = SparseIoBackend::Zarr;
            }
            "h5" => {
                resolved_backend = SparseIoBackend::HDF5;
            }
            _ => {
                // there is no extension
                backend_file = match resolved_backend {
                    SparseIoBackend::HDF5 => format!("{}.h5", file_path),
                    SparseIoBackend::Zarr => format!("{}.zarr", file_path),
                }
            }
        };

        Ok((resolved_backend, backend_file.into_boxed_str()))
    } else {
        // backend has to be inferred
        let resolved_backend = match ext.as_ref() {
            "zarr" => SparseIoBackend::Zarr,
            "h5" => SparseIoBackend::HDF5,
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", file_path)),
        };

        let backend_file = match resolved_backend {
            SparseIoBackend::HDF5 => file_path.to_string(),
            SparseIoBackend::Zarr => file_path.to_string(),
        };

        Ok((resolved_backend, backend_file.into_boxed_str()))
    }
}

/// update a v2 zarrs array to the v3 one
pub fn update_zarr_to_v3(
    store: std::sync::Arc<dyn ZStorageTraits>, // zarrs::filesystem::FilesystemStore
    key_name: &str,
) -> anyhow::Result<()> {
    use anyhow::Context;
    use zarrs::array::Array as ZArray;
    use zarrs::config::{MetadataEraseVersion, MetadataRetrieveVersion};
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

/// 10x xenium's n x 2 `cell_id` array into a vector of string codes
/// see [here](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/3.4/advanced/xoa-output-zarr#cellID)
pub fn parse_10x_cell_id(
    input: ndarray::ArrayView<u32, ndarray::IxDyn>,
) -> anyhow::Result<Vec<Box<str>>> {
    if input.ndim() != 2 || input.shape()[1] != 2 {
        return Err(anyhow::anyhow!("Must be 2D with shape [N, 2]"));
    }

    // Precomputed lookup table for hex-to-shifted conversion
    let mut lookup = [None; 256];
    for (i, ch) in "0123456789abcdef".chars().enumerate() {
        lookup[ch as usize] = Some(
            [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            ][i],
        );
    }
    input
        .outer_iter()
        .map(|row| -> anyhow::Result<Box<str>> {
            let barcode: String = format!("{:08x}", row[0])
                .chars()
                .map(|ch| {
                    lookup[ch as usize].ok_or_else(|| anyhow::anyhow!("invalid hex char: {}", ch))
                })
                .collect::<anyhow::Result<String>>()?;
            Ok(format!("{}-{}", barcode, row[1]).into_boxed_str())
        })
        .collect()
}

/// read a full ndarray from zarr storage `store`
/// * `store` - filesystem storage
/// * `key_name` - key name
pub fn read_zarr_ndarray<T>(
    store: std::sync::Arc<dyn ZStorageTraits>, // zarrs::filesystem::FilesystemStore
    key_name: &str,
) -> anyhow::Result<ndarray::ArrayD<T>>
where
    T: zarrs::array::ElementOwned + num_traits::FromPrimitive,
{
    use zarrs::array::Array as ZArray;
    use zarrs::config::MetadataRetrieveVersion;

    update_zarr_to_v3(store.clone(), key_name)?;

    let arr = ZArray::open_opt(store.clone(), key_name, &MetadataRetrieveVersion::Default)?;

    use zarrs::array::data_type::DataType;
    match arr.data_type() {
        DataType::Float32 => {
            let array: ndarray::ArrayD<f32> =
                arr.retrieve_array_subset_ndarray::<f32>(&arr.subset_all())?;
            Ok(array.mapv(|x| T::from_f32(x).unwrap()))
        }
        DataType::Float64 => {
            let array: ndarray::ArrayD<f64> =
                arr.retrieve_array_subset_ndarray::<f64>(&arr.subset_all())?;
            Ok(array.mapv(|x| T::from_f64(x).unwrap()))
        }
        DataType::UInt32 => {
            let array: ndarray::ArrayD<u32> =
                arr.retrieve_array_subset_ndarray::<u32>(&arr.subset_all())?;
            Ok(array.mapv(|x| T::from_u32(x).unwrap()))
        }
        DataType::UInt64 => {
            let array: ndarray::ArrayD<u64> =
                arr.retrieve_array_subset_ndarray::<u64>(&arr.subset_all())?;
            Ok(array.mapv(|x| T::from_u64(x).unwrap()))
        }
        _ => Err(anyhow::anyhow!("not supported data type")),
    }
}

/// read numeric vector from zarr storage `store`
/// * `store` - filesystem storage
/// * `key_name` - key name
pub fn read_zarr_numerics<T>(
    store: std::sync::Arc<dyn ZStorageTraits>, // zarrs::filesystem::FilesystemStore
    key_name: &str,
) -> anyhow::Result<Vec<T>>
where
    T: zarrs::array::ElementOwned + num_traits::FromPrimitive,
{
    use zarrs::array::Array as ZArray;
    use zarrs::config::MetadataRetrieveVersion;

    update_zarr_to_v3(store.clone(), key_name)?;

    let arr = ZArray::open_opt(store.clone(), key_name, &MetadataRetrieveVersion::Default)?;

    use zarrs::array::data_type::DataType;
    let ret = match arr.data_type() {
        DataType::Float32 => arr
            .retrieve_array_subset_elements::<f32>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_f32(x).unwrap())
            .collect(),
        DataType::Float64 => arr
            .retrieve_array_subset_elements::<f64>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_f64(x).unwrap())
            .collect(),
        DataType::UInt32 => arr
            .retrieve_array_subset_elements::<u32>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_u32(x).unwrap())
            .collect(),
        DataType::UInt64 => arr
            .retrieve_array_subset_elements::<u64>(&arr.subset_all())?
            .into_iter()
            .map(|x| T::from_u64(x).unwrap())
            .collect(),
        _ => return Err(anyhow::anyhow!("not supported data type")),
    };

    Ok(ret)
}

/// Extract `attr_name` attribute from `group_name` group
pub fn read_zarr_attr<V>(
    store: std::sync::Arc<dyn ZStorageTraits>,
    key_name: &str,
) -> anyhow::Result<V>
where
    V: serde::de::DeserializeOwned,
{
    use anyhow::Context;
    use zarrs::config::MetadataRetrieveVersion;

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
            ), // No "/" in the string
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

pub fn read_zarr_strings(
    store: std::sync::Arc<zarrs::filesystem::FilesystemStore>,
    key_name: &str,
) -> anyhow::Result<Vec<Box<str>>> {
    use zarrs::array::Array as ZArray;
    use zarrs::config::MetadataRetrieveVersion;
    update_zarr_to_v3(store.clone(), key_name)?;
    let arr = ZArray::open_opt(store.clone(), key_name, &MetadataRetrieveVersion::Default)?;

    Ok(arr
        .retrieve_array_subset_elements::<String>(&arr.subset_all())?
        .into_iter()
        .map(|x| x.into_boxed_str())
        .collect())
}

/// Read a single column from an AnnData HDF5 DataFrame group (obs or var).
///
/// Handles:
/// - Categorical columns (subgroups with `codes` + `categories` datasets)
/// - String datasets (VarLenUnicode, FixedAscii, FixedUnicode)
/// - Boolean datasets
/// - Integer and float datasets (converted to string)
pub fn read_h5ad_column(group: &hdf5::Group, col_name: &str) -> anyhow::Result<Vec<Box<str>>> {
    // Try as categorical first: column is a subgroup with codes + categories
    if let Ok(col_group) = group.group(col_name) {
        let categories = read_hdf5_strings(col_group.dataset("categories")?)?;
        let codes_ds = col_group.dataset("codes")?;
        let dtype = codes_ds.dtype()?;
        let desc = dtype.to_descriptor()?;

        let codes: Vec<i32> = match desc {
            TypeDescriptor::Integer(sz) => match sz {
                hdf5::types::IntSize::U1 => codes_ds
                    .read_1d::<i8>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
                hdf5::types::IntSize::U2 => codes_ds
                    .read_1d::<i16>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
                hdf5::types::IntSize::U4 => codes_ds.read_1d::<i32>()?.to_vec(),
                hdf5::types::IntSize::U8 => codes_ds
                    .read_1d::<i64>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
            },
            TypeDescriptor::Unsigned(sz) => match sz {
                hdf5::types::IntSize::U1 => codes_ds
                    .read_1d::<u8>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
                hdf5::types::IntSize::U2 => codes_ds
                    .read_1d::<u16>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
                hdf5::types::IntSize::U4 => codes_ds
                    .read_1d::<u32>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
                hdf5::types::IntSize::U8 => codes_ds
                    .read_1d::<u64>()?
                    .iter()
                    .map(|&x| x as i32)
                    .collect(),
            },
            _ => {
                return Err(anyhow::anyhow!(
                    "unsupported codes dtype for categorical '{}'",
                    col_name
                ));
            }
        };

        let result: Vec<Box<str>> = codes
            .iter()
            .map(|&c| {
                if c < 0 {
                    "NA".to_string().into_boxed_str()
                } else {
                    categories[c as usize].clone()
                }
            })
            .collect();

        return Ok(result);
    }

    // Otherwise it's a direct dataset
    let ds = group.dataset(col_name)?;
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor()?;

    match desc {
        TypeDescriptor::VarLenUnicode
        | TypeDescriptor::FixedAscii(_)
        | TypeDescriptor::FixedUnicode(_) => read_hdf5_strings(ds),
        TypeDescriptor::Boolean => {
            let data = ds.read_1d::<bool>()?;
            Ok(data
                .iter()
                .map(|&b| if b { "true" } else { "false" }.into())
                .collect())
        }
        TypeDescriptor::Integer(_) => {
            let data = ds.read_1d::<i64>()?;
            Ok(data
                .iter()
                .map(|x| x.to_string().into_boxed_str())
                .collect())
        }
        TypeDescriptor::Unsigned(_) => {
            let data = ds.read_1d::<u64>()?;
            Ok(data
                .iter()
                .map(|x| x.to_string().into_boxed_str())
                .collect())
        }
        TypeDescriptor::Float(sz) => match sz {
            hdf5::types::FloatSize::U4 => {
                let data = ds.read_1d::<f32>()?;
                Ok(data
                    .iter()
                    .map(|x| x.to_string().into_boxed_str())
                    .collect())
            }
            hdf5::types::FloatSize::U8 => {
                let data = ds.read_1d::<f64>()?;
                Ok(data
                    .iter()
                    .map(|x| x.to_string().into_boxed_str())
                    .collect())
            }
        },
        _ => Err(anyhow::anyhow!(
            "unsupported dtype for column '{}'",
            col_name
        )),
    }
}

/// Read all columns from an AnnData HDF5 DataFrame group (obs or var).
///
/// Uses the `column-order` attribute to discover column names,
/// then reads each column via `read_h5ad_column`.
/// Columns that fail to read are skipped with a warning.
///
/// Returns `(column_names, columns_data)` where each entry in
/// `columns_data` is a `Vec<Box<str>>` of the same length.
pub fn read_h5ad_dataframe(
    group: &hdf5::Group,
) -> anyhow::Result<H5adDataFrame> {
    let col_order: Vec<String> = group
        .attr("column-order")?
        .read_1d::<VarLenUnicode>()?
        .iter()
        .map(|x| x.to_string())
        .collect();

    let mut col_names = Vec::new();
    let mut col_data = Vec::new();

    for col_name in &col_order {
        match read_h5ad_column(group, col_name) {
            Ok(data) => {
                col_names.push(col_name.clone().into_boxed_str());
                col_data.push(data);
            }
            Err(e) => {
                log::warn!("Skipping obs column '{}': {}", col_name, e);
            }
        }
    }

    Ok(H5adDataFrame {
        col_names,
        col_data,
    })
}

/// Read strings from `HDF5` dataset
pub fn read_hdf5_strings(data: hdf5::dataset::Dataset) -> anyhow::Result<Vec<Box<str>>> {
    let dtype = data.dtype()?;
    let desc = dtype.to_descriptor()?;

    let ret: Vec<Box<str>> = match desc {
        TypeDescriptor::VarLenUnicode => data
            .read_1d::<VarLenUnicode>()?
            .map(|x| x.to_string().into_boxed_str())
            .into_iter()
            .collect(),
        TypeDescriptor::FixedAscii(n) => {
            if n < 24 {
                data.read_1d::<FixedAscii<24>>()?
                    .map(|x| x.to_string().into_boxed_str())
                    .into_iter()
                    .collect()
            } else if n < 128 {
                data.read_1d::<FixedAscii<128>>()?
                    .map(|x| x.to_string().into_boxed_str())
                    .into_iter()
                    .collect()
            } else {
                data.read_1d::<FixedAscii<1024>>()?
                    .map(|x| x.to_string().into_boxed_str())
                    .into_iter()
                    .collect()
            }
        }
        TypeDescriptor::FixedUnicode(n) => {
            if n < 24 {
                data.read_1d::<FixedUnicode<24>>()?
                    .map(|x| x.to_string().into_boxed_str())
                    .into_iter()
                    .collect()
            } else if n < 128 {
                data.read_1d::<FixedUnicode<128>>()?
                    .map(|x| x.to_string().into_boxed_str())
                    .into_iter()
                    .collect()
            } else {
                data.read_1d::<FixedUnicode<1024>>()?
                    .map(|x| x.to_string().into_boxed_str())
                    .into_iter()
                    .collect()
            }
        }
        _ => {
            return Err(anyhow::anyhow!("unsupported string"));
        }
    };

    Ok(ret)
}

// ndarray v0.17 issue
// use ndarray::Array1;
// fn ndarray_into_box_str<T, U, S>(data: Array1<T>) -> Vec<Box<str>>
// where
//     T: RawData<Elem = U> + Data + ToString,
// {
//     data.into_iter()
//         .map(|x| x.to_string().into_boxed_str())
//         .collect()
// }
