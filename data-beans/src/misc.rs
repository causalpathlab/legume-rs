use hdf5::types::FixedAscii;
use hdf5::types::FixedUnicode;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenUnicode;

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
pub fn read_h5ad_dataframe(group: &hdf5::Group) -> anyhow::Result<H5adDataFrame> {
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
