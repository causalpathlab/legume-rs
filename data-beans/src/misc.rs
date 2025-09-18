use hdf5::types::FixedAscii;
use hdf5::types::FixedUnicode;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenUnicode;

use zarrs::storage::ReadableWritableListableStorageTraits as ZStorageTraits;

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

    match arr.metadata() {
        ArrayMetadata::V2(_v2) => {
            let arr = arr.to_v3().with_context(|| "unable to convert to v3")?;
            arr.store_metadata()
                .with_context(|| "failed to store meta data")?;
            arr.erase_metadata_opt(MetadataEraseVersion::V2)
                .with_context(|| "failed to erase the old one")?;
        }
        _ => {}
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

/// Read strings from `HDF5` dataset
pub fn read_hdf5_strings(data: hdf5::dataset::Dataset) -> anyhow::Result<Vec<Box<str>>> {
    let dtype = data.dtype()?;
    let desc = dtype.to_descriptor()?;

    let ret: Vec<Box<str>> = match desc {
        TypeDescriptor::VarLenUnicode => ndarray_into_box_str(&data.read_1d::<VarLenUnicode>()?),
        TypeDescriptor::FixedAscii(n) => {
            if n < 24 {
                ndarray_into_box_str(&data.read_1d::<FixedAscii<24>>()?)
            } else if n < 128 {
                ndarray_into_box_str(&data.read_1d::<FixedAscii<128>>()?)
            } else {
                ndarray_into_box_str(&data.read_1d::<FixedAscii<1024>>()?)
            }
        }
        TypeDescriptor::FixedUnicode(n) => {
            if n < 24 {
                ndarray_into_box_str(&data.read_1d::<FixedUnicode<24>>()?)
            } else if n < 128 {
                ndarray_into_box_str(&data.read_1d::<FixedUnicode<128>>()?)
            } else {
                ndarray_into_box_str(&data.read_1d::<FixedUnicode<1024>>()?)
            }
        }
        _ => {
            return Err(anyhow::anyhow!("unsupported string"));
        }
    };

    Ok(ret)
}

use ndarray::{ArrayBase, Data, Dim, RawData};
use rand_distr::num_traits;

fn ndarray_into_box_str<T, U>(data: &ArrayBase<T, Dim<[usize; 1]>>) -> Vec<Box<str>>
where
    T: RawData<Elem = U> + Data,
    U: ToString,
{
    data.into_iter()
        .map(|x| x.to_string().into_boxed_str())
        .collect()
}
