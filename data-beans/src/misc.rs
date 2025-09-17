use hdf5::types::FixedAscii;
use hdf5::types::FixedUnicode;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenUnicode;


/// update a v2 zarrs array to the v3 one
pub fn update_zarr_to_v3(
    store: std::sync::Arc<zarrs::filesystem::FilesystemStore>,
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

/// read numeric vector from zarr storage `store`
/// * `store` - filesystem storage
/// * `key_name` - key name
pub fn read_zarr_array<T>(
    store: std::sync::Arc<zarrs::filesystem::FilesystemStore>,
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
