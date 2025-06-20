use hdf5::types::FixedAscii;
use hdf5::types::FixedUnicode;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenUnicode;

pub fn read_hdf5_strings(names: hdf5::dataset::Dataset) -> anyhow::Result<Vec<Box<str>>> {
    let dtype = names.dtype()?;
    let desc = dtype.to_descriptor()?;

    let ret: Vec<Box<str>> = match desc {
        TypeDescriptor::VarLenUnicode => into_box_str(&names.read_1d::<VarLenUnicode>()?),
        TypeDescriptor::FixedAscii(n) => {
            if n < 24 {
                into_box_str(&names.read_1d::<FixedAscii<24>>()?)
            } else if n < 128 {
                into_box_str(&names.read_1d::<FixedAscii<128>>()?)
            } else {
                into_box_str(&names.read_1d::<FixedAscii<1024>>()?)
            }
        }
        TypeDescriptor::FixedUnicode(n) => {
            if n < 24 {
                into_box_str(&names.read_1d::<FixedUnicode<24>>()?)
            } else if n < 128 {
                into_box_str(&names.read_1d::<FixedUnicode<128>>()?)
            } else {
                into_box_str(&names.read_1d::<FixedUnicode<1024>>()?)
            }
        }
        _ => {
            return Err(anyhow::anyhow!("unsupported string"));
        }
    };

    Ok(ret)
}

use ndarray::{ArrayBase, Data, Dim, RawData};

fn into_box_str<T, U>(data: &ArrayBase<T, Dim<[usize; 1]>>) -> Vec<Box<str>>
where
    T: RawData<Elem = U> + Data,
    U: ToString,
{
    data.into_iter()
        .map(|x| x.to_string().into_boxed_str())
        .collect()
}
