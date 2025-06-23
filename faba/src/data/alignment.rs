#![allow(dead_code)]

use noodles::sam::alignment::record::data::field::value::Value as SamTagValue;

/// alignment file sample name
///
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum SamSampleName {
    Combined,
    Barcode(Box<str>),
}

pub fn sam_sample_name(vv: SamTagValue<'_>) -> anyhow::Result<SamSampleName> {
    Ok(match vv {
        SamTagValue::String(bstr) => SamSampleName::Barcode(std::str::from_utf8(bstr)?.into()),
        _ => SamSampleName::Combined,
    })
}

/// alignment file UMI name
///
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum SamUmiName {
    Combined,
    Barcode(Box<str>),
}

/// Converts a `SamTagValue` into a `UmiName`.
pub fn sam_umi_name(vv: SamTagValue<'_>) -> anyhow::Result<SamUmiName> {
    Ok(match vv {
        SamTagValue::String(bstr) => SamUmiName::Barcode(std::str::from_utf8(bstr)?.into()),
        _ => SamUmiName::Combined,
    })
}

/// Display sample names
///
impl std::fmt::Display for SamSampleName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamSampleName::Combined => write!(f, "."),
            SamSampleName::Barcode(barcode) => write!(f, "{}", barcode),
        }
    }
}

impl std::fmt::Display for SamUmiName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamUmiName::Combined => write!(f, "."),
            SamUmiName::Barcode(barcode) => write!(f, "{}", barcode),
        }
    }
}
