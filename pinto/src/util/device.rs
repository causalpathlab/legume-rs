//! Compute-device selection for subcommands that run a candle model.
//!
//! `cage` is the only such command today. This lives here rather than in its
//! `args.rs` so a second one can reach it without a cross-subcommand import.

use clap::ValueEnum;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl ComputeDevice {
    pub fn to_device(&self, device_no: usize) -> anyhow::Result<candle_util::candle_core::Device> {
        use candle_util::candle_core::Device;
        Ok(match self {
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::new_cuda(device_no)?,
            ComputeDevice::Metal => Device::new_metal(device_no)?,
        })
    }
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ComputeDevice::Cpu => "cpu",
            ComputeDevice::Cuda => "cuda",
            ComputeDevice::Metal => "metal",
        })
    }
}
