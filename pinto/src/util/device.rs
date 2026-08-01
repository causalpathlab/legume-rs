//! Compute-device selection, shared by every subcommand that runs a candle
//! model.
//!
//! Lives here rather than in one subcommand's `args.rs` because more than one
//! command needs it: `cage` defined it first, and `lc-etm` ran CPU-only until
//! it could reach this.

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
