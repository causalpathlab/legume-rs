// Export a data-beans backend back out to interchange formats.
// The h5ad exporter needs libhdf5, so it compiles out without the feature.
// The mtx (10x MEX) exporter has no HDF5 dependency and is always available.

#[cfg(feature = "hdf5")]
mod to_h5ad;
mod to_mtx;

#[cfg(feature = "hdf5")]
pub use to_h5ad::*;
pub use to_mtx::*;
