//! Re-export dense mat helpers from [`matrix_util::dense_mat_io`].
//!
//! Kept as a thin module so existing `crate::mat_io::*` paths inside this crate
//! stay stable after the helpers moved out of senna to break the crate cycle.

pub use matrix_util::dense_mat_io::{
    axis_id_names, l2_normalize_rows_inplace, read_mat, Mat, MatWithNames,
};
