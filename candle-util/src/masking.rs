//! Padded batches score every column, so a pad column must lose every softmax
//! it appears in: PBG's `−1e9` added to its score gives it exactly zero
//! probability and no gradient.

use candle_core::{Result, Tensor};

/// PBG's "ignore this column" score.
pub const MASK_NEG: f64 = -1e9;

/// `(1 − valid) · MASK_NEG`: zero on valid columns, `MASK_NEG` on pads, from
/// a `{1, 0}` validity indicator of any shape.
pub fn additive_pad_mask(valid: &Tensor) -> Result<Tensor> {
    valid.affine(-MASK_NEG, MASK_NEG)
}
