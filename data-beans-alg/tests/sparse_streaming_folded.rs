//! The folded streaming pass must agree with a pass over the already-pooled
//! matrix, on all four statistics, exactly.
//!
//! Two of the four are the point. `s2` of a sum is not the sum of `s2`, and
//! `npos` of a sum is not the sum of `npos`, so a fold applied to a row-axis
//! result is wrong in a way that a fold applied inside the pass is not. The
//! last test pins that difference rather than leaving it as a claim.

use data_beans::sparse_io::*;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::sparse_streaming::{
    streaming_sparse_running_stats, streaming_sparse_running_stats_folded,
};
use ndarray::Array2;
use std::sync::Arc;

const N_CELLS: usize = 6;
const N_GENES: usize = 4;

/// Row `r` carries gene `ROW_TO_GENE[r]`. Deliberately interleaved and out of
/// gene order, so a fold that quietly assumes rows are grouped fails here.
/// Gene 3 has a single row, which is the common case on real data.
const ROW_TO_GENE: [u32; 7] = [1, 0, 2, 0, 1, 3, 2];

/// `[7 x 6]`, chosen so every case is present: cells firing on both rows of a
/// gene (which is what separates folded `npos` from summed `npos`), cells
/// firing on one, and cells firing on neither.
fn channelized() -> Array2<f32> {
    Array2::from_shape_vec(
        (7, N_CELLS),
        vec![
            3.0, 0.0, 1.0, 0.0, 5.0, 2.0, // g1 nascent
            4.0, 2.0, 0.0, 7.0, 1.0, 0.0, // g0 mature
            0.0, 6.0, 3.0, 0.0, 0.0, 8.0, // g2 mature
            1.0, 5.0, 0.0, 2.0, 0.0, 0.0, // g0 nascent  (cells 0,1,3 double-fire with row 1)
            2.0, 9.0, 0.0, 0.0, 4.0, 0.0, // g1 mature   (cells 0,4 double-fire with row 0)
            0.0, 0.0, 7.0, 3.0, 0.0, 1.0, // g3 mature   (single-row gene)
            5.0, 0.0, 0.0, 6.0, 2.0, 0.0, // g2 nascent
        ],
    )
    .expect("shape")
}

/// Fold by walking rows in ASCENDING order, which is the order a CSC column
/// presents them, so the f32 additions happen in the same sequence the folded
/// accumulator uses and the comparison can be exact rather than approximate.
fn pooled(from: &Array2<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((N_GENES, N_CELLS));
    for r in 0..from.nrows() {
        let g = ROW_TO_GENE[r] as usize;
        for c in 0..N_CELLS {
            out[(g, c)] += from[(r, c)];
        }
    }
    out
}

fn names(n: usize, prefix: &str) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

fn as_data(raw: &Array2<f32>, prefix: &str) -> anyhow::Result<SparseIoVec> {
    let mut sp = create_sparse_from_ndarray(raw, None, None)?;
    sp.register_row_names_vec(&names(raw.nrows(), prefix));
    sp.register_column_names_vec(&names(N_CELLS, "cell"));
    sp.preload_columns()?;
    let mut data = SparseIoVec::new();
    data.push(Arc::from(sp), Some("batch0".into()))?;
    Ok(data)
}

#[test]
fn folded_statistics_equal_the_pooled_matrix_statistics() -> anyhow::Result<()> {
    let chan = channelized();
    let pool = pooled(&chan);

    let chan_data = as_data(&chan, "row")?;
    let pool_data = as_data(&pool, "gene")?;

    let (row_stats, folded) =
        streaming_sparse_running_stats_folded(&chan_data, None, "test", &ROW_TO_GENE, N_GENES)?;
    let reference = streaming_sparse_running_stats(&pool_data, None, "test")?;

    let (f_npos, f_sum, f_mean, f_std) = folded.to_vecs();
    let (r_npos, r_sum, r_mean, r_std) = reference.to_vecs();

    assert_eq!(f_npos, r_npos, "npos");
    assert_eq!(f_sum, r_sum, "sum");
    assert_eq!(f_mean, r_mean, "mean");
    assert_eq!(f_std, r_std, "std");

    // The row-axis half of the return value must be untouched by the fold.
    let plain = streaming_sparse_running_stats(&chan_data, None, "test")?;
    assert_eq!(row_stats.to_vecs(), plain.to_vecs(), "row stats");
    Ok(())
}

#[test]
fn no_post_hoc_fold_of_row_statistics_recovers_npos_or_dispersion() -> anyhow::Result<()> {
    let chan = channelized();
    let chan_data = as_data(&chan, "row")?;
    let (row_stats, folded) =
        streaming_sparse_running_stats_folded(&chan_data, None, "test", &ROW_TO_GENE, N_GENES)?;

    let (r_npos, r_sum, _, r_std) = row_stats.to_vecs();
    let (f_npos, f_sum, _, f_std) = folded.to_vecs();

    let mut post_npos = vec![0.0f32; N_GENES];
    let mut post_sum = vec![0.0f32; N_GENES];
    let mut post_std = [0.0f32; N_GENES];
    for (r, &g) in ROW_TO_GENE.iter().enumerate() {
        post_npos[g as usize] += r_npos[r];
        post_sum[g as usize] += r_sum[r];
        post_std[g as usize] += r_std[r];
    }

    // `s1` is additive, so a post-hoc fold gets the sum right.
    assert_eq!(post_sum, f_sum, "sum is additive and must agree");

    // `npos` is not. A post-hoc fold over-counts by exactly the number of
    // cells that fire on BOTH of a gene's rows, so the error is a property of
    // the overlap and not of the gene count:
    //   gene 0  rows overlap on 3 cells  -> over-counts by 3
    //   gene 1  rows overlap on 2 cells  -> over-counts by 2
    //   gene 2  rows never co-fire       -> happens to agree
    //   gene 3  owns a single row        -> nothing to double count
    // That third case is the trap. A post-hoc fold is not uniformly wrong, it
    // is wrong in proportion to co-detection, which is precisely the quantity
    // a detection-based cutoff is trying to measure.
    let overlap = [3.0f32, 2.0, 0.0, 0.0];
    for g in 0..N_GENES {
        assert_eq!(
            post_npos[g] - f_npos[g],
            overlap[g],
            "gene {g}: post-hoc {post_npos:?} vs folded {f_npos:?}"
        );
    }

    // The dispersion cannot be recovered either, and the interesting part is
    // that the two natural post-hoc rules fail in OPPOSITE directions, so the
    // truth is sandwiched between them and neither can be repaired by a
    // constant:
    //
    //   sum of the per-row std      OVER-states  (std is subadditive)
    //   sqrt of summed variances    UNDER-states (the 2*cov cross term is gone)
    //
    // Gene 0's two rows co-fire on three of six cells, so its cross term is
    // large and the gap is wide.
    let pythagorean = (r_std[1] * r_std[1] + r_std[3] * r_std[3]).sqrt(); // rows 1 and 3 are gene 0
    assert!(
        pythagorean < f_std[0] && f_std[0] < post_std[0],
        "folded std {} should sit strictly between {} and {}",
        f_std[0],
        pythagorean,
        post_std[0]
    );
    Ok(())
}

#[test]
fn a_bucket_index_past_the_declared_count_is_an_error() -> anyhow::Result<()> {
    let chan_data = as_data(&channelized(), "row")?;
    let err = streaming_sparse_running_stats_folded(&chan_data, None, "test", &ROW_TO_GENE, 3)
        .err()
        .expect("bucket 3 is named but only 3 buckets declared");
    assert!(err.to_string().contains("only 3"), "{err}");

    let short = [0u32; 2];
    let err = streaming_sparse_running_stats_folded(&chan_data, None, "test", &short, N_GENES)
        .err()
        .expect("row_to_gene shorter than the matrix");
    assert!(err.to_string().contains("2 entries"), "{err}");
    Ok(())
}
