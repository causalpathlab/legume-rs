use super::*;
use candle_core::Device;

const P: usize = 7;
const D: usize = 40;

fn dev() -> Device {
    Device::Cpu
}

/// A deterministic `[P, D]` rate matrix with genuine zeros: the dense encoder
/// reads those too, so a fixture without them would not exercise the change.
fn rows() -> Mat {
    Mat::from_fn(P, D, |r, c| {
        if (r + c) % 3 == 0 {
            0.0
        } else {
            ((r * D + c) % 11) as f32
        }
    })
}

/// The same fixture in the `[D, P]` orientation the loader is handed.
fn rows_dp() -> Mat {
    rows().transpose()
}

fn level() -> DenseMaskedLevel {
    DenseMaskedLevel::from_mats(&rows_dp(), None, &rows_dp(), &vec![1.0f32; D], &dev()).unwrap()
}

fn draw(frac: f64) -> MaskedDraw {
    MaskedDraw {
        schedule: MaskSchedule::Fixed,
        mask_fraction: frac,
    }
}

/// `row id -> its visible row`, collected across every minibatch of an epoch.
fn visible_by_row(ep: &DenseMaskedEpoch<'_>) -> std::collections::HashMap<u32, Vec<f32>> {
    let mut out = std::collections::HashMap::new();
    for b in 0..ep.n_batches() {
        let mb = ep.batch(b).unwrap();
        let ids: Vec<u32> = mb.row_ids.to_vec1().unwrap();
        let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
        for (id, v) in ids.iter().zip(vis) {
            out.insert(*id, v);
        }
    }
    out
}

/// The draw is keyed on `(epoch seed, source row)` and nothing else.
///
/// A mask that depended on the batch size or on which rayon worker happened to
/// take a row would make a seeded run unreproducible in the one place a reader
/// would never look. Two epochs at different batch sizes, one of them forced
/// onto a single thread, must hand every row the identical hidden set.
#[test]
fn the_hidden_set_depends_on_the_seed_and_the_row_alone() {
    let lv = level();
    let d = draw(0.4);
    let a = visible_by_row(&lv.begin_epoch(1234, &d, 3).unwrap());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let b = pool.install(|| visible_by_row(&lv.begin_epoch(1234, &d, 5).unwrap()));

    assert_eq!(a.len(), P, "every source row appears once per epoch");
    for r in 0..P as u32 {
        assert_eq!(
            a.get(&r),
            b.get(&r),
            "row {r}: the hidden set moved with the batch size or the thread count"
        );
    }

    // ... and a different seed must actually move it, or the check above is vacuous.
    let c = visible_by_row(&lv.begin_epoch(99, &d, 3).unwrap());
    assert!(
        (0..P as u32).any(|r| a.get(&r) != c.get(&r)),
        "a different epoch seed drew the same mask"
    );
}

/// The mask is over the WHOLE gene axis, zeros included — that is the change.
#[test]
fn the_mask_covers_every_gene_not_just_the_expressed_ones() {
    let lv = level();
    let ep = lv.begin_epoch(7, &draw(0.5), P).unwrap();
    let mb = ep.batch(0).unwrap();
    let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
    let x: Vec<Vec<f32>> = mb.x_nd.to_vec2().unwrap();
    let mut hidden_zero = 0usize;
    let mut visible_zero = 0usize;
    for (vrow, xrow) in vis.iter().zip(&x) {
        assert_eq!(vrow.len(), D, "the mask spans the gene axis");
        for (&v, &val) in vrow.iter().zip(xrow) {
            assert!(v == 0.0 || v == 1.0, "the mask is an indicator");
            if val == 0.0 {
                if v == 0.0 {
                    hidden_zero += 1;
                } else {
                    visible_zero += 1;
                }
            }
        }
    }
    assert!(
        hidden_zero > 0 && visible_zero > 0,
        "zero-count genes must be maskable and visible alike: {hidden_zero} hidden, \
         {visible_zero} visible"
    );
}

/// A minibatch's rows are its source rows: input, target and mask all agree on
/// which pseudobulk each row is.
#[test]
fn a_minibatch_slices_the_resident_rows_by_row_id() {
    let lv = level();
    let src = rows();
    let ep = lv.begin_epoch(11, &draw(0.3), 2).unwrap();
    for b in 0..ep.n_batches() {
        let mb = ep.batch(b).unwrap();
        let ids: Vec<u32> = mb.row_ids.to_vec1().unwrap();
        let x: Vec<Vec<f32>> = mb.x_nd.to_vec2().unwrap();
        let y: Vec<Vec<f32>> = mb.target_nd.to_vec2().unwrap();
        for (i, &r) in ids.iter().enumerate() {
            for g in 0..D {
                assert_eq!(x[i][g], src[(r as usize, g)], "input row {r} gene {g}");
                assert_eq!(y[i][g], src[(r as usize, g)], "target row {r} gene {g}");
            }
        }
    }
}

/// `Uniform` draws the rate per row, so rows differ in how much is hidden.
#[test]
fn the_uniform_schedule_varies_the_rate_across_rows() {
    let lv = level();
    let d = MaskedDraw {
        schedule: MaskSchedule::Uniform { lo: 0.1, hi: 0.9 },
        mask_fraction: 0.4,
    };
    let ep = lv.begin_epoch(5, &d, P).unwrap();
    let mb = ep.batch(0).unwrap();
    let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
    let shares: Vec<f32> = vis
        .iter()
        .map(|r| r.iter().sum::<f32>() / D as f32)
        .collect();
    let lo = shares.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = shares.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(hi - lo > 0.1, "rates did not vary: {shares:?}");
}

/// The resident rows are exactly what the host transpose used to upload.
///
/// A level is handed the pseudobulk posterior in its natural `[D, P]` layout
/// and transposes it nowhere: nalgebra's column-major `[D, P]` buffer IS the
/// row-major `[P, D]` buffer, element `(d, p)` sitting at `d + p·D` in both. The
/// reference below is the construction this replaced — `DMatrix::transpose()`
/// on the host, then `to_tensor`'s transposed view made `contiguous()` on the
/// device — and the two must agree element for element, or every fit quietly
/// changed while the timing improved.
#[test]
fn the_resident_rows_equal_the_old_host_transpose() {
    use crate::data::loader_util::{upload_columns_as_rows, upload_to_device};

    // Planted `[D, P]`, every entry distinct, so a transposed read cannot pass
    // by symmetry.
    let m_dp = Mat::from_fn(D, P, |d, p| (d * P + p) as f32 + 0.5);

    let old = upload_to_device(&m_dp.transpose(), &dev()).unwrap();
    let new = upload_columns_as_rows(&m_dp, &dev()).unwrap();

    assert_eq!(old.dims(), &[P, D], "the reference is [P, D]");
    assert_eq!(new.dims(), old.dims(), "the upload changed shape");
    let old_rows: Vec<Vec<f32>> = old.to_vec2().unwrap();
    let new_rows: Vec<Vec<f32>> = new.to_vec2().unwrap();
    assert_eq!(
        new_rows, old_rows,
        "the uploaded rows are not the same tensor"
    );
    // ... and they are the transpose, not the matrix read row-wise.
    for p in 0..P {
        for d in 0..D {
            assert_eq!(new_rows[p][d], m_dp[(d, p)], "row {p}, gene {d}");
        }
    }
}

///////////////////////////////
// Fixed-count hidden draw   //
///////////////////////////////

/// `row id -> its hidden ids`, collected across every minibatch of an epoch.
fn hidden_by_row(ep: &DenseMaskedEpoch<'_>) -> std::collections::HashMap<u32, Vec<u32>> {
    let mut out = std::collections::HashMap::new();
    for b in 0..ep.n_batches() {
        let mb = ep.batch(b).unwrap();
        let ids: Vec<u32> = mb.row_ids.to_vec1().unwrap();
        let hid: Vec<Vec<u32>> = mb.hidden_ids.to_vec2().unwrap();
        for (id, h) in ids.iter().zip(hid) {
            out.insert(*id, h);
        }
    }
    out
}

/// Every row hides the SAME number of genes, `round(rate · D)`.
///
/// The per-gene Bernoulli this replaced gave a binomial count, so the hidden
/// set was ragged and could not be carried as ids at all. A fixed count is
/// what makes `[N, d_h]` a shape rather than an average.
#[test]
fn every_row_hides_exactly_the_rounded_fraction() {
    let lv = level();
    for frac in [0.25f64, 0.4, 0.5, 0.75] {
        let want = (frac * D as f64).round() as usize;
        let ep = lv.begin_epoch(3, &draw(frac), P).unwrap();
        let mb = ep.batch(0).unwrap();
        assert_eq!(
            mb.hidden_ids.dims(),
            &[P, want],
            "rate {frac}: the hidden block is [N, round(rate·D)]"
        );
        assert!(
            mb.hidden_weight.is_none(),
            "rate {frac}: a fixed count needs no padding weight"
        );
        let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
        for (n, v) in vis.iter().enumerate() {
            let hidden = v.iter().filter(|&&x| x == 0.0).count();
            assert_eq!(
                hidden, want,
                "rate {frac}, row {n}: hid {hidden}, want {want}"
            );
        }
    }
}

/// The visible mask and the hidden ids are two views of ONE draw: visible is 1
/// everywhere except exactly at the hidden ids. Two draws that merely agreed on
/// average would let the encoder read a gene the decoder is scored on.
#[test]
fn the_visible_mask_and_the_hidden_ids_are_one_draw() {
    let lv = level();
    let ep = lv.begin_epoch(17, &draw(0.4), 3).unwrap();
    for b in 0..ep.n_batches() {
        let mb = ep.batch(b).unwrap();
        let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
        let hid: Vec<Vec<u32>> = mb.hidden_ids.to_vec2().unwrap();
        for (n, (v, h)) in vis.iter().zip(&hid).enumerate() {
            let mut want = vec![1.0f32; D];
            for &g in h {
                want[g as usize] = 0.0;
            }
            assert_eq!(
                v, &want,
                "row {n}: the mask and the ids are different draws"
            );
        }
    }
}

/// Floyd draws WITHOUT replacement: a row's ids are strictly ascending, so no
/// gene is scored twice and the block is in gather order.
#[test]
fn a_rows_hidden_ids_are_strictly_ascending() {
    let lv = level();
    let ep = lv.begin_epoch(23, &draw(0.5), P).unwrap();
    let mb = ep.batch(0).unwrap();
    for (n, h) in mb.hidden_ids.to_vec2::<u32>().unwrap().iter().enumerate() {
        assert!(
            h.windows(2).all(|w| w[0] < w[1]),
            "row {n} repeats or unsorts an id: {h:?}"
        );
    }
}

/// The hidden ids, like the mask they agree with, are keyed on `(epoch seed,
/// source row)` and nothing else.
#[test]
fn the_hidden_ids_depend_on_the_seed_and_the_row_alone() {
    let lv = level();
    let d = draw(0.4);
    let a = hidden_by_row(&lv.begin_epoch(1234, &d, 3).unwrap());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let b = pool.install(|| hidden_by_row(&lv.begin_epoch(1234, &d, 5).unwrap()));

    assert_eq!(a.len(), P, "every source row appears once per epoch");
    for r in 0..P as u32 {
        assert_eq!(
            a.get(&r),
            b.get(&r),
            "row {r}: the hidden ids moved with the batch size or the thread count"
        );
    }
    let c = hidden_by_row(&lv.begin_epoch(99, &d, 3).unwrap());
    assert!(
        (0..P as u32).any(|r| a.get(&r) != c.get(&r)),
        "a different epoch seed drew the same hidden ids"
    );
}

/// A row is never fully hidden or fully visible: the count is clamped to
/// `[1, D-1]`, so the encoder always has something to read and the decoder
/// always has something to answer for.
#[test]
fn the_hidden_count_is_clamped_away_from_both_ends() {
    let lv = level();
    for (frac, want) in [(0.0f64, 1usize), (1.0, D - 1), (0.001, 1), (0.999, D - 1)] {
        let ep = lv.begin_epoch(41, &draw(frac), P).unwrap();
        let mb = ep.batch(0).unwrap();
        assert_eq!(
            mb.hidden_ids.dims(),
            &[P, want],
            "rate {frac} must clamp to {want} hidden"
        );
        let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
        for v in &vis {
            assert_eq!(v.iter().filter(|&&x| x == 0.0).count(), want);
        }
    }
}

/// `Uniform` draws the row's rate first, so the COUNT differs across rows. The
/// block is then padded to the widest row and carries the weight that makes the
/// short rows exact.
#[test]
fn the_uniform_schedule_varies_the_hidden_count_across_rows() {
    let lv = level();
    let d = MaskedDraw {
        schedule: MaskSchedule::Uniform { lo: 0.1, hi: 0.9 },
        mask_fraction: 0.4,
    };
    let ep = lv.begin_epoch(5, &d, P).unwrap();
    let mb = ep.batch(0).unwrap();
    let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
    let counts: Vec<usize> = vis
        .iter()
        .map(|r| r.iter().filter(|&&x| x == 0.0).count())
        .collect();
    let lo = *counts.iter().min().unwrap();
    let hi = *counts.iter().max().unwrap();
    assert!(
        hi > lo,
        "the per-row rate did not move the count: {counts:?}"
    );
    assert_eq!(
        mb.hidden_ids.dims(),
        &[P, hi],
        "the block is as wide as the widest row"
    );
    let w = mb
        .hidden_weight
        .as_ref()
        .expect("a ragged draw carries the weight that makes the short rows exact")
        .to_vec2::<f32>()
        .unwrap();
    for (n, (row, &c)) in w.iter().zip(&counts).enumerate() {
        assert_eq!(
            row.iter().sum::<f32>(),
            c as f32,
            "row {n}: the weight must count exactly the row's hidden genes"
        );
        assert!(row[..c].iter().all(|&x| x == 1.0) && row[c..].iter().all(|&x| x == 0.0));
    }
    // The padded slots still index real hidden genes, so a gather is in range
    // and the mask agrees with the weighted ids.
    let hid: Vec<Vec<u32>> = mb.hidden_ids.to_vec2().unwrap();
    for (n, (h, &c)) in hid.iter().zip(&counts).enumerate() {
        for &g in h {
            assert_eq!(vis[n][g as usize], 0.0, "row {n}: id {g} is not hidden");
        }
        assert!(h[..c].windows(2).all(|x| x[0] < x[1]), "row {n}: {h:?}");
    }
}
