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
    DenseMaskedLevel::from_mats(&rows_dp(), None, &rows_dp(), &[1.0f32; D], &dev()).unwrap()
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

/// At a rate the CLI admits, a row is never fully hidden or fully visible: the
/// encoder always has something to read and the decoder always has something to
/// answer for, and the count is exactly `round(rate · D)` with no reinterpretation.
#[test]
fn an_admissible_rate_hides_the_rounded_fraction_and_no_more() {
    let lv = level();
    for (frac, want) in [(0.05f64, 2usize), (0.4, 16), (0.95, D - 2)] {
        let ep = lv.begin_epoch(41, &draw(frac), P).unwrap();
        let mb = ep.batch(0).unwrap();
        assert_eq!(
            mb.hidden_ids.dims(),
            &[P, want],
            "rate {frac} must hide {want} genes"
        );
        let vis: Vec<Vec<f32>> = mb.visible_nd.to_vec2().unwrap();
        for v in &vis {
            assert_eq!(v.iter().filter(|&&x| x == 0.0).count(), want);
            assert!(v.contains(&1.0), "rate {frac} left nothing visible");
        }
    }
}

/// A degenerate rate is a caller bug, not a draw to be silently rounded up to
/// one gene. The CLI refuses it (`--mask-fraction` and the uniform bounds); the
/// loader asserts it, so a caller that reaches past the CLI hears about it.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "open interval (0, 1)")]
fn a_rate_of_zero_trips_the_invariant() {
    let lv = level();
    let _ = lv.begin_epoch(41, &draw(0.0), P).unwrap().batch(0);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "open interval (0, 1)")]
fn a_rate_of_one_trips_the_invariant() {
    let lv = level();
    let _ = lv.begin_epoch(41, &draw(1.0), P).unwrap().batch(0);
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

/// The `[N, D]` visible row the loader used to fill on the host and upload
/// every minibatch: ones, with a zero at every hidden id.
fn host_visible(hidden: &[Vec<u32>], d: usize) -> Vec<Vec<f32>> {
    hidden
        .iter()
        .map(|row| {
            let mut v = vec![1f32; d];
            for &g in row {
                v[g as usize] = 0.0;
            }
            v
        })
        .collect()
}

/// `hidden_ids` already says everything the visible mask says, so the mask is
/// derived from it ON THE DEVICE rather than built `[N, D]` on the host and
/// uploaded every step.
///
/// The derivation must not care about the order of a row's ids (a planted
/// block can be unsorted) nor about a repeat (the `Uniform` schedule pads a
/// short row with its own last hidden id), so all three shapes are planted
/// here and checked against the host build kept above as the reference.
#[test]
fn the_visible_mask_is_derived_from_the_hidden_ids() {
    let hidden = vec![
        vec![0u32, 3, 7], // ascending, as a draw produces it
        vec![9u32, 1, 4], // unsorted: a scatter must not care
        vec![5u32, 5, 5], // a padded row repeats its own last hidden id
    ];
    let d = 12;
    let flat: Vec<u32> = hidden.iter().flatten().copied().collect();
    let ids = Tensor::from_vec(flat, (hidden.len(), 3), &dev()).unwrap();
    let got: Vec<Vec<f32>> = visible_from_hidden(&ids, d).unwrap().to_vec2().unwrap();
    assert_eq!(got, host_visible(&hidden, d));
}

/// A level handed ONE matrix twice holds ONE device buffer.
///
/// Without a batch-adjusted target the trainer's `mixed` IS the target, and
/// uploading it as two resident `[P, D]` tensors doubles the level's device
/// footprint for two copies of the same numbers. `Tensor` is `Arc`-backed, so
/// the reused clone carries the same `id()`; two genuinely separate uploads
/// cannot.
#[test]
fn one_matrix_handed_in_twice_is_uploaded_once() {
    let m = rows_dp();
    let mean = vec![1.0f32; D];
    let shared = DenseMaskedLevel::from_mats(&m, None, &m, &mean, &dev()).unwrap();
    assert_eq!(
        shared.input_pd.id(),
        shared.target_pd.id(),
        "the same Mat handed in as input and target must upload once"
    );

    // Two distinct matrices — equal in content, so only the identity can tell
    // them apart — still get one buffer each.
    let other = rows_dp();
    let distinct = DenseMaskedLevel::from_mats(&m, None, &other, &mean, &dev()).unwrap();
    assert_ne!(
        distinct.input_pd.id(),
        distinct.target_pd.id(),
        "two separate Mats must not be collapsed onto one buffer"
    );

    // And either way the rows read back as the caller's matrix.
    let want: Vec<Vec<f32>> = rows()
        .row_iter()
        .map(|r| r.iter().copied().collect())
        .collect();
    for lv in [&shared, &distinct] {
        let got: Vec<Vec<f32>> = lv.target_pd.to_vec2().unwrap();
        assert_eq!(got, want);
    }
}

/// A valid rate can round onto an edge of a small axis. The CLI refuses the
/// degenerate RATES; this is the guard on the degenerate ROUNDING, and it must
/// hold in release builds, where a debug assertion would not.
#[test]
fn a_valid_rate_never_hides_everything_or_nothing() {
    use super::hidden_count;
    assert_eq!(
        hidden_count(2, 0.4),
        1,
        "0.4 of two genes rounds to one, not zero"
    );
    assert_eq!(
        hidden_count(5, 0.9),
        4,
        "0.9 of five genes rounds to all five; one must stay visible"
    );
    assert_eq!(
        hidden_count(3, 0.5),
        2,
        "plain rounding when it lands inside the axis"
    );
    assert_eq!(
        hidden_count(34_008, 0.4),
        13_603,
        "the production case is untouched by the bound"
    );
}
