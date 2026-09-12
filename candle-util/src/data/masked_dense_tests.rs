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

fn level() -> DenseMaskedLevel {
    DenseMaskedLevel::from_mats(&rows(), None, &rows(), &vec![1.0f32; D], &dev()).unwrap()
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
