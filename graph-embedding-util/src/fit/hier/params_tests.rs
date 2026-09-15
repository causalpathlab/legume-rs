use super::*;

#[test]
fn init_is_seeded_and_biases_are_zero() {
    let a = HierParams::new(3, 2, 5, 4, 7);
    let b = HierParams::new(3, 2, 5, 4, 7);
    let c = HierParams::new(3, 2, 5, 4, 8);
    assert_eq!(a.e_u, b.e_u);
    assert_ne!(a.e_u, c.e_u);
    assert_eq!(a.e_u.len(), 12);
    assert_eq!(a.mu.len(), 8);
    assert_eq!(a.r.len(), 20);
    assert!(a.b_m.iter().all(|&x| x == 0.0) && a.b_g.iter().all(|&x| x == 0.0));
    // N(0, 0.1²): entries are small
    assert!(a.e_u.iter().all(|x| x.abs() < 1.0));
}

#[test]
fn adagrad_first_step_moves_by_lr_times_sign_and_later_steps_shrink() {
    let mut opt = RowAdagrad::new(1, 0.5);
    let mut row = vec![0f32, 0.0];
    let g = [3f32, -4.0]; // mean(g²) = 12.5, sqrt = 3.5355
    opt.update(0, &mut row, &g);
    assert!((row[0] - (-0.5 * 3.0 / 3.5355)).abs() < 1e-4);
    assert!((row[1] - (0.5 * 4.0 / 3.5355)).abs() < 1e-4);
    let before = row.clone();
    opt.update(0, &mut row, &g); // acc doubles → denominator ×√2
    let d0 = (row[0] - before[0]).abs();
    assert!((d0 - 0.5 * 3.0 / (2.0f32 * 12.5).sqrt()).abs() < 1e-4);
}

#[test]
fn a_zero_gradient_leaves_the_row_and_accumulator_untouched() {
    let mut opt = RowAdagrad::new(2, 0.1);
    let mut row = vec![1f32, 2.0];
    opt.update(1, &mut row, &[0.0, 0.0]);
    assert_eq!(row, vec![1.0, 2.0]);
    assert_eq!(opt.acc[1], 0.0);
}

#[test]
fn the_non_base_tracks_start_at_zero_and_leave_the_base_tables_untouched() {
    let (n_u, n_m, n_g, h) = (3usize, 2usize, 5usize, 4usize);
    let plain = HierParams::new(n_u, n_m, n_g, h, 7);
    let tracked = HierParams::new_tracked(n_u, n_m, n_g, 3, h, 7);
    // same seed, same draws: the base tables are identical
    assert_eq!(plain.e_u, tracked.e_u);
    assert_eq!(plain.mu, tracked.mu);
    assert_eq!(plain.r, tracked.r);
    assert_eq!(plain.b_m, tracked.b_m);
    assert_eq!(plain.b_g, tracked.b_g);
    // one offset table per non-base track, all zero
    assert!(plain.offsets.is_empty());
    assert!(plain.offset(0).is_none());
    assert!(plain.offset(1).is_none());
    assert_eq!(tracked.offsets.len(), 2);
    assert!(tracked.offset(0).is_none());
    for t in 1..3 {
        let o = tracked.offset(t).expect("a non-base track has an offset");
        assert_eq!(o.d_mu.len(), n_m * h);
        assert_eq!(o.d_b_m.len(), n_m);
        assert_eq!(o.d_r.len(), n_g * h);
        assert_eq!(o.d_b_g.len(), n_g);
        assert!(o.d_mu.iter().all(|&x| x == 0.0));
        assert!(o.d_b_m.iter().all(|&x| x == 0.0));
        assert!(o.d_r.iter().all(|&x| x == 0.0));
        assert!(o.d_b_g.iter().all(|&x| x == 0.0));
    }
    assert!(tracked.offset(3).is_none());
}
