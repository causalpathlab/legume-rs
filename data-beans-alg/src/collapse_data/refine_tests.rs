use super::*;

#[test]
fn build_reproject_offsets_extracts_extra_bits() {
    // 4 pb-samples (one cell each); 4-bit fine codes differing only in the
    // high 2 bits. finest-first dims: child = 4 bits, parent = 2 bits.
    let fine_codes = vec![0b0000usize, 0b0100, 0b1000, 0b1100];
    let pb_cells = vec![vec![0usize], vec![1], vec![2], vec![3]];
    let dims = vec![4usize, 2];
    let off = build_reproject_offsets(&fine_codes, &pb_cells, &dims);
    assert_eq!(off.len(), 2);
    // level 0: extra = (code >> 2) & 0b11 — semantic high bits, not a
    // positional index, so identical extra bits map to the same offset.
    assert_eq!(off[0], vec![0, 1, 2, 3]);
    // coarsest level has no parent → empty (unused).
    assert!(off[1].is_empty());
}

#[test]
fn build_reproject_offsets_bounded_by_extra_bit_width() {
    // child 5 bits, parent 3 bits → extra ∈ [0, 2^2). Codes that share the
    // high 2 bits collapse to one offset regardless of low bits.
    let fine_codes = vec![0b00000usize, 0b00111, 0b01000, 0b01011, 0b11111];
    let pb_cells: Vec<Vec<usize>> = (0..5).map(|i| vec![i]).collect();
    let dims = vec![5usize, 3];
    let off = build_reproject_offsets(&fine_codes, &pb_cells, &dims);
    // extra = (code >> 3) & 0b11
    assert_eq!(off[0], vec![0, 0, 1, 1, 3]);
    assert!(off[0].iter().all(|&x| x < 4));
}
