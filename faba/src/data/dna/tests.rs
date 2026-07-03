use super::*;

#[test]
fn test_base_count_add_get() {
    let mut count = DnaBaseCount::new();
    count.add(Some(&Dna::A), 5);
    count.add(Some(&Dna::T), 3);
    count.add(Some(&Dna::G), 7);
    count.add(Some(&Dna::C), 2);

    assert_eq!(count.count_a(), 5);
    assert_eq!(count.count_t(), 3);
    assert_eq!(count.count_g(), 7);
    assert_eq!(count.count_c(), 2);

    assert_eq!(count.get(Some(&Dna::A)), 5);
    assert_eq!(count.get(Some(&Dna::T)), 3);
    assert_eq!(count.get(None), 0);

    // Add more to existing
    count.add(Some(&Dna::A), 10);
    assert_eq!(count.count_a(), 15);
}

#[test]
fn test_add_assign() {
    let mut a = DnaBaseCount::new();
    a.add(Some(&Dna::A), 10);
    a.add(Some(&Dna::T), 5);

    let mut b = DnaBaseCount::new();
    b.add(Some(&Dna::A), 3);
    b.add(Some(&Dna::G), 7);

    a += &b;
    assert_eq!(a.count_a(), 13);
    assert_eq!(a.count_t(), 5);
    assert_eq!(a.count_g(), 7);
    assert_eq!(a.count_c(), 0);
}
