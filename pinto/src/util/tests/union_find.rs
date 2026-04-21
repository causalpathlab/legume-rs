use crate::util::union_find::UnionFind;

#[test]
fn test_union_find_basic() {
    let mut uf = UnionFind::new(5);
    assert_eq!(uf.find(0), 0);
    assert_eq!(uf.find(4), 4);
    assert_eq!(uf.size(0), 1);

    let rep = uf.union(0, 1);
    assert_eq!(uf.find(0), uf.find(1));
    assert_eq!(uf.size(rep), 2);

    let rep2 = uf.union(2, 3);
    assert_eq!(uf.size(rep2), 2);

    let rep3 = uf.union(0, 3);
    assert_eq!(uf.find(0), uf.find(3));
    assert_eq!(uf.find(1), uf.find(2));
    assert_eq!(uf.size(rep3), 4);
}

#[test]
fn test_union_find_flatten() {
    // Build a tall chain via repeated unions that favor asymmetric ranks.
    let mut uf = UnionFind::new(6);
    uf.union(0, 1);
    uf.union(2, 3);
    uf.union(0, 2);
    uf.union(4, 5);
    uf.union(0, 4);

    uf.flatten();
    let root = uf.find(0);
    for i in 0..6 {
        assert_eq!(uf.parent(i), root, "flatten should make parent(i) == root");
    }
}
