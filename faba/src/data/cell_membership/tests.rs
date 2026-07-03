use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_membership_file() -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "AAACCTGAGAAACCAT\tT_cell").unwrap();
    writeln!(file, "AAACCTGAGCCCAATT\tB_cell").unwrap();
    writeln!(file, "AAACCTGCATACTCTT\tMonocyte").unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn test_exact_matching() {
    let file = create_test_membership_file();
    let membership = CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, false).unwrap();

    assert_eq!(membership.num_cells(), 3);

    let barcode = CellBarcode::Barcode("AAACCTGAGAAACCAT".into());
    assert_eq!(membership.matches_barcode(&barcode), Some("T_cell".into()));

    // No prefix matching
    let barcode_with_suffix = CellBarcode::Barcode("AAACCTGAGAAACCAT-1".into());
    assert_eq!(membership.matches_barcode(&barcode_with_suffix), None);
}

#[test]
fn test_prefix_matching() {
    let file = create_test_membership_file();
    let membership = CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

    let barcode = CellBarcode::Barcode("AAACCTGAGCCCAATT".into());
    assert_eq!(membership.matches_barcode(&barcode), Some("B_cell".into()));

    let barcode_with_suffix = CellBarcode::Barcode("AAACCTGAGAAACCAT-1".into());
    assert_eq!(
        membership.matches_barcode(&barcode_with_suffix),
        Some("T_cell".into())
    );

    let unknown = CellBarcode::Barcode("ZZZZZZZZZZZZZZZ".into());
    assert_eq!(membership.matches_barcode(&unknown), None);
}

#[test]
fn test_missing_barcode() {
    let file = create_test_membership_file();
    let membership = CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

    let missing = CellBarcode::Missing;
    assert_eq!(membership.matches_barcode(&missing), None);
}

#[test]
fn test_caching() {
    let file = create_test_membership_file();
    let membership = CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

    let barcode = CellBarcode::Barcode("AAACCTGAGAAACCAT-1".into());

    let result1 = membership.matches_barcode(&barcode);
    assert_eq!(result1, Some("T_cell".into()));

    let result2 = membership.matches_barcode(&barcode);
    assert_eq!(result2, Some("T_cell".into()));

    assert_eq!(result1, result2);
}

#[test]
fn test_stats() {
    let file = create_test_membership_file();
    let membership = CellMembership::from_file(file.path().to_str().unwrap(), 0, 1, true).unwrap();

    let barcode1 = CellBarcode::Barcode("AAACCTGAGAAACCAT".into());
    let barcode2 = CellBarcode::Barcode("UNKNOWN".into());
    let missing = CellBarcode::Missing;

    membership.matches_barcode(&barcode1);
    membership.matches_barcode(&barcode2);
    membership.matches_barcode(&missing);

    let (matched, total) = membership.match_stats();
    assert_eq!(matched, 1);
    assert_eq!(total, 3);
}
