use super::*;

/// 10x barcode whitelist collision rate measured between unrelated
/// GSE139369 backends: 6–21 shared barcodes out of ~6k. Anything the
/// planner does must stay far above this floor.
const WHITELIST_NOISE: usize = 12;

fn bc(prefix: &str, n: usize) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{prefix}{i:06}-1").into_boxed_str())
        .collect()
}

/// `n_shared` barcodes drawn from `a`'s namespace, the rest from `b`'s —
/// the shape of a whitelist collision between unrelated samples.
fn bc_noisy(shared_prefix: &str, own_prefix: &str, n: usize, n_shared: usize) -> Vec<Box<str>> {
    let mut v = bc(shared_prefix, n_shared);
    v.extend((0..n - n_shared).map(|i| format!("{own_prefix}{i:06}-1").into_boxed_str()));
    v
}

fn genes(n: usize) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("GENE{i}").into_boxed_str())
        .collect()
}

fn adt() -> Vec<Box<str>> {
    ["ADT-CD14", "ADT-CD19", "ADT-CD3", "ADT-CD4", "ADT-CD8A"]
        .iter()
        .map(|s| (*s).into())
        .collect()
}

fn peaks(n: usize) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("chr1:{}-{}", i * 5000, i * 5000 + 5000).into_boxed_str())
        .collect()
}

fn axes(file: &str, rows: Vec<Box<str>>, cols: Vec<Box<str>>) -> FileAxes {
    FileAxes {
        file: file.into(),
        rows,
        cols,
    }
}

/// The GSE139369 CITE-seq shape: four donor cohorts, each an
/// RNA + ADT pair on identical barcodes, with whitelist noise between
/// cohorts. Expect 4 groups × 2 modalities, named from the filenames.
#[test]
fn citeseq_cohorts_group_by_barcode_and_split_by_modality() {
    let cohorts = ["BMMC_D1T1", "BMMC_D1T2", "PBMC_D4T1", "PBMC_D4T2"];
    let mut input = Vec::new();
    for (k, c) in cohorts.iter().enumerate() {
        let cells = bc_noisy("SHARED", &format!("C{k}"), 6000, WHITELIST_NOISE);
        input.push(axes(
            &format!("scRNA_{c}.zarr.zip"),
            genes(20000),
            cells.clone(),
        ));
        input.push(axes(&format!("scADT_{c}.zarr.zip"), adt(), cells));
    }

    let plan = plan_from_axes(&input).unwrap().expect("multiome expected");

    assert_eq!(plan.group_sizes, vec![2, 2, 2, 2]);
    assert_eq!(plan.n_groups(), 4);
    // Files are reordered so each group is contiguous.
    for (g, c) in cohorts.iter().enumerate() {
        assert_eq!(&*plan.files[2 * g], &*format!("scRNA_{c}.zarr.zip"));
        assert_eq!(&*plan.files[2 * g + 1], &*format!("scADT_{c}.zarr.zip"));
        assert_eq!(&*plan.group[2 * g], *c);
        assert_eq!(&*plan.group[2 * g + 1], *c);
    }
    assert_eq!(&*plan.modality[0], "scRNA");
    assert_eq!(&*plan.modality[1], "scADT");
    // Four groups ⇒ barcodes must be namespaced by cohort, or the 12
    // whitelist collisions would merge cells across donors.
    let suffix = plan.barcode_suffix().expect("multi-group needs suffixes");
    assert_eq!(suffix[0].as_deref(), Some("BMMC_D1T1"));
    assert_eq!(suffix[7].as_deref(), Some("PBMC_D4T2"));
    assert_eq!(plan.n_bridge_cells, Some(4 * 6000));
}

/// One sample, two modalities: no cross-group collisions to guard, so no
/// barcode suffix — that keeps the modality-presence auto-batch alive.
#[test]
fn single_group_gets_no_barcode_suffix() {
    let cells = bc("AAA", 500);
    let input = vec![
        axes("rna.zarr", genes(2000), cells.clone()),
        axes("atac.zarr", peaks(3000), cells),
    ];
    let plan = plan_from_axes(&input).unwrap().expect("multiome expected");
    assert_eq!(plan.group_sizes, vec![2]);
    assert!(!plan.barcode_tagged);
    assert_eq!(&*plan.modality[0], "rna");
    assert_eq!(&*plan.modality[1], "atac");
}

/// Patchy pairing (per-modality QC dropped cells) still reads as one group.
#[test]
fn patchy_pairing_is_still_one_group() {
    let input = vec![
        axes("rna.zarr", genes(2000), bc("AAA", 1000)),
        axes("atac.zarr", peaks(3000), bc("AAA", 600)),
    ];
    let plan = plan_from_axes(&input).unwrap().expect("multiome expected");
    assert_eq!(plan.group_sizes, vec![2]);
    assert_eq!(plan.n_bridge_cells, Some(600));
}

/// Several samples of ONE modality is the ordinary single-modality load.
/// Auto-detection must keep its hands off it.
#[test]
fn many_samples_one_modality_is_not_multiome() {
    let input = vec![
        axes(
            "s1.zarr",
            genes(2000),
            bc_noisy("SHARED", "A", 900, WHITELIST_NOISE),
        ),
        axes(
            "s2.zarr",
            genes(2000),
            bc_noisy("SHARED", "B", 900, WHITELIST_NOISE),
        ),
        axes(
            "s3.zarr",
            genes(2000),
            bc_noisy("SHARED", "C", 900, WHITELIST_NOISE),
        ),
    ];
    assert!(plan_from_axes(&input).unwrap().is_none());
}

/// Two modalities but no matched cells (GSE139369's scRNA vs scATAC: the
/// assays were run on different donors). Without a bridge there is no
/// evidence of a paired design, so fall back to today's behaviour.
#[test]
fn disjoint_modalities_without_matched_cells_is_not_multiome() {
    let input = vec![
        axes(
            "scRNA_BMMC_D1T1.zarr",
            genes(20000),
            bc_noisy("SHARED", "R", 6000, 6),
        ),
        axes(
            "scATAC_BMMC_D5T1.zarr",
            peaks(37000),
            bc_noisy("SHARED", "A", 62000, 6),
        ),
    ];
    assert!(plan_from_axes(&input).unwrap().is_none());
}

/// Two gene panels that overlap heavily are one modality, even when the
/// row counts differ (a different reference build, not another assay).
#[test]
fn overlapping_gene_panels_are_one_modality() {
    let mut short = genes(2000);
    short.truncate(1400);
    let input = vec![
        axes("a.zarr", genes(2000), bc("A", 500)),
        axes("b.zarr", short, bc("B", 500)),
    ];
    assert!(plan_from_axes(&input).unwrap().is_none());
}

/// Two files of the same modality glued onto one sample is ambiguous:
/// which one owns the modality row block? Fail with both names.
#[test]
fn duplicate_modality_within_a_group_is_an_error() {
    let cells = bc("AAA", 500);
    let err = plan_from_axes(&[
        axes("rna_a.zarr", genes(2000), cells.clone()),
        axes("rna_b.zarr", genes(2000), cells.clone()),
        axes("atac.zarr", peaks(3000), cells),
    ])
    .unwrap_err()
    .to_string();
    assert!(err.contains("rna_a.zarr"), "{err}");
    assert!(err.contains("rna_b.zarr"), "{err}");
}

/// A modality present in only some cohorts still plans; the cohort with
/// no partner is its own group.
#[test]
fn unpaired_cohort_rides_along_as_its_own_group() {
    let paired = bc("P", 800);
    let input = vec![
        axes("scRNA_S1.zarr", genes(2000), paired.clone()),
        axes("scADT_S1.zarr", adt(), paired),
        axes("scRNA_S2.zarr", genes(2000), bc("Q", 700)),
    ];
    let plan = plan_from_axes(&input).unwrap().expect("multiome expected");
    assert_eq!(plan.group_sizes, vec![2, 1]);
    assert_eq!(&*plan.files[2], "scRNA_S2.zarr");
    assert_eq!(&*plan.modality[2], "scRNA");
}

/// Filenames that share nothing fall back to positional tags rather than
/// inventing a label.
#[test]
fn unnameable_labels_fall_back_to_positional_tags() {
    let a = bc("A", 400);
    let b = bc("B", 400);
    let input = vec![
        axes("x.zarr", genes(2000), a.clone()),
        axes("y.zarr", peaks(2000), a),
        axes("x2.zarr", genes(2000), b.clone()),
        axes("y2.zarr", peaks(2000), b),
    ];
    let plan = plan_from_axes(&input).unwrap().expect("multiome expected");
    assert_eq!(plan.group_sizes, vec![2, 2]);
    assert_eq!(&*plan.group[0], "g0");
    assert_eq!(&*plan.group[2], "g1");
}

/// Labels reach feature names as `{name}/{modality}` and barcodes as
/// `{bc}@{group}`, so neither may carry those separators.
#[test]
fn labels_are_free_of_name_separators() {
    let cells = bc("A", 300);
    let other = bc("B", 300);
    let input = vec![
        axes("s@1/rna.zarr", genes(500), cells.clone()),
        axes("s@1/atac.zarr", peaks(500), cells),
        axes("s@2/rna.zarr", genes(500), other.clone()),
        axes("s@2/atac.zarr", peaks(500), other),
    ];
    let plan = plan_from_axes(&input).unwrap().expect("multiome expected");
    for lab in plan.modality.iter().chain(plan.group.iter()) {
        assert!(!lab.contains('/') && !lab.contains('@'), "bad label {lab}");
    }
}

#[test]
fn fewer_than_two_files_is_never_multiome() {
    let input = vec![axes("rna.zarr", genes(100), bc("A", 10))];
    assert!(plan_from_axes(&input).unwrap().is_none());
    assert!(plan_from_axes(&[]).unwrap().is_none());
}
