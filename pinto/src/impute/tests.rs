//! What the profile projection must guarantee: a cell whose counts sit on
//! one community's genes lands on that community, gene matching goes
//! through the canonicalizer (not raw row names), and cells the model
//! cannot see keep a zero row instead of a made-up assignment.

use super::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};

fn make_data(
    dir: &tempfile::TempDir,
    name: &str,
    gene_names: &[&str],
    triplets: &[(u64, u64, f32)],
    n_cells: usize,
) -> anyhow::Result<SparseIoVec> {
    let path = dir.path().join(name);
    let mut backend = create_sparse_from_triplets(
        triplets,
        (gene_names.len(), n_cells, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    let genes: Vec<Box<str>> = gene_names.iter().map(|s| Box::from(*s)).collect();
    let cells: Vec<Box<str>> = (0..n_cells).map(|i| format!("cell_{i}").into()).collect();
    backend.register_row_names_vec(&genes);
    backend.register_column_names_vec(&cells);
    let mut data_vec = SparseIoVec::new();
    data_vec.push(std::sync::Arc::from(backend), None)?;
    Ok(data_vec)
}

/// Two communities with disjoint expression: C0 lives on {a, b}, C1 on
/// {c, d}.
fn disjoint_profiles() -> (Mat, Vec<Box<str>>) {
    let profiles = Mat::from_row_slice(
        4,
        2,
        &[
            5.0, 0.0, // a
            5.0, 0.0, // b
            0.0, 4.0, // c
            0.0, 6.0, // d
        ],
    );
    let genes: Vec<Box<str>> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    (profiles, genes)
}

#[test]
fn pure_cells_land_on_their_community() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, genes) = disjoint_profiles();
    // cell_0 expresses only C0 genes, cell_1 only C1 genes, cell_2 both.
    let data = make_data(
        &dir,
        "q.zarr",
        &["a", "b", "c", "d"],
        &[
            (0, 0, 5.0),
            (1, 0, 3.0),
            (2, 1, 4.0),
            (3, 1, 2.0),
            (0, 2, 3.0),
            (2, 2, 3.0),
        ],
        3,
    )?;
    let prop = project_profile_propensity(
        &data,
        &profiles,
        &genes,
        &auxiliary_data::feature_names::FeatureNameKind::Exact,
        100,
        None,
        "test",
    )?;
    assert!(prop[(0, 0)] > 0.99, "pure C0 cell: {}", prop[(0, 0)]);
    assert!(prop[(1, 1)] > 0.99, "pure C1 cell: {}", prop[(1, 1)]);
    assert!(
        prop[(2, 0)] > 0.2 && prop[(2, 1)] > 0.2,
        "mixed cell stays mixed: {:?}",
        (prop[(2, 0)], prop[(2, 1)])
    );
    let row_sum: f32 = prop.row(2).iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-4, "rows are proportions");
    Ok(())
}

#[test]
fn gene_matching_goes_through_the_canonicalizer() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, genes) = disjoint_profiles();
    // Query rows carry `<ensembl>_<symbol>` names; the Gene kind resolves
    // them onto the model's bare symbols.
    let data = make_data(
        &dir,
        "ens.zarr",
        &["ENSG01_a", "ENSG02_b", "ENSG03_c", "ENSG04_d"],
        &[(0, 0, 5.0), (1, 0, 3.0)],
        1,
    )?;
    let prop = project_profile_propensity(
        &data,
        &profiles,
        &genes,
        &auxiliary_data::feature_names::FeatureNameKind::Gene { delim: '_' },
        100,
        None,
        "test",
    )?;
    assert!(prop[(0, 0)] > 0.99, "canonicalized match: {}", prop[(0, 0)]);
    Ok(())
}

#[test]
fn a_cell_off_the_model_axis_keeps_a_zero_row() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, genes) = disjoint_profiles();
    // cell_0 expresses only a gene the model never saw; cell_1 is normal.
    let data = make_data(
        &dir,
        "off.zarr",
        &["zzz", "a"],
        &[(0, 0, 9.0), (1, 1, 4.0)],
        2,
    )?;
    let prop = project_profile_propensity(
        &data,
        &profiles,
        &genes,
        &auxiliary_data::feature_names::FeatureNameKind::Exact,
        100,
        None,
        "test",
    )?;
    assert!(prop.row(0).iter().all(|&x| x == 0.0));
    assert!(prop[(1, 0)] > 0.99);
    Ok(())
}

#[test]
fn no_shared_gene_is_an_error_not_a_uniform_guess() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (profiles, genes) = disjoint_profiles();
    let data = make_data(&dir, "none.zarr", &["x", "y"], &[(0, 0, 1.0)], 1)?;
    assert!(project_profile_propensity(
        &data,
        &profiles,
        &genes,
        &auxiliary_data::feature_names::FeatureNameKind::Exact,
        10,
        None,
        "test",
    )
    .is_err());
    Ok(())
}

#[test]
fn explicit_reference_data_bypasses_the_manifest() -> anyhow::Result<()> {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        args: ImputeArgs,
    }
    let wrap = Wrap::try_parse_from([
        "impute",
        "q.zarr",
        "--model",
        "no/such/model",
        "-o",
        "out",
        "--reference-data",
        "ref.zarr",
    ])?;
    let files = resolve_reference_data(&wrap.args)?;
    assert_eq!(files, vec![Box::<str>::from("ref.zarr")]);
    Ok(())
}

#[test]
fn a_missing_manifest_names_the_flag_as_the_way_out() -> anyhow::Result<()> {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        args: ImputeArgs,
    }
    let wrap = Wrap::try_parse_from(["impute", "q.zarr", "--model", "no/such/model", "-o", "out"])?;
    let err = resolve_reference_data(&wrap.args).unwrap_err();
    assert!(err.to_string().contains("--reference-data"), "{err}");
    Ok(())
}
