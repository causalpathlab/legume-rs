//! The projection has to obey the caller's query-axis rules. It once built
//! its own defaults, which made it deaf to `--ablate-features`: the latent
//! was fitted on the very genes it was then scored against, so an ablated
//! svd run reported a reconstruction wearing a prediction's name.

use super::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use std::collections::HashSet;

fn backend(dir: &tempfile::TempDir, genes: &[&str], n_cells: usize) -> anyhow::Result<SparseIoVec> {
    let path = dir.path().join("q.zarr");
    // Every gene carries counts in every cell, so hiding some genuinely
    // removes evidence rather than dropping empty rows.
    let triplets: Vec<(u64, u64, f32)> = (0..genes.len())
        .flat_map(|g| (0..n_cells).map(move |c| (g as u64, c as u64, 1.0 + (g + c) as f32)))
        .collect();
    let mut b = create_sparse_from_triplets(
        &triplets,
        (genes.len(), n_cells, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    let gn: Vec<Box<str>> = genes.iter().map(|s| Box::from(*s)).collect();
    let cn: Vec<Box<str>> = (0..n_cells).map(|i| format!("cell_{i}").into()).collect();
    b.register_row_names_vec(&gn);
    b.register_column_names_vec(&cn);
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None)?;
    Ok(v)
}

#[test]
fn hidden_features_are_withheld_from_the_projection() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let genes = ["a", "b", "c", "d"];
    let train: Vec<Box<str>> = genes.iter().map(|s| Box::from(*s)).collect();
    let data = backend(&dir, &genes, 3)?;
    let u_dk = Mat::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 0.7, 0.3, 0.2, 0.8]);

    let plain = project_onto_dictionary(
        &data,
        &train,
        &u_dk,
        1e4,
        &QueryNameOpts::default(),
        None,
        "plain",
    )?;

    let hiding = QueryNameOpts {
        hide: Some(std::sync::Arc::new(
            ["c", "d"]
                .iter()
                .map(|s| Box::from(*s))
                .collect::<HashSet<Box<str>>>(),
        )),
        ..Default::default()
    };
    let ablated = project_onto_dictionary(&data, &train, &u_dk, 1e4, &hiding, None, "ablated")?;

    assert_eq!(plain.shape(), ablated.shape());
    let moved = (&plain - &ablated).norm();
    assert!(
        moved > 1e-6,
        "hiding half the genes left the latent unchanged ({moved}): the projection \
         is ignoring the hide set, so an ablated run is scored on genes it saw"
    );
    Ok(())
}

/// The coverage floor is the caller's too — it used to be hardcoded to zero,
/// so `--min-gene-overlap` could not refuse a thin query.
#[test]
fn the_callers_coverage_floor_is_enforced() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let train: Vec<Box<str>> = ["a", "b", "c", "d"].iter().map(|s| Box::from(*s)).collect();
    // Only one of the model's four genes is present here.
    let data = backend(&dir, &["a", "x", "y"], 2)?;
    let u_dk = Mat::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 0.7, 0.3, 0.2, 0.8]);

    let lenient = QueryNameOpts::default();
    assert!(project_onto_dictionary(&data, &train, &u_dk, 1e4, &lenient, None, "ok").is_ok());

    let strict = QueryNameOpts {
        min_overlap: 0.9,
        ..Default::default()
    };
    assert!(
        project_onto_dictionary(&data, &train, &u_dk, 1e4, &strict, None, "strict").is_err(),
        "a 90% floor must refuse a query carrying one of four model genes"
    );
    Ok(())
}
