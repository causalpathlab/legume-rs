//! Contract of the pre-trained gene-embedding loader, stated as equalities.
//!
//! The loader aligns an external `gene x H` dictionary to the run's own gene
//! axis, and every defect it guards against is an ordering or identity
//! mistake: rows silently following the dictionary's order instead of the
//! run's, an unmatched gene left at an arbitrary init instead of the closest
//! matched profile, or a co-embed artifact (track-suffixed rows) accepted as
//! if it were a dictionary. Each test pins one of those with exact values.

use super::pretrained::{load_pretrained_gene_embedding, InitKind, PretrainedArgs};
use crate::util::common::Mat;
use auxiliary_data::feature_names::FeatureNameKind;
use matrix_util::traits::IoOps;

/// Write a small dictionary parquet: rows named by `genes`, `h` columns of
/// values `base + row * 10 + col`, so every row is unique and recognizable.
fn write_dictionary(
    dir: &tempfile::TempDir,
    tag: &str,
    genes: &[Box<str>],
    h: usize,
    base: f32,
) -> anyhow::Result<String> {
    let d = genes.len();
    let m = Mat::from_fn(d, h, |r, c| base + (r * 10 + c) as f32);
    let cols: Vec<Box<str>> = (0..h).map(|c| format!("H{c}").into()).collect();
    let path = dir
        .path()
        .join(format!("{tag}.parquet"))
        .to_string_lossy()
        .into_owned();
    m.to_parquet_with_names(&path, (Some(genes), Some("gene")), Some(&cols))?;
    Ok(path)
}

fn names(list: &[&str]) -> Vec<Box<str>> {
    list.iter().map(|s| (*s).into()).collect()
}

/// The run's gene axis, deliberately in a different order from the
/// dictionary, with an UNMATCHED gene first so the matched positions are
/// shifted: an implementation that writes dictionary rows at their compact
/// (loader) index instead of their target index cannot pass this.
#[test]
fn rows_follow_the_runs_gene_axis_not_the_dictionarys() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let dict_genes = names(&["G3", "G1", "G2"]);
    let path = write_dictionary(&dir, "dict", &dict_genes, 4, 100.0)?;

    let run_genes = names(&["G9", "G1", "G2", "G3"]);
    let profiles = Mat::zeros(4, 2);
    let out = load_pretrained_gene_embedding(PretrainedArgs {
        dictionary_path: &path,
        bias_path: None,
        gene_names: &run_genes,
        name_kind: FeatureNameKind::Exact,
        gene_profiles: &profiles,
    })?;

    assert_eq!(out.h, 4);
    assert_eq!(out.e_gene.nrows(), 4);
    assert_eq!(out.matched, vec![false, true, true, true]);
    // Dictionary row for G1 is dictionary-row 1 => values 100 + 10 + col,
    // and it must land on run-axis row 1 (after the unmatched G9), not row 0.
    for c in 0..4 {
        assert_eq!(out.e_gene[(1, c)], 100.0 + 10.0 + c as f32, "G1 col {c}");
        assert_eq!(out.e_gene[(2, c)], 100.0 + 20.0 + c as f32, "G2 col {c}");
        assert_eq!(out.e_gene[(3, c)], 100.0 + c as f32, "G3 col {c}");
    }
    assert!(out.b_gene.iter().all(|&b| b == 0.0), "no bias file => zeros");
    Ok(())
}

/// An unmatched gene must be seeded from the matched gene whose count profile
/// it resembles most, and the report must say which gene and how strongly.
#[test]
fn unmatched_gene_takes_the_closest_matched_profile_neighbor() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let dict_genes = names(&["G1", "G2"]);
    let path = write_dictionary(&dir, "dict", &dict_genes, 3, 0.0)?;

    let run_genes = names(&["G1", "G2", "G9"]);
    // G9's profile is parallel to G2's (cosine 1) and orthogonal to G1's.
    let profiles = Mat::from_row_slice(
        3,
        2,
        &[
            1.0, 0.0, // G1
            0.0, 1.0, // G2
            0.0, 2.0, // G9
        ],
    );
    let out = load_pretrained_gene_embedding(PretrainedArgs {
        dictionary_path: &path,
        bias_path: None,
        gene_names: &run_genes,
        name_kind: FeatureNameKind::Exact,
        gene_profiles: &profiles,
    })?;

    assert_eq!(out.matched, vec![true, true, false]);
    // G9's row equals G2's dictionary row (0 + 10 + col).
    for c in 0..3 {
        assert_eq!(out.e_gene[(2, c)], 10.0 + c as f32, "G9 col {c}");
    }
    let rec = &out.records[2];
    assert_eq!(rec.init, InitKind::Neighbor);
    assert_eq!(rec.neighbor_gene.as_deref(), Some("G2"));
    assert!((rec.cosine - 1.0).abs() < 1e-6, "cosine {}", rec.cosine);
    // Matched genes report themselves as matched, with no neighbor.
    assert_eq!(out.records[0].init, InitKind::Matched);
    assert!(out.records[0].neighbor_gene.is_none());
    Ok(())
}

/// A track-suffixed co-embed table is not a dictionary. Reject it by name,
/// before any alignment, and say which rows offended.
#[test]
fn track_suffixed_rows_are_rejected_with_the_offending_name() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let dict_genes = names(&["G1/count/spliced", "G1/count/unspliced"]);
    let path = write_dictionary(&dir, "dict", &dict_genes, 2, 0.0)?;

    let run_genes = names(&["G1"]);
    let profiles = Mat::zeros(1, 2);
    let err = load_pretrained_gene_embedding(PretrainedArgs {
        dictionary_path: &path,
        bias_path: None,
        gene_names: &run_genes,
        name_kind: FeatureNameKind::Exact,
        gene_profiles: &profiles,
    })
    .err()
    .expect("a co-embed table must be rejected")
    .to_string();
    assert!(
        err.contains("G1/count/spliced"),
        "error must name an offending row: {err}"
    );
    Ok(())
}

/// An unmatched gene whose profile is all zero has no closest neighbor; it
/// takes the matched-row mean and reports no neighbor gene.
#[test]
fn zero_profile_gene_takes_the_matched_mean() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let dict_genes = names(&["G1", "G2"]);
    let path = write_dictionary(&dir, "dict", &dict_genes, 2, 0.0)?;

    let run_genes = names(&["G1", "G2", "G9"]);
    let profiles = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let out = load_pretrained_gene_embedding(PretrainedArgs {
        dictionary_path: &path,
        bias_path: None,
        gene_names: &run_genes,
        name_kind: FeatureNameKind::Exact,
        gene_profiles: &profiles,
    })?;

    // Mean of rows (0,1) and (10,11) is (5,6).
    assert_eq!(out.e_gene[(2, 0)], 5.0);
    assert_eq!(out.e_gene[(2, 1)], 6.0);
    let rec = &out.records[2];
    assert_eq!(rec.init, InitKind::Neighbor);
    assert!(rec.neighbor_gene.is_none(), "mean seeding names no neighbor");
    Ok(())
}

/// A bias parquet fills matched genes; unmatched genes stay at zero.
#[test]
fn bias_loads_for_matched_genes_and_zeros_elsewhere() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let dict_genes = names(&["G1", "G2"]);
    let path = write_dictionary(&dir, "dict", &dict_genes, 2, 0.0)?;
    let bias = Mat::from_row_slice(2, 1, &[7.0, 8.0]);
    let bias_path = dir
        .path()
        .join("bias.parquet")
        .to_string_lossy()
        .into_owned();
    bias.to_parquet_with_names(
        &bias_path,
        (Some(&dict_genes), Some("gene")),
        Some(&names(&["bias"])),
    )?;

    let run_genes = names(&["G2", "G9"]);
    let profiles = Mat::from_row_slice(2, 2, &[1.0, 0.0, 1.0, 0.0]);
    let out = load_pretrained_gene_embedding(PretrainedArgs {
        dictionary_path: &path,
        bias_path: Some(&bias_path),
        gene_names: &run_genes,
        name_kind: FeatureNameKind::Exact,
        gene_profiles: &profiles,
    })?;

    assert_eq!(out.b_gene, vec![8.0, 0.0]);
    Ok(())
}

/// No overlap at all is a hard error, not an empty model.
#[test]
fn zero_matched_genes_is_a_hard_error() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let dict_genes = names(&["G1", "G2"]);
    let path = write_dictionary(&dir, "dict", &dict_genes, 2, 0.0)?;

    let run_genes = names(&["G8", "G9"]);
    let profiles = Mat::zeros(2, 2);
    assert!(load_pretrained_gene_embedding(PretrainedArgs {
        dictionary_path: &path,
        bias_path: None,
        gene_names: &run_genes,
        name_kind: FeatureNameKind::Exact,
        gene_profiles: &profiles,
    })
    .is_err());
    Ok(())
}

/// The freeze is a post-step restore, not a gradient mask: after an optimizer
/// step has moved every row, restoring must put frozen rows back exactly and
/// leave the trainable rows where the step left them.
#[test]
fn restore_puts_frozen_rows_back_and_leaves_trainable_rows_alone() -> anyhow::Result<()> {
    use super::pretrained::restore_frozen_rows;
    use candle_util::candle_core::{Device, Tensor, Var};

    let dev = Device::Cpu;
    let init = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &dev)?;
    let var = Var::from_tensor(&init)?;
    let fixed = var.as_tensor().copy()?;
    // Rows 0 and 2 frozen, row 1 trainable.
    let mask = Tensor::from_vec(vec![1.0f32, 0.0, 1.0], (3, 1), &dev)?;

    // Simulate an optimizer step that moved every value.
    let moved = (var.as_tensor() + 10.0)?;
    var.set(&moved)?;

    restore_frozen_rows(&var, &fixed, &mask)?;
    let got = var.as_tensor().to_vec2::<f32>()?;
    assert_eq!(got[0], vec![1.0, 2.0], "frozen row 0 restored");
    assert_eq!(got[2], vec![5.0, 6.0], "frozen row 2 restored");
    assert_eq!(got[1], vec![13.0, 14.0], "trainable row 1 keeps its step");
    Ok(())
}
