//! Load a pre-trained per-gene embedding (and optional per-gene bias)
//! from parquet and strictly intersect its row axis against a caller's
//! target feature axis.
//!
//! Used by `senna gbe / topic / cell-embedded-topic` to freeze the
//! gene-side parameter table (`E_feat` in gbe, ρ in the ETM topic models)
//! so cells train on a shared, pre-fit gene-relation space.
//!
//! Source formats supported via [`FrozenLoadArgs`]:
//! - **gbe**: `{prefix}.dictionary.parquet` + `{prefix}.feature_bias.parquet`
//!   (gene × H plus gene × 1 bias).
//! - **topic / cell-embedded-topic**: `{prefix}.feature_embedding.parquet`
//!   alone — bias defaults to zeros, which is what the topic models use
//!   internally (no per-gene additive bias on ρ).
//!
//! Name resolution goes through [`FeatureNameKind`] so `TGFB1` and
//! `ENSG00000105329_TGFB1` resolve to the same row.

use crate::feature_names::FeatureNameKind;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;
use rustc_hash::{FxHashMap, FxHashSet};

/// Loaded + aligned frozen feature side ready to hand off to a candle
/// engine (`graph-embedding-util` or one of the topic-model encoders).
///
/// Rows are reordered to follow the *target* axis. `keep_target_indices`
/// records which positions in the caller's target feature axis survived
/// the intersection — the caller MUST restrict its data (triplets,
/// encoder D, decoder β) to these indices, otherwise the row order
/// disagrees with the embedding rows.
pub struct FrozenFeatureHost {
    /// `[|keep|, H]`, rows in the same order as `keep_target_indices`.
    pub e_feat: DMatrix<f32>,
    /// `[|keep|]`. Zeros when no `bias_path` was given.
    pub b_feat: Vec<f32>,
    /// Indices into the *target* feature axis that matched a source row.
    /// Length equals `e_feat.nrows()`.
    pub keep_target_indices: Vec<usize>,
    pub h: usize,
}

pub struct FrozenLoadArgs<'a> {
    /// Path to the `[D_src, H]` parquet (gbe `dictionary.parquet` or
    /// topic `feature_embedding.parquet`). Row column 0 is the gene name.
    pub dictionary_path: &'a str,
    /// Optional path to a `[D_src, 1]` per-gene bias parquet (gbe
    /// `feature_bias.parquet`). `None` → bias filled with zeros, which
    /// matches the topic models' implicit "no per-gene bias on ρ".
    pub bias_path: Option<&'a str>,
    /// Caller's feature axis (e.g. `unified.feature_names` for gbe;
    /// the topic models' `gene_names`). Output rows follow this order
    /// after dropping unmatched entries.
    pub target_feature_names: &'a [Box<str>],
    /// Per-name canonicalization rule applied to both source and target
    /// names before intersection. [`FeatureNameKind::Exact`] for strict
    /// matching; [`FeatureNameKind::Gene { delim: '_' }`] is the typical
    /// choice for scRNA gene IDs.
    pub name_kind: FeatureNameKind,
}

pub fn load_frozen_feature_host(args: FrozenLoadArgs) -> anyhow::Result<FrozenFeatureHost> {
    let dict = <DMatrix<f32> as IoOps>::from_parquet(args.dictionary_path)?;
    let n_src = dict.rows.len();
    let h = dict.mat.ncols();
    anyhow::ensure!(
        h > 0 && dict.mat.nrows() == n_src,
        "{}: malformed dictionary (rows={}, mat dims={}x{})",
        args.dictionary_path,
        n_src,
        dict.mat.nrows(),
        h
    );

    let src_bias: Vec<f32> = match args.bias_path {
        None => vec![0.0; n_src],
        Some(p) => {
            let bias = <DMatrix<f32> as IoOps>::from_parquet(p)?;
            anyhow::ensure!(
                bias.rows == dict.rows,
                "{} row names disagree with {} (both files must come from the same training run)",
                p,
                args.dictionary_path
            );
            anyhow::ensure!(
                bias.mat.ncols() == 1,
                "{}: expected 1 data column (bias), got {}",
                p,
                bias.mat.ncols()
            );
            (0..n_src).map(|i| bias.mat[(i, 0)]).collect()
        }
    };

    let mut src_by_canon: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut src_dupes = 0usize;
    for (i, name) in dict.rows.iter().enumerate() {
        let canon = args.name_kind.canonicalize(name);
        if src_by_canon.insert(canon, i).is_some() {
            src_dupes += 1;
        }
    }
    if src_dupes > 0 {
        log::warn!(
            "{}: {} source rows had duplicate canonical names — kept first occurrence",
            args.dictionary_path,
            src_dupes
        );
    }

    let mut keep_target_indices = Vec::new();
    let mut keep_src_indices = Vec::new();
    for (target_i, name) in args.target_feature_names.iter().enumerate() {
        let canon = args.name_kind.canonicalize(name);
        if let Some(&src_i) = src_by_canon.get(&canon) {
            keep_target_indices.push(target_i);
            keep_src_indices.push(src_i);
        }
    }
    anyhow::ensure!(
        !keep_target_indices.is_empty(),
        "No feature names matched between {} (n={}) and target axis (n={}) under {:?} \
         — check the gene-name kind (Exact / Gene / Locus / Mixed) and source axis",
        args.dictionary_path,
        n_src,
        args.target_feature_names.len(),
        args.name_kind
    );

    let unique_src_used: FxHashSet<usize> = keep_src_indices.iter().copied().collect();
    log::info!(
        "Frozen feature side from {}: {}/{} target features matched (H={}, {} of {} source rows reused, kind={:?})",
        args.dictionary_path,
        keep_target_indices.len(),
        args.target_feature_names.len(),
        h,
        unique_src_used.len(),
        n_src,
        args.name_kind
    );

    let k = keep_target_indices.len();
    let mut e_feat = DMatrix::<f32>::zeros(k, h);
    let mut b_feat = Vec::with_capacity(k);
    for (out_i, &src_i) in keep_src_indices.iter().enumerate() {
        for j in 0..h {
            e_feat[(out_i, j)] = dict.mat[(src_i, j)];
        }
        b_feat.push(src_bias[src_i]);
    }

    Ok(FrozenFeatureHost {
        e_feat,
        b_feat,
        keep_target_indices,
        h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_util::traits::IoOps;

    fn write_test_parquet(
        path: &str,
        rows: &[&str],
        row_axis: &str,
        cols: &[&str],
        data: &DMatrix<f32>,
    ) {
        let row_names: Vec<Box<str>> = rows.iter().map(|s| (*s).into()).collect();
        let col_names: Vec<Box<str>> = cols.iter().map(|s| (*s).into()).collect();
        data.to_parquet_with_names(path, (Some(&row_names), Some(row_axis)), Some(&col_names))
            .unwrap();
    }

    #[test]
    fn strict_intersection_drops_unmatched_and_preserves_target_order() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();

        // Source: 4 genes × H=3. Source row "ENSG_DROP" has no target match.
        let src = DMatrix::<f32>::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, // TGFB1
                4.0, 5.0, 6.0, // MYC
                7.0, 8.0, 9.0, // ENSG_DROP (unmatched)
                10.0, 11.0, 12.0, // TP53
            ],
        );
        write_test_parquet(
            &dict_path,
            &["TGFB1", "MYC", "ENSG_DROP", "TP53"],
            "gene",
            &["h0", "h1", "h2"],
            &src,
        );

        // Target: 5 genes; "FOO" and "BAR" don't appear in source.
        let target: Vec<Box<str>> = ["FOO", "TP53", "TGFB1", "BAR", "MYC"]
            .iter()
            .map(|s| (*s).into())
            .collect();

        let host = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Exact,
        })
        .unwrap();

        // Kept target indices = positions of TP53, TGFB1, MYC in target order.
        assert_eq!(host.keep_target_indices, vec![1, 2, 4]);
        assert_eq!(host.h, 3);
        assert_eq!(host.e_feat.nrows(), 3);
        assert_eq!(host.b_feat, vec![0.0, 0.0, 0.0]);

        // Row 0 of e_feat should be source row for TP53 (= source row 3).
        assert_eq!(host.e_feat[(0, 0)], 10.0);
        assert_eq!(host.e_feat[(0, 2)], 12.0);
        // Row 1: TGFB1 → source row 0.
        assert_eq!(host.e_feat[(1, 0)], 1.0);
        // Row 2: MYC → source row 1.
        assert_eq!(host.e_feat[(2, 1)], 5.0);
    }

    #[test]
    fn gene_canon_matches_across_delim_variants() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();

        // Source uses ENSG-prefixed; target uses bare gene symbols.
        let src = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(
            &dict_path,
            &["ENSG00000105329_TGFB1", "ENSG00000141510_TP53"],
            "gene",
            &["h0", "h1"],
            &src,
        );
        let target: Vec<Box<str>> = ["TP53", "TGFB1"].iter().map(|s| (*s).into()).collect();

        let host = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Gene { delim: '_' },
        })
        .unwrap();

        assert_eq!(host.keep_target_indices, vec![0, 1]);
        // Row 0 (target TP53) ← source row 1.
        assert_eq!(host.e_feat[(0, 0)], 3.0);
        // Row 1 (target TGFB1) ← source row 0.
        assert_eq!(host.e_feat[(1, 0)], 1.0);
    }

    #[test]
    fn empty_intersection_errors() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();
        let src = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(&dict_path, &["A", "B"], "gene", &["h0", "h1"], &src);
        let target: Vec<Box<str>> = ["C", "D"].iter().map(|s| (*s).into()).collect();
        let result = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Exact,
        });
        let err = match result {
            Ok(_) => panic!("expected empty-intersection error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("No feature names matched"));
    }

    #[test]
    fn bias_loaded_when_provided() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();
        let bias_path = dir.path().join("b.parquet").to_str().unwrap().to_string();
        let src = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(&dict_path, &["A", "B"], "gene", &["h0", "h1"], &src);
        let bias = DMatrix::<f32>::from_row_slice(2, 1, &[0.5, -0.3]);
        write_test_parquet(&bias_path, &["A", "B"], "gene", &["bias"], &bias);

        let target: Vec<Box<str>> = ["B", "A"].iter().map(|s| (*s).into()).collect();
        let host = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: Some(&bias_path),
            target_feature_names: &target,
            name_kind: FeatureNameKind::Exact,
        })
        .unwrap();
        // Row 0 of output = target "B" = source row 1.
        assert_eq!(host.b_feat, vec![-0.3, 0.5]);
    }
}
