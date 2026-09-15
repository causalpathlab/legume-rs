//! Which read a masked checkpoint takes, decided from its metadata alone.

use super::MaskedRead;

/// A model trained window-free records no context size, and reads dense.
#[test]
fn no_recorded_window_reads_dense() {
    let read = MaskedRead::resolve(None, None).expect("a window-free model needs nothing else");
    assert!(
        matches!(read, MaskedRead::Dense),
        "a model with no recorded window must read every gene"
    );
    // A stale shortlist alongside a window-free model does not resurrect the
    // window: the metadata decides, not what happens to be on disk.
    let w = vec![1.0f32; 4];
    let read = MaskedRead::resolve(None, Some(&w)).unwrap();
    assert!(matches!(read, MaskedRead::Dense));
}

/// An OLD model records its window and keeps scoring exactly as it did: top-K
/// per cell, ranked by the shortlist weights it was trained with.
#[test]
fn a_recorded_window_reads_the_indexed_path() {
    let w = vec![0.5f32, 1.0, 2.0, 0.25];
    match MaskedRead::resolve(Some(3), Some(&w)).expect("an old model loads") {
        MaskedRead::Windowed {
            context_size,
            shortlist_weights,
        } => {
            assert_eq!(context_size, 3);
            assert_eq!(shortlist_weights, w.as_slice());
        }
        MaskedRead::Dense => panic!("a recorded window must take the indexed path"),
    }
}

/// A recorded window with no weights on disk is a broken model, not a reason to
/// silently read it some other way: scoring it dense would be a different model.
#[test]
fn a_recorded_window_without_its_weights_is_refused_by_name() {
    let msg = MaskedRead::resolve(Some(3), None)
        .expect_err("a windowed model cannot be read without its shortlist")
        .to_string();
    assert!(
        msg.contains("shortlist_weights"),
        "the error must name the missing file; got: {msg}"
    );
}

/// `enc_context_size` is what decides the read, so it has to survive the round
/// trip through `model.json` — a `None` that came back as anything else would
/// score a window-free model as if it had a window, and the other way round.
#[test]
fn the_field_that_decides_the_read_round_trips() {
    use crate::topic::model_metadata::TopicModelMetadata;

    let base = TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_MASKED_VAE.into(),
        decoder_types: vec!["nb".into()],
        decoder_weights: vec![1.0],
        n_features_encoder: 6,
        n_features_full: 6,
        n_topics: 2,
        encoder_hidden: vec![8],
        num_levels: 1,
        level_decoder_dims: vec![6],
        adj_method: "residual".into(),
        has_coarsening: false,
        embedding_dim: Some(4),
        enc_context_size: None,
        theta_mean: None,
        n_train_cells: None,
        n_gene_modules: None,
        query_rank: None,
    };
    let dir = tempfile::tempdir().unwrap();

    let dense = dir.path().join("dense").to_string_lossy().into_owned();
    base.save(&dense).unwrap();
    let back = TopicModelMetadata::load(&dense).unwrap();
    assert_eq!(
        back.enc_context_size, None,
        "a window-free model records none"
    );
    assert!(matches!(
        MaskedRead::resolve(back.enc_context_size, None).unwrap(),
        MaskedRead::Dense
    ));

    let windowed = dir.path().join("windowed").to_string_lossy().into_owned();
    let mut m = base;
    m.enc_context_size = Some(512);
    m.save(&windowed).unwrap();
    assert_eq!(
        TopicModelMetadata::load(&windowed)
            .unwrap()
            .enc_context_size,
        Some(512),
        "an old model's window must come back as itself"
    );
}

//////////////////////////////////////////////////////////////
// The dense block, and how many of them run at once         //
//////////////////////////////////////////////////////////////

/// One block of the window-free read equals the straightforward dense
/// conversion of the same sparse columns.
///
/// Three things a scatter gets wrong quietly, all planted here: an **empty
/// column** (a cell with nothing in it must come out as a zero row, not as the
/// previous cell's row or a shifted one), a column whose **nonzeros are out of
/// row order** (the block is built by scattering, so nothing may depend on the
/// order they arrive in), and a **remap that sends two query genes to one
/// training gene** (the fill has to ADD, or the second silently replaces the
/// first). The reference below is the obvious `[n, D_train]` fill, written out
/// so the comparison is against arithmetic rather than against another copy of
/// the same scatter.
#[test]
fn a_dense_block_equals_the_straightforward_conversion() {
    use crate::candle_core::Device;
    use nalgebra_sparse::CscMatrix;

    const D_QUERY: usize = 6;
    const D_TRAIN: usize = 4;
    const N: usize = 4;

    // Column 1 is empty; column 2's rows are given out of order; column 3
    // repeats nothing but lands on a remapped gene.
    let col_offsets = vec![0, 3, 3, 6, 8];
    let row_indices = vec![
        0, 2, 5, /* col 2, unsorted: */ 4, 1, 0, /* col 3: */ 3, 5,
    ];
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_dn =
        CscMatrix::try_from_unsorted_csc_data(D_QUERY, N, col_offsets, row_indices, values.clone())
            .expect("a valid CSC block");

    // Two query genes (0 and 5) land on training gene 0; query gene 4 is not on
    // the training axis at all and must be dropped, not folded into gene 0.
    let remap: Vec<Option<usize>> = vec![Some(0), Some(1), Some(2), Some(3), None, Some(0)];

    // `None` means the columns are already on the training axis, so it is read
    // at the block's own width; the remap is what crosses axes.
    for (map, d_out) in [(None, D_QUERY), (Some(remap.as_slice()), D_TRAIN)] {
        // The reference: walk the triplets and add, exactly as the definition
        // reads, with no tensor in sight.
        let mut want = vec![vec![0.0f32; d_out]; N];
        for (r, c, v) in x_dn.triplet_iter() {
            let g = match map {
                Some(m) => m[r],
                None => Some(r),
            };
            if let Some(g) = g {
                want[c][g] += v;
            }
        }

        let got: Vec<Vec<f32>> = super::csc_block_to_dense(&x_dn, d_out, map, &Device::Cpu)
            .expect("the block builds")
            .to_vec2()
            .expect("[n, D_out]");

        assert_eq!(got.len(), N, "one row per cell in the block");
        assert_eq!(
            got,
            want,
            "the block is not the dense conversion (remap: {})",
            map.is_some()
        );
        assert!(
            got[1].iter().all(|&v| v == 0.0),
            "the empty column must come out as a zero row"
        );
    }

    // Reading an off-axis block without a remap is a wiring mistake, not
    // something to index past the end of the buffer for.
    let msg = super::csc_block_to_dense(&x_dn, D_TRAIN, None, &Device::Cpu)
        .expect_err("D_query > D_train with no remap is not a readable block")
        .to_string();
    assert!(
        msg.contains("gene_remap"),
        "the refusal must name what is missing; got: {msg}"
    );

    // ... and the planted cases are actually present, or the test proves nothing.
    let nnz_per_col: Vec<usize> = (0..N).map(|j| x_dn.col(j).nnz()).collect();
    assert!(nnz_per_col.contains(&0), "no empty column was planted");
    let with_remap: Vec<Vec<f32>> =
        super::csc_block_to_dense(&x_dn, D_TRAIN, Some(&remap), &Device::Cpu)
            .unwrap()
            .to_vec2()
            .unwrap();
    assert_eq!(
        with_remap[0][0], 4.0,
        "query genes 0 and 5 of cell 0 must ADD onto training gene 0"
    );
}

/// The dense read holds a handful of blocks at once; the windowed read is left
/// on the device rule.
///
/// A dense block's `[n, D]` working set is the encoder chain over it, and its
/// own matmul already asks rayon for every core — so running one per thread
/// costs more than it buys (measured: 395 s of summed block wall wide open
/// against 19 s held at 8). The windowed `[n, K]` block never had the problem
/// and must not be slowed down by the fix.
#[test]
fn the_dense_read_bounds_how_many_blocks_run_at_once() {
    use crate::candle_core::Device;

    let cpu = Device::Cpu;
    let threads = rayon::current_num_threads();
    let dense = super::masked_block_concurrency(super::MaskedRead::Dense, &cpu, 100, 34008);
    assert!(
        (1..=super::DENSE_BLOCKS_IN_FLIGHT).contains(&dense),
        "a whole-transcriptome dense block must be capped, got {dense}"
    );

    let w = vec![1.0f32; 8];
    let windowed = super::masked_block_concurrency(
        super::MaskedRead::Windowed {
            context_size: 1000,
            shortlist_weights: &w,
        },
        &cpu,
        100,
        34008,
    );
    assert_eq!(
        windowed, threads,
        "the windowed read keeps the device rule: one block per thread"
    );
}

/////////////////////////////////////////////////
// The query head is no longer wired anywhere  //
/////////////////////////////////////////////////

/// A model trained with a query decoder cannot be scored by this build.
///
/// Its `dec_query.*` tensors ride in the checkpoint, and `VarMap::load` fills
/// only the vars a rebuild has already registered — anything else is skipped in
/// silence. With the head unwired nothing registers them, so scoring such a
/// model would quietly use a different rate from the one it was trained on.
/// Refuse by name instead.
#[test]
fn a_model_with_a_query_head_is_refused_by_name() {
    use crate::topic::model_metadata::ensure_query_head_not_wired;

    assert!(ensure_query_head_not_wired(None).is_ok());
    assert!(
        ensure_query_head_not_wired(Some(0)).is_ok(),
        "rank 0 is no head at all"
    );
    let msg = ensure_query_head_not_wired(Some(32))
        .expect_err("a model whose weights would be silently skipped must be refused")
        .to_string();
    for needle in ["query", "32"] {
        assert!(
            msg.contains(needle),
            "the message must name {needle}; got: {msg}"
        );
    }
}

/// `query_rank` stays deserialisable: an old model records it, this build reads
/// it back unchanged (so it can refuse), and a model written now has none.
#[test]
fn the_query_rank_round_trips_with_and_without_a_value() {
    use crate::topic::model_metadata::TopicModelMetadata;

    let base = TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_MASKED_VAE.into(),
        decoder_types: vec!["nb".into()],
        decoder_weights: vec![1.0],
        n_features_encoder: 6,
        n_features_full: 6,
        n_topics: 2,
        encoder_hidden: vec![8],
        num_levels: 1,
        level_decoder_dims: vec![6],
        adj_method: "residual".into(),
        has_coarsening: false,
        embedding_dim: Some(4),
        enc_context_size: None,
        theta_mean: None,
        n_train_cells: None,
        n_gene_modules: None,
        query_rank: None,
    };
    let dir = tempfile::tempdir().unwrap();

    let plain = dir.path().join("plain").to_string_lossy().into_owned();
    base.save(&plain).unwrap();
    assert_eq!(
        TopicModelMetadata::load(&plain).unwrap().query_rank,
        None,
        "a model this build writes has no query head"
    );

    let old = dir.path().join("old").to_string_lossy().into_owned();
    let mut m = base;
    m.query_rank = Some(32);
    m.save(&old).unwrap();
    let back = TopicModelMetadata::load(&old).unwrap();
    assert_eq!(
        back.query_rank,
        Some(32),
        "the field has to survive, or the refusal cannot fire"
    );
    assert!(crate::topic::model_metadata::ensure_query_head_not_wired(back.query_rank).is_err());
}

/// ONE seeded hold-out draw, shared by both evaluation arms.
///
/// The two arms threshold different buffers — the dense arm the `[n, D]` block,
/// the windowed arm the `[n, K]` pack — but they drew the mask the same way,
/// spelled out twice. Both spellings are kept inline here as the reference: the
/// helper has to reproduce them exactly, RNG stream included, or an old
/// checkpoint's held-out number moves.
#[test]
fn both_arms_draw_the_same_holdout_mask() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    // A planted buffer with zeros scattered through it: a zero is never held
    // out and must never consume a draw, which is what pins the RNG stream.
    let values: Vec<f32> = (0..37)
        .map(|i| {
            if i % 4 == 0 {
                0.0
            } else {
                (i % 7) as f32 + 0.5
            }
        })
        .collect();
    let seed = 42u64;
    let lb = 512u64;
    let rate = 0.4;

    // The dense arm as it was written: 1 everywhere, 0 on a held-out slot.
    let dense_ref = {
        let mut rng = StdRng::seed_from_u64(seed ^ lb);
        let mut vis = vec![1f32; values.len()];
        for (slot, &v) in values.iter().enumerate() {
            if v > 0.0 && rng.random::<f64>() < rate {
                vis[slot] = 0.0;
            }
        }
        vis
    };
    // The windowed arm as it was written: 0 everywhere, 1 on a held-out slot.
    let windowed_ref = {
        let mut rng = StdRng::seed_from_u64(seed ^ lb);
        let mut mask_buf = vec![0f32; values.len()];
        for (slot, &v) in values.iter().enumerate() {
            if v > 0.0 && rng.random::<f64>() < rate {
                mask_buf[slot] = 1.0;
            }
        }
        mask_buf
    };

    let held = super::seeded_holdout_mask(&values, seed ^ lb, rate);
    assert_eq!(held, windowed_ref, "the windowed arm's held-out indicator");
    let vis: Vec<f32> = held.iter().map(|m| 1.0 - m).collect();
    assert_eq!(vis, dense_ref, "the dense arm's visible mask");
    assert!(
        held.iter().any(|&m| m > 0.0) && held.contains(&0.0),
        "a fixture that holds out everything or nothing proves nothing"
    );
}

/// ONE aggregation for both evaluation arms.
///
/// The dense arm calls `dense_module_targets`; the windowed arm used to rebuild
/// the same four tensors by hand from the sparse columns. That hand build is
/// kept inline below as the reference: routing the windowed arm through the
/// shared helper must reproduce it exactly, because this path scores OLD
/// checkpoints and its numbers must not move.
mod windowed_aggregation {
    use super::super::{
        csc_to_indexed, seeded_holdout_mask, windowed_module_targets, PerGeneContext,
    };
    use candle_util::candle_core::{DType, Device, Tensor};
    use candle_util::decoder::coarsening_map::CoarseningMap;
    use candle_util::fast_index::scatter_add_cols;
    use candle_util::vae::masked_topic::DenseModuleTargets;

    const D: usize = 8;
    const N: usize = 4;
    const M: usize = 5;
    /// Narrower than the widest column, so genes fall OUTSIDE the window and
    /// the whole-column tensors and the window tensors have to differ.
    const K: usize = 3;

    /// `[D, N]` counts: column 2 is empty (a cell with nothing observed) and
    /// several genes repeat across columns.
    fn block() -> nalgebra_sparse::CscMatrix<f32> {
        let mut m = nalgebra::DMatrix::<f32>::zeros(D, N);
        for (g, j, v) in [
            (0, 0, 3.0),
            (1, 0, 7.0),
            (3, 0, 2.0),
            (5, 0, 5.0),
            (1, 1, 4.0),
            (2, 1, 1.0),
            (6, 1, 9.0),
            (7, 1, 6.0),
            (0, 3, 8.0),
            (4, 3, 2.0),
            (7, 3, 3.0),
        ] {
            m[(g, j)] = v;
        }
        nalgebra_sparse::CscMatrix::from(&m)
    }

    /// A coarsening with repeated target rows: genes 0,1 share a row, as do
    /// 3,4 and 6,7. Nothing here may depend on the map being the identity.
    fn coarsening(dev: &Device) -> CoarseningMap {
        let f2c = [0usize, 0, 1, 2, 2, 3, 4, 4];
        let share = [0.5f32, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5];
        CoarseningMap::new(&f2c, &share, dev).unwrap()
    }

    fn close(got: &Tensor, want: &Tensor, what: &str) {
        let g: Vec<Vec<f32>> = got.to_vec2().unwrap();
        let w: Vec<Vec<f32>> = want.to_vec2().unwrap();
        assert_eq!(g.len(), w.len(), "{what}: row count");
        for (i, (gr, wr)) in g.iter().zip(&w).enumerate() {
            assert_eq!(gr.len(), wr.len(), "{what}: row {i} width");
            for (j, (a, b)) in gr.iter().zip(wr).enumerate() {
                assert!((a - b).abs() <= 1e-6, "{what}: [{i},{j}] {a} vs {b}");
            }
        }
    }

    #[test]
    fn the_windowed_targets_equal_the_hand_built_ones() {
        let dev = Device::Cpu;
        let x_dn = block();
        let map = coarsening(&dev);
        let weights = vec![1.0f32; D];
        let pack =
            csc_to_indexed(&x_dn, K, &weights, None, PerGeneContext::default(), &dev).unwrap();

        // The arm's own hold-out draw, on the pack it encodes from.
        let values_host: Vec<f32> = pack.values.flatten_all().unwrap().to_vec1().unwrap();
        let held = seeded_holdout_mask(&values_host, 7, 0.5);
        let dims = pack.values.dims2().unwrap();
        let masked = Tensor::from_vec(held, dims, &dev).unwrap();
        let real = pack.values.gt(0.0).unwrap().to_dtype(DType::F32).unwrap();
        let visible = (&real - &masked).unwrap();
        let vis_host: Vec<f32> = visible.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            vis_host.iter().any(|&v| v > 0.0) && vis_host.contains(&0.0),
            "a fixture with nothing hidden, or nothing visible, proves nothing"
        );

        // OLD: the four tensors rebuilt by hand from the sparse columns.
        let f2c = map.host_fine_to_coarse();
        let mut dense = vec![0f32; N * M];
        let mut lib = vec![1f32; N];
        for j in 0..N {
            let c = x_dn.col(j);
            for (&r, &v) in c.row_indices().iter().zip(c.values().iter()) {
                dense[j * M + f2c[r]] += v;
                lib[j] += v;
            }
        }
        let m_ctx = map.groups_of(&pack.indices).unwrap();
        let share_ctx = map.log_share_at(&pack.indices).unwrap().exp().unwrap();
        let want = DenseModuleTargets {
            values_nm: Tensor::from_vec(dense, (N, M), &dev).unwrap(),
            visible_counts_nm: scatter_add_cols(&m_ctx, &(&pack.values * &visible).unwrap(), M)
                .unwrap(),
            visible_share_nm: scatter_add_cols(&m_ctx, &(share_ctx * &visible).unwrap(), M)
                .unwrap(),
            lib_n1: Tensor::from_vec(lib, (N, 1), &dev).unwrap(),
        };

        // NEW: the dense arm's aggregation, on the densified block and the
        // pack's visible slots put back on the gene axis.
        let got = windowed_module_targets(&map, &x_dn, &pack.indices, &visible, D, &dev).unwrap();

        close(&got.values_nm, &want.values_nm, "values_nm");
        close(
            &got.visible_counts_nm,
            &want.visible_counts_nm,
            "visible_counts_nm",
        );
        close(
            &got.visible_share_nm,
            &want.visible_share_nm,
            "visible_share_nm",
        );
        close(&got.lib_n1, &want.lib_n1, "lib_n1");

        // The window really is narrower than the data, or the two would agree
        // for an uninteresting reason.
        let counts: Vec<Vec<f32>> = want.visible_counts_nm.to_vec2().unwrap();
        let totals: Vec<Vec<f32>> = want.values_nm.to_vec2().unwrap();
        let seen: f32 = counts.iter().flatten().sum();
        let all: f32 = totals.iter().flatten().sum();
        assert!(
            seen < all,
            "every count landed in the window: {seen} vs {all}"
        );
    }
}
