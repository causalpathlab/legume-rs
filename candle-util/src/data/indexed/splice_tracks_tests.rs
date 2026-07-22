use super::*;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// `G` genes laid out as `2G` rows: row `2g` nascent, row `2g+1` mature.
fn map_of(n_genes: usize) -> GeneTrackMap {
    GeneTrackMap {
        row_to_gene: (0..2 * n_genes).map(|r| (r / 2) as u32).collect(),
        row_is_nascent: (0..2 * n_genes).map(|r| r % 2 == 0).collect(),
        n_genes,
    }
}

/// `entries` are `(gene, track_is_nascent, cell, value)`.
fn csc(n_genes: usize, n_cells: usize, entries: &[(usize, bool, usize, f32)]) -> CscMatrix<f32> {
    let mut coo = CooMatrix::new(2 * n_genes, n_cells);
    for &(g, nascent, c, v) in entries {
        coo.push(2 * g + usize::from(!nascent), c, v);
    }
    CscMatrix::from(&coo)
}

fn sample_of(s: &GemSample) -> Vec<(u32, f32, f32)> {
    let mut v: Vec<(u32, f32, f32)> = s
        .genes
        .iter()
        .zip(s.nascent.iter())
        .zip(s.mature.iter())
        .map(|((&g, &n), &m)| (g, n, m))
        .collect();
    v.sort_by_key(|x| x.0);
    v
}

/// The load-bearing property of the chunked scatter: the dense scratch buffers
/// are REUSED across every cell in a chunk, so a cell must not inherit the
/// previous cell's counts. Cells here have disjoint gene sets, so any leak shows
/// up as a gene appearing in a cell that never observed it.
///
/// This is not hypothetical — the sparse rewrite initially dropped the reset
/// entirely and every existing test still passed, because none of them exercise
/// this function.
#[test]
fn a_cell_does_not_inherit_the_previous_cells_counts() {
    let n_genes = 8;
    let map = map_of(n_genes);
    let w = vec![1.0f32; n_genes];

    // Cell 0 observes genes {0,1}; cell 1 observes {5,6}. Disjoint on purpose.
    let x = csc(
        n_genes,
        2,
        &[
            (0, true, 0, 3.0),
            (0, false, 0, 7.0),
            (1, true, 0, 2.0),
            (5, false, 1, 9.0),
            (6, true, 1, 4.0),
        ],
    );

    let s = gem_samples_from_csc(&x, &map, &w, 16);
    assert_eq!(s.len(), 2);
    assert_eq!(sample_of(&s[0]), vec![(0, 3.0, 7.0), (1, 2.0, 0.0)]);
    assert_eq!(sample_of(&s[1]), vec![(5, 0.0, 9.0), (6, 4.0, 0.0)]);
}

/// Both tracks of one gene must pool onto that single gene, and the top-K score
/// must rank on the POOLED total — that is what keeps a gene's two tracks
/// selected together.
#[test]
fn the_two_tracks_of_a_gene_pool_onto_one_entry() {
    let map = map_of(4);
    let w = vec![1.0f32; 4];
    // Gene 2 is small on each track alone (2+3) but largest pooled; gene 0 has 4.
    let x = csc(
        4,
        1,
        &[
            (0, true, 0, 4.0),
            (2, true, 0, 2.0),
            (2, false, 0, 3.0),
        ],
    );

    let s = gem_samples_from_csc(&x, &map, &w, 1);
    assert_eq!(s[0].genes.len(), 1, "context_size = 1 must select one gene");
    assert_eq!(
        sample_of(&s[0]),
        vec![(2, 2.0, 3.0)],
        "top-K must rank on pooled nascent+mature, so gene 2 (5) beats gene 0 (4)"
    );
}

/// Correctness must not depend on how cells fall across chunk boundaries. Run a
/// cell count well past the 64-cell chunk size and check every cell independently.
#[test]
fn results_are_identical_across_chunk_boundaries() {
    let n_genes = 64;
    let n_cells = 200; // spans several chunks
    let map = map_of(n_genes);
    let w = vec![1.0f32; n_genes];

    // Cell c observes exactly gene (c % n_genes), with a value keyed to the cell.
    let entries: Vec<(usize, bool, usize, f32)> = (0..n_cells)
        .map(|c| (c % n_genes, true, c, (c + 1) as f32))
        .collect();
    let x = csc(n_genes, n_cells, &entries);

    let s = gem_samples_from_csc(&x, &map, &w, 8);
    assert_eq!(s.len(), n_cells);
    for (c, sc) in s.iter().enumerate() {
        assert_eq!(
            sample_of(sc),
            vec![((c % n_genes) as u32, (c + 1) as f32, 0.0)],
            "cell {c} is wrong — chunking changed the result"
        );
    }
}

/// A cell with fewer observed genes than `context_size` yields only what it
/// observed, rather than padding the shortlist with zero-count genes. This
/// matches the sibling `csc_columns_to_indexed_samples`, which never sees the
/// zero entries at all.
#[test]
fn fewer_observed_genes_than_k_yields_only_the_observed_ones() {
    let map = map_of(32);
    let w = vec![1.0f32; 32];
    let x = csc(32, 1, &[(7, true, 0, 5.0), (9, false, 0, 6.0)]);

    let s = gem_samples_from_csc(&x, &map, &w, 16);
    assert_eq!(s[0].genes.len(), 2, "only two genes were observed");
    assert_eq!(sample_of(&s[0]), vec![(7, 5.0, 0.0), (9, 0.0, 6.0)]);
}

/// The batch null must be read at each gene's OWN track row.
///
/// It used to be read at the mature row and applied to both tracks, justified as
/// "a per-cell library/batch effect, not a per-track one". The collapse fits it
/// per ROW, and intronic capture varies by batch differently from exonic — so
/// dividing both tracks by the spliced factor leaves `r^u/r^s`, which is itself
/// a per-gene splice-ratio distortion and lands directly in `δ`, the estimand.
#[test]
fn the_batch_null_is_read_per_track_not_shared() {
    let dev = candle_core::Device::Cpu;
    let n_genes = 4;
    let map = map_of(n_genes);
    let w = vec![1.0f32; n_genes];
    let x = csc(n_genes, 1, &[(1, true, 0, 5.0), (1, false, 0, 7.0)]);

    // Row 2*g = nascent, 2*g+1 = mature. Gene 1 gets clearly different values.
    let mut null = vec![1.0f32; 2 * n_genes];
    null[2] = 0.25; // gene 1 nascent
    null[3] = 4.0; // gene 1 mature

    let data = GemIndexedData::from_samples(
        gem_samples_from_csc(&x, &map, &w, 8),
        &map,
        8,
        None,
        None,
        Some(vec![null]),
    ).unwrap();
    let mb = data.minibatch_ordered(0, 1, &dev).unwrap();

    let nas: Vec<f32> = mb.nascent_residual.unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let mat: Vec<f32> = mb.mature_residual.unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(nas[0], 0.25, "nascent null must come from the NASCENT row");
    assert_eq!(mat[0], 4.0, "mature null must come from the MATURE row");
    assert!(
        (nas[0] - mat[0]).abs() > 1e-6,
        "the two tracks collapsed to one null — the shared-null bug is back"
    );
}

/// With no decoder target supplied (the inference path, and the un-adjusted
/// training path) the fields stay `None`, so the trainer falls back to scoring
/// the encoder's own values.
#[test]
fn no_decoder_target_when_none_is_supplied() {
    let dev = candle_core::Device::Cpu;
    let map = map_of(4);
    let w = vec![1.0f32; 4];
    let x = csc(4, 1, &[(0, true, 0, 3.0)]);
    let data =
        GemIndexedData::from_samples(gem_samples_from_csc(&x, &map, &w, 4), &map, 4, None, None, None).unwrap();
    let mb = data.minibatch_ordered(0, 1, &dev).unwrap();
    assert!(mb.nascent_adjusted.is_none() && mb.mature_adjusted.is_none());
    assert!(mb.nascent_residual.is_none() && mb.mature_residual.is_none());
}

/// The `(observed, residual, adjusted)` triple must be ONE matrix shape on ONE
/// feature axis, and a violation must be caught here rather than downstream.
///
/// The gathers index `rows[si][r]` with `r` taken from the gene/track map, which
/// is built against `observed`. A SHORT matrix panics out-of-bounds mid-epoch; a
/// WIDE one is worse — it reads a real value from the wrong row and the decoder
/// is scored against a different gene's counts, with nothing to notice. Only
/// `observed` used to be checked, so the two that fail quietly were the two that
/// went unchecked.
#[test]
fn a_target_on_a_different_axis_is_refused_not_read_at_the_wrong_row() {
    let map = map_of(3); // 3 genes → 6 rows
    let w = vec![1.0f32; 3];
    let observed = nalgebra::DMatrix::<f32>::from_element(2, 6, 1.0);

    // Same shape → accepted.
    let ok = GemIndexedData::from_dense(GemIndexedArgs {
        observed: &observed,
        residual: None,
        adjusted: Some(&observed),
        map: &map,
        context_size: 3,
        gene_weights: &w,
        nascent_mean: None,
        mature_mean: None,
    });
    assert!(ok.is_ok(), "a matching target must be accepted");

    // Wider target — the silent case.
    let wide = nalgebra::DMatrix::<f32>::from_element(2, 8, 1.0);
    let err = GemIndexedData::from_dense(GemIndexedArgs {
        observed: &observed,
        residual: None,
        adjusted: Some(&wide),
        map: &map,
        context_size: 3,
        gene_weights: &w,
        nascent_mean: None,
        mature_mean: None,
    })
    .map(|_| ())
    .expect_err("a target on a wider axis must be refused");
    assert!(
        err.to_string().contains("adjusted"),
        "the error must name which matrix is wrong, got: {err}"
    );

    // Wrong sample count on the residual.
    let short = nalgebra::DMatrix::<f32>::from_element(1, 6, 1.0);
    let err = GemIndexedData::from_dense(GemIndexedArgs {
        observed: &observed,
        residual: Some(&short),
        adjusted: None,
        map: &map,
        context_size: 3,
        gene_weights: &w,
        nascent_mean: None,
        mature_mean: None,
    })
    .map(|_| ())
    .expect_err("a residual with the wrong sample count must be refused");
    assert!(err.to_string().contains("residual"), "got: {err}");
}

/// `from_samples` states the same axis invariant in its docstring and used to
/// enforce none of it. Inference is where a mismatch is invisible: the encoder
/// is simply handed a different input distribution than it was fitted on, and
/// still returns a perfectly plausible latent.
#[test]
fn inference_residual_rows_must_match_the_sample_and_row_axes() {
    let map = map_of(3);
    let w = vec![1.0f32; 3];
    let x = csc(3, 2, &[(0, true, 0, 2.0), (1, false, 1, 3.0)]);
    let samples = gem_samples_from_csc(&x, &map, &w, 3);
    assert_eq!(samples.len(), 2);

    // One residual row per sample, each covering every feature row → fine.
    let good = vec![vec![1.0f32; 6], vec![1.0f32; 6]];
    assert!(
        GemIndexedData::from_samples(samples.clone(), &map, 3, None, None, Some(good)).is_ok()
    );

    // Too few rows for the samples.
    let err = GemIndexedData::from_samples(
        samples.clone(),
        &map,
        3,
        None,
        None,
        Some(vec![vec![1.0f32; 6]]),
    )
    .map(|_| ())
    .expect_err("one residual row for two samples must be refused");
    assert!(err.to_string().contains("residual"), "got: {err}");

    // Right count, wrong width — the row ids would index off the end.
    let err = GemIndexedData::from_samples(
        samples,
        &map,
        3,
        None,
        None,
        Some(vec![vec![1.0f32; 4], vec![1.0f32; 4]]),
    )
    .map(|_| ())
    .expect_err("residual rows narrower than the feature axis must be refused");
    assert!(err.to_string().contains("feature rows"), "got: {err}");
}
