//! Stream a column selection (and an optional ascending row selection) of a
//! sparse backend into a fresh backend without materialising the survivors.
//! Shared by the `split` / `subsample` CLI handlers and by downstream crates
//! (faba's `qc`) that filter a matrix by cells and features.

/// Stream a column selection (with an optional ASCENDING row filter) into a
/// fresh backend, without ever materialising the survivors.
///
/// The shape `split` and `subsample` share: pick columns, optionally keep an
/// ascending subset of rows, write a new file. Both used to read every
/// surviving triplet into one `Vec` and hand it to the sorting triplet writer —
/// 24 B/nnz of residency that OOMs at imaging scale. Here the exact output nnz
/// comes from the resident indptr (or one counting pass when rows are
/// filtered), and survivors flow through the validated streaming writer in
/// byte-budgeted slabs.
///
/// `row_filter`, when given, must be ascending: the renumbering is then
/// monotone, so within-column row order survives and no per-column sort is
/// needed. Both callers sample or select ascending; the debug assert keeps the
/// next caller honest rather than silently writing unsorted columns (which the
/// streaming writer would refuse anyway).
pub fn stream_column_selection(
    data: &dyn crate::sparse_io::SparseIo<IndexIter = Vec<usize>>,
    selected_columns: &[usize],
    row_filter: Option<&[usize]>,
    out_row_names: &[Box<str>],
    out_col_names: &[Box<str>],
    file_out: &str,
    backend_out: &crate::sparse_io::SparseIoBackend,
) -> anyhow::Result<(usize, usize, usize)> {
    use crate::sparse_io::*;

    let nrow_full = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;

    // Old-row → new-row, monotone by construction.
    let row_map: Option<Vec<Option<u64>>> = row_filter.map(|keep| {
        debug_assert!(
            keep.windows(2).all(|w| w[0] < w[1]),
            "row filter must ascend"
        );
        let mut map = vec![None; nrow_full];
        for (new, &old) in keep.iter().enumerate() {
            map[old] = Some(new as u64);
        }
        map
    });
    let out_nrow = row_filter.map_or(nrow_full, <[usize]>::len);
    let out_ncol = selected_columns.len();

    // Exact per-output-column nnz. Free from the indptr when every row
    // survives; one counting pass — counts, never entries — otherwise.
    let per_col_nnz: Vec<u64> = match &row_map {
        None => selected_columns
            .iter()
            .map(|&c| {
                data.column_nnz(c)
                    .ok_or_else(|| anyhow::anyhow!("no indptr entry for column {c}"))
            })
            .collect::<anyhow::Result<_>>()?,
        Some(map) => {
            // Block reads here too; counts only, never entries.
            let mut counts = vec![0u64; out_ncol];
            let coarse = matrix_util::utils::generate_minibatch_intervals(out_ncol, 0, Some(8192));
            for (lb, ub) in coarse {
                let (_, _, triplets) =
                    data.read_triplets_by_columns(selected_columns[lb..ub].to_vec())?;
                for (r, c_local, _) in triplets {
                    if map[r as usize].is_some() {
                        counts[lb + c_local as usize] += 1;
                    }
                }
            }
            counts
        }
    };
    let nnz: u64 = per_col_nnz.iter().sum();

    let mut out = create_sparse_streaming_empty(Some(file_out), Some(backend_out))?;
    out.begin_streaming_csc((out_nrow, out_ncol, nnz as usize))?;

    let blocks = matrix_util::utils::byte_budget_intervals(
        &per_col_nnz,
        crate::sparse_io::SLAB_BUDGET_BYTES,
        crate::sparse_io::TRIPLET_BYTES,
    );

    let t_stream = std::time::Instant::now();
    let mut nnz_offset = 0u64;
    for (lb, ub) in blocks {
        // ONE block read per slab, never a read per column: a single-column
        // read pays the cached-subset machinery per call, and at hundreds of
        // thousands of columns that made this path slower than the memory wall
        // it replaced. The block read returns LOCAL column ids for the
        // requested set, rows ascending within each column — the writer's
        // invariant already.
        let (_, _, triplets) = data.read_triplets_by_columns(selected_columns[lb..ub].to_vec())?;

        let n_block = ub - lb;
        let mut per_col: Vec<Vec<(u64, f32)>> = vec![Vec::new(); n_block];
        for (r, c_local, x) in triplets {
            let kept = match &row_map {
                None => Some(r),
                Some(map) => map[r as usize],
            };
            if let Some(new_r) = kept {
                per_col[c_local as usize].push((new_r, x));
            }
        }
        let mut local_colptr = Vec::with_capacity(n_block);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        for entries in &per_col {
            local_colptr.push(row_indices.len() as u64);
            for &(r, x) in entries {
                row_indices.push(r);
                values.push(x);
            }
        }
        out.append_csc_slab(lb as u64, nnz_offset, &local_colptr, &row_indices, &values)?;
        nnz_offset += values.len() as u64;
    }

    out.finalize_streaming_csc()?;
    out.build_csr_from_csc_streaming()?;
    log::info!(
        "streamed {nnz} entries in {out_ncol} columns ({:.1}s)",
        t_stream.elapsed().as_secs_f32()
    );
    out.register_row_names_vec(out_row_names);
    out.register_column_names_vec(out_col_names);
    Ok((out_nrow, out_ncol, nnz as usize))
}
