use super::*;

/// Softmax probability of column `idx` within a score row (numerically stable;
/// uniform fallback for an all-equal/degenerate row). A bounded [0,1]
/// confidence for the selected label.
fn softmax_prob_at(row: &[f32], idx: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = row.iter().map(|&z| (z - max).exp()).sum();
    if denom > 0.0 {
        (row[idx] - max).exp() / denom
    } else {
        1.0 / row.len().max(1) as f32
    }
}

/// Write the two per-cell label files shared by BOTH annotation methods, so a
/// downstream consumer never has to know which produced them:
/// * `{prefix}.argmax.tsv` — `cell⇥cell_type⇥probability` (with header), the
///   argmax reader `senna plot` consumes.
/// * `{prefix}.membership.tsv` — `cell⇥cell_type` (no header), for
///   `data-beans stat -g` / `faba gem-summary` grouping.
///
/// `labels`/`probs` are parallel to `cell_names`. The single writer keeps the
/// two annotation passes (`-by-projection`, `-by-enrichment`) in lock-step.
pub fn write_label_tsvs(
    prefix: &str,
    cell_names: &[Box<str>],
    labels: &[Box<str>],
    probs: &[f32],
) -> Result<()> {
    let n = cell_names.len();
    let mut argmax = Vec::with_capacity(n + 1);
    argmax.push(Box::from("cell\tcell_type\tprobability"));
    argmax.par_extend(
        (0..n)
            .into_par_iter()
            .map(|i| format!("{}\t{}\t{:.4}", cell_names[i], labels[i], probs[i]).into_boxed_str()),
    );
    let argmax_path = format!("{prefix}.argmax.tsv");
    write_lines(&argmax, &argmax_path).with_context(|| format!("writing {argmax_path}"))?;
    info!("wrote {argmax_path}");

    let membership: Vec<Box<str>> = cell_names
        .iter()
        .zip(labels)
        .map(|(cell, label)| format!("{cell}\t{label}").into_boxed_str())
        .collect();
    let membership_path = format!("{prefix}.membership.tsv");
    write_lines(&membership, &membership_path)
        .with_context(|| format!("writing {membership_path}"))?;
    info!("wrote {membership_path}");
    Ok(())
}

/// Write the (co-embedded) embeddings of every matched marker gene, labelled by
/// type + weight, to `{out_prefix}.marker_embedding.parquet`. Rows are the
/// `(gene, type)` entries actually used by the type signatures (a gene shared by
/// two types appears twice). Columns: `type`, `weight`, then `h0..h{H-1}`.
pub(super) fn write_marker_embeddings(
    out_prefix: &str,
    feature_emb: &DMatrix<f32>,
    gene_names: &[Box<str>],
    type_names: &[Box<str>],
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
) -> Result<()> {
    let mut genes: Vec<Box<str>> = Vec::new();
    let mut types: Vec<Box<str>> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let mut h_cols: Vec<Vec<f32>> = vec![Vec::new(); h];
    for (ti, markers) in type_markers.iter().enumerate() {
        for &(fi, w) in markers {
            let fi = fi as usize;
            genes.push(gene_names[fi].clone());
            types.push(type_names[ti].clone());
            weights.push(w);
            for (c, col) in h_cols.iter_mut().enumerate() {
                col.push(feature_emb[(fi, c)]);
            }
        }
    }
    if genes.is_empty() {
        return Ok(());
    }
    let mut columns: Vec<(Box<str>, Column)> = Vec::with_capacity(h + 2);
    columns.push((Box::from("type"), Column::Str(&types)));
    columns.push((Box::from("weight"), Column::F32(&weights)));
    let h_names = crate::embedding_col_names(h);
    for (col, name) in h_cols.iter().zip(h_names) {
        columns.push((name, Column::F32(col)));
    }
    let path = format!("{out_prefix}.marker_embedding.parquet");
    write_named_table(&path, "gene", &genes, &columns)?;
    info!("wrote {path} ({} matched markers × {h} dims)", genes.len());
    Ok(())
}

/// Co-embed the cell-type signatures (`res.type_emb_ch` fine, `res.coarse_emb_kh`
/// coarse) onto the cell manifold via the SIMBA softmax-over-cells transform —
/// the same operator used for genes in `senna bge` — so each type lands at the
/// weighted centroid of *its* cells. Writes `{out}.{type,coarse}_embedding.parquet`.
/// Returns the co-embed locations `(type [C×H], coarse [K×H])` so the layout can
/// place the anchors *firmly* on the cell manifold (Nyström through the cell
/// layout) instead of re-centering raw centroids.
pub(super) fn write_type_coembeddings(
    out_prefix: &str,
    cell_emb: &DMatrix<f32>,
    type_names: &[Box<str>],
    h: usize,
    res: &AnnotateProjOutputs,
) -> Result<(DMatrix<f32>, DMatrix<f32>)> {
    use candle_util::candle_core::{Device, Tensor};
    use matrix_util::traits::ConvertMatOps;
    let cpu = Device::Cpu;
    let cell_t = cell_emb.to_tensor(&cpu)?; // [N, H]
                                            // Eff-cells temperature target = median size of the coarse communities we
                                            // ALREADY clustered (`annotate_by_projection`'s Leiden); no second Leiden
                                            // pass just to derive it.
    let target_eff = crate::postprocess::target_eff_from_labels(&res.community, res.n_coarse);
    let col_names = crate::embedding_col_names(h);
    let place = |sig: &[f32], names: &[Box<str>], suffix: &str| -> Result<DMatrix<f32>> {
        let rows = names.len();
        if rows == 0 {
            return Ok(DMatrix::<f32>::zeros(0, h));
        }
        let sig_t = Tensor::from_vec(sig.to_vec(), (rows, h), &cpu)?;
        let (co, _t) = crate::postprocess::feature_coembedding(&cell_t, &sig_t, target_eff)?;
        let flat: Vec<f32> = co.to_vec2::<f32>()?.into_iter().flatten().collect();
        let mat = DMatrix::<f32>::from_row_iterator(rows, h, flat);
        let path = format!("{out_prefix}.{suffix}.parquet");
        mat.to_parquet_with_names(&path, (Some(names), Some("cell_type")), Some(&col_names))
            .with_context(|| format!("writing {path}"))?;
        info!("wrote {path} ({rows} types co-embedded onto the cell manifold)");
        Ok(mat)
    };
    let type_co = place(&res.type_emb_ch, type_names, "type_embedding")?;
    let coarse_co = place(&res.coarse_emb_kh, &res.coarse_names, "coarse_embedding")?;
    Ok((type_co, coarse_co))
}

/// Write the tidy annotation tables:
/// * `{prefix}.annot.parquet` — one row per cell: community, coarse + fine
///   label, score (z), and p-value for each layer.
/// * `{prefix}.membership.tsv` — `cell⇥coarse_label` (no header) for
///   `data-beans stat`/`gem-summary` grouping.
/// * `{prefix}.argmax.tsv` — `cell⇥cell_type⇥probability` (with header), the
///   uniform argmax contract shared with `annotate-by-enrichment`.
/// * `{prefix}.community_profile.parquet` — one row per community.
/// * `{prefix}.type_map.parquet` — fine → coarse merge record.
/// * `{prefix}.{type,coarse}_embedding.parquet` — signature plot anchors.
pub(super) fn write_annotation_outputs(
    out_prefix: &str,
    cell_names: &[Box<str>],
    type_names: &[Box<str>],
    res: &AnnotateProjOutputs,
    cfg: &AnnotateProjConfig,
) -> Result<()> {
    let (n, c, k) = (res.n_cells, res.n_types, res.n_coarse);
    let nan = f32::NAN;

    ////////////////////////////
    // per-cell annotation table
    ////////////////////////////
    let mut community = Vec::with_capacity(n);
    let mut coarse_label = Vec::with_capacity(n);
    let mut coarse_z = Vec::with_capacity(n);
    let mut coarse_p = Vec::with_capacity(n);
    let mut coarse_prob = Vec::with_capacity(n);
    let mut fine_z = Vec::with_capacity(n);
    let mut fine_p = Vec::with_capacity(n);
    let mut fine_margin = Vec::with_capacity(n);
    for cell in 0..n {
        let kk = res.community[cell];
        let ff = res.fine_label[cell];
        community.push(kk as i32);
        coarse_label.push(res.coarse_names[kk].clone());
        coarse_z.push(res.coarse_z[cell * k + kk]);
        coarse_p.push(res.coarse_p.as_ref().map_or(nan, |p| p[cell * k + kk]));
        coarse_prob.push(softmax_prob_at(&res.coarse_z[cell * k..(cell + 1) * k], kk));
        fine_z.push(res.fine_z[cell * c + ff]);
        fine_p.push(res.fine_p.as_ref().map_or(nan, |p| p[cell * c + ff]));
        fine_margin.push(res.fine_margin[cell]);
    }
    // BH q-values across the N per-cell calls of each layer (FDR over the
    // selected-label p-values); NaN-filled when the null was skipped.
    let coarse_q = if res.coarse_p.is_some() {
        enrichment::fdr::bh_fdr(&coarse_p)
    } else {
        vec![nan; n]
    };
    let fine_q = if res.fine_p.is_some() {
        enrichment::fdr::bh_fdr(&fine_p)
    } else {
        vec![nan; n]
    };

    // Definitiveness gate (only with a permutation null): a fine call is
    // `unassigned` when its best type is not significant at `fine_fdr` (no
    // marker set fits the cell). With `min_margin > 0` it is ALSO gated on the
    // top1−top2 margin; fine types are nested/correlated by design, so margins
    // are inherently small and margin gating is opt-in (default off) — the
    // significance gate is the safe default.
    let unassigned: Box<str> = Box::from("unassigned");
    let fine_label: Vec<Box<str>> = (0..n)
        .map(|cell| {
            // Both gates off by default (fine_fdr ≥ 1, min_margin ≤ 0) ⇒ every
            // cell keeps its (smoothed) argmax label; per-cell fine calls are
            // inherently modest here, so gating is opt-in and the fine_q /
            // fine_margin columns are there for the user to threshold instead.
            let confident = cfg.n_perm == 0
                || ((cfg.fine_fdr >= 1.0 || fine_q[cell] < cfg.fine_fdr)
                    && (cfg.min_margin <= 0.0 || fine_margin[cell] >= cfg.min_margin));
            if confident {
                type_names[res.fine_label[cell]].clone()
            } else {
                unassigned.clone()
            }
        })
        .collect();

    let annot_path = format!("{out_prefix}.annot.parquet");
    write_named_table(
        &annot_path,
        "cell",
        cell_names,
        &[
            (Box::from("community"), Column::I32(&community)),
            (Box::from("coarse_label"), Column::Str(&coarse_label)),
            (Box::from("coarse_z"), Column::F32(&coarse_z)),
            (Box::from("coarse_p"), Column::F32(&coarse_p)),
            (Box::from("coarse_q"), Column::F32(&coarse_q)),
            (Box::from("fine_label"), Column::Str(&fine_label)),
            (Box::from("fine_z"), Column::F32(&fine_z)),
            (Box::from("fine_p"), Column::F32(&fine_p)),
            (Box::from("fine_q"), Column::F32(&fine_q)),
            (Box::from("fine_margin"), Column::F32(&fine_margin)),
        ],
    )
    .with_context(|| format!("writing {annot_path}"))?;
    info!("wrote {annot_path}");

    // Per-cell label files (argmax.tsv + membership.tsv), keyed on the robust
    // COARSE label + its softmax confidence. Shared with `annotate-by-enrichment`
    // via the one writer below, so both methods emit an identical contract.
    write_label_tsvs(out_prefix, cell_names, &coarse_label, &coarse_prob)?;

    ////////////////////////////
    // community profile table
    ////////////////////////////
    let mut comm_sizes = vec![0i32; k];
    for &kk in &res.community {
        comm_sizes[kk] += 1;
    }
    let comm_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    let comm_label: Vec<Box<str>> = res.coarse_names.clone();
    // Show the SAME ranked lineage set that produced the label (weighted
    // ordering), displaying each member's raw centered enrichment.
    let top_fine: Vec<Box<str>> = (0..k)
        .map(|kk| {
            res.community_members[kk]
                .iter()
                .take(5)
                .map(|&t| format!("{}({:.1})", type_names[t], res.enrich[kk * c + t]))
                .collect::<Vec<_>>()
                .join(",")
                .into_boxed_str()
        })
        .collect();
    let profile_path = format!("{out_prefix}.community_profile.parquet");
    write_named_table(
        &profile_path,
        "community",
        &comm_names,
        &[
            (Box::from("n_cells"), Column::I32(&comm_sizes)),
            (Box::from("coarse_label"), Column::Str(&comm_label)),
            (Box::from("top_fine_types"), Column::Str(&top_fine)),
        ],
    )
    .with_context(|| format!("writing {profile_path}"))?;
    info!("wrote {profile_path}");

    ////////////////////////////
    // fine → coarse merge map
    ////////////////////////////
    let mut group_members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (t, &kk) in res.coarse_of_fine.iter().enumerate() {
        group_members[kk].push(t);
    }
    let map_coarse: Vec<Box<str>> = (0..c)
        .map(|t| res.coarse_names[res.coarse_of_fine[t]].clone())
        .collect();
    let map_members: Vec<Box<str>> = (0..c)
        .map(|t| {
            group_members[res.coarse_of_fine[t]]
                .iter()
                .map(|&m| type_names[m].as_ref())
                .collect::<Vec<_>>()
                .join(",")
                .into_boxed_str()
        })
        .collect();
    let map_path = format!("{out_prefix}.type_map.parquet");
    write_named_table(
        &map_path,
        "fine_type",
        type_names,
        &[
            (Box::from("coarse_label"), Column::Str(&map_coarse)),
            (Box::from("members"), Column::Str(&map_members)),
        ],
    )
    .with_context(|| format!("writing {map_path}"))?;
    info!("wrote {map_path}");

    Ok(())
}

/// Log a per-coarse-label cell-count histogram (a quick console sanity check).
pub(super) fn log_label_histogram(res: &AnnotateProjOutputs) {
    let mut counts: FxHashMap<&str, usize> = FxHashMap::default();
    for &kk in &res.community {
        *counts.entry(res.coarse_names[kk].as_ref()).or_insert(0) += 1;
    }
    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    info!(
        "annotation summary ({} cells, {} communities → {} coarse labels):",
        res.n_cells,
        res.n_coarse,
        ranked.len()
    );
    for (label, n) in ranked {
        info!("  {label:24} {n:6}");
    }
}
