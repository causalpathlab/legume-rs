//! Multi-donor HCA bone-marrow stress test — `#[ignore]`d (needs the BM zarrs +
//! a real fit; `--release`):
//!
//! ```text
//! cargo test -p graph-embedding-util --release --test posterior_hca -- --ignored --nocapture
//! ```
//!
//! This is the collapse regime (`--sort-dim 13`, multi-donor): the cross-donor
//! second moment inflates the feature variance, and the *variational* gate /
//! null-call over-call null. The question (plan verification item 4): does the
//! **exact gate posterior** report calibrated uncertainty — credible sets that
//! *widen* rather than a confident collapse — and what does the global effect
//! scale σ₀² look like?
//!
//! Donor count + epochs are consts so the run can be sized to the machine.

use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::posterior::{
    build_index, gate_posterior, hyper_ss, FrozenMap, GateConfig, HyperSsConfig,
};
use graph_embedding_util::{fit, load_unified_data, FitConfig, LoadUnifiedArgs, SoftmaxGateConfig};
use matrix_util::knn_graph::{modularity_to_cpm_resolution, run_leiden, KnnGraph, KnnGraphArgs};
use nalgebra::DMatrix;

/// Real Leiden (the pipeline's own — matrix_util) on a row-major `[n × h]`
/// embedding: kNN graph → fuzzy weights → Leiden communities. Returns per-row
/// labels. Not kmeans.
fn leiden_labels(emb: &[f32], n: usize, h: usize) -> Vec<usize> {
    let data = DMatrix::<f32>::from_row_slice(n, h, emb);
    let knn = KnnGraph::from_rows(
        &data,
        KnnGraphArgs {
            knn: 15,
            block_size: 1024,
            reciprocal: false,
        },
    )
    .expect("knn");
    let (net, tew) = knn.to_leiden_network();
    run_leiden(&net, n, modularity_to_cpm_resolution(1.0, tew), Some(1))
}

// Where the subsampled input zarrs live and where the TSV exports go. An env var,
// not a constant: this is a local-only harness, and a hardcoded absolute path is
// unrunnable by anyone but whoever first wrote it.
//
//   GEU_HCA_SCRATCH=/path/to/dir cargo test -p graph-embedding-util --release \
//       --test posterior_hca -- --ignored --nocapture
//
// The dir must hold `BM{1..N_DONORS}_2k.zarr.zip`, made with `data-beans subsample`
// from the per-donor zarrs in `/data3/AML/BoneMarrowMap/beans`.
fn scratch() -> String {
    std::env::var("GEU_HCA_SCRATCH").unwrap_or_else(|_| "/tmp/geu-hca".to_string())
}
const N_DONORS: usize = 8; // all HCA BM donors (2k cells each, CUDA fit)
const EPOCHS: usize = 300; // GPU-fast SGD, so afford more
                           // `--sort-dim 13` (8192 pb) is the collapse *stress* config, but its BBKNN
                           // refine did not finish in 540 s here; d=10 (1024 pb) is the memory's
                           // documented "realistic operating point" and is feasible. Still multi-donor.
const SORT_DIM: usize = 10;
const H: usize = 16;
const TOP_GENES: usize = 400;
const N_GENE_UMAP: usize = 3000; // live genes (top by ‖e_feat‖) for the gene UMAP
const N_PARTITION: usize = 256;

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
#[ignore = "needs HCA BM zarrs + a real fit; run with --release --ignored"]
fn gate_posterior_on_hca_bm() -> anyhow::Result<()> {
    // Small multi-donor subsets (3k cells/donor) made with `data-beans subsample`,
    // so the fit is feasible interactively (cells are the fit's binding cost).
    let scratch = scratch();
    let files: Vec<Box<str>> = (1..=N_DONORS)
        .map(|d| format!("{scratch}/BM{d}_2k.zarr.zip").into_boxed_str())
        .collect();
    if !std::path::Path::new(&*files[0]).exists() {
        eprintln!("SKIP: {} not present (set GEU_HCA_SCRATCH)", files[0]);
        return Ok(());
    }

    // Multi-donor: disjoint barcodes, batch = per file.
    let mut unified = load_unified_data(LoadUnifiedArgs {
        data_files: files.clone(),
        feature_kind: Some(graph_embedding_util::FeatureNameKind::Gene { delim: '_' }),
        column_alignment: ColumnAlignment::Disjoint,
        preload: true, // hold columns in memory — avoids tiny-block zarr I/O
        ..Default::default()
    })?;
    let (n_cells, n_features, n_batches) =
        (unified.n_cells(), unified.n_features(), unified.n_batches());
    eprintln!(
        "loaded {N_DONORS} donors: {n_cells} cells × {n_features} genes, {n_batches} batches"
    );

    let cfg = FitConfig {
        embedding_dim: H,
        num_levels: 1,
        sort_dim: SORT_DIM,
        knn_pb_samples: 10,
        num_opt_iter: 3, // lighten the collapse (feasibility)
        proj_dim: H,
        epochs: EPOCHS,
        batches_per_epoch: None,
        batch_size: 256,
        num_negatives: 5,
        learning_rate: 0.05,
        seed: 1,
        device: candle_util::candle_core::Device::new_cuda(0)?,
        block_size: Some(2048), // NOT None → None clamps to ~100 cols (glacial disk I/O)
        feature_network: None,
        hvg_weights: None,
        // Lighten the DC-Poisson refine (default 20+10 sweeps/level) for feasibility.
        refine: Some(graph_embedding_util::RefineParams {
            num_gibbs: 3,
            num_greedy: 2,
            ..Default::default()
        }),
        feature_embedding_l2: 0.0, // default (unbounded) — the collapse regime
        weight_decay: 0.0,
        max_grad_norm: 0.0,
        delta_l2: 0.0,
        cell_weight_mult: None,
        phase1_cells_per_pb: 0,
        feat_factor: None,
        lineage_dag: false,
        lineage_smooth: false,
        lineage_mst: false,
        joint_velocity: false,
        feature_projection: None,
        nce_objective: graph_embedding_util::loss::NceObjective::Softmax,
        softmax_gate: Some(SoftmaxGateConfig { temperature: 1.0 }),
    };
    let t0 = std::time::Instant::now();
    let out = fit(&mut unified, cfg)?;
    eprintln!(
        "fit ({EPOCHS} epochs) done in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let e_cell: Vec<f32> = out.model.e_cell.flatten_all()?.to_vec1::<f32>()?;
    let b_cell: Vec<f32> = out.model.b_cell.flatten_all()?.to_vec1::<f32>()?;
    let b_feat: Vec<f32> = out.model.b_feat.flatten_all()?.to_vec1::<f32>()?;
    let e_feat: Vec<f32> = out.model.e_feat.flatten_all()?.to_vec1::<f32>()?; // gated gene dict
    let donor: Vec<u32> = unified.batch_membership.clone(); // per-cell donor id

    // Real Leiden (matrix_util) on the cell embedding — the pipeline's clustering,
    // not the earlier kmeans stand-in (R lacks igraph, so it is done here).
    let cell_leiden = leiden_labels(&e_cell, n_cells, H);
    let var_sel: Vec<f32> = out
        .model
        .feature_selection()?
        .expect("gate on")
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Calibrated by construction — an uncalibrated index collapses every anchor
    // onto the count-weighted mean direction.
    let frozen = FrozenMap {
        e_cell: e_cell.clone(),
        b_cell: b_cell.clone(),
        b_feat: b_feat.clone(),
        h: H,
    };
    let idx = build_index(&unified, &frozen, N_PARTITION, 7, None, None)?;
    let total: Vec<f32> = idx
        .pos
        .iter()
        .map(|p: &Vec<(u32, f32)>| p.iter().map(|&(_, c)| c).sum())
        .collect();
    let mut order: Vec<usize> = (0..n_features).collect();
    order.sort_by(|&a, &b| total[b].partial_cmp(&total[a]).unwrap());
    let picked: Vec<usize> = order.into_iter().take(TOP_GENES).collect();

    let all_nodes = idx.node_terms();
    let sub_nodes: Vec<_> = picked.iter().map(|&g| all_nodes[g]).collect();
    let side = idx.frozen_side();

    // Gate posterior on the top genes.
    let t1 = std::time::Instant::now();
    let post = gate_posterior(&sub_nodes, &side, &GateConfig::new(200, 100, 13));
    eprintln!(
        "gate posterior: {} genes in {:.1}s",
        post.len(),
        t1.elapsed().as_secs_f32()
    );

    // Collapse signature: variational non-null mass (how much the gate selects).
    let rowsum = |g: usize| -> f32 { var_sel[g * H..(g + 1) * H].iter().sum() };
    let mut nn: Vec<f32> = picked.iter().map(|&g| rowsum(g)).collect();
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "variational non-null mass: max={:.3} median={:.3} (collapse ⇒ near 0)",
        nn[nn.len() - 1],
        nn[nn.len() / 2]
    );

    // Posterior: concentration + credible-set width (the uncertainty read).
    let mean_max_pip: f64 = post
        .iter()
        .map(|g| f64::from(g.pip.iter().cloned().fold(0.0, f32::max)))
        .sum::<f64>()
        / post.len() as f64;
    let mean_cs: f64 = post
        .iter()
        .map(|g| g.credible_set.len() as f64)
        .sum::<f64>()
        / post.len() as f64;
    eprintln!(
        "posterior: mean max-PIP = {mean_max_pip:.3} (uniform {:.3}); mean 95% credible-set size = {mean_cs:.2} of {H} dims",
        1.0 / H as f64
    );

    // Agreement with the variational gate on the most-selected genes.
    let mut by_conf: Vec<usize> = (0..picked.len()).collect();
    by_conf.sort_by(|&a, &b| rowsum(picked[b]).partial_cmp(&rowsum(picked[a])).unwrap());
    let top = 100.min(by_conf.len());
    let agree = by_conf
        .iter()
        .take(top)
        .filter(|&&i| argmax(&post[i].pip) == argmax(&var_sel[picked[i] * H..(picked[i] + 1) * H]))
        .count();
    eprintln!(
        "top-{top} argmax agreement: {:.3} (chance {:.3})",
        agree as f64 / top as f64,
        1.0 / H as f64
    );

    // Complete Tier-1: interleaved spike-and-slab → σ₀² AND π₀ with real ESS.
    let t2 = std::time::Instant::now();
    let hres = hyper_ss(&sub_nodes, &side, &HyperSsConfig::new(300, 100, 9));
    eprintln!(
        "hyper Tier-1: σ₀²={:.3} (ESS {:.0}), π₀={:.3} (ESS {:.0}) in {:.1}s",
        hres.sigma2_mean,
        hres.sigma_diag.min_ess,
        hres.pi0_mean,
        hres.pi0_diag.min_ess,
        t2.elapsed().as_secs_f32()
    );

    // ---- export raw tables for R (UMAP + plots live there) ----
    // Cells: the MAP embedding (e0..e{H-1}) + donor + depth + real Leiden. R lays out.
    let mut cells = String::with_capacity(n_cells * 64);
    cells.push_str("donor\tdepth\tleiden");
    for k in 0..H {
        cells.push_str(&format!("\te{k}"));
    }
    cells.push('\n');
    for i in 0..n_cells {
        cells.push_str(&format!(
            "{}\t{:.4}\t{}",
            donor[i], b_cell[i], cell_leiden[i]
        ));
        for k in 0..H {
            cells.push_str(&format!("\t{:.5}", e_cell[i * H + k]));
        }
        cells.push('\n');
    }
    std::fs::write(format!("{scratch}/hca_cells.tsv"), cells)?;

    // Gene embedding: the live (gate-selected) genes — top by ‖e_feat_g‖ — with
    // their gene-space Leiden module + the embedding for R's gene UMAP. Most genes
    // are gated to ~0 (null), so filtering by norm keeps the real modules.
    let gnorm = |g: usize| -> f32 {
        e_feat[g * H..(g + 1) * H]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt()
    };
    let mut gorder: Vec<usize> = (0..n_features).collect();
    gorder.sort_by(|&a, &b| gnorm(b).partial_cmp(&gnorm(a)).unwrap());
    let live_genes: Vec<usize> = gorder
        .into_iter()
        .take(N_GENE_UMAP.min(n_features))
        .collect();
    let gene_emb: Vec<f32> = live_genes
        .iter()
        .flat_map(|&g| e_feat[g * H..(g + 1) * H].iter().copied())
        .collect();
    let gene_leiden = leiden_labels(&gene_emb, live_genes.len(), H);
    let mut gtab = String::from("gene\tleiden\tnorm");
    for k in 0..H {
        gtab.push_str(&format!("\te{k}"));
    }
    gtab.push('\n');
    for (i, &g) in live_genes.iter().enumerate() {
        gtab.push_str(&format!("{g}\t{}\t{:.4}", gene_leiden[i], gnorm(g)));
        for k in 0..H {
            gtab.push_str(&format!("\t{:.5}", e_feat[g * H + k]));
        }
        gtab.push('\n');
    }
    std::fs::write(format!("{scratch}/hca_gene_emb.tsv"), gtab)?;
    let n_gene_clusters = gene_leiden
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len();
    eprintln!(
        "gene modules: {} live genes → {} Leiden clusters",
        live_genes.len(),
        n_gene_clusters
    );

    // Genes: per top gene, the variational vs posterior gate summaries.
    let mut genes = String::from("gene\ttotal\tvar_nonnull\tpost_max_pip\tpost_cs_size\n");
    for (i, &g) in picked.iter().enumerate() {
        let pmax = post[i].pip.iter().cloned().fold(0.0f32, f32::max);
        genes.push_str(&format!(
            "{g}\t{:.2}\t{:.4}\t{:.4}\t{}\n",
            total[g],
            rowsum(g),
            pmax,
            post[i].credible_set.len()
        ));
    }
    std::fs::write(format!("{scratch}/hca_genes.tsv"), genes)?;
    eprintln!("wrote {scratch}/hca_cells.tsv + hca_genes.tsv");

    // Sanity only — this is a diagnostic run, not a pass/fail gate.
    assert!(post.iter().all(|g| g.pip.iter().all(|x| x.is_finite())));
    assert!(hres.sigma2_mean.is_finite() && hres.sigma2_mean > 0.0);
    assert!((0.0..=1.0).contains(&hres.pi0_mean));
    Ok(())
}
